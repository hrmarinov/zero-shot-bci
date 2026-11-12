"""The project's best cross-session decoder (R11) as one entry point, with fit counts as parameters.

Families (each trained on the training session only and averaged with EQUAL weight per family; within a
family the seeds are averaged first):
  cnn   ShallowConvNet on per-group Riemannian-aligned 4-38 Hz, 0-4 s input, seeds 0..n_cnn-1.
  fts   the same network trained with FTSurrogate augmentation (p = 0.5), seeds 0..n_fts-1.
  tslr  OAS covariance -> tangent space -> logistic regression (C = 1) on aligned 8-30 Hz, 0.5-2.5 s.

Named configurations (docs/diag_gates_d1_d4.md):
  R11  = (CNN3 + TS+LR + FTS2) / 3       the best pipeline (82.45% IV-2a, 80.86% IV-2b), 5 GPU fits.
  R10  = (CNN3 + TS+LR) / 2              the previous best, 3 GPU fits.
  LITE = (FTS1 + TS+LR) / 2              1 GPU fit; keeps ~82% of R11's gain over a single CNN in a
                                         post-hoc ablation (reports/r14_council_and_ideation.md, §3).
                                         Not yet confirmed; do not treat as selected.

Alignment is fitted per group (IV-2a: per session; IV-2b: per recording) on that group's own trials,
which uses the test session's unlabelled trials (transductive), exactly as R3-R14.

``experiments/check_pipeline_reproduces_r11.py`` retrains R11 through this module and checks that every
member equals the cached member predictions and every subject's accuracy equals R11's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.geometry.mean import mean_riemann
from sklearn.linear_model import LogisticRegression

from src.adaptation.augmentation import make_augmentation
from src.adaptation.convnet import ShallowConvNet, TrainConfig, predict_proba, set_deterministic, train_model
from src.adaptation.riemannian_align import RiemannianAlignment
from src.adaptation.stacked_convnet import (StackedShallowConvNet, dropout_rng_state, predict_proba_stacked,
                                            train_stacked)

CNN_BAND = dict(fmin=4.0, fmax=38.0, tmin=0.0, tmax=4.0)  # R3's WIDE input
CNN_TRAIN = TrainConfig(epochs=300, batch_size=48, lr=1e-3, weight_decay=1e-3, max_shift=0)  # R3's CFG
RA_SELF_CHECK_TOL = 1e-8


@dataclass(frozen=True)
class PipelineConfig:
    """How many members of each family to fit. ``one_hot_targets`` only selects the loss-target format
    (IV-2a's cached members used integer targets, IV-2b's one-hot); both are cross-entropy.

    ``fast`` trains all CNN members of a subject as one stacked model (src/adaptation/stacked_convnet.py):
    same seeds, initial weights, batch orders, dropout masks and augmentation draws, ~4x faster, but not
    bit-identical (float rounding; measured in docs/diag_gates_d1_d4.md, F1). Off by default."""

    n_cnn: int = 3
    n_fts: int = 2
    use_tslr: bool = True
    one_hot_targets: bool = False
    train: TrainConfig = field(default_factory=lambda: CNN_TRAIN)
    fast: bool = False

    def __post_init__(self) -> None:
        if self.n_cnn < 0 or self.n_fts < 0:
            raise ValueError("member counts must be >= 0")
        if self.n_cnn == 0 and self.n_fts == 0 and not self.use_tslr:
            raise ValueError("the pipeline needs at least one member")


R11 = PipelineConfig(n_cnn=3, n_fts=2)
R10 = PipelineConfig(n_cnn=3, n_fts=0)
LITE = PipelineConfig(n_cnn=0, n_fts=1)


@dataclass(frozen=True)
class SubjectData:
    """One subject's trials in recording order. ``train`` marks the training trials; the rest are scored."""

    X_cnn: np.ndarray  # (n, channels, times), CNN_BAND
    X_tslr: np.ndarray  # (n, channels, times), 8-30 Hz, 0.5-2.5 s
    y: np.ndarray  # integer labels 0..n_classes-1
    groups: np.ndarray  # alignment group per trial (session or recording)
    train: np.ndarray  # bool

    def __post_init__(self) -> None:
        n = len(self.y)
        if not (len(self.X_cnn) == len(self.X_tslr) == len(self.groups) == len(self.train) == n):
            raise ValueError("all SubjectData fields need one entry per trial")
        if self.train.all() or not self.train.any():
            raise ValueError("need both training and test trials")
        if set(self.groups[self.train]) & set(self.groups[~self.train]):
            raise ValueError("an alignment group mixes training and test trials")


def align_by_group(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Riemannian alignment fitted on each group's own trials (self-checked: aligned mean = identity)."""
    out = np.empty_like(X)
    for g in np.unique(groups):
        idx = groups == g
        ra = RiemannianAlignment().fit(X[idx])
        out[idx] = ra.transform(X[idx])
        W = ra.whitening_
        mean = mean_riemann(Covariances(estimator=ra.estimator).fit_transform(X[idx]))
        dev = float(np.abs(W @ mean @ W - np.eye(W.shape[0])).max())
        if dev > RA_SELF_CHECK_TOL:
            raise AssertionError(f"alignment self-check failed for group {g!r}: {dev:.2e}")
    return out


def tslr_member(data: SubjectData) -> np.ndarray:
    A = align_by_group(data.X_tslr, data.groups)
    cov = Covariances(estimator="oas")
    ts = TangentSpace(metric="riemann").fit(cov.fit_transform(A[data.train]))
    Ft, Fe = ts.transform(cov.transform(A[data.train])), ts.transform(cov.transform(A[~data.train]))
    return LogisticRegression(C=1.0, max_iter=5000).fit(Ft, data.y[data.train]).predict_proba(Fe)


def _cnn_tensors(data: SubjectData, A: np.ndarray, n_classes: int, cfg: PipelineConfig,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Xt = torch.as_tensor(A[data.train].astype(np.float32), device=device)
    Xe = torch.as_tensor(A[~data.train].astype(np.float32), device=device)
    yt = data.y[data.train]
    target = (torch.as_tensor(np.eye(n_classes)[yt].astype(np.float32), device=device) if cfg.one_hot_targets
              else torch.as_tensor(yt, dtype=torch.long, device=device))
    return Xt, Xe, target


def cnn_member(data: SubjectData, A: np.ndarray, n_classes: int, seed: int, augmentation: str | None,
               cfg: PipelineConfig, device: torch.device) -> np.ndarray:
    """One ShallowConvNet on the aligned CNN input ``A``, built after seeding (bit-exact reruns)."""
    Xt, Xe, target = _cnn_tensors(data, A, n_classes, cfg, device)
    set_deterministic(seed)
    augment = None if augmentation is None else make_augmentation(augmentation, seed)
    model = ShallowConvNet(A.shape[1], n_classes, A.shape[2]).to(device)
    model = train_model(model, Xt, target, cfg.train, seed=seed, augment=augment)
    probs = predict_proba(model, Xe)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


def cnn_members_stacked(data: SubjectData, A: np.ndarray, n_classes: int,
                        specs: list[tuple[int, str | None]], cfg: PipelineConfig,
                        device: torch.device) -> list[np.ndarray]:
    """The members ``specs`` = [(seed, augmentation), ...] trained together as one stacked model.

    Each member is built after ``set_deterministic(seed)`` exactly as in :func:`cnn_member`, and keeps its
    own batch order, dropout stream and augmentation draws (src/adaptation/stacked_convnet.py)."""
    Xt, Xe, target = _cnn_tensors(data, A, n_classes, cfg, device)
    members, states, augments = [], [], []
    for seed, aug in specs:
        set_deterministic(seed)
        augments.append(None if aug is None else make_augmentation(aug, seed))
        members.append(ShallowConvNet(A.shape[1], n_classes, A.shape[2]).to(device))
        states.append(dropout_rng_state(device))
    model = StackedShallowConvNet(members).to(device)
    del members
    train_stacked(model, Xt, target, cfg.train, [s for s, _ in specs], augments, states)
    probs = predict_proba_stacked(model, Xe)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return list(probs)


def combine(families: dict[str, np.ndarray]) -> np.ndarray:
    """Equal-weight mean over families (each already the mean of its seeds)."""
    if not families:
        raise ValueError("no members to combine")
    return np.mean(list(families.values()), axis=0)


def fit_predict(data: SubjectData, n_classes: int, cfg: PipelineConfig = R11,
                device: torch.device | None = None, cache_dir: Path | None = None,
                before_fit: Callable[[], None] | None = None) -> dict[str, np.ndarray]:
    """Fit every member on the training trials; return test-trial probabilities per family and 'ensemble'.

    ``cache_dir`` stores each CNN member's probabilities (``{family}_seed{k}.npy``) so an interrupted
    run resumes; ``before_fit`` runs before each GPU fit (e.g. src.gpu_health.wait_until_healthy).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aligned: list[np.ndarray] = []  # CNN input, aligned once on first use
    suffix = "_fast" if cfg.fast else ""  # fast members never mix with the exact cache
    specs = [("cnn", k, None) for k in range(cfg.n_cnn)] + [("fts", k, "ftsurrogate") for k in range(cfg.n_fts)]

    def cache_path(family: str, seed: int) -> Path | None:
        return None if cache_dir is None else Path(cache_dir) / f"{family}_seed{seed}{suffix}.npy"

    stacked: dict[tuple[str, int], np.ndarray] = {}
    if cfg.fast:
        todo = [s for s in specs if (p := cache_path(s[0], s[1])) is None or not p.exists()]
        if todo:
            if before_fit is not None:
                before_fit()
            A = align_by_group(data.X_cnn, data.groups)
            probs = cnn_members_stacked(data, A, n_classes, [(k, aug) for _, k, aug in todo], cfg, device)
            for (fam, k, _), p in zip(todo, probs):
                stacked[(fam, k)] = p

    def member(family: str, seed: int, aug: str | None) -> np.ndarray:
        path = cache_path(family, seed)
        if (family, seed) in stacked:
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, stacked[(family, seed)])
            return stacked[(family, seed)]
        if path is not None and path.exists():
            return np.load(path)
        if before_fit is not None:
            before_fit()
        if not aligned:
            aligned.append(align_by_group(data.X_cnn, data.groups))
        p = cnn_member(data, aligned[0], n_classes, seed, aug, cfg, device)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, p)
        return p

    families: dict[str, np.ndarray] = {}
    if cfg.n_cnn:
        families["cnn"] = np.mean([member("cnn", k, None) for k in range(cfg.n_cnn)], axis=0)
    if cfg.use_tslr:
        families["tslr"] = tslr_member(data)
    if cfg.n_fts:
        families["fts"] = np.mean([member("fts", k, "ftsurrogate") for k in range(cfg.n_fts)], axis=0)
    return {**families, "ensemble": combine(families)}


IV2B_CLASSES = ["left_hand", "right_hand"]


def load_subject_data(dataset: str, subject: int) -> tuple[SubjectData, int]:
    """(SubjectData, n_classes) for 'IV-2a' (T -> E, per-session alignment) or 'IV-2b'
    (recordings 0-2train -> 3test, 4test, per-recording alignment); trial order is checked across bands."""
    from src.adaptation.convnet import encode_labels
    from src.config import FMAX, FMIN, IV2A_CLASSES, TMAX, TMIN
    from src.datasets import IV2B_CALIB_SESSIONS, load_iv2a_subject_window, load_iv2b_subject_window

    narrow = dict(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
    if dataset == "IV-2a":
        w, d = load_iv2a_subject_window(subject, **CNN_BAND), load_iv2a_subject_window(subject, **narrow)
        if not (np.array_equal(w.y_calib, d.y_calib) and np.array_equal(w.y_eval, d.y_eval)):
            raise AssertionError(f"IV-2a s{subject}: trial order differs between bands")
        n_t, n_e = len(d.y_calib), len(d.y_eval)
        data = SubjectData(X_cnn=np.concatenate([w.X_calib, w.X_eval]), X_tslr=np.concatenate([d.X_calib, d.X_eval]),
                           y=encode_labels(np.concatenate([d.y_calib, d.y_eval]), IV2A_CLASSES),
                           groups=np.array(["train"] * n_t + ["test"] * n_e),
                           train=np.r_[np.ones(n_t, bool), np.zeros(n_e, bool)])
        return data, len(IV2A_CLASSES)
    if dataset == "IV-2b":
        w, d = load_iv2b_subject_window(subject, **CNN_BAND), load_iv2b_subject_window(subject, **narrow)
        if not (np.array_equal(w.y, d.y) and np.array_equal(w.session, d.session)):
            raise AssertionError(f"IV-2b s{subject}: trial order differs between bands")
        data = SubjectData(X_cnn=w.X, X_tslr=d.X, y=encode_labels(d.y, IV2B_CLASSES), groups=d.session,
                           train=np.isin(d.session, sorted(IV2B_CALIB_SESSIONS)))
        return data, len(IV2B_CLASSES)
    raise ValueError(f"unknown dataset {dataset!r}; choose 'IV-2a' or 'IV-2b'")
