"""R11 as a live decoder: calibrate once, then decode one trial at a time.

    dec = LiveDecoder.calibrate(X_cnn, X_tslr, y, groups, n_classes)   # training session (labelled)
    dec.start_session()                                                # a new (test) session begins
    out = dec.decode(x_cnn, x_tslr)                                    # each new trial, in order

``x_cnn`` is one epoch band-passed 4-38 Hz, 0-4 s post-cue, 250 Hz; ``x_tslr`` the same trial 8-30 Hz,
0.5-2.5 s (src.pipeline's two inputs; filtering the stream is the caller's job). Each band keeps its own
running Riemannian reference, warm-started from the training session's mean and updated with every decoded
trial exactly as :func:`src.pipeline.align_causal` (C1). Decoding a whole session trial by trial therefore
gives the same probabilities as ``fit_predict(..., alignment="causal")`` (checked in tests and in
``experiments/live_replay_check.py``). No labels are used after calibration.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import torch
from pyriemann.estimation import Covariances
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.geodesic import geodesic_riemann
from pyriemann.geometry.mean import mean_riemann

from src.adaptation.augmentation import make_augmentation
from src.adaptation.convnet import ShallowConvNet, predict_proba, set_deterministic, train_model
from src.adaptation.stacked_convnet import (StackedShallowConvNet, dropout_rng_state, predict_proba_stacked,
                                            train_stacked)
from src.pipeline import R11, PipelineConfig, align_by_group, combine, fit_tslr

Predictor = Callable[[torch.Tensor], list[np.ndarray]]  # (n, chans, times) -> one (n, n_classes) per member


@dataclass
class _RunningReference:
    """Recursive Riemannian mean of one band's trial stream (src.pipeline.causal_references, one step at a time)."""

    prior: np.ndarray
    prior_weight: float
    estimator: str = "oas"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.R, self.n = self.prior.copy(), 0

    def whiten(self, x: np.ndarray) -> np.ndarray:
        """Update the reference with trial ``x`` (channels, times) and return ``x`` whitened by it."""
        C = Covariances(estimator=self.estimator).transform(x[None])[0]
        self.n += 1
        if np.isfinite(self.prior_weight):
            self.R = geodesic_riemann(self.R, C, 1.0 / (self.prior_weight + self.n))
        return invsqrtm(self.R) @ x


def _train_cnns(A: np.ndarray, target: torch.Tensor, n_classes: int, specs: list[tuple[int, str | None]],
                cfg: PipelineConfig, device: torch.device) -> Predictor:
    """The CNN members (seed, augmentation), trained as src.pipeline does (exact or stacked)."""
    Xt = torch.as_tensor(A.astype(np.float32), device=device)
    if not specs:
        return lambda X: []
    if cfg.fast:
        members, states, augments = [], [], []
        for seed, aug in specs:
            set_deterministic(seed)
            augments.append(None if aug is None else make_augmentation(aug, seed))
            members.append(ShallowConvNet(A.shape[1], n_classes, A.shape[2]).to(device))
            states.append(dropout_rng_state(device))
        model = train_stacked(StackedShallowConvNet(members).to(device), Xt, target, cfg.train,
                              [s for s, _ in specs], augments, states)
        return lambda X: list(predict_proba_stacked(model, X))
    models = []
    for seed, aug in specs:
        set_deterministic(seed)
        augment = None if aug is None else make_augmentation(aug, seed)
        model = ShallowConvNet(A.shape[1], n_classes, A.shape[2]).to(device)
        models.append(train_model(model, Xt, target, cfg.train, seed=seed, augment=augment))
    return lambda X: [predict_proba(m, X) for m in models]


class LiveDecoder:
    """The R11 ensemble (or any PipelineConfig) decoding a session causally, one trial at a time."""

    def __init__(self, nets: Predictor, n_cnn: int, tslr: Callable[[np.ndarray], np.ndarray] | None,
                 ref_cnn: _RunningReference, ref_tslr: _RunningReference, device: torch.device) -> None:
        self._nets, self._n_cnn, self._tslr = nets, n_cnn, tslr
        self._ref_cnn, self._ref_tslr = ref_cnn, ref_tslr
        self._device = device
        self.latencies_ms: list[float] = []

    @classmethod
    def calibrate(cls, X_cnn: np.ndarray, X_tslr: np.ndarray, y: np.ndarray, groups: np.ndarray, n_classes: int,
                  cfg: PipelineConfig = R11, device: torch.device | None = None) -> "LiveDecoder":
        """Train every member on the labelled calibration trials (aligned per group, as R11)."""
        cfg = replace(cfg, alignment="causal")
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not (len(X_cnn) == len(X_tslr) == len(y) == len(groups)):
            raise ValueError("calibration inputs need one entry per trial")
        y = np.asarray(y)
        target = (torch.as_tensor(np.eye(n_classes)[y].astype(np.float32), device=device) if cfg.one_hot_targets
                  else torch.as_tensor(y, dtype=torch.long, device=device))
        A_cnn = align_by_group(X_cnn, groups)
        specs = [(k, None) for k in range(cfg.n_cnn)] + [(k, "ftsurrogate") for k in range(cfg.n_fts)]
        nets = _train_cnns(A_cnn, target, n_classes, specs, cfg, device)
        tslr = fit_tslr(align_by_group(X_tslr, groups), y) if cfg.use_tslr else None

        def prior(X: np.ndarray) -> np.ndarray:
            return mean_riemann(Covariances(estimator="oas").fit_transform(X))

        w = cfg.causal_prior_weight
        return cls(nets, cfg.n_cnn, tslr, _RunningReference(prior(X_cnn), w),
                   _RunningReference(prior(X_tslr), w), device)

    def start_session(self) -> None:
        """Forget the previous session's adaptation: both references return to the training prior."""
        self._ref_cnn.reset()
        self._ref_tslr.reset()

    def decode(self, x_cnn: np.ndarray, x_tslr: np.ndarray) -> dict[str, np.ndarray]:
        """Class probabilities for one trial: per family and the equal-weight 'ensemble' (n_classes,)."""
        t = time.perf_counter()
        a_cnn, a_tslr = self._ref_cnn.whiten(x_cnn), self._ref_tslr.whiten(x_tslr)
        xt = torch.as_tensor(a_cnn[None].astype(np.float32), device=self._device)
        probs = [p[0] for p in self._nets(xt)]
        cnn, fts = probs[:self._n_cnn], probs[self._n_cnn:]
        families: dict[str, np.ndarray] = {}
        if cnn:
            families["cnn"] = np.mean(cnn, axis=0)
        if self._tslr is not None:
            families["tslr"] = self._tslr(a_tslr[None])[0]
        if fts:
            families["fts"] = np.mean(fts, axis=0)
        out = {**families, "ensemble": combine(families)}
        self.latencies_ms.append(1000 * (time.perf_counter() - t))
        return out
