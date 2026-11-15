"""Population pretraining of R11's ShallowConvNets (P1, confirmed on the lockbox).

    pool = population_pool([other_subject_data, ...])                       # other people's TRAINING trials
    init = pretrain_members(*pool, n_classes)                               # once per population
    out  = fit_predict(new_user_data, n_classes, R11_PT, pretrained=init)   # fine-tune on the user's trials
    zs   = zero_shot_predict(init, X_new_session)                           # no calibration at all

Each subject in the pool is aligned on its own trials (src.pipeline.align_by_group); subjects only share a
supervised loss, and no computation aligns or compares two subjects' covariances (the project's rule). Use only
training sessions of other people, never the new user's data. P1 settings: 100 pretraining epochs, stacked
fast path, members = base seeds 0..n_cnn-1 then FTSurrogate seeds 0..n_fts-1 (FTSurrogate on during
pretraining). Measured gain (docs/diag_gates_d1_d4.md, P1): +3-4 pp at 48 labelled trials, +2 pp at ~100,
none with a full session.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.adaptation.augmentation import make_augmentation
from src.adaptation.convnet import ShallowConvNet, set_deterministic
from src.adaptation.stacked_convnet import (StackedShallowConvNet, dropout_rng_state, predict_proba_stacked,
                                            train_stacked)
from src.pipeline import CNN_TRAIN, R11_PT, PipelineConfig, SubjectData, align_by_group

PRETRAIN_EPOCHS = 100


def population_pool(subjects: Sequence[SubjectData]) -> tuple[np.ndarray, np.ndarray]:
    """The training trials of each subject, aligned on themselves (per group), pooled: (X float32, y)."""
    if not subjects:
        raise ValueError("need at least one subject")
    Xs, ys = [], []
    for d in subjects:
        Xs.append(align_by_group(d.X_cnn[d.train], d.groups[d.train]).astype(np.float32))
        ys.append(d.y[d.train])
    return np.concatenate(Xs), np.concatenate(ys)


def _specs(cfg: PipelineConfig) -> list[tuple[int, str | None]]:
    return [(k, None) for k in range(cfg.n_cnn)] + [(k, "ftsurrogate") for k in range(cfg.n_fts)]


def _stack(specs, n_chans: int, n_classes: int, n_times: int, inits, device):
    members, states, augs = [], [], []
    for (seed, aug), init in zip(specs, inits):
        set_deterministic(seed)
        augs.append(None if aug is None else make_augmentation(aug, seed))
        m = ShallowConvNet(n_chans, n_classes, n_times)
        if init is not None:
            m.load_state_dict(init)
        members.append(m.to(device))
        states.append(dropout_rng_state(device))
    return StackedShallowConvNet(members).to(device), states, augs


def pretrain_members(X: np.ndarray, y: np.ndarray, n_classes: int, cfg: PipelineConfig = R11_PT,
                     one_hot_targets: bool = False, epochs: int = PRETRAIN_EPOCHS,
                     device: torch.device | None = None) -> list[dict]:
    """Pretrain cfg's CNN members on a pooled, self-aligned population (population_pool); returns one CPU
    state dict per member, in fit_predict's order (base seeds, then FTSurrogate seeds)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = _specs(cfg)
    if not specs:
        raise ValueError("the config has no CNN members to pretrain")
    model, states, augs = _stack(specs, X.shape[1], n_classes, X.shape[2], [None] * len(specs), device)
    target = (torch.as_tensor(np.eye(n_classes)[y].astype(np.float32), device=device) if one_hot_targets
              else torch.as_tensor(y, dtype=torch.long, device=device))
    train_stacked(model, torch.as_tensor(X.astype(np.float32), device=device), target,
                  replace(CNN_TRAIN, epochs=epochs), [s for s, _ in specs], augs, states)
    out = [{k: v.detach().cpu() for k, v in model.unstack(i, ShallowConvNet(X.shape[1], n_classes, X.shape[2]))
            .state_dict().items()} for i in range(len(specs))]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def zero_shot_predict(pretrained: list[dict], X: np.ndarray, groups: np.ndarray | None = None,
                      cfg: PipelineConfig = R11_PT, device: torch.device | None = None) -> np.ndarray:
    """Calibration-free probabilities: the pretrained CNN families (base, FTSurrogate) averaged with equal
    weight, on trials aligned on their own session (per group; transductive, as R11)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = _specs(cfg)
    if len(pretrained) != len(specs):
        raise ValueError(f"pretrained has {len(pretrained)} members, the config has {len(specs)}")
    groups = np.zeros(len(X)) if groups is None else np.asarray(groups)
    A = align_by_group(X, groups)
    n_classes = pretrained[0]["classifier.bias"].shape[0]
    model, _, _ = _stack(specs, A.shape[1], n_classes, A.shape[2], pretrained, device)
    probs = predict_proba_stacked(model, torch.as_tensor(A.astype(np.float32), device=device))
    families = [np.mean(probs[:cfg.n_cnn], 0)] if cfg.n_cnn else []
    families += [np.mean(probs[cfg.n_cnn:], 0)] if cfg.n_fts else []
    return np.mean(families, axis=0)


def save(pretrained: list[dict], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pretrained, path)


def load(path: Path) -> list[dict]:
    return torch.load(path)
