"""Within-session drift tracking: a per-trial re-centering reference instead of one per session.

Riemannian alignment (``riemannian_align.py``) whitens every trial of a session by the same
reference, the session's mean covariance. D5 found a slow, monotone within-session drift
(rising alpha / broadband power, docs/diag_gates_d1_d4.md §D5). The references here follow
that drift, so trial t is whitened by a reference built from trials near t:

- ``kernel_references``: Gaussian-weighted Riemannian mean over trial index (non-causal, uses
  trials before and after t). This is the most generous tracker and is used as a gate: if it
  does not help, no causal tracker can.
- ``causal_references``: the inductive (geodesic) running mean. With ``half_life=None`` it is the
  cumulative mean of trials 1..t (no forgetting, the causal control). With a half-life it
  switches to an exponential moving average once 1/t falls below the EMA weight.

All functions take one session's trials in chronological order and use no labels and no other
session (per-session, per-subject).
"""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.geodesic import geodesic_riemann
from pyriemann.geometry.mean import mean_riemann


def trial_covariances(X: np.ndarray, estimator: str = "oas") -> np.ndarray:
    return Covariances(estimator=estimator).fit_transform(X)


def kernel_references(covs: np.ndarray, sigma: float) -> np.ndarray:
    """Per-trial Riemannian mean with Gaussian weights exp(-(i-t)^2 / 2 sigma^2) over trial index."""
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    n = len(covs)
    idx = np.arange(n)
    init = mean_riemann(covs)
    refs = np.empty_like(covs)
    for t in range(n):
        w = np.exp(-0.5 * ((idx - t) / sigma) ** 2)
        refs[t] = mean_riemann(covs, sample_weight=w / w.sum(), init=init)
    return refs


def ema_weight(half_life: float) -> float:
    """Geodesic step size whose influence halves every ``half_life`` trials."""
    if half_life <= 0:
        raise ValueError(f"half_life must be > 0, got {half_life}")
    return 1.0 - 2.0 ** (-1.0 / half_life)


def causal_references(covs: np.ndarray, half_life: float | None = None) -> np.ndarray:
    """Reference for trial t from trials 1..t only (the trial's own covariance is available online).

    Step weight w_t = max(1/t, ema_weight(half_life)): the cumulative mean early on, then an EMA.
    """
    floor = 0.0 if half_life is None else ema_weight(half_life)
    refs = np.empty_like(covs)
    refs[0] = covs[0]
    for t in range(1, len(covs)):
        refs[t] = geodesic_riemann(refs[t - 1], covs[t], alpha=max(1.0 / (t + 1), floor))
    return refs


def whiten_per_trial(X: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """X_t <- refs_t^{-1/2} X_t for each trial t."""
    if len(X) != len(refs):
        raise ValueError(f"{len(X)} trials but {len(refs)} references")
    return np.einsum("ncd,ndt->nct", invsqrtm(refs), X)
