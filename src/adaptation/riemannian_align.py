"""Riemannian alignment: per-session recentering of trial covariances.

This is the recentering step of Riemannian Procrustes Analysis (Rodrigues et
al.) / the Riemannian-metric analogue of Euclidean Alignment (He & Wu 2020):
each session's trials are whitened by the inverse square root of that
session's own Riemannian (affine-invariant) mean covariance, so the session's
mean covariance becomes the identity. Fit only needs unlabeled trials from
the session being aligned — no labels, no trials from any other session —
which is what makes this usable at the start of a live session with zero
calibration trials.
"""

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_riemann
from sklearn.base import BaseEstimator, TransformerMixin


class RiemannianAlignment(BaseEstimator, TransformerMixin):
    """Whiten trials so their own session's mean covariance becomes identity.

    Must be fit separately per session (per subject, per calibration/eval
    split) — fitting on one session and transforming another would just apply
    an arbitrary linear map rather than correcting for that session's own
    covariance shift.
    """

    def __init__(self, estimator: str = "oas"):
        self.estimator = estimator

    def fit(self, X: np.ndarray, y=None) -> "RiemannianAlignment":
        covs = Covariances(estimator=self.estimator).fit_transform(X)
        mean_cov = mean_riemann(covs)
        self.whitening_ = invsqrtm(mean_cov)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.einsum("cd,ndt->nct", self.whitening_, X)
