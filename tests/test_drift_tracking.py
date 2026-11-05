"""Drift trackers must reduce to the session mean in the limits, be causal, and undo a planted drift."""

import numpy as np
import pytest
from pyriemann.geometry.mean import mean_riemann

from src.adaptation.drift_tracking import (
    causal_references,
    ema_weight,
    kernel_references,
    trial_covariances,
    whiten_per_trial,
)


def _covs(n: int, c: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, c, 3 * c))
    return A @ A.transpose(0, 2, 1) / (3 * c)


def test_kernel_with_huge_sigma_equals_session_mean() -> None:
    C = _covs(30, 4, 0)
    refs = kernel_references(C, sigma=1e6)
    np.testing.assert_allclose(refs, np.broadcast_to(mean_riemann(C), refs.shape), atol=1e-7)


def test_kernel_with_tiny_sigma_returns_each_trial() -> None:
    C = _covs(10, 4, 1)
    np.testing.assert_allclose(kernel_references(C, sigma=0.05), C, atol=1e-7)


def test_cumulative_mean_is_exact_for_commuting_matrices() -> None:
    # for diagonal (commuting) SPD matrices the inductive mean IS the geometric mean of trials 1..t
    rng = np.random.default_rng(2)
    d = np.exp(rng.standard_normal((20, 5)))
    C = np.stack([np.diag(v) for v in d])
    refs = causal_references(C)
    for t in range(20):
        expected = np.diag(np.exp(np.log(d[: t + 1]).mean(0)))
        np.testing.assert_allclose(refs[t], expected, rtol=1e-10)


def test_causal_references_ignore_future_trials() -> None:
    C = _covs(40, 4, 3)
    C2 = C.copy()
    C2[25:] = _covs(15, 4, 99)
    for hl in (None, 8.0):
        a, b = causal_references(C, hl), causal_references(C2, hl)
        np.testing.assert_array_equal(a[:25], b[:25])
        assert not np.allclose(a[25:], b[25:])


def test_ema_forgets_and_cumulative_does_not() -> None:
    # session switches from I to 4I halfway; the EMA must end near 4I, the cumulative mean near 2I
    C = np.concatenate([np.tile(np.eye(3), (200, 1, 1)), np.tile(4 * np.eye(3), (200, 1, 1))])
    np.testing.assert_allclose(causal_references(C, half_life=10)[-1], 4 * np.eye(3), rtol=1e-3)
    np.testing.assert_allclose(causal_references(C)[-1], 2 * np.eye(3), rtol=1e-10)


def test_ema_weight_halves_influence() -> None:
    assert (1 - ema_weight(48)) ** 48 == pytest.approx(0.5)
    with pytest.raises(ValueError):
        ema_weight(0)


def test_whitening_removes_planted_linear_gain_drift() -> None:
    # a power ramp (gain x1 -> x3 over the session) is removed by the kernel tracker, not by one global mean
    rng = np.random.default_rng(4)
    X = rng.standard_normal((120, 4, 500))
    gain = np.linspace(1.0, 3.0, 120)
    Xd = X * gain[:, None, None]
    C = trial_covariances(Xd)
    tracked = trial_covariances(whiten_per_trial(Xd, kernel_references(C, sigma=5.0)))
    static = trial_covariances(whiten_per_trial(Xd, np.broadcast_to(mean_riemann(C), C.shape)))
    tr = np.trace(tracked, axis1=1, axis2=2)
    st = np.trace(static, axis1=1, axis2=2)
    assert st[-10:].mean() / st[:10].mean() > 5.0
    assert 0.8 < tr[-10:].mean() / tr[:10].mean() < 1.25


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        kernel_references(_covs(5, 3, 0), sigma=0)
    with pytest.raises(ValueError):
        whiten_per_trial(np.zeros((4, 3, 10)), _covs(5, 3, 0))
