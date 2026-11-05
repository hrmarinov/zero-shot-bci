"""The oracle-rotation solver must recover a planted rotation (else the D1 bound is understated)."""

import numpy as np
import pytest
from scipy.stats import ortho_group

from src.adaptation.rotation_oracle import fit_orthogonal_congruence


def _spd(k: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((k, n, n))
    return A @ A.transpose(0, 2, 1) / n + 0.1 * np.eye(n)


@pytest.mark.parametrize("reflect", [False, True])
def test_recovers_planted_rotation_and_reflection(reflect: bool) -> None:
    n = 8
    M = _spd(4, n, 0)
    Q = ortho_group.rvs(n, random_state=3)
    if (np.linalg.det(Q) < 0) != reflect:
        Q[:, 0] *= -1
    source = np.stack([Q.T @ m @ Q for m in M])
    R, cost = fit_orthogonal_congruence(source, M, n_restarts=16, steps=800)
    np.testing.assert_allclose(R @ R.T, np.eye(n), atol=1e-8)
    np.testing.assert_allclose(np.stack([R @ s @ R.T for s in source]), M, atol=1e-4)
    assert cost < 1e-6


def test_identity_when_sets_already_match() -> None:
    M = _spd(4, 6, 1)
    R, cost = fit_orthogonal_congruence(M, M, n_restarts=4, steps=300)
    assert cost < 1e-10
    np.testing.assert_allclose(np.stack([R @ m @ R.T for m in M]), M, atol=1e-5)


def test_rejects_mismatched_input() -> None:
    with pytest.raises(ValueError):
        fit_orthogonal_congruence(_spd(4, 6, 0), _spd(3, 6, 0))
