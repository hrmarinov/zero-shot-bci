"""src.pipeline causal (online) alignment: recursive mean, causality, training rows unchanged, end-to-end."""

import dataclasses

import numpy as np
import pytest
import torch
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import distance_riemann
from pyriemann.geometry.mean import mean_riemann

from src.adaptation.convnet import TrainConfig
from src.pipeline import (PipelineConfig, SubjectData, align_by_group, align_causal, causal_references,
                          fit_predict)


def _covs(n: int, c: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, c, 50))
    return Covariances("scm").fit_transform(A)


def _data(seed: int = 0) -> SubjectData:
    rng = np.random.default_rng(seed)
    groups = np.repeat(["a", "b", "c", "d"], 24)
    y = np.tile(np.arange(2), 48)
    X = rng.standard_normal((len(y), 3, 256))
    X[:, 0] *= 1 + y[:, None]
    X[groups == "c"] *= 3.0
    X[groups == "d"] *= 0.5
    return SubjectData(X_cnn=X, X_tslr=X.copy(), y=y, groups=groups, train=np.isin(groups, ["a", "b"]))


def test_recursive_mean_cold_start_is_exact_for_two_trials() -> None:
    C = _covs(2)
    refs = causal_references(C, np.eye(3), prior_weight=0)
    np.testing.assert_allclose(refs[0], C[0], atol=1e-10)
    np.testing.assert_allclose(refs[1], mean_riemann(C), atol=1e-8)


def test_infinite_prior_weight_never_adapts_and_negative_is_rejected() -> None:
    C, P = _covs(5), _covs(1, seed=1)[0]
    np.testing.assert_allclose(causal_references(C, P, np.inf), np.repeat(P[None], 5, 0))
    with pytest.raises(ValueError):
        causal_references(C, P, -1)


def test_running_reference_moves_from_prior_towards_the_test_mean() -> None:
    C = _covs(200) * 4.0
    P = np.eye(3)
    refs = causal_references(C, P, prior_weight=10)
    full = mean_riemann(C)
    assert distance_riemann(refs[-1], full) < 0.1 * distance_riemann(P, full)
    assert distance_riemann(refs[0], P) < distance_riemann(refs[0], full)  # early trials stay near the prior


def test_causal_alignment_never_uses_future_trials_and_keeps_training_rows() -> None:
    d = _data()
    A = align_causal(d.X_cnn, d.groups, d.train)
    np.testing.assert_array_equal(A[d.train], align_by_group(d.X_cnn, d.groups)[d.train])
    X2 = d.X_cnn.copy()
    test_idx = np.flatnonzero(~d.train)
    cut = test_idx[10]
    X2[cut:] = np.random.default_rng(9).standard_normal(X2[cut:].shape) * 7  # rewrite the future
    A2 = align_causal(X2, d.groups, d.train)
    np.testing.assert_array_equal(A2[:cut], A[:cut])
    assert not np.allclose(A2[cut:], A[cut:])


def test_test_groups_are_adapted_independently() -> None:
    d = _data()
    A = align_causal(d.X_cnn, d.groups, d.train)
    X2 = d.X_cnn.copy()
    X2[d.groups == "c"] *= 10.0  # group c changes; group d must not
    A2 = align_causal(X2, d.groups, d.train)
    np.testing.assert_array_equal(A2[d.groups == "d"], A[d.groups == "d"])


def test_config_validates_alignment() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(alignment="online")
    with pytest.raises(ValueError):
        PipelineConfig(alignment="causal", causal_prior_weight=-1)


def test_fit_predict_causal_end_to_end_and_separate_cache(tmp_path) -> None:
    d = _data()
    cfg = PipelineConfig(n_cnn=1, n_fts=0, train=TrainConfig(epochs=1, batch_size=16), alignment="causal")
    out = fit_predict(d, 2, cfg, device=torch.device("cpu"), cache_dir=tmp_path)
    assert (tmp_path / "cnn_seed0_causal.npy").exists() and not (tmp_path / "cnn_seed0.npy").exists()
    for p in out.values():
        assert p.shape == ((~d.train).sum(), 2)
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    trans = fit_predict(d, 2, dataclasses.replace(cfg, alignment="transductive"), device=torch.device("cpu"))
    assert not np.allclose(out["tslr"], trans["tslr"])  # the test input differs, so predictions differ
