"""src.pipeline: config validation, family weighting, group alignment, and a tiny CPU end-to-end fit."""

import dataclasses

import numpy as np
import pytest
import torch

from src.adaptation.convnet import TrainConfig
from src.pipeline import LITE, R10, R11, PipelineConfig, SubjectData, align_by_group, combine, fit_predict


def _data(n_per_group: int = 24, n_classes: int = 2, seed: int = 0) -> SubjectData:
    rng = np.random.default_rng(seed)
    groups = np.repeat(["a", "b", "c"], n_per_group)
    y = np.tile(np.arange(n_classes), len(groups) // n_classes)
    X = rng.standard_normal((len(y), 3, 256))
    X[:, 0] *= 1 + y[:, None]  # class-dependent variance on channel 0
    X[groups == "c"] *= 3.0  # a session-level scale change, removed by alignment
    return SubjectData(X_cnn=X, X_tslr=X.copy(), y=y, groups=groups, train=groups != "c")


def test_named_configs() -> None:
    assert (R11.n_cnn, R11.n_fts, R11.use_tslr) == (3, 2, True)
    assert (R10.n_cnn, R10.n_fts) == (3, 0) and (LITE.n_cnn, LITE.n_fts) == (0, 1)


@pytest.mark.parametrize("kwargs", [dict(n_cnn=-1), dict(n_cnn=0, n_fts=0, use_tslr=False)])
def test_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(**kwargs)


def test_combine_weights_families_equally() -> None:
    a, b, c = np.full((2, 2), 0.9), np.full((2, 2), 0.3), np.full((2, 2), 0.0)
    np.testing.assert_allclose(combine({"cnn": a, "tslr": b, "fts": c}), (a + b + c) / 3)
    with pytest.raises(ValueError):
        combine({})


def test_subject_data_rejects_group_spanning_train_and_test() -> None:
    d = _data()
    with pytest.raises(ValueError):
        dataclasses.replace(d, groups=np.repeat(["a", "b"], len(d.y) // 2))


def test_align_by_group_removes_a_per_group_scale() -> None:
    d = _data()
    X2 = d.X_cnn.copy()
    X2[d.groups == "b"] *= 5.0  # OAS is scale-equivariant, so per-group alignment must undo this exactly
    np.testing.assert_allclose(align_by_group(X2, d.groups), align_by_group(d.X_cnn, d.groups), atol=1e-9)


def test_fit_predict_cpu_end_to_end() -> None:
    d = _data()
    cfg = PipelineConfig(n_cnn=1, n_fts=1, train=TrainConfig(epochs=1, batch_size=16))
    out = fit_predict(d, 2, cfg, device=torch.device("cpu"))
    assert set(out) == {"cnn", "tslr", "fts", "ensemble"}
    for p in out.values():
        assert p.shape == ((~d.train).sum(), 2)
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    np.testing.assert_allclose(out["ensemble"], (out["cnn"] + out["tslr"] + out["fts"]) / 3)


def test_fit_predict_cache_resumes(tmp_path) -> None:
    d = _data()
    cfg = PipelineConfig(n_cnn=1, n_fts=0, use_tslr=False, train=TrainConfig(epochs=1, batch_size=16))
    first = fit_predict(d, 2, cfg, device=torch.device("cpu"), cache_dir=tmp_path)
    calls = []
    second = fit_predict(d, 2, cfg, device=torch.device("cpu"), cache_dir=tmp_path, before_fit=lambda: calls.append(1))
    assert not calls and (tmp_path / "cnn_seed0.npy").exists()
    np.testing.assert_array_equal(first["ensemble"], second["ensemble"])
