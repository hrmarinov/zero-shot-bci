"""src.pretrain + pretrained init in src.pipeline / src.live."""

import dataclasses

import numpy as np
import pytest
import torch

from src.adaptation.convnet import TrainConfig
from src.live import LiveDecoder
from src.pipeline import PipelineConfig, SubjectData, align_by_group, fit_predict
from src.pretrain import population_pool, pretrain_members, zero_shot_predict

CPU = torch.device("cpu")
CFG = PipelineConfig(n_cnn=2, n_fts=1, fast=True, train=TrainConfig(epochs=2, batch_size=16))


def _subject(seed: int) -> SubjectData:
    rng = np.random.default_rng(seed)
    groups = np.repeat(["tr", "te"], 32)
    y = np.tile(np.arange(2), 32)
    X = rng.standard_normal((64, 3, 256)) * (1 + seed)
    X[:, 0] *= 1 + y[:, None]
    return SubjectData(X_cnn=X, X_tslr=X[:, :, ::2].copy(), y=y, groups=groups, train=groups == "tr")


def _pretrained():
    X, y = population_pool([_subject(1), _subject(2)])
    return pretrain_members(X, y, 2, CFG, epochs=2, device=CPU)


def test_pool_aligns_each_subject_on_itself_and_uses_training_trials_only() -> None:
    a, b = _subject(1), _subject(2)
    X, y = population_pool([a, b])
    assert X.dtype == np.float32 and len(X) == a.train.sum() + b.train.sum()
    np.testing.assert_allclose(X[:32], align_by_group(a.X_cnn[a.train], a.groups[a.train]), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(y[32:], b.y[b.train])
    with pytest.raises(ValueError):
        population_pool([])


def test_pretraining_is_deterministic_and_returns_one_state_per_member() -> None:
    p1, p2 = _pretrained(), _pretrained()
    assert len(p1) == 3
    for s1, s2 in zip(p1, p2):
        for k in s1:
            assert torch.equal(s1[k], s2[k])


def test_zero_epoch_finetune_equals_zero_shot() -> None:
    pre, d = _pretrained(), _subject(3)
    cfg = dataclasses.replace(CFG, use_tslr=False, train=TrainConfig(epochs=0, batch_size=16))
    out = fit_predict(d, 2, cfg, device=CPU, pretrained=pre)
    zs = zero_shot_predict(pre, d.X_cnn[~d.train], d.groups[~d.train], cfg, device=CPU)
    np.testing.assert_allclose(out["ensemble"], zs, atol=1e-6)


def test_pretrained_init_changes_the_result_caches_separately_and_validates(tmp_path) -> None:
    pre, d = _pretrained(), _subject(3)
    scratch = fit_predict(d, 2, CFG, device=CPU, cache_dir=tmp_path)
    tuned = fit_predict(d, 2, CFG, device=CPU, cache_dir=tmp_path, pretrained=pre)
    assert (tmp_path / "cnn_seed0_fast_pt.npy").exists() and (tmp_path / "cnn_seed0_fast.npy").exists()
    assert not np.allclose(scratch["cnn"], tuned["cnn"])
    with pytest.raises(ValueError):
        fit_predict(d, 2, CFG, device=CPU, pretrained=pre[:2])
    with pytest.raises(ValueError):
        zero_shot_predict(pre[:2], d.X_cnn, cfg=CFG, device=CPU)


@pytest.mark.parametrize("fast", [True, False])
def test_live_decoder_with_pretrained_equals_causal_fit_predict(fast: bool) -> None:
    pre, d = _pretrained(), _subject(3)
    cfg = dataclasses.replace(CFG, fast=fast, alignment="causal")
    ref = fit_predict(d, 2, cfg, device=CPU, pretrained=pre)
    dec = LiveDecoder.calibrate(d.X_cnn[d.train], d.X_tslr[d.train], d.y[d.train], d.groups[d.train], 2, cfg, CPU,
                                pretrained=pre)
    dec.start_session()
    got = np.stack([dec.decode(d.X_cnn[i], d.X_tslr[i])["ensemble"] for i in np.flatnonzero(~d.train)])
    np.testing.assert_allclose(got, ref["ensemble"], atol=1e-5)
    with pytest.raises(ValueError):
        LiveDecoder.calibrate(d.X_cnn[d.train], d.X_tslr[d.train], d.y[d.train], d.groups[d.train], 2, cfg, CPU,
                              pretrained=pre[:1])
