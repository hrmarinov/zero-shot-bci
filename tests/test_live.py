"""src.live.LiveDecoder: trial-by-trial decoding reproduces fit_predict(alignment='causal')."""

import dataclasses

import numpy as np
import pytest
import torch

from src.adaptation.convnet import TrainConfig
from src.live import LiveDecoder
from src.pipeline import PipelineConfig, SubjectData, fit_predict

CPU = torch.device("cpu")


def _data(seed: int = 0) -> SubjectData:
    rng = np.random.default_rng(seed)
    groups = np.repeat(["a", "b", "c", "d"], 24)
    y = np.tile(np.arange(2), 48)
    X = rng.standard_normal((len(y), 3, 256))
    X[:, 0] *= 1 + y[:, None]
    X[groups == "c"] *= 3.0
    X[groups == "d"] *= 0.5
    return SubjectData(X_cnn=X, X_tslr=X[:, :, ::2].copy(), y=y, groups=groups, train=np.isin(groups, ["a", "b"]))


def _replay(dec: LiveDecoder, d: SubjectData) -> dict[str, np.ndarray]:
    """Decode every test trial in order; a new test group starts a new session (as align_causal)."""
    outs, current = [], None
    for i in np.flatnonzero(~d.train):
        if d.groups[i] != current:
            dec.start_session()
            current = d.groups[i]
        outs.append(dec.decode(d.X_cnn[i], d.X_tslr[i]))
    return {k: np.stack([o[k] for o in outs]) for k in outs[0]}


@pytest.mark.parametrize("fast", [False, True])
def test_live_replay_equals_causal_fit_predict(fast: bool) -> None:
    d = _data()
    cfg = PipelineConfig(n_cnn=2, n_fts=1, train=TrainConfig(epochs=2, batch_size=16), alignment="causal", fast=fast)
    ref = fit_predict(d, 2, cfg, device=CPU)
    dec = LiveDecoder.calibrate(d.X_cnn[d.train], d.X_tslr[d.train], d.y[d.train], d.groups[d.train], 2, cfg, CPU)
    got = _replay(dec, d)
    assert set(got) == set(ref)
    for k in ref:
        np.testing.assert_allclose(got[k], ref[k], atol=1e-5)
    assert len(dec.latencies_ms) == (~d.train).sum() and min(dec.latencies_ms) > 0


def test_start_session_resets_adaptation_and_order_matters() -> None:
    d = _data()
    cfg = PipelineConfig(n_cnn=1, n_fts=0, train=TrainConfig(epochs=1, batch_size=16))
    dec = LiveDecoder.calibrate(d.X_cnn[d.train], d.X_tslr[d.train], d.y[d.train], d.groups[d.train], 2, cfg, CPU)
    i, j = np.flatnonzero(d.groups == "c")[:2]
    first = dec.decode(d.X_cnn[i], d.X_tslr[i])["ensemble"]
    after = dec.decode(d.X_cnn[i], d.X_tslr[i])["ensemble"]  # the reference has moved
    assert not np.allclose(first, after)
    dec.start_session()
    np.testing.assert_allclose(dec.decode(d.X_cnn[i], d.X_tslr[i])["ensemble"], first, atol=1e-12)
    assert dec.decode(d.X_cnn[j], d.X_tslr[j])["ensemble"].shape == (2,)


def test_tslr_only_and_bad_input() -> None:
    d = _data()
    cfg = dataclasses.replace(PipelineConfig(n_cnn=0, n_fts=0), train=TrainConfig(epochs=1))
    dec = LiveDecoder.calibrate(d.X_cnn[d.train], d.X_tslr[d.train], d.y[d.train], d.groups[d.train], 2, cfg, CPU)
    out = dec.decode(d.X_cnn[60], d.X_tslr[60])
    assert set(out) == {"tslr", "ensemble"}
    with pytest.raises(ValueError):
        LiveDecoder.calibrate(d.X_cnn[:10], d.X_tslr[:9], d.y[:10], d.groups[:10], 2, cfg, CPU)
