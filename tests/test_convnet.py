"""Tests for src/adaptation/convnet.py — the bugs a ShallowConvNet + AdaBN harness invites.

Each test names the failure mode it guards against (from the R3 red-team checklist in
docs/audit_log.md). All run on CPU in seconds; none needs a dataset download except the
loader-identity test, which is skipped when the IV-2a cache is absent.
"""

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from src.adaptation.convnet import (
    ShallowConvNet,
    TrainConfig,
    _random_shift,
    adapt_bn_statistics,
    encode_labels,
    make_batches,
    predict_proba,
    predict_proba_groupwise,
    train_model,
)

C, T, K = 6, 200, 4


def _model(seed: int = 0, n_times: int = T) -> ShallowConvNet:
    torch.manual_seed(seed)
    return ShallowConvNet(C, K, n_times, n_filters=8, pool_size=30, pool_stride=10)


def _data(n: int = 64, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    return scale * torch.randn(n, C, T, generator=torch.Generator().manual_seed(seed))


def test_unfold_einsum_equals_reference_conv2d_pair() -> None:
    """Implementation error guard: the matmul convs must equal Schirrmeister's Conv2d pair."""
    m = _model()
    x = _data(5)
    conv_t = nn.Conv2d(1, 8, (1, 25))
    conv_s = nn.Conv2d(8, 8, (C, 1), bias=False)
    with torch.no_grad():
        conv_t.weight.copy_(m.temporal_w[:, None, None, :])
        conv_t.bias.copy_(m.temporal_b)
        conv_s.weight.copy_(m.spatial_w[:, :, :, None])
        ref = conv_s(conv_t(x[:, None])).squeeze(2)
        got = m.features_pre_bn(x)
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_forward_shape_and_too_short_input() -> None:
    assert _model()(_data(3)).shape == (3, K)
    with pytest.raises(ValueError):
        ShallowConvNet(C, K, 40)


def test_adabn_sets_exact_moments_and_leaves_parameters_untouched() -> None:
    m = _model()
    before = {k: v.clone() for k, v in m.named_parameters()}
    x = _data(50, seed=1, scale=3.0)
    adapt_bn_statistics(m, x)
    with torch.no_grad():
        f = m.features_pre_bn(x)  # (B, F, T')
    flat = f.transpose(0, 1).reshape(f.shape[1], -1)
    torch.testing.assert_close(m.bn.running_mean, flat.mean(1), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(m.bn.running_var, flat.var(1, unbiased=True), rtol=1e-4, atol=1e-5)
    for k, v in m.named_parameters():
        assert torch.equal(v, before[k]), f"AdaBN changed parameter {k}"
    assert not m.training and not m.bn.training
    assert m.bn.momentum == 0.1, "momentum must be restored after AdaBN"


def test_adabn_is_order_invariant_and_takes_no_labels() -> None:
    m1, m2 = _model(), _model()
    x = _data(40, seed=2)
    adapt_bn_statistics(m1, x)
    adapt_bn_statistics(m2, x[torch.randperm(40, generator=torch.Generator().manual_seed(9))])
    torch.testing.assert_close(m1.bn.running_mean, m2.bn.running_mean)
    torch.testing.assert_close(m1.bn.running_var, m2.bn.running_var)
    assert list(inspect.signature(adapt_bn_statistics).parameters) == ["model", "X"]


def test_adabn_is_not_a_no_op_and_rejects_bad_input() -> None:
    m = _model()
    x = _data(30, seed=3, scale=5.0)
    p_before = predict_proba(m, x)
    adapt_bn_statistics(m, x)
    assert not np.allclose(p_before, predict_proba(m, x))
    with pytest.raises(ValueError):
        adapt_bn_statistics(m, x[:1])
    with pytest.raises(ValueError):
        adapt_bn_statistics(nn.Linear(2, 2), torch.zeros(4, 2))


def test_eval_prediction_does_not_depend_on_batch_composition() -> None:
    """Guard against accidental train-mode BN (silent transductive AdaBN in a baseline arm)."""
    m = _model()
    x = _data(20, seed=4)
    full = predict_proba(m, x)
    one_by_one = np.vstack([predict_proba(m, x[i : i + 1]) for i in range(20)])
    np.testing.assert_allclose(full, one_by_one, rtol=1e-5, atol=1e-6)
    assert m.bn.num_batches_tracked.item() == 0, "prediction updated BN buffers"


def test_groupwise_adabn_equals_per_group_manual_and_single_group_equals_global() -> None:
    x = _data(24, seed=5)
    groups = np.repeat([0, 1, 2], 8)
    m = _model()
    got = predict_proba_groupwise(m, x, groups)
    for g in range(3):
        ref_m = _model()
        adapt_bn_statistics(ref_m, x[g * 8 : (g + 1) * 8])
        np.testing.assert_allclose(got[g * 8 : (g + 1) * 8], predict_proba(ref_m, x[g * 8 : (g + 1) * 8]), rtol=1e-5)
    one = predict_proba_groupwise(_model(), x, np.zeros(24))
    glob = _model()
    adapt_bn_statistics(glob, x)
    np.testing.assert_allclose(one, predict_proba(glob, x), rtol=1e-5, atol=1e-6)


def test_make_batches_partitions_and_respects_groups() -> None:
    gen = torch.Generator().manual_seed(0)
    plain = make_batches(100, None, 32, gen)
    assert sorted(np.concatenate(plain)) == list(range(100))
    groups = np.repeat(np.arange(6), 48)
    grouped = make_batches(288, groups, 48, gen)
    assert len(grouped) == 6
    assert sorted(np.concatenate(grouped)) == list(range(288))
    for b in grouped:
        assert len(set(groups[b])) == 1, "a SAT batch mixed two runs"
    # a trailing batch of one trial is dropped (BN cannot normalise it)
    assert all(len(b) >= 2 for b in make_batches(33, None, 32, gen))


def test_random_shift_is_circular_and_off_by_default() -> None:
    x = _data(4)
    gen = torch.Generator().manual_seed(0)
    assert torch.equal(_random_shift(x, 0, gen), x)
    s = _random_shift(x, 10, gen)
    for i in range(4):
        rolls = [torch.equal(s[i], torch.roll(x[i], k, -1)) for k in range(-10, 11)]
        assert any(rolls)


def test_training_is_deterministic_and_learns_a_separable_problem() -> None:
    gen = torch.Generator().manual_seed(1)
    y = torch.arange(80) % K
    t = torch.arange(T, dtype=torch.float32)
    x = 0.3 * torch.randn(80, C, T, generator=gen)
    for k in range(K):  # class k = a strong oscillation on channel k
        x[y == k, k] += torch.sin(0.3 * t)
    cfg = TrainConfig(epochs=40, batch_size=16, lr=3e-3)
    m1 = train_model(_model(1), x, y, cfg, seed=7)
    m2 = train_model(_model(1), x, y, cfg, seed=7)
    p1, p2 = predict_proba(m1, x), predict_proba(m2, x)
    np.testing.assert_array_equal(p1, p2)
    assert (p1.argmax(1) == y.numpy()).mean() > 0.9
    assert "y_eval" not in inspect.signature(train_model).parameters


def test_encode_labels_fixed_order_and_unknown_label() -> None:
    classes = ["left_hand", "right_hand", "feet", "tongue"]
    np.testing.assert_array_equal(encode_labels(np.array(["feet", "left_hand"]), classes), [2, 0])
    with pytest.raises(ValueError):
        encode_labels(np.array(["rest"]), classes)


def test_window_loader_reproduces_the_audited_loader() -> None:
    """The new loader must return the audited Phase 1/2 arrays at the Phase 1/2 settings."""
    pytest.importorskip("moabb")
    from src.config import FMAX, FMIN, TMAX, TMIN
    from src.datasets import load_iv2a_subject, load_iv2a_subject_window

    try:
        ref = load_iv2a_subject(1)
    except Exception as exc:  # noqa: BLE001 - offline / no cache
        pytest.skip(f"IV-2a not available: {exc!r}")
    new = load_iv2a_subject_window(1, FMIN, FMAX, TMIN, TMAX)
    np.testing.assert_array_equal(new.X_calib, ref.X_calib)
    np.testing.assert_array_equal(new.X_eval, ref.X_eval)
    np.testing.assert_array_equal(new.y_eval, ref.y_eval)
    assert sorted(np.unique(new.runs_calib)) == list(range(6))
    assert np.all(np.bincount(new.runs_eval) == 48)
    with pytest.raises(ValueError):
        load_iv2a_subject_window(1, 4, 38, -0.5, 4)


def test_matches_braindecode_shallowfbcspnet_given_the_same_weights() -> None:
    """Our matmul ShallowConvNet must be the reference architecture, not a look-alike.

    Copies braindecode's (randomly initialised) weights and BN buffers into ours and
    requires the same eval-mode outputs, so any drift in layer order, pooling, log,
    flatten order or classifier layout fails here.
    """
    bd = pytest.importorskip("braindecode.models")
    n_chans, n_times = 22, 1001
    torch.manual_seed(0)
    ref = bd.ShallowFBCSPNet(n_chans=n_chans, n_outputs=K, n_times=n_times).eval()
    with torch.no_grad():
        ref.bnorm.running_mean.uniform_(-0.5, 0.5)
        ref.bnorm.running_var.uniform_(0.5, 2.0)
        ref.bnorm.weight.uniform_(0.5, 1.5)
        ref.bnorm.bias.uniform_(-0.2, 0.2)
    ours = ShallowConvNet(n_chans, K, n_times).eval()
    ct, cs = ref.conv_time_spat.conv_time, ref.conv_time_spat.conv_spat
    clf = ref.final_layer.conv_classifier
    assert ours.bn.eps == ref.bnorm.eps
    with torch.no_grad():
        ours.temporal_w.copy_(ct.weight[:, 0, :, 0])
        ours.temporal_b.copy_(ct.bias)
        ours.spatial_w.copy_(cs.weight[:, :, 0, :])
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(ours.bn, name).copy_(getattr(ref.bnorm, name))
        ours.classifier.weight.copy_(clf.weight.reshape(K, -1))
        ours.classifier.bias.copy_(clf.bias)
    x = torch.randn(8, n_chans, n_times, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(ours(x), ref(x), rtol=1e-4, atol=1e-4)


def test_train_model_applies_augment_to_every_batch_and_none_is_unchanged() -> None:
    from src.adaptation.convnet import ShallowConvNet, TrainConfig, set_deterministic, train_model

    gen = torch.Generator().manual_seed(0)
    X, y = torch.randn(20, 3, 120, generator=gen), torch.arange(20) % 2
    cfg = TrainConfig(epochs=2, batch_size=10)

    def fit(augment):
        set_deterministic(0)
        net = ShallowConvNet(3, 2, 120, n_filters=4, pool_size=30, pool_stride=10)
        return train_model(net, X, y, cfg, seed=0, augment=augment).state_dict()

    calls = []

    def identity(xb):
        calls.append(xb.shape[0])
        return xb

    base, with_identity = fit(None), fit(identity)
    assert calls == [10, 10, 10, 10]
    for k in base:
        assert torch.equal(base[k], with_identity[k])
    zeroed = fit(lambda xb: torch.zeros_like(xb))
    assert not torch.equal(base["spatial_w"], zeroed["spatial_w"])
