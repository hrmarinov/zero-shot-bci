"""src.adaptation.stacked_convnet: K stacked ShallowConvNets behave like K separate ones (to float rounding)."""

import dataclasses

import numpy as np
import pytest
import torch

from src.adaptation.augmentation import make_augmentation
from src.adaptation.convnet import ShallowConvNet, TrainConfig, predict_proba, set_deterministic, train_model
from src.adaptation.stacked_convnet import (StackedShallowConvNet, dropout_rng_state, predict_proba_stacked,
                                            train_stacked)
from src.pipeline import PipelineConfig, SubjectData, fit_predict

CPU = torch.device("cpu")
C, T, N_CLASSES = 4, 200, 2


def _members(seeds, augs=None):
    augs = augs or [None] * len(seeds)
    members, states, augments = [], [], []
    for s, a in zip(seeds, augs):
        set_deterministic(s)
        augments.append(None if a is None else make_augmentation(a, s))
        members.append(ShallowConvNet(C, N_CLASSES, T))
        states.append(dropout_rng_state(CPU))
    return members, states, augments


def _xy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(N_CLASSES), n // N_CLASSES)
    X = rng.standard_normal((n, C, T)).astype(np.float32)
    X[:, 0] *= 1 + y[:, None]
    return torch.as_tensor(X), torch.as_tensor(y)


def test_eval_forward_equals_each_member() -> None:
    members, _, _ = _members([0, 1, 2])
    st = StackedShallowConvNet(members).eval()
    X, _ = _xy()
    got = st(X.expand(3, *X.shape))
    for i, m in enumerate(members):
        torch.testing.assert_close(got[i], m.eval()(X), atol=1e-4, rtol=1e-4)


def test_train_forward_reproduces_each_members_dropout_and_batchnorm() -> None:
    members, states, _ = _members([3, 4])
    X, _ = _xy()
    st = StackedShallowConvNet([m for m in members]).train()
    st.dropout_states = [s.clone() for s in states]
    got = st(torch.stack([X[:16], X[16:32]]))
    for i, (m, s, xb) in enumerate(zip(members, states, (X[:16], X[16:32]))):
        torch.set_rng_state(s)
        torch.testing.assert_close(got[i], m.train()(xb), atol=1e-4, rtol=1e-4)


def test_training_matches_separate_runs_including_augmentation() -> None:
    X, y = _xy()
    cfg = TrainConfig(epochs=3, batch_size=16)
    seeds, augs = [0, 1, 0], [None, None, "ftsurrogate"]
    members, states, augments = _members(seeds, augs)
    st = train_stacked(StackedShallowConvNet(members), X, y, cfg, seeds, augments, states)
    got = predict_proba_stacked(st, X)
    for i, (s, a) in enumerate(zip(seeds, augs)):
        set_deterministic(s)
        aug = None if a is None else make_augmentation(a, s)
        ref = predict_proba(train_model(ShallowConvNet(C, N_CLASSES, T), X, y, cfg, seed=s, augment=aug), X)
        np.testing.assert_allclose(got[i], ref, atol=1e-3)


def test_members_do_not_influence_each_other() -> None:
    X, y = _xy()
    cfg = TrainConfig(epochs=2, batch_size=16)
    m2, s2, a2 = _members([0, 1])
    pair = predict_proba_stacked(train_stacked(StackedShallowConvNet(m2), X, y, cfg, [0, 1], a2, s2), X)
    m1, s1, a1 = _members([0])
    alone = predict_proba_stacked(train_stacked(StackedShallowConvNet(m1), X, y, cfg, [0], a1, s1), X)
    np.testing.assert_allclose(pair[0], alone[0], atol=1e-4)  # equal up to float rounding (other kernel shapes)


def test_one_hot_targets_and_unstack_roundtrip() -> None:
    X, y = _xy()
    members, states, augs = _members([0, 1])
    st = train_stacked(StackedShallowConvNet(members), X, torch.eye(N_CLASSES)[y], TrainConfig(epochs=1, batch_size=16),
                       [0, 1], augs, states)
    probs = predict_proba_stacked(st, X)
    assert probs.shape == (2, len(y), N_CLASSES)
    np.testing.assert_allclose(probs.sum(-1), 1, atol=1e-6)
    single = st.unstack(1, ShallowConvNet(C, N_CLASSES, T))
    np.testing.assert_allclose(predict_proba(single, X), probs[1], atol=1e-5)


def test_rejects_bad_input() -> None:
    X, y = _xy()
    members, states, augs = _members([0, 1])
    with pytest.raises(ValueError):
        StackedShallowConvNet([members[0], ShallowConvNet(C + 1, N_CLASSES, T)])
    with pytest.raises(ValueError):
        StackedShallowConvNet([])
    st = StackedShallowConvNet(members)
    with pytest.raises(ValueError):
        train_stacked(st, X, y, TrainConfig(epochs=1), [0], augs, states)
    with pytest.raises(ValueError):
        train_stacked(st, X, y, TrainConfig(epochs=1, max_shift=5), [0, 1], augs, states)
    with pytest.raises(ValueError):
        st(X.expand(3, *X.shape))


def _subject() -> SubjectData:
    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b", "c"], 24)
    y = np.tile(np.arange(2), 36)
    X = rng.standard_normal((len(y), 3, 256))
    X[:, 0] *= 1 + y[:, None]
    return SubjectData(X_cnn=X, X_tslr=X.copy(), y=y, groups=groups, train=groups != "c")


def test_pipeline_fast_option_is_close_to_exact_and_caches_separately(tmp_path) -> None:
    d = _subject()
    cfg = PipelineConfig(n_cnn=2, n_fts=1, train=TrainConfig(epochs=2, batch_size=16))
    exact = fit_predict(d, 2, cfg, device=CPU, cache_dir=tmp_path)
    fast_cfg = dataclasses.replace(cfg, fast=True)
    calls = []
    fast = fit_predict(d, 2, fast_cfg, device=CPU, cache_dir=tmp_path, before_fit=lambda: calls.append(1))
    assert len(calls) == 1  # one stacked fit for all three CNN members
    for name in ("cnn_seed0", "cnn_seed1", "fts_seed0"):
        assert (tmp_path / f"{name}.npy").exists() and (tmp_path / f"{name}_fast.npy").exists()
    for fam in ("cnn", "fts", "tslr", "ensemble"):
        np.testing.assert_allclose(fast[fam], exact[fam], atol=1e-3)
    again = fit_predict(d, 2, fast_cfg, device=CPU, cache_dir=tmp_path, before_fit=lambda: calls.append(1))
    assert len(calls) == 1  # resumed from the fast cache
    np.testing.assert_array_equal(again["ensemble"], fast["ensemble"])
