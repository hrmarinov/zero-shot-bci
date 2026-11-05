"""Confident selection must be class-balanced; IM-TTA must touch only BN affine params and lower IM loss;
soft-target training must work through the unchanged train_model."""

import numpy as np
import pytest
import torch

from src.adaptation.convnet import ShallowConvNet, TrainConfig, predict_proba, set_deterministic, train_model
from src.adaptation.self_training import (
    im_tta,
    information_maximisation_loss,
    one_hot,
    select_confident,
)


def test_select_confident_is_per_predicted_class() -> None:
    # class 0 has 6 very confident trials, class 1 has 4 weakly confident ones
    probs = np.array([[0.99, 0.01]] * 6 + [[0.4, 0.6]] * 2 + [[0.45, 0.55]] * 2)
    idx = select_confident(probs, 0.5)
    np.testing.assert_array_equal(idx, [0, 1, 2, 6, 7])  # 3 of class 0, the 2 best of class 1
    assert len(select_confident(probs, 1.0)) == 10
    with pytest.raises(ValueError):
        select_confident(probs, 0.0)


def test_select_confident_picks_highest_within_class() -> None:
    probs = np.array([[0.6, 0.4], [0.9, 0.1], [0.7, 0.3], [0.8, 0.2]])
    np.testing.assert_array_equal(select_confident(probs, 0.5), [1, 3])


def test_one_hot() -> None:
    np.testing.assert_array_equal(one_hot(np.array([2, 0]), 3), [[0, 0, 1], [1, 0, 0]])


def test_im_loss_prefers_confident_and_diverse() -> None:
    confident_diverse = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    confident_collapsed = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    uncertain = torch.zeros(2, 2)
    loss = information_maximisation_loss
    assert loss(confident_diverse) < loss(uncertain) - 0.5
    assert loss(confident_diverse) < loss(confident_collapsed) - 0.5


def _toy(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(40, 4, 120, generator=gen)
    y = torch.arange(40) % 2
    X[y == 1, 0] *= 3.0  # class 1: more power on channel 0
    return X, y


def test_im_tta_changes_only_bn_affine_and_lowers_loss() -> None:
    set_deterministic(0)
    X, y = _toy()
    model = train_model(ShallowConvNet(4, 2, 120, n_filters=8, pool_size=30, pool_stride=10), X, y,
                        TrainConfig(epochs=5, batch_size=10), seed=0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    Xs = X * 1.5  # a shifted "session"
    adapted = im_tta(model, Xs, steps=30, lr=1e-2)
    after = adapted.state_dict()
    for k, v in before.items():
        if k in ("bn.weight", "bn.bias") or k.startswith("bn.running") or k == "bn.num_batches_tracked":
            continue
        torch.testing.assert_close(after[k], v, rtol=0, atol=0)
    assert not torch.equal(after["bn.weight"], before["bn.weight"])
    for k, v in before.items():  # the original model is untouched
        torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    with torch.no_grad():
        model_adabn = im_tta(model, Xs, steps=0, lr=1e-2)
        assert information_maximisation_loss(adapted(Xs)) < information_maximisation_loss(model_adabn(Xs))


def test_train_model_accepts_soft_targets() -> None:
    set_deterministic(0)
    X, y = _toy(1)
    soft = torch.as_tensor(one_hot(y.numpy(), 2)) * 0.9 + 0.05
    model = train_model(ShallowConvNet(4, 2, 120, n_filters=8, pool_size=30, pool_stride=10), X, soft,
                        TrainConfig(epochs=30, batch_size=10), seed=0)
    assert (predict_proba(model, X).argmax(1) == y.numpy()).mean() > 0.9
