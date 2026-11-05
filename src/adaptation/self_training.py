"""Label-free use of the evaluation session through the model's own predictions.

Three transplants from semi-supervised learning / test-time adaptation, all fed only by
unlabelled E trials and a teacher's probabilities (never E labels):

- pseudo-label self-training: retrain on T plus the teacher's *confident* E trials;
  :func:`select_confident` picks them class-balanced so a dominant class cannot take over;
- ensemble distillation: retrain on T plus all E trials with the teacher's soft probabilities
  (``train_model`` accepts probability targets: ``nn.CrossEntropyLoss`` supports them);
- information-maximisation TTA (Tent / SHOT-IM): :func:`im_tta` updates only BatchNorm affine
  parameters to make E predictions confident (low mean entropy) yet diverse (high entropy
  of the mean prediction).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from src.adaptation.convnet import adapt_bn_statistics, bn_layers


def select_confident(probs: np.ndarray, frac: float) -> np.ndarray:
    """Indices of the most confident ``frac`` of trials *within each predicted class* (sorted)."""
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    pred, conf = probs.argmax(1), probs.max(1)
    keep = []
    for k in np.unique(pred):
        idx = np.flatnonzero(pred == k)
        n_keep = int(np.ceil(frac * len(idx)))
        keep.append(idx[np.argsort(-conf[idx], kind="stable")[:n_keep]])
    return np.sort(np.concatenate(keep))


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes, dtype=np.float32)[y]


def information_maximisation_loss(logits: torch.Tensor) -> torch.Tensor:
    """mean per-trial entropy minus entropy of the mean prediction (SHOT-IM)."""
    p = torch.softmax(logits, dim=1)
    log_p = torch.log_softmax(logits, dim=1)
    cond_entropy = -(p * log_p).sum(1).mean()
    p_mean = p.mean(0)
    marginal_entropy = -(p_mean * torch.log(p_mean + 1e-12)).sum()
    return cond_entropy - marginal_entropy


def im_tta(model: nn.Module, X: torch.Tensor, steps: int, lr: float) -> nn.Module:
    """Return a copy of ``model`` with BN affine parameters tuned by IM on ``X`` (full batch).

    BN uses the batch statistics of ``X`` during adaptation (as in Tent) and, afterwards, its
    running statistics are set to exactly those of ``X`` (AdaBN), so the returned model in eval
    mode normalises with E's own statistics. Dropout stays off. All other weights are frozen.
    """
    adapted = copy.deepcopy(model)
    layers = bn_layers(adapted)
    if not layers:
        raise ValueError("model has no BatchNorm layers")
    for p in adapted.parameters():
        p.requires_grad_(False)
    params = [p for m in layers for p in (m.weight, m.bias)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    adapted.eval()
    for m in layers:
        m.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        information_maximisation_loss(adapted(X)).backward()
        opt.step()
    for p in adapted.parameters():
        p.requires_grad_(True)
    return adapt_bn_statistics(adapted, X)
