"""Subject-specific ShallowConvNet + per-session normalisation at every layer.

Why this module exists (docs/convnet_session_norm.md has the full argument):
the project's one working principle is *per-session self-normalisation* — Riemannian
alignment whitens each session by its own mean covariance, label-free. The same principle
has exact counterparts in two other fields, and this module transplants both:

* vision domain adaptation — **AdaBN** (Li et al., 2016): keep the trained weights, but
  recompute every BatchNorm layer's statistics on the *unlabelled target domain*;
* speech recognition — **CMVN + speaker-adaptive training (SAT)**: normalise features per
  utterance/speaker *during training too*, so the model never learns domain-specific
  offsets it would later have to unlearn.

Mechanism, specific to ShallowConvNet: BatchNorm sits *before* the square -> mean-pool ->
log stage, so dividing a filter's output by its per-session standard deviation subtracts
a per-session constant from that filter's log-variance feature. A multiplicative,
band- and filter-specific session gain (electrode impedance, cap shift) is exactly such a
constant. Input alignment (one broadband whitening matrix) cannot remove a gain that
differs per band; per-session BN can.

Every statistic here is computed from one subject's own data, never across subjects
(the project's design rule). Nothing ever sees evaluation labels: :func:`adapt_bn_statistics`
takes no ``y``, and :func:`train_model` receives the calibration session only.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- model


class ShallowConvNet(nn.Module):
    """Schirrmeister et al. (2017) shallow ConvNet — a learned, end-to-end FBCSP.

    temporal conv (band-pass) -> spatial conv (CSP) -> BN -> square -> mean pool
    (variance) -> log -> dropout -> linear.  Input: ``(batch, n_chans, n_times)``.

    The two convolutions are implemented as ``unfold`` + ``einsum`` (mathematically the
    same as the reference Conv2d pair) because cuDNN's *deterministic* Conv2d backward was
    measured 3.6x slower on this machine, and the project requires bit-exact reruns.
    """

    def __init__(
        self,
        n_chans: int,
        n_classes: int,
        n_times: int,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
        log_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.temporal_kernel = temporal_kernel
        self.log_eps = log_eps
        self.temporal_w = nn.Parameter(torch.empty(n_filters, temporal_kernel))
        self.temporal_b = nn.Parameter(torch.zeros(n_filters))
        self.spatial_w = nn.Parameter(torch.empty(n_filters, n_filters, n_chans))
        nn.init.kaiming_uniform_(self.temporal_w, a=5**0.5)
        nn.init.kaiming_uniform_(self.spatial_w.view(n_filters, -1), a=5**0.5)
        self.bn = nn.BatchNorm1d(n_filters)
        self.pool = nn.AvgPool1d(pool_size, stride=pool_stride)
        self.drop = nn.Dropout(dropout)
        n_out = (n_times - temporal_kernel + 1 - pool_size) // pool_stride + 1
        if n_out < 1:
            raise ValueError(f"n_times={n_times} too short for the pooling configuration")
        self.classifier = nn.Linear(n_filters * n_out, n_classes)

    def features_pre_bn(self, x: torch.Tensor) -> torch.Tensor:
        """Output of temporal + spatial convolution, shape (batch, n_filters, n_times')."""
        patches = x.unfold(-1, self.temporal_kernel, 1)  # (B, C, T', K)
        temporal = torch.einsum("bctk,fk->bfct", patches, self.temporal_w)
        temporal = temporal + self.temporal_b[None, :, None, None]
        return torch.einsum("bfct,gfc->bgt", temporal, self.spatial_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bn(self.features_pre_bn(x))
        z = torch.log(torch.clamp(self.pool(z * z), min=self.log_eps))
        return self.classifier(self.drop(z).flatten(1))


# --------------------------------------------------------------------------- AdaBN


def bn_layers(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def adapt_bn_statistics(model: nn.Module, X: torch.Tensor) -> nn.Module:
    """AdaBN: set every BatchNorm layer's running statistics to those of ``X`` (one pass).

    Label-free by construction (no ``y`` argument). Weights, BN affine parameters and the
    classifier are untouched; dropout stays off while statistics are collected. ``X`` is
    forwarded as a single batch, so the result is the exact mean and unbiased variance of
    the BN inputs over ``X`` — independent of trial order. Returns the model in eval mode.
    """
    layers = bn_layers(model)
    if not layers:
        raise ValueError("model has no BatchNorm layers; AdaBN would be a no-op")
    if X.shape[0] < 2:
        raise ValueError("AdaBN needs at least 2 trials")
    saved = [m.momentum for m in layers]
    model.eval()
    for m in layers:
        m.reset_running_stats()
        m.momentum = None  # cumulative average; with one batch = exact batch moments
        m.train()
    model(X)
    for m, mom in zip(layers, saved):
        m.momentum = mom
    model.eval()
    return model


@torch.no_grad()
def predict_proba_groupwise(model: nn.Module, X: torch.Tensor, groups: np.ndarray) -> np.ndarray:
    """AdaBN applied separately per group (e.g. per recording run), then predict.

    Each trial is normalised with the statistics of its own group only — the test-time
    counterpart of :func:`train_model`'s ``group_batches=True`` (speaker-adaptive training).
    Modifies the model's BN buffers; callers should pass a copy if they need it afterwards.
    """
    groups = np.asarray(groups)
    out = np.zeros((X.shape[0], model.classifier.out_features), dtype=np.float64)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        sel = torch.as_tensor(idx, device=X.device)
        adapt_bn_statistics(model, X[sel])
        out[idx] = predict_proba(model, X[sel])
    return out


@torch.no_grad()
def predict_proba(model: nn.Module, X: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    model.eval()
    chunks = [torch.softmax(model(X[s : s + batch_size]), dim=1) for s in range(0, X.shape[0], batch_size)]
    return torch.cat(chunks).double().cpu().numpy()


def encode_labels(y: np.ndarray, classes: list[str]) -> np.ndarray:
    """Map string labels to indices in a *fixed* class order; raise on any unknown label.

    One shared order for both sessions prevents the silent int-vs-str / per-session
    encoding mismatch that would make ``y_pred == y_true`` meaningless.
    """
    lookup = {c: i for i, c in enumerate(classes)}
    unknown = set(np.unique(y)) - set(lookup)
    if unknown:
        raise ValueError(f"labels {sorted(unknown)} not in class list {classes}")
    return np.array([lookup[v] for v in y], dtype=np.int64)


# --------------------------------------------------------------------------- training


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 300
    batch_size: int = 48
    lr: float = 1e-3
    weight_decay: float = 1e-3
    max_shift: int = 0  # random circular time-shift augmentation (samples); 0 = off


def make_batches(
    n: int, groups: np.ndarray | None, batch_size: int, gen: torch.Generator
) -> list[np.ndarray]:
    """Shuffled index batches for one epoch.

    ``groups=None``: ordinary random batches (every batch mixes all runs).
    ``groups`` given: every batch contains trials from **one group only**
    (speaker-adaptive / domain-specific BN), so BN statistics during training are always
    within-group, exactly as at test time with :func:`predict_proba_groupwise`.
    Batches smaller than 2 are dropped (BN cannot normalise one trial).
    """
    if groups is None:
        perm = torch.randperm(n, generator=gen).numpy()
        batches = [perm[s : s + batch_size] for s in range(0, n, batch_size)]
    else:
        groups = np.asarray(groups)
        if groups.shape[0] != n:
            raise ValueError("groups must have one entry per trial")
        batches = []
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            idx = idx[torch.randperm(len(idx), generator=gen).numpy()]
            batches += [idx[s : s + batch_size] for s in range(0, len(idx), batch_size)]
        order = torch.randperm(len(batches), generator=gen).numpy()
        batches = [batches[i] for i in order]
    return [b for b in batches if len(b) >= 2]


def _random_shift(x: torch.Tensor, max_shift: int, gen: torch.Generator) -> torch.Tensor:
    if max_shift <= 0:
        return x
    shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0],), generator=gen)
    t = torch.arange(x.shape[-1])
    idx = (t[None, :] - shifts[:, None]) % x.shape[-1]
    return torch.gather(x, -1, idx.to(x.device)[:, None, :].expand(-1, x.shape[1], -1))


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
    seed: int,
    groups: np.ndarray | None = None,
) -> nn.Module:
    """Train on the calibration session only, for a fixed a-priori epoch budget.

    No early stopping: stopping on the evaluation session would select on its labels, and
    the budget is pre-registered instead (see the experiment's prereg file). Cosine-annealed
    AdamW, cross-entropy. ``groups`` switches on within-group batches (see make_batches).
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y length mismatch")
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(cfg.epochs):
        for b in make_batches(X.shape[0], groups, cfg.batch_size, gen):
            idx = torch.as_tensor(b, device=X.device)
            xb = _random_shift(X[idx], cfg.max_shift, gen)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), y[idx]).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def set_deterministic(seed: int) -> None:
    """Seed python/numpy/torch and force deterministic GPU kernels (bit-exact reruns)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
