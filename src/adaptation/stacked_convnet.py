"""K ShallowConvNets trained as one stacked model: a faster path for the R11 ensemble's CNN members.

Why: R11 trains 5 ShallowConvNets per subject one after another (3 base seeds + 2 FTSurrogate seeds).
Each is small, so the GPU is mostly waiting on launches and on one large intermediate tensor. This
module trains all K members in one forward/backward pass per step.

What is the same as K separate :func:`src.adaptation.convnet.train_model` runs (by construction):
  * the function each member computes, its initial weights (copied from a ShallowConvNet built after
    ``set_deterministic(seed)``), its batch order (its own ``torch.Generator(seed)``), its dropout masks
    (each member's own global-RNG stream, swapped in around its dropout call), its augmentation draws
    (its own braindecode transform, applied to its own batch only);
  * the optimiser: AdamW and cosine annealing act element-wise, so one AdamW over stacked parameters is
    K independent AdamWs; the loss is the sum of the members' mean losses, so gradients do not mix.

What differs (so results are NOT bit-identical; quantified in docs/diag_gates_d1_d4.md, F1):
  * the temporal and spatial convolutions are applied as one combined kernel
    ``W[g, c, j] = sum_f spatial[g, f, c] * temporal[f, j]`` (the same linear map, different summation
    order), which avoids the (batch, filters, channels, time) intermediate;
  * the matrix products are batched over members, so other GEMM kernels run.
bf16 autocast was measured and dropped (no speed gain on this GPU; the run is bound by per-step
work, not FLOPs). torch.compile is unavailable here (no triton on Windows).
Every rounding difference is a different path through the same seeded training run.

Same rules as convnet.py: training sees the calibration trials only; no early stopping.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.adaptation.convnet import ShallowConvNet, TrainConfig, make_batches

Augment = Callable[[torch.Tensor], torch.Tensor] | None


class StackedShallowConvNet(nn.Module):
    """K independent ShallowConvNets with stacked parameters. Input ``(K, batch, chans, times)``."""

    def __init__(self, members: Sequence[ShallowConvNet]) -> None:
        super().__init__()
        if not members:
            raise ValueError("need at least one member")
        m0 = members[0]
        shapes = [(m.temporal_w.shape, m.spatial_w.shape, m.classifier.weight.shape) for m in members]
        if any(s != shapes[0] for s in shapes):
            raise ValueError("all members must have the same architecture")
        self.k = len(members)
        self.temporal_kernel = m0.temporal_kernel
        self.log_eps = m0.log_eps
        self.pool_size, self.pool_stride = m0.pool.kernel_size[0], m0.pool.stride[0]
        self.p_drop = m0.drop.p
        self.bn_eps, self.bn_momentum = m0.bn.eps, m0.bn.momentum

        def stack(get) -> nn.Parameter:
            return nn.Parameter(torch.stack([get(m).detach().clone() for m in members]))

        self.temporal_w = stack(lambda m: m.temporal_w)  # (K, F, J)
        self.temporal_b = stack(lambda m: m.temporal_b)  # (K, F)
        self.spatial_w = stack(lambda m: m.spatial_w)  # (K, G, F, C)
        self.bn_w = stack(lambda m: m.bn.weight)  # (K, G)
        self.bn_b = stack(lambda m: m.bn.bias)
        self.cls_w = stack(lambda m: m.classifier.weight)  # (K, O, G * n_out)
        self.cls_b = stack(lambda m: m.classifier.bias)  # (K, O)
        self.register_buffer("bn_mean", torch.cat([m.bn.running_mean.detach().clone() for m in members]))
        self.register_buffer("bn_var", torch.cat([m.bn.running_var.detach().clone() for m in members]))
        self.dropout_states: list[torch.Tensor] | None = None  # one RNG state per member while training

    def combined_kernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Temporal x spatial convolution as one kernel (K, G, C, J) and bias (K, G)."""
        w = torch.einsum("kgfc,kfj->kgcj", self.spatial_w, self.temporal_w)
        b = torch.einsum("kgfc,kf->kg", self.spatial_w, self.temporal_b)
        return w, b

    def features_pre_bn(self, x: torch.Tensor) -> torch.Tensor:
        """(K, B, C, T) -> (B, K * G, T'), the members' conv outputs side by side."""
        k, b, c, _ = x.shape
        w, bias = self.combined_kernel()
        g = w.shape[1]
        # grouped conv1d: input channels = K * C, groups = K, one combined kernel per member
        z = F.conv1d(x.permute(1, 0, 2, 3).reshape(b, k * c, -1), w.reshape(k * g, c, -1),
                     bias.reshape(k * g), groups=k)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(K, B, C, T) -> logits (K, B, n_classes)."""
        if x.shape[0] != self.k:
            raise ValueError(f"expected {self.k} member batches, got {x.shape[0]}")
        z = F.batch_norm(self.features_pre_bn(x), self.bn_mean, self.bn_var, self.bn_w.reshape(-1), self.bn_b.reshape(-1),
                         self.training, self.bn_momentum, self.bn_eps)
        z = torch.log(torch.clamp(F.avg_pool1d(z * z, self.pool_size, self.pool_stride), min=self.log_eps))
        bsz, _, n_out = z.shape
        z = z.reshape(bsz, self.k, -1, n_out).transpose(0, 1)  # (K, B, G, n_out), each member contiguous
        z = self._dropout(z.contiguous())
        return torch.einsum("kbi,koi->kbo", z.flatten(2), self.cls_w) + self.cls_b[:, None, :]

    def _dropout(self, z: torch.Tensor) -> torch.Tensor:
        """Each member's dropout drawn from its own RNG stream, exactly as in a separate run."""
        if not self.training or self.p_drop == 0:
            return z
        if self.dropout_states is None:
            return F.dropout(z, self.p_drop, True)
        get_state, set_state = _rng_accessors(z.device)
        outer = get_state()
        outs = []
        for i in range(self.k):
            set_state(self.dropout_states[i])
            outs.append(F.dropout(z[i], self.p_drop, True))
            self.dropout_states[i] = get_state()
        set_state(outer)
        return torch.stack(outs)

    def unstack(self, i: int, member: ShallowConvNet) -> ShallowConvNet:
        """Copy member ``i``'s weights and BN statistics into ``member`` (same architecture)."""
        with torch.no_grad():
            member.temporal_w.copy_(self.temporal_w[i])
            member.temporal_b.copy_(self.temporal_b[i])
            member.spatial_w.copy_(self.spatial_w[i])
            member.bn.weight.copy_(self.bn_w[i])
            member.bn.bias.copy_(self.bn_b[i])
            g = self.bn_w.shape[1]
            member.bn.running_mean.copy_(self.bn_mean[i * g:(i + 1) * g])
            member.bn.running_var.copy_(self.bn_var[i * g:(i + 1) * g])
            member.classifier.weight.copy_(self.cls_w[i])
            member.classifier.bias.copy_(self.cls_b[i])
        return member.eval()


def _rng_accessors(device: torch.device):
    """(get, set) for the generator that ``F.dropout`` draws from on ``device``."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return (lambda: torch.cuda.get_rng_state(idx)), (lambda s: torch.cuda.set_rng_state(s, idx))
    return torch.get_rng_state, torch.set_rng_state


def dropout_rng_state(device: torch.device) -> torch.Tensor:
    """The current state of the dropout generator on ``device`` (call right after building a member)."""
    return _rng_accessors(device)[0]()


def train_stacked(
    model: StackedShallowConvNet,
    X: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
    seeds: Sequence[int],
    augments: Sequence[Augment],
    dropout_states: Sequence[torch.Tensor] | None = None,
) -> StackedShallowConvNet:
    """Train the K members together on the same calibration trials (X, y), each with its own seed.

    ``seeds[i]`` drives member i's batch order (as ``train_model``'s ``seed``), ``augments[i]`` its
    augmentation (``None`` = none), ``dropout_states[i]`` its dropout stream (``None`` = shared stream).
    No ``max_shift``/``groups`` support: R11 uses neither.
    """
    k = model.k
    if not (len(seeds) == len(augments) == k) or (dropout_states is not None and len(dropout_states) != k):
        raise ValueError("need one seed, augmentation (and dropout state) per member")
    if cfg.max_shift:
        raise ValueError("train_stacked does not support max_shift")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y length mismatch")
    gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
    model.dropout_states = None if dropout_states is None else [s.clone() for s in dropout_states]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    model.train()
    for _ in range(cfg.epochs):
        per_member = [make_batches(X.shape[0], None, cfg.batch_size, g) for g in gens]
        for step in zip(*per_member):
            idx = [torch.as_tensor(b, device=X.device) for b in step]
            xb = torch.stack([X[i] if a is None else a(X[i]) for i, a in zip(idx, augments)])
            yb = torch.stack([y[i] for i in idx])  # (K, B) or (K, B, n_classes)
            logits = model(xb)
            loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten(0, 1), reduction="none")
            opt.zero_grad(set_to_none=True)
            loss.view(k, -1).mean(1).sum().backward()
            opt.step()
        sched.step()
    model.dropout_states = None
    model.eval()
    return model


@torch.no_grad()
def predict_proba_stacked(model: StackedShallowConvNet, X: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    """Softmax of every member on the same trials: (K, n, n_classes), float64."""
    model.eval()
    chunks = []
    for s in range(0, X.shape[0], batch_size):
        xb = X[s:s + batch_size]
        chunks.append(torch.softmax(model(xb.expand(model.k, *xb.shape)), dim=-1))
    return torch.cat(chunks, dim=1).double().cpu().numpy()
