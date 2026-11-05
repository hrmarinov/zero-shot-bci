"""Orthogonal congruence fit between two sets of matched SPD matrices.

Solves  argmin_{R in O(n)}  sum_k || R S_k R^T - M_k ||_F^2  for matched pairs (S_k, M_k),
e.g. per-class mean covariances of two sessions after per-session whitening.

Why not ``pyriemann.optimization.grassmann._get_rotation_manifold``: on 22-channel planted
rotations it stalls in local minima (max residual 0.13-0.23 even with 10 restarts, measured
2026-11-04), which would *understate* an oracle upper bound. Here R = expm(A - A^T) is
optimised with Adam from many restarts in one batch, once for SO(n) and once for the
reflected coset, and the lowest-cost solution is returned; planted 22-channel rotations are
recovered to ~1e-4 (tests/test_rotation_oracle.py).
"""

from __future__ import annotations

import numpy as np
import torch


def _congruence_cost(R: torch.Tensor, S: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    RS = R[:, None] @ S[None] @ R[:, None].transpose(-1, -2)
    return ((RS - M[None]) ** 2).sum(dim=(1, 2, 3))


def _fit_special_orthogonal(S: torch.Tensor, M: torch.Tensor, n_restarts: int, steps: int, seed: int):
    n = S.shape[-1]
    gen = torch.Generator().manual_seed(seed)
    A = 0.5 * torch.randn(n_restarts, n, n, generator=gen, dtype=torch.float64)
    A[0] = 0.0  # one restart from the identity
    A.requires_grad_(True)
    opt = torch.optim.Adam([A], lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        cost = _congruence_cost(torch.matrix_exp(A - A.transpose(1, 2)), S, M)
        opt.zero_grad()
        cost.sum().backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        best = int(_congruence_cost(torch.matrix_exp(A - A.transpose(1, 2)), S, M).argmin())
    return _polish(A[best : best + 1].detach().clone(), S, M, steps)


def _polish(A: torch.Tensor, S: torch.Tensor, M: torch.Tensor, steps: int):
    """Refine the best restart with L-BFGS (the batched Adam phase only finds the basin)."""
    A.requires_grad_(True)
    opt = torch.optim.LBFGS([A], lr=1.0, max_iter=steps, tolerance_grad=1e-14,
                            tolerance_change=1e-18, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        cost = _congruence_cost(torch.matrix_exp(A - A.transpose(1, 2)), S, M).sum()
        cost.backward()
        return cost

    opt.step(closure)
    with torch.no_grad():
        R = torch.matrix_exp(A - A.transpose(1, 2))
        return R[0].numpy(), float(_congruence_cost(R, S, M)[0])


def fit_orthogonal_congruence(
    source: np.ndarray, target: np.ndarray, n_restarts: int = 32, steps: int = 1500, seed: int = 0
) -> tuple[np.ndarray, float]:
    """Return (R, cost) with R orthogonal minimising sum_k ||R source_k R^T - target_k||_F^2.

    ``source`` and ``target``: (n_pairs, n, n) SPD matrices, matched by index. Searches
    both det(R)=+1 and det(R)=-1 (via a fixed reflection P: R = R' P).
    """
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError(f"need matched (k, n, n) stacks, got {source.shape} and {target.shape}")
    S = torch.as_tensor(source, dtype=torch.float64)
    M = torch.as_tensor(target, dtype=torch.float64)
    R_pos, c_pos = _fit_special_orthogonal(S, M, n_restarts, steps, seed)
    P = torch.eye(S.shape[-1], dtype=torch.float64)
    P[0, 0] = -1.0
    R_neg, c_neg = _fit_special_orthogonal(P @ S @ P, M, n_restarts, steps, seed + 1)
    R_neg = R_neg @ P.numpy()
    return (R_pos, c_pos) if c_pos <= c_neg else (R_neg, c_neg)
