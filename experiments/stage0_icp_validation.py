"""Stage 0 gate for the ICP geometric primitives (src/adaptation/riemannian_icp.py).

Classic ICP correctness test, before touching anything EEG-related: build
a random point cloud, apply a *known* rotation (+ noise), verify ICP
recovers an alignment that undoes it - checked via (a) the recovered
rotation matrix being close to the true inverse rotation, and (b) mean
alignment error dropping close to the injected noise floor, not by eye.

Then a second, EEG-shaped check: two synthetic "sessions" of the same
synthetic subject (src/synthetic_eeg.py), with a *known* injected channel-
space transform applied to the second session only (simulating session
drift), checking whether recentering+rescaling+ICP recovers enough of the
alignment to reduce cross-session classification error versus no
alignment at all - still fully synthetic, still known ground truth, before
spending real IV-2a compute on it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import ortho_group

from src.adaptation.riemannian_icp import icp_align, procrustes_rotation, recenter


def _anisotropic_clustered_sample(n_points: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """An isotropic Gaussian point cloud looks statistically identical
    after *any* rotation - no rotation is identifiable from it at all, by
    construction. Real EEG tangent-space data isn't isotropic: different
    classes occupy different regions (that's the entire discriminative
    signal CSP/LDA rely on). Simulate that here with a few off-center,
    anisotropically-scaled clusters, so recovering "the" rotation is a
    well-posed problem, not an artifact of a degenerate test.
    """
    n_clusters = 4
    cluster_centers = rng.normal(scale=3.0, size=(n_clusters, dim))
    cluster_scales = rng.uniform(0.3, 1.5, size=(n_clusters, dim))
    assignments = rng.integers(0, n_clusters, size=n_points)
    return cluster_centers[assignments] + rng.normal(size=(n_points, dim)) * cluster_scales[assignments]


def _small_angle_rotation(dim: int, angle_scale: float, rng: np.random.Generator) -> np.ndarray:
    """A rotation close to identity (matrix exponential of a small random
    skew-symmetric generator) - modeling *realistic* session-to-session
    drift, which should be a modest perturbation, not an arbitrary large
    rotation. A fully random SO(dim) rotation (scipy's ortho_group) has no
    such "closeness to identity" guarantee and, tested directly, sent ICP
    into a bad local optimum from an identity start - a known, standard
    ICP limitation (it needs a reasonably close starting alignment), not a
    realistic stand-in for what cross-session drift actually looks like.
    """
    A = rng.normal(size=(dim, dim)) * angle_scale
    skew = A - A.T
    from scipy.linalg import expm
    return expm(skew)


def check_toy_point_cloud(dim: int = 10, n_points: int = 200, noise_std: float = 0.02, random_state: int = 0) -> None:
    """Standard ICP test: the *same* underlying points, reordered (breaking
    any known correspondence) and rotated (+ small noise) - isolates what
    ICP specifically adds over plain Procrustes (correspondence discovery
    via nearest-neighbor search), without conflating it with the
    irreducible noise floor two genuinely different distribution samples
    would introduce.
    """
    rng = np.random.default_rng(random_state)
    source = _anisotropic_clustered_sample(n_points, dim, rng)

    true_rotation = _small_angle_rotation(dim, angle_scale=0.15, rng=rng)
    shuffle = rng.permutation(n_points)
    target = source[shuffle] @ true_rotation.T + rng.normal(scale=noise_std, size=(n_points, dim))

    result = icp_align(source, target, max_iters=100)
    print(f"  ICP converged={result.converged} in {result.n_iters} iterations, "
          f"final MSE={result.mse_history[-1]:.5f}")
    print(f"  MSE history (first 5): {[round(m, 4) for m in result.mse_history[:5]]}")

    rotation_error = np.linalg.norm(result.rotation - true_rotation, ord="fro") / np.linalg.norm(true_rotation, ord="fro")
    print(f"  relative Frobenius error vs true rotation: {rotation_error:.4f}")

    noise_floor = noise_std**2 * dim  # expected per-point squared-error floor from injected noise alone
    assert result.mse_history[-1] < 10 * noise_floor, (
        f"ICP failed to converge to a good alignment (final MSE={result.mse_history[-1]:.4f}, "
        f"expected close to the noise floor {noise_floor:.4f})"
    )
    assert rotation_error < 0.05, f"recovered rotation too far from the true injected rotation (rel. error={rotation_error:.4f})"
    print("  [PASS] toy point-cloud ICP recovers the injected rotation and correspondence")


def check_procrustes_exact_recovery(dim: int = 8, n_points: int = 30, random_state: int = 1) -> None:
    """Sanity check on procrustes_rotation alone (no ICP loop, no
    correspondence search): with EXACTLY matched pairs and zero noise, the
    closed-form solution must recover the true rotation essentially
    exactly."""
    rng = np.random.default_rng(random_state)
    source = rng.normal(size=(n_points, dim))
    source_centered, _ = recenter(source)
    true_rotation = ortho_group.rvs(dim, random_state=random_state)
    target_centered = source_centered @ true_rotation.T

    recovered = procrustes_rotation(source_centered, target_centered)
    error = np.linalg.norm(recovered - true_rotation, ord="fro")
    print(f"  exact-correspondence Procrustes recovery error: {error:.2e}")
    assert error < 1e-6, f"closed-form Procrustes should recover an exact rotation from exact correspondences (error={error:.2e})"
    print("  [PASS] procrustes_rotation exactly recovers a known rotation from matched pairs")


def run() -> None:
    print("=== Procrustes exact-recovery check (no ICP loop) ===")
    check_procrustes_exact_recovery()

    print("\n=== Toy point-cloud ICP check (distribution alignment, not exact correspondence) ===")
    check_toy_point_cloud()

    print("\nICP geometric primitives validated on synthetic ground truth - "
          "safe to proceed to synthetic EEG session-drift validation.")


if __name__ == "__main__":
    run()
