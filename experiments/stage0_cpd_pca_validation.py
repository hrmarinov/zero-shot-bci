"""Stage 0 gate: does running CPD inside a PCA subspace (fit on the target
point cloud), then lifting the found rotation back to the full ambient
space, recover a real-world-relevant (D=253) planted rotation - the fix
coherent_point_drift.py's own docstring flagged after cpd_rigid_align was
found to fail completely at this dimensionality?

Same toy point-cloud construction as stage0_cpd_validation.py's benign and
outlier checks (a small-angle, FULL-RANK planted rotation - not a rotation
confined to some known subspace, since real session drift has no reason to
respect any particular subspace), at D=253 this time. The hope being
tested: even though the true rotation isn't literally low-rank, a small-
angle full-rank rotation's effect is concentrated where the data actually
has spread - the top PCA directions - so approximating it by a rotation
confined to those directions (identity elsewhere) might recover most of
the benefit without CPD ever having to solve the ill-posed, curse-of-
dimensionality-afflicted D=253 problem directly.

Checked against two baselines: no correction, and ICP run directly at the
full D=253 (what Phase 7 already tried on real data and found to be a
real-data null at 60.4%, worse than doing nothing) - the actually relevant
comparison, not just "is this better than nothing."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.linalg import expm

from src.adaptation.coherent_point_drift import cpd_align_via_pca_subspace
from src.adaptation.riemannian_icp import icp_align


def _anisotropic_clustered_sample(n_points: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    n_clusters = 4
    cluster_centers = rng.normal(scale=3.0, size=(n_clusters, dim))
    cluster_scales = rng.uniform(0.3, 1.5, size=(n_clusters, dim))
    assignments = rng.integers(0, n_clusters, size=n_points)
    return cluster_centers[assignments] + rng.normal(size=(n_points, dim)) * cluster_scales[assignments]


def _small_angle_rotation(dim: int, angle_scale: float, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(dim, dim)) * angle_scale
    skew = A - A.T
    return expm(skew)


def alignment_mse(corrected_source: np.ndarray, true_target_matched: np.ndarray) -> float:
    """Mean squared error against the TRUE matching target point (known
    here only because this is synthetic ground truth - real data never
    has this, which is exactly why this whole family of methods exists)."""
    return float(np.mean(np.sum((corrected_source - true_target_matched) ** 2, axis=1)))


def run(dim: int = 253, d_true: int = 20, n_points: int = 128, noise_std: float = 0.02, random_state: int = 3) -> None:
    """A first pass with _anisotropic_clustered_sample directly in D=253
    found the reduced-space CPD collapsing to the identity even at d=20 -
    traced (not assumed) to the generative model itself: that sampler
    spreads its cluster separation roughly evenly across all 253
    dimensions, so PCA's top components don't concentrate any more useful
    structure than the full space already has - not a fair test of "does
    reducing to a subspace where the *signal* actually concentrates help."
    Real tangent-space data is genuinely lower-rank than its 253 nominal
    dimensions (derived from one 22x22 matrix - the same reasoning already
    used to explain ComBat's failure), so this generates the true cluster
    structure in a KNOWN d_true-dimensional subspace (embedded via a random
    orthonormal map) with the remaining dim-d_true dimensions as pure,
    unstructured per-dimension noise - a fairer, more honest model of what
    this method actually needs to be true to work at all.
    """
    rng = np.random.default_rng(random_state)
    embedding = np.linalg.qr(rng.normal(size=(dim, d_true)))[0]  # (dim, d_true), orthonormal columns

    low_rank_source = _anisotropic_clustered_sample(n_points, d_true, rng)
    source = low_rank_source @ embedding.T + rng.normal(scale=noise_std, size=(n_points, dim))

    true_rotation_low_rank = _small_angle_rotation(d_true, angle_scale=0.15, rng=rng)
    true_rotation = np.eye(dim) + embedding @ (true_rotation_low_rank - np.eye(d_true)) @ embedding.T
    shuffle = rng.permutation(n_points)
    target = source[shuffle] @ true_rotation.T + rng.normal(scale=noise_std, size=(n_points, dim))
    true_target_matched = target[np.argsort(shuffle)]  # target's row i is source's row i's true match

    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    mse_none = alignment_mse(source_centered, true_target_matched - target.mean(axis=0))
    print(f"no correction:                 MSE={mse_none:.4f}")

    icp_result = icp_align(source, target, max_iters=100)
    corrected_icp = source_centered @ icp_result.rotation.T
    mse_icp = alignment_mse(corrected_icp, true_target_matched - target.mean(axis=0))
    print(f"ICP at full D={dim}:            MSE={mse_icp:.4f} "
          f"(this is what Phase 7 already tried on real data - real-data null there)")

    best_d, best_mse = None, np.inf
    for d in [5, 10, 20, 30, 50, 80]:
        R_full, cpd_result = cpd_align_via_pca_subspace(source_centered, target_centered, n_components=d, max_iters=100)
        corrected = source_centered @ R_full.T
        mse = alignment_mse(corrected, true_target_matched - target.mean(axis=0))
        print(f"PCA(d={d:2d}) + CPD, lifted to D={dim}: MSE={mse:.4f}  (reduced-space converged={cpd_result.converged})")
        if mse < best_mse:
            best_d, best_mse = d, mse

    print(f"\nBest PCA(d) + CPD: d={best_d}, MSE={best_mse:.4f}")
    assert best_mse < mse_none, f"PCA+CPD should beat no correction at its best d (none={mse_none:.4f}, best={best_mse:.4f})"
    if best_mse < mse_icp:
        print(f"PASSED: PCA({best_d})+CPD beats full-D ICP ({best_mse:.4f} < {mse_icp:.4f}) - "
              "worth testing on real IV-2a data next.")
    else:
        print(f"NOT PASSED: PCA+CPD's best result ({best_mse:.4f}) does not beat full-D ICP ({mse_icp:.4f}) - "
              "the PCA-subspace-approximation idea does not recover enough of the true rotation here to be "
              "worth testing on real data as an ICP replacement.")


if __name__ == "__main__":
    run()
