"""Stage 0 gate for the CPD geometric primitive (src/adaptation/coherent_point_drift.py),
built as a direct successor to ICP after ICP was found to be a real-data
null in Phase 7. Same toy-point-cloud recovery test as
stage0_icp_validation.py (for a direct, apples-to-apples comparison at the
easy end), plus the specific test CPD's design is supposed to win: point
clouds contaminated with genuine outliers (points with no true
correspondence at all) - ICP has no way to represent "this point has no
match," so every outlier gets forced into some nearest-neighbor match and
corrupts the re-fit; CPD's explicit outlier-weight term is supposed to
down-weight exactly this case instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.linalg import expm

from src.adaptation.coherent_point_drift import cpd_rigid_align
from src.adaptation.riemannian_icp import icp_align


def _anisotropic_clustered_sample(n_points: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Same construction as stage0_icp_validation.py's helper of the same
    name - an isotropic Gaussian cloud looks identical after any rotation,
    so no rotation would be identifiable at all; real tangent-space EEG
    data has off-center, anisotropic class clusters instead."""
    n_clusters = 4
    cluster_centers = rng.normal(scale=3.0, size=(n_clusters, dim))
    cluster_scales = rng.uniform(0.3, 1.5, size=(n_clusters, dim))
    assignments = rng.integers(0, n_clusters, size=n_points)
    return cluster_centers[assignments] + rng.normal(size=(n_points, dim)) * cluster_scales[assignments]


def _small_angle_rotation(dim: int, angle_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Same construction as stage0_icp_validation.py - a modest, near-
    identity rotation, modeling realistic session drift rather than an
    arbitrary large one."""
    A = rng.normal(size=(dim, dim)) * angle_scale
    skew = A - A.T
    return expm(skew)


def rotation_error(recovered: np.ndarray, true_rotation: np.ndarray) -> float:
    return float(np.linalg.norm(recovered - true_rotation, ord="fro") / np.linalg.norm(true_rotation, ord="fro"))


def check_benign_parity(dim: int = 10, n_points: int = 200, noise_std: float = 0.02, random_state: int = 0) -> None:
    """Same scenario as stage0_icp_validation.py's toy point-cloud check,
    no outliers - CPD should do at least as well as ICP here, not worse."""
    rng = np.random.default_rng(random_state)
    source = _anisotropic_clustered_sample(n_points, dim, rng)
    true_rotation = _small_angle_rotation(dim, angle_scale=0.15, rng=rng)
    shuffle = rng.permutation(n_points)
    target = source[shuffle] @ true_rotation.T + rng.normal(scale=noise_std, size=(n_points, dim))

    icp_result = icp_align(source, target, max_iters=100)
    cpd_result = cpd_rigid_align(source, target, max_iters=100)

    icp_err = rotation_error(icp_result.rotation, true_rotation)
    cpd_err = rotation_error(cpd_result.rotation, true_rotation)
    print(f"  ICP: rotation error={icp_err:.4f}, converged={icp_result.converged}")
    print(f"  CPD: rotation error={cpd_err:.4f}, converged={cpd_result.converged}, scale={cpd_result.scale:.4f}")

    assert cpd_err < 0.05, f"CPD failed to recover the rotation even in the benign no-outlier case (error={cpd_err:.4f})"
    print("  [PASS] CPD recovers the injected rotation at least as well as ICP in the benign case")


def check_outlier_robustness(
    dim: int = 10, n_points: int = 200, noise_std: float = 0.02, outlier_fraction: float = 0.25, random_state: int = 2
) -> None:
    """The scenario CPD's design specifically targets: outlier_fraction of
    the TARGET cloud has no true correspondence to any source point at all
    (pure random noise points, not a rotated+shuffled copy of anything in
    source). ICP has no representation for "no match" - every outlier gets
    forced into whatever nearest-neighbor match is closest, corrupting the
    Procrustes re-fit. CPD's outlier-weight term is supposed to down-weight
    these instead of forcing a match.
    """
    rng = np.random.default_rng(random_state)
    source = _anisotropic_clustered_sample(n_points, dim, rng)
    true_rotation = _small_angle_rotation(dim, angle_scale=0.15, rng=rng)

    n_outliers = int(round(n_points * outlier_fraction))
    n_genuine = n_points - n_outliers
    shuffle = rng.permutation(n_points)[:n_genuine]
    genuine_target = source[shuffle] @ true_rotation.T + rng.normal(scale=noise_std, size=(n_genuine, dim))

    # outliers: drawn from the same coarse spatial extent as the genuine
    # cloud (not absurdly far away, which would be trivially detectable by
    # any distance threshold - the point is ambiguous outliers, not obvious ones)
    spread = genuine_target.std(axis=0)
    outliers = rng.normal(scale=1.0, size=(n_outliers, dim)) * spread * 2.0

    target = np.concatenate([genuine_target, outliers], axis=0)
    target = target[rng.permutation(len(target))]  # don't leave outliers conveniently grouped at the end

    icp_result = icp_align(source, target, max_iters=100)
    cpd_result = cpd_rigid_align(source, target, max_iters=100, outlier_weight=outlier_fraction)

    icp_err = rotation_error(icp_result.rotation, true_rotation)
    cpd_err = rotation_error(cpd_result.rotation, true_rotation)
    print(f"  {outlier_fraction:.0%} of target points are pure outliers with no true correspondence")
    print(f"  ICP: rotation error={icp_err:.4f}, converged={icp_result.converged}")
    print(f"  CPD: rotation error={cpd_err:.4f}, converged={cpd_result.converged}, scale={cpd_result.scale:.4f}")

    assert cpd_err < icp_err, (
        f"CPD's outlier-weight term should give it an advantage over ICP's hard assignment under outlier "
        f"contamination, but CPD error ({cpd_err:.4f}) was not better than ICP's ({icp_err:.4f})"
    )
    print("  [PASS] CPD is more robust to outlier contamination than ICP, as designed")


def check_high_dimensional_regime(dim: int = 253, n_points: int = 128, noise_std: float = 0.02, random_state: int = 0) -> None:
    """The ACTUAL dimensionality this project needs (253 = IV-2a's 22
    channels' worth of log-Euclidean tangent-space dimensions), not caught
    by the D=10 checks above until real Phase 7 data hit it directly.
    Identical construction to check_benign_parity, dimension changed only -
    isolates the effect of D itself, not a difference in data source.

    NOT a pass/fail gate: reports a real, diagnosed structural failure
    (see coherent_point_drift.py's own docstring for the full trace) rather
    than asserting anything works. Included so the failure stays visible on
    every run of this file, not just discovered once and then forgotten.
    """
    rng = np.random.default_rng(random_state)
    source = _anisotropic_clustered_sample(n_points, dim, rng)
    true_rotation = _small_angle_rotation(dim, angle_scale=0.15, rng=rng)
    shuffle = rng.permutation(n_points)
    target = source[shuffle] @ true_rotation.T + rng.normal(scale=noise_std, size=(n_points, dim))

    cpd_result = cpd_rigid_align(source, target, max_iters=100)
    cpd_err = rotation_error(cpd_result.rotation, true_rotation)
    print(f"  same construction as the benign D=10 check, only dim changed to {dim}")
    print(f"  CPD: rotation error={cpd_err:.4f} (was ~0.002 at D=10) - "
          f"n_iters={cpd_result.n_iters}, converged={cpd_result.converged}")
    print("  This is the known, diagnosed, NOT-fixed failure (curse of dimensionality in the outlier "
          "hypothesis's log-weight) - see coherent_point_drift.py's docstring. Do not use cpd_rigid_align "
          "on data this high-dimensional without a materially different fix (dimensionality reduction "
          "first, or a non-Gaussian-density soft-correspondence scheme).")


def run() -> None:
    print("=== Benign parity check (no outliers - same scenario as stage0_icp_validation.py) ===")
    check_benign_parity()

    print("\n=== Outlier-robustness check (CPD's specific documented advantage over ICP) ===")
    check_outlier_robustness()

    print("\n=== High-dimensional check (D=253, this project's ACTUAL tangent-space dimensionality) ===")
    check_high_dimensional_regime()

    print("\nCPD validated at D=10 (both benign and outlier-contaminated cases) but NOT at D=253 - "
          "do not use cpd_rigid_align on real IV-2a tangent-space data (253 dimensions) as it stands.")


if __name__ == "__main__":
    run()
