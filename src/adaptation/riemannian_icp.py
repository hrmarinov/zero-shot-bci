"""Within-subject cross-session realignment via Iterative Closest Point (ICP).

Transplanted from robotics / 3D scanning / computer vision (Besl & McKay,
1992) - the standard algorithm for aligning two point clouds representing
the *same underlying object*, captured at different times/sensor
positions, with no known point-to-point correspondence. That's exactly
this problem: a calibration session's trial cloud and an eval session's
trial cloud represent the same subject's neural signature, shifted by
session-specific drift, with no eval-side labels available to anchor a
supervised correspondence.

Operates entirely in log-Euclidean tangent-space coordinates (same
convention as spd_shrinkage.py, geometric_correction.py, and
prototypical_network.py - reference=I, metric="logeuclid"), which turns
the SPD manifold into an ordinary Euclidean vector space, so "point cloud"
alignment is literal: standard rigid-transform registration (Procrustes/
Kabsch), not a special manifold-valued variant.

Never uses another subject's data anywhere - calibration and eval trials
of the *same* subject are the only two point clouds involved, matching the
principle that fixed docs/progress_and_direction.md's prototypical-network
training (never directly compare two different subjects' data inside one
computation).

Three composable pieces, each independently testable:

1. **Recentering + rescaling** (unsupervised, no labels needed on either
   side): subtract each session's own mean tangent vector, then match
   dispersion (spread) between sessions. The tangent-space analogue of the
   first two steps of Riemannian Procrustes Analysis (Rodrigues et al.) -
   RiemannianAlignment (riemannian_align.py) already does an affine-
   invariant-metric version of recentering at the raw-signal level; this
   is the same idea expressed natively in the tangent-space framework the
   rest of this module operates in.
2. **Supervised rotation** (needs labels/pseudo-labels on both sides):
   solve the closed-form Procrustes rotation that best aligns matched
   class-mean pairs between calibration and eval. Classic RPA's rotation
   step, but applied within-subject (calibration -> eval), never across
   subjects.
3. **ICP** (needs no labels at all): iteratively alternates finding
   nearest-neighbor correspondences between the (already recentered/
   rescaled) point clouds and re-solving the Procrustes rotation from
   those correspondences, until convergence. Replaces step 2's dependency
   on labels with a data-driven, self-discovered correspondence - at the
   cost of being sensitive to a poor starting alignment, which is exactly
   why step 1 exists: to give ICP a good initial guess before it takes
   over.
"""

from dataclasses import dataclass

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.geometry.tangentspace import tangent_space, untangent_space

EPS = 1e-12


def trial_tangent_vectors(X: np.ndarray, estimator: str = "oas") -> np.ndarray:
    """Per-trial covariances -> log-Euclidean tangent vectors (reference=I)."""
    n_channels = X.shape[1]
    reference = np.eye(n_channels)
    covs = Covariances(estimator=estimator).fit_transform(X)
    return tangent_space(covs, reference, metric="logeuclid")


def recenter(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the point cloud's own mean. Returns (centered, mean)."""
    mean = vecs.mean(axis=0)
    return vecs - mean, mean


def dispersion(centered_vecs: np.ndarray) -> float:
    """Mean squared norm of an already-centered point cloud - the tangent-
    space analogue of RPA's dispersion statistic."""
    return float(np.mean(np.sum(centered_vecs**2, axis=1)))


def rescale(centered_vecs: np.ndarray, target_dispersion: float) -> np.ndarray:
    """Scale a centered point cloud so its own dispersion matches
    target_dispersion - unsupervised, no correspondence needed."""
    own_dispersion = dispersion(centered_vecs)
    scale = np.sqrt(target_dispersion / (own_dispersion + EPS))
    return centered_vecs * scale


def procrustes_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Closed-form orthogonal Procrustes / Kabsch solution: the rotation R
    minimizing ||target - source @ R.T||^2, given MATCHED (same-order,
    already-centered) point pairs. Standard SVD solution: R = V @ U.T where
    U, S, Vt = svd(source.T @ target) (with a reflection-correction if
    det(V @ U.T) < 0, to keep R a proper rotation, not a reflection).
    """
    if source.shape != target.shape:
        raise ValueError(f"source and target must have matching shape for Procrustes, got {source.shape} vs {target.shape}")
    cross_cov = source.T @ target
    U, _, Vt = np.linalg.svd(cross_cov)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.eye(U.shape[0])
    correction[-1, -1] = d
    R = Vt.T @ correction @ U.T
    return R


@dataclass
class ICPResult:
    rotation: np.ndarray  # (D, D) - maps a centered source point toward the target frame
    source_mean: np.ndarray
    target_mean: np.ndarray
    n_iters: int
    converged: bool
    mse_history: list[float]

    def transform(self, vecs: np.ndarray) -> np.ndarray:
        """Map new source-space tangent vectors into the target frame."""
        centered = vecs - self.source_mean
        return centered @ self.rotation.T + self.target_mean


def icp_align(
    source: np.ndarray,
    target: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-6,
    init_rotation: np.ndarray | None = None,
) -> ICPResult:
    """Iterative Closest Point: alternates nearest-neighbor correspondence
    and closed-form Procrustes re-fitting until the alignment MSE stops
    improving. source and target need NOT have the same number of points
    or any known correspondence - that's the whole point of ICP versus a
    single supervised Procrustes call.

    Brute-force nearest-neighbor search (fine at the trial counts here -
    tens to a few hundred points; not meant to scale to anything larger).
    """
    source_centered, source_mean = recenter(source)
    target_centered, target_mean = recenter(target)
    dim = source_centered.shape[1]

    R = init_rotation if init_rotation is not None else np.eye(dim)
    mse_history = []
    converged = False
    n_iters = 0

    for iteration in range(max_iters):
        transformed = source_centered @ R.T
        # nearest neighbor in target for each transformed source point
        dists = np.linalg.norm(transformed[:, None, :] - target_centered[None, :, :], axis=2)
        nn_idx = dists.argmin(axis=1)
        matched_target = target_centered[nn_idx]

        mse = float(np.mean(np.sum((transformed - matched_target) ** 2, axis=1)))
        mse_history.append(mse)
        n_iters = iteration + 1

        R = procrustes_rotation(source_centered, matched_target)

        if len(mse_history) >= 2 and abs(mse_history[-2] - mse_history[-1]) < tol:
            converged = True
            break

    return ICPResult(
        rotation=R, source_mean=source_mean, target_mean=target_mean,
        n_iters=n_iters, converged=converged, mse_history=mse_history,
    )


def supervised_rotation_from_labels(
    source_vecs: np.ndarray, source_labels: np.ndarray,
    target_vecs: np.ndarray, target_labels: np.ndarray,
    classes: list[str],
) -> np.ndarray:
    """RPA-style rotation step: solve Procrustes using matched *class-mean*
    landmark pairs, rather than ICP's self-discovered point correspondences.
    Needs labels (or pseudo-labels) on both source and target - the
    within-subject analogue of classic RPA's cross-subject rotation, using
    a subject's own calibration labels and some eval labels/pseudo-labels
    as the matched pairs instead of two different subjects' class means.
    """
    source_means = np.stack([source_vecs[source_labels == c].mean(axis=0) for c in classes])
    target_means = np.stack([target_vecs[target_labels == c].mean(axis=0) for c in classes])
    source_centered = source_means - source_means.mean(axis=0)
    target_centered = target_means - target_means.mean(axis=0)
    return procrustes_rotation(source_centered, target_centered)
