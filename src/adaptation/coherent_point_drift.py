"""Within-subject cross-session realignment via Coherent Point Drift (CPD;
Myronenko & Song 2010) - a direct successor to riemannian_icp.py's ICP,
built specifically because ICP was found to be a real-data null in Phase 7
("unsupervised nearest-neighbor correspondence-finding across real, noisy,
4-class EEG trial clouds is evidently not reliable enough for ICP's
iterative refinement to converge somewhere useful"). CPD's whole documented
advantage over ICP, in the point-cloud-registration literature it comes
from, is exactly the failure mode ICP hit here: ICP's hard nearest-
neighbor correspondence has no way to express "I'm not sure which point
this corresponds to," so a single wrong match at low SNR pulls the entire
re-fit in the wrong direction. CPD replaces that hard assignment with a
*soft*, probabilistic one (every point contributes to every match,
weighted by a Gaussian Mixture Model likelihood, with an explicit uniform
"outlier" component so ambiguous/noisy points don't have to be forced into
any single-correspondence at all) - the standard fix for exactly this
brittleness in the registration literature.

A public implementation (pycpd) exists but was found unusable as-is: its
RigidRegistration hard-codes a restriction to 2D/3D point clouds (an
implementation choice for its typical computer-vision use cases, not a
mathematical limitation of CPD itself - the EM derivation below has no
dependence on dimensionality). This project's points are log-Euclidean
tangent vectors of SPD covariance matrices - 253-dimensional for IV-2a's
22 channels - so pycpd could not be used and this is a direct, dimension-
general implementation of the same published EM algorithm instead.

Same conventions as riemannian_icp.py: operates on tangent-space vectors,
never uses another subject's data, and is meant to be tested downstream of
recenter+rescale (this project's established, validated-safe initialization
step), the same way ICP was.
"""

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class CPDResult:
    rotation: np.ndarray  # (D, D)
    scale: float
    source_mean: np.ndarray
    target_mean: np.ndarray
    n_iters: int
    converged: bool
    sigma2_history: list[float]

    def transform(self, vecs: np.ndarray) -> np.ndarray:
        """Map new source-space tangent vectors into the target frame."""
        centered = vecs - self.source_mean
        return self.scale * (centered @ self.rotation.T) + self.target_mean


def cpd_align_via_pca_subspace(
    source: np.ndarray,
    target: np.ndarray,
    n_components: int,
    max_iters: int = 100,
    tol: float = 1e-6,
    outlier_weight: float = 0.1,
) -> tuple[np.ndarray, "CPDResult"]:
    """Runs cpd_rigid_align inside a PCA subspace fit on `target` - the
    dimensionality reduction flagged, but not attempted, in
    cpd_rigid_align's own docstring after it was found to fail entirely at
    D=253. Does NOT reduce what the classifier ultimately sees: the
    found d x d rotation is extended to a full (D, D) transform that acts
    as that rotation *within* the PCA subspace and as the identity on
    everything orthogonal to it (R_full = I + V @ (R_d - I_d) @ V.T, where
    V's columns are the subspace's orthonormal basis vectors) - a valid,
    well-defined orthogonal transform of the full ambient space, not an
    approximation that discards the other D-d dimensions. This keeps
    "does PCA hurt classification" (already found to hurt once in this
    project - see docs/progress_and_direction.md's dimensionality-reduction
    section) separate from "does CPD's correction help", by only ever
    using the reduced subspace to *find* a rotation, never to represent the
    data the downstream classifier actually sees.

    Returns (R_full, cpd_result): R_full is the (D, D) lifted rotation to
    apply to *centered* source vectors; cpd_result is the underlying
    reduced-space CPDResult, kept for its diagnostics (converged, sigma2
    history) even though its own .rotation/.transform are d-dimensional
    and not what should be applied to full-dimensional data directly.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    pca.fit(target)
    V = pca.components_.T  # (D, d), orthonormal columns

    source_reduced = source @ V
    target_reduced = target @ V
    cpd_result = cpd_rigid_align(source_reduced, target_reduced, max_iters, tol, outlier_weight, estimate_scale=False)

    D = source.shape[1]
    R_full = np.eye(D) + V @ (cpd_result.rotation - np.eye(n_components)) @ V.T
    return R_full, cpd_result


def cpd_rigid_align(
    source: np.ndarray,
    target: np.ndarray,
    max_iters: int = 100,
    tol: float = 1e-6,
    outlier_weight: float = 0.1,
    estimate_scale: bool = False,
) -> CPDResult:
    """Rigid Coherent Point Drift: models target (X) as drawn from a
    Gaussian Mixture whose M centroids are the transformed source (Y)
    points, sharing one variance sigma2, plus a uniform outlier component
    weighted by outlier_weight - then alternates (E) computing each
    target point's soft-assignment posterior over source centroids, and
    (M) re-solving the weighted rigid transform (rotation, scale,
    translation) and sigma2 from those soft assignments, until sigma2
    stops improving. source and target need NOT have the same number of
    points or any known correspondence, same as icp_align.

    Follows Myronenko & Song (2010)'s closed-form rigid-CPD M-step
    (weighted Kabsch/Procrustes via SVD, reflection-corrected the same way
    riemannian_icp.py's procrustes_rotation is), generalized to arbitrary
    dimensionality D (not hard-restricted to 2D/3D the way pycpd's
    RigidRegistration is - see this module's own docstring for why pycpd
    could not be used directly).

    Three real problems found and fixed at D=253 (this project's actual
    tangent-space dimensionality), not caught by stage0_cpd_validation.py's
    original D=10 toy test:
    1. The Gaussian kernel exp(-||x-y||^2/(2*sigma2)) as literally stated
       in the paper underflows to numerical noise once D gets this large -
       squared distances scale with D, so "close" and "far" point pairs
       end up differing by factors like 10^40 in raw likelihood, collapsing
       the E-step's supposedly *soft* assignment into something worse than
       ICP's hard one (and numerically unstable: the responsibility sums
       came out essentially zero at low point counts, corrupting every
       downstream M-step statistic). Fixed by normalizing the squared
       distance by D before dividing by sigma2 - consistent with sigma2
       itself already being estimated as a *per-dimension* average
       variance, not a total one.
    2. That fix alone still wasn't enough: the outlier term's own
       normalization constant, (2*pi*sigma2_total)^(D/2), is *itself*
       numerically catastrophic at D=253 - checked directly: with
       sigma2_total around 10, this evaluates to ~1e230, completely
       swamping every genuine-match term (which is always <= 1 by
       construction) and driving every responsibility to ~0 regardless of
       true match quality. This isn't a parametrization mistake, it's the
       well-known "curse of dimensionality" behavior of Gaussian
       normalization constants in high dimensions - unavoidable if the
       E-step is computed in raw probability space at all. Fixed with the
       standard remedy: do the whole E-step's normalization in log-space
       (log-sum-exp over the genuine-match log-likelihoods and the
       outlier's log-weight together, subtracting the shared max before
       exponentiating), which is exact regardless of how extreme the raw
       values would have been.
    3. estimate_scale=True let scale drift smoothly to zero given few
       source points at this dimensionality (verified: 16 points trying to
       constrain a rigid transform in 253 dimensions is drastically
       underdetermined - a general D-dimensional rotation alone has
       D*(D-1)/2 ~= 32000 degrees of freedom). riemannian_icp.py's own
       procrustes_rotation never estimates scale for exactly this kind of
       reason - recenter+rescale (fit beforehand, unsupervised, on much
       more data) already matches dispersion between the two sessions, so
       CPD only needs to find the rotation correction on top of that, not
       re-derive scale from a handful of points. estimate_scale now
       defaults to False to match.

    A FOURTH problem was found after all three fixes above, and it is NOT
    fixed - this function still does not work at D~253 (verified: the same
    toy point-cloud construction stage0_cpd_validation.py already validates
    cleanly at D=10 was re-run at D=253 with nothing else changed, and
    rotation error went from 0.002 to 1.4 - essentially no correspondence
    recovered at all, confirmed to be dimension-driven and not an artifact
    of real EEG data specifically). The cause, traced directly rather than
    guessed: even with the log-space fix above removing the numerical
    overflow/underflow, the outlier hypothesis's log-weight
    ((D/2)*log(2*pi*sigma2_total)) is *mathematically*, not just
    numerically, hundreds of nats larger than any genuine match's
    log-likelihood at this dimensionality, for any sigma2_total this
    module's own initialization produces - the model correctly (given its
    own assumptions) concludes every point is an outlier. This is a real
    instance of the curse of dimensionality: in ~253 dimensions,
    concentration of measure makes essentially all pairwise distances
    converge toward a similar "typical" value, so no single correspondence
    can look distinctively better than pure noise under a literal Gaussian-
    density comparison, regardless of tuning outlier_weight (its own
    contribution to the log-weight is negligible next to the (D/2)*log(...)
    term). Fixing this for real would need a materially different
    approach - e.g. reducing to a lower-dimensional subspace before
    computing correspondences (the same move state_space_flow.py's PCA
    step makes), or a different soft-correspondence formulation not based
    on an absolute Gaussian density comparison (optimal-transport/Sinkhorn-
    based soft assignment was flagged in this project's own literature
    review as a documented alternative) - not attempted here. Do not use
    this function on data with more than a few tens of dimensions without
    building and validating one of those first.
    """
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    M, D = source_centered.shape
    N = target_centered.shape[0]

    R = np.eye(D)
    s = 1.0
    t = np.zeros(D)
    # standard CPD initialization: sigma2 = mean squared distance between
    # every source and every target point, normalized by (N*M*D)
    diff = target_centered[None, :, :] - source_centered[:, None, :]
    sigma2 = float(np.sum(diff**2)) / (N * M * D)

    sigma2_history = []
    converged = False
    n_iters = 0
    w = outlier_weight

    for iteration in range(max_iters):
        transformed = s * (source_centered @ R.T) + t  # (M, D)

        # E-step: P[m, n] = posterior that target point n corresponds to
        # transformed source centroid m. sigma2 (this function's variable)
        # is a *per-dimension average* variance (that's what its init and
        # the M-step's own update formula both produce) - the paper's
        # Gaussian kernel needs the *total* isotropic variance across all D
        # dimensions, sigma2*D, not sigma2 itself.
        #
        # Computed entirely in log-space, then normalized via log-sum-exp:
        # both the raw exponent and the outlier term's own normalization
        # constant are individually unrepresentable in ordinary float64 at
        # D=253 (astronomically large or small), even though their *ratio*
        # (what P actually needs) is always a well-behaved probability -
        # see this function's own docstring for the numbers that showed this.
        sigma2_total = sigma2 * D
        sq_dists = np.sum((target_centered[None, :, :] - transformed[:, None, :]) ** 2, axis=2)  # (M, N)
        log_kernel = -sq_dists / (2 * sigma2_total)  # (M, N)
        log_outlier = (D / 2) * np.log(2 * np.pi * sigma2_total) + np.log(w / (1 - w)) + np.log(M / N)  # scalar

        stacked = np.concatenate([log_kernel, np.full((1, N), log_outlier)], axis=0)  # (M+1, N)
        log_norm = np.logaddexp.reduce(stacked, axis=0, keepdims=True)  # (1, N)
        P = np.exp(log_kernel - log_norm)  # (M, N)

        # M-step: weighted rigid transform from the soft correspondences.
        N_p = P.sum()
        P_row_sums = P.sum(axis=1)  # (M,) - each source point's total responsibility
        P_col_sums = P.sum(axis=0)  # (N,) - each target point's total responsibility

        mu_x = (target_centered * P_col_sums[:, None]).sum(axis=0) / (N_p + EPS)
        mu_y = (source_centered * P_row_sums[:, None]).sum(axis=0) / (N_p + EPS)
        X_hat = target_centered - mu_x
        Y_hat = source_centered - mu_y

        A = X_hat.T @ P.T @ Y_hat  # (D, D)
        U, Sigma, Vt = np.linalg.svd(A)
        d = np.sign(np.linalg.det(U @ Vt))
        C = np.eye(D)
        C[-1, -1] = d
        R = U @ C @ Vt

        denom_s = np.trace(np.diag(Sigma) @ C)
        y_hat_weighted_var = np.sum(P_row_sums[:, None] * Y_hat**2)
        if estimate_scale:
            s = denom_s / (y_hat_weighted_var + EPS)
        t = mu_x - s * (R @ mu_y)

        x_hat_weighted_var = np.sum(P_col_sums[:, None] * X_hat**2)
        sigma2_new = (x_hat_weighted_var - s * denom_s) / (N_p * D + EPS)
        sigma2_new = max(sigma2_new, EPS)

        sigma2_history.append(sigma2_new)
        n_iters = iteration + 1

        if len(sigma2_history) >= 2 and abs(sigma2_history[-2] - sigma2_history[-1]) < tol:
            sigma2 = sigma2_new
            converged = True
            break
        sigma2 = sigma2_new

    return CPDResult(
        rotation=R, scale=float(s), source_mean=source_mean, target_mean=target_mean,
        n_iters=n_iters, converged=converged, sigma2_history=sigma2_history,
    )
