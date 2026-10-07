"""Linear dynamical-system ("flow field over neural state space") fitting.

Distinct from flow_field.py: that module computes optical flow of a
*spatial* scalp topography (physical 2D electrode positions evolving over
time). This module fits a flow field in an abstract *state space* where
each axis is a channel's (or a PCA component's) activity level - the
xdot = f(x) picture from Churchland/Shenoy/Sussillo's neural population
dynamics work (jPCA, rotational dynamics; Kaufman et al.'s null-space/
potent-space account of movement preparation), not scalp-surface motion.
See docs/progress_and_direction.md's literature-check correction for the
full context, including the one EEG-specific precedent found (Zhang et al.
2016, "Low-Rank Linear Dynamical Systems for Motor Imagery EEG" - fits
exactly this kind of state-space system directly to raw EEG channels,
evaluated on this project's own BCI IV-2a dataset, beating CSP within-
session; cross-session stability, this project's actual question, is
untested there).

jPCA's own method (Churchland et al. 2012): reduce the observed high-
dimensional trajectory to a low-dimensional subspace (PCA), then fit the
best-fit linear operator M in xdot ~= M x via least squares, and constrain
M to be skew-symmetric (M = (M - M^T)/2). A skew-symmetric matrix has
purely imaginary eigenvalues in conjugate pairs - each pair describes one
2D plane of pure rotation at a fixed angular frequency, with no growth or
decay - which is what makes the fitted flow "rotational dynamics" rather
than an arbitrary linear system. Fitting the unconstrained M first and
only then symmetrizing (rather than constraining rotational structure from
the start) is deliberate and matches the published method: it lets the fit
quality (before symmetrizing) diagnose whether the data supports rotational
structure at all, versus forcing rotation onto data that doesn't have any.

Known limitation, found and diagnosed in stage0_state_space_flow_validation.py
before this touched anything else: naive least-squares fitting recovers the
rotational *plane* robustly even under harsh observation noise (subspace
cosine ~0.999 at -3dB SNR) but *systematically underestimates the rotation
frequency* under noise (recovered ratio 0.83 at the same SNR, not 1.0) - a
textbook errors-in-variables attenuation bias, since the regressor (the
noisy observed state) is measured with error. fit_skew_symmetric_flow_debiased
corrects for this given a known noise variance, verified to recover the
frequency to within ~2% at the same harsh SNR - confirming the mechanism is
exactly this bias and nothing else. See that function's docstring for what
"known noise variance" costs when real data doesn't hand it over for free.
"""

import numpy as np


def fit_skew_symmetric_flow(X: np.ndarray, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Fits M in Xdot ~= X @ M.T, with M constrained skew-symmetric. X:
    (T, d) state trajectory - already dimensionality-reduced (e.g. via
    PCA), since fitting a d x d matrix needs T >> d timesteps to be
    identifiable, which is why this is applied downstream of a reduction,
    never to raw high-channel-count data directly.

    Xdot is estimated by central difference (T-2 interior points used).
    Returns (M_skew, Xdot, r_squared): r_squared is the fraction of Xdot's
    variance explained by X @ M_skew.T - a diagnostic for how much of the
    observed dynamics looks like clean rotational flow at all, used by the
    spurious-rotation-in-noise guard in
    experiments/stage0_state_space_flow_validation.py.
    """
    Xdot = (X[2:] - X[:-2]) / (2 * dt)
    X_interior = X[1:-1]
    M_full, *_ = np.linalg.lstsq(X_interior, Xdot, rcond=None)  # solves X_interior @ M_full ~= Xdot
    M = M_full.T
    M_skew = (M - M.T) / 2
    Xdot_pred = X_interior @ M_skew.T
    ss_res = np.sum((Xdot - Xdot_pred) ** 2)
    ss_tot = np.sum((Xdot - Xdot.mean(axis=0)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    return M_skew, Xdot, r_squared


def fit_skew_symmetric_flow_debiased(X: np.ndarray, dt: float, noise_variance: float) -> tuple[np.ndarray, np.ndarray, float]:
    """fit_skew_symmetric_flow, corrected for errors-in-variables
    attenuation bias. Naive least squares on Xdot ~= X @ M.T systematically
    *underestimates* the magnitude of M (hence the recovered rotation
    frequency) whenever X - the regressor - is itself measured with noise,
    even though the recovered rotational *plane* stays accurate (found
    empirically in experiments/stage0_state_space_flow_validation.py:
    subspace cosine ~0.999 while frequency ratio was only 0.83 at -3dB
    SNR - a systematic, not random, shortfall, confirmed to be exactly
    this mechanism by this correction recovering ratio ~0.98 at the same
    SNR). Standard correction: subtract the *known* noise contribution
    from the regressor's Gram matrix before solving - X_true's Gram
    matrix is Xi.T @ Xi minus the noise's own expected contribution,
    T * noise_variance * I, since independent per-sample noise adds
    exactly that much to the diagonal in expectation.

    noise_variance is the per-dimension observation noise variance *in
    X's own coordinate space* (e.g. the reduced PCA space X actually
    lives in) - a synthetic-validation luxury (the true injected noise
    power is known exactly here). Real EEG data doesn't hand this value
    over for free; using this correction there needs either a variance
    estimate (e.g. from repeated trials) or a full EM-based LDS fit that
    estimates observation noise jointly with the dynamics (the more
    complete solution - see Zhang et al. 2016's {A, C, Gamma, Sigma}
    formulation, referenced in docs/progress_and_direction.md).
    """
    Xdot = (X[2:] - X[:-2]) / (2 * dt)
    X_interior = X[1:-1]
    n_samples, d = X_interior.shape
    gram_corrected = X_interior.T @ X_interior - n_samples * noise_variance * np.eye(d)
    M_full = np.linalg.solve(gram_corrected, X_interior.T @ Xdot)
    M = M_full.T
    M_skew = (M - M.T) / 2
    Xdot_pred = X_interior @ M_skew.T
    ss_res = np.sum((Xdot - Xdot_pred) ** 2)
    ss_tot = np.sum((Xdot - Xdot.mean(axis=0)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    return M_skew, Xdot, r_squared


def fit_pooled_flow_debiased(trials: list[np.ndarray], dt: float, noise_variance: float) -> np.ndarray:
    """Fits one shared, debiased skew-symmetric M from multiple trials of
    the same condition (e.g. all of one class's calibration trials, in a
    shared reduced space) - a more stable template than any single trial's
    own fit, the same way a class mean is more stable than any one sample.
    Concatenates each trial's own (X_interior, Xdot) pairs into a single
    regression; never computes a derivative *across* a trial boundary.
    trials: list of (T, d) reduced trajectories (same d, T can differ per
    trial).
    """
    all_X, all_Xdot = [], []
    for X in trials:
        all_Xdot.append((X[2:] - X[:-2]) / (2 * dt))
        all_X.append(X[1:-1])
    X_pooled = np.concatenate(all_X, axis=0)
    Xdot_pooled = np.concatenate(all_Xdot, axis=0)
    n_samples, d = X_pooled.shape
    gram_corrected = X_pooled.T @ X_pooled - n_samples * noise_variance * np.eye(d)
    M_full = np.linalg.solve(gram_corrected, X_pooled.T @ Xdot_pooled)
    M = M_full.T
    return (M - M.T) / 2


def dominant_rotation(M_skew: np.ndarray, X: np.ndarray) -> tuple[float, np.ndarray]:
    """Eigendecomposes M_skew (purely imaginary eigenvalues in conjugate
    pairs, each pair = one 2D rotation plane) and picks the plane that
    captures the most variance of the actual trajectory X - not simply the
    fastest-spinning fitted mode. Frequency-only selection would be wrong:
    noise readily produces spuriously fast, low-amplitude "rotations" that
    carry almost none of the data's actual variance, and a criterion that
    prefers speed over amplitude would latch onto those instead of the
    real signal.

    Returns (frequency, plane): frequency in radians per unit time
    (dt-scaled, matching fit_skew_symmetric_flow's dt), plane a (d, 2)
    array of two real, orthonormal vectors spanning the winning rotation.
    """
    eigvals, eigvecs = np.linalg.eig(M_skew)
    best_variance = -1.0
    best_frequency = 0.0
    best_plane = None
    for i, eigval in enumerate(eigvals):
        if eigval.imag <= 1e-9:
            continue  # skip near-zero/real eigenvalues and one of each conjugate pair
        v = eigvecs[:, i]
        b1 = v.real / np.linalg.norm(v.real)
        b2 = v.imag - np.dot(v.imag, b1) * b1
        b2 = b2 / np.linalg.norm(b2)
        plane = np.stack([b1, b2], axis=1)
        variance_captured = float(np.sum((X @ plane) ** 2))
        if variance_captured > best_variance:
            best_variance = variance_captured
            best_frequency = float(np.abs(eigval.imag))
            best_plane = plane
    return best_frequency, best_plane


def subspace_cosine(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Smallest principal-angle cosine between the subspaces spanned by
    basis_a's and basis_b's columns (each (d, k)). 1.0 = identical
    subspaces, 0.0 = orthogonal - the conservative (worst-axis) summary,
    not an average, matching this project's other subspace/direction-
    recovery checks (e.g. riemannian_icp.py's rotation-recovery error).
    """
    q_a, _ = np.linalg.qr(basis_a)
    q_b, _ = np.linalg.qr(basis_b)
    singular_values = np.linalg.svd(q_a.T @ q_b, compute_uv=False)
    return float(np.min(singular_values))
