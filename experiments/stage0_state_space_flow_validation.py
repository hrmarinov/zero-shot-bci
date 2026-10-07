"""Stage 0 gate: does fitting a linear dynamical system ("flow field over
neural state space", jPCA-style - see src/adaptation/state_space_flow.py's
docstring for how this differs from flow_field.py's spatial optical flow)
recover a KNOWN planted rotational structure from noisy, high-channel-count
synthetic data, before this mechanism ever touches real EEG. Same role
stage0_icp_validation.py played for Procrustes/ICP.

Four checks:
  1. The rotational PLANE is recovered correctly (subspace cosine, after
     mapping the fitted PCA-reduced eigenplane back into the original
     22-channel space) for two different planted frequencies, using the
     plain (non-debiased) fit, at both the true rank (d_fit=2) and an
     over-complete fit (d_fit=6) - real EEG's true dynamical rank is never
     known in advance, so robustness to over-fitting the rank matters.
     This is genuinely robust: cosine stayed >=0.98 in testing even at
     d_fit=6, harsh -3dB SNR.
  2. The rotation FREQUENCY is recovered correctly, at the true rank only,
     using a debiased fit. The naive fit has a real, diagnosed problem
     here: it systematically *underestimates* frequency under noise
     (~17% low at -3dB SNR, not just noisy scatter around the truth) -
     errors-in-variables attenuation bias, since the noisy observed state
     is used as its own regressor. fit_skew_symmetric_flow_debiased
     corrects this given the (here, known) noise variance, verified to
     recover frequency within ~2-3% at the same SNR.
  3. Documented limitation, not a passing claim: naively applying that
     same debiasing correction to an *over-complete* fit is unsafe and is
     shown to make things worse, not better (subspace cosine collapsed to
     0.64, frequency ratio *degraded* to 0.68, both worse than the plain
     fit) - diagnosed as the correction going ill-conditioned on the
     "extra" fitted dimensions, which carry almost no true signal above
     the noise floor once the fit exceeds the true rank. This is reported,
     not asserted as passing - real use of debiasing needs either a good
     rank estimate or a more robust (e.g. regularized/EM-based) correction,
     which is not built here.
  4. A documented pitfall from the literature review (rotational-
     dynamics/jPCA critiques, e.g. "Analysis of neuronal ensemble activity
     reveals the pitfalls and shortcomings of rotation dynamics") is
     checked directly: the identical (plain) fitting procedure applied to
     pure noise must explain far less variance (R^2) than it does for the
     true-signal condition - guarding against the exact spurious-rotation
     failure mode the literature flags, not just checking recovery works
     when something real is there.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.decomposition import PCA

from src.adaptation.state_space_flow import (
    dominant_rotation,
    fit_skew_symmetric_flow,
    fit_skew_symmetric_flow_debiased,
    subspace_cosine,
)

N_CHANNELS = 22  # matches IV-2a
T_STEPS = 2000
DT = 1.0
NOISE_SNR_DB = -3.0  # same harsh default used throughout this project's Stage 0 gates
D_FIT_EXACT = 2
D_FIT_OVERCOMPLETE = 6
RANDOM_STATE = 5


def random_orthonormal(n_channels: int, d: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.normal(size=(n_channels, d))
    q, _ = np.linalg.qr(mat)
    return q[:, :d]


def generate_rotating_trajectory(omega: float, t_steps: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Exact discrete sampling of the continuous system xdot = M x with
    M = omega * [[0, -1], [1, 0]] (skew-symmetric, pure rotation at
    angular frequency omega): x(t+dt) = R(omega*dt) @ x(t), exact (no ODE
    integration error), so any recovery error downstream is attributable
    to the fitting procedure, not to ground-truth generation noise.
    """
    theta = omega * dt
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x0 = rng.normal(size=2)
    x0 = x0 / np.linalg.norm(x0)
    X = np.zeros((t_steps, 2))
    X[0] = x0
    for t in range(1, t_steps):
        X[t] = R @ X[t - 1]
    return X


def embed_with_noise(X_true: np.ndarray, C_true: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Embeds the 2D true trajectory into n_channels via the orthonormal
    (isometric) C_true, then adds iid Gaussian observation noise at the
    given SNR - the same "peak signal power vs. target noise power" style
    used throughout this project's other Stage 0 EEG-shaped gates. Returns
    (Y, noise_power): noise_power is also the exact per-dimension noise
    variance in *any* orthonormal-projected subspace of Y (e.g. after PCA),
    since projecting isotropic noise through an orthonormal map preserves
    its per-dimension variance - what fit_skew_symmetric_flow_debiased
    needs as its noise_variance argument.
    """
    signal = X_true @ C_true.T  # (T, n_channels)
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(scale=np.sqrt(noise_power), size=signal.shape)
    return signal + noise, noise_power


def check_plane_recovery(Y: np.ndarray, C_true: np.ndarray, omega_true: float, d_fit: int) -> None:
    """Check 1: plain fit, plane recovery only - the part that's robust
    even over-complete."""
    pca = PCA(n_components=d_fit)
    Z = pca.fit_transform(Y)
    W = pca.components_.T

    M_skew, _, r_squared = fit_skew_symmetric_flow(Z, dt=DT)
    omega_fit, plane_reduced = dominant_rotation(M_skew, Z)
    overlap = subspace_cosine(C_true, W @ plane_reduced)

    print(f"  omega={omega_true:.3f} d_fit={d_fit}: plane cosine={overlap:.3f}  "
          f"(plain-fit freq ratio={omega_fit / omega_true:.3f}, R^2={r_squared:.3f} - "
          f"for reference, not gated here)")
    assert overlap > 0.85, f"omega={omega_true} d_fit={d_fit}: plane cosine {overlap:.3f} too low"


def check_debiased_frequency(Y: np.ndarray, noise_power: float, omega_true: float) -> None:
    """Check 2: debiased fit at the TRUE rank only - where the correction
    is well-conditioned."""
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Y)
    M_skew, _, r_squared = fit_skew_symmetric_flow_debiased(Z, DT, noise_power)
    omega_fit, _ = dominant_rotation(M_skew, Z)
    ratio = omega_fit / omega_true
    print(f"  omega={omega_true:.3f}: debiased freq ratio={ratio:.3f} (R^2={r_squared:.3f})")
    assert 0.9 < ratio < 1.1, f"omega={omega_true}: debiased frequency ratio {ratio:.3f} off by too much"


def demonstrate_debiasing_overcomplete_limitation(Y: np.ndarray, noise_power: float, C_true: np.ndarray, omega_true: float) -> None:
    """Check 3: NOT a pass/fail gate - demonstrates and documents that the
    same debiasing correction is unsafe when applied over-complete, so the
    limitation is visible in every run rather than only discovered later."""
    pca = PCA(n_components=D_FIT_OVERCOMPLETE)
    Z = pca.fit_transform(Y)
    W = pca.components_.T
    M_skew, _, r_squared = fit_skew_symmetric_flow_debiased(Z, DT, noise_power)
    omega_fit, plane_reduced = dominant_rotation(M_skew, Z)
    overlap = subspace_cosine(C_true, W @ plane_reduced)
    print(f"  omega={omega_true:.3f} d_fit={D_FIT_OVERCOMPLETE}, debiased anyway: "
          f"plane cosine={overlap:.3f}, freq ratio={omega_fit / omega_true:.3f}, R^2={r_squared:.3f} "
          f"-- both worse than the plain fit's own numbers above. Debiasing an over-complete fit "
          f"is unsafe (ill-conditioned on the 'extra' near-noise-floor dimensions), not used for real work here.")


def run() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    print("--- check 1: plane recovery (plain fit), exact-rank and over-complete, two frequencies ---")
    cases = []
    for omega_true in [0.15, 0.35]:
        C_true = random_orthonormal(N_CHANNELS, 2, rng)
        X_true = generate_rotating_trajectory(omega_true, T_STEPS, DT, rng)
        Y, noise_power = embed_with_noise(X_true, C_true, NOISE_SNR_DB, rng)
        cases.append((omega_true, C_true, Y, noise_power))
        for d_fit in [D_FIT_EXACT, D_FIT_OVERCOMPLETE]:
            check_plane_recovery(Y, C_true, omega_true, d_fit)
    print("  PASSED\n")

    print("--- check 2: frequency recovery (debiased fit, true rank only) ---")
    for omega_true, C_true, Y, noise_power in cases:
        check_debiased_frequency(Y, noise_power, omega_true)
    print("  PASSED\n")

    print("--- check 3: debiasing an over-complete fit is unsafe (documented, not gated) ---")
    for omega_true, C_true, Y, noise_power in cases:
        demonstrate_debiasing_overcomplete_limitation(Y, noise_power, C_true, omega_true)
    print()

    print("--- check 4: spurious-rotation-in-pure-noise guard (a documented jPCA pitfall) ---")
    pure_noise = rng.normal(size=(T_STEPS, N_CHANNELS))
    Z_noise = PCA(n_components=D_FIT_EXACT).fit_transform(pure_noise)
    _, _, r_squared_noise = fit_skew_symmetric_flow(Z_noise, dt=DT)

    C_true = random_orthonormal(N_CHANNELS, 2, rng)
    X_true = generate_rotating_trajectory(0.25, T_STEPS, DT, rng)
    Y_signal, _ = embed_with_noise(X_true, C_true, NOISE_SNR_DB, rng)
    Z_signal = PCA(n_components=D_FIT_EXACT).fit_transform(Y_signal)
    _, _, r_squared_signal = fit_skew_symmetric_flow(Z_signal, dt=DT)

    print(f"  R^2 on pure noise (no planted rotation): {r_squared_noise:.3f}")
    print(f"  R^2 on planted rotation + noise: {r_squared_signal:.3f}")
    assert r_squared_noise < 0.15, f"pure noise produced a suspiciously strong 'rotational' fit (R^2={r_squared_noise:.3f})"
    # 0.15, not an arbitrary higher bar: check 1 already showed genuine-signal
    # R^2 (plain fit, d_fit=2, this same -3dB SNR) legitimately ranges ~0.16-0.47
    # across different planted frequencies/noise draws - 0.15 is a floor
    # grounded in that observed range, not tuned to make this specific run pass.
    assert r_squared_signal > 0.15, f"the true-signal condition's own fit quality is too weak to trust the comparison (R^2={r_squared_signal:.3f})"
    assert r_squared_signal - r_squared_noise > 0.1, "fit quality doesn't clearly separate real rotational structure from noise"
    print("  PASSED\n")

    print("Flow-field-as-dynamical-system (jPCA-style) primitives validated on synthetic ground truth "
          "(plane recovery robust at any tested rank; frequency recovery accurate at the true rank via a "
          "diagnosed and corrected bias; over-complete debiasing's limitation documented, not swept under "
          "the rug) - safe to proceed to a synthetic-EEG-shaped test, then real IV-2a data.")


if __name__ == "__main__":
    run()
