"""Section 10.5 test - does an EEG analogue of interferometric closure phase exist?

Radio interferometry self-calibrates without any external reference using
"closure phase/amplitude": combinations of antenna-pair measurements where
each antenna's unknown multiplicative gain error appears once with each
algebraic sign around a closed loop, and cancels exactly.

Math for the EEG case: if channel i's recorded signal is an unknown real
scalar gain g_i times its true signal (impedance mismatch, amplifier drift -
a per-channel corruption, NOT the cross-subject/cross-session mixing-matrix
difference Riemannian alignment targets), then covariance under linear
rescaling gives, exactly:
    Cov(g_i x_i, g_j x_j) = g_i * g_j * Cov(x_i, x_j)
So for four distinct channels i, j, k, l, the ratio
    (Sigma_ij * Sigma_kl) / (Sigma_ik * Sigma_jl)
is exactly invariant to g_i, g_j, g_k, g_l, since each gain appears once in
the numerator and once in the denominator - the EEG closure-amplitude
analogue. This script verifies that numerically on real EEG data, and
verifies the flip side: the same ratio is NOT invariant under a full-matrix
(non-diagonal) distortion, which is what actually causes cross-session drift
- confirming this technique addresses a real but narrower problem (per-
channel gain/impedance calibration) than the one motivating this project.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.datasets import load_iv2a_subject

QUADRUPLES = [(0, 1, 2, 3), (5, 10, 15, 20), (2, 8, 14, 19), (0, 21, 7, 13)]


def closure_amplitude_ratios(Sigma: np.ndarray, quadruples: list[tuple[int, int, int, int]]) -> list[float]:
    return [
        (Sigma[i, j] * Sigma[k, l]) / (Sigma[i, k] * Sigma[j, l])
        for i, j, k, l in quadruples
    ]


def sample_covariance(X: np.ndarray) -> np.ndarray:
    n_channels = X.shape[1]
    flat = X.transpose(1, 0, 2).reshape(n_channels, -1)
    return np.cov(flat)


def run() -> None:
    data = load_iv2a_subject(1)
    X = data.X_calib
    n_channels = X.shape[1]
    print(f"X shape: {X.shape}")

    Sigma = sample_covariance(X)
    raw_ratios = closure_amplitude_ratios(Sigma, QUADRUPLES)

    rng = np.random.default_rng(0)
    gains = rng.uniform(0.2, 3.0, size=n_channels)
    X_gain_corrupted = X * gains[np.newaxis, :, np.newaxis]
    Sigma_gain = sample_covariance(X_gain_corrupted)
    gain_ratios = closure_amplitude_ratios(Sigma_gain, QUADRUPLES)

    print("\nPer-channel scalar gain corruption (the model closure amplitude targets):")
    print(f"{'quad':<20}{'ratio (raw)':>15}{'ratio (gain-corrupted)':>25}{'rel. diff':>15}")
    for quad, raw, corrupted in zip(QUADRUPLES, raw_ratios, gain_ratios):
        rel_diff = abs(raw - corrupted) / abs(raw)
        print(f"{str(quad):<20}{raw:>15.6f}{corrupted:>25.6f}{rel_diff:>15.2e}")

    A = rng.normal(size=(n_channels, n_channels)) * 0.3 + np.eye(n_channels)
    X_mixed = np.einsum("cd,nds->ncs", A, X)
    Sigma_mixed = sample_covariance(X_mixed)
    mixed_ratios = closure_amplitude_ratios(Sigma_mixed, QUADRUPLES)

    print("\nFull-matrix (non-diagonal) distortion - the kind that actually")
    print("causes cross-session drift, not a per-channel scalar gain:")
    print(f"{'quad':<20}{'ratio (raw)':>15}{'ratio (full-matrix mix)':>25}{'rel. diff':>15}")
    for quad, raw, mixed in zip(QUADRUPLES, raw_ratios, mixed_ratios):
        rel_diff = abs(raw - mixed) / abs(raw)
        print(f"{str(quad):<20}{raw:>15.6f}{mixed:>25.6f}{rel_diff:>15.2e}")


if __name__ == "__main__":
    run()
