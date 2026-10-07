"""Stage 0 gate: does RIDE's actual documented advantage over plain
Woody's filter - separating a fixed-latency component (S) from a jittered
one (C) *before* estimating the jittered component's per-trial latency -
provide a real, measurable improvement, tested directly rather than
assumed from the literature?

Two-component generative model (Stage A: clean amplitude-domain, no
oscillatory-carrier confound - see src/adaptation/latency_alignment.py's
ride_two_component docstring and stage0_latency_alignment_validation.py's
carrier-envelope-randomness finding for why that confound is deliberately
excluded here first): each trial is S(t) + C(t - jitter_i) + noise, S a
Gaussian bump at a FIXED latency (e.g. a cue-evoked response), C a
Gaussian bump with its own per-trial jitter (e.g. this project's ERD/ERS
onset). RIDE's claim: estimating C's latency from (trial - S_estimate) is
cleaner than estimating it from the raw, unseparated trial, since S never
moves and only ever adds interference.

Three checks, same discipline as every other Stage 0 gate in this project:
  1. Zero true jitter (control): does RIDE's alternating S/C estimation
     avoid manufacturing spurious latency structure, the same pitfall
     already found (twice now) in this project's other alignment work?
  2. Known nonzero jitter, N_REPEATS independent draws (never trust a
     single draw - established twice already this session): does RIDE's
     C-latency recovery correlate with true jitter?
  3. Head-to-head against naive Woody's filter applied directly to the
     raw S+C mixture (no separation) - RIDE's actual claimed advantage,
     not just "does RIDE recover jitter" in isolation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import wilcoxon

from src.adaptation.latency_alignment import cross_correlate_lag, ride_two_component, woody_align

SFREQ = 250.0
DURATION_S = 4.0
N_TRIALS = 60
SNR_DB = -3.0
JITTER_STD_S = 0.15
MAX_LAG_S = 0.5
N_ITERS = 5
N_REPEATS = 20
RANDOM_STATE = 21

S_ONSET_S, S_SIGMA_S, S_AMPLITUDE = 0.3, 0.15, 1.0
C_ONSET_S, C_SIGMA_S, C_AMPLITUDE = 1.5, 0.3, 1.0


def gaussian_bump(t: np.ndarray, onset: float, sigma: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-((t - onset) ** 2) / (2 * sigma**2))


def generate_trials(jitter_std_s: float, n_times: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (trials, true_jitters, s_component): trials is (N_TRIALS,
    n_times), the raw S+C+noise mixture; true_jitters is each trial's own
    known C-latency shift; s_component is the true, noiseless S(t) shape
    (ground truth for checking S recovery)."""
    t = np.arange(n_times) / SFREQ
    s_component = gaussian_bump(t, S_ONSET_S, S_SIGMA_S, S_AMPLITUDE)
    signal_power = np.mean((s_component + gaussian_bump(t, C_ONSET_S, C_SIGMA_S, C_AMPLITUDE)) ** 2)
    noise_power = signal_power / (10 ** (SNR_DB / 10))

    trials, true_jitters = [], []
    for _ in range(N_TRIALS):
        jitter = rng.normal(0.0, jitter_std_s) if jitter_std_s > 0 else 0.0
        c_component = gaussian_bump(t, C_ONSET_S + jitter, C_SIGMA_S, C_AMPLITUDE)
        noise = rng.normal(scale=np.sqrt(noise_power), size=n_times)
        trials.append(s_component + c_component + noise)
        true_jitters.append(jitter)
    return np.array(trials), np.array(true_jitters), s_component


def run() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    n_times = int(DURATION_S * SFREQ)
    max_lag_samples = int(round(MAX_LAG_S * SFREQ))
    t = np.arange(n_times) / SFREQ
    true_c_shape = gaussian_bump(t, C_ONSET_S, C_SIGMA_S, C_AMPLITUDE)

    print("--- check 1: zero true jitter (control - same pitfall already found twice in this project) ---")
    trials0, _, s_true0 = generate_trials(0.0, n_times, rng)
    S_est0, C_est0, lags0 = ride_two_component(trials0, max_lag_samples, N_ITERS)
    lags0_s = np.array(lags0) / SFREQ
    s_corr0 = np.corrcoef(S_est0, s_true0)[0, 1]
    print(f"  recovered lag distribution (should be small/centered near zero): "
          f"mean={lags0_s.mean() * 1000:.1f}ms  std={lags0_s.std() * 1000:.1f}ms")
    print(f"  recovered S component correlation with true S shape: {s_corr0:.4f} (not gated - see note below)")
    assert lags0_s.std() < 0.1, f"RIDE manufactured spurious C-latency structure with no true jitter (std={lags0_s.std()*1000:.1f}ms)"
    print("  PASSED (lag estimation stayed at zero, as it should)")
    print("  Note: S/C *separation* is mathematically ill-posed at zero jitter - with no trial-to-trial C "
          "movement, S and C are indistinguishable from the data alone (the naive trial mean already matches "
          "S+C combined at r=0.97), so a weak S-recovery number here is expected, not a bug. S recovery is a "
          "meaningful check only once real jitter gives the algorithm something to separate on - see check 2.\n")

    print(f"--- check 2+3: known nonzero jitter, {N_REPEATS} independent repeats, RIDE vs. naive Woody-on-raw-mixture ---")
    ride_lag_corrs, woody_lag_corrs, s_corrs = [], [], []
    for repeat in range(N_REPEATS):
        rep_rng = np.random.default_rng(500 + repeat)
        trials, true_jitters, s_true = generate_trials(JITTER_STD_S, n_times, rep_rng)

        S_est, C_est, ride_lags = ride_two_component(trials, max_lag_samples, N_ITERS)
        ride_lag_corrs.append(np.corrcoef(np.array(ride_lags) / SFREQ, true_jitters)[0, 1])
        s_corrs.append(np.corrcoef(S_est, s_true)[0, 1])

        # naive Woody's filter applied directly to the raw S+C mixture, no separation
        _, _, woody_lags = woody_align(trials, max_lag_samples, n_iters=3)
        woody_lag_corrs.append(np.corrcoef(np.array(woody_lags) / SFREQ, true_jitters)[0, 1])

    ride_lag_corrs = np.array(ride_lag_corrs)
    woody_lag_corrs = np.array(woody_lag_corrs)
    s_corrs = np.array(s_corrs)

    print(f"  RIDE: C-latency-vs-true-jitter correlation: mean={ride_lag_corrs.mean():.3f} std={ride_lag_corrs.std():.3f}")
    print(f"  naive Woody-on-raw-mixture: latency-vs-true-jitter correlation: mean={woody_lag_corrs.mean():.3f} std={woody_lag_corrs.std():.3f}")
    print(f"  RIDE's recovered S component vs. true S shape: mean correlation={s_corrs.mean():.4f}")
    print(f"  RIDE beats naive Woody in {(ride_lag_corrs > woody_lag_corrs).sum()}/{N_REPEATS} repeats")
    stat = wilcoxon(ride_lag_corrs, woody_lag_corrs)
    print(f"  Wilcoxon (RIDE vs. naive Woody): p={stat.pvalue:.4f}")

    print("\nCONCLUSION printed above, not gated on a hard assert here - see docs/progress_and_direction.md "
          "for the full interpretation once this has been run.")


if __name__ == "__main__":
    run()
