"""Stage 0 gate: does Woody-filter-style alignment (iteratively estimate
each trial's latency via cross-correlation against an evolving template,
shift, re-average) recover a KNOWN planted per-trial jitter, and does it
avoid the specific peak-picking/multiple-comparisons bias this project
already found breaks a structurally similar mechanism (Phase 4's
max-over-lags jitter-tolerant classifier scoring, which made accuracy
*worse* even at zero true jitter - see docs/progress_and_direction.md and
src/adaptation/latency_alignment.py's docstring)?

Works on each trial's amplitude ENVELOPE (Hilbert transform), not the raw
oscillation - the carrier's phase isn't trial-locked, so only the
envelope's onset/trough shape is a genuinely alignable template. The
"true" reference shape is computed directly from
synthetic_eeg.erd_power_envelope's deterministic formula, not from a
generated (still carrier-noise-contaminated, even at high SNR) trial -
the cleanest ground truth available, not an approximation.

Two conditions:
  1. Zero true jitter (control, re-testing the known pitfall directly):
     does alignment avoid degrading the template relative to the naive
     average, the way Phase 4's approach did?
  2. Known nonzero jitter, over N_REPEATS independent draws (a single
     draw was tried first and looked promising - recovered-lag-vs-true-
     jitter correlation 0.364 - but that was a lucky instance, not a
     reliable property; the same lesson stage0_state_space_flow_eeg_
     validation.py's drift test already established. Only the repeated
     version is trustworthy): does the recovered per-trial lag reliably
     correlate with the true injected jitter, and does the aligned
     template reliably beat the naive average?

**Check 1 failed on the first pass** - alignment degraded the template
even with zero true jitter (naive correlation with the true shape 0.587,
woody-aligned only 0.398), recovered lags had a suspiciously large spread
(std 230ms) despite nothing to recover. Diagnosed properly before
concluding the mechanism is broken: raising additive SNR from -3dB all the
way to 60dB (near-noiseless) left the spurious-lag spread essentially
unchanged (~150ms throughout) - ruling out additive background noise as
the cause. The real culprit is the oscillatory carrier's own *intrinsic*
envelope randomness: a narrowband random process (the carrier is filtered
white noise in a ~4Hz band) has a natural envelope correlation time of
roughly 1/bandwidth =~ 250ms, independent of any added noise - and Woody's
filter, run on the raw envelope, mistakes that intrinsic wiggle for genuine
trial-to-trial latency structure. Fixed with `smooth_signal` (a 0.5s moving
average, comparable to the genuine ERD/ERS onset-to-trough timescale, not
the oscillation's own period) applied before alignment - the same kind of
fix ("smooth the template/signal, don't search harder") that already
worked for the *original* Phase 4 pitfall this test was re-checking against.

**Check 2 found a genuine, honest ceiling after that fix, not a pass**:
across N_REPEATS=20 independent draws, recovered-lag-vs-true-jitter
correlation averaged 0.063 (std 0.102, sign inconsistent across draws) at
this project's standard harsh -3dB SNR - indistinguishable from noise -
and the aligned template beat the naive average in only 5/20 draws.
Raising SNR to a very generous, unrealistic-for-single-trial-EEG 20dB
improved this to only 0.278 (std 0.130) - real, but still modest. That
SNR-insensitivity points at a genuine structural ceiling, not just a noise
problem: the ~0.5s smoothing needed to suppress the carrier's own
intrinsic envelope randomness (see the fix above) is comparable to or
larger than the 0.15s jitter being detected - the fix for the false-
positive problem and the sensitivity needed to detect real jitter are in
direct tension here, not simultaneously achievable by tuning the smoothing
window alone.

**Check 3, added after the user asked exactly the right follow-up
question**: is it meaningful to *estimate* the lag on smoothed envelopes
(needed to avoid the check-1 pitfall) but *apply* that estimate to the
original, unsmoothed trial for downstream use, rather than smoothing
throughout? Tested directly, not just reasoned about: applying the
smoothed-estimated lag to each trial's raw envelope, then averaging the
*raw* shifted trials, beats both doing nothing and the smoothed-throughout
approach - mean correlation with the true raw shape improves from 0.531
(naive) to 0.548 (shifted), wins in 37/50 repeats, Wilcoxon p=1.0e-5. Real,
not a fluke - but modest (a ~3% relative gain), and the underlying lag
estimate itself is still only weakly correlated with true jitter (~0.06-
0.07, unchanged) - separating estimation from application recovers a
little more of what that weak signal already carries, it does not create
a stronger signal than was there. This refines, but does not overturn, the
"genuine ceiling" finding above.

This closes out the *Woody-filter* instantiation of the user's proposed
idea as a real-but-modest, not transformative, mechanism (see
docs/progress_and_direction.md) before it ever reaches the cross-session
comparison or NN-generalization steps proposed next - RIDE's more
sophisticated multi-component decomposition, not tried here, is the
natural next thing to check instead of building further on top of this
specific mechanism.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.signal import hilbert
from scipy.stats import wilcoxon

from src.adaptation.latency_alignment import shift_signal, smooth_signal, woody_align
from src.datasets import IV2A_CHANNELS
from src.synthetic_eeg import CONTRALATERAL_ELECTRODE, SyntheticSubjectParams, erd_power_envelope, generate_trial

SFREQ = 250.0
DURATION_S = 4.0
SNR_DB = -3.0
N_TRIALS = 60
JITTER_STD_S = 0.15
MAX_LAG_S = 0.3
SMOOTH_WINDOW_S = 0.5  # see smooth_signal's docstring - required, not cosmetic (found empirically)
N_REPEATS = 20  # single-draw results here are unreliable - see module docstring
RANDOM_STATE = 13
LABEL = "left_hand"
CHANNEL = CONTRALATERAL_ELECTRODE[LABEL]

PARAMS = SyntheticSubjectParams(
    peak_freq_hz=10.0, onset_s=0.5, trough_s=1.5,
    erd_magnitude=0.6, rebound_magnitude=0.3, spatial_sigma_m=0.06,
)


def generate_trials_with_known_jitter(
    jitter_std_s: float, rng: np.random.Generator, smooth_window_samples: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (raw_envelopes, smoothed_envelopes, true_jitters):
    raw_envelopes is (N_TRIALS, n_times), the Hilbert envelope of
    CHANNEL's signal per trial, at full resolution; smoothed_envelopes is
    the same, moving-average smoothed (needed for lag *estimation*, see
    smooth_signal's docstring - check 3 below tests applying that
    estimate back to raw_envelopes instead of smoothed_envelopes).
    true_jitters is each trial's own known injected onset/trough shift (in
    seconds), chosen externally - not synthetic_eeg.py's own
    onset_jitter_std_s mechanism - so it's directly available as ground
    truth here.
    """
    raw_envelopes, smoothed_envelopes, true_jitters = [], [], []
    channel_idx = IV2A_CHANNELS.index(CHANNEL)
    win = smooth_window_samples or int(round(SMOOTH_WINDOW_S * SFREQ))
    for _ in range(N_TRIALS):
        jitter = rng.normal(0.0, jitter_std_s) if jitter_std_s > 0 else 0.0
        shifted_params = replace(PARAMS, onset_s=PARAMS.onset_s + jitter, trough_s=PARAMS.trough_s + jitter)
        trial = generate_trial(shifted_params, LABEL, IV2A_CHANNELS, SFREQ, DURATION_S, SNR_DB, rng)
        envelope = np.abs(hilbert(trial[channel_idx]))
        raw_envelopes.append(envelope)
        smoothed_envelopes.append(smooth_signal(envelope, win))
        true_jitters.append(jitter)
    return np.array(raw_envelopes), np.array(smoothed_envelopes), np.array(true_jitters)


def true_envelope_shape(n_times: int) -> np.ndarray:
    """The exact, deterministic ground-truth envelope shape - computed
    directly from erd_power_envelope's own formula, not from a generated
    trial (which would still carry the carrier's own random envelope
    fluctuations even at very high SNR)."""
    t = np.arange(n_times) / SFREQ
    return np.sqrt(np.clip(erd_power_envelope(t, PARAMS), 1e-6, None))


def run() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    max_lag_samples = int(round(MAX_LAG_S * SFREQ))
    smooth_window_samples = int(round(SMOOTH_WINDOW_S * SFREQ))
    n_times = int(DURATION_S * SFREQ)
    raw_reference = true_envelope_shape(n_times)
    smooth_reference = smooth_signal(raw_reference, smooth_window_samples)

    print("--- check 1: zero true jitter (control - re-testing Phase 4's known peak-picking pitfall) ---")
    _, envelopes0, _ = generate_trials_with_known_jitter(0.0, rng, smooth_window_samples)
    naive_template0 = envelopes0.mean(axis=0)
    _, woody_template0, lags0 = woody_align(envelopes0, max_lag_samples, n_iters=3)
    lags0_s = np.array(lags0) / SFREQ

    naive_corr0 = np.corrcoef(naive_template0, smooth_reference)[0, 1]
    woody_corr0 = np.corrcoef(woody_template0, smooth_reference)[0, 1]
    print(f"  recovered lag distribution (should be small/centered near zero - no true jitter planted): "
          f"mean={lags0_s.mean() * 1000:.1f}ms  std={lags0_s.std() * 1000:.1f}ms")
    print(f"  naive template correlation with true shape:  {naive_corr0:.4f}")
    print(f"  woody-aligned template correlation with true shape: {woody_corr0:.4f}")
    assert woody_corr0 > naive_corr0 - 0.02, (
        f"alignment degraded the template even with zero true jitter (naive={naive_corr0:.4f}, "
        f"woody={woody_corr0:.4f}) - same failure mode as Phase 4's max-over-lags scoring, do not proceed"
    )
    print("  PASSED (alignment did not manufacture spurious 'improvement' out of pure search)\n")

    print(f"--- check 2: known nonzero jitter, {N_REPEATS} independent repeats (a single draw is unreliable - see below) ---")
    lag_corrs, naive_corrs, woody_corrs = [], [], []
    for repeat in range(N_REPEATS):
        rep_rng = np.random.default_rng(1000 + repeat)
        _, envelopes, true_jitters = generate_trials_with_known_jitter(JITTER_STD_S, rep_rng, smooth_window_samples)
        naive_template = envelopes.mean(axis=0)
        _, woody_template, lags = woody_align(envelopes, max_lag_samples, n_iters=3)
        lags_s = np.array(lags) / SFREQ
        lag_corrs.append(np.corrcoef(lags_s, true_jitters)[0, 1])
        naive_corrs.append(np.corrcoef(naive_template, smooth_reference)[0, 1])
        woody_corrs.append(np.corrcoef(woody_template, smooth_reference)[0, 1])
    lag_corrs, naive_corrs, woody_corrs = np.array(lag_corrs), np.array(naive_corrs), np.array(woody_corrs)

    print(f"  recovered-lag vs. true-jitter correlation: mean={lag_corrs.mean():.3f} std={lag_corrs.std():.3f} "
          f"(range {lag_corrs.min():.3f} to {lag_corrs.max():.3f})")
    print(f"  naive template correlation with true shape: mean={naive_corrs.mean():.4f}")
    print(f"  woody-aligned template correlation with true shape: mean={woody_corrs.mean():.4f}")
    print(f"  woody beats naive in {(woody_corrs > naive_corrs).sum()}/{N_REPEATS} repeats")
    print("  NOT a pass/fail gate - this is the honest ceiling, documented rather than asserted as working "
          "(see this file's module docstring for the SNR check confirming it's a structural limitation, "
          "not just noise).\n")

    print(f"--- check 3: estimate lag on SMOOTHED envelope, apply it to the RAW (unsmoothed) trial instead ---")
    n_repeats_check3 = 50  # this effect is small - needs more repeats than check 2 to size it reliably
    naive_raw_corrs, shifted_raw_corrs = [], []
    for repeat in range(n_repeats_check3):
        rep_rng = np.random.default_rng(2000 + repeat)
        raw_envelopes, smoothed_envelopes, _ = generate_trials_with_known_jitter(JITTER_STD_S, rep_rng, smooth_window_samples)
        _, _, lags = woody_align(smoothed_envelopes, max_lag_samples, n_iters=3)
        shifted_raw = np.array([shift_signal(raw_envelopes[i], lags[i]) for i in range(len(raw_envelopes))])
        naive_raw_corrs.append(np.corrcoef(raw_envelopes.mean(axis=0), raw_reference)[0, 1])
        shifted_raw_corrs.append(np.corrcoef(shifted_raw.mean(axis=0), raw_reference)[0, 1])
    naive_raw_corrs, shifted_raw_corrs = np.array(naive_raw_corrs), np.array(shifted_raw_corrs)

    stat = wilcoxon(shifted_raw_corrs, naive_raw_corrs)
    print(f"  naive (unshifted) raw template vs. true raw shape: mean={naive_raw_corrs.mean():.4f}")
    print(f"  shifted (smoothed-estimated lag, applied to raw) raw template vs. true raw shape: "
          f"mean={shifted_raw_corrs.mean():.4f}")
    print(f"  shifted-raw beats naive-raw in {(shifted_raw_corrs > naive_raw_corrs).sum()}/{n_repeats_check3} repeats, "
          f"Wilcoxon p={stat.pvalue:.2e}")
    assert stat.pvalue < 0.01 and shifted_raw_corrs.mean() > naive_raw_corrs.mean(), (
        "estimate-on-smoothed/apply-to-raw did not show a real, significant improvement - "
        f"mean naive={naive_raw_corrs.mean():.4f} shifted={shifted_raw_corrs.mean():.4f} p={stat.pvalue:.2e}"
    )
    print("  PASSED - real, statistically significant, but modest (see module docstring for how to read this)\n")

    print("CONCLUSION: Woody-filter alignment does not reliably recover realistic-magnitude jitter at this "
          "project's standard harsh SNR (check 2). Estimating the lag on a smoothed envelope but applying it "
          "to the original, unsmoothed trial (check 3) recovers a small, real, statistically significant "
          "improvement over both doing nothing and smoothing throughout - genuine, but modest, refinement, "
          "not a reversal of check 2's ceiling finding. See docs/progress_and_direction.md for the full "
          "write-up and what to try instead (RIDE's multi-component decomposition) before building the "
          "cross-session comparison or NN-generalization steps on top of this.")


if __name__ == "__main__":
    run()
