"""Stage 0 synthetic validation harness (docs/wavelet_personalization_pipeline.md §6).

Confirms the synthetic EEG generator (src/synthetic_eeg.py) faithfully
encodes its own intended ground truth *before* it's trusted as the
validation tool for Module A/B. A generator that runs without errors but
gets its own physics wrong would silently invalidate every later check
built on top of it - the same class of bug already caught once here
(see check 4's history: an earlier version anchored noise power to the
trial-average signal, which let a near-zero-weight channel show
almost-full apparent ERD).

Four checks, each gating on an assertion (nonzero exit = Stage 0 not
validated, do not proceed to Module A):

1. Envelope shape - trough near params.trough_s, rebound exceeds trough,
   correct sign for both ERD and ERS subjects. Deterministic, no RNG.
2. Contralaterality - left_hand modulates C4, right_hand modulates C3,
   not swapped or symmetric.
3. Spatial monotonicity - modulation strength decreases with distance from
   the contralateral electrode, measured via band-limited power (matching
   both the literature's ERD measurement convention and this project's own
   8-30Hz pipeline) at n_reps large enough that the check isn't dominated
   by estimator noise (n=60 was; verified by rerunning at n=400 during
   development - see session history).
4. SNR-dependent detectability - distant channels' apparent modulation
   washes out toward a ratio of 1.0 (no visible effect) as SNR drops, while
   the contralateral channel's stays clearly visible - the property Stage
   2's noise curriculum is meant to exploit later.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.datasets import IV2A_CHANNELS
from src.synthetic_eeg import (
    CONTRALATERAL_ELECTRODE,
    SyntheticSubjectParams,
    erd_power_envelope,
    generate_trial,
    spatial_weights,
)

N_REPS = 400  # see check 3/4 docstring - 60 was too noisy to trust the ordering


def check_envelope_shape() -> None:
    t = np.linspace(0.0, 4.0, 2000)

    erd_params = SyntheticSubjectParams(
        peak_freq_hz=10.0, onset_s=0.5, trough_s=1.5,
        erd_magnitude=0.6, rebound_magnitude=0.4, spatial_sigma_m=0.06,
    )
    env = erd_power_envelope(t, erd_params)
    baseline_val = env[t < erd_params.onset_s].mean()
    trough_val = env[np.argmin(np.abs(t - erd_params.trough_s))]
    rebound_peak_val = env.max()

    assert abs(baseline_val - 1.0) < 0.01, f"ERD baseline should be 1.0, got {baseline_val:.3f}"
    assert trough_val < 0.45, f"ERD trough should be near 1-0.6=0.4, got {trough_val:.3f}"
    assert rebound_peak_val > 1.3, f"ERD rebound should exceed baseline (~1.4), got {rebound_peak_val:.3f}"

    ers_params = SyntheticSubjectParams(
        peak_freq_hz=10.0, onset_s=0.5, trough_s=1.5,
        erd_magnitude=-0.3, rebound_magnitude=0.2, spatial_sigma_m=0.06,
    )
    env_ers = erd_power_envelope(t, ers_params)
    trough_val_ers = env_ers[np.argmin(np.abs(t - ers_params.trough_s))]
    assert trough_val_ers > 1.2, (
        f"ERS subject (erd_magnitude<0) should show a power INCREASE at trough "
        f"(~1.3), got {trough_val_ers:.3f}"
    )
    print("[1/4] envelope shape: PASS "
          f"(ERD trough={trough_val:.3f}, rebound_peak={rebound_peak_val:.3f}, "
          f"ERS trough={trough_val_ers:.3f})")


def check_contralaterality() -> None:
    """Compares C3 vs C4 modulation strength at moderate (0dB) SNR, not high
    SNR. At high SNR the noise floor is negligible everywhere, so every
    channel's *ratio* converges to the same value regardless of spatial
    weight - each channel's own signal component has identical relative
    modulation shape (weight only scales absolute amplitude, not the
    trough/baseline ratio). Spatial/contralateral specificity only shows up
    in the ratio metric once the fixed noise floor is comparable to the
    (weight-scaled) signal - confirmed by first trying this check at 20dB,
    where it failed (C3 and C4 both ~0.6, indistinguishable), then finding
    it passes cleanly at 0dB, matching check_spatial_monotonicity's choice
    of SNR for the same reason.
    """
    sfreq, duration = 250.0, 3.0
    params = SyntheticSubjectParams(
        peak_freq_hz=10.0, onset_s=0.5, trough_s=1.5,
        erd_magnitude=0.6, rebound_magnitude=0.4, spatial_sigma_m=0.06,
    )
    t = np.arange(int(duration * sfreq)) / sfreq
    baseline_mask = t < params.onset_s
    trough_mask = (t > params.trough_s - 0.1) & (t < params.trough_s + 0.1)

    rng = np.random.default_rng(0)
    ratios = {("left_hand", "C4"): [], ("left_hand", "C3"): [],
              ("right_hand", "C3"): [], ("right_hand", "C4"): []}
    for _ in range(N_REPS):
        for label in ["left_hand", "right_hand"]:
            trial = generate_trial(params, label, IV2A_CHANNELS, sfreq, duration, snr_db=0.0, rng=rng)
            for ch in ["C3", "C4"]:
                idx = IV2A_CHANNELS.index(ch)
                base_p = _band_limited_variance(trial[idx], sfreq, params.peak_freq_hz, baseline_mask)
                trough_p = _band_limited_variance(trial[idx], sfreq, params.peak_freq_hz, trough_mask)
                ratios[(label, ch)].append(trough_p / base_p)

    left_c4 = np.mean(ratios[("left_hand", "C4")])
    left_c3 = np.mean(ratios[("left_hand", "C3")])
    right_c3 = np.mean(ratios[("right_hand", "C3")])
    right_c4 = np.mean(ratios[("right_hand", "C4")])

    assert left_c4 < left_c3 - 0.05, (
        f"left_hand should show stronger ERD at C4 (contralateral) than C3, "
        f"got C4={left_c4:.3f} C3={left_c3:.3f}"
    )
    assert right_c3 < right_c4 - 0.05, (
        f"right_hand should show stronger ERD at C3 (contralateral) than C4, "
        f"got C3={right_c3:.3f} C4={right_c4:.3f}"
    )
    print(f"[2/4] contralaterality: PASS "
          f"(left_hand: C4={left_c4:.3f} < C3={left_c3:.3f}; "
          f"right_hand: C3={right_c3:.3f} < C4={right_c4:.3f})")


def _band_limited_variance(x: np.ndarray, sfreq: float, center_hz: float, mask: np.ndarray) -> float:
    sos = butter(4, [center_hz - 2.0, center_hz + 2.0], btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, x)
    return float(np.var(filtered[mask]))


def _measure_ratios(params, snr_db, channels, sfreq, duration, n_reps, seed):
    t = np.arange(int(duration * sfreq)) / sfreq
    baseline_mask = t < params.onset_s
    trough_mask = (t > params.trough_s - 0.1) & (t < params.trough_s + 0.1)
    rng = np.random.default_rng(seed)
    ratios = {}
    for ch in channels:
        idx = IV2A_CHANNELS.index(ch)
        vals = np.zeros((n_reps, 2))
        for i in range(n_reps):
            trial = generate_trial(params, "left_hand", IV2A_CHANNELS, sfreq, duration, snr_db=snr_db, rng=rng)
            vals[i] = [
                _band_limited_variance(trial[idx], sfreq, params.peak_freq_hz, baseline_mask),
                _band_limited_variance(trial[idx], sfreq, params.peak_freq_hz, trough_mask),
            ]
        ratios[ch] = vals[:, 1].mean() / vals[:, 0].mean()
    return ratios


def check_spatial_monotonicity() -> None:
    sfreq, duration = 250.0, 3.0
    params = SyntheticSubjectParams(
        peak_freq_hz=10.0, onset_s=0.5, trough_s=1.5,
        erd_magnitude=0.6, rebound_magnitude=0.4, spatial_sigma_m=0.06,
    )
    channels = ["C4", "FCz", "POz", "P1"]
    weights = spatial_weights(IV2A_CHANNELS, "C4", params.spatial_sigma_m)
    ordered = sorted(channels, key=lambda c: -weights[IV2A_CHANNELS.index(c)])

    # Mid SNR (0dB): the regime where spatial falloff is most cleanly
    # observable (high SNR washes differences out toward the true envelope
    # ratio for every channel; very low SNR washes everything toward 1.0 -
    # see check 4).
    ratios = _measure_ratios(params, 0.0, channels, sfreq, duration, N_REPS, seed=2)
    ordered_ratios = [ratios[c] for c in ordered]
    monotonic = all(a <= b + 0.03 for a, b in zip(ordered_ratios, ordered_ratios[1:]))
    detail = ", ".join(f"{c}={ratios[c]:.3f}" for c in ordered)
    assert monotonic, (
        f"channel ordering by spatial weight should match ordering by ERD "
        f"strength (lower ratio = stronger) at n={N_REPS}: {detail}"
    )
    print(f"[3/4] spatial monotonicity (0dB, n={N_REPS}): PASS ({detail})")


def check_snr_detectability() -> None:
    sfreq, duration = 250.0, 3.0
    params = SyntheticSubjectParams(
        peak_freq_hz=10.0, onset_s=0.5, trough_s=1.5,
        erd_magnitude=0.6, rebound_magnitude=0.4, spatial_sigma_m=0.06,
    )
    # P1 has near-negligible spatial weight to C4 - at low SNR its apparent
    # modulation should wash out to ~1.0 (undetectable), while C4's should
    # remain clearly below 1.0 (detectable) throughout.
    high_snr = _measure_ratios(params, 20.0, ["C4", "P1"], sfreq, duration, N_REPS, seed=3)
    low_snr = _measure_ratios(params, -10.0, ["C4", "P1"], sfreq, duration, N_REPS, seed=3)

    assert high_snr["C4"] < 0.6, f"C4 should show clear ERD at 20dB, got {high_snr['C4']:.3f}"
    assert low_snr["C4"] < 0.75, f"C4 should still show ERD at -10dB, got {low_snr['C4']:.3f}"
    assert low_snr["P1"] > 0.8, (
        f"P1 (near-zero spatial weight) should wash out toward 1.0 at -10dB, "
        f"got {low_snr['P1']:.3f}"
    )
    assert low_snr["P1"] > low_snr["C4"] + 0.1, (
        f"at -10dB, P1's apparent modulation should be far weaker than C4's: "
        f"P1={low_snr['P1']:.3f} C4={low_snr['C4']:.3f}"
    )
    print(f"[4/4] SNR-dependent detectability: PASS "
          f"(C4: 20dB={high_snr['C4']:.3f} -10dB={low_snr['C4']:.3f}; "
          f"P1: 20dB={high_snr['P1']:.3f} -10dB={low_snr['P1']:.3f})")


def run() -> None:
    check_envelope_shape()
    check_contralaterality()
    check_spatial_monotonicity()
    check_snr_detectability()
    print("\nStage 0 synthetic generator validated - safe to build Module A on top of it.")


if __name__ == "__main__":
    run()
