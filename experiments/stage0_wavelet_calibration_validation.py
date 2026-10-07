"""Stage 0 gate for the wavelet-calibration mechanism (src/adaptation/wavelet_calibration.py).

Same validation discipline as stage0_module_a_validation.py: before trusting
a per-session backprop-fit wavelet calibration on real IV-2a data, check
that it actually recovers *known* ground truth on synthetic data where the
true onset/trough/magnitude/frequency/spatial-topology are known exactly.

Uses the same fixed-parameter synthetic subject as
stage0_synthetic_validation.py's checks (peak_freq_hz=10, onset_s=0.5,
trough_s=1.5, erd_magnitude=0.6, rebound_magnitude=0.4, spatial_sigma_m=0.06)
so the "known answer" is identical to what's already been independently
verified physiologically plausible - not a fresh, untested target.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.adaptation.wavelet_calibration import fit_wavelet_calibration_multistart
from src.datasets import IV2A_CHANNELS
from src.synthetic_eeg import CONTRALATERAL_ELECTRODE, SyntheticSubjectParams, generate_trial, spatial_weights

SFREQ = 250.0
DURATION_S = 4.5  # covers baseline (0-0.5s) through recovery_end (trough+2.0=3.5s) with margin
SNR_DB = 0.0
N_TRIALS = 80
RANDOM_STATE = 7

TRUE_PARAMS = SyntheticSubjectParams(
    peak_freq_hz=10.0, onset_s=0.5, trough_s=1.5,
    erd_magnitude=0.6, rebound_magnitude=0.4, spatial_sigma_m=0.06,
)


def generate_condition_trials(label: str, rng: np.random.Generator) -> np.ndarray:
    trials = [
        generate_trial(TRUE_PARAMS, label, IV2A_CHANNELS, SFREQ, DURATION_S, SNR_DB, rng)
        for _ in range(N_TRIALS)
    ]
    return np.array(trials)


def run() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    print(f"Generating {N_TRIALS} left_hand + {N_TRIALS} right_hand synthetic trials "
          f"(known params: onset={TRUE_PARAMS.onset_s}, trough={TRUE_PARAMS.trough_s}, "
          f"erd_magnitude={TRUE_PARAMS.erd_magnitude}, rebound_magnitude={TRUE_PARAMS.rebound_magnitude}, "
          f"peak_freq={TRUE_PARAMS.peak_freq_hz}Hz)...")
    X_left = generate_condition_trials("left_hand", rng)
    X_right = generate_condition_trials("right_hand", rng)

    true_weights = spatial_weights(IV2A_CHANNELS, "C4", TRUE_PARAMS.spatial_sigma_m)

    results = {}
    for label, X in [("left_hand", X_left), ("right_hand", X_right)]:
        print(f"\nFitting wavelet calibration on {label} trials...")
        result = fit_wavelet_calibration_multistart(
            X, IV2A_CHANNELS, SFREQ, t0_s=0.0, baseline_window_s=(0.0, 0.5),
            random_state=RANDOM_STATE, verbose=True,
        )
        results[label] = result
        print(f"  fitted: onset={result.onset_s:.3f} (true {TRUE_PARAMS.onset_s})  "
              f"trough={result.trough_s:.3f} (true {TRUE_PARAMS.trough_s})  "
              f"erd_magnitude={result.erd_magnitude:.3f} (true {TRUE_PARAMS.erd_magnitude})  "
              f"rebound_magnitude={result.rebound_magnitude:.3f} (true {TRUE_PARAMS.rebound_magnitude})")

    print("\nGate checks:")
    all_passed = True

    for label in ["left_hand", "right_hand"]:
        result = results[label]
        contra = CONTRALATERAL_ELECTRODE[label]
        contra_idx = IV2A_CHANNELS.index(contra)
        contra_freq = result.wavelet.center_freq[contra_idx].item()
        freq_ok = 7.0 <= contra_freq <= 13.0
        all_passed &= freq_ok
        print(f"  [{label}] contralateral ({contra}) fitted center_freq={contra_freq:.2f}Hz "
              f"(true {TRUE_PARAMS.peak_freq_hz}Hz)  [{'PASS' if freq_ok else 'FAIL'}]")

        onset_ok = abs(result.onset_s - TRUE_PARAMS.onset_s) < 0.2
        trough_ok = abs(result.trough_s - TRUE_PARAMS.trough_s) < 0.3
        all_passed &= onset_ok and trough_ok
        print(f"  [{label}] timing: onset diff={abs(result.onset_s - TRUE_PARAMS.onset_s):.3f} "
              f"[{'PASS' if onset_ok else 'FAIL'}]  trough diff={abs(result.trough_s - TRUE_PARAMS.trough_s):.3f} "
              f"[{'PASS' if trough_ok else 'FAIL'}]")

        contra_gain = abs(result.channel_gain[contra_idx])
        far_channels = ["POz", "P1"]
        far_gains = [abs(result.channel_gain[IV2A_CHANNELS.index(c)]) for c in far_channels]
        spatial_ok = all(contra_gain > g for g in far_gains)
        all_passed &= spatial_ok
        print(f"  [{label}] spatial: |gain[{contra}]|={contra_gain:.3f}  "
              f"|gain[far]|={[f'{g:.3f}' for g in far_gains]}  [{'PASS' if spatial_ok else 'FAIL'}]")

    if not all_passed:
        raise AssertionError(
            "wavelet calibration failed to recover known synthetic ground truth - "
            "the mechanism is not working, do not proceed to real IV-2a data"
        )

    print("\nWavelet calibration mechanism validated on synthetic ground truth - "
          "safe to proceed to real IV-2a data.")


if __name__ == "__main__":
    run()
