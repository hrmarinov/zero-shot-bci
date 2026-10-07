"""Qualitative sanity check: does wavelet calibration fitting on REAL IV-2a
calibration-session data produce physiologically plausible results (before
building the full accuracy-benchmark harness)?

Not a pass/fail gate like the synthetic one - real subjects don't have a
known ground truth to check against. This just prints the fitted params for
a few subjects so a human (or a second pass of reasoning) can sanity-check
them: is the fitted frequency in a plausible EEG band, is onset/trough
timing plausible for motor imagery, does the contralateral channel actually
come out with the largest gain more often than chance?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.adaptation.wavelet_calibration import fit_wavelet_calibration_multistart
from src.config import IV2A_SUBJECTS
from src.datasets import IV2A_CHANNELS, WAVELET_CALIB_TMIN, load_iv2a_subject_wide_window
from src.synthetic_eeg import CONTRALATERAL_ELECTRODE

SFREQ = 250.0


def run(subjects: list[int]) -> None:
    for subject in subjects:
        print(f"\n{'=' * 60}\n[subject {subject}]\n{'=' * 60}")
        data = load_iv2a_subject_wide_window(subject)

        for label in ["left_hand", "right_hand"]:
            mask = data.y_calib == label
            X = data.X_calib[mask]
            print(f"\n  {label}: {X.shape[0]} calibration trials")

            result = fit_wavelet_calibration_multistart(
                X, IV2A_CHANNELS, SFREQ, t0_s=WAVELET_CALIB_TMIN, baseline_window_s=(-1.0, -0.2),
                random_state=subject, verbose=False,
            )
            contra = CONTRALATERAL_ELECTRODE[label]
            contra_idx = IV2A_CHANNELS.index(contra)
            contra_freq = result.wavelet.center_freq[contra_idx].item()

            gains = np.abs(result.channel_gain)
            ranked = np.argsort(-gains)
            top5 = [(IV2A_CHANNELS[i], gains[i]) for i in ranked[:5]]

            print(f"    onset={result.onset_s:.3f}  trough={result.trough_s:.3f}  "
                  f"erd_magnitude={result.erd_magnitude:.3f}  rebound_magnitude={result.rebound_magnitude:.3f}")
            print(f"    contralateral channel {contra}: freq={contra_freq:.2f}Hz  "
                  f"|gain|={gains[contra_idx]:.3f}  (rank {list(ranked).index(contra_idx) + 1}/{len(IV2A_CHANNELS)})")
            print(f"    top-5 channels by |gain|: {top5}")
            print(f"    final train loss: {result.train_losses[-1]:.5f}  "
                  f"(initial: {result.train_losses[0]:.5f})")


if __name__ == "__main__":
    run(subjects=IV2A_SUBJECTS[:6])
