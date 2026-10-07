"""Diagnose the onset-near-boundary pattern seen in check_wavelet_calibration_real.py.

Two checks:
1. Direct inspection: compute a FIXED-frequency (not fitted) wavelet power
   envelope at the nominal contralateral channel, sampled at regular
   intervals, to see the actual shape of the real trial-averaged response -
   an early sharp spike (evoked-potential-like) vs a later, broader dip
   (ERD-like) should be visually distinguishable in the printed numbers.
2. Initialization sensitivity: refit with onset initialized late (0.8s)
   instead of the default (0.5s) and see whether it still converges back
   near the lower bound regardless of starting point - if so, that's
   evidence of a real early feature in the data pulling the fit there, not
   an optimization accident.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.adaptation.wavelet_calibration import MorletBank, fit_wavelet_calibration
from src.datasets import IV2A_CHANNELS, WAVELET_CALIB_TMIN, load_iv2a_subject_wide_window
from src.synthetic_eeg import CONTRALATERAL_ELECTRODE

SFREQ = 250.0


def inspect_fixed_frequency_envelope(X: np.ndarray, channel_idx: int, freq_hz: float = 10.0) -> None:
    n_times = X.shape[-1]
    t = WAVELET_CALIB_TMIN + np.arange(n_times) / SFREQ

    bank = MorletBank(n_channels=1, sfreq=SFREQ, init_freq_hz=freq_hz, init_sigma_s=0.15)
    with torch.no_grad():
        x_ch = torch.tensor(X[:, channel_idx : channel_idx + 1, :], dtype=torch.float32)
        power = bank(x_ch).mean(dim=0).squeeze(0).numpy()  # (n_times,)

    baseline_mask = (t >= -1.0) & (t < -0.2)
    baseline = power[baseline_mask].mean()
    normalized = power / (baseline + 1e-8)

    print(f"    fixed-{freq_hz}Hz power ratio (1.0=baseline) at sampled times:")
    sample_times = np.arange(-0.5, 3.01, 0.25)
    for st in sample_times:
        idx = np.argmin(np.abs(t - st))
        print(f"      t={st:+.2f}s  ratio={normalized[idx]:.3f}")


def check_onset_init_sensitivity(X: np.ndarray, label: str, subject: int) -> None:
    for init_onset_s in [0.15, 0.5, 0.9]:
        result = fit_wavelet_calibration(
            X, IV2A_CHANNELS, SFREQ, t0_s=WAVELET_CALIB_TMIN, baseline_window_s=(-1.0, -0.2),
            random_state=subject, init_onset_s=init_onset_s, verbose=False,
        )
        print(f"    [init_onset={init_onset_s:.2f}] fitted onset={result.onset_s:.3f}  "
              f"trough={result.trough_s:.3f}  erd_magnitude={result.erd_magnitude:.3f}")


def run() -> None:
    for subject, label in [(1, "left_hand"), (1, "right_hand"), (3, "left_hand"), (3, "right_hand")]:
        print(f"\n{'=' * 60}\n[subject {subject}] {label}\n{'=' * 60}")
        data = load_iv2a_subject_wide_window(subject)
        mask = data.y_calib == label
        X = data.X_calib[mask]
        contra = CONTRALATERAL_ELECTRODE[label]
        contra_idx = IV2A_CHANNELS.index(contra)

        print(f"  --- direct inspection at {contra} (fixed 10Hz wavelet, not fitted) ---")
        inspect_fixed_frequency_envelope(X, contra_idx, freq_hz=10.0)

        print(f"  --- init sensitivity ---")
        check_onset_init_sensitivity(X, label, subject)


if __name__ == "__main__":
    run()
