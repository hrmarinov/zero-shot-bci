"""Phase 4 - per-session wavelet calibration as a template-matching classifier.

Decisive real-data test of the mechanism discussed in this session: per
class, fit a wavelet-calibration template (src/adaptation/wavelet_calibration.py)
on the calibration session's own trials for that class - no population data,
no cross-subject borrowing anywhere. At test time, classify an eval-session
trial by whichever class's template it matches best (lowest score_trials
loss) - a template-matching classifier that falls directly out of the
calibration mechanism itself, not a separate model layered on top.

Same cross-session protocol and same 9 IV-2a subjects as every other phase
in this project, so the result is directly comparable to the already-
reported baselines: Phase 1 (non-adapted CSP+LDA, 62.2%), Phase 2
(Riemannian alignment, 66.9%).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.adaptation.wavelet_calibration import fit_wavelet_calibration_multistart, score_trials
from src.config import IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import IV2A_CHANNELS, WAVELET_CALIB_TMIN, load_iv2a_subject_wide_window
from src.evaluate import accuracy_summary, wilson_interval

SFREQ = 250.0
BASELINE_WINDOW_S = (-1.0, -0.2)


def run(subjects: list[int] = IV2A_SUBJECTS, out_name: str = "phase4_wavelet_calibration.csv") -> pd.DataFrame:
    rows = []

    for subject in subjects:
        print(f"\n[subject {subject}] loading (wide window)...")
        data = load_iv2a_subject_wide_window(subject)

        models = {}
        for label in IV2A_CLASSES:
            mask = data.y_calib == label
            X = data.X_calib[mask]
            print(f"  fitting {label} template ({X.shape[0]} calibration trials)...")
            models[label] = fit_wavelet_calibration_multistart(
                X, IV2A_CHANNELS, SFREQ, t0_s=WAVELET_CALIB_TMIN, baseline_window_s=BASELINE_WINDOW_S,
                random_state=subject, verbose=False,
            )
            print(f"    onset={models[label].onset_s:.3f}  trough={models[label].trough_s:.3f}  "
                  f"final_loss={models[label].train_losses[-1]:.5f}")

        scores = {
            label: score_trials(models[label], data.X_eval, SFREQ, WAVELET_CALIB_TMIN, BASELINE_WINDOW_S)
            for label in IV2A_CLASSES
        }
        score_matrix = np.stack([scores[label] for label in IV2A_CLASSES], axis=1)
        pred_idx = np.argmin(score_matrix, axis=1)
        y_pred = np.array([IV2A_CLASSES[i] for i in pred_idx])

        acc, n_correct, n_total = accuracy_summary(data.y_eval, y_pred)
        ci_low, ci_high = wilson_interval(n_correct, n_total)
        print(f"[subject {subject}] cross-session accuracy: {acc:.3f} ({n_correct}/{n_total})  "
              f"95% CI [{ci_low:.3f}, {ci_high:.3f}]")

        rows.append({
            "subject": subject, "accuracy": acc, "n_correct": n_correct, "n_total": n_total,
            "ci_low": ci_low, "ci_high": ci_high,
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / out_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(f"\nMean cross-session accuracy: {df['accuracy'].mean():.3f} (std {df['accuracy'].std():.3f})")
    print("Comparison: Phase 1 (raw, no adaptation) = 62.2%, Phase 2 (Riemannian alignment) = 66.9%, "
          "chance level = 25.0%")
    return df


if __name__ == "__main__":
    run()
