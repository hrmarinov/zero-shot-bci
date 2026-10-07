"""Phase 4b - wavelet calibration with per-channel, jitter-tolerant features + LDA.

Supersedes phase4_wavelet_calibration.py's pooled nearest-template
classifier. Two changes, both validated on synthetic ground truth first
(diagnose_jitter_hypothesis.py) before being trusted here:

1. Per-channel matched-filter scores (match_filter_score_per_channel)
   instead of one pooled scalar per class - lets a classifier learn which
   channels matter instead of committing to a single fixed pooling rule.
2. A Gaussian-smoothed template (jitter tolerance) instead of a fixed-lag
   comparison - real trial-to-trial ERD onset jitter (confirmed via known-
   ground-truth synthetic tests) otherwise penalizes genuine matches that
   are merely time-shifted from the trial-averaged template.

On synthetic data with known ground truth, this combination raised
per-trial classification accuracy from 75.0% (old pooled score) to 86.3%
at zero jitter, and stayed ahead of the old approach at every tested
jitter level (0.1-0.3s). Still no population data, no cross-subject
borrowing anywhere - templates and the combining LDA are both fit only on
this one session's own calibration trials.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.adaptation.wavelet_calibration import fit_wavelet_calibration_multistart, match_filter_score_per_channel
from src.config import IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import IV2A_CHANNELS, WAVELET_CALIB_TMIN, load_iv2a_subject_wide_window
from src.evaluate import accuracy_summary, wilson_interval

SFREQ = 250.0
BASELINE_WINDOW_S = (-1.0, -0.2)


def run(subjects: list[int] = IV2A_SUBJECTS, out_name: str = "phase4b_wavelet_per_channel.csv") -> pd.DataFrame:
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

        calib_features = np.concatenate(
            [match_filter_score_per_channel(models[l], data.X_calib, SFREQ, WAVELET_CALIB_TMIN, BASELINE_WINDOW_S) for l in IV2A_CLASSES],
            axis=1,
        )
        eval_features = np.concatenate(
            [match_filter_score_per_channel(models[l], data.X_eval, SFREQ, WAVELET_CALIB_TMIN, BASELINE_WINDOW_S) for l in IV2A_CLASSES],
            axis=1,
        )

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(calib_features, data.y_calib)
        y_pred = lda.predict(eval_features)

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
    print("Comparison: Phase 4 (pooled score) mean was 25.0%, Phase 1 (raw CSP+LDA) = 62.2%, "
          "Phase 2 (Riemannian alignment) = 66.9%, chance level = 25.0%")
    return df


if __name__ == "__main__":
    run()
