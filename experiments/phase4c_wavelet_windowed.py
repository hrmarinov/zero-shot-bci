"""Phase 4c - wavelet calibration with per-channel, per-phase-window features + LDA.

Extends Phase 4b (per-channel, jitter-tolerant, LDA - 31.4% mean) with a
time-window breakdown: instead of one pooled-over-time correlation score
per (channel, class), five scores per (channel, class) - one per envelope
phase (baseline, onset-transition, trough/hold, rebound-transition,
recovery), using each template's own fitted phase boundaries.

Validated on synthetic ground truth first (diagnose_jitter_hypothesis.py):
slightly behind per-channel-only at zero jitter (0.844 vs 0.863 - more
features without more signal costs a little), but consistently *ahead* at
every non-zero jitter level tested (+3.7 to +6.3pp at 0.1-0.3s jitter).
Real IV-2a data has confirmed non-trivial jitter (the onset-floor-hugging
symptom diagnosed earlier), so this is the regime the real test matters in.

4 classes x 22 channels x 5 windows = 440 features from ~288 calibration
trials - a much tighter feature-to-sample ratio than Phase 4b's 88. Watch
for this being too aggressive even for shrinkage LDA; if Phase 4c
underperforms 4b on real data despite the synthetic result, that's the
likely reason.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.adaptation.wavelet_calibration import fit_wavelet_calibration_multistart, match_filter_score_windowed
from src.config import IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import IV2A_CHANNELS, WAVELET_CALIB_TMIN, load_iv2a_subject_wide_window
from src.evaluate import accuracy_summary, wilson_interval

SFREQ = 250.0
BASELINE_WINDOW_S = (-1.0, -0.2)


def _windowed_features(models: dict, X: np.ndarray) -> np.ndarray:
    blocks = [
        match_filter_score_windowed(models[l], X, SFREQ, WAVELET_CALIB_TMIN, BASELINE_WINDOW_S).reshape(len(X), -1)
        for l in IV2A_CLASSES
    ]
    return np.concatenate(blocks, axis=1)


def run(subjects: list[int] = IV2A_SUBJECTS, out_name: str = "phase4c_wavelet_windowed.csv") -> pd.DataFrame:
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

        calib_features = _windowed_features(models, data.X_calib)
        eval_features = _windowed_features(models, data.X_eval)

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(calib_features, data.y_calib)
        y_pred = lda.predict(eval_features)

        acc, n_correct, n_total = accuracy_summary(data.y_eval, y_pred)
        ci_low, ci_high = wilson_interval(n_correct, n_total)
        print(f"[subject {subject}] cross-session accuracy: {acc:.3f} ({n_correct}/{n_total})  "
              f"95% CI [{ci_low:.3f}, {ci_high:.3f}]  feature dim={calib_features.shape[1]}")

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
    print("Comparison: Phase 4b (per-channel only) = 31.4%, Phase 1 (raw CSP+LDA) = 62.2%, "
          "Phase 2 (Riemannian alignment) = 66.9%, chance level = 25.0%")
    return df


if __name__ == "__main__":
    run()
