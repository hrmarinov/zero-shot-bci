"""Diagnose Phase 4's chance-level result: is single-trial noise swamping
the (trial-averaged) template signal, or is it a genuine cross-session
generalization failure?

Classifies the CALIBRATION session's own trials (same session the templates
were fit on, single-trial scoring, no averaging) via the same nearest-
template rule Phase 4 uses for the eval session. If in-sample calibration
accuracy is *also* near chance, the problem is single-trial noise
overwhelming the template match at test time - not a cross-session
generalization gap.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict

from src.adaptation.wavelet_calibration import fit_wavelet_calibration_multistart, match_filter_score, score_trials
from src.config import IV2A_CLASSES, RANDOM_STATE
from src.datasets import IV2A_CHANNELS, WAVELET_CALIB_TMIN, load_iv2a_subject_wide_window
from src.evaluate import accuracy_summary

SFREQ = 250.0
BASELINE_WINDOW_S = (-1.0, -0.2)


def run(subject: int = 1) -> None:
    print(f"[subject {subject}] loading...")
    data = load_iv2a_subject_wide_window(subject)

    models = {}
    for label in IV2A_CLASSES:
        mask = data.y_calib == label
        X = data.X_calib[mask]
        print(f"  fitting {label} template ({X.shape[0]} trials)...")
        models[label] = fit_wavelet_calibration_multistart(
            X, IV2A_CHANNELS, SFREQ, t0_s=WAVELET_CALIB_TMIN, baseline_window_s=BASELINE_WINDOW_S,
            random_state=subject, verbose=False,
        )

    print("\n--- in-sample: classify calibration trials (same session, single-trial), MSE (argmin) ---")
    mse_scores = {
        label: score_trials(models[label], data.X_calib, SFREQ, WAVELET_CALIB_TMIN, BASELINE_WINDOW_S)
        for label in IV2A_CLASSES
    }
    mse_matrix = np.stack([mse_scores[label] for label in IV2A_CLASSES], axis=1)
    pred_idx = np.argmin(mse_matrix, axis=1)
    y_pred = np.array([IV2A_CLASSES[i] for i in pred_idx])
    acc, n_correct, n_total = accuracy_summary(data.y_calib, y_pred)
    print(f"  in-sample calibration accuracy (MSE): {acc:.3f} ({n_correct}/{n_total})")

    print("\n--- in-sample: classify calibration trials, matched-filter (argmax) ---")
    scores = {
        label: match_filter_score(models[label], data.X_calib, SFREQ, WAVELET_CALIB_TMIN, BASELINE_WINDOW_S)
        for label in IV2A_CLASSES
    }
    score_matrix = np.stack([scores[label] for label in IV2A_CLASSES], axis=1)
    pred_idx = np.argmax(score_matrix, axis=1)
    y_pred = np.array([IV2A_CLASSES[i] for i in pred_idx])
    acc, n_correct, n_total = accuracy_summary(data.y_calib, y_pred)
    print(f"  in-sample calibration accuracy (matched-filter): {acc:.3f} ({n_correct}/{n_total})")

    print("\n--- per-class own-template score distribution (calibration trials) ---")
    for true_label in IV2A_CLASSES:
        mask = data.y_calib == true_label
        own_scores = scores[true_label][mask]
        other_scores = {ol: scores[ol][mask].mean() for ol in IV2A_CLASSES if ol != true_label}
        print(f"  true={true_label:10s}  own_template_score: mean={own_scores.mean():.4f} std={own_scores.std():.4f}  "
              f"other_template_scores: {[(ol, round(float(s), 4)) for ol, s in other_scores.items()]}")

    print("\n--- LDA on the 4-dim matched-filter score vector, 5-fold CV within calibration session ---")
    lda = LinearDiscriminantAnalysis()
    y_pred_cv = cross_val_predict(lda, score_matrix, data.y_calib, cv=5)
    acc, n_correct, n_total = accuracy_summary(data.y_calib, y_pred_cv)
    print(f"  LDA-on-scores cross-validated accuracy: {acc:.3f} ({n_correct}/{n_total})")


if __name__ == "__main__":
    run(subject=1)
