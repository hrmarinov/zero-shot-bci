"""Phase 9 - scalp flow-field features on real IV-2a data.

The flow-field mechanism (src/adaptation/flow_field.py) is validated on
synthetic ground truth (stage0_flow_field_validation.py's primitives,
stage0_flow_field_eeg_validation.py's planted-drift recovery) but has not
touched real EEG until now. Per subject, per trial: trial_flow_features
reduces the trial's Horn-Schunck flow sequence to 6 summary scalars (mean
and std of u, v, and speed), classified by an LDA trained on that subject's
own calibration session and evaluated on their own eval session - the same
single-subject, no-cross-subject-data paradigm as Phase 1/2, so none of the
population-pooling failure mode applies here.

Two feature-set conditions, since phase8's channel-restriction result (hurts)
and the flow-field's own anisotropy finding (better-resolved on the densely-
sampled central row) cut in different directions and neither should be
assumed without a real test:
  - whole_grid: pooled over the entire interpolated grid
  - central_roi: pooled only within radius 0.6 of center (~central_17's span)

Compared against chance (25%, binomial test) first, since this is an honest
first look at whether single-trial flow features carry any usable signal at
all - the trial-averaged Stage 0 EEG gate needed ~200 trials to resolve
cleanly even in the easy direction, and single, unaveraged real trials are
a much harder, noisier regime than that gate tested.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.adaptation.flow_field import electrode_positions_2d, trial_flow_features
from src.config import IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import IV2A_CHANNELS, load_iv2a_subject
from src.evaluate import accuracy_summary, wilson_interval

GRID_SIZE = 16
WINDOW_S, STRIDE_S = 0.25, 0.125
CENTRAL_ROI_RADIUS = 0.6
FEATURE_SETS = {"whole_grid": None, "central_roi": CENTRAL_ROI_RADIUS}


def extract_features(X: np.ndarray, positions_2d: np.ndarray, sfreq: float, roi_radius: float | None) -> np.ndarray:
    return np.stack([
        trial_flow_features(trial, positions_2d, sfreq, WINDOW_S, STRIDE_S, GRID_SIZE, roi_radius=roi_radius)
        for trial in X
    ])


def run() -> pd.DataFrame:
    positions_2d = electrode_positions_2d(IV2A_CHANNELS)
    rows = []

    for subject in IV2A_SUBJECTS:
        print(f"[subject {subject}] loading...")
        data = load_iv2a_subject(subject)
        sfreq = 250.0  # IV-2a's native rate, matches src.config.FMIN/FMAX filtering already applied

        for set_name, roi_radius in FEATURE_SETS.items():
            print(f"[subject {subject}] extracting {set_name} flow features "
                  f"({len(data.X_calib)} calib + {len(data.X_eval)} eval trials)...")
            F_calib = extract_features(data.X_calib, positions_2d, sfreq, roi_radius)
            F_eval = extract_features(data.X_eval, positions_2d, sfreq, roi_radius)

            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(F_calib, data.y_calib)
            y_pred = lda.predict(F_eval)
            acc, n_correct, n_total = accuracy_summary(data.y_eval, y_pred)
            ci_low, ci_high = wilson_interval(n_correct, n_total)
            p_vs_chance = binomtest(n_correct, n_total, p=0.25, alternative="greater").pvalue

            print(f"[subject {subject}] {set_name}: acc={acc:.3f} [{ci_low:.3f}, {ci_high:.3f}] "
                  f"p_vs_chance={p_vs_chance:.4f}")
            rows.append({
                "subject": subject, "feature_set": set_name, "accuracy": acc,
                "ci_low": ci_low, "ci_high": ci_high, "n_correct": n_correct, "n_total": n_total,
                "p_vs_chance": p_vs_chance,
            })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase9_flow_field_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    summary = df.groupby("feature_set")["accuracy"].agg(["mean", "std"]).reindex(FEATURE_SETS.keys())
    print(summary.to_string())
    return df


if __name__ == "__main__":
    run()
