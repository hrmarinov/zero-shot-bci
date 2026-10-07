"""Phase 9b - do flow-field features add anything on top of CSP?

Phase 9 found flow-field features carry real but weak signal standalone
(28.7% mean, chance 25%, Wilcoxon p=0.02 across subjects) - nowhere near
CSP alone (62.2%). The question worth asking before shelving the idea,
matching Phase 5's precedent exactly (wavelet+MiniRocket+CSP combination):
does concatenating flow features onto CSP's own log-variance features and
fitting a combined LogisticRegressionCV classifier help or hurt versus CSP
alone? There's a principled reason to hope for complementary value here
that Phase 5 didn't have as cleanly - flow features capture *within-trial
temporal evolution* of the spatial pattern, something CSP's log-variance
features (one static number per spatial filter, no time-course information
at all) structurally cannot see - but "principled reason to hope" is not
evidence, so this is tested, not assumed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from mne.decoding import CSP
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from src.adaptation.flow_field import electrode_positions_2d, trial_flow_features
from src.config import IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import IV2A_CHANNELS, load_iv2a_subject
from src.evaluate import accuracy_summary, fit_predict, mcnemar_test, wilson_interval
from src.baselines import make_csp_lda

GRID_SIZE = 16
WINDOW_S, STRIDE_S = 0.25, 0.125
SFREQ = 250.0


def extract_flow_features(X: np.ndarray, positions_2d: np.ndarray) -> np.ndarray:
    return np.stack([
        trial_flow_features(trial, positions_2d, SFREQ, WINDOW_S, STRIDE_S, GRID_SIZE, roi_radius=None)
        for trial in X
    ])


def run() -> pd.DataFrame:
    positions_2d = electrode_positions_2d(IV2A_CHANNELS)
    rows = []

    for subject in IV2A_SUBJECTS:
        print(f"[subject {subject}] loading...")
        data = load_iv2a_subject(subject)

        csp_lda = make_csp_lda()
        y_pred_csp = fit_predict(csp_lda, data.X_calib, data.y_calib, data.X_eval)
        acc_csp, correct_csp, n_total = accuracy_summary(data.y_eval, y_pred_csp)

        csp = CSP(n_components=8, reg="ledoit_wolf", log=True, norm_trace=False)
        F_csp_calib = csp.fit_transform(data.X_calib, data.y_calib)
        F_csp_eval = csp.transform(data.X_eval)

        print(f"[subject {subject}] extracting flow features...")
        F_flow_calib = extract_flow_features(data.X_calib, positions_2d)
        F_flow_eval = extract_flow_features(data.X_eval, positions_2d)

        scaler = StandardScaler().fit(F_flow_calib)
        F_flow_calib = scaler.transform(F_flow_calib)
        F_flow_eval = scaler.transform(F_flow_eval)

        F_combined_calib = np.hstack([F_csp_calib, F_flow_calib])
        F_combined_eval = np.hstack([F_csp_eval, F_flow_eval])

        clf = LogisticRegressionCV(max_iter=2000, cv=5)
        clf.fit(F_combined_calib, data.y_calib)
        y_pred_combined = clf.predict(F_combined_eval)
        acc_combined, correct_combined, _ = accuracy_summary(data.y_eval, y_pred_combined)

        ci_low, ci_high = wilson_interval(correct_combined, n_total)
        p_value = mcnemar_test(data.y_eval, y_pred_csp, y_pred_combined)
        improvement = acc_combined - acc_csp

        print(f"[subject {subject}] csp={acc_csp:.3f} combined={acc_combined:.3f} "
              f"improvement={improvement:+.3f} p={p_value:.3f}")
        rows.append({
            "subject": subject, "acc_csp": acc_csp, "acc_combined": acc_combined,
            "ci_low": ci_low, "ci_high": ci_high, "improvement": improvement, "mcnemar_p": p_value,
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase9b_flow_field_plus_csp.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(df.to_string(index=False))

    stat = wilcoxon(df["acc_combined"], df["acc_csp"])
    print(f"\nMean CSP alone: {df['acc_csp'].mean():.3f}   Mean combined: {df['acc_combined'].mean():.3f}")
    print(f"Wilcoxon (combined vs CSP alone): {stat}")
    print(f"Combined beats CSP alone in {(df['acc_combined'] > df['acc_csp']).sum()}/9 subjects")
    return df


if __name__ == "__main__":
    run()
