"""Phase 5 - combined classifier: CSP + wavelet-calibration + MiniRocket.

Per user direction: use CSP as leverage (the wavelet-calibration mechanism
alone, Phase 4, tops out well below CSP - concatenate rather than compete),
and add a ROCKET-style random-convolutional-kernel representation of the
raw temporal dynamics as a third, independent feature source. Uses sktime's
MiniRocketMultivariate (the modern, deterministic, multivariate-aware
ROCKET variant) rather than a hand-rolled implementation - a well-tested,
widely-used method, not something to reinvent.

Three feature blocks, each summarizing the data differently:
  - CSP (8-dim): spatial covariance/variance-ratio structure - the existing
    strong baseline (Phase 1/2's ~62-67%).
  - wavelet-calibration (4-dim): per-class matched-filter score against a
    per-session-fitted, known-shape ERD/ERS template - no population data,
    no cross-subject borrowing (Phase 4, ~25-38% depending on subject).
  - MiniRocket+Ridge decision scores (4-dim): raw temporal waveform shape
    via ~10,000 random convolutional kernels, distilled through a Ridge
    classifier (the standard ROCKET recipe - Ridge specifically because it
    handles high-dimensional, small-sample settings the way plain LDA
    can't) down to one decision score per class, so the *final* combining
    classifier sees a compact summary rather than being swamped by 10,000
    raw dimensions relative to ~250 calibration trials.

All three blocks and the final LDA are fit once on the calibration session,
evaluated once on the held-out eval session - the exact same cross-session
protocol as every other phase in this project, not a novel evaluation
methodology.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate

from src.adaptation.wavelet_calibration import fit_wavelet_calibration_multistart, match_filter_score
from src.config import IV2A_CLASSES, IV2A_SUBJECTS, RANDOM_STATE, RESULTS_DIR
from src.datasets import IV2A_CHANNELS, SubjectSessions, WAVELET_CALIB_TMIN, load_iv2a_subject, load_iv2a_subject_wide_window
from src.evaluate import accuracy_summary

SFREQ = 250.0
BASELINE_WINDOW_S = (-1.0, -0.2)
CSP_COMPONENTS = 8


def extract_csp_features(data: SubjectSessions) -> tuple[np.ndarray, np.ndarray]:
    csp = CSP(n_components=CSP_COMPONENTS, reg="ledoit_wolf", log=True, norm_trace=False)
    calib_feat = csp.fit_transform(data.X_calib, data.y_calib)
    eval_feat = csp.transform(data.X_eval)
    return calib_feat, eval_feat


def extract_wavelet_features(data_wide: SubjectSessions, subject: int) -> tuple[np.ndarray, np.ndarray]:
    models = {}
    for label in IV2A_CLASSES:
        mask = data_wide.y_calib == label
        X = data_wide.X_calib[mask]
        models[label] = fit_wavelet_calibration_multistart(
            X, IV2A_CHANNELS, SFREQ, t0_s=WAVELET_CALIB_TMIN, baseline_window_s=BASELINE_WINDOW_S,
            random_state=subject, verbose=False,
        )
    calib_feat = np.stack(
        [match_filter_score(models[l], data_wide.X_calib, SFREQ, WAVELET_CALIB_TMIN, BASELINE_WINDOW_S) for l in IV2A_CLASSES],
        axis=1,
    )
    eval_feat = np.stack(
        [match_filter_score(models[l], data_wide.X_eval, SFREQ, WAVELET_CALIB_TMIN, BASELINE_WINDOW_S) for l in IV2A_CLASSES],
        axis=1,
    )
    return calib_feat, eval_feat


def extract_rocket_features(data: SubjectSessions) -> tuple[np.ndarray, np.ndarray, RidgeClassifierCV]:
    rocket = MiniRocketMultivariate(random_state=RANDOM_STATE)
    calib_raw = rocket.fit_transform(data.X_calib)
    eval_raw = rocket.transform(data.X_eval)

    scaler = StandardScaler()
    calib_scaled = scaler.fit_transform(calib_raw)
    eval_scaled = scaler.transform(eval_raw)

    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(calib_scaled, data.y_calib)

    calib_feat = clf.decision_function(calib_scaled)
    eval_feat = clf.decision_function(eval_scaled)
    return calib_feat, eval_feat, clf


def run(subjects: list[int] = IV2A_SUBJECTS, out_name: str = "phase5_combined_features.csv") -> pd.DataFrame:
    rows = []

    for subject in subjects:
        print(f"\n[subject {subject}] loading...")
        data = load_iv2a_subject(subject)
        data_wide = load_iv2a_subject_wide_window(subject)

        print("  extracting CSP features...")
        csp_calib, csp_eval = extract_csp_features(data)

        print("  fitting wavelet-calibration templates...")
        wc_calib, wc_eval = extract_wavelet_features(data_wide, subject)

        print("  extracting MiniRocket features...")
        rocket_calib, rocket_eval, _ = extract_rocket_features(data)

        csp_lda = LinearDiscriminantAnalysis().fit(csp_calib, data.y_calib)
        csp_acc, _, _ = accuracy_summary(data.y_eval, csp_lda.predict(csp_eval))

        wc_lda = LinearDiscriminantAnalysis().fit(wc_calib, data.y_calib)
        wc_acc, _, _ = accuracy_summary(data.y_eval, wc_lda.predict(wc_eval))

        rocket_lda = LinearDiscriminantAnalysis().fit(rocket_calib, data.y_calib)
        rocket_acc, _, _ = accuracy_summary(data.y_eval, rocket_lda.predict(rocket_eval))

        # Each block standardized *independently* before concatenation - CSP
        # log-variance, wavelet matched-filter correlation, and Ridge
        # decision_function scores live on unrelated natural scales, which
        # otherwise distorts the combined LDA's covariance estimate and can
        # let a noisier block dilute a cleaner one (found happening here:
        # naive concatenation made accuracy *worse* than CSP alone). Shrinkage
        # LDA (solver="lsqr", shrinkage="auto") is also more appropriate than
        # plain LDA for 16 features from ~288 calibration trials split across
        # blocks of very different individual quality.
        block_scaler = StandardScaler()
        combined_calib = block_scaler.fit_transform(np.concatenate([csp_calib, wc_calib, rocket_calib], axis=1))
        combined_eval = block_scaler.transform(np.concatenate([csp_eval, wc_eval, rocket_eval], axis=1))
        combined_lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(combined_calib, data.y_calib)
        combined_lda_acc, _, _ = accuracy_summary(data.y_eval, combined_lda.predict(combined_eval))

        combined_logreg = LogisticRegressionCV(Cs=10, penalty="l2", max_iter=2000, cv=5).fit(combined_calib, data.y_calib)
        y_pred = combined_logreg.predict(combined_eval)
        combined_acc, n_correct, n_total = accuracy_summary(data.y_eval, y_pred)

        print(f"[subject {subject}] csp_only={csp_acc:.3f}  wavelet_only={wc_acc:.3f}  "
              f"rocket_only={rocket_acc:.3f}  combined_lda={combined_lda_acc:.3f}  combined_logreg={combined_acc:.3f}")

        rows.append({
            "subject": subject, "csp_only": csp_acc, "wavelet_only": wc_acc,
            "rocket_only": rocket_acc, "combined_lda": combined_lda_acc, "combined": combined_acc,
            "n_correct": n_correct, "n_total": n_total,
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / out_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    print(f"\n{'method':15s}  {'mean':>6s}  {'std':>6s}")
    for col in ["csp_only", "wavelet_only", "rocket_only", "combined_lda", "combined"]:
        print(f"{col:15s}  {df[col].mean():.3f}  {df[col].std():.3f}")
    print("\nComparison: Phase 1 (raw, no adaptation) = 62.2%, Phase 2 (Riemannian alignment) = 66.9%, "
          "chance level = 25.0%")
    return df


if __name__ == "__main__":
    run()
