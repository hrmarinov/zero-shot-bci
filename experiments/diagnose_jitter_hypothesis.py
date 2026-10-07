"""Test the trial-to-trial onset-jitter hypothesis with known ground truth.

Real-data diagnostics (diagnose_wavelet_onset.py) found fitted onset
consistently hugging its parameter floor, and Phase 4's per-trial
classification only modestly above chance - both consistent with real
trial-to-trial ERD onset jitter smearing the trial-averaged fit and
penalizing a fixed-lag matched-filter score, but not proven, since real
data has no known ground truth to check against.

Here: generate synthetic subjects with a KNOWN injected per-trial onset
jitter (src/synthetic_eeg.py's onset_jitter_std_s), fit with the *current*
mechanism (no jitter tolerance anywhere), and check whether (a) the fitted
onset is biased/distorted the way real data's was, and (b) per-trial
matched-filter classification accuracy degrades as jitter increases - a
clean, known-ground-truth confirmation (or disconfirmation) before building
a jitter-tolerant fix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from src.adaptation.wavelet_calibration import (
    fit_wavelet_calibration_multistart,
    match_filter_score,
    match_filter_score_per_channel,
    match_filter_score_windowed,
)
from src.datasets import IV2A_CHANNELS
from src.evaluate import accuracy_summary
from src.synthetic_eeg import CONTRALATERAL_ELECTRODE, SyntheticSubjectParams, generate_trial

SFREQ = 250.0
DURATION_S = 4.5
SNR_DB = 0.0
N_TRIALS = 80
RANDOM_STATE = 7

TRUE_PARAMS = SyntheticSubjectParams(
    peak_freq_hz=10.0, onset_s=0.5, trough_s=1.5,
    erd_magnitude=0.6, rebound_magnitude=0.4, spatial_sigma_m=0.06,
)


def run(jitter_std_s: float) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    print(f"\n=== onset_jitter_std_s={jitter_std_s} ===")

    X_left = np.array([
        generate_trial(TRUE_PARAMS, "left_hand", IV2A_CHANNELS, SFREQ, DURATION_S, SNR_DB, rng, jitter_std_s)
        for _ in range(N_TRIALS)
    ])
    X_right = np.array([
        generate_trial(TRUE_PARAMS, "right_hand", IV2A_CHANNELS, SFREQ, DURATION_S, SNR_DB, rng, jitter_std_s)
        for _ in range(N_TRIALS)
    ])

    models = {}
    for label, X in [("left_hand", X_left), ("right_hand", X_right)]:
        result = fit_wavelet_calibration_multistart(
            X, IV2A_CHANNELS, SFREQ, t0_s=0.0, baseline_window_s=(0.0, 0.5),
            random_state=RANDOM_STATE, verbose=False,
        )
        models[label] = result
        print(f"  [{label}] fitted onset={result.onset_s:.3f} (true {TRUE_PARAMS.onset_s})  "
              f"trough={result.trough_s:.3f} (true {TRUE_PARAMS.trough_s})")

    X_all = np.concatenate([X_left, X_right], axis=0)
    y_all = np.array(["left_hand"] * N_TRIALS + ["right_hand"] * N_TRIALS)

    scores = {
        label: match_filter_score(models[label], X_all, SFREQ, 0.0, (0.0, 0.5))
        for label in ["left_hand", "right_hand"]
    }
    score_matrix = np.stack([scores["left_hand"], scores["right_hand"]], axis=1)
    pred_idx = np.argmax(score_matrix, axis=1)
    y_pred = np.array(["left_hand", "right_hand"])[pred_idx]
    acc, n_correct, n_total = accuracy_summary(y_all, y_pred)
    print(f"  fixed-lag pooled classification accuracy: {acc:.3f} ({n_correct}/{n_total})")

    # Jitter-tolerant per-channel score, pooled via mean-over-channels (not
    # max - max-over-channels adds its own multiple-comparisons bias) for a
    # like-for-like comparison against the pooled fixed-lag score above -
    # a richer per-channel-feature classifier comes next, this just
    # isolates whether jitter tolerance alone recovers the lost accuracy.
    per_channel_scores = {
        label: match_filter_score_per_channel(models[label], X_all, SFREQ, 0.0, (0.0, 0.5))
        for label in ["left_hand", "right_hand"]
    }
    pooled_jt_scores = {label: per_channel_scores[label].mean(axis=1) for label in per_channel_scores}
    jt_score_matrix = np.stack([pooled_jt_scores["left_hand"], pooled_jt_scores["right_hand"]], axis=1)
    jt_pred_idx = np.argmax(jt_score_matrix, axis=1)
    jt_y_pred = np.array(["left_hand", "right_hand"])[jt_pred_idx]
    jt_acc, jt_n_correct, jt_n_total = accuracy_summary(y_all, jt_y_pred)
    print(f"  jitter-tolerant, naive mean-pooled accuracy: {jt_acc:.3f} ({jt_n_correct}/{jt_n_total})")

    # The real test: let a classifier learn how to weight the per-channel
    # features, instead of a hand-picked pooling rule (mean/max) that has
    # no way to know which channels actually carry signal.
    full_features = np.concatenate([per_channel_scores["left_hand"], per_channel_scores["right_hand"]], axis=1)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    y_pred_cv = cross_val_predict(lda, full_features, y_all, cv=5)
    lda_acc, lda_n_correct, lda_n_total = accuracy_summary(y_all, y_pred_cv)
    print(f"  jitter-tolerant, LDA-on-per-channel-features (5-fold CV) accuracy: "
          f"{lda_acc:.3f} ({lda_n_correct}/{lda_n_total})")

    # Next increment: per-channel, per-phase-window features (5x richer
    # again) - does the extra temporal resolution help, or is 220 features
    # from only 160 trials too much for shrinkage LDA to handle well?
    windowed_scores = {
        label: match_filter_score_windowed(models[label], X_all, SFREQ, 0.0, (0.0, 0.5))
        for label in ["left_hand", "right_hand"]
    }
    windowed_flat = {label: windowed_scores[label].reshape(len(X_all), -1) for label in windowed_scores}
    windowed_features = np.concatenate([windowed_flat["left_hand"], windowed_flat["right_hand"]], axis=1)
    lda_w = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    y_pred_w_cv = cross_val_predict(lda_w, windowed_features, y_all, cv=5)
    w_acc, w_n_correct, w_n_total = accuracy_summary(y_all, y_pred_w_cv)
    print(f"  jitter-tolerant, LDA-on-windowed-features (5-fold CV) accuracy: "
          f"{w_acc:.3f} ({w_n_correct}/{w_n_total})  [feature dim={windowed_features.shape[1]}]")

    # Phase 4c's windowed features didn't help on real data, hypothesized
    # to be a sample-efficiency cost (220-440 raw features from far fewer
    # trials). Test whether PCA-reducing the windowed features *before*
    # LDA recovers the synthetic gain without paying the full dimension
    # cost - PCA is refit inside each CV fold (via Pipeline) specifically
    # so this comparison isn't leaking eval-fold information into the
    # component directions.
    for n_components in [10, 20, 40]:
        pca_lda = Pipeline([
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])
        y_pred_pca_cv = cross_val_predict(pca_lda, windowed_features, y_all, cv=5)
        pca_acc, pca_n_correct, pca_n_total = accuracy_summary(y_all, y_pred_pca_cv)
        print(f"  jitter-tolerant, PCA({n_components})+LDA-on-windowed-features (5-fold CV) accuracy: "
              f"{pca_acc:.3f} ({pca_n_correct}/{pca_n_total})")

    # PCA is unsupervised - it keeps high-variance directions regardless of
    # whether they separate the classes, and most of a 220-dim matched-
    # filter feature space's variance is plausibly noise, not signal. Try a
    # *supervised* alternative instead: L1-penalized logistic regression,
    # which uses the labels themselves to zero out uninformative features.
    l1_logreg = LogisticRegressionCV(Cs=10, penalty="l1", solver="liblinear", max_iter=2000, cv=5)
    y_pred_l1_cv = cross_val_predict(l1_logreg, windowed_features, y_all, cv=5)
    l1_acc, l1_n_correct, l1_n_total = accuracy_summary(y_all, y_pred_l1_cv)
    print(f"  jitter-tolerant, L1-logreg-on-windowed-features (5-fold CV) accuracy: "
          f"{l1_acc:.3f} ({l1_n_correct}/{l1_n_total})")


if __name__ == "__main__":
    for jitter in [0.0, 0.1, 0.2, 0.3]:
        run(jitter)
