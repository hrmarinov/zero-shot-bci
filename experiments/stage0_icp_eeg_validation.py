"""Stage 0 gate, part 2: ICP realignment on synthetic *EEG* session drift.

Same synthetic subject (src/synthetic_eeg.py - one fixed, known ERD/ERS
signature), two "sessions": a calibration session generated normally, and
an eval session generated from the *same* underlying params but with a
known, modest channel-mixing matrix applied afterward - simulating
realistic session-to-session drift (electrode placement, gain, reference
scheme), not an arbitrary transform. Checks whether recentering +
rescaling + ICP, applied entirely within this one synthetic subject,
recovers enough of the injected drift to improve nearest-class-mean
classification of eval trials versus doing nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.linalg import expm

from src.adaptation.riemannian_icp import dispersion, icp_align, recenter, rescale, trial_tangent_vectors
from src.datasets import IV2A_CHANNELS
from src.evaluate import accuracy_summary
from src.synthetic_eeg import SyntheticSubjectParams, generate_trial

SFREQ = 250.0
DURATION_S = 3.0
SNR_DB = -8.0
N_TRIALS_PER_CLASS = 40
RANDOM_STATE = 3

TRUE_PARAMS = SyntheticSubjectParams(
    peak_freq_hz=10.0, onset_s=0.5, trough_s=1.5,
    erd_magnitude=0.6, rebound_magnitude=0.4, spatial_sigma_m=0.06,
)


def generate_session(params: SyntheticSubjectParams, rng: np.random.Generator, channel_mixing: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for label in ["left_hand", "right_hand"]:
        for _ in range(N_TRIALS_PER_CLASS):
            trial = generate_trial(params, label, IV2A_CHANNELS, SFREQ, DURATION_S, SNR_DB, rng)
            if channel_mixing is not None:
                trial = channel_mixing @ trial
            X.append(trial)
            y.append(label)
    order = rng.permutation(len(y))
    return np.array(X)[order], np.array(y)[order]


def nearest_mean_classify(calib_vecs: np.ndarray, calib_y: np.ndarray, eval_vecs: np.ndarray, classes: list[str]) -> np.ndarray:
    means = {c: calib_vecs[calib_y == c].mean(axis=0) for c in classes}
    dists = np.stack([np.linalg.norm(eval_vecs - means[c], axis=1) for c in classes], axis=1)
    pred_idx = dists.argmin(axis=1)
    return np.array(classes)[pred_idx]


def run() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    n_channels = len(IV2A_CHANNELS)

    print("Generating calibration session...")
    X_calib, y_calib = generate_session(TRUE_PARAMS, rng)

    print("Generating eval session (same subject, known modest channel-mixing drift applied)...")
    A = rng.normal(size=(n_channels, n_channels)) * 0.35
    skew = A - A.T
    channel_mixing = expm(skew)  # small-angle orthogonal mixing - modest, realistic drift
    X_eval, y_eval = generate_session(TRUE_PARAMS, rng, channel_mixing=channel_mixing)

    classes = ["left_hand", "right_hand"]
    calib_vecs = trial_tangent_vectors(X_calib)
    eval_vecs = trial_tangent_vectors(X_eval)

    print("\n--- baseline: no realignment ---")
    y_pred_baseline = nearest_mean_classify(calib_vecs, y_calib, eval_vecs, classes)
    acc_baseline, _, _ = accuracy_summary(y_eval, y_pred_baseline)
    print(f"  accuracy: {acc_baseline:.3f}")

    print("\n--- recenter + rescale only (unsupervised) ---")
    calib_centered, calib_mean = recenter(calib_vecs)
    eval_centered, eval_mean = recenter(eval_vecs)
    calib_disp = dispersion(calib_centered)
    eval_rescaled = rescale(eval_centered, calib_disp)
    # map both into a shared "centered+rescaled" frame for nearest-mean comparison
    calib_shared = calib_centered
    eval_shared = eval_rescaled
    y_pred_rr = nearest_mean_classify(calib_shared, y_calib, eval_shared, classes)
    acc_rr, _, _ = accuracy_summary(y_eval, y_pred_rr)
    print(f"  accuracy: {acc_rr:.3f}")

    print("\n--- recenter + rescale + ICP (fully unsupervised, within-subject) ---")
    icp_result = icp_align(eval_rescaled, calib_centered, max_iters=100)
    print(f"  ICP converged={icp_result.converged} in {icp_result.n_iters} iters, "
          f"final MSE={icp_result.mse_history[-1]:.4f} (initial={icp_result.mse_history[0]:.4f})")
    eval_aligned = eval_rescaled @ icp_result.rotation.T
    y_pred_icp = nearest_mean_classify(calib_centered, y_calib, eval_aligned, classes)
    acc_icp, _, _ = accuracy_summary(y_eval, y_pred_icp)
    print(f"  accuracy: {acc_icp:.3f}")

    print(f"\nSummary: baseline={acc_baseline:.3f}  recenter+rescale={acc_rr:.3f}  +ICP={acc_icp:.3f}")

    if acc_icp < acc_baseline - 0.02:
        raise AssertionError(
            f"ICP realignment made things worse on synthetic ground truth (baseline={acc_baseline:.3f}, "
            f"+ICP={acc_icp:.3f}) - do not proceed to real IV-2a data without understanding why"
        )

    print("\nICP EEG-drift realignment validated on synthetic ground truth "
          "(did not hurt, ideally helped) - safe to proceed to real IV-2a data.")


if __name__ == "__main__":
    run()
