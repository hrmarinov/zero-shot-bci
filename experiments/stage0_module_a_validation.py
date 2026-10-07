"""Stage 0 gate for Module A (docs/wavelet_personalization_pipeline.md §6, item 1).

"Does Module A's correction network actually recover the known clean
covariance from noisy/few-trial input, at a known SNR?" - answered here on
synthetic data, with a known ground truth, before spending any real-data
compute (PhysioNet) on it. Real EEG never gives a ground-truth covariance to
check against; synthetic data does.

Training pool: 40 synthetic subjects, 200 trials each (moderate SNR).
Held-out test pool: 10 *fresh* synthetic subjects, never seen in training,
500 trials each - the extra trials aren't used as training signal, only to
compute a high-fidelity proxy for that subject's true covariance (large-n
convergence of the empirical Riemannian mean), which few-trial subsamples
from the *same* subject are then compared against.

Gate: at small n (8, 16 trials - the calibration-free regime this whole
pipeline targets), the learned correction's log-Euclidean distance to the
held-out ground truth must be lower than both uncorrected baselines (raw
affine-invariant mean, raw log-Euclidean mean). If Module A can't beat raw
estimates in this easy, abundant-data synthetic setting, something is
broken in the mechanism itself - not a "real reference pool too small" issue
(that risk, per doc §9, is specifically about real-data scale, and doesn't
apply here since synthetic subjects are free to generate) - and Module A
should not proceed to real PhysioNet-scale training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.geometry.mean import mean_riemann
from pyriemann.geometry.tangentspace import tangent_space

from src.adaptation.geometric_correction import predict_corrected_mean, train_geometric_correction
from src.adaptation.spd_shrinkage import logeuclid_mean_cov
from src.datasets import IV2A_CHANNELS
from src.synthetic_eeg import generate_synthetic_subject

SFREQ = 250.0
DURATION_S = 3.0
SNR_DB = 0.0
N_TRAIN_SUBJECTS = 40
N_TRAIN_TRIALS_PER_CLASS = 100  # 200 trials/subject
N_TEST_SUBJECTS = 10
N_TEST_TRIALS_PER_CLASS = 250  # 500 trials/subject - for a high-fidelity ground-truth proxy
TEST_SAMPLE_SIZES = [8, 16, 32, 64]
TEST_REPEATS = 10
RANDOM_STATE = 123


def log_euclid_distance(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    n = cov_a.shape[0]
    reference = np.eye(n)
    vec_a = tangent_space(cov_a[np.newaxis, :, :], reference, metric="logeuclid")[0]
    vec_b = tangent_space(cov_b[np.newaxis, :, :], reference, metric="logeuclid")[0]
    return float(np.linalg.norm(vec_a - vec_b))


def generate_pool(n_subjects: int, n_trials_per_class: int, seed_offset: int) -> list[np.ndarray]:
    pool = []
    for i in range(n_subjects):
        rng = np.random.default_rng(RANDOM_STATE + seed_offset + i)
        ds = generate_synthetic_subject(
            IV2A_CHANNELS, sfreq=SFREQ, n_trials_per_class=n_trials_per_class,
            duration_s=DURATION_S, snr_db=SNR_DB, rng=rng,
        )
        pool.append(ds.X)
    return pool


def run() -> pd.DataFrame:
    print(f"Generating {N_TRAIN_SUBJECTS} synthetic training subjects "
          f"({N_TRAIN_TRIALS_PER_CLASS * 2} trials each)...")
    train_X = generate_pool(N_TRAIN_SUBJECTS, N_TRAIN_TRIALS_PER_CLASS, seed_offset=0)

    print(f"Generating {N_TEST_SUBJECTS} held-out synthetic test subjects "
          f"({N_TEST_TRIALS_PER_CLASS * 2} trials each, never used in training)...")
    test_X = generate_pool(N_TEST_SUBJECTS, N_TEST_TRIALS_PER_CLASS, seed_offset=10_000)

    print("\nTraining Module A on the synthetic reference pool...")
    model = train_geometric_correction(
        reference_subjects_X=train_X,
        sample_sizes=[8, 16, 32, 64, 128],
        random_state=RANDOM_STATE,
    )

    print(f"\nEvaluating on {N_TEST_SUBJECTS} held-out subjects "
          f"at n_trials={TEST_SAMPLE_SIZES}, {TEST_REPEATS} repeats each...")
    rng = np.random.default_rng(RANDOM_STATE + 99)
    rows = []
    for subject_idx, X_full in enumerate(test_X):
        ground_truth_cov = mean_riemann(Covariances(estimator="oas").fit_transform(X_full))
        n_trials = X_full.shape[0]

        for n in TEST_SAMPLE_SIZES:
            for repeat in range(TEST_REPEATS):
                idx = rng.choice(n_trials, size=n, replace=False)
                X_sub = X_full[idx]

                raw_ai_cov = mean_riemann(Covariances(estimator="oas").fit_transform(X_sub))
                raw_le_cov = logeuclid_mean_cov(X_sub)
                learned_cov = predict_corrected_mean(model, X_sub)

                for method, cov in [("raw_ai", raw_ai_cov), ("raw_le", raw_le_cov), ("learned", learned_cov)]:
                    rows.append({
                        "subject": subject_idx,
                        "n_trials": n,
                        "repeat": repeat,
                        "method": method,
                        "distance_to_truth": log_euclid_distance(cov, ground_truth_cov),
                    })

    df = pd.DataFrame(rows)
    summary = df.groupby(["method", "n_trials"])["distance_to_truth"].agg(["mean", "std"]).reset_index()
    print("\n" + summary.to_string(index=False))

    print("\nGate checks (learned must beat both raw baselines at small n):")
    all_passed = True
    for n in [8, 16]:
        learned_mean = df[(df.method == "learned") & (df.n_trials == n)]["distance_to_truth"].mean()
        raw_ai_mean = df[(df.method == "raw_ai") & (df.n_trials == n)]["distance_to_truth"].mean()
        raw_le_mean = df[(df.method == "raw_le") & (df.n_trials == n)]["distance_to_truth"].mean()
        passed = learned_mean < raw_ai_mean and learned_mean < raw_le_mean
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  n={n:3d}: learned={learned_mean:.4f}  raw_ai={raw_ai_mean:.4f}  "
              f"raw_le={raw_le_mean:.4f}  [{status}]")

    if not all_passed:
        raise AssertionError(
            "Module A failed to beat raw estimates on synthetic data at small n - "
            "the mechanism itself is not working; do not proceed to real-data training "
            "(see docstring - this gate is specifically about the mechanism, not reference-pool size)"
        )

    print("\nModule A validated on synthetic ground truth - safe to proceed to real-data training.")
    return df


if __name__ == "__main__":
    run()
