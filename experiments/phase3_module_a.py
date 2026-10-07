"""Phase 3 - Module A (learned geometric correction) with a PhysioNet-scale reference pool.

Real-data counterpart to experiments/stage0_module_a_validation.py, now that
the synthetic gate has passed (docs/wavelet_personalization_pipeline.md §6,
build order item 2). Same evaluation protocol as phase2b/phase2d (paired
comparison across the 9 IV-2a subjects, same sample-size sweep), with a
fourth "learned" column added alongside raw_ai/raw_le/shrunk - directly,
quantitatively comparable to the already-reported shrinkage null result
(phase2b/phase2d), not a fresh unvalidated claim.

PhysioNet is a full 22/22-channel match with IV-2a (unlike OpenBMI, which is
missing FCz) - see src/datasets.py's PHYSIONET_CHANNELS assertion - so no
channel restriction is needed on the IV-2a side, only on PhysioNet's side
(load_physionet_reference_subject already does this). Trace-normalization is
still used (different recording hardware, same justification as Phase 2d).

Module A is trained *once* on the full PhysioNet reference pool and reused
across all 9 IV-2a eval subjects - PhysioNet subjects are external to IV-2a,
so there's no leave-one-out concern (unlike Phase 2b's within-IV-2a prior).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_riemann

from src.adaptation.geometric_correction import predict_corrected_mean, train_geometric_correction
from src.adaptation.riemannian_align import RiemannianAlignment
from src.adaptation.spd_shrinkage import fit_population_prior, logeuclid_mean_cov, shrink_subject_mean
from src.baselines import make_csp_lda
from src.config import IV2A_SUBJECTS, RANDOM_STATE, RESULTS_DIR
from src.datasets import PHYSIONET_EXCLUDED_SUBJECTS, load_iv2a_subject, load_physionet_reference_subject
from src.evaluate import accuracy_summary

SAMPLE_SIZES = [8, 16, 32, 64, 128, 288]
N_REPEATS = 5


def whiten(X: np.ndarray, mean_cov: np.ndarray) -> np.ndarray:
    whitening = invsqrtm(mean_cov)
    return np.einsum("cd,ndt->nct", whitening, X)


def build_reference_pool(physionet_subjects: list[int]) -> list[np.ndarray]:
    print(f"Loading {len(physionet_subjects)} PhysioNet reference subjects...")
    reference_X = []
    for s in physionet_subjects:
        X = load_physionet_reference_subject(s)
        reference_X.append(X)
        print(f"  PhysioNet subject {s}: {X.shape[0]} trials")
    return reference_X


def run(physionet_subjects: list[int], out_name: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    reference_X = build_reference_pool(physionet_subjects)

    print("\nFitting closed-form shrinkage prior on the reference pool...")
    prior = fit_population_prior(reference_X, normalize_trace=True)
    print(f"Population prior: between_subject_var={prior.between_subject_var_:.6f}  "
          f"trial_sampling_var={prior.trial_sampling_var_:.6f}  "
          f"ratio={prior.trial_sampling_var_ / prior.between_subject_var_:.3f}")

    print("\nTraining Module A on the reference pool...")
    model = train_geometric_correction(
        reference_subjects_X=reference_X,
        normalize_trace=True,
        random_state=RANDOM_STATE,
    )

    for subject in IV2A_SUBJECTS:
        print(f"\n[subject {subject}] loading (full 22-channel)...")
        data = load_iv2a_subject(subject)

        pipeline = make_csp_lda()
        X_calib_aligned = RiemannianAlignment().fit_transform(data.X_calib)
        clf = pipeline.fit(X_calib_aligned, data.y_calib)

        for n in SAMPLE_SIZES:
            n_repeats = 1 if n == data.X_eval.shape[0] else N_REPEATS
            for repeat in range(n_repeats):
                idx = rng.choice(data.X_eval.shape[0], size=n, replace=False)
                X_sub = data.X_eval[idx]

                raw_ai_cov = mean_riemann(Covariances(estimator="oas").fit_transform(X_sub))
                raw_le_cov = logeuclid_mean_cov(X_sub)
                shrunk_cov = shrink_subject_mean(prior, X_sub, normalize_trace=True)
                learned_cov = predict_corrected_mean(model, X_sub, normalize_trace=True)

                conditions = {
                    "raw_ai": whiten(data.X_eval, raw_ai_cov),
                    "raw_le": whiten(data.X_eval, raw_le_cov),
                    "shrunk": whiten(data.X_eval, shrunk_cov),
                    "learned": whiten(data.X_eval, learned_cov),
                }

                for method, X_eval_transformed in conditions.items():
                    y_pred = clf.predict(X_eval_transformed)
                    acc, _, _ = accuracy_summary(data.y_eval, y_pred)
                    rows.append({"subject": subject, "n_trials": n, "repeat": repeat, "method": method, "accuracy": acc})

        print(f"[subject {subject}] done")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / out_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    summary = df.groupby(["method", "n_trials"])["accuracy"].agg(["mean", "std"]).reset_index()
    print(summary.to_string(index=False))
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=108)
    parser.add_argument("--out-name", type=str, default="phase3_module_a_iv2a.csv")
    args = parser.parse_args()

    all_subjects = [s for s in range(1, 110) if s not in PHYSIONET_EXCLUDED_SUBJECTS]
    run(physionet_subjects=all_subjects[: args.n_subjects], out_name=args.out_name)
