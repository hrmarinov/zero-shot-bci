"""Phase 2d - SPD shrinkage with an OpenBMI-scale reference pool.

Same low-data alignment question as Phase 2b (does shrinking a short-
recording covariance toward a population prior help?), but with the
diagnosed bottleneck fixed: Phase 2b/2c used 8 leave-one-out IV-2a subjects
as the reference pool and found no benefit, most likely because 8 subjects
can't reliably estimate between-subject variance in a 253-dimensional (IV-2a,
22-channel) or even a lower-dimensional tangent space. Here the reference
pool is OpenBMI subjects (up to 54, external to IV-2a - no leave-one-out
needed), restricted to the 21 channels the two datasets share (verified
compatible; see docs/state_of_the_art.md Section 10.1).

Trace-normalization is used when building the prior and when shrinking a
target subject's covariance, since OpenBMI and IV-2a were recorded on
different hardware and absolute EEG power scale differs for reasons
unrelated to real between-subject neurophysiology - see
src/adaptation/spd_shrinkage.py's normalize_trace docstring for why this is
provably harmless to downstream CSP+LDA accuracy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_riemann

from src.adaptation.riemannian_align import RiemannianAlignment
from src.adaptation.spd_shrinkage import fit_population_prior, logeuclid_mean_cov, shrink_subject_mean
from src.baselines import make_csp_lda
from src.config import IV2A_SUBJECTS, RANDOM_STATE, RESULTS_DIR
from src.datasets import load_iv2a_subject_shared_channels, load_openbmi_reference_subject
from src.evaluate import accuracy_summary

SAMPLE_SIZES = [8, 16, 32, 64, 128, 288]
N_REPEATS = 5


def whiten(X: np.ndarray, mean_cov: np.ndarray) -> np.ndarray:
    whitening = invsqrtm(mean_cov)
    return np.einsum("cd,ndt->nct", whitening, X)


def build_reference_pool(openbmi_subjects: list[int]):
    print(f"Loading {len(openbmi_subjects)} OpenBMI reference subjects...")
    reference_X = []
    for s in openbmi_subjects:
        X = load_openbmi_reference_subject(s)
        reference_X.append(X)
        print(f"  OpenBMI subject {s}: {X.shape[0]} trials")
    return fit_population_prior(reference_X, normalize_trace=True)


def run(openbmi_subjects: list[int], out_name: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    prior = build_reference_pool(openbmi_subjects)
    print(f"Population prior: between_subject_var={prior.between_subject_var_:.6f}  "
          f"trial_sampling_var={prior.trial_sampling_var_:.6f}  "
          f"ratio={prior.trial_sampling_var_ / prior.between_subject_var_:.3f}")

    for subject in IV2A_SUBJECTS:
        print(f"[subject {subject}] loading (21 shared channels)...")
        data = load_iv2a_subject_shared_channels(subject)

        pipeline = make_csp_lda()
        X_calib_aligned = RiemannianAlignment().fit_transform(data.X_calib)
        model = pipeline.fit(X_calib_aligned, data.y_calib)

        for n in SAMPLE_SIZES:
            n_repeats = 1 if n == data.X_eval.shape[0] else N_REPEATS
            for repeat in range(n_repeats):
                idx = rng.choice(data.X_eval.shape[0], size=n, replace=False)
                X_sub = data.X_eval[idx]

                raw_ai_cov = mean_riemann(Covariances(estimator="oas").fit_transform(X_sub))
                raw_le_cov = logeuclid_mean_cov(X_sub)
                shrunk_cov = shrink_subject_mean(prior, X_sub, normalize_trace=True)

                conditions = {
                    "raw_ai": whiten(data.X_eval, raw_ai_cov),
                    "raw_le": whiten(data.X_eval, raw_le_cov),
                    "shrunk": whiten(data.X_eval, shrunk_cov),
                }

                for method, X_eval_transformed in conditions.items():
                    y_pred = model.predict(X_eval_transformed)
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
    parser.add_argument("--n-subjects", type=int, default=20)
    parser.add_argument("--out-name", type=str, default="phase2d_openbmi_shrinkage_iv2a.csv")
    args = parser.parse_args()

    run(openbmi_subjects=list(range(1, args.n_subjects + 1)), out_name=args.out_name)
