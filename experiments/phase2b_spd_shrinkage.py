"""Phase 2b - SPD shrinkage for low-data cross-session alignment.

Transplanted from diffusion-tensor-imaging covariance-shrinkage statistics
(see docs/state_of_the_art.md Section 10.1). Simulates a calibration-free
live deployment: only n trials of *unlabeled* eval-session data are available
to estimate the alignment reference, instead of the full 288 used in Phase 2.

Leave-one-subject-out population prior (fit from the other 8 IV-2a subjects'
full calibration-session data) supplies the shrinkage target. At each n we
compare three ways of estimating the eval-session alignment reference from
only those n trials:
  - raw_ai:  affine-invariant Riemannian mean (same estimator as Phase 2,
             just fit on n trials instead of 288) - isolates the cost of
             having less data, holding the estimator family fixed.
  - raw_le:  log-Euclidean mean, no shrinkage - the shrinkage estimator's
             fair baseline (same metric family as shrunk, isolates the
             effect of shrinkage specifically).
  - shrunk:  log-Euclidean mean, shrunk toward the population prior.

The calibration side (CSP+LDA training) always uses the full, standard
Phase 2 alignment - only the eval side's reference is data-starved, matching
"you just walked into a new session with almost no data yet."
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
from src.adaptation.spd_shrinkage import (
    fit_eigenspace_prior,
    fit_population_prior,
    logeuclid_mean_cov,
    shrink_subject_mean,
    shrink_subject_mean_eigenspace,
)
from src.baselines import make_csp_lda
from src.config import IV2A_SUBJECTS, RANDOM_STATE, RESULTS_DIR
from src.datasets import load_iv2a_subject
from src.evaluate import accuracy_summary

DEFAULT_SAMPLE_SIZES = [8, 16, 32, 64, 128, 288]
DEFAULT_N_REPEATS = 5


def whiten(X: np.ndarray, mean_cov: np.ndarray) -> np.ndarray:
    whitening = invsqrtm(mean_cov)
    return np.einsum("cd,ndt->nct", whitening, X)


def run(
    subjects=IV2A_SUBJECTS,
    loader=load_iv2a_subject,
    sample_sizes=None,
    n_repeats: int = DEFAULT_N_REPEATS,
    out_name: str = "phase2b_spd_shrinkage_iv2a.csv",
    csp_components: int = 8,
    eigen_components: int | None = None,
) -> pd.DataFrame:
    sample_sizes = sample_sizes or DEFAULT_SAMPLE_SIZES
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    print("Loading all subjects once...")
    all_data = {s: loader(s) for s in subjects}
    full_n_eval = min(d.X_eval.shape[0] for d in all_data.values())
    sample_sizes = [n for n in sample_sizes if n <= full_n_eval]
    if sample_sizes[-1] != full_n_eval:
        sample_sizes.append(full_n_eval)

    for subject in subjects:
        print(f"[subject {subject}] building leave-one-out population prior...")
        data = all_data[subject]

        reference_X = [all_data[s].X_calib for s in subjects if s != subject]
        prior = fit_population_prior(reference_X)
        eigen_prior = fit_eigenspace_prior(prior, eigen_components) if eigen_components else None

        pipeline = make_csp_lda(n_components=csp_components)
        X_calib_aligned = RiemannianAlignment().fit_transform(data.X_calib)
        model = pipeline.fit(X_calib_aligned, data.y_calib)

        for n in sample_sizes:
            n_repeats_here = 1 if n == data.X_eval.shape[0] else n_repeats
            for repeat in range(n_repeats_here):
                idx = rng.choice(data.X_eval.shape[0], size=n, replace=False)
                X_sub = data.X_eval[idx]

                raw_ai_cov = mean_riemann(Covariances(estimator="oas").fit_transform(X_sub))
                raw_le_cov = logeuclid_mean_cov(X_sub)
                shrunk_cov = shrink_subject_mean(prior, X_sub)

                conditions = {
                    "raw_ai": whiten(data.X_eval, raw_ai_cov),
                    "raw_le": whiten(data.X_eval, raw_le_cov),
                    "shrunk": whiten(data.X_eval, shrunk_cov),
                }
                if eigen_prior is not None:
                    eigen_cov = shrink_subject_mean_eigenspace(eigen_prior, X_sub)
                    conditions["eigen"] = whiten(data.X_eval, eigen_cov)

                for method, X_eval_transformed in conditions.items():
                    y_pred = model.predict(X_eval_transformed)
                    acc, _, _ = accuracy_summary(data.y_eval, y_pred)
                    rows.append(
                        {
                            "subject": subject,
                            "n_trials": n,
                            "repeat": repeat,
                            "method": method,
                            "accuracy": acc,
                        }
                    )

        print(f"[subject {subject}] done")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / out_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved raw results to {out_path}")

    summary = df.groupby(["method", "n_trials"])["accuracy"].agg(["mean", "std"]).reset_index()
    print(summary.to_string(index=False))
    return df


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-name",
        default="phase2b_spd_shrinkage_iv2a.csv",
        help="Filename written under results/ (default: %(default)s)",
    )
    parser.add_argument(
        "--eigen-components",
        type=int,
        default=None,
        help=(
            "Add the anisotropic eigenvoice condition, shrinking in this many "
            "principal components (isotropic-only when omitted). The committed "
            "phase2c_eigenspace_k3/k7 CSVs were produced with 3 and 7."
        ),
    )
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Trial counts to sweep (default: %s)" % (DEFAULT_SAMPLE_SIZES,),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        sample_sizes=args.sample_sizes,
        n_repeats=args.n_repeats,
        out_name=args.out_name,
        eigen_components=args.eigen_components,
    )
