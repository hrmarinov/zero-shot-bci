"""Diagnostic: is Module A's real-data underperformance (see phase3_module_a
smoke test) a covariance-prediction bug, or an underpowered reference pool?

Isolates the covariance prediction from the downstream CSP+LDA classifier by
comparing, in log-Euclidean distance, few-trial raw/learned covariance
estimates against each IV-2a eval subject's own full-288-trial affine-
invariant mean (a reasonable "gold" proxy for that subject's true
covariance - same methodology as stage0_module_a_validation.py, applied to
real data instead of synthetic). If "learned" is *closer* to gold than raw
here, the network is doing something directionally sensible and the harness/
whitening code is the more likely place to look; if "learned" is *farther*
even in this distance metric, the network itself is the problem.

Also prints ||pred_dev|| vs ||input_dev|| per subject - if pred_dev is near
enough zero to answer independent of the input, that's the "collapsed to
predicting the grand mean" failure mode; if it's large but still far from
gold, that's "confidently wrong," implying overfitting to too few reference
subjects rather than a training-loop bug.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from pyriemann.estimation import Covariances
from pyriemann.geometry.mean import mean_riemann
from pyriemann.geometry.tangentspace import tangent_space

from src.adaptation.geometric_correction import (
    _sample_size_feature,
    _trial_tangent_vectors,
    predict_corrected_mean,
    train_geometric_correction,
)
from src.adaptation.spd_shrinkage import logeuclid_mean_cov
from src.config import IV2A_SUBJECTS, RANDOM_STATE
from src.datasets import PHYSIONET_EXCLUDED_SUBJECTS, load_iv2a_subject, load_physionet_reference_subject


def log_euclid_distance(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    n = cov_a.shape[0]
    reference = np.eye(n)
    vec_a = tangent_space(cov_a[np.newaxis, :, :], reference, metric="logeuclid")[0]
    vec_b = tangent_space(cov_b[np.newaxis, :, :], reference, metric="logeuclid")[0]
    return float(np.linalg.norm(vec_a - vec_b))


def run(n_reference_subjects: int, n_eval_subjects: int, sample_sizes: list[int]) -> None:
    physionet_ids = [s for s in range(1, 110) if s not in PHYSIONET_EXCLUDED_SUBJECTS][:n_reference_subjects]
    print(f"Loading {len(physionet_ids)} PhysioNet reference subjects...")
    reference_X = [load_physionet_reference_subject(s) for s in physionet_ids]

    print("Training Module A...")
    model = train_geometric_correction(
        reference_subjects_X=reference_X, normalize_trace=True, random_state=RANDOM_STATE,
    )
    print(f"  final train_loss={model.train_losses_[-1]:.5f}  final val_loss={model.val_losses_[-1]:.5f}")
    print(f"  ||grand_mean|| = {np.linalg.norm(model.grand_mean_):.4f}")

    rng = np.random.default_rng(RANDOM_STATE)
    for subject in IV2A_SUBJECTS[:n_eval_subjects]:
        data = load_iv2a_subject(subject)
        gold_cov = mean_riemann(Covariances(estimator="oas").fit_transform(data.X_eval))
        print(f"\n[subject {subject}] gold covariance from all {data.X_eval.shape[0]} eval trials")

        for n in sample_sizes:
            idx = rng.choice(data.X_eval.shape[0], size=n, replace=False)
            X_sub = data.X_eval[idx]

            raw_ai_cov = mean_riemann(Covariances(estimator="oas").fit_transform(X_sub))
            raw_le_cov = logeuclid_mean_cov(X_sub)
            learned_cov = predict_corrected_mean(model, X_sub, normalize_trace=True)

            trial_vecs = _trial_tangent_vectors(X_sub, model.reference_, estimator="oas", normalize_trace=True)
            input_dev = trial_vecs.mean(axis=0) - model.grand_mean_
            x = np.concatenate([input_dev, _sample_size_feature(n)])
            model.net.eval()
            with torch.no_grad():
                pred_dev_t, alpha_t = model.net(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
                pred_dev = pred_dev_t.squeeze(0).numpy()
                alpha = alpha_t.item()

            d_raw_ai = log_euclid_distance(raw_ai_cov, gold_cov)
            d_raw_le = log_euclid_distance(raw_le_cov, gold_cov)
            d_learned = log_euclid_distance(learned_cov, gold_cov)

            print(f"  n={n:3d}  d(raw_ai,gold)={d_raw_ai:.4f}  d(raw_le,gold)={d_raw_le:.4f}  "
                  f"d(learned,gold)={d_learned:.4f}  alpha={alpha:.3f}  ||input_dev||={np.linalg.norm(input_dev):.4f}  "
                  f"||pred_dev||={np.linalg.norm(pred_dev):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reference-subjects", type=int, default=20)
    parser.add_argument("--n-eval-subjects", type=int, default=3)
    args = parser.parse_args()

    run(
        n_reference_subjects=args.n_reference_subjects,
        n_eval_subjects=args.n_eval_subjects,
        sample_sizes=[8, 32, 128],
    )
