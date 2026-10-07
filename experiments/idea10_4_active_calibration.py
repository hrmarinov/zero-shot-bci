"""Section 10.4 test — Bayesian-optimal (active-learning) calibration protocols.

For each IV-2a subject: instead of using the calibration session's 288
trials in a fixed order, simulate revealing them adaptively (uncertainty
sampling: reveal whichever remaining trial the current partial CSP+LDA model
is least certain about) vs. random order vs. the trials' original
as-collected order. Track cross-session accuracy as a function of how many
calibration trials have been revealed, for each strategy — the question is
whether active selection reaches good accuracy in fewer trials than a fixed
protocol needs.

Calibration and eval alignment (RiemannianAlignment) is computed once on the
full session, held fixed across all conditions — isolating trial-selection
strategy as the only variable, consistent with the Phase 2b/2c experiments.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.adaptation.active_calibration import fixed_order_batches, uncertainty_sampling_batches
from src.adaptation.riemannian_align import RiemannianAlignment
from src.baselines import make_csp_lda
from src.config import IV2A_SUBJECTS, RANDOM_STATE, RESULTS_DIR
from src.datasets import load_iv2a_subject
from src.evaluate import accuracy_summary

BATCH_SIZE = 16
N_RANDOM_REPEATS = 5


def stratified_seed(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    idx = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        idx.extend(rng.choice(cls_idx, size=per_class, replace=False))
    return np.array(idx)


def learning_curve(pipeline, X_pool, y_pool, X_eval, y_eval, batches: list[np.ndarray]) -> list[dict]:
    rows = []
    revealed: list[int] = []
    for batch in batches:
        revealed.extend(batch.tolist())
        model = clone(pipeline)
        model.fit(X_pool[revealed], y_pool[revealed])
        y_pred = model.predict(X_eval)
        acc, _, _ = accuracy_summary(y_eval, y_pred)
        rows.append({"n_trials": len(revealed), "accuracy": acc})
    return rows


def run() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for subject in IV2A_SUBJECTS:
        print(f"[subject {subject}] loading + aligning...")
        data = load_iv2a_subject(subject)
        X_calib = RiemannianAlignment().fit_transform(data.X_calib)
        X_eval = RiemannianAlignment().fit_transform(data.X_eval)
        y_calib, y_eval = data.y_calib, data.y_eval

        n_classes = len(np.unique(y_calib))
        per_class = BATCH_SIZE // n_classes
        pipeline = make_csp_lda()

        seed_idx = stratified_seed(y_calib, per_class, rng)

        print(f"[subject {subject}] uncertainty sampling...")
        active_batches = uncertainty_sampling_batches(pipeline, X_calib, y_calib, seed_idx, BATCH_SIZE)
        for row in learning_curve(pipeline, X_calib, y_calib, X_eval, y_eval, active_batches):
            rows.append({"subject": subject, "strategy": "active", "repeat": 0, **row})

        print(f"[subject {subject}] as-collected order...")
        collected_batches = fixed_order_batches(np.arange(len(y_calib)), seed_idx, BATCH_SIZE)
        for row in learning_curve(pipeline, X_calib, y_calib, X_eval, y_eval, collected_batches):
            rows.append({"subject": subject, "strategy": "as_collected", "repeat": 0, **row})

        for repeat in range(N_RANDOM_REPEATS):
            perm = rng.permutation(len(y_calib))
            random_batches = fixed_order_batches(perm, seed_idx, BATCH_SIZE)
            for row in learning_curve(pipeline, X_calib, y_calib, X_eval, y_eval, random_batches):
                rows.append({"subject": subject, "strategy": "random", "repeat": repeat, **row})

        print(f"[subject {subject}] done")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "idea10_4_active_calibration_iv2a.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    summary = df.groupby(["strategy", "n_trials"])["accuracy"].agg(["mean", "std"]).reset_index()
    print(summary.to_string(index=False))
    return df


if __name__ == "__main__":
    run()
