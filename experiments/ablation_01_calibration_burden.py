"""Ablation 01 - calibration burden curve (TODO.md A29).

Every accuracy this project has ever reported is measured at one fixed calibration
budget: all 288 labelled calibration trials. The question this script asks is the one
that actually matters for a BCI product, and the one the field (and industry) optimizes:
**how few labelled trials are needed to reach accuracy X?**

Riemannian alignment consumes *no labels* from the evaluation session - it only needs
that session's own unlabeled trials. So its advantage over non-adapted CSP+LDA should
*grow* as the labelled budget shrinks. If that holds, the project's real claim is not
"66.9% vs 62.2%" but something like "matches the 66.9% baseline with half the
calibration" - a strictly more useful and more defensible result.

Design
------
For each subject, sweep labelled calibration trials
`n in {32, 64, 96, 144, 192, 288}` (stratified: n/4 per class), with `--n-repeats`
independent draws for n < 288 (288 is the full session, so one deterministic draw).
Accuracy is always measured on the subject's full 288-trial evaluation session.

Three conditions, differing only in what the pipeline is allowed to see:

  none              CSP+LDA fit on the n labelled calibration trials alone, no alignment.
                    (The honest "no adaptation" baseline at this budget.)
  aligned           RiemannianAlignment fit on those same n trials (unsupervised use of
                    the same data - exactly what is available at deployment), CSP+LDA on
                    the aligned subset. Evaluation-session alignment uses that session's
                    full unlabeled trials, as in Phase 2.
  aligned_ref_full  Diagnostic: alignment reference estimated from the FULL calibration
                    session (288 unlabeled trials) while CSP+LDA is still trained on only
                    the n labelled trials. Separates "the alignment reference needs more
                    data" from "the classifier needs more labels".

Leakage discipline
------------------
Evaluation-session *labels* are never used for any fitting, in any condition. The
evaluation session's unlabeled trials are used to estimate its own whitening matrix -
the same transductive, label-free step as Phase 2 (`experiments/phase2_riemannian_alignment.py`).
Subsampling is stratified and seeded, so a rerun reproduces the committed CSV exactly.

Output: `results/ablation_calibration_burden.csv`
    subject, condition, n_trials, repeat, accuracy, n_correct, n_total

Usage:
    .venv\\Scripts\\python experiments\\ablation_01_calibration_burden.py
    .venv\\Scripts\\python experiments\\ablation_01_calibration_burden.py --n-repeats 5 --subjects 1 2
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import mem_report, setup_threads

setup_threads(8)  # before numpy

import numpy as np
import pandas as pd

from src.adaptation.riemannian_align import RiemannianAlignment
from src.baselines import make_csp_lda
from src.config import IV2A_CHANCE_LEVEL, IV2A_SUBJECTS, RANDOM_STATE, RESULTS_DIR
from src.datasets import load_iv2a_subject
from src.evaluate import accuracy_summary

DEFAULT_SAMPLE_SIZES = [32, 64, 96, 144, 192, 288]
DEFAULT_N_REPEATS = 20
N_CLASSES = 4


def stratified_subsample(y: np.ndarray, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of a class-balanced subsample of exactly `n_trials` trials.

    n_trials must be divisible by the number of classes, so that every class
    contributes equally - otherwise a small draw could accidentally be class-skewed,
    which would confound the accuracy-vs-budget curve with class imbalance.
    """
    if n_trials % N_CLASSES != 0:
        raise ValueError(f"n_trials={n_trials} must be divisible by n_classes={N_CLASSES}")
    per_class = n_trials // N_CLASSES
    picked: list[np.ndarray] = []
    for label in np.unique(y):
        class_idx = np.flatnonzero(y == label)
        if per_class > class_idx.size:
            raise ValueError(f"requested {per_class} trials of class {label!r}, only {class_idx.size} available")
        picked.append(rng.choice(class_idx, size=per_class, replace=False))
    return np.sort(np.concatenate(picked))


def run(
    subjects: list[int] | None = None,
    sample_sizes: list[int] | None = None,
    n_repeats: int = DEFAULT_N_REPEATS,
    n_components: int = 8,
    out_name: str = "ablation_calibration_burden.csv",
) -> pd.DataFrame:
    subjects = subjects if subjects is not None else IV2A_SUBJECTS
    sample_sizes = sample_sizes if sample_sizes is not None else DEFAULT_SAMPLE_SIZES
    rows: list[dict] = []

    for subject in subjects:
        print(f"\n[subject {subject}] loading...", flush=True)
        data = load_iv2a_subject(subject)
        n_calib, n_eval = data.X_calib.shape[0], data.X_eval.shape[0]
        print(f"[subject {subject}] calib={n_calib} trials, eval={n_eval} trials", flush=True)
        mem_report(f"subject {subject} loaded")

        # Evaluation-session whitening is unsupervised and identical across every
        # condition and budget, so compute it once. This is the only place evaluation
        # data is touched, and it never sees a label.
        X_eval_aligned = RiemannianAlignment().fit_transform(data.X_eval)
        # Full-calibration reference, used only by the `aligned_ref_full` diagnostic.
        calib_ref_full = RiemannianAlignment().fit(data.X_calib)
        X_calib_aligned_full = calib_ref_full.transform(data.X_calib)

        # Deterministic per-subject stream, independent of subject order.
        rng = np.random.default_rng(RANDOM_STATE + subject)

        for n in sample_sizes:
            if n > n_calib:
                print(f"[subject {subject}] skipping n={n} (only {n_calib} calibration trials)")
                continue
            repeats = 1 if n == n_calib else n_repeats
            for repeat in range(repeats):
                idx = np.sort(np.arange(n_calib)) if n == n_calib else stratified_subsample(data.y_calib, n, rng)
                X_sub, y_sub = data.X_calib[idx], data.y_calib[idx]

                conditions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

                # none: no adaptation at all.
                conditions["none"] = (X_sub, y_sub, data.X_eval)

                # aligned: alignment reference estimated from the same n trials the
                # classifier is trained on - what deployment actually has.
                align_sub = RiemannianAlignment()
                conditions["aligned"] = (align_sub.fit_transform(X_sub), y_sub, X_eval_aligned)

                # aligned_ref_full: same classifier budget, richer (unlabeled) reference.
                conditions["aligned_ref_full"] = (
                    X_calib_aligned_full[idx],
                    y_sub,
                    X_eval_aligned,
                )

                for condition, (X_train, y_train, X_test) in conditions.items():
                    pipeline = make_csp_lda(n_components=n_components)
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    acc, n_correct, n_total = accuracy_summary(data.y_eval, y_pred)
                    rows.append(
                        {
                            "subject": subject,
                            "condition": condition,
                            "n_trials": int(n),
                            "repeat": repeat,
                            "accuracy": acc,
                            "n_correct": n_correct,
                            "n_total": n_total,
                        }
                    )

        del data, X_eval_aligned, X_calib_aligned_full, calib_ref_full
        gc.collect()
        print(f"[subject {subject}] done", flush=True)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / out_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved raw results to {out_path}")

    summary = (
        df.groupby(["condition", "n_trials"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std", "count": "n_samples"})
        .sort_values(["condition", "n_trials"])
    )
    print("\n=== mean accuracy on the full 288-trial eval session, by budget ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    _print_trials_to_threshold(summary)
    return df


def _print_trials_to_threshold(
    summary: pd.DataFrame, thresholds: tuple[float, ...] = (0.55, 0.60, 0.62, 0.65)
) -> None:
    """Report the smallest swept budget whose mean accuracy reaches each threshold.

    Granularity-bound by construction: the sweep only visits the budgets in
    DEFAULT_SAMPLE_SIZES, so the answer is "the first visited budget at or above the
    threshold", never an interpolated value.
    """
    print("\n=== trials-to-threshold (first swept budget reaching the threshold) ===")
    for condition, block in summary.groupby("condition"):
        block = block.sort_values("n_trials")
        parts = []
        for threshold in thresholds:
            hit = block[block["accuracy_mean"] >= threshold]
            parts.append(f"{threshold:.0%}: {'never' if hit.empty else int(hit.iloc[0]['n_trials'])}")
        print(f"  {condition:<18} " + "  ".join(parts))
    print(
        f"\n(chance = {IV2A_CHANCE_LEVEL:.1%}; thresholds are on the mean across subjects, "
        "and the sweep is too coarse to resolve differences below ~32 trials)"
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="default: all 9")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=None, help=f"default: {DEFAULT_SAMPLE_SIZES}")
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument(
        "--out-name",
        default="ablation_calibration_burden.csv",
        help="filename written under results/ (use a smoketest name for partial runs)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        subjects=args.subjects,
        sample_sizes=args.sample_sizes,
        n_repeats=args.n_repeats,
        n_components=args.n_components,
        out_name=args.out_name,
    )
