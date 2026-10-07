"""Phase 2 — Riemannian alignment for cross-session adaptation.

For each IV-2a subject: independently recenter the calibration and
evaluation session's trial covariances to identity (unsupervised, using only
each session's own unlabeled trials — no eval-session labels, no calibration
session trials involved in aligning the eval session), then run the same
CSP+LDA pipeline used in the Phase 1 baseline. Compare against the Phase 1
non-adapted cross-session accuracy with a paired McNemar test on the same
held-out trials, since the two conditions differ by a preprocessing step and
should be compared trial-by-trial rather than via two independent CIs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.adaptation.riemannian_align import RiemannianAlignment
from src.baselines import make_csp_lda
from src.config import IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import load_iv2a_subject
from src.evaluate import accuracy_summary, fit_predict, mcnemar_test, wilson_interval


def run() -> pd.DataFrame:
    rows = []
    for subject in IV2A_SUBJECTS:
        print(f"[subject {subject}] loading...")
        data = load_iv2a_subject(subject)
        pipeline = make_csp_lda()

        # Non-adapted baseline, recomputed here (deterministic, matches Phase 1).
        y_pred_baseline = fit_predict(pipeline, data.X_calib, data.y_calib, data.X_eval)
        baseline_acc, _, n_total = accuracy_summary(data.y_eval, y_pred_baseline)

        # Riemannian alignment: each session recentered independently, unsupervised.
        X_calib_aligned = RiemannianAlignment().fit_transform(data.X_calib)
        X_eval_aligned = RiemannianAlignment().fit_transform(data.X_eval)

        y_pred_aligned = fit_predict(pipeline, X_calib_aligned, data.y_calib, X_eval_aligned)
        aligned_acc, aligned_correct, _ = accuracy_summary(data.y_eval, y_pred_aligned)
        ci_low, ci_high = wilson_interval(aligned_correct, n_total)

        p_value = mcnemar_test(data.y_eval, y_pred_baseline, y_pred_aligned)
        improvement = aligned_acc - baseline_acc
        significant = p_value < 0.05

        print(
            f"[subject {subject}] baseline={baseline_acc:.3f} aligned={aligned_acc:.3f} "
            f"[{ci_low:.3f}, {ci_high:.3f}] improvement={improvement:+.3f} "
            f"p={p_value:.3f} significant={significant}"
        )
        rows.append(
            {
                "subject": subject,
                "cross_session_baseline_acc": baseline_acc,
                "cross_session_aligned_acc": aligned_acc,
                "cross_session_aligned_ci_low": ci_low,
                "cross_session_aligned_ci_high": ci_high,
                "improvement": improvement,
                "mcnemar_p_value": p_value,
                "improvement_significant": significant,
            }
        )

    df = pd.DataFrame(rows)
    avg_row = {
        "subject": "average",
        "cross_session_baseline_acc": df["cross_session_baseline_acc"].mean(),
        "cross_session_aligned_acc": df["cross_session_aligned_acc"].mean(),
        "cross_session_aligned_ci_low": df["cross_session_aligned_ci_low"].mean(),
        "cross_session_aligned_ci_high": df["cross_session_aligned_ci_high"].mean(),
        "improvement": df["improvement"].mean(),
        "mcnemar_p_value": float("nan"),
        "improvement_significant": df["improvement_significant"].sum(),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase2_riemannian_alignment_iv2a.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run()
