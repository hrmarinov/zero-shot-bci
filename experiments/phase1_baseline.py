"""Phase 1 — baseline reproduction.

For each IV-2a subject: reproduce a within-session CSP+LDA baseline (sanity
check against published numbers), then measure the cross-session accuracy
drop when training on the calibration session and testing on the evaluation
session with no adaptation at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.baselines import make_csp_lda
from src.config import IV2A_CHANCE_LEVEL, IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import load_iv2a_subject
from src.evaluate import (
    SubjectResult,
    cross_session_accuracy,
    wilson_interval,
    within_session_accuracy,
)


def run() -> pd.DataFrame:
    rows = []
    for subject in IV2A_SUBJECTS:
        print(f"[subject {subject}] loading...")
        data = load_iv2a_subject(subject)
        pipeline = make_csp_lda()

        within_acc, within_std = within_session_accuracy(pipeline, data.X_calib, data.y_calib)
        cross_acc, n_correct, n_total = cross_session_accuracy(
            pipeline, data.X_calib, data.y_calib, data.X_eval, data.y_eval
        )
        ci_low, ci_high = wilson_interval(n_correct, n_total)
        result = SubjectResult(
            subject=subject,
            within_session_acc=within_acc,
            within_session_std=within_std,
            cross_session_acc=cross_acc,
            cross_session_ci_low=ci_low,
            cross_session_ci_high=ci_high,
            cross_session_n=n_total,
        )
        # Flags whether the cross-session point estimate falls outside the
        # spread of the within-session repeated-CV scores (~95% band), i.e.
        # whether the drop looks bigger than within-session fold-to-fold noise.
        beyond_noise = abs(result.gap) > 1.96 * within_std
        above_chance = ci_low > IV2A_CHANCE_LEVEL
        print(
            f"[subject {subject}] within-session={within_acc:.3f}+-{within_std:.3f} "
            f"cross-session={cross_acc:.3f} [{ci_low:.3f}, {ci_high:.3f}] "
            f"gap={result.gap:.3f} beyond_noise={beyond_noise} above_chance={above_chance}"
        )
        rows.append(
            {
                "subject": subject,
                "within_session_acc": within_acc,
                "within_session_std": within_std,
                "cross_session_acc": cross_acc,
                "cross_session_ci_low": ci_low,
                "cross_session_ci_high": ci_high,
                "gap": result.gap,
                "gap_beyond_within_session_noise": beyond_noise,
                "cross_session_above_chance": above_chance,
            }
        )

    df = pd.DataFrame(rows)
    avg_row = {
        "subject": "average",
        "within_session_acc": df["within_session_acc"].mean(),
        "within_session_std": df["within_session_std"].mean(),
        "cross_session_acc": df["cross_session_acc"].mean(),
        "cross_session_ci_low": df["cross_session_ci_low"].mean(),
        "cross_session_ci_high": df["cross_session_ci_high"].mean(),
        "gap": df["gap"].mean(),
        "gap_beyond_within_session_noise": df["gap_beyond_within_session_noise"].sum(),
        "cross_session_above_chance": df["cross_session_above_chance"].sum(),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase1_baseline_iv2a.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run()
