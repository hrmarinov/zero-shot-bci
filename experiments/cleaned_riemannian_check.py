"""Artifact-cleaned pipeline check - CSP+LDA on ICA-cleaned IV-2a data.

Compares two cross-session conditions on the artifact-cleaned pipeline built in
`src/preprocessing.py` (notch -> wide bandpass -> ICA removal of EOG- and
muscle-correlated components), to test whether cleaned data is a usable drop-in
replacement for the raw MOABB pulls every other phase uses:

  - no realignment: the Phase 1 protocol applied to cleaned data.
  - + Riemannian alignment: the Phase 2 protocol applied to cleaned data.

Only the aligned column is written to `results/cleaned_riemannian_check.csv`
(the no-realignment number is printed; see the write-up in
`docs/progress_and_direction.md` for why it is the more diagnostic of the two -
each session's ICA is fit independently, so cleaning adds its own
session-specific spatial transform on top of whatever drift already existed).

Entry point reconstructed after the fact: the committed CSV was produced by an
ad-hoc script that was not kept in the tree, and nothing else in the repository
consumed `load_iv2a_subject_cleaned`. This reproduces that CSV's schema and
values from the loaders and evaluation protocol that remain, so the artifact is
no longer orphaned.

Note: unlike every other phase, this one does *not* re-download anything, but it
does depend on the cached cleaned recordings under `data/cleaned/` - the first
run creates them (~25-35s per subject/session).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.adaptation.riemannian_align import RiemannianAlignment
from src.baselines import make_csp_lda
from src.config import IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import load_iv2a_subject_cleaned
from src.evaluate import accuracy_summary, fit_predict


def run() -> pd.DataFrame:
    rows = []
    for subject in IV2A_SUBJECTS:
        print(f"[subject {subject}] loading cleaned sessions...")
        data = load_iv2a_subject_cleaned(subject)
        pipeline = make_csp_lda()

        # Phase 1 protocol on cleaned data.
        y_pred_none = fit_predict(pipeline, data.X_calib, data.y_calib, data.X_eval)
        acc_none, _, _ = accuracy_summary(data.y_eval, y_pred_none)

        # Phase 2 protocol on cleaned data: each session recentered independently.
        X_calib_aligned = RiemannianAlignment().fit_transform(data.X_calib)
        X_eval_aligned = RiemannianAlignment().fit_transform(data.X_eval)
        y_pred_aligned = fit_predict(pipeline, X_calib_aligned, data.y_calib, X_eval_aligned)
        acc_aligned, _, _ = accuracy_summary(data.y_eval, y_pred_aligned)

        print(f"[subject {subject}] cleaned no-realign={acc_none:.4f} cleaned+aligned={acc_aligned:.4f}")
        rows.append({"subject": subject, "acc_cleaned_aligned": acc_aligned})

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cleaned_riemannian_check.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(f"mean acc_cleaned_aligned = {df['acc_cleaned_aligned'].mean():.4f}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run()
