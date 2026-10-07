"""Phase 7c - ComBat (replacing recenter+rescale) on real IV-2a data,
validated on synthetic ground truth first (stage0_combat_validation.py,
where ComBat consistently underperformed recenter+rescale across four
different eval trial counts - a real, honest null, not expected to reverse
on real data, but checked anyway for completeness).

CPD was meant to be tested here too (replacing ICP in the exact role Phase
7 already tested it in), but stage0_cpd_validation.py found - AFTER its
D=10 toy tests passed cleanly - that cpd_rigid_align does not work at
D=253 (this project's actual tangent-space dimensionality): a genuine
curse-of-dimensionality problem in the outlier hypothesis's log-weight,
confirmed to be dimension-driven (reproduced on a clean, non-EEG synthetic
point cloud, not an EEG-data quirk) and NOT fixed (see
coherent_point_drift.py's docstring for the full diagnosis). Running it on
real data would only produce garbage, so it is dropped from this real-data
phase entirely - Phase 7c ended up ComBat-only.

ComBat cannot be tested across sample sizes the way Phase 7's other
methods were: it has to see every trial it harmonizes in one call (no
separate "fit on a subsample, apply to unseen trials" step - see
combat_align.py's docstring for why), so it is only tested at n=288 (the
full eval session) - a narrower, less comparable condition than the other
phases' SAMPLE_SIZES ablation, reported as such, not padded out to look
like the same thing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.adaptation.combat_align import combat_harmonize
from src.adaptation.riemannian_icp import trial_tangent_vectors
from src.config import IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import load_iv2a_subject
from src.evaluate import accuracy_summary


def run() -> pd.DataFrame:
    rows = []
    for subject in IV2A_SUBJECTS:
        print(f"[subject {subject}] loading...")
        data = load_iv2a_subject(subject)

        calib_vecs = trial_tangent_vectors(data.X_calib)
        all_eval_vecs = trial_tangent_vectors(data.X_eval)

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(calib_vecs, data.y_calib)

        y_pred_none = lda.predict(all_eval_vecs)
        acc_none, _, _ = accuracy_summary(data.y_eval, y_pred_none)

        eval_combat = combat_harmonize(calib_vecs, all_eval_vecs)
        y_pred_combat = lda.predict(eval_combat)
        acc_combat, _, _ = accuracy_summary(data.y_eval, y_pred_combat)

        print(f"[subject {subject}] none={acc_none:.3f}  combat={acc_combat:.3f}")
        rows.append({"subject": subject, "acc_none": acc_none, "acc_combat": acc_combat})

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase7c_combat.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(f"mean none={df.acc_none.mean():.3f}  mean combat={df.acc_combat.mean():.3f}")
    print(f"combat beats none in {(df.acc_combat > df.acc_none).sum()}/{len(df)} subjects")
    return df


if __name__ == "__main__":
    run()
