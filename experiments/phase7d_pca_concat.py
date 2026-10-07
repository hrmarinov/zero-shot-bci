"""Phase 7d - does giving the classifier BOTH the full-resolution (253-dim)
and a PCA-reduced (low-res) view of the same recenter+rescaled tangent
vectors, concatenated, help - rather than picking one or the other?

Motivated by stage0_cpd_pca_validation.py's finding: PCA-reduced CPD didn't
beat full-D ICP for *alignment*, but that doesn't settle whether a coarse,
low-dimensional summary is still a *useful, complementary classification
feature* alongside the full-resolution one (a coarse view could be more
robust to overfitting in the small-sample regime, even if it discards
detail).

Same caution as every other combination attempt in this project (Phase 5's
CSP+wavelet+MiniRocket, Phase 9b's CSP+flow-field): naively concatenating a
weaker/different feature onto a strong one has consistently *diluted* the
strong one so far, even through a regularized combiner. Tested honestly
here with the same regularized combiner (LogisticRegressionCV) Phase 5
found least-bad, not assumed to help.

Three conditions at n=288 (full eval session, matching this phase's other
methods' most-supervised condition): full-253-dim alone (matches Phase 7's
recenter_rescale=66.6%), PCA-reduced alone, and both concatenated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from src.adaptation.riemannian_icp import dispersion, recenter, rescale, trial_tangent_vectors
from src.config import IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import load_iv2a_subject
from src.evaluate import accuracy_summary

N_COMPONENTS = 20  # the d that gave PCA+CPD its best (if still losing) result in stage0_cpd_pca_validation.py


def run() -> pd.DataFrame:
    rows = []
    for subject in IV2A_SUBJECTS:
        print(f"[subject {subject}] loading...")
        data = load_iv2a_subject(subject)

        calib_vecs = trial_tangent_vectors(data.X_calib)
        eval_vecs = trial_tangent_vectors(data.X_eval)

        calib_centered, calib_mean = recenter(calib_vecs)
        eval_centered, eval_mean = recenter(eval_vecs)
        target_disp = dispersion(calib_centered)
        eval_rr = rescale(eval_centered, target_disp)  # recenter+rescale, matching Phase 7's own method
        calib_full = calib_centered
        eval_full = eval_rr

        pca = PCA(n_components=N_COMPONENTS)
        pca.fit(calib_full)
        calib_reduced = pca.transform(calib_full)
        eval_reduced = pca.transform(eval_full)

        scaler = StandardScaler().fit(calib_reduced)
        calib_reduced_s = scaler.transform(calib_reduced)
        eval_reduced_s = scaler.transform(eval_reduced)

        # full-dim alone
        lda_full = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda_full.fit(calib_full, data.y_calib)
        acc_full, _, _ = accuracy_summary(data.y_eval, lda_full.predict(eval_full))

        # reduced-dim alone
        lda_reduced = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda_reduced.fit(calib_reduced_s, data.y_calib)
        acc_reduced, _, _ = accuracy_summary(data.y_eval, lda_reduced.predict(eval_reduced_s))

        # concatenated (full + reduced), regularized combiner
        calib_concat = np.hstack([calib_full, calib_reduced_s])
        eval_concat = np.hstack([eval_full, eval_reduced_s])
        clf = LogisticRegressionCV(max_iter=2000, cv=5)
        clf.fit(calib_concat, data.y_calib)
        acc_concat, _, _ = accuracy_summary(data.y_eval, clf.predict(eval_concat))

        print(f"[subject {subject}] full={acc_full:.3f}  reduced(d={N_COMPONENTS})={acc_reduced:.3f}  concat={acc_concat:.3f}")
        rows.append({"subject": subject, "acc_full": acc_full, "acc_reduced": acc_reduced, "acc_concat": acc_concat})

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase7d_pca_concat.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(f"mean full={df.acc_full.mean():.3f}  mean reduced={df.acc_reduced.mean():.3f}  mean concat={df.acc_concat.mean():.3f}")
    print(f"concat beats full in {(df.acc_concat > df.acc_full).sum()}/{len(df)} subjects")
    return df


if __name__ == "__main__":
    run()
