"""Does the R11 pipeline beat CSP+LDA with Riemannian alignment? Paired, per subject, IV-2a + IV-2b.

CSP+LDA+RA is the Phase 2 reference: per-group Riemannian alignment, CSP (8 components, Ledoit-Wolf)
+ LDA on 8-30 Hz, 0.5-2.5 s, trained on the training session only. On IV-2b it is run here for the first
time under R11's protocol (train 0-2train, test 3test + 4test, per-recording alignment, subject score =
mean over the two test recordings).
  Check: the IV-2a recomputation equals Phase 2's cross_session_aligned_acc per subject (< 1e-12).
R11 per subject is read from results/r11_round3_diverse_members.csv; the EOG-clean R11 from
results/r14_eog_gate.csv. One-sided sign-flip test over the 18 subjects. CPU, ~2 min.

    .venv\\Scripts\\python experiments\\compare_r11_vs_csp_lda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.baselines import make_csp_lda  # noqa: E402
from src.config import IV2A_SUBJECTS, RESULTS_DIR  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402
from src.pipeline import align_by_group, load_subject_data  # noqa: E402

OUT_CSV = RESULTS_DIR / "compare_r11_vs_csp_lda.csv"
mne.set_log_level("ERROR")


def csp_lda_ra(ds: str, s: int) -> float:
    data, _ = load_subject_data(ds, s)
    A = align_by_group(data.X_tslr, data.groups)
    pred = make_csp_lda().fit(A[data.train], data.y[data.train]).predict(A[~data.train])
    y, rec = data.y[~data.train], data.groups[~data.train]
    return float(np.mean([(pred[rec == r] == y[rec == r]).mean() for r in np.unique(rec)]))


def main() -> int:
    r11 = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv").set_index(["dataset", "subject", "arm"]).acc
    r14 = pd.read_csv(RESULTS_DIR / "r14_eog_gate.csv").set_index(["dataset", "subject", "arm"]).acc
    phase2 = pd.read_csv(RESULTS_DIR / "phase2_riemannian_alignment_iv2a.csv")
    phase2 = phase2[phase2.subject != "average"].astype({"subject": int}).set_index("subject")
    rows = []
    for ds in ("IV-2a", "IV-2b"):
        for s in IV2A_SUBJECTS:
            csp = csp_lda_ra(ds, s)
            if ds == "IV-2a" and not abs(csp - phase2.loc[s, "cross_session_aligned_acc"]) < 1e-12:
                raise AssertionError(f"IV-2a s{s}: CSP+LDA+RA does not reproduce Phase 2")
            rows.append({"dataset": ds, "subject": s, "csp_lda_ra": csp, "r11": r11[(ds, s, "fts")],
                         "r11_eog_clean": r14[(ds, s, "B_clean")]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("Phase 2 IV-2a CSP+LDA+RA reproduced for all 9 subjects\n")
    print((100 * df.groupby("dataset")[["csp_lda_ra", "r11", "r11_eog_clean"]].mean()).round(2).to_string())
    for arm in ("r11", "r11_eog_clean"):
        d = 100 * (df[arm] - df.csp_lda_ra)
        p = sign_flip_permutation_test(d.to_numpy(), "greater")
        per = d.groupby(df.dataset).mean()
        print(f"\n{arm} - CSP+LDA+RA: pooled {d.mean():+.2f} pp, {(d > 0).sum()}/18 up, {(d < 0).sum()} down, "
              f"min {d.min():+.2f}, p = {p:.5f}; IV-2a {per['IV-2a']:+.2f}, IV-2b {per['IV-2b']:+.2f}")
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
