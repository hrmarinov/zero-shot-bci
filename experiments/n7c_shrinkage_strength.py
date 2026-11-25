"""N7c, strength of the population shrinkage, chosen by accuracy on the OTHER subjects. PRE-REGISTERED.

Committed BEFORE any arm is scored. N7b (experiments/n7b_shrinkage_control.py) found that shrink1 (translate the subject's
session-1 trials by lam * (mu1 - m1_s), mu1 = population mean of the session-1 class-mean tangent features) gives R11 +1.09 pp
over base_I on Lee2019 with lam chosen by forecast error (median 0.4), while a ridge with shuffled pairing gave +1.94 pp, which
suggests that the accuracy-optimal pull is stronger. Choosing lam by session-2 accuracy of the SAME 54 subjects would be selection
on test labels; here lam is chosen for each held-out subject s ONLY from the other 53 subjects' session-2 accuracy (leave-one-
subject-out, nested). Same features, protocol, LR (C = 1) and R11 stack as N7; CPU only; Zhou2020 sealed, BNCI2015_001 unused.
N7b's base_I and shrink1 R11 accuracies are recomputed and must equal results/n7b_shrinkage_control_lee2019.csv exactly.

GRID (fixed here, not tuned): lam in {0, 0.2, 0.4, 0.6, 0.8, 1.0}; lam = 1 sets the subject's session-1 class means to the population's.
ARMS:
  fixed_lam  shrink1 with each lam of the grid for every subject (descriptive: the accuracy-vs-lam curve; NOT for selection).
  auto       lam(s) = argmax over the grid of the mean R11 session-2 accuracy of the other 53 subjects o, where each o is shrunk
             toward the population mean of the 52 subjects other than o and s (ties: the smaller lam). Applied to s with the
             population mean of the 53 others, exactly as N7b's shrink1.
PRIMARY ENDPOINT. R11[auto] - R11[shrink1 of N7b], Lee2019 n = 54, per-subject, one-sided sign-flip Monte Carlo (as D2); a single
contrast, no Holm. SECONDARY (reported): R11[auto] - base_I, the D3 transfer-limited subjects, the bottom third by csp_lda, share
>= 70 %, up/down, the distribution of the chosen lam, and the oracle-in-hindsight fixed lam (reported to show the optimism of picking
it on the cohort, never used).
DECISION. auto replaces shrink1 as the candidate only if R11[auto] - R11[shrink1] >= +0.5 pp with one-sided p < 0.05; otherwise
shrink1 (lam by forecast error) stays, and the lane is closed with the finding "strength tuning adds nothing reliable". Either
way the result remains a screen on the development cohort: a Zhou2020 opening is justified only for a candidate gain >= +3 pp
over base_I (R11[auto] - base_I), which is not expected.

    .venv\Scripts\python experiments\n7c_shrinkage_strength.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.d2_lee2019_dev import LITERACY, sign_flip_p  # noqa: E402
from experiments.n7_drift_forecast_screen import N_SUB, class_means, fit_predict, subject_features  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402

GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
MIN_GAIN_PP, ALPHA = 0.5, 0.05


def r11_acc(v: dict, m1: np.ndarray, mu: np.ndarray, lam: float) -> float:
    """R11 session-2 accuracy of a subject whose session-1 class means are shrunk toward ``mu`` with strength ``lam``."""
    shift = None if lam == 0 else lam * (mu - m1)
    p = fit_predict(v["F_tr"], v["y_tr"], shift, v["F_te"])
    return float((((v["cnn3"] + p + v["fts2"]) / 3).argmax(1) == v["y_te"]).mean())


def main() -> int:
    d3 = pd.read_csv(RESULTS_DIR / "d3_weak_subject_diagnosis.csv").set_index("subject")
    d2 = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    n7b = pd.read_csv(RESULTS_DIR / "n7b_shrinkage_control_lee2019.csv", index_col=0)
    S = {s: subject_features(s) for s in range(1, N_SUB + 1)}
    M1 = {s: class_means(v["F_tr"], v["y_tr"]) for s, v in S.items()}
    total = sum(M1.values())
    rows, chosen = {}, {}
    for s, v in S.items():
        others = [o for o in S if o != s]
        mu_s = (total - M1[s]) / (N_SUB - 1)
        row = {f"lam{lam:g}": 100 * r11_acc(v, M1[s], mu_s, lam) for lam in GRID}
        sel = np.zeros(len(GRID))
        for o in others:
            mu_o = (total - M1[o] - M1[s]) / (N_SUB - 2)
            sel += [r11_acc(S[o], M1[o], mu_o, lam) for lam in GRID]
        chosen[s] = GRID[int(np.argmax(sel))]  # argmax returns the first (smallest lam) on ties
        row["auto"] = row[f"lam{chosen[s]:g}"]
        rows[s] = row
    df = pd.DataFrame(rows).T
    if (df["lam0"] - n7b.r11_base_I).abs().max() > 1e-9:
        raise AssertionError("lam = 0 does not reproduce N7b base_I")
    ref = n7b.r11_shrink1
    print("reproduces N7b base_I; N7b shrink1 reference mean", round(ref.mean(), 2))
    print("accuracy vs fixed lam (descriptive; the best fixed lam below is picked in hindsight, optimistic):")
    for lam in GRID:
        print(f"  lam {lam:.1f}: R11 {df[f'lam{lam:g}'].mean():.2f}")
    print(f"auto: R11 {df.auto.mean():.2f}; chosen lam counts {pd.Series(chosen).value_counts().sort_index().to_dict()}")
    d = df.auto - ref
    p = sign_flip_p(d.to_numpy())
    print(f"PRIMARY auto - shrink1(N7b): {d.mean():+.2f} pp ({(d > 0).sum()} up/{(d < 0).sum()} down), p = {p:.4f}")
    g = df.auto - df.lam0
    print(f"secondary auto - base_I: {g.mean():+.2f} pp ({(g > 0).sum()} up/{(g < 0).sum()} down), p = {sign_flip_p(g.to_numpy()):.4f}")
    tl, third = d3.index[d3.category == "transfer_limited"], d2.csp_lda.sort_values().index[: N_SUB // 3]
    for name, idx in (("transfer_limited", tl), ("bottom third", third)):
        print(f"{name} n={len(idx)}: base_I {df.loc[idx, 'lam0'].mean():.2f}, shrink1(N7b) {ref[idx].mean():.2f}, auto {df.loc[idx, 'auto'].mean():.2f}")
    print(f"share >= 70 %: base_I {(df.lam0 >= 100 * LITERACY).mean():.1%}, auto {(df.auto >= 100 * LITERACY).mean():.1%}")
    ok = d.mean() >= MIN_GAIN_PP and p < ALPHA
    print(f"\nauto replaces shrink1 as the candidate: {ok}; gain over base_I {g.mean():+.2f} pp (Zhou2020 bar +3 pp: {g.mean() >= 3})")
    df.to_csv(RESULTS_DIR / "n7c_shrinkage_strength_lee2019.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
