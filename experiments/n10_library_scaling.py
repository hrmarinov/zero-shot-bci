"""N10, library-size scaling of the pooled cross-subject decoder. PRE-REGISTERED.

Committed BEFORE any arm is scored. N8 found a zero-shot pooled decoder (53 other Lee2019 subjects, none of the subject's own labels)
at R11 76.83 % against base_I 74.48 %; N9 found no such gain on IV-2a with a library of 8 subjects and 4 classes. Library size,
class count and montage are confounded there. N10 separates library size on Lee2019 itself. CPU only, cached; Zhou2020 sealed.
Same features, leave-one-subject-out protocol and R11 stack as N7 / N8 (src: experiments/n7_drift_forecast_screen.py).

ARMS (R11 with the TS+LR member swapped for a pooled LR, w = 0 i.e. zero-shot, C = 1, max_iter 2000). For each held-out subject s the
library is K other subjects drawn at random (numpy default_rng(1000 * s + draw), without replacement, from the 53 others), both of
their labelled sessions; K in {8, 13, 27, 53}; 3 independent draws per (s, K) for K < 53 (K = 53 is the whole library, one fit);
the subject's accuracy at K is the mean over its draws. base_I is recomputed and must equal results/n7_drift_forecast_lee2019.csv;
the K = 53 arm must equal N8's pool_w0 exactly.
ENDPOINT. gain(K) = R11[pool, K] - R11[base_I] per subject, Lee2019 n = 54; one-sided sign-flip Monte Carlo (as D2) for each K, and
the paired contrast gain(53) - gain(8).
DECISION (the branch the next step follows). SCALES iff gain(53) - gain(8) >= +1.0 pp with one-sided p < 0.05. FLAT otherwise.
  SCALES: IV-2a's null in N9 is attributed to the small library; the next step is a large EXTERNAL library (other datasets'
          left/right-hand trials, if cached or cheaply obtainable) tested on IV-2a / IV-2b as a binary task.
  FLAT:   the cross-subject line is closed as Lee2019-specific and the project moves to the calibration-burden reframe on cached data only.
REPORTED, NOT FOR DECISION: gain(K) mean, up/down, the transfer-limited subjects and the bottom third by csp_lda at each K; the
within-subject spread across draws.

    .venv\Scripts\python experiments\n10_library_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.d2_lee2019_dev import sign_flip_p  # noqa: E402
from experiments.n7_drift_forecast_screen import N_SUB, fit_predict, subject_features  # noqa: E402
from experiments.n8_pooled_decoder_screen import acc, pool_proba, r11_pred  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402

SIZES, N_DRAWS, MIN_DIFF_PP, ALPHA = (8, 13, 27, 53), 3, 1.0, 0.05


def pooled_acc(S: dict, s: int, k: int) -> tuple[float, float]:
    """Mean R11 accuracy (swap) over the draws of a K-subject library, and its spread across draws."""
    others = [o for o in S if o != s]
    draws = 1 if k == len(others) else N_DRAWS
    accs = []
    for d in range(draws):
        lib = set(np.random.default_rng(1000 * s + d).choice(others, k, replace=False)) if k < len(others) else set(others)
        accs.append(100 * acc(S[s], r11_pred(S[s], pool_proba(S, set(S) - lib, None, 0, S[s]))))
    return float(np.mean(accs)), float(np.std(accs))


def main() -> int:
    n7 = pd.read_csv(RESULTS_DIR / "n7_drift_forecast_lee2019.csv", index_col=0)
    n8 = pd.read_csv(RESULTS_DIR / "n8_pooled_decoder_lee2019.csv", index_col=0)
    d3 = pd.read_csv(RESULTS_DIR / "d3_weak_subject_diagnosis.csv").set_index("subject")
    d2 = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    S = {s: subject_features(s) for s in range(1, N_SUB + 1)}
    rows = {}
    for s, v in S.items():
        base = 100 * acc(v, r11_pred(v, fit_predict(v["F_tr"], v["y_tr"], None, v["F_te"])))
        rows[s] = {"base_I": base}
        for k in SIZES:
            rows[s][f"K{k}"], rows[s][f"K{k}_sd"] = pooled_acc(S, s, k)
        print(f"s{s}: base_I {base:.0f}, " + ", ".join(f"K{k} {rows[s][f'K{k}']:.1f}" for k in SIZES), flush=True)
    df = pd.DataFrame(rows).T
    if (df.base_I - n7.r11_base_I).abs().max() > 1e-9 or (df.K53 - n8.pool_w0).abs().max() > 1e-9:
        raise AssertionError("base_I or K = 53 does not reproduce N7 / N8")
    print("\nreproduces N7 base_I and N8 pool_w0")
    tl, third = d3.index[d3.category == "transfer_limited"], d2.csp_lda.sort_values().index[: N_SUB // 3]
    for k in SIZES:
        g = df[f"K{k}"] - df.base_I
        print(f"K = {k:2d}: R11 {df[f'K{k}'].mean():.2f}, gain {g.mean():+.2f} pp ({(g > 0).sum()} up/{(g < 0).sum()} down), "
              f"p = {sign_flip_p(g.to_numpy()):.4f}; transfer_limited {g[tl].mean():+.2f}, bottom third {g[third].mean():+.2f}; "
              f"draw sd {df[f'K{k}_sd'].mean():.2f}")
    d = (df.K53 - df.K8)
    p = sign_flip_p(d.to_numpy())
    scales = d.mean() >= MIN_DIFF_PP and p < ALPHA
    print(f"\ngain(53) - gain(8) = {d.mean():+.2f} pp ({(d > 0).sum()} up/{(d < 0).sum()} down), p = {p:.4f}")
    print("VERDICT:", "SCALES" if scales else "FLAT")
    df.to_csv(RESULTS_DIR / "n10_library_scaling_lee2019.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
