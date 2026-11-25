"""N11, calibration burden with a population prior: how few own trials does the TS+LR member need? PRE-REGISTERED.

Committed BEFORE any arm is scored. N10 showed that a zero-shot pooled decoder (no own labels) beats R11's own TS+LR member on
Lee2019 even with 8 library subjects, and that the gain does not grow with library size, which suggests the prior helps when the
subject's own data is scarce (100 trials per session here). N11 measures the TS+LR MEMBER ALONE (not R11: the cached CNN members
are fitted on all 100 trials and cannot be subsampled without new fits) on session 2 of Lee2019, n = 54, leave-one-subject-out,
features as N7 (per-session alignment, OAS, tangent space at I; the held-out session-2 trials are aligned on themselves as always).

ARMS at n_own in {5, 10, 20, 40, 70, 100} own labelled session-1 trials (stratified random subsample of the subject's session 1,
3 draws per (subject, n_own), seeds 1000 * s + draw; n_own = 100 is the full session and uses one draw):
  own    LR (C = 1) on the subject's n_own trials only (needs two classes in the draw; guaranteed by stratification).
  pool0  zero-shot: LR on the 53 other subjects' trials (both sessions), constant in n_own.
  poolw  the pool plus the n_own own trials at sample weight w = 20 (fixed here, not tuned; the middle of N8's grid).
Accuracy per subject = mean over draws. own at n_own = 100 must equal results/n7_drift_forecast_lee2019.csv column ts_base_I.
ENDPOINTS. For each n_own: mean accuracy of the three arms; poolw - own and pool0 - own with one-sided sign-flip p (as D2); Holm over
the 5 sample sizes below 100 for poolw - own. DERIVED: n_own* = smallest n_own at which own's mean accuracy reaches pool0's mean
accuracy (the burden a subject-only decoder needs to match a calibration-free one); n_pool = smallest n_own at which poolw's mean
accuracy reaches own's accuracy at n_own = 100 minus 0.5 pp, and the burden reduction 100 / n_pool.
DECISION. CALIBRATION-FREE VIABLE iff pool0 >= own(100) - 0.5 pp. BURDEN CLAIM iff n_pool <= 50 (a reduction of at least 2x) AND poolw
- own at n_own = n_pool has Holm p < 0.05. If the claim holds, README "Next" and TODO get a short note (the plan changes: the
deliverable becomes calibration burden). Otherwise the line is closed and nothing is added to the plan. TS+LR member only; a claim
about the full R11 pipeline would need new CNN fits and is not made here. IV-2a is not tested (N9: no library effect there).

    .venv\Scripts\python experiments\n11_calibration_burden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from experiments.d2_lee2019_dev import sign_flip_p  # noqa: E402
from experiments.n7_drift_forecast_screen import N_SUB, subject_features  # noqa: E402
from experiments.n8_pooled_decoder_screen import pool_proba  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.evaluate import holm_adjust  # noqa: E402

SIZES, N_DRAWS, W, TOL_PP, ALPHA, MAX_N_POOL = (5, 10, 20, 40, 70, 100), 3, 20, 0.5, 0.05, 50


def subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Stratified own trials: n // 2 of class 0 and the rest of class 1."""
    rng = np.random.default_rng(seed)
    take = {0: n // 2, 1: n - n // 2}
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), k, replace=False) for c, k in take.items()]))


def ts_acc(p: np.ndarray, v: dict) -> float:
    return 100 * float((p.argmax(1) == v["y_te"]).mean())


def main() -> int:
    n7 = pd.read_csv(RESULTS_DIR / "n7_drift_forecast_lee2019.csv", index_col=0)
    S = {s: subject_features(s) for s in range(1, N_SUB + 1)}
    rows = {}
    for s, v in S.items():
        rows[s] = {"pool0": ts_acc(pool_proba(S, {s}, None, 0, v), v)}
        for n in SIZES:
            draws = 1 if n == 100 else N_DRAWS
            own, pw = [], []
            for d in range(draws):
                idx = np.arange(len(v["y_tr"])) if n == 100 else subsample(v["y_tr"], n, 1000 * s + d)
                sub = {"F_tr": v["F_tr"][idx], "y_tr": v["y_tr"][idx]}
                own.append(ts_acc(LogisticRegression(C=1.0, max_iter=5000).fit(sub["F_tr"], sub["y_tr"]).predict_proba(v["F_te"]), v))
                pw.append(ts_acc(pool_proba(S, {s}, sub, W, v), v))
            rows[s][f"own{n}"], rows[s][f"poolw{n}"] = float(np.mean(own)), float(np.mean(pw))
        print(f"s{s}: pool0 {rows[s]['pool0']:.0f}, own5 {rows[s]['own5']:.0f}, poolw5 {rows[s]['poolw5']:.0f}, own100 {rows[s]['own100']:.0f}", flush=True)
    df = pd.DataFrame(rows).T
    if (df.own100 - n7.ts_base_I).abs().max() > 1e-9:
        raise AssertionError("own at n_own = 100 does not reproduce N7 ts_base_I")
    print("\nreproduces N7 ts_base_I (own, n = 100)")
    ct = []
    for n in SIZES:
        for a in ("poolw", "pool0"):
            col = df[f"{a}{n}"] if a == "poolw" else df.pool0
            d = col - df[f"own{n}"]
            ct.append({"n_own": n, "contrast": f"{a}-own", "mean_pp": d.mean(), "up": int((d > 0).sum()), "down": int((d < 0).sum()),
                       "p": sign_flip_p(d.to_numpy())})
    ct = pd.DataFrame(ct)
    m = (ct.contrast == "poolw-own") & (ct.n_own < 100)
    ct.loc[m, "holm_p"] = holm_adjust(ct.loc[m, "p"].tolist())
    for n in SIZES:
        print(f"n_own {n:3d}: own {df[f'own{n}'].mean():.2f}, pool0 {df.pool0.mean():.2f}, poolw {df[f'poolw{n}'].mean():.2f}")
    print(ct.round(4).to_string(index=False))
    ref = df.own100.mean()
    n_star = next((n for n in SIZES if df[f"own{n}"].mean() >= df.pool0.mean()), None)
    n_pool = next((n for n in SIZES if df[f"poolw{n}"].mean() >= ref - TOL_PP), None)
    viable = df.pool0.mean() >= ref - TOL_PP
    holm_at = ct[(ct.n_own == n_pool) & (ct.contrast == "poolw-own")].holm_p.iloc[0] if n_pool and n_pool < 100 else np.nan
    claim = bool(viable and n_pool and n_pool <= MAX_N_POOL and holm_at < ALPHA)
    print(f"\nown(100) {ref:.2f}; pool0 {df.pool0.mean():.2f}; n_own* (own reaches pool0) = {n_star}; n_pool = {n_pool} "
          f"(reduction {100 / n_pool if n_pool else float('nan'):.1f}x), Holm p at n_pool {holm_at}")
    print(f"CALIBRATION-FREE VIABLE: {viable}; BURDEN CLAIM: {claim}")
    df.to_csv(RESULTS_DIR / "n11_calibration_burden_lee2019.csv")
    ct.to_csv(RESULTS_DIR / "n11_calibration_burden_contrasts.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
