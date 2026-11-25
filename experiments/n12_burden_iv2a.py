"""N12, does the N11 calibration-burden result hold on IV-2a? PRE-REGISTERED.

Committed BEFORE any arm is scored. N11 (Lee2019, TS+LR member alone): zero-shot pooled 75.26 % against own-data 70.72 % at 100 trials,
and pool + 5 own trials already matches. N9 found no pooled gain on IV-2a for R11, so the claim must be checked there before it enters the
plan. Same protocol as N11 on IV-2a (9 subjects, 4 classes, 22 channels, 288 trials per session), TS+LR member alone, library = the 8 other
subjects (both labelled sessions), features as N7 (per-session alignment, OAS, tangent space at I), LR C = 1, pool weight w = 20, 3 draws.
n_own in {8, 20, 40, 80, 160, 288}: stratified own trials (n // 4 per class, the remainder given to the lowest classes); n_own = 288 is the
full session. Descriptive only: 9 subjects, no p values; per-subject up/down counts.
CONSISTENT iff (a) pool0 >= own(288) - 0.5 pp AND (b) poolw reaches own(288) - 0.5 pp at some n_pool <= 144 with poolw > own for at least 7 of
9 subjects at that n_pool. Then the burden claim is stated for both cohorts; otherwise it is stated for Lee2019 only (100 trials per
session, binary) and the plan note says so. IV-2b is not tested (3 channels, a different recording design).

    .venv\Scripts\python experiments\n12_burden_iv2a.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from experiments.n8_pooled_decoder_screen import pool_proba  # noqa: E402
from experiments.n9_ens4_controls import iv2a_set  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402

SIZES, N_DRAWS, W, TOL_PP, MAX_N_POOL, MIN_UP = (8, 20, 40, 80, 160, 288), 3, 20, 0.5, 144, 7


def subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per = [n // 4 + (1 if c < n % 4 else 0) for c in range(4)]
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), k, replace=False) for c, k in enumerate(per)]))


def ts_acc(p: np.ndarray, v: dict) -> float:
    return 100 * float((p.argmax(1) == v["y_te"]).mean())


def main() -> int:
    r11 = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv")
    S = iv2a_set(r11[(r11.dataset == "IV-2a") & (r11.arm == "fts")].set_index("subject").acc)
    rows = {}
    for s, v in S.items():
        rows[s] = {"pool0": ts_acc(pool_proba(S, {s}, None, 0, v), v)}
        for n in SIZES:
            own, pw = [], []
            for d in range(1 if n == 288 else N_DRAWS):
                idx = np.arange(len(v["y_tr"])) if n == 288 else subsample(v["y_tr"], n, 1000 * s + d)
                sub = {"F_tr": v["F_tr"][idx], "y_tr": v["y_tr"][idx]}
                own.append(ts_acc(LogisticRegression(C=1.0, max_iter=5000).fit(sub["F_tr"], sub["y_tr"]).predict_proba(v["F_te"]), v))
                pw.append(ts_acc(pool_proba(S, {s}, sub, W, v), v))
            rows[s][f"own{n}"], rows[s][f"poolw{n}"] = float(np.mean(own)), float(np.mean(pw))
        print(f"s{s}: pool0 {rows[s]['pool0']:.1f}, own288 {rows[s]['own288']:.1f}, poolw8 {rows[s]['poolw8']:.1f}", flush=True)
    df = pd.DataFrame(rows).T
    for n in SIZES:
        d = df[f"poolw{n}"] - df[f"own{n}"]
        print(f"n_own {n:3d}: own {df[f'own{n}'].mean():.2f}, pool0 {df.pool0.mean():.2f}, poolw {df[f'poolw{n}'].mean():.2f} "
              f"(poolw-own {d.mean():+.2f} pp, {(d > 0).sum()} up/{(d < 0).sum()} down)")
    ref = df.own288.mean()
    n_pool = next((n for n in SIZES if df[f"poolw{n}"].mean() >= ref - TOL_PP), None)
    up = int(((df[f"poolw{n_pool}"] - df[f"own{n_pool}"]) > 0).sum()) if n_pool else 0
    a, b = df.pool0.mean() >= ref - TOL_PP, bool(n_pool and n_pool <= MAX_N_POOL and up >= MIN_UP)
    print(f"\nown(288) {ref:.2f}; pool0 {df.pool0.mean():.2f}; n_pool = {n_pool}, up at n_pool {up}/9")
    print(f"(a) zero-shot viable: {a}; (b) burden reduced: {b}; CONSISTENT: {a and b}")
    df.to_csv(RESULTS_DIR / "n12_burden_iv2a.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
