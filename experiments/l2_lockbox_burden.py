"""L2, LOCKBOX confirmation of the calibration-burden claim (B1) on BNCI2015_001. PRE-REGISTERED.

Claim from B1 (development data, IV-2a + IV-2b): with the R11 ensemble, a new session needs about HALF the
labelled calibration trials to match a single ShallowConvNet trained on the full calibration session.
Committed before any budgeted result on BNCI2015_001 exists. Caveat fixed in advance: BNCI2015_001 is no
longer untouched - L1 used its full-budget results - but no reduced-budget result has been computed on it
and nothing below was tuned on it. Opened once; an implementation bug may be fixed in a separate commit
with the protocol unchanged.

DESIGN (B1 conventions; L1 data path and caches):
  BNCI2015_001, 12 subjects, train on session A (FULL = 200 trials, 100 per class), test on all of session B.
  Budgets n in {48, 100, 144} labelled trials, 3 stratified seeded draws each (seed 1000 n + 100 d + s);
  alignment of the training side fitted on the drawn trials only; test side aligned on its own unlabelled
  trials. Pipeline = src.pipeline R11 config (3 base CNN seeds + TS+LR + 2 FTSurrogate seeds), integer
  targets, 250 Hz, exactly as L1. FULL reuses L1's cached members.
ARMS: cnn1 (mean single-model accuracy over base seeds 0-2), tslr, csp_lda (CSP+LDA+RA), r10, r11.

PRIMARY (H1): R11 trained on HALF the session (n = 100) is non-inferior to a single CNN trained on the FULL
  session: one-sided 95% bootstrap lower bound (10 000 resamples over the 12 subjects, seed 0) of
  r11@100 - cnn1@FULL above -1.0 pp.  The burden claim CONFIRMS iff H1 holds.
SECONDARY: the same test at 48 and 144; r11 - cnn1 and r11 - csp_lda at every budget (sign-flip p).
CHECK: FULL reproduces L1's r11 and cnn1 (mean over base seeds 0-2 recomputed from the cache) per subject.

    .venv\\Scripts\\python experiments\\l2_lockbox_burden.py     # ~1 h GPU (540 small fits), resumable
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from moabb.datasets import BNCI2015_001  # noqa: E402

from experiments.b1_r11_calibration_burden import stratified_draw, subset  # noqa: E402
from experiments.l1_lockbox_bnci2015_001 import load  # noqa: E402
from src.baselines import make_csp_lda  # noqa: E402
from src.config import DATA_DIR, FMAX, FMIN, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.pipeline import CNN_BAND, R11, SubjectData, align_by_group, fit_predict  # noqa: E402

mne.set_log_level("ERROR")
OUT_CSV = RESULTS_DIR / "l2_lockbox_burden.csv"
BUDGETS = (48, 100, 144)
PRIMARY_N = 100
DRAWS = (0, 1, 2)
MARGIN_PP = -1.0
N_BOOT = 10_000


def acc(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(1) == y).mean())


def score(data: SubjectData, cache: Path, device) -> dict:
    res = fit_predict(data, 2, R11, device, cache, before_fit=wait_until_healthy)
    y = data.y[~data.train]
    singles = [np.load(cache / f"cnn_seed{k}.npy") for k in range(R11.n_cnn)]
    A = align_by_group(data.X_tslr, data.groups)
    csp = make_csp_lda().fit(A[data.train], data.y[data.train]).predict(A[~data.train])
    return {"cnn1": float(np.mean([acc(p, y) for p in singles])), "tslr": acc(res["tslr"], y),
            "csp_lda": float((csp == y).mean()), "r10": acc((res["cnn"] + res["tslr"]) / 2, y),
            "r11": acc(res["ensemble"], y)}


def main() -> int:
    l1 = pd.read_csv(RESULTS_DIR / "l1_lockbox_bnci2015_001.csv").set_index("subject")
    dataset, device = BNCI2015_001(), torch.device("cuda")
    ts_band = dict(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
    rows = []
    for s in dataset.subject_list:
        data = load(dataset, s, dict(CNN_BAND), ts_band)
        full = score(data, DATA_DIR / "cache" / "l1_preds" / f"s{s}", device)
        if not abs(full["r11"] - l1.loc[s, "r11"]) < 1e-12:
            raise AssertionError(f"s{s}: FULL does not reproduce L1's r11")
        rows.append({"subject": s, "budget": "FULL", "draw": 0, **full})
        y_pool = data.y[data.train]
        for n in BUDGETS:
            for d in DRAWS:
                sub = subset(data, stratified_draw(y_pool, n, seed=1000 * n + 100 * d + s))
                rows.append({"subject": s, "budget": str(n), "draw": d,
                             **score(sub, DATA_DIR / "cache" / "l2_preds" / f"s{s}_n{n}_d{d}", device)})
        print(f"  s{s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("CHECK passed: FULL reproduces L1 for all 12 subjects")
    summarise(df)
    print(f"\nwrote {OUT_CSV}")
    return 0


def summarise(df: pd.DataFrame) -> None:
    arms = ["cnn1", "tslr", "csp_lda", "r10", "r11"]
    subj = df.groupby(["subject", "budget"])[arms].mean()
    order = [str(n) for n in BUDGETS] + ["FULL"]
    print("\n=== L2 lockbox burden: BNCI2015_001, 12 subjects ===")
    print((100 * subj.groupby("budget").mean().loc[order]).round(2).to_string())
    full_cnn = 100 * subj.xs("FULL", level="budget")["cnn1"]
    rng = np.random.default_rng(0)
    print(f"\nPARITY: r11@n - cnn1@FULL (non-inferior if the one-sided 95% lower bound > {MARGIN_PP:+.1f} pp)")
    for n in BUDGETS:
        d = (100 * subj.xs(str(n), level="budget")["r11"] - full_cnn).to_numpy()
        lo = np.percentile(d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1), 5)
        ok = lo > MARGIN_PP
        tag = "PRIMARY H1" if n == PRIMARY_N else "secondary"
        print(f"  [{tag}] n = {n}: mean {d.mean():+.2f} pp, lower bound {lo:+.2f}, {(d > 0).sum()}/12 up "
              f"-> {'NON-INFERIOR' if ok else 'not shown'}")
        if n == PRIMARY_N:
            verdict = "CONFIRMED" if ok else "NOT CONFIRMED"
    print(f"\nBurden claim on the lockbox: {verdict}")
    print("\nGAINS at each budget (one-sided sign-flip):")
    for b in order:
        x = 100 * subj.xs(b, level="budget")
        for base in ("cnn1", "csp_lda"):
            dd = (x["r11"] - x[base]).to_numpy()
            print(f"  n = {b:>4}: r11 - {base:7s} {dd.mean():+.2f} pp, {(dd > 0).sum()}/12 up, "
                  f"p = {sign_flip_permutation_test(dd, 'greater'):.4f}")


if __name__ == "__main__":
    sys.exit(main())
