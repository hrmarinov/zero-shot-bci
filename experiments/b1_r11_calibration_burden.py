"""B1, calibration-burden curve for the R11 pipeline (IV-2a + IV-2b, 18 subjects). PRE-REGISTERED.

Why (TODO Part E #7, reports/r14_council_and_ideation.md §4): the project's goal is less calibration, yet
every R-number is measured with the full training session. Riemannian alignment and TS+LR need few
trials; CNNs may not. Does the ensemble move the point at which a user reaches a given accuracy?

DESIGN (A29 conventions, experiments/ablation_01_calibration_burden.py):
  budgets n in {48, 96, 144} labelled training trials, plus FULL (all training trials: 288 on IV-2a,
  ~400 on IV-2b). For n < FULL: 3 draws (seeds 0-2), stratified by class (n / n_classes per class),
  drawn from the whole training pool (IV-2b: all three training recordings). Alignment of the training
  side is fitted on the drawn trials only (per recording on IV-2b); the test session is aligned on its
  own unlabelled trials, as everywhere else. Test = the full evaluation session. Pipeline = src.pipeline,
  unchanged (R11 config; IV-2b one-hot targets), so FULL reuses the reproduction-check caches.
ARMS at every budget: cnn1 (mean single-model accuracy over base seeds 0-2), tslr, csp_lda (CSP+LDA+RA),
  r10, r11. Subject score per budget = mean over the draws; IV-2b = mean over its two test recordings.

PRIMARY (H1, parity): R11 trained on 96 trials is non-inferior to a single CNN trained on the FULL
  session: one-sided 95% bootstrap lower bound (10 000 resamples over the 18 subjects, seed 0) of
  r11@96 - cnn1@FULL above -1.0 pp.
SECONDARY (reported): the same parity test at 48 and 144; r11 - cnn1 and r11 - csp_lda at every budget
  (sign-flip p); the smallest budget at which each arm reaches cnn1@FULL - 1 pp on the pooled mean.

CHECKS: FULL reproduces R11 per subject exactly (< 1e-12 vs results/r11_round3_diverse_members.csv) and
  cnn1@FULL equals the mean of the cached single models; every draw is stratified and seeded; draws of
  different budgets are independent.

    .venv\\Scripts\\python experiments\\b1_r11_calibration_burden.py     # ~1.5 h GPU (810 small fits), resumable
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiments.r11_round3_diverse_members import CACHE, rec_acc  # noqa: E402
from src.baselines import make_csp_lda  # noqa: E402
from src.config import IV2A_SUBJECTS, RESULTS_DIR  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.pipeline import R11, SubjectData, align_by_group, fit_predict, load_subject_data  # noqa: E402

mne.set_log_level("ERROR")
OUT_CSV = RESULTS_DIR / "b1_r11_calibration_burden.csv"
BUDGETS = (48, 96, 144)
DRAWS = (0, 1, 2)
PARITY_MARGIN_PP = -1.0
N_BOOT = 10_000
ARMS = ("cnn1", "tslr", "csp_lda", "r10", "r11")


def stratified_draw(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    classes = np.unique(y)
    if n % len(classes):
        raise ValueError(f"budget {n} not divisible by {len(classes)} classes")
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), n // len(classes), replace=False)
                                   for c in classes]))


def subset(data: SubjectData, keep_train: np.ndarray) -> SubjectData:
    """Keep the chosen training trials (indices into the training pool) and every test trial."""
    tr = np.flatnonzero(data.train)[keep_train]
    idx = np.sort(np.concatenate([tr, np.flatnonzero(~data.train)]))
    return SubjectData(X_cnn=data.X_cnn[idx], X_tslr=data.X_tslr[idx], y=data.y[idx],
                       groups=data.groups[idx], train=data.train[idx])


def score(ds: str, data: SubjectData, n_classes: int, cache: Path, device) -> dict:
    cfg = dataclasses.replace(R11, one_hot_targets=(ds == "IV-2b"))
    res = fit_predict(data, n_classes, cfg, device, cache, before_fit=wait_until_healthy)
    y, rec = data.y[~data.train], (data.groups[~data.train] if ds == "IV-2b" else np.zeros((~data.train).sum()))
    singles = [np.load(cache / f"cnn_seed{k}.npy") for k in range(R11.n_cnn)]
    A = align_by_group(data.X_tslr, data.groups)
    csp = make_csp_lda().fit(A[data.train], data.y[data.train]).predict(A[~data.train])
    one_hot = np.eye(n_classes)[csp]
    return {"cnn1": float(np.mean([rec_acc(p, y, rec) for p in singles])), "tslr": rec_acc(res["tslr"], y, rec),
            "csp_lda": rec_acc(one_hot, y, rec), "r10": rec_acc((res["cnn"] + res["tslr"]) / 2, y, rec),
            "r11": rec_acc(res["ensemble"], y, rec)}


def run(ds: str, s: int, ref: pd.Series, device) -> list[dict]:
    data, n_classes = load_subject_data(ds, s)
    rows = []
    full = score(ds, data, n_classes, CACHE / "pipeline_preds" / f"{ds}_s{s}", device)
    if not abs(full["r11"] - ref[(ds, s, "fts")]) < 1e-12:
        raise AssertionError(f"{ds} s{s}: FULL does not reproduce R11")
    rows.append({"dataset": ds, "subject": s, "budget": "FULL", "n_train": int(data.train.sum()), "draw": 0, **full})
    y_pool = data.y[data.train]
    for n in BUDGETS:
        for d in DRAWS:
            keep = stratified_draw(y_pool, n, seed=1000 * n + 100 * d + s)
            sub = subset(data, keep)
            r = score(ds, sub, n_classes, CACHE / "b1_preds" / f"{ds}_s{s}_n{n}_d{d}", device)
            rows.append({"dataset": ds, "subject": s, "budget": str(n), "n_train": n, "draw": d, **r})
    return rows


def summarise(df: pd.DataFrame) -> None:
    subj = df.groupby(["dataset", "subject", "budget"])[list(ARMS)].mean()
    order = [str(n) for n in BUDGETS] + ["FULL"]
    print("\n=== B1 R11 calibration burden (pooled 18 subjects; per-dataset below) ===")
    print((100 * subj.groupby("budget").mean().loc[order]).round(2).to_string())
    print((100 * subj.groupby(["dataset", "budget"]).mean()).round(2).to_string())
    full_cnn = 100 * subj.xs("FULL", level="budget")["cnn1"]
    rng = np.random.default_rng(0)
    print(f"\nPARITY: r11@n - cnn1@FULL (non-inferior if the one-sided 95% lower bound > {PARITY_MARGIN_PP:+.1f} pp)")
    for n in BUDGETS:
        d = (100 * subj.xs(str(n), level="budget")["r11"] - full_cnn).to_numpy()
        lo = np.percentile(d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1), 5)
        tag = "PRIMARY H1" if n == 96 else "secondary"
        print(f"  [{tag}] n = {n}: mean {d.mean():+.2f} pp, lower bound {lo:+.2f}, {(d > 0).sum()}/18 up "
              f"-> {'NON-INFERIOR' if lo > PARITY_MARGIN_PP else 'not shown'}")
    print("\nGAINS at each budget (one-sided sign-flip):")
    for b in order:
        x = 100 * subj.xs(b, level="budget")
        for base in ("cnn1", "csp_lda"):
            dd = (x["r11"] - x[base]).to_numpy()
            print(f"  n = {b:>4}: r11 - {base:7s} {dd.mean():+.2f} pp, {(dd > 0).sum()}/18 up, "
                  f"p = {sign_flip_permutation_test(dd, 'greater'):.4f}")
    target = full_cnn.mean() - 1.0
    print(f"\nSmallest budget reaching cnn1@FULL - 1 pp ({target:.2f}%) on the pooled mean:")
    pooled = 100 * subj.groupby("budget").mean().loc[order]
    for arm in ARMS:
        hit = [b for b in order if pooled.loc[b, arm] >= target]
        print(f"  {arm:8s}: {hit[0] if hit else 'never'}")


def main() -> int:
    ref = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv").set_index(["dataset", "subject", "arm"]).acc
    device = torch.device("cuda")
    rows: list[dict] = []
    for s in IV2A_SUBJECTS:
        for ds in ("IV-2a", "IV-2b"):
            rows += run(ds, s, ref, device)
        print(f"  s{s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("CHECK passed: FULL reproduces R11 for all 18 subjects")
    summarise(df)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
