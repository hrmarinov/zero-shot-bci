"""F1: the stacked fast-training path (PipelineConfig.fast) against the exact R11 path. PRE-REGISTERED.

Engineering check, no new science. Committed before the first real run. The default (exact) path is not
changed by this step; the fast path is an option.

PER SUBJECT (IV-2a s1-9 and IV-2b s1-9, data and settings exactly as src/pipeline.py's R11):
  exact  fit_predict(R11) into a FRESH cache (data/cache/f1_preds/<ds>_s<s>/), one ShallowConvNet after
         another. V1: every exact member equals its cached R3/R7/R8/R9 prediction (identical argmax,
         max |dp| < 1e-5), i.e. the refactored default path still reproduces R11 bit-for-bit within the
         project's tolerance, and the R11 accuracy equals results/r11_round3_diverse_members.csv (< 1e-12).
  fast   fit_predict(R11, fast=True): the same 5 members trained as one stacked model.
  Timing: wall time of the CNN fits per subject (torch.cuda.synchronize around each fit_predict; time spent
  in the GPU health guard is subtracted). TS+LR is identical in both and not counted.

REPORTED (no gate on the default; it stays exact):
  speedup = total exact time / total fast time;
  per member: argmax agreement with the exact member, max |dp|;
  per subject: accuracy delta fast - exact for each member, CNN3, FTS2 and the R11 ensemble;
  reference noise: per-subject |acc(base seed 0) - acc(base seed 1)| of the exact members (what a
  different seed changes), to put the fast-vs-exact deltas in scale.
DECISION RULE (fixed in advance): the fast path is RECOMMENDED for screens if the pooled 18-subject R11
mean delta lies within +-0.5 pp and the mean |R11 delta| is below the mean seed-to-seed |delta| of a
single member; otherwise it is kept as an option with the measured caveat. The default stays exact either way.

    .venv\\Scripts\\python experiments\\f1_fast_training_check.py      # ~1 h GPU, resumable
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiments.check_pipeline_reproduces_r11 import REFERENCE, TOL_ACC, TOL_PROB  # noqa: E402
from experiments.r11_round3_diverse_members import CACHE, rec_acc  # noqa: E402
from src.config import IV2A_SUBJECTS, RESULTS_DIR  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.pipeline import R11, fit_predict, load_subject_data  # noqa: E402

F1_CACHE = CACHE / "f1_preds"
OUT_CSV = RESULTS_DIR / "f1_fast_training_check.csv"
MEMBERS = [("cnn", k) for k in range(R11.n_cnn)] + [("fts", k) for k in range(R11.n_fts)]


class Guard:
    """wait_until_healthy, with the time it spends waiting recorded so it can be excluded from timings."""

    def __init__(self) -> None:
        self.waited = 0.0

    def __call__(self) -> None:
        t = time.perf_counter()
        wait_until_healthy()
        self.waited += time.perf_counter() - t


def timed_fit(data, n_classes, cfg, device, cache_dir: Path) -> tuple[dict, float]:
    guard = Guard()
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fit_predict(data, n_classes, cfg, device, cache_dir, before_fit=guard)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t - guard.waited


def run(ds: str, s: int, ref: pd.Series, device) -> dict:
    data, n_classes = load_subject_data(ds, s)
    rec = data.groups[~data.train] if ds == "IV-2b" else np.zeros((~data.train).sum())
    y = data.y[~data.train]
    cache = F1_CACHE / f"{ds}_s{s}"
    cfg = dataclasses.replace(R11, one_hot_targets=(ds == "IV-2b"))
    timing_file = cache / "timing.npy"
    already = timing_file.exists()
    exact, t_exact = timed_fit(data, n_classes, cfg, device, cache)
    fast, t_fast = timed_fit(data, n_classes, dataclasses.replace(cfg, fast=True), device, cache)
    if already:  # resumed: keep the timings measured on the first, uncached pass
        t_exact, t_fast = np.load(timing_file)
    else:
        np.save(timing_file, np.array([t_exact, t_fast]))
    row = {"dataset": ds, "subject": s, "t_exact_s": float(t_exact), "t_fast_s": float(t_fast)}
    for fam, k in MEMBERS:
        e, f = np.load(cache / f"{fam}_seed{k}.npy"), np.load(cache / f"{fam}_seed{k}_fast.npy")
        r = np.load(REFERENCE[ds][fam](s, k))
        diff = float(np.abs(e - r).max())
        if not (np.array_equal(e.argmax(1), r.argmax(1)) and diff < TOL_PROB):
            raise AssertionError(f"V1: {ds} s{s} exact {fam} seed{k} differs from its cache (max diff {diff:.2e})")
        row[f"{fam}{k}_agree"] = float((e.argmax(1) == f.argmax(1)).mean())
        row[f"{fam}{k}_maxdp"] = float(np.abs(e - f).max())
        row[f"{fam}{k}_exact"], row[f"{fam}{k}_fast"] = rec_acc(e, y, rec), rec_acc(f, y, rec)
    for name in ("cnn", "fts", "ensemble"):
        row[f"{name}_exact"], row[f"{name}_fast"] = rec_acc(exact[name], y, rec), rec_acc(fast[name], y, rec)
    if not abs(row["ensemble_exact"] - ref[(ds, s, "fts")]) < TOL_ACC:
        raise AssertionError(f"V1: {ds} s{s} exact R11 {row['ensemble_exact']:.6f} != {ref[(ds, s, 'fts')]:.6f}")
    return row


def summarise(df: pd.DataFrame) -> None:
    print("\n=== F1: fast (stacked) vs exact R11 training ===")
    for ds, g in df.groupby("dataset"):
        print(f"  {ds}: exact {g.t_exact_s.mean() / 60:.2f} min/subject, fast {g.t_fast_s.mean() / 60:.2f} "
              f"min/subject, speedup {g.t_exact_s.sum() / g.t_fast_s.sum():.2f}x")
    print(f"  overall speedup {df.t_exact_s.sum() / df.t_fast_s.sum():.2f}x")
    agree = df[[f"{f}{k}_agree" for f, k in MEMBERS]]
    maxdp = df[[f"{f}{k}_maxdp" for f, k in MEMBERS]]
    print(f"  member argmax agreement: mean {100 * agree.values.mean():.2f}%, min {100 * agree.values.min():.2f}%; "
          f"max |dp| median {np.median(maxdp.values):.3f}, max {maxdp.values.max():.3f}")
    print("\n  accuracy (pp), mean per dataset, fast - exact:")
    for name in ("cnn", "fts", "ensemble"):
        d = 100 * (df[f"{name}_fast"] - df[f"{name}_exact"])
        per = ", ".join(f"{ds} {d[df.dataset == ds].mean():+.2f}" for ds in sorted(df.dataset.unique()))
        print(f"    {name:8s}: pooled {d.mean():+.2f} (mean |d| {d.abs().mean():.2f}, max |d| {d.abs().max():.2f}, "
              f"{(d > 0).sum()} up / {(d < 0).sum()} down, two-sided p = "
              f"{sign_flip_permutation_test(d.to_numpy(), 'two-sided'):.3f}); {per}")
    member_d = pd.concat([100 * (df[f"{f}{k}_fast"] - df[f"{f}{k}_exact"]) for f, k in MEMBERS])
    seed_noise = (100 * (df["cnn0_exact"] - df["cnn1_exact"])).abs()
    print(f"    single member mean |d|: {member_d.abs().mean():.2f} pp; reference seed 0 vs seed 1 mean |d|: "
          f"{seed_noise.mean():.2f} pp")
    ens = 100 * (df.ensemble_fast - df.ensemble_exact)
    ok = abs(ens.mean()) <= 0.5 and ens.abs().mean() < seed_noise.mean()
    print(f"\n  R11 fast {100 * df.ensemble_fast.mean():.2f} vs exact {100 * df.ensemble_exact.mean():.2f} (pooled) -> "
          f"{'RECOMMENDED for screens' if ok else 'kept as an option only (caveat)'}; default stays exact")


def main() -> int:
    ref = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv").set_index(["dataset", "subject", "arm"]).acc
    device = torch.device("cuda")
    rows = []
    for s in IV2A_SUBJECTS:
        for ds in ("IV-2a", "IV-2b"):
            rows.append(run(ds, s, ref, device))
            r = rows[-1]
            print(f"  {ds} s{s}: exact {r['t_exact_s']:.0f}s fast {r['t_fast_s']:.0f}s, R11 "
                  f"{100 * r['ensemble_exact']:.2f} -> {100 * r['ensemble_fast']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summarise(df)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
