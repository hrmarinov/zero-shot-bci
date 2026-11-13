"""C1: causal (online) alignment for R11 vs the transductive R11. PRE-REGISTERED.

Committed BEFORE the first real run (development or lockbox). Implementation bugs found later are fixed in a
separate commit that leaves this protocol unchanged.

QUESTION. R11 aligns each test session with the Riemannian mean of ALL its unlabelled trials, computed up
front (transductive), so it cannot run live. Does a causal reference, updated trial by trial, keep R11's
accuracy?

CAUSAL ALIGNMENT (src.pipeline.align_causal, all settings fixed a priori):
  training trials aligned exactly as R11 (per group, own full mean: the training session is complete before
  the test session starts), so every trained model is R11's own model;
  each test group (IV-2a: the evaluation session; IV-2b: each test recording; BNCI2015_001: session B) is
  processed in recording order; trial i is whitened by R_i, the recursive Riemannian mean
  R_i = geodesic(R_{i-1}, C_i, 1 / (w + i)), started at R_0 = Riemannian mean of all training-trial OAS
  covariances counted as w pseudo-trials. R_i uses trial i itself (its unlabelled covariance is available
  when it must be classified) and earlier trials only; never a label, never a later trial.
ARMS (same trained members, same weights (CNN3 + TSLR + FTS2) / 3; only the test-trial alignment differs):
  trans   R11 as cached (full test-session mean).                   <- reference
  causal  w = 10 (src.pipeline.CAUSAL_PRIOR_WEIGHT).                <- the arm under test
  static  w = inf: the training reference only, never updated (no test-session adaptation at all).
  cold    w = 0: no prior; the first trial is whitened by its own covariance (secondary, descriptive).
Applied identically to the CNN input (4-38 Hz, 0-4 s) and the TS+LR input (8-30 Hz, 0.5-2.5 s).

HYPOTHESES (development: IV-2a + IV-2b, 18 subjects pooled; score = accuracy, IV-2b averaged over its test
recordings as R11):
  H1 (primary)  causal is NON-INFERIOR to trans: one-sided 95% bootstrap lower bound (10 000 resamples over
                subjects, seed 0) of causal - trans above -1.0 pp.
  H2            causal - static > 0: one-sided exact sign-flip p < 0.05 (online adaptation beats none).
REPORTED (no gate): cold - causal; accuracy over the first N = 10, 20, 40, 80 trials of each test group for
every arm (the warm-up cost); per-dataset means; up/down counts.
LOCKBOX (once, `--lockbox`, BNCI2015_001 via L1's data path; members retrained, checked against L1's cache):
  the same H1 and H2 over 12 subjects. Run once, whatever the development result.

VALIDITY: V1 trial order (loaders) and training rows of every arm's input identical to R11's;
V2 every retrained member's trans prediction equals its cached R11 prediction (identical argmax,
max |dp| < 1e-5; IV-2a/2b: r3/r7/r8/r9 caches, lockbox: data/cache/l1_preds), so all arms share R11's models;
V3 TS+LR's trans prediction equals src.pipeline.tslr_member exactly; V4 (dev only) trans reproduces
results/r11_round3_diverse_members.csv (arm 'fts') exactly.

    .venv\\Scripts\\python experiments\\c1_causal_alignment.py             # development (~50 min GPU, resumable)
    .venv\\Scripts\\python experiments\\c1_causal_alignment.py --lockbox   # BNCI2015_001, once (~30 min GPU)
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from experiments.check_pipeline_reproduces_r11 import REFERENCE, TOL_PROB  # noqa: E402
from experiments.r11_round3_diverse_members import CACHE, rec_acc  # noqa: E402
from src.config import FMAX, FMIN, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.pipeline import (CAUSAL_PRIOR_WEIGHT, CNN_BAND, R11, SubjectData, align_by_group, align_causal,  # noqa: E402
                          cnn_member, load_subject_data, tslr_member)

ARMS = {"causal": CAUSAL_PRIOR_WEIGHT, "static": np.inf, "cold": 0.0}
FIRST_N = (10, 20, 40, 80)
MEMBERS = [("cnn", k, None) for k in range(R11.n_cnn)] + [("fts", k, "ftsurrogate") for k in range(R11.n_fts)]
NONINF_MARGIN_PP = -1.0
N_BOOT = 10_000
C1_CACHE = CACHE / "c1_preds"


def inputs(data: SubjectData, X: np.ndarray) -> dict[str, np.ndarray]:
    """Every arm's aligned input; asserts the training rows are R11's in every arm (V1)."""
    out = {"trans": align_by_group(X, data.groups)}
    for arm, w in ARMS.items():
        out[arm] = align_causal(X, data.groups, data.train, w)
        if not np.array_equal(out[arm][data.train], out["trans"][data.train]):
            raise AssertionError(f"V1: {arm} changes the training rows")
    return out


def tslr_all(data: SubjectData, A: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cov = Covariances(estimator="oas")
    ts = TangentSpace(metric="riemann").fit(cov.fit_transform(A["trans"][data.train]))
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(ts.transform(cov.transform(A["trans"][data.train])),
                                                      data.y[data.train])
    out = {arm: lr.predict_proba(ts.transform(cov.transform(a[~data.train]))) for arm, a in A.items()}
    if not np.array_equal(out["trans"], tslr_member(data)):
        raise AssertionError("V3: TS+LR trans prediction differs from src.pipeline.tslr_member")
    return out


def members(data: SubjectData, A: dict[str, np.ndarray], n_classes: int, one_hot: bool, reference, cache: Path,
            device) -> dict[str, list[np.ndarray]]:
    """{arm: [CNN3 probs, FTS2 probs]} from the 5 R11 members, each trained once and applied to every arm."""
    cfg = dataclasses.replace(R11, one_hot_targets=one_hot)
    per_arm: dict[str, dict[str, list[np.ndarray]]] = {arm: {"cnn": [], "fts": []} for arm in A}
    for fam, k, aug in MEMBERS:
        paths = {arm: cache / f"{fam}_seed{k}_{arm}.npy" for arm in A}
        if all(p.exists() for p in paths.values()):
            probs = {arm: np.load(p) for arm, p in paths.items()}
        else:
            wait_until_healthy()
            got = cnn_member(data, A["trans"], n_classes, k, aug, cfg, device,
                             extra_tests={arm: a[~data.train] for arm, a in A.items() if arm != "trans"})
            probs = {"trans": got["test"], **{arm: got[arm] for arm in A if arm != "trans"}}
            cache.mkdir(parents=True, exist_ok=True)
            for arm, p in probs.items():
                np.save(paths[arm], p)
        ref = np.load(reference(fam, k))
        diff = float(np.abs(probs["trans"] - ref).max())
        if not (np.array_equal(probs["trans"].argmax(1), ref.argmax(1)) and diff < TOL_PROB):
            raise AssertionError(f"V2: {cache.name} {fam} seed{k} differs from its R11 cache (max diff {diff:.2e})")
        for arm in A:
            per_arm[arm][fam].append(probs[arm])
    return {arm: [np.mean(v["cnn"], 0), np.mean(v["fts"], 0)] for arm, v in per_arm.items()}


def first_n_acc(p: np.ndarray, y: np.ndarray, rec: np.ndarray, n: int) -> float:
    pr = p.argmax(1)
    return float(np.mean([(pr[rec == r][:n] == y[rec == r][:n]).mean() for r in np.unique(rec)]))


def score_subject(data: SubjectData, n_classes: int, one_hot: bool, reference, cache: Path, device) -> dict:
    y = data.y[~data.train]
    rec = data.groups[~data.train]
    A_cnn, A_ts = inputs(data, data.X_cnn), inputs(data, data.X_tslr)
    ts = tslr_all(data, A_ts)
    fam = members(data, A_cnn, n_classes, one_hot, reference, cache, device)
    row = {}
    for arm in A_cnn:
        ens = (fam[arm][0] + ts[arm] + fam[arm][1]) / 3
        row[arm] = rec_acc(ens, y, rec)
        for n in FIRST_N:
            row[f"{arm}_first{n}"] = first_n_acc(ens, y, rec, n)
    return row


def boot_lower(d: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    means = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return float(np.percentile(means, 5)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(df: pd.DataFrame, label: str) -> None:
    pp = df.set_index(["dataset", "subject"]) * 100
    n = len(pp)
    arms = ["trans", *ARMS]
    print(f"\n=== C1 {label}: causal vs transductive alignment, {n} subjects ===")
    print(pp[arms].groupby(level="dataset").mean().round(2).to_string())
    print("\nfirst-N accuracy (mean over subjects; each test group's first N trials):")
    tab = pd.DataFrame({arm: [pp[f"{arm}_first{k}"].mean() for k in FIRST_N] for arm in arms},
                       index=[f"first {k}" for k in FIRST_N])
    tab.loc["all"] = [pp[a].mean() for a in arms]
    print(tab.round(2).to_string())
    d = (pp.causal - pp.trans).to_numpy()
    lo90, lo, hi = boot_lower(d)
    h1 = lo90 > NONINF_MARGIN_PP
    print(f"\nH1 causal - trans: mean {d.mean():+.2f} pp [95% CI {lo:+.2f}, {hi:+.2f}], {(d > 0).sum()} up / "
          f"{(d < 0).sum()} down; one-sided 95% lower bound {lo90:+.2f} vs margin {NONINF_MARGIN_PP:+.1f} -> "
          f"{'NON-INFERIOR' if h1 else 'not shown'}")
    d2 = (pp.causal - pp.static).to_numpy()
    p2 = sign_flip_permutation_test(d2, "greater")
    print(f"H2 causal - static: mean {d2.mean():+.2f} pp, {(d2 > 0).sum()} up / {(d2 < 0).sum()} down, "
          f"p = {p2:.4f} -> {'SUPPORTED' if p2 < 0.05 and d2.mean() > 0 else 'not supported'}")
    d3 = (pp.cold - pp.causal).to_numpy()
    print(f"(reported) cold - causal: mean {d3.mean():+.2f} pp, {(d3 > 0).sum()} up / {(d3 < 0).sum()} down")
    for ds, g in pp.groupby(level="dataset"):
        print(f"  {ds}: causal - trans {(g.causal - g.trans).mean():+.2f}, causal - static {(g.causal - g.static).mean():+.2f}")


def run_dev(device) -> None:
    ref = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv").set_index(["dataset", "subject", "arm"]).acc
    rows = []
    for s in IV2A_SUBJECTS:
        for ds in ("IV-2a", "IV-2b"):
            data, n_classes = load_subject_data(ds, s)
            reference = lambda fam, k, ds=ds, s=s: REFERENCE[ds][fam](s, k)  # noqa: E731
            row = score_subject(data, n_classes, ds == "IV-2b", reference, C1_CACHE / f"{ds}_s{s}", device)
            if not abs(row["trans"] - ref[(ds, s, "fts")]) < 1e-12:
                raise AssertionError(f"V4: {ds} s{s} trans {row['trans']:.6f} != R11 {ref[(ds, s, 'fts')]:.6f}")
            rows.append({"dataset": ds, "subject": s, **row})
            print(f"  {ds} s{s}: trans {100 * row['trans']:.2f} causal {100 * row['causal']:.2f} "
                  f"static {100 * row['static']:.2f} cold {100 * row['cold']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "c1_causal_alignment.csv"
    df.to_csv(out, index=False)
    summarise(df, "development (IV-2a + IV-2b)")
    print(f"\nwrote {out}")


def run_lockbox(device) -> None:
    from moabb.datasets import BNCI2015_001

    from experiments.l1_lockbox_bnci2015_001 import load
    dataset = BNCI2015_001()
    ts_band = dict(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
    rows = []
    for s in dataset.subject_list:
        data = load(dataset, s, dict(CNN_BAND), ts_band)
        reference = lambda fam, k, s=s: CACHE / "l1_preds" / f"s{s}" / f"{fam}_seed{k}.npy"  # noqa: E731
        row = score_subject(data, 2, False, reference, C1_CACHE / f"lockbox_s{s}", device)
        rows.append({"dataset": "BNCI2015_001", "subject": s, **row})
        print(f"  lockbox s{s}: trans {100 * row['trans']:.2f} causal {100 * row['causal']:.2f} "
              f"static {100 * row['static']:.2f} cold {100 * row['cold']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "c1_causal_alignment_lockbox.csv"
    df.to_csv(out, index=False)
    summarise(df, "LOCKBOX (BNCI2015_001)")
    print(f"\nwrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lockbox", action="store_true", help="the single BNCI2015_001 run")
    args = ap.parse_args()
    device = torch.device("cuda")
    run_lockbox(device) if args.lockbox else run_dev(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
