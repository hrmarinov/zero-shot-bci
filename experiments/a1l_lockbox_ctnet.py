"""A1L, LOCKBOX confirmation of A1's winner on BNCI2015_001. PRE-REGISTERED.

Committed BEFORE any CTNet fit on BNCI2015_001. The dataset is opened once for this question; no reruns, no
changes of protocol after the data are seen. Bugs found later are fixed in a separate commit that leaves the
protocol unchanged.

A1 (docs/diag_gates_d1_d4.md) passed one arm: aug_ctn = (CNN3 + TSLR + FTS2 + CTN2) / 4, i.e. R11 plus a
fourth equal-weight family of two CTNet seeds (pooled +0.89 pp over R11 on IV-2a + IV-2b, 14/18 up, Holm
p = 0.045). This script tests it once on the untouched lockbox.

DATA AND MEMBERS: L1's data path (experiments/l1_lockbox_bnci2015_001.load: session A -> B, 250 Hz, both
bands, trial-order check). CNN3 = base seeds 0-2, CNNx = base seeds 3-4, FTS2 = FTSurrogate seeds 0-1, all
from L1's cache (data/cache/l1_preds); TSLR recomputed (src.pipeline.tslr_member). CTN2 = braindecode CTNet
seeds 0-1 trained exactly as in A1 (experiments/a1_arch_member_screen.arch_member: per-session RA input,
4-38 Hz, 0-4 s, R3 CFG, 300 epochs, integer targets, built after set_deterministic(seed)); cached in
data/cache/a1l_preds.

HYPOTHESES (12 subjects, score = accuracy on session B):
  H1 (primary)  aug_ctn - r11 > 0, one-sided exact sign-flip p < 0.05. The winner CONFIRMS iff H1 holds.
  H2            aug_ctn - ctrl > 0 (ctrl = (CNN3 + TSLR + FTS2 + CNNx) / 4, size-matched), same test;
                reported, no gate (it decides whether the gain is architecture diversity or member count).
Reported: mean, median, up/down, 95% bootstrap CI (10 000 resamples, seed 0) for both; CTNet alone vs a
single ShallowConvNet (seed mean).
DECISION: H1 confirmed -> `aug_ctn` becomes the recommended pipeline (R15) and is added to src/pipeline.py as
a named configuration; R11 stays the default until then. Not confirmed -> R11 stays; documented stop.

VALIDITY: V1 r11 recomputed from L1's cache reproduces results/l1_lockbox_bnci2015_001.csv exactly (L1's own
ctrl had no FTS family, so it is not comparable);
V2 every member has one row per session-B trial.

    .venv\\Scripts\\python experiments\\a1l_lockbox_ctnet.py      # THE run, once (~20 min GPU)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiments.a1_arch_member_screen import ARCH_SEEDS, arch_member  # noqa: E402
from experiments.l1_lockbox_bnci2015_001 import load  # noqa: E402
from src.config import DATA_DIR, FMAX, FMIN, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.pipeline import CNN_BAND, CNN_TRAIN, align_by_group, tslr_member  # noqa: E402

L1_CACHE = DATA_DIR / "cache" / "l1_preds"
CACHE = DATA_DIR / "cache" / "a1l_preds"
OUT_CSV = RESULTS_DIR / "a1l_lockbox_ctnet.csv"
N_BOOT = 10_000


def acc(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(1) == y).mean())


def score_subject(dataset, s: int, l1: pd.DataFrame, device) -> dict:
    data = load(dataset, s, dict(CNN_BAND), dict(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX))
    y = data.y[~data.train]
    base = [np.load(L1_CACHE / f"s{s}" / f"cnn_seed{k}.npy") for k in range(5)]
    fts = [np.load(L1_CACHE / f"s{s}" / f"fts_seed{k}.npy") for k in range(2)]
    ctn = []
    A = None
    for k in ARCH_SEEDS:
        path = CACHE / f"s{s}" / f"ctn_seed{k}.npy"
        if not path.exists():
            if A is None:
                A = align_by_group(data.X_cnn, data.groups)
            wait_until_healthy()
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, arch_member("ctn", data, A, 2, k, False, CNN_TRAIN, device))
        ctn.append(np.load(path))
    for p in base + fts + ctn:
        if p.shape[0] != len(y):
            raise AssertionError(f"V2: s{s}: a member has {p.shape[0]} rows, expected {len(y)}")
    ts = tslr_member(data)
    cnn3, cnnx, fts2, ctn2 = np.mean(base[:3], 0), np.mean(base[3:], 0), np.mean(fts, 0), np.mean(ctn, 0)
    row = {"subject": s, "r11": acc((cnn3 + ts + fts2) / 3, y), "ctrl": acc((cnn3 + ts + fts2 + cnnx) / 4, y),
           "aug_ctn": acc((cnn3 + ts + fts2 + ctn2) / 4, y),
           "cnn1": float(np.mean([acc(p, y) for p in base])), "ctn1": float(np.mean([acc(p, y) for p in ctn]))}
    ref = l1.loc[s]
    if not abs(row["r11"] - ref.r11) < 1e-12:
        raise AssertionError(f"V1: s{s} r11 {row['r11']:.4f} != L1 {ref.r11:.4f}")
    # L1's ctrl was (CNN3 + TSLR + CNNx) / 3; ours adds FTS2, so only r11 is checked against L1.
    print(f"  s{s}: r11 {100 * row['r11']:.2f} aug_ctn {100 * row['aug_ctn']:.2f} ctrl {100 * row['ctrl']:.2f}",
          flush=True)
    return row


def report(name: str, d: np.ndarray) -> float:
    rng = np.random.default_rng(0)
    means = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    p = sign_flip_permutation_test(d, "greater")
    print(f"  {name}: mean {d.mean():+.2f} pp [95% CI {np.percentile(means, 2.5):+.2f}, "
          f"{np.percentile(means, 97.5):+.2f}], median {np.median(d):+.2f}, {(d > 0).sum()}/{len(d)} up, "
          f"{(d < 0).sum()} down, p = {p:.4f}")
    return p


def main() -> int:
    from moabb.datasets import BNCI2015_001
    dataset = BNCI2015_001()
    l1 = pd.read_csv(RESULTS_DIR / "l1_lockbox_bnci2015_001.csv").set_index("subject")
    device = torch.device("cuda")
    df = pd.DataFrame([score_subject(dataset, s, l1, device) for s in dataset.subject_list])
    df.to_csv(OUT_CSV, index=False)
    pp = 100 * df.set_index("subject")
    print("\n=== A1L lockbox: R11 + CTNet family, BNCI2015_001, 12 subjects ===")
    print(pp.mean().round(2).to_string())
    p1 = report("H1 aug_ctn - r11", (pp.aug_ctn - pp.r11).to_numpy())
    report("H2 aug_ctn - ctrl", (pp.aug_ctn - pp.ctrl).to_numpy())
    report("CTNet alone - ShallowConvNet alone", (pp.ctn1 - pp.cnn1).to_numpy())
    ok = p1 < 0.05 and (pp.aug_ctn - pp.r11).mean() > 0
    print(f"\nH1: {'CONFIRMED - aug_ctn replicates on the lockbox' if ok else 'NOT CONFIRMED - R11 stays'}")
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
