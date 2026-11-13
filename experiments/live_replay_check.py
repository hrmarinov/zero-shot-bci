"""Live replay: src.live.LiveDecoder on the recorded test sessions, one trial at a time. Engineering check.

Per subject (IV-2a s1-9, IV-2b s1-9): calibrate R11 on the training session (exact path), then decode the
test session trial by trial in recording order (IV-2b: start_session() at each test recording).
  V1 the live ensemble probabilities equal C1's causal R11 rebuilt from its cached members
     (data/cache/c1_preds) + causal TS+LR (max |dp| < 1e-5), and the accuracy equals
     results/c1_causal_alignment.csv 'causal' (< 1e-12): the live decoder IS C1's causal R11.
Reported: calibration time, per-trial decode latency (median, 95th percentile, max), and for --fast the
same with PipelineConfig(fast=True) (accuracy reported, not checked: fast is not bit-identical).

    .venv\\Scripts\\python experiments\\live_replay_check.py          # ~30 min GPU
    .venv\\Scripts\\python experiments\\live_replay_check.py --fast   # ~12 min GPU
"""

from __future__ import annotations

import argparse
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

from experiments.r11_round3_diverse_members import CACHE, rec_acc  # noqa: E402
from src.config import IV2A_SUBJECTS, RESULTS_DIR  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.live import LiveDecoder  # noqa: E402
from src.pipeline import R11, load_subject_data, tslr_member  # noqa: E402

TOL_PROB = 1e-5


def run(ds: str, s: int, fast: bool, ref: pd.Series, device) -> dict:
    data, n_classes = load_subject_data(ds, s)
    tr = data.train
    cfg = dataclasses.replace(R11, one_hot_targets=(ds == "IV-2b"), fast=fast)
    wait_until_healthy()
    torch.cuda.synchronize()
    t = time.perf_counter()
    dec = LiveDecoder.calibrate(data.X_cnn[tr], data.X_tslr[tr], data.y[tr], data.groups[tr], n_classes, cfg, device)
    torch.cuda.synchronize()
    t_cal = time.perf_counter() - t
    outs, current = [], None
    for i in np.flatnonzero(~tr):
        if data.groups[i] != current:
            dec.start_session()
            current = data.groups[i]
        outs.append(dec.decode(data.X_cnn[i], data.X_tslr[i])["ensemble"])
    live = np.stack(outs)
    y, rec = data.y[~tr], data.groups[~tr]
    acc = rec_acc(live, y, rec)
    if not fast:
        c1 = CACHE / "c1_preds" / f"{ds}_s{s}"
        ens = (np.mean([np.load(c1 / f"cnn_seed{k}_causal.npy") for k in range(3)], 0)
               + tslr_member(data, dataclasses.replace(cfg, alignment="causal"))
               + np.mean([np.load(c1 / f"fts_seed{k}_causal.npy") for k in range(2)], 0)) / 3
        diff = float(np.abs(live - ens).max())
        if not (diff < TOL_PROB and abs(acc - ref[(ds, s)]) < 1e-12):
            raise AssertionError(f"V1: {ds} s{s} live differs from C1 causal (max |dp| {diff:.2e}, "
                                 f"acc {acc:.6f} vs {ref[(ds, s)]:.6f})")
    lat = np.array(dec.latencies_ms)
    return {"dataset": ds, "subject": s, "fast": fast, "acc": acc, "calibrate_s": t_cal,
            "lat_median_ms": float(np.median(lat)), "lat_p95_ms": float(np.percentile(lat, 95)),
            "lat_max_ms": float(lat.max()), "n_trials": len(lat)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    ref = pd.read_csv(RESULTS_DIR / "c1_causal_alignment.csv").set_index(["dataset", "subject"]).causal
    device = torch.device("cuda")
    rows = []
    for s in IV2A_SUBJECTS:
        for ds in ("IV-2a", "IV-2b"):
            r = run(ds, s, args.fast, ref, device)
            rows.append(r)
            print(f"  {ds} s{s}: acc {100 * r['acc']:.2f} calibrate {r['calibrate_s']:.0f}s, decode median "
                  f"{r['lat_median_ms']:.1f} ms p95 {r['lat_p95_ms']:.1f} ms max {r['lat_max_ms']:.1f} ms", flush=True)
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / f"live_replay_check{'_fast' if args.fast else ''}.csv"
    df.to_csv(out, index=False)
    print(("\nfast path" if args.fast else "\nexact path: V1 passed, live decoding equals C1's causal R11 on all 18"))
    print(df.groupby("dataset")[["acc", "calibrate_s", "lat_median_ms", "lat_p95_ms", "lat_max_ms"]].mean().round(3)
          .to_string())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
