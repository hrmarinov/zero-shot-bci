"""Check that src/pipeline.py (the consolidated R11 pipeline) reproduces R11 exactly, on all 18 subjects.

Retrains every R11 member through ``src.pipeline.fit_predict`` and asserts:
  C1 constants: CNN_BAND and CNN_TRAIN equal R3's WIDE and CFG.
  C2 every CNN member (base seeds 0-2, FTSurrogate seeds 0-1) equals its cached R3/R7/R8/R9 prediction
     (identical argmax, max |prob diff| < 1e-5, as R13's V2).
  C3 per subject, the TS+LR member, the R10 configuration and the R11 configuration give exactly the
     accuracies recorded in results/r11_round3_diverse_members.csv (m_tslr, r10, fts; < 1e-12).
The LITE configuration (FTS1 + TS+LR) is printed from the same fits, descriptive only.

    .venv\\Scripts\\python experiments\\check_pipeline_reproduces_r11.py     # ~30 min GPU (90 fits), resumable
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiments.r3_convnet_session_norm import CFG, WIDE  # noqa: E402
from experiments.r11_round3_diverse_members import CACHE, rec_acc  # noqa: E402
from src.config import IV2A_SUBJECTS, RESULTS_DIR  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.pipeline import CNN_BAND, CNN_TRAIN, LITE, R10, R11, fit_predict, load_subject_data  # noqa: E402

PIPE_CACHE = CACHE / "pipeline_preds"
TOL_PROB, TOL_ACC = 1e-5, 1e-12
REFERENCE = {  # cached member predictions per dataset and family
    "IV-2a": {"cnn": lambda s, k: CACHE / "r3_preds" / f"s{s}_cnn_ra_seed{k}.npy",
              "fts": lambda s, k: CACHE / "r8_preds" / f"s{s}_ftsurrogate_seed{k}.npy"},
    "IV-2b": {"cnn": lambda s, k: CACHE / "r7_preds" / f"s{s}_teacher_seed{k}.npy",
              "fts": lambda s, k: CACHE / "r9_preds" / f"s{s}_ftsurrogate_seed{k}.npy"},
}


def check_members(ds: str, s: int, cache_dir: Path) -> None:
    for fam, n in (("cnn", R11.n_cnn), ("fts", R11.n_fts)):
        for k in range(n):
            got, ref = np.load(cache_dir / f"{fam}_seed{k}.npy"), np.load(REFERENCE[ds][fam](s, k))
            diff = np.abs(got - ref).max()
            if not (np.array_equal(got.argmax(1), ref.argmax(1)) and diff < TOL_PROB):
                raise AssertionError(f"C2: {ds} s{s} {fam} seed{k} differs from its cache (max diff {diff:.2e})")


def run(ds: str, s: int, ref: pd.Series, device) -> dict:
    data, n_classes = load_subject_data(ds, s)
    test_rec = data.groups[~data.train] if ds == "IV-2b" else np.zeros((~data.train).sum())
    y = data.y[~data.train]
    cache_dir = PIPE_CACHE / f"{ds}_s{s}"
    out = {}
    for name, cfg in (("r11", R11), ("r10", R10), ("lite", LITE)):
        cfg = dataclasses.replace(cfg, one_hot_targets=(ds == "IV-2b"))
        res = fit_predict(data, n_classes, cfg, device, cache_dir, before_fit=wait_until_healthy)
        out[name] = rec_acc(res["ensemble"], y, test_rec)
        if name == "r11":
            out["tslr"] = rec_acc(res["tslr"], y, test_rec)
    check_members(ds, s, cache_dir)
    for name, arm in (("tslr", "m_tslr"), ("r10", "r10"), ("r11", "fts")):
        if not abs(out[name] - ref[(ds, s, arm)]) < TOL_ACC:
            raise AssertionError(f"C3: {ds} s{s} {name} {out[name]:.6f} != R11's {arm} {ref[(ds, s, arm)]:.6f}")
    return {"dataset": ds, "subject": s, **out}


def main() -> int:
    if not (CNN_BAND == WIDE and CNN_TRAIN == CFG):
        raise AssertionError("C1: src.pipeline constants differ from R3's WIDE/CFG")
    ref = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv").set_index(["dataset", "subject", "arm"]).acc
    device = torch.device("cuda")
    rows = []
    for s in IV2A_SUBJECTS:
        for ds in ("IV-2a", "IV-2b"):
            rows.append(run(ds, s, ref, device))
        print(f"  s{s} reproduced", flush=True)
    df = pd.DataFrame(rows)
    print("\nC1-C3 passed: src.pipeline reproduces R11 (members, TS+LR, R10, R11) on all 18 subjects")
    print((100 * df.groupby("dataset")[["r11", "r10", "tslr", "lite"]].mean()).round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
