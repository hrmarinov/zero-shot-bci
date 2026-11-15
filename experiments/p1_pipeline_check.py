"""Check that the pretraining option in src.pretrain / src.pipeline reproduces P1 bit-for-bit (engineering).

For IV-2a s1 and IV-2b s1:
  C1 src.pretrain.pretrain_members(population_pool(other 8 subjects)) equals P1's cached pretrained weights;
  C2 src.pretrain.zero_shot_predict equals P1's cached zero-shot members' ensemble;
  C3 fit_predict(R11_PT, pretrained=...) equals P1's cached fine-tuned members at FULL and at n=48 draw 0.
All comparisons are exact (torch.equal / np.array_equal).

    .venv\\Scripts\\python experiments\\p1_pipeline_check.py     # ~6 min GPU
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.b1_r11_calibration_burden import stratified_draw, subset  # noqa: E402
from experiments.p1_pretrain_burden import P1_CACHE, SPECS  # noqa: E402
from src.config import IV2A_SUBJECTS  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.pipeline import R11_PT, fit_predict, load_subject_data  # noqa: E402
from src.pretrain import population_pool, pretrain_members, zero_shot_predict  # noqa: E402

NAMES = [f"{'cnn' if a is None else 'fts'}_seed{s}" for s, a in SPECS]


def main() -> int:
    device = torch.device("cuda")
    s = 1
    for ds in ("IV-2a", "IV-2b"):
        one_hot = ds == "IV-2b"
        cfg = dataclasses.replace(R11_PT, one_hot_targets=one_hot)
        data, n_classes = load_subject_data(ds, s)
        X, y = population_pool([load_subject_data(ds, o)[0] for o in IV2A_SUBJECTS if o != s])
        wait_until_healthy()
        pre = pretrain_members(X, y, n_classes, cfg, one_hot, device=device)
        cached = torch.load(P1_CACHE / f"{ds}_s{s}" / "pretrained.pt")
        if not all(torch.equal(a[k], b[k]) for a, b in zip(pre, cached) for k in a):
            raise AssertionError(f"C1: {ds} s{s} pretrained weights differ from P1")
        zs = zero_shot_predict(pre, data.X_cnn[~data.train], data.groups[~data.train], cfg, device)
        z = [np.load(P1_CACHE / f"{ds}_s{s}" / "zeroshot" / f"{n}.npy") for n in NAMES]
        if not np.array_equal(zs, (np.mean(z[:3], 0) + np.mean(z[3:], 0)) / 2):
            raise AssertionError(f"C2: {ds} s{s} zero-shot differs from P1")
        for budget, sub in (("FULL", data),
                            ("48", subset(data, stratified_draw(data.y[data.train], 48, seed=1000 * 48 + s)))):
            wait_until_healthy()
            out = fit_predict(sub, n_classes, cfg, device, pretrained=pre)
            ft = [np.load(P1_CACHE / f"{ds}_s{s}" / f"n{budget}_d0" / f"{n}.npy") for n in NAMES]
            if not (np.array_equal(out["cnn"], np.mean(ft[:3], 0)) and np.array_equal(out["fts"], np.mean(ft[3:], 0))):
                raise AssertionError(f"C3: {ds} s{s} n{budget} fine-tuned members differ from P1")
        print(f"  {ds} s{s}: C1-C3 passed (bit-identical to P1)", flush=True)
    print("src.pretrain + fit_predict(pretrained=...) reproduce P1 exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
