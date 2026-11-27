"""Fixed evaluation harness (see program.md). The loop must not edit this file.

    python autoresearch/prepare.py [module]     # TUNE score of autoresearch/<module>.py (default: candidate)
    python autoresearch/prepare.py --final      # the single HOLDOUT run
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "autoresearch")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.d2_lee2019_dev import CACHE, load, sign_flip_p  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402

ORDER = np.random.default_rng(2026).permutation(np.arange(1, 55))
TUNE, HOLDOUT = sorted(int(s) for s in ORDER[:27]), sorted(int(s) for s in ORDER[27:])


def read(s: int) -> dict:
    d = load(None, s, {}, {}, ["cached"])
    c = CACHE / "preds" / f"s{s}"
    return {"id": s, "d": d, "cnn3": np.mean([np.load(c / f"cnn_seed{k}_fast.npy") for k in range(3)], 0),
            "fts2": np.mean([np.load(c / f"fts_seed{k}_fast.npy") for k in range(2)], 0)}


def split(r: dict) -> tuple[dict, dict, np.ndarray]:
    """(own labelled session 1, unlabelled session 2 + cached members, hidden session-2 labels)."""
    d, tr = r["d"], r["d"].train
    own = {"id": r["id"], "X_ts": d.X_tslr[tr], "X_cnn": d.X_cnn[tr], "y": d.y[tr]}
    test = {"id": r["id"], "X_ts": d.X_tslr[~tr], "X_cnn": d.X_cnn[~tr], "cnn3": r["cnn3"], "fts2": r["fts2"]}
    return own, test, d.y[~tr]


def evaluate(module: str, ids: list[int], data: dict) -> pd.Series:
    predict = importlib.import_module(module).predict
    out = {}
    for s in ids:
        own, test, y = split(data[s])
        library = []
        for o in TUNE:
            if o != s:
                a, b, yb = split(data[o])
                library.append({"id": o, "sessions": [{"X_ts": a["X_ts"], "X_cnn": a["X_cnn"], "y": a["y"]},
                                                      {"X_ts": b["X_ts"], "X_cnn": b["X_cnn"], "y": yb}]})
        p = predict(own, test, library)
        if p.shape != (len(y), 2) or not np.isfinite(p).all():
            raise ValueError(f"s{s}: bad probabilities")
        out[s] = 100 * float((p.argmax(1) == y).mean())
    return pd.Series(out)


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "candidate"
    d2 = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    if arg != "--final":
        data = {s: read(s) for s in TUNE}
        acc = evaluate(arg, TUNE, data)
        acc.to_csv(ROOT / "autoresearch" / f"last_{arg}.csv", header=["acc"])
        print(f"tune_acc: {acc.mean():.3f}  (R11 on TUNE {100 * d2.loc[TUNE, 'r11'].mean():.3f})")
        return 0
    flag = ROOT / "autoresearch" / "final_holdout.csv"
    if flag.exists():
        raise SystemExit("the holdout was already scored; no second run")
    data = {s: read(s) for s in range(1, 55)}
    df = pd.DataFrame({m: evaluate(m, HOLDOUT, data) for m in ("candidate", "reference")})
    df["r11"] = 100 * d2.loc[HOLDOUT, "r11"]
    df.to_csv(flag)
    print(df.mean().round(3).to_string())
    for a, b in (("candidate", "r11"), ("candidate", "reference"), ("reference", "r11")):
        d = df[a] - df[b]
        print(f"{a} - {b}: {d.mean():+.2f} pp ({(d > 0).sum()} up/{(d < 0).sum()} down), p = {sign_flip_p(d.to_numpy()):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
