"""Greedy keep / discard runner for program.md: one change at a time against the incumbent, at most 30 experiments."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import prepare  # noqa: I001  (first: it puts the repo root and this folder on sys.path)
import candidate

HERE = Path(__file__).resolve().parent
MIN_GAIN, TIE = 0.3, 0.1


def windows(*bands: tuple, t: tuple = (0.5, 2.5)) -> list:
    return [(b, t) for b in bands]


IDEAS = [  # (description, change, is_simplification)
    ("add pooled 4th member (own weight 20)", {"w_pool": 1.0}, False),
    ("pooled member zero-shot", {"pool_w": 0.0}, True),
    ("pooled own weight 5", {"pool_w": 5.0}, False),
    ("pooled ensemble weight 2", {"w_pool": 2.0}, False),
    ("own TS: fixed-I + shrink lam 0.8", {"own_mode": "I", "lam": 0.8}, False),
    ("own TS: fixed-I + shrink lam 1.0", {"own_mode": "I", "lam": 1.0}, False),
    ("drop own TS member", {"w_own": 0.0}, True),
    ("FTS weight 2", {"w_fts": 2.0}, False),
    ("CNN weight 2", {"w_cnn": 2.0}, False),
    ("geometric mean", {"geo": True}, False),
    ("class balancing", {"balance": True}, False),
    ("pooled C 0.1", {"pool_C": 0.1}, False),
    ("pooled C 0.01", {"pool_C": 0.01}, False),
    ("pooled C 10", {"pool_C": 10.0}, False),
    ("pooled window 0.5-3.5 s", {"pool_feats": windows((8, 30), t=(0.5, 3.5))}, False),
    ("pooled window 1.0-4.0 s", {"pool_feats": windows((8, 30), t=(1.0, 4.0))}, False),
    ("pooled filterbank mu+beta", {"pool_feats": windows((8, 13), (13, 30))}, False),
    ("pooled band 4-38 Hz", {"pool_feats": windows((4, 38))}, False),
    ("pooled filterbank 3 bands, 0.5-3.5 s", {"pool_feats": windows((8, 13), (13, 20), (20, 30), t=(0.5, 3.5))}, False),
    ("second pooled member (filterbank)", {"w_pool2": 1.0}, False),
    ("own filterbank mu+beta", {"own_mode": "I", "own_feats": windows((8, 13), (13, 30))}, False),
    ("own window 0.5-3.5 s", {"own_mode": "I", "own_feats": windows((8, 30), t=(0.5, 3.5))}, False),
    ("pooled library: 13 nearest subjects", {"knn": 13}, False),
    ("pooled classifier: shrinkage LDA", {"pool_clf": "lda"}, False),
    ("drop FTS member", {"w_fts": 0.0}, True),
]


def git(*args: str) -> None:
    subprocess.run(["git", *args], cwd=HERE.parent, check=True, capture_output=True)


def main() -> None:
    assert len(IDEAS) <= 30
    data = {s: prepare.read(s) for s in prepare.TUNE}
    best_cfg: dict = {}
    best = prepare.evaluate("candidate", prepare.TUNE, data)
    log = HERE / "results.tsv"
    rows = [f"0\t{best.mean():.3f}\tkeep\tbaseline R11"]
    print(rows[0], flush=True)
    for i, (name, change, simpler) in enumerate(IDEAS, 1):
        candidate.CFG = {**best_cfg, **change}
        acc = prepare.evaluate("candidate", prepare.TUNE, data)
        d = acc - best
        keep = bool((d.mean() >= MIN_GAIN and (d > 0).sum() > (d < 0).sum())
                    or (simpler and d.mean() >= -TIE and (d != 0).any()))
        rows.append(f"{i}\t{acc.mean():.3f}\t{'keep' if keep else 'discard'}\t{name} "
                    f"({d.mean():+.2f} pp, {(d > 0).sum()} up / {(d < 0).sum()} down)")
        print(rows[-1], flush=True)
        log.write_text("exp\ttune_acc\tstatus\tdescription\n" + "\n".join(rows) + "\n", encoding="utf-8")
        if keep:
            best, best_cfg = acc, candidate.CFG
            path = HERE / "candidate.py"
            text = re.sub(r"^CFG = .*$", lambda _: f"CFG = {best_cfg!r}", path.read_text(encoding="utf-8"), count=1, flags=re.M)
            path.write_text(text, encoding="utf-8")
            git("add", "autoresearch/candidate.py")
            git("commit", "-qm", f"feat(autoresearch): keep exp {i}, {name} (tune {acc.mean():.3f})")
    print(f"DONE best tune_acc {best.mean():.3f} CFG {best_cfg}")


if __name__ == "__main__":
    main()
