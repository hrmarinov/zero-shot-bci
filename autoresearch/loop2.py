"""Rounds 2+ of program.md: greedy keep / discard until a whole round keeps nothing (or 90 experiments in total)."""

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


W = windows
ROUND2 = [
    ("feature dispersion matching per session", {"scale": True}, False),
    ("Euclidean alignment for tangent features", {"align": "euclid"}, False),
    ("Ledoit-Wolf covariance", {"cov": "lwf"}, False),
    ("Tyler covariance", {"cov": "tyl"}, False),
    ("pooled band 8-35 Hz", {"pool_feats": W((8, 35))}, False),
    ("pooled band 6-32 Hz", {"pool_feats": W((6, 32))}, False),
    ("pooled window 0.5-2.5 + 1.5-3.5 s", {"pool_feats": W((8, 30)) + W((8, 30), t=(1.5, 3.5))}, False),
    ("library: session 1 only", {"lib_session": 0}, False),
    ("library: session 2 only", {"lib_session": 1}, False),
    ("second pooled member: 4-38 Hz", {"w_pool2": 1.0, "pool2_feats": W((4, 38))}, False),
    ("second pooled member: LDA", {"w_pool2": 1.0, "pool2_feats": [None], "pool2_clf": "lda"}, False),
    ("second pooled member: late window", {"w_pool2": 1.0, "pool2_feats": W((8, 30), t=(1.5, 3.5))}, False),
    ("add own TS member back at 0.5", {"w_own": 0.5}, False),
    ("pooled with own weight 1", {"pool_w": 1.0}, False),
    ("drop CNN member", {"w_cnn": 0.0}, True),
]


def neighbours(cfg: dict) -> list:
    full = {**candidate.DEFAULT, **cfg}
    out = []
    for key, values in (("w_pool", (1.5, 3.0)), ("w_cnn", (0.5, 1.5)), ("w_fts", (0.5, 1.5)), ("pool_C", (0.3, 3.0))):
        out += [(f"{key} {v}", {key: v}, False) for v in values if full[key] != v]
    return out


def git(*args: str) -> None:
    subprocess.run(["git", *args], cwd=HERE.parent, check=True, capture_output=True)


def main() -> None:
    data = {s: prepare.read(s) for s in prepare.TUNE}
    best_cfg = dict(candidate.CFG)
    best = prepare.evaluate("candidate", prepare.TUNE, data)
    log = HERE / "results.tsv"
    base = log.read_text(encoding="utf-8").rstrip("\n")
    rows, n, rnd = [f"r2-start	{best.mean():.3f}	keep	round 1 candidate without balancing"], 25, 2
    print(rows[0], flush=True)
    while n < 90:
        ideas = ROUND2 + neighbours(best_cfg) if rnd == 2 else neighbours(best_cfg) + ROUND2
        kept = 0
        for name, change, simpler in ideas:
            if n >= 90:
                break
            trial = {**best_cfg, **change}
            if {**candidate.DEFAULT, **trial} == {**candidate.DEFAULT, **best_cfg}:
                continue
            n += 1
            candidate.CFG = trial
            acc = prepare.evaluate("candidate", prepare.TUNE, data)
            d = acc - best
            keep = bool((d.mean() >= MIN_GAIN and (d > 0).sum() > (d < 0).sum())
                        or (simpler and d.mean() >= -TIE and (d != 0).any()))
            rows.append(f"{n}	{acc.mean():.3f}	{'keep' if keep else 'discard'}	round {rnd}: {name} "
                        f"({d.mean():+.2f} pp, {(d > 0).sum()} up / {(d < 0).sum()} down)")
            print(rows[-1], flush=True)
            log.write_text(base + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
            if keep:
                kept += 1
                best, best_cfg = acc, trial
                path = HERE / "candidate.py"
                text = re.sub(r"^CFG = .*$", lambda _: f"CFG = {best_cfg!r}", path.read_text(encoding="utf-8"), count=1, flags=re.M)
                path.write_text(text, encoding="utf-8")
                git("add", "autoresearch/candidate.py")
                git("commit", "-qm", f"feat(autoresearch): keep exp {n}, {name} (tune {acc.mean():.3f})")
        print(f"ROUND {rnd} kept {kept}", flush=True)
        if kept == 0:
            break
        rnd += 1
    candidate.CFG = best_cfg
    print(f"DONE experiments {n} best tune_acc {best.mean():.3f} CFG {best_cfg}")


if __name__ == "__main__":
    main()
