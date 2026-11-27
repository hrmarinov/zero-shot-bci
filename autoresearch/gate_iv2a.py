"""Gate before any Zhou2020 opening: does the frozen candidate transfer to a multi-class, small-library setting? PRE-REGISTERED.

Committed BEFORE the candidate is scored on IV-2a. Zhou2020 (protocol) is 3-class, 7 sessions, and its montage allows a pooled library
only within a group (subjects 1-12: 11 others, 41 unnamed channels; 13-20: 7 others, 26 channels). IV-2a (4-class, library of 8) is the
closest cached setting. N9 / N12 already showed the pooled MEMBER failing there; the frozen candidate itself has not been scored.
CANDIDATE (autoresearch/candidate.py, balancing off): (CNN3 + FTS2 + 2 x zero-shot pooled TS+LR) / 4.
GATE: PASS iff mean(candidate - R11) >= 0 over the 9 subjects AND at least 5 of 9 subjects are not worse. Class balancing is
reported separately and does not enter the gate. FAIL means Zhou2020 is not opened for this candidate without the user's decision.

    .venv\\Scripts\\python autoresearch\\gate_iv2a.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.n8_pooled_decoder_screen import acc, pool_proba, r11_pred  # noqa: E402
from experiments.n9_ens4_controls import iv2a_set  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402


def balance(p: np.ndarray) -> np.ndarray:
    for _ in range(200):
        p = p / p.sum(0, keepdims=True)
        p = p / p.sum(1, keepdims=True)
    return p


def main() -> int:
    r11 = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv")
    ref = r11[(r11.dataset == "IV-2a") & (r11.arm == "fts")].set_index("subject").acc
    S = iv2a_set(ref)
    out = {}
    for s, v in S.items():
        p = (v["cnn3"] + v["fts2"] + 2 * pool_proba(S, {s}, None, 0, v)) / 4
        out[s] = {"r11": 100 * ref[s], "candidate": 100 * acc(v, p), "candidate_balanced": 100 * acc(v, balance(p))}
    df = pd.DataFrame(out).T
    d = df.candidate - df.r11
    print(df.round(2).to_string())
    print(df.mean().round(2).to_string())
    ok = bool(d.mean() >= 0 and (d >= 0).sum() >= 5)
    print(f"candidate - R11: {d.mean():+.2f} pp, {(d > 0).sum()} up / {(d < 0).sum()} down. GATE: {'PASS' if ok else 'FAIL'}")
    df.to_csv(ROOT / "autoresearch" / "gate_iv2a.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
