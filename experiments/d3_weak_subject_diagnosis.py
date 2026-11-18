"""D3, why do weak Lee2019 subjects stay weak? Diagnostic, CPU only, no model is fitted for deployment.
PRE-REGISTERED (committed before the first run). Uses the D2 caches (no new data, no new GPU fits).

Evaluation labels are used here ONLY to diagnose (ceilings); nothing in the pipeline is fitted or selected on them.

Per subject (Lee2019_MI, 20 channels, 8-30 Hz, 0.5-2.5 s, per-session Riemannian alignment, TS+LR and CSP+LDA):
  within1, within2  5-fold stratified CV accuracy inside session 1 / session 2 (shuffle seed 0): the signal
                    ceiling of each session for a subject-specific linear decoder (max over the two decoders)
  cross12, cross21  train on one session, test on the other (same decoders, max)
  r11               D2 result (session 1 -> 2)
FIXED CATEGORIES (thresholds fixed here, before the run):
  no_signal         max(within1, within2) < 0.60        (even a within-session decoder is near chance)
  transfer_limited  not no_signal and within2 - r11 >= 0.10   (signal exists but does not carry across days)
  other             the rest
REPORTED: counts per category among all subjects and among r11 < 70 % subjects; mean r11 per category;
  headroom = mean over subjects of max(0, within2 - r11) overall and per category (an upper bound on what a
  better cross-session method could add; within2 is itself a noisy 100-trial CV estimate and is optimistic);
  Spearman of r11 with within2, within1 and (within2 - cross12 style) drift; share of the r11 shortfall to the
  top-quartile subject mean that each category explains.

    .venv\\Scripts\\python experiments\\d3_weak_subject_diagnosis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.baselines import make_csp_lda  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.pipeline import align_by_group, fit_tslr  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "d2_lee2019"
NO_SIGNAL, TRANSFER_GAP, WEAK = 0.60, 0.10, 0.70


def fit_predict_csp(A, y, B):
    return make_csp_lda().fit(A, y).predict(B)


def fit_predict_ts(A, y, B):
    return fit_tslr(A, y)(B).argmax(1)


DECODERS = {"csp": fit_predict_csp, "ts": fit_predict_ts}


def within(A: np.ndarray, y: np.ndarray) -> float:
    best = 0.0
    for fp in DECODERS.values():
        hits = 0
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(A, y):
            hits += int((fp(A[tr], y[tr], A[te]) == y[te]).sum())
        best = max(best, hits / len(y))
    return best


def cross(A_tr, y_tr, A_te, y_te) -> float:
    return max(float((fp(A_tr, y_tr, A_te) == y_te).mean()) for fp in DECODERS.values())


def diagnose(subject: int) -> dict:
    z = np.load(CACHE / f"s{subject}.npz", allow_pickle=True)
    A = align_by_group(z["X_tslr"].astype(np.float64), z["groups"])
    y, tr = z["y"], z["train"]
    return {"subject": subject, "within1": within(A[tr], y[tr]), "within2": within(A[~tr], y[~tr]),
            "cross12": cross(A[tr], y[tr], A[~tr], y[~tr]), "cross21": cross(A[~tr], y[~tr], A[tr], y[tr])}


def main() -> int:
    d2 = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    df = pd.DataFrame([diagnose(s) for s in d2.index]).set_index("subject")
    df["r11"] = d2.r11
    ceiling = df[["within1", "within2"]].max(axis=1)
    df["category"] = np.where(ceiling < NO_SIGNAL, "no_signal",
                              np.where(df.within2 - df.r11 >= TRANSFER_GAP, "transfer_limited", "other"))
    df["headroom"] = (df.within2 - df.r11).clip(lower=0)
    df.to_csv(RESULTS_DIR / "d3_weak_subject_diagnosis.csv")
    weak = df.r11 < WEAK
    print(f"=== D3 weak-subject diagnosis, Lee2019_MI, {len(df)} subjects, {int(weak.sum())} with r11 < 70 % ===")
    print(f"mean r11 {100 * df.r11.mean():.2f} | within1 {100 * df.within1.mean():.2f} | within2 "
          f"{100 * df.within2.mean():.2f} | cross12 {100 * df.cross12.mean():.2f} | cross21 {100 * df.cross21.mean():.2f}")
    g = df.groupby("category").agg(n=("r11", "size"), n_weak=("r11", lambda s: int((s < WEAK).sum())),
                                    r11=("r11", "mean"), within2=("within2", "mean"), headroom=("headroom", "mean"))
    g[["r11", "within2", "headroom"]] *= 100
    g["headroom_share_of_total"] = g.headroom * g.n / (100 * df.headroom.sum() + 1e-12) / 100 * 100
    print(g.round(2).to_string())
    print(f"\ntotal headroom (mean over all subjects of max(0, within2 - r11)): {100 * df.headroom.mean():.2f} pp")
    top = df.r11.sort_values().index[-len(df) // 4:]
    short = (df.r11[top].mean() - df.r11).clip(lower=0)
    print("share of the shortfall to the top-quartile mean explained by category: " +
          ", ".join(f"{c} {short[df.category == c].sum() / short.sum():.0%}" for c in g.index))
    for col in ("within1", "within2", "cross12"):
        r, p = spearmanr(df[col], df.r11)
        print(f"Spearman r11 vs {col}: {r:+.2f} (p = {p:.3g})")
    drift = df.within2 - df.cross12
    r, p = spearmanr(drift, df.r11)
    print(f"Spearman r11 vs (within2 - cross12): {r:+.2f} (p = {p:.3g}); mean gap {100 * drift.mean():.2f} pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
