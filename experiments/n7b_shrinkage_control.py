"""N7b, shrinkage control for the N7 drift forecast. PRE-REGISTERED.

Committed BEFORE any control arm is scored. N7 (experiments/n7_drift_forecast_screen.py) found R11[ridge] - R11[base_I] = +1.69 pp
on Lee2019 and ridge - const = +1.15 pp, but registered no control that separates "forecasting a subject's drift" from "pulling
the subject's class means toward a population prior" (or from any 53-subject ridge fit that ignores the subject's own pairing).
Same data, features, leave-one-subject-out protocol, LR (C = 1) and R11 stack as N7; CPU only; Zhou2020 sealed, BNCI2015_001 unused.
N7's ridge, base_I and const numbers are recomputed here and must equal N7's results/n7_drift_forecast_lee2019.csv exactly.

CONTROL ARMS (same translation of session-1 trials by delta_c per class as N7; only delta differs):
  shrink2  delta = (1 - lam) m1_s + lam mu2 - m1_s, mu2 = mean over the 53 other subjects of their session-2 class means:
           pulls the subject toward the population session-2 class means. lam in {0, 0.1, ..., 1} chosen by leave-one-out over
           the 53 others (each other subject's m2 predicted from its own m1 and mu2 of the remaining 52; squared error).
  shrink1  delta = lam (mu1 - m1_s), mu1 = population mean of session-1 class means (pure regularisation toward the population
           with no drift term); lam chosen the same way, on the error of predicting m2 from the shrunk m1.
  ridge_perm  as N7 ridge, but the 53 others' session-2 targets are cyclically shifted by one subject relative to their session-1
           inputs (pairing destroyed; sorted subject order, fixed): a ridge that cannot know subject-specific drift.
CONTRASTS (R11 with the arm's TS+LR member, Lee2019 n = 54, one-sided sign-flip Monte Carlo as D2, Holm over the 3):
  ridge - shrink2, ridge - shrink1, ridge - ridge_perm.
DECISION. The N7 gain is attributed to SUBJECT-SPECIFIC DRIFT FORECASTING only if ridge beats ALL THREE controls by >= +0.5 pp with
Holm p < 0.05. If not: the mechanism is a population prior / generic shrinkage, and the candidate carried forward is the SIMPLEST
control arm whose gain over base_I is >= +0.5 pp with one-sided p < 0.05 (ties on significance are broken by fewer parameters:
shrink1 < shrink2 < ridge_perm), because a one-parameter shrinkage deploys with no subject library beyond two population means.
REPORTED, NOT FOR DECISION: every arm's mean R11, gain over base_I, up/down, D3 transfer-limited and bottom-third means.

    .venv\Scripts\python experiments\n7b_shrinkage_control.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402

from experiments.d2_lee2019_dev import sign_flip_p  # noqa: E402
from experiments.n7_drift_forecast_screen import ALPHAS, N_SUB, class_means, fit_predict, subject_features  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.evaluate import holm_adjust  # noqa: E402

LAMS = np.round(np.arange(0, 1.01, 0.1), 1)
MIN_GAIN_PP, ALPHA = 0.5, 0.05
CONTROLS = ("shrink2", "shrink1", "ridge_perm")
SIMPLICITY = ("shrink1", "shrink2", "ridge_perm")


def choose_lambda(M1: dict, M2: dict, others: list[int], target: str) -> float:
    """Leave-one-out over ``others``: squared error of predicting m2 from the shrunk m1 (population mean of the remaining ones)."""
    errs = []
    for lam in LAMS:
        e = 0.0
        for o in others:
            rest = [r for r in others if r != o]
            mu = np.mean([(M2 if target == "shrink2" else M1)[r] for r in rest], 0)
            e += ((M2[o] - ((1 - lam) * M1[o] + lam * mu)) ** 2).sum()
        errs.append(e)
    return float(LAMS[int(np.argmin(errs))])


def main() -> int:
    d3 = pd.read_csv(RESULTS_DIR / "d3_weak_subject_diagnosis.csv").set_index("subject")
    d2 = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    n7 = pd.read_csv(RESULTS_DIR / "n7_drift_forecast_lee2019.csv", index_col=0)
    S = {s: subject_features(s) for s in range(1, N_SUB + 1)}
    M1 = {s: class_means(v["F_tr"], v["y_tr"]) for s, v in S.items()}
    M2 = {s: class_means(v["F_te"], v["y_te"]) for s, v in S.items()}
    rows, lams = {}, {"shrink1": [], "shrink2": []}
    for s, v in S.items():
        others = [o for o in S if o != s]
        X = np.stack([M1[o].ravel() for o in others])
        Y = np.stack([M2[o].ravel() for o in others])
        delta = {"ridge": RidgeCV(alphas=ALPHAS).fit(X, Y).predict(M1[s].ravel()[None])[0].reshape(2, -1) - M1[s],
                 "ridge_perm": RidgeCV(alphas=ALPHAS).fit(X, np.roll(Y, 1, axis=0)).predict(M1[s].ravel()[None])[0].reshape(2, -1) - M1[s]}
        l2, l1 = choose_lambda(M1, M2, others, "shrink2"), choose_lambda(M1, M2, others, "shrink1")
        lams["shrink2"].append(l2)
        lams["shrink1"].append(l1)
        delta["shrink2"] = l2 * (np.mean([M2[o] for o in others], 0) - M1[s])
        delta["shrink1"] = l1 * (np.mean([M1[o] for o in others], 0) - M1[s])
        delta["base_I"] = None
        rows[s] = {}
        for k, d in delta.items():
            p = fit_predict(v["F_tr"], v["y_tr"], d, v["F_te"])
            rows[s][f"r11_{k}"] = 100 * ((((v["cnn3"] + p + v["fts2"]) / 3).argmax(1)) == v["y_te"]).mean()
    df = pd.DataFrame(rows).T
    for k in ("base_I", "ridge"):
        if (df[f"r11_{k}"] - n7[f"r11_{k}"]).abs().max() > 1e-9:
            raise AssertionError(f"{k} does not reproduce N7")
    print("reproduces N7: base_I, ridge")
    print(f"lambda chosen: shrink2 median {np.median(lams['shrink2']):.1f}, shrink1 median {np.median(lams['shrink1']):.1f}")
    arms = ["base_I", "ridge", *CONTROLS]
    tl, third = d3.index[d3.category == "transfer_limited"], d2.csp_lda.sort_values().index[: N_SUB // 3]
    for a in arms:
        g = df[f"r11_{a}"] - df.r11_base_I
        print(f"{a:11s} R11 {df[f'r11_{a}'].mean():.2f}  gain over base_I {g.mean():+.2f} pp ({(g > 0).sum()} up/{(g < 0).sum()} down), "
              f"p {sign_flip_p(g.to_numpy()):.4f}; transfer_limited {df.loc[tl, f'r11_{a}'].mean():.2f}, "
              f"bottom third {df.loc[third, f'r11_{a}'].mean():.2f}")
    ct = pd.DataFrame([{"contrast": f"ridge-{c}", "mean_pp": (df.r11_ridge - df[f"r11_{c}"]).mean(),
                        "up": int((df.r11_ridge > df[f"r11_{c}"]).sum()), "down": int((df.r11_ridge < df[f"r11_{c}"]).sum()),
                        "p": sign_flip_p((df.r11_ridge - df[f"r11_{c}"]).to_numpy())} for c in CONTROLS])
    ct["holm_p"] = holm_adjust(ct.p.tolist())
    print(ct.round(4).to_string(index=False))
    specific = bool(((ct.mean_pp >= MIN_GAIN_PP) & (ct.holm_p < ALPHA)).all())
    print(f"\nSUBJECT-SPECIFIC DRIFT FORECASTING: {specific}")
    if not specific:
        ok = [c for c in SIMPLICITY if (df[f"r11_{c}"] - df.r11_base_I).mean() >= MIN_GAIN_PP
              and sign_flip_p((df[f"r11_{c}"] - df.r11_base_I).to_numpy()) < ALPHA]
        print(f"carried forward (simplest qualifying control): {ok[0] if ok else 'none: only ridge, mechanism unresolved'}")
    df.to_csv(RESULTS_DIR / "n7b_shrinkage_control_lee2019.csv")
    ct.to_csv(RESULTS_DIR / "n7b_shrinkage_control_contrasts.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
