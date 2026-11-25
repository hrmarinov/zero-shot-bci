"""N7, drift forecast: learn how a session-1 decoder drifts into session 2 from the OTHER subjects. PRE-REGISTERED.

Committed BEFORE any arm is scored. Motivation: N0 found no flag that helps, and D3 found that 5 of the 12 weakest Lee2019
subjects are "transfer-limited" (a decoder fitted inside session 2 reaches ~76 %, but session 1 -> 2 reaches 61 %): signal
exists, it moves between days. The 54 subjects each have two labelled sessions, so the day-to-day movement can be learned
across subjects, which no single subject can supply. CPU only, cached data only; Zhou2020 stays sealed, BNCI2015_001 unused.

FEATURES. Per-session Riemannian alignment (src.pipeline.align_by_group, as R11), OAS covariances, tangent space at the
FIXED reference I (both sessions of every subject are aligned to I, so the feature space is common to all subjects), 210-d.
m1 = (mean feature of class 0, mean of class 1) in session 1, m2 the same in session 2; delta = m2 - m1 (2 x 210).

ARMS (TS+LR member of R11 only; CNN3 and FTS2 cached as in D2; LR with C = 1 as R11):
  base    R11's own TS+LR (subject-fitted tangent space), the D2 reference; must reproduce D2 exactly or stop.
  base_I  TS+LR on the fixed-I tangent features, no shift. The PAIRED control for the two arms below.
  const   session-1 trials of class c are translated by the mean delta_c of the 53 other subjects, then LR is fitted.
  ridge   as const, but with a subject-specific forecast: delta_hat = Ridge(m1) - m1 fitted on the 53 other subjects
          (m1 -> m2, 420 -> 420, alpha from {1, 1e1 ... 1e5} by leave-one-subject-out inside those 53, RidgeCV).
Leave-one-subject-out: the held-out subject contributes only its own labelled session 1 and its unlabelled session-2 trials
(aligned on themselves, as R11). Its session-2 labels are used only to score.

PRIMARY ENDPOINT. R11 with its TS+LR member replaced by the arm's, Lee2019 n = 54, per-subject accuracy; contrasts ridge -
base_I and const - base_I, one-sided sign-flip Monte Carlo (as D2), Holm over the 2. SECONDARY (reported): the same on the TS+LR
member alone; ridge - const; vs base; the D3 "transfer_limited" subjects (categories fixed in results/d3_weak_subject_diagnosis.csv),
the bottom third by csp_lda, share >= 70 %, up/down counts. DIAGNOSTIC: leave-one-subject-out forecast skill of delta,
R2 = 1 - sum|delta - delta_ridge|^2 / sum|delta - delta_const|^2 over subjects.
IV-2a / IV-2b: NOT applicable (8-9 subjects, a different montage, nothing to learn the drift from); the replication a pass
needs is therefore a Zhou2020 opening, justified only for an effect >= +3 pp.

KILL RULE. If the forecast skill R2 <= 0, drift is not forecastable and the track ends (ridge is then not scored for adoption).
ADOPT ridge if ALL hold: R11[ridge] - R11[base_I] >= +0.5 pp, Holm p < 0.05, AND ridge - const > 0 with one-sided p < 0.05
(the gain comes from subject-specific drift, not from a population shift). Otherwise the drift-forecast lane is closed.

    .venv\Scripts\python experiments\n7_drift_forecast_screen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from sklearn.linear_model import LogisticRegression, RidgeCV  # noqa: E402

from experiments.d2_lee2019_dev import CACHE, LITERACY, load, sign_flip_p  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.evaluate import holm_adjust  # noqa: E402
from src.pipeline import align_by_group, tslr_member  # noqa: E402

ALPHAS = (1.0, 1e1, 1e2, 1e3, 1e4, 1e5)
MIN_GAIN_PP, ALPHA = 0.5, 0.05
N_SUB = 54


def tangent_I(A: np.ndarray, ref: np.ndarray) -> np.ndarray:
    C = Covariances(estimator="oas").transform(A)
    return TangentSpace(metric="riemann").fit(ref[None]).transform(C)


def subject_features(s: int) -> dict:
    data = load(None, s, {}, {}, ["cached"])
    A = align_by_group(data.X_tslr, data.groups)
    ref = np.eye(A.shape[1])
    F = tangent_I(A, ref)
    tr = data.train
    c = CACHE / "preds" / f"s{s}"
    return {"F_tr": F[tr], "y_tr": data.y[tr], "F_te": F[~tr], "y_te": data.y[~tr], "data": data,
            "cnn3": np.mean([np.load(c / f"cnn_seed{k}_fast.npy") for k in range(3)], 0),
            "fts2": np.mean([np.load(c / f"fts_seed{k}_fast.npy") for k in range(2)], 0)}


def class_means(F: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([F[y == k].mean(0) for k in (0, 1)])  # (2, d)


def fit_predict(F_tr: np.ndarray, y_tr: np.ndarray, shift: np.ndarray | None, F_te: np.ndarray) -> np.ndarray:
    X = F_tr if shift is None else F_tr + shift[y_tr]
    return LogisticRegression(C=1.0, max_iter=5000).fit(X, y_tr).predict_proba(F_te)


def main() -> int:
    d2 = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    d3 = pd.read_csv(RESULTS_DIR / "d3_weak_subject_diagnosis.csv").set_index("subject")
    S = {s: subject_features(s) for s in range(1, N_SUB + 1)}
    # session-2 labelled class means exist for the OTHER subjects only (used inside the leave-one-out below)
    M1 = {s: class_means(v["F_tr"], v["y_tr"]) for s, v in S.items()}
    M2 = {s: class_means(v["F_te"], v["y_te"]) for s, v in S.items()}
    rows, skill = {}, []
    for s, v in S.items():
        others = [o for o in S if o != s]
        X = np.stack([M1[o].ravel() for o in others])
        Y = np.stack([M2[o].ravel() for o in others])
        delta_const = (Y - X).mean(0).reshape(2, -1)
        rcv = RidgeCV(alphas=ALPHAS).fit(X, Y)
        delta_ridge = rcv.predict(M1[s].ravel()[None])[0].reshape(2, -1) - M1[s]
        true = M2[s] - M1[s]
        skill.append((((true - delta_ridge) ** 2).sum(), ((true - delta_const) ** 2).sum(), rcv.alpha_))
        y, ye = v["y_te"], v["y_te"]
        ts = {"base": tslr_member(v["data"]),
              "base_I": fit_predict(v["F_tr"], v["y_tr"], None, v["F_te"]),
              "const": fit_predict(v["F_tr"], v["y_tr"], delta_const, v["F_te"]),
              "ridge": fit_predict(v["F_tr"], v["y_tr"], delta_ridge, v["F_te"])}
        r11 = {k: (v["cnn3"] + t + v["fts2"]) / 3 for k, t in ts.items()}
        if abs((r11["base"].argmax(1) == ye).mean() - d2.loc[s, "r11"]) > 1e-9:
            raise AssertionError(f"s{s}: rebuilt R11 differs from D2")
        rows[s] = {**{f"r11_{k}": 100 * (p.argmax(1) == ye).mean() for k, p in r11.items()},
                   **{f"ts_{k}": 100 * (p.argmax(1) == ye).mean() for k, p in ts.items()}}
    df = pd.DataFrame(rows).T
    sk = np.array(skill)
    r2 = 1 - sk[:, 0].sum() / sk[:, 1].sum()
    print(f"forecast skill R2 (leave-one-subject-out, delta_ridge vs delta_const): {r2:+.4f}; median alpha {np.median(sk[:, 2]):g}")
    print(f"R11 base {df.r11_base.mean():.2f}, base_I {df.r11_base_I.mean():.2f}, const {df.r11_const.mean():.2f}, "
          f"ridge {df.r11_ridge.mean():.2f}")
    out = []
    for lab, a, b in (("ridge-base_I", "ridge", "base_I"), ("const-base_I", "const", "base_I"), ("ridge-const", "ridge", "const"),
                      ("ridge-base", "ridge", "base")):
        for tier in ("r11", "ts"):
            d = df[f"{tier}_{a}"] - df[f"{tier}_{b}"]
            out.append({"contrast": lab, "tier": tier, "mean_pp": d.mean(), "up": int((d > 0).sum()), "down": int((d < 0).sum()),
                        "p": sign_flip_p(d.to_numpy())})
    ct = pd.DataFrame(out)
    prim = ct[(ct.tier == "r11") & ct.contrast.isin(["ridge-base_I", "const-base_I"])]
    ct["holm_p"] = np.nan
    ct.loc[prim.index, "holm_p"] = holm_adjust(prim.p.tolist())
    print(ct.round(4).to_string(index=False))
    tl = d3.index[d3.category == "transfer_limited"]
    third = d2.csp_lda.sort_values().index[: N_SUB // 3]
    for name, idx in (("transfer_limited (D3)", tl), ("bottom third (csp_lda)", third)):
        print(f"{name} n={len(idx)}: R11 base_I {df.loc[idx, 'r11_base_I'].mean():.2f}, const {df.loc[idx, 'r11_const'].mean():.2f}, "
              f"ridge {df.loc[idx, 'r11_ridge'].mean():.2f}")
    for k in ("base_I", "ridge"):
        print(f"share >= 70 % {k}: {(df[f'r11_{k}'] >= 100 * LITERACY).mean():.1%}")
    gain = ct.set_index(["contrast", "tier"])
    g, hp = gain.loc[("ridge-base_I", "r11"), "mean_pp"], gain.loc[("ridge-base_I", "r11"), "holm_p"]
    p_rc = gain.loc[("ridge-const", "r11"), "p"]
    adopt = r2 > 0 and g >= MIN_GAIN_PP and hp < ALPHA and p_rc < ALPHA
    print(f"\nADOPT ridge: {adopt} (R2 {r2:+.3f}, gain {g:+.2f} pp, Holm p {hp:.4f}, ridge-const p {p_rc:.4f})")
    df.to_csv(RESULTS_DIR / "n7_drift_forecast_lee2019.csv")
    ct.to_csv(RESULTS_DIR / "n7_drift_forecast_contrasts.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
