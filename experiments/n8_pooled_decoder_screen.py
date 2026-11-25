"""N8, pooled cross-subject decoder as the TS+LR member of R11. PRE-REGISTERED.

Committed BEFORE any arm is scored. N7c found that the accuracy-optimal pull of a subject's session-1 class means toward the
population is 0.8-1.0 (R11 +1.91 pp over base_I), i.e. the TS+LR member under-uses cross-subject information. N8 tests the direct
version: a logistic regression trained on the other subjects' labelled trials. Same features (per-session Riemannian alignment,
OAS covariance, tangent space at the fixed reference I, 210-d), leave-one-subject-out protocol and R11 stack as N7/N7c; CPU only;
Zhou2020 sealed, BNCI2015_001 unused. The held-out subject's session-2 labels are used only to score; the other 53 subjects
contribute both labelled sessions (session 1 and 2) as training trials.
N7's base_I and N7c's auto accuracies are recomputed here and must equal results/n7_drift_forecast_lee2019.csv and
results/n7c_shrinkage_strength_lee2019.csv exactly.

ARMS (TS+LR member of R11 replaced; CNN3 and FTS2 cached as in D2; LR with C = 1, max_iter 2000):
  pool_w  LR fitted on the 53 other subjects' trials (weight 1 each) plus the subject's own labelled session-1 trials with
          sample weight w, w in {0, 1, 5, 20, 100} (w = 0 is zero-shot: no data of the subject's own enters). Fixed grid, not tuned.
  auto_w  w(s) = argmax over the grid of the mean R11 session-2 accuracy over 10 FIXED validator subjects (the first 10 of
          numpy default_rng(0).permutation(1..54) that are not s), each validator o scored with a pool of the 52 subjects other
          than o and s plus o's own session-1 trials at weight w (ties: the smaller w). Applied to s with the pool of the 53 others.
  ens4    (CNN3 + TS+LR[N7c auto] + FTS2 + pool[auto_w]) / 4: the pooled decoder as an additional member next to N7c's candidate.
  The comparison arm n7c_auto is N7c's candidate (lam chosen on the other subjects) recomputed here.
PRIMARY ENDPOINT. R11 session-2 accuracy, Lee2019 n = 54, per subject. Contrasts against n7c_auto: pool[auto_w] swapped in for TS+LR,
and ens4; one-sided sign-flip Monte Carlo (as D2), Holm over the 2.
SECONDARY (reported): the contrasts against base_I; every fixed-w arm (descriptive, not for selection); the chosen-w histogram; the
D3 transfer-limited subjects, the bottom third by csp_lda, share >= 70 %, up/down.
DECISION. A pooled arm replaces n7c_auto as the candidate only if its gain over n7c_auto is >= +0.5 pp with Holm p < 0.05. If neither
passes, the cross-subject lane is closed at N7c's candidate with the finding "the pooled decoder adds nothing over the population
class-mean prior". Either way a screen on the development cohort: a Zhou2020 opening is justified only for a candidate gain >= +3 pp
over base_I, and IV-2a / IV-2b cannot replicate it (8-9 subjects, a different montage).

    .venv\Scripts\python experiments\n8_pooled_decoder_screen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from experiments.d2_lee2019_dev import LITERACY, sign_flip_p  # noqa: E402
from experiments.n7_drift_forecast_screen import N_SUB, class_means, fit_predict, subject_features  # noqa: E402
from experiments.n7c_shrinkage_strength import GRID as LAMS  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.evaluate import holm_adjust  # noqa: E402

WEIGHTS = (0, 1, 5, 20, 100)
N_VAL = 10
MIN_GAIN_PP, ALPHA = 0.5, 0.05


def pool_proba(S: dict, exclude: set[int], own: dict | None, w: float, target: dict) -> np.ndarray:
    """LR on all subjects not in ``exclude`` (both labelled sessions) plus ``own`` session-1 trials at weight ``w``."""
    ids = [o for o in S if o not in exclude]
    X = np.vstack([np.vstack([S[o]["F_tr"], S[o]["F_te"]]) for o in ids])
    y = np.concatenate([np.r_[S[o]["y_tr"], S[o]["y_te"]] for o in ids])
    sw = np.ones(len(y))
    if w > 0:
        X, y, sw = np.vstack([X, own["F_tr"]]), np.r_[y, own["y_tr"]], np.r_[sw, np.full(len(own["y_tr"]), float(w))]
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, y, sample_weight=sw).predict_proba(target["F_te"])


def r11_pred(v: dict, ts: np.ndarray) -> np.ndarray:
    return (v["cnn3"] + ts + v["fts2"]) / 3


def acc(v: dict, p: np.ndarray) -> float:
    return float((p.argmax(1) == v["y_te"]).mean())


def n7c_lam(S: dict, M1: dict, total: np.ndarray, s: int) -> float:
    """N7c's auto lam for subject s (other subjects' R11 accuracy, nested)."""
    sel = np.zeros(len(LAMS))
    for o in S:
        if o == s:
            continue
        mu = (total - M1[o] - M1[s]) / (N_SUB - 2)
        for i, lam in enumerate(LAMS):
            shift = None if lam == 0 else lam * (mu - M1[o])
            sel[i] += acc(S[o], r11_pred(S[o], fit_predict(S[o]["F_tr"], S[o]["y_tr"], shift, S[o]["F_te"])))
    return float(LAMS[int(np.argmax(sel))])


def main() -> int:
    d3 = pd.read_csv(RESULTS_DIR / "d3_weak_subject_diagnosis.csv").set_index("subject")
    d2 = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    n7 = pd.read_csv(RESULTS_DIR / "n7_drift_forecast_lee2019.csv", index_col=0)
    n7c = pd.read_csv(RESULTS_DIR / "n7c_shrinkage_strength_lee2019.csv", index_col=0)
    S = {s: subject_features(s) for s in range(1, N_SUB + 1)}
    M1 = {s: class_means(v["F_tr"], v["y_tr"]) for s, v in S.items()}
    total = sum(M1.values())
    order = np.random.default_rng(0).permutation(np.arange(1, N_SUB + 1))
    rows, chosen = {}, {}
    for s, v in S.items():
        mu_s = (total - M1[s]) / (N_SUB - 1)
        lam = n7c_lam(S, M1, total, s)
        ts_auto = fit_predict(v["F_tr"], v["y_tr"], None if lam == 0 else lam * (mu_s - M1[s]), v["F_te"])
        row = {"base_I": 100 * acc(v, r11_pred(v, fit_predict(v["F_tr"], v["y_tr"], None, v["F_te"]))),
               "n7c_auto": 100 * acc(v, r11_pred(v, ts_auto))}
        pools = {w: pool_proba(S, {s}, v, w, v) for w in WEIGHTS}
        for w, p in pools.items():
            row[f"pool_w{w}"] = 100 * acc(v, r11_pred(v, p))
        vals = [o for o in order if o != s][:N_VAL]
        sel = np.zeros(len(WEIGHTS))
        for o in vals:
            for i, w in enumerate(WEIGHTS):
                sel[i] += acc(S[o], r11_pred(S[o], pool_proba(S, {o, s}, S[o], w, S[o])))
        chosen[s] = WEIGHTS[int(np.argmax(sel))]
        p_auto = pools[chosen[s]]
        row["auto_w"] = row[f"pool_w{chosen[s]}"]
        row["ens4"] = 100 * acc(v, (v["cnn3"] + ts_auto + v["fts2"] + p_auto) / 4)
        rows[s] = row
        print(f"s{s}: lam {lam}, w {chosen[s]}, n7c_auto {row['n7c_auto']:.0f}, auto_w {row['auto_w']:.0f}, ens4 {row['ens4']:.0f}", flush=True)
    df = pd.DataFrame(rows).T
    for a, b in (("base_I", n7.r11_base_I), ("n7c_auto", n7c.auto)):
        if (df[a] - b).abs().max() > 1e-9:
            raise AssertionError(f"{a} does not reproduce its earlier result")
    print("\nreproduces N7 base_I and N7c auto")
    for a in ["base_I", "n7c_auto", *[f"pool_w{w}" for w in WEIGHTS], "auto_w", "ens4"]:
        g = df[a] - df.base_I
        print(f"{a:10s} R11 {df[a].mean():.2f}  gain over base_I {g.mean():+.2f} pp ({(g > 0).sum()} up/{(g < 0).sum()} down)")
    print("chosen w:", pd.Series(chosen).value_counts().sort_index().to_dict())
    ct = pd.DataFrame([{"contrast": f"{a}-n7c_auto", "mean_pp": (df[a] - df.n7c_auto).mean(), "up": int((df[a] > df.n7c_auto).sum()),
                        "down": int((df[a] < df.n7c_auto).sum()), "p": sign_flip_p((df[a] - df.n7c_auto).to_numpy())}
                       for a in ("auto_w", "ens4")])
    ct["holm_p"] = holm_adjust(ct.p.tolist())
    print(ct.round(4).to_string(index=False))
    tl, third = d3.index[d3.category == "transfer_limited"], d2.csp_lda.sort_values().index[: N_SUB // 3]
    for name, idx in (("transfer_limited", tl), ("bottom third", third)):
        print(f"{name} n={len(idx)}: " + ", ".join(f"{a} {df.loc[idx, a].mean():.2f}" for a in ("base_I", "n7c_auto", "auto_w", "ens4")))
    print("share >= 70 %: " + ", ".join(f"{a} {(df[a] >= 100 * LITERACY).mean():.1%}" for a in ("base_I", "n7c_auto", "auto_w", "ens4")))
    wins = ct[(ct.mean_pp >= MIN_GAIN_PP) & (ct.holm_p < ALPHA)].contrast.tolist()
    print(f"\nREPLACES n7c_auto: {wins or 'none: the cross-subject lane is closed at the N7c candidate'}")
    df.to_csv(RESULTS_DIR / "n8_pooled_decoder_lee2019.csv")
    ct.to_csv(RESULTS_DIR / "n8_pooled_decoder_contrasts.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
