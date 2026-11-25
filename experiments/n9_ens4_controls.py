"""N9, equal-size control and IV-2a replication for the N8 four-member ensemble. PRE-REGISTERED.

Committed BEFORE any arm of this script is scored. N8 (experiments/n8_pooled_decoder_screen.py) found ens4 = (CNN3 + TS+LR[N7c] +
FTS2 + pool[auto_w]) / 4 at R11 +2.94 pp over base_I on Lee2019 (+1.04 pp over N7c auto, Holm p 0.021), but registered no
equal-size control (the gain may be member count / diversity in general) and no replication. CPU only, cached members only;
Zhou2020 sealed, BNCI2015_001 unused. Same features, leave-one-subject-out protocol and R11 stack as N7 / N7c / N8.

PART A, Lee2019 n = 54 (inference). ENS4_CTRL = (CNN3 + TS+LR[N7c auto] + FTS2 + CNN45) / 4, CNN45 = mean softmax of the cached base
CNN seeds 3-4 (a fourth member that is subject-specific and equally large). lam (N7c) and w (N8) are chosen exactly as there; ens4 and
n7c_auto must reproduce results/n8_pooled_decoder_lee2019.csv exactly or the script stops.
  PRIMARY: ens4 - ens4_ctrl, one-sided sign-flip Monte Carlo (as D2), a single contrast. The pooled member's gain is credited to
  CROSS-SUBJECT INFORMATION only if the mean is >= +0.5 pp with p < 0.05; otherwise to member count / diversity in general.
  REPORTED: ens4_ctrl - n7c_auto (the effect of a fourth member of the old kind), ens4 - base_I, transfer-limited and bottom third.
PART B, IV-2a n = 9 (descriptive replication, NO inference: 9 subjects, 4 classes, 22 channels). Library = the 8 other subjects (both
labelled sessions, 576 trials each); features as in N7 (per-session alignment, OAS, tangent space at I, 253-d); lam grid {0, .2, .4,
.6, .8, 1} chosen on the 8 others (nested), w grid {0, 1, 5, 20, 100} chosen on the 7 other validators (each o scored with the pool of
the subjects other than o and s); arms base_I, n7c_auto, pool[auto_w], ens4, ens4_ctrl with CNN45 = cached r3_preds cnn_ra seeds 3-4;
R11 base must reproduce the stored IV-2a R11 accuracy exactly. IV-2b is not tested (a different recording design and 3 channels).
  REPLICATION CONSISTENT iff ens4 - base_I mean >= +0.5 pp AND ens4 > base_I for >= 7 of the 9 subjects AND ens4 - ens4_ctrl mean > 0.
DECISION. ens4 stays the candidate (and is worth packaging as a pipeline option) only if Part A credits cross-subject information and
Part B is consistent. If A fails, the N8 gain is generic member count and ens4 is dropped; if only B fails, the result is Lee2019-
specific and stays a development-cohort screen. A Zhou2020 opening still needs a candidate gain >= +3 pp over base_I.

    .venv\Scripts\python experiments\n9_ens4_controls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binom  # noqa: E402

from experiments.d2_lee2019_dev import CACHE, sign_flip_p  # noqa: E402
from experiments.n7_drift_forecast_screen import fit_predict, subject_features, tangent_I  # noqa: E402
from experiments.n7c_shrinkage_strength import GRID as LAMS  # noqa: E402
from experiments.n8_pooled_decoder_screen import WEIGHTS, acc, pool_proba, r11_pred  # noqa: E402
from src.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.pipeline import align_by_group, load_subject_data, tslr_member  # noqa: E402

N_VAL, MIN_GAIN_PP, ALPHA, MIN_UP = 10, 0.5, 0.05, 7


def class_means(F: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    return np.stack([F[y == c].mean(0) for c in range(k)])


def lee_set() -> dict:
    S = {}
    for s in range(1, 55):
        v = subject_features(s)
        c = CACHE / "preds" / f"s{s}"
        v["cnn45"] = np.mean([np.load(c / f"cnn_seed{k}_fast.npy") for k in (3, 4)], 0)
        S[s] = v
    return S


def iv2a_set(ref: pd.Series) -> dict:
    S = {}
    for s in range(1, 10):
        data, _ = load_subject_data("IV-2a", s)
        A = align_by_group(data.X_tslr, data.groups)
        F, tr = tangent_I(A, np.eye(A.shape[1])), data.train
        r3, r8 = DATA_DIR / "cache" / "r3_preds", DATA_DIR / "cache" / "r8_preds"
        v = {"F_tr": F[tr], "y_tr": data.y[tr], "F_te": F[~tr], "y_te": data.y[~tr],
             "cnn3": np.mean([np.load(r3 / f"s{s}_cnn_ra_seed{k}.npy") for k in range(3)], 0),
             "cnn45": np.mean([np.load(r3 / f"s{s}_cnn_ra_seed{k}.npy") for k in (3, 4)], 0),
             "fts2": np.mean([np.load(r8 / f"s{s}_ftsurrogate_seed{k}.npy") for k in range(2)], 0)}
        if abs(acc(v, r11_pred(v, tslr_member(data))) - ref[s]) > 1e-9:
            raise AssertionError(f"IV-2a s{s}: rebuilt R11 differs from the stored result")
        S[s] = v
    return S


def run(S: dict, n_cls: int, val_order: np.ndarray | None, tag: str) -> pd.DataFrame:
    n = len(S)
    M1 = {s: class_means(v["F_tr"], v["y_tr"], n_cls) for s, v in S.items()}
    total = sum(M1.values())
    rows = {}
    for s, v in S.items():
        others = [o for o in S if o != s]
        sel = np.zeros(len(LAMS))
        for o in others:
            mu = (total - M1[o] - M1[s]) / (n - 2)
            for i, lam in enumerate(LAMS):
                sh = None if lam == 0 else lam * (mu - M1[o])
                sel[i] += acc(S[o], r11_pred(S[o], fit_predict(S[o]["F_tr"], S[o]["y_tr"], sh, S[o]["F_te"])))
        lam = float(LAMS[int(np.argmax(sel))])
        mu_s = (total - M1[s]) / (n - 1)
        ts_auto = fit_predict(v["F_tr"], v["y_tr"], None if lam == 0 else lam * (mu_s - M1[s]), v["F_te"])
        vals = others if val_order is None else [o for o in val_order if o != s][:N_VAL]
        wsel = np.zeros(len(WEIGHTS))
        for o in vals:
            for i, w in enumerate(WEIGHTS):
                wsel[i] += acc(S[o], r11_pred(S[o], pool_proba(S, {o, s}, S[o], w, S[o])))
        w = WEIGHTS[int(np.argmax(wsel))]
        p_pool = pool_proba(S, {s}, v, w, v)
        a = lambda p: 100 * acc(v, p)  # noqa: E731
        rows[s] = {"base_I": a(r11_pred(v, fit_predict(v["F_tr"], v["y_tr"], None, v["F_te"]))), "n7c_auto": a(r11_pred(v, ts_auto)),
                   "auto_w": a(r11_pred(v, p_pool)),
                   "ens4": a((v["cnn3"] + ts_auto + v["fts2"] + p_pool) / 4),
                   "ens4_ctrl": a((v["cnn3"] + ts_auto + v["fts2"] + v["cnn45"]) / 4), "lam": lam, "w": w}
        print(f"{tag} s{s}: lam {lam}, w {w}, " + ", ".join(f"{k} {rows[s][k]:.0f}" for k in ("base_I", "n7c_auto", "ens4", "ens4_ctrl")), flush=True)
    return pd.DataFrame(rows).T


def summarize(df: pd.DataFrame, name: str, infer: bool) -> None:
    print(f"\n=== {name} (n = {len(df)}) ===")
    for a in ("base_I", "n7c_auto", "auto_w", "ens4_ctrl", "ens4"):
        g = df[a] - df.base_I
        print(f"{a:10s} R11 {df[a].mean():.2f}  gain over base_I {g.mean():+.2f} pp ({(g > 0).sum()} up/{(g < 0).sum()} down)")
    for a, b in (("ens4", "ens4_ctrl"), ("ens4_ctrl", "n7c_auto"), ("ens4", "n7c_auto")):
        d = df[a] - df[b]
        p = f", p = {sign_flip_p(d.to_numpy()):.4f}" if infer else ""
        print(f"  {a} - {b}: {d.mean():+.2f} pp ({(d > 0).sum()} up/{(d < 0).sum()} down){p}")


def main() -> int:
    n8 = pd.read_csv(RESULTS_DIR / "n8_pooled_decoder_lee2019.csv", index_col=0)
    r11 = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv")
    lee = run(lee_set(), 2, np.random.default_rng(0).permutation(np.arange(1, 55)), "lee")
    for a in ("n7c_auto", "ens4"):
        if (lee[a] - n8[a]).abs().max() > 1e-9:
            raise AssertionError(f"{a} does not reproduce N8")
    print("\nreproduces N8 ens4 and n7c_auto")
    summarize(lee, "Part A, Lee2019", True)
    d = lee.ens4 - lee.ens4_ctrl
    a_ok = d.mean() >= MIN_GAIN_PP and sign_flip_p(d.to_numpy()) < ALPHA
    iv = run(iv2a_set(r11[(r11.dataset == "IV-2a") & (r11.arm == "fts")].set_index("subject").acc), 4, None, "iv2a")
    summarize(iv, "Part B, IV-2a (descriptive)", False)
    gb = iv.ens4 - iv.base_I
    b_ok = gb.mean() >= MIN_GAIN_PP and (gb > 0).sum() >= MIN_UP and (iv.ens4 - iv.ens4_ctrl).mean() > 0
    print(f"\nPART A credits cross-subject information: {a_ok}; PART B replication consistent: {b_ok}")
    print("ens4 stays the candidate:", a_ok and b_ok)
    lee.to_csv(RESULTS_DIR / "n9_ens4_controls_lee2019.csv")
    iv.to_csv(RESULTS_DIR / "n9_ens4_controls_iv2a.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
