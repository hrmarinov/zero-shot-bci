"""S1, cheap stack screen: five zero-fit changes to R11, scored on cached predictions. PRE-REGISTERED.

Committed BEFORE any arm of this script is scored. Source: docs/benefit_complexity_review.md (section 3, S1-S5). Every
change below costs no GPU time per user. S4 (per-member weights from out-of-fold predictions) is NOT in this screen: it
needs out-of-fold CNN fits that are not cached. A post hoc look at S1 on R11 probabilities exists (Lee2019 +0.70 pp,
IV-2a +0.27 pp); it is the reason S1 is listed, and it is repeated here as the pre-registered test.

DATA. Primary: Lee2019_MI, 54 subjects, session 1 -> 2 (D2 cache: R11 members, CNN seeds 0-4, FTS seeds 0-1; the TS+LR
member is recomputed). Secondary: IV-2a, 9 subjects (cached R3/R8 members as in R11). R11 must reproduce the D2 and IV-2a
accuracies exactly or the script stops.

ARMS (each differs from R11 in one thing; "family mean" = (CNN3 + TS+LR + FTS2) / 3 on probabilities):
  bal     S1  rescale the probabilities of the test session so predicted class counts are equal (iterative proportional
              fitting, 200 steps, then argmax). Assumes equal class counts in the test session, which both datasets
              state in their design (IV-2a 72 per class, Lee2019 50 per class). No test labels are used.
  geo     S2  geometric mean of the three family probabilities (renormalised) instead of the arithmetic mean.
  cv      S3a TS+LR with C chosen by stratified 5-fold CV accuracy on the TRAINING session, from {0.01, 0.1, 1, 10, 100}.
  tyl     S3b TS+LR with the Tyler covariance estimator instead of OAS (C = 1).
  euc     S5  TS+LR aligned with the Euclidean (arithmetic) mean covariance instead of the Riemannian mean.
Only the TS+LR member changes in cv, tyl and euc; the CNN members are the cached R11 ones.

ENDPOINT. Per-subject accuracy difference arm - R11 on Lee2019. One-sided sign-flip test (Monte Carlo as D2), Holm over
the 5 contrasts.
ADOPT an arm if: Lee2019 mean gain >= +0.5 pp AND Holm p < 0.05 AND IV-2a mean gain >= 0 (9 subjects, no test).
STACK. Apply all adopted arms together (if more than one of cv, tyl, euc is adopted, keep only the one with the largest
Lee2019 gain). Score the stack once. STACK PASS if its Lee2019 gain >= +1.0 pp and one-sided p < 0.05 and its IV-2a gain
>= 0. The effect sizes here are below the 1.9 pp detectable limit of the cohort, so a stack pass is a screen, not a
confirmation; a Zhou2020 opening is justified only for a stack gain >= +3 pp, which this screen is not expected to reach.
REPORTED, NOT FOR DECISION: share of subjects >= 70 %, bottom third by csp_lda, up/down counts.

    .venv\\Scripts\\python experiments\\s1_cheap_stack_screen.py --dry-run   # 3 subjects, plumbing only, no accuracy
    .venv\\Scripts\\python experiments\\s1_cheap_stack_screen.py             # THE run (CPU, minutes)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from scipy.linalg import fractional_matrix_power  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from experiments.d2_lee2019_dev import CACHE, LITERACY, load, sign_flip_p  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.evaluate import holm_adjust  # noqa: E402
from src.pipeline import align_by_group  # noqa: E402

ARMS = ("bal", "geo", "cv", "tyl", "euc")
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
MIN_GAIN_PP, STACK_GAIN_PP, ALPHA = 0.5, 1.0, 0.05
IPF_STEPS = 200


def balance(p: np.ndarray) -> np.ndarray:
    """Iterative proportional fitting of the class marginals to uniform; label-free."""
    q = p.copy()
    for _ in range(IPF_STEPS):
        q = q * (q.shape[0] / q.shape[1]) / q.sum(0, keepdims=True)
        q /= q.sum(1, keepdims=True)
    return q


def geo_mean(parts: list[np.ndarray]) -> np.ndarray:
    g = np.exp(np.mean([np.log(np.clip(p, 1e-12, None)) for p in parts], axis=0))
    return g / g.sum(1, keepdims=True)


def euclid_align(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.empty_like(X)
    for g in np.unique(groups):
        idx = groups == g
        mean_cov = Covariances(estimator="oas").fit_transform(X[idx]).mean(0)
        out[idx] = np.einsum("cd,ndt->nct", np.real(fractional_matrix_power(mean_cov, -0.5)), X[idx])
    return out


def tslr(A_tr: np.ndarray, y_tr: np.ndarray, A_te: np.ndarray, estimator: str = "oas", c: float | str = 1.0) -> np.ndarray:
    cov = Covariances(estimator=estimator)
    ts = TangentSpace(metric="riemann").fit(cov.fit_transform(A_tr))
    F_tr, F_te = ts.transform(cov.transform(A_tr)), ts.transform(cov.transform(A_te))
    if c == "cv":
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        c = max(C_GRID, key=lambda k: cross_val_score(LogisticRegression(C=k, max_iter=5000), F_tr, y_tr, cv=cv).mean())
    return LogisticRegression(C=c, max_iter=5000).fit(F_tr, y_tr).predict_proba(F_te)


def arm_probs(cnn3: np.ndarray, fts2: np.ndarray, X_ts: np.ndarray, groups: np.ndarray, train: np.ndarray, y_tr: np.ndarray,
              stack: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Test-session probabilities for R11 and every arm (or only R11 plus the stack when ``stack`` is given)."""
    A = align_by_group(X_ts, groups)
    ts = {"base": tslr(A[train], y_tr, A[~train]), "cv": None, "tyl": None, "euc": None}
    want = ARMS if not stack else stack
    if "cv" in want:
        ts["cv"] = tslr(A[train], y_tr, A[~train], c="cv")
    if "tyl" in want:
        ts["tyl"] = tslr(A[train], y_tr, A[~train], estimator="tyl")
    if "euc" in want:
        E = euclid_align(X_ts, groups)
        ts["euc"] = tslr(E[train], y_tr, E[~train])
    mean3 = lambda t: (cnn3 + t + fts2) / 3  # noqa: E731
    out = {"r11": mean3(ts["base"])}
    if not stack:
        out.update(bal=balance(out["r11"]), geo=geo_mean([cnn3, ts["base"], fts2]),
                   cv=mean3(ts["cv"]), tyl=mean3(ts["tyl"]), euc=mean3(ts["euc"]))
        return out
    member = next((a for a in ("cv", "tyl", "euc") if a in stack), None)
    parts = [cnn3, ts[member] if member else ts["base"], fts2]
    p = geo_mean(parts) if "geo" in stack else sum(parts) / 3
    out["stack"] = balance(p) if "bal" in stack else p
    return out


def lee_subject(s: int, d2: pd.DataFrame, stack: tuple[str, ...]) -> dict[str, float]:
    data = load(None, s, {}, {}, ["cached"])
    y = data.y[~data.train]
    c = CACHE / "preds" / f"s{s}"
    cnn3 = np.mean([np.load(c / f"cnn_seed{k}_fast.npy") for k in range(3)], 0)
    fts2 = np.mean([np.load(c / f"fts_seed{k}_fast.npy") for k in range(2)], 0)
    probs = arm_probs(cnn3, fts2, data.X_tslr, data.groups, data.train, data.y[data.train], stack)
    acc = {k: float((v.argmax(1) == y).mean()) for k, v in probs.items()}
    if abs(acc["r11"] - d2.loc[s, "r11"]) > 1e-9:
        raise AssertionError(f"s{s}: rebuilt R11 differs from the D2 result")
    return acc


def iv2a_subject(s: int, ref: pd.Series, stack: tuple[str, ...]) -> dict[str, float]:
    import experiments.r11_round3_diverse_members as m
    from src.config import FMAX, FMIN, TMAX, TMIN

    d = m.load_iv2a_subject_window(s, FMIN, FMAX, TMIN, TMAX)
    yt, ye = m.encode_labels(d.y_calib, m.IV2A_CLASSES), m.encode_labels(d.y_eval, m.IV2A_CLASSES)
    X = np.concatenate([d.X_calib, d.X_eval]).astype(np.float32)
    groups = np.r_[np.zeros(len(yt), int), np.ones(len(ye), int)]
    cnn3 = np.mean([np.load(m.CACHE / "r3_preds" / f"s{s}_cnn_ra_seed{k}.npy") for k in range(3)], 0)
    fts2 = np.mean([np.load(m.CACHE / "r8_preds" / f"s{s}_ftsurrogate_seed{k}.npy") for k in range(2)], 0)
    probs = arm_probs(cnn3, fts2, X, groups, groups == 0, yt, stack)
    acc = {k: float((v.argmax(1) == ye).mean()) for k, v in probs.items()}
    if abs(acc["r11"] - ref[s]) > 1e-9:
        raise AssertionError(f"IV-2a s{s}: rebuilt R11 differs from the stored result")
    return acc


def contrasts(pp: pd.DataFrame, arms: tuple[str, ...]) -> pd.DataFrame:
    rows = [{"arm": a, "mean_pp": (pp[a] - pp.r11).mean(), "up": int((pp[a] > pp.r11).sum()),
             "down": int((pp[a] < pp.r11).sum()), "p": sign_flip_p((pp[a] - pp.r11).to_numpy())} for a in arms]
    out = pd.DataFrame(rows)
    out["holm_p"] = holm_adjust(out.p.tolist())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    dry = ap.parse_args().dry_run
    d2 = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    ref = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv")
    ref = ref[(ref.dataset == "IV-2a") & (ref.arm == "fts")].set_index("subject").acc
    lee_subjects, iv_subjects = (range(1, 4), range(1, 3)) if dry else (range(1, 55), range(1, 10))
    lee = pd.DataFrame({s: lee_subject(s, d2, ()) for s in lee_subjects}).T * 100
    iv = pd.DataFrame({s: iv2a_subject(s, ref, ()) for s in iv_subjects}).T * 100
    if dry:
        print(f"DRY RUN OK: arms {list(lee.columns)} on {len(lee)} Lee2019 and {len(iv)} IV-2a subjects (no accuracy printed)")
        return 0
    cl, ci = contrasts(lee, ARMS), contrasts(iv, ARMS)
    cl["iv2a_mean_pp"] = ci.mean_pp.to_numpy()
    cl["adopt"] = (cl.mean_pp >= MIN_GAIN_PP) & (cl.holm_p < ALPHA) & (cl.iv2a_mean_pp >= 0)
    print("\n=== S1 cheap stack screen (Lee2019 n = 54; IV-2a n = 9), accuracy gain over R11 ===")
    print(f"R11: Lee2019 {lee.r11.mean():.2f}, IV-2a {iv.r11.mean():.2f}")
    print(cl.round(4).to_string(index=False))
    third = d2.csp_lda.sort_values().index[: len(lee) // 3]
    print(f"R11 share >= 70 %: {(lee.r11 >= 100 * LITERACY).mean():.1%}; bottom third {lee.loc[third, 'r11'].mean():.2f}")
    for a in ARMS:
        print(f"  {a}: share >= 70 % {(lee[a] >= 100 * LITERACY).mean():.1%}, bottom third {lee.loc[third, a].mean():.2f}")
    adopted = tuple(cl.loc[cl.adopt, "arm"])
    members = [a for a in adopted if a in ("cv", "tyl", "euc")]
    if len(members) > 1:
        keep = max(members, key=lambda a: cl.set_index("arm").loc[a, "mean_pp"])
        adopted = tuple(a for a in adopted if a not in members or a == keep)
    print(f"\nADOPTED: {list(adopted) or 'none'}")
    verdict = "no stack (nothing adopted)"
    if adopted:
        sl = pd.DataFrame({s: lee_subject(s, d2, adopted) for s in lee.index}).T * 100
        si = pd.DataFrame({s: iv2a_subject(s, ref, adopted) for s in iv.index}).T * 100
        gl, gi = (sl.stack - sl.r11), (si.stack - si.r11)
        p = sign_flip_p(gl.to_numpy())
        print(f"STACK {list(adopted)}: Lee2019 {gl.mean():+.2f} pp ({(gl > 0).sum()} up / {(gl < 0).sum()} down), "
              f"p = {p:.4f}; IV-2a {gi.mean():+.2f} pp")
        verdict = "STACK PASS" if gl.mean() >= STACK_GAIN_PP and p < ALPHA and gi.mean() >= 0 else "STACK FAIL"
    print(f"\nTHE SCREEN: {verdict}")
    lee.to_csv(RESULTS_DIR / "s1_cheap_stack_lee2019.csv")
    cl.to_csv(RESULTS_DIR / "s1_cheap_stack_contrasts.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
