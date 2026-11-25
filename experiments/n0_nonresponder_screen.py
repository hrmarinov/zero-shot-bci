"""N0, non-responder flag screen: can a flag identify weak subjects better than training-session CV? PRE-REGISTERED.

Committed BEFORE any flag of this script is scored. Source: docs/benefit_complexity_review.md and the D2 finding that the
bottom third of Lee2019 sits at chance for every member (a new source of information is needed, not a better classifier).
CPU only, cached members only, no new network fit. Zhou2020 stays sealed, BNCI2015_001 is not used. No test labels are used
to build any flag (they are used only to score the endpoints).

TARGET. Non-responder = session-2 accuracy of R11 not above the one-sided binomial chance bound (p < 0.05) for the number
of test trials n (Lee2019: n = 100, bound 58 %, i.e. "non-responder" if accuracy <= 58 %; computed with scipy binom.ppf).
Cohorts: Lee2019 n = 54 (primary, the only one with inference); IV-2a n = 9 and IV-2b n = 9 (descriptive only).
R11 must reproduce the stored D2 / IV-2a accuracies or the script stops.

FLAGS (higher flag value = predicted BETTER responder; AUC is computed for "responder", equivalently 1 - AUC for non-responder
after sign alignment; all flags use the training session or unlabelled test trials only):
  N1 baseline  stratified 5-fold CV accuracy of TS+LR (OAS, C = 1) on the aligned training session (seed 0), plus a 200-
               permutation p against chance (reported, not a flag).
  N2 geometry  classDistinct and classStability on the aligned training covariances (OAS), pyRiemann distances (Lotte & Jeunet
               2018, Ivanov & Chau 2023): classDistinct = d(M_1, M_2) / mean within-class dispersion, classStability =
               1 / mean within-class dispersion (dispersion = mean Riemannian distance of trials to their class mean). Two
               flags: N2a distinct, N2b stability.
  N3 spectrum  mu/beta peak over C3, Cz, C4 on the training-session trials (CNN band input, 0-4 s): Welch PSD averaged over
               trials and channels, 1/f line fitted in log-log on 4-38 Hz excluding 8-30 Hz; flag N3a = peak-to-fit SNR in dB
               (max over 8-30 Hz of log10 PSD minus fitted line, x10); N3b = the 1/f slope (flatter = lower); both reported.
  N4 test-time ensemble disagreement / uncertainty (label-free, session-2 trials): N4a = minus mean pairwise disagreement of
               the five R11 members (CNN seeds 0-2, TS+LR, FTS-mean) on argmax; N4b = minus mean predictive entropy of R11;
               N4c = mean top-two margin of R11.
  N5 conformal split-conformal (LAC score 1 - p_true) per subject, alpha = 0.10, calibrated on out-of-fold training-session
               TS+LR probabilities (5-fold), applied to R11 session-2 probabilities. Flag = minus mean set size; empirical
               coverage on session 2 reported (undercoverage from session shift expected and stated).
  N6 pre-cue SMR baseline: testable only if the cached epochs contain a pre-cue period. Cached epochs start at the cue (tmin = 0),
               so N6 is reported as "not testable from caches"; raw data is NOT re-extracted.
ENDPOINTS per flag: Spearman rho with session-2 R11 accuracy, AUC for responder vs non-responder, PPV of the "flag = non-
responder" call when the lowest-flag subjects are called non-responders in the number equal to the observed prevalence; bootstrap
95 % CI over subjects (2000 resamples, seed 0). Paired AUC difference against N1 with its bootstrap CI (same resamples).
RISK-COVERAGE: for R11 and each cached member (CSP+LDA, TS+LR, CNN3), accuracy over the subjects kept at 100 / 80 / 60 % coverage
when subjects are ranked by each flag, against a random-abstention control (mean of 1000 random subsets) and the N1 flag.
Dropping subjects raises mean accuracy mechanically; coverage is always shown next to accuracy.
KILL RULE: a flag N2-N6 is "better than N1" if AUC(flag) - AUC(N1) >= +0.05 AND the bootstrap 95 % CI of the difference excludes
zero AND the Holm-adjusted one-sided bootstrap p (over N2a, N2b, N3a, N3b, N4a, N4b, N4c, N5, the testable flags) < 0.05. If no flag
qualifies the track ends with the finding "training CV is enough". Primary decision on Lee2019 only.
DESCRIPTIVE: which flags agree on the 12 weakest subjects (by R11 accuracy: overlap of each flag's 12 lowest); whether IV-2a
subjects 2 and 6 (weak for R11, 73-75 % for TCFormer) are in the bottom 2 of each flag within IV-2a.

    .venv\Scripts\python experiments\n0_nonresponder_screen.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from pyriemann.utils.distance import distance_riemann  # noqa: E402
from pyriemann.utils.mean import mean_riemann  # noqa: E402
from scipy.signal import welch  # noqa: E402
from scipy.stats import binom, rankdata, spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score  # noqa: E402

from experiments.d2_lee2019_dev import CACHE, load  # noqa: E402
from src.baselines import make_csp_lda  # noqa: E402
from src.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.evaluate import holm_adjust  # noqa: E402
from src.pipeline import align_by_group, load_subject_data, tslr_member  # noqa: E402

FS, N_BOOT, N_PERM, ALPHA_CP = 250, 2000, 200, 0.10
MIN_DAUC, ALPHA = 0.05, 0.05
MOTOR = {"lee": (7, 9, 11), "iv2a": (7, 9, 11), "iv2b": (0, 1, 2)}  # C3, Cz, C4 in each cached channel order
TEST_FLAGS = ("N2a", "N2b", "N3a", "N3b", "N4a", "N4b", "N4c", "N5")
COVERAGES = (1.0, 0.8, 0.6)
CV = StratifiedKFold(5, shuffle=True, random_state=0)


def cv_features(A: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C = Covariances(estimator="oas").fit_transform(A[train])
    return C, TangentSpace(metric="riemann").fit(C).transform(C)


def n1_n2(A: np.ndarray, y: np.ndarray, train: np.ndarray) -> dict[str, float]:
    C, F = cv_features(A, train)
    yt = y[train]
    lr = LogisticRegression(C=1.0, max_iter=5000)
    cv_acc = float(cross_val_score(lr, F, yt, cv=CV).mean())
    rng = np.random.default_rng(0)
    perm = [cross_val_score(lr, F, rng.permutation(yt), cv=CV).mean() for _ in range(N_PERM)]
    classes = np.unique(yt)
    means = [mean_riemann(C[yt == k]) for k in classes]
    disp = np.mean([distance_riemann(C[yt == k], M).mean() for k, M in zip(classes, means)])
    between = np.mean([float(distance_riemann(a, b)) for a, b in combinations(means, 2)])
    return {"N1": cv_acc, "N1_perm_p": float((1 + sum(p >= cv_acc for p in perm)) / (1 + N_PERM)),
            "N2a": float(between / disp), "N2b": float(1 / disp)}


def n3(X_cnn: np.ndarray, train: np.ndarray, chans: tuple[int, ...]) -> dict[str, float]:
    f, P = welch(X_cnn[train][:, list(chans)], fs=FS, nperseg=FS, noverlap=FS // 2, axis=-1)
    psd = np.log10(P.mean((0, 1)))
    fit_m = ((f >= 4) & (f < 8)) | ((f > 30) & (f <= 38))
    slope, icpt = np.polyfit(np.log10(f[fit_m]), psd[fit_m], 1)
    band = (f >= 8) & (f <= 30)
    snr = 10 * (psd[band] - (icpt + slope * np.log10(f[band]))).max()
    return {"N3a": float(snr), "N3b": float(slope)}


def n4_n5(members: list[np.ndarray], r11: np.ndarray, ts_oof: np.ndarray, y_tr: np.ndarray, y_te: np.ndarray) -> dict[str, float]:
    votes = [m.argmax(1) for m in members]
    dis = np.mean([(a != b).mean() for a, b in combinations(votes, 2)])
    ent = -(r11 * np.log(np.clip(r11, 1e-12, None))).sum(1).mean()
    top2 = np.sort(r11, 1)[:, -2:]
    n = len(y_tr)
    s_cal = 1 - ts_oof[np.arange(n), y_tr]
    q = np.quantile(s_cal, min(1.0, np.ceil((n + 1) * (1 - ALPHA_CP)) / n), method="higher")
    sets = (1 - r11) <= q
    return {"N4a": float(-dis), "N4b": float(-ent), "N4c": float((top2[:, 1] - top2[:, 0]).mean()),
            "N5": float(-sets.sum(1).mean()), "N5_cov": float(sets[np.arange(len(y_te)), y_te].mean())}


def lee_subject(s: int, d2: pd.DataFrame) -> tuple[dict, dict]:
    data = load(None, s, {}, {}, ["cached"])
    c = CACHE / "preds" / f"s{s}"
    cnn = [np.load(c / f"cnn_seed{k}_fast.npy") for k in range(3)]
    fts2 = np.mean([np.load(c / f"fts_seed{k}_fast.npy") for k in range(2)], 0)
    return _subject("lee", data, cnn, fts2, d2.loc[s, "r11"], d2.loc[s, "csp_lda"])


def iv2a_subject(s: int, ref: pd.Series) -> tuple[dict, dict]:
    data, _ = load_subject_data("IV-2a", s)
    cnn = [np.load(DATA_DIR / "cache" / "r3_preds" / f"s{s}_cnn_ra_seed{k}.npy") for k in range(3)]
    fts2 = np.mean([np.load(DATA_DIR / "cache" / "r8_preds" / f"s{s}_ftsurrogate_seed{k}.npy") for k in range(2)], 0)
    return _subject("iv2a", data, cnn, fts2, ref[s], None)


def _subject(tag: str, data, cnn: list[np.ndarray], fts2: np.ndarray, r11_ref: float, csp_acc: float | None) -> tuple[dict, dict]:
    tr = data.train
    yt, ye = data.y[tr], data.y[~tr]
    A = align_by_group(data.X_tslr, data.groups)
    ts = tslr_member(data)
    cnn3 = np.mean(cnn, 0)
    r11 = (cnn3 + ts + fts2) / 3
    acc = lambda p: float((p.argmax(1) == ye).mean())  # noqa: E731
    if abs(acc(r11) - r11_ref) > 1e-9:
        raise AssertionError(f"{tag}: rebuilt R11 {acc(r11)} differs from stored {r11_ref}")
    _, F = cv_features(A, tr)
    ts_oof = cross_val_predict(LogisticRegression(C=1.0, max_iter=5000), F, yt, cv=CV, method="predict_proba")
    flags = {**n1_n2(A, data.y, tr), **n3(data.X_cnn, tr, MOTOR[tag]), **n4_n5([*cnn, ts, fts2], r11, ts_oof, yt, ye)}
    if csp_acc is None:
        csp_acc = (make_csp_lda().fit(A[tr], yt).predict(A[~tr]) == ye).mean()
    return flags, {"r11": acc(r11), "csp_lda": float(csp_acc), "tslr": acc(ts), "cnn3": acc(cnn3), "n_test": int(len(ye))}


def iv2b_subject(s: int, ref: pd.Series) -> tuple[dict, dict]:
    """Only the training-session flags N1-N3 (no R11 members are rebuilt); accuracy from the stored R11 result."""
    data, _ = load_subject_data("IV-2b", s)
    A = align_by_group(data.X_tslr, data.groups)
    return {**n1_n2(A, data.y, data.train), **n3(data.X_cnn, data.train, MOTOR["iv2b"])}, \
        {"r11": float(ref[s]), "n_test": int((~data.train).sum())}


# ------------------------------------------------------------------ endpoints
def auc(flag: np.ndarray, resp: np.ndarray) -> float:
    """P(flag of a random responder > flag of a random non-responder), ties 1/2."""
    if resp.all() or not resp.any():
        return np.nan
    r = rankdata(flag)
    n1, n0 = resp.sum(), (~resp).sum()
    return float((r[resp].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ppv(flag: np.ndarray, resp: np.ndarray) -> float:
    """Call the lowest-flag subjects non-responders (as many as the observed prevalence; ties all called)."""
    k = int((~resp).sum())
    if k == 0:
        return np.nan
    called = flag <= np.sort(flag)[k - 1]
    return float((~resp[called]).mean())


def endpoints(df: pd.DataFrame, flags: list[str], rng: np.random.Generator, boot: bool) -> pd.DataFrame:
    resp, acc = df.responder.to_numpy(), df.r11.to_numpy()
    idx = rng.integers(0, len(df), (N_BOOT, len(df))) if boot else None
    rows, base_boot = [], None
    for f in flags:
        x = df[f].to_numpy().astype(float)
        row = {"flag": f, "rho": spearmanr(x, acc)[0], "auc": auc(x, resp), "ppv": ppv(x, resp)}
        if boot:
            b = np.array([[spearmanr(x[i], acc[i])[0], auc(x[i], resp[i]), ppv(x[i], resp[i])] for i in idx])
            for j, m in enumerate(("rho", "auc", "ppv")):
                row[f"{m}_lo"], row[f"{m}_hi"] = np.nanpercentile(b[:, j], [2.5, 97.5])
            if f == "N1":
                base_boot, base_auc = b[:, 1], row["auc"]
            else:
                d = b[:, 1] - base_boot
                row["dauc"] = row["auc"] - base_auc
                row["dauc_lo"], row["dauc_hi"] = np.nanpercentile(d, [2.5, 97.5])
                row["p_one_sided"] = float((1 + np.nansum(d <= 0)) / (1 + np.isfinite(d).sum()))
        rows.append(row)
    out = pd.DataFrame(rows)
    if boot:
        m = out.flag.isin(TEST_FLAGS)
        out.loc[m, "holm_p"] = holm_adjust(out.loc[m, "p_one_sided"].tolist())
        out["beats_n1"] = (out.dauc >= MIN_DAUC) & (out.dauc_lo > 0) & (out.holm_p < ALPHA)
    return out


def risk_coverage(df: pd.DataFrame, flags: list[str], members: list[str], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for m in members:
        a = df[m].to_numpy()
        for cov in COVERAGES:
            k = round(cov * len(df))
            rand = np.array([a[rng.choice(len(df), k, replace=False)].mean() for _ in range(1000)])
            for f in flags:
                keep = np.argsort(-df[f].to_numpy(), kind="stable")[:k]  # highest flag value kept
                rows.append({"member": m, "coverage": cov, "flag": f, "acc": 100 * a[keep].mean(),
                             "random_acc": 100 * rand.mean(), "random_sd": 100 * rand.std()})
    return pd.DataFrame(rows)


def lee_extras(df: pd.DataFrame, flags: list[str], ep: pd.DataFrame, rng: np.random.Generator) -> None:
    print(f"N5 empirical coverage (target {1 - ALPHA_CP:.0%}): mean {df.N5_cov.mean():.3f}, "
          f"non-responders {df.loc[~df.responder, 'N5_cov'].mean():.3f}")
    rc = risk_coverage(df, flags, ["r11", "csp_lda", "tslr", "cnn3"], rng)
    rc.to_csv(RESULTS_DIR / "n0_nonresponder_risk_coverage.csv", index=False)
    r = rc[rc.member == "r11"]
    print("R11 accuracy (%) by coverage, subjects ranked by each flag:")
    print(r.pivot_table(index="flag", columns="coverage", values="acc").round(2).to_string())
    print("random-abstention control:", r.groupby("coverage").random_acc.first().round(2).to_dict())
    weak = set(df.r11.nsmallest(12).index)
    low = {f: set(df[f].nsmallest(12).index) for f in flags}
    agree = pd.Series({f: len(low[f] & weak) for f in flags}, name="overlap_with_12_weakest_by_r11")
    agree.to_csv(RESULTS_DIR / "n0_nonresponder_weak12_agreement.csv")
    named = {int(s): sorted(f for f in flags if s in low[f]) for s in sorted(weak)}
    print("overlap of each flag's 12 lowest with the 12 weakest by R11:\n" + agree.to_string())
    print("flags naming each of the 12 weakest subjects:", {s: len(v) for s, v in named.items()})
    ok = list(ep.loc[ep.get("beats_n1", False) == True, "flag"])  # noqa: E712
    print(f"\nKILL RULE: flags beating N1: {ok or 'none: training CV is enough'}")


def main() -> int:
    d2 = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    r11 = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv")
    ref = lambda ds: r11[(r11.dataset == ds) & (r11.arm == "fts")].set_index("subject").acc  # noqa: E731
    cohorts = {"lee": (range(1, 55), lambda s: lee_subject(s, d2), "Lee2019", 2),
               "iv2a": (range(1, 10), lambda s: iv2a_subject(s, ref("IV-2a")), "IV-2a", 4),
               "iv2b": (range(1, 10), lambda s: iv2b_subject(s, ref("IV-2b")), "IV-2b", 2)}
    rng = np.random.default_rng(0)
    for tag, (ids, fn, name, n_cls) in cohorts.items():
        recs = {s: fn(s) for s in ids}
        df = pd.DataFrame({s: {**f, **m} for s, (f, m) in recs.items()}).T
        df.index.name = "subject"
        n_t = int(df.n_test.iloc[0])
        bound = binom.ppf(0.95, n_t, 1 / n_cls) / n_t
        df["responder"] = df.r11 > bound
        flags = ["N1", *(f for f in TEST_FLAGS if f in df.columns)]
        print(f"\n=== {name}: n = {len(df)}, {n_t} test trials, chance bound {100 * bound:.1f} %, "
              f"non-responders {(~df.responder).sum()}, R11 mean {100 * df.r11.mean():.2f} ===")
        ep = endpoints(df, flags, rng, boot=(tag == "lee"))
        print(ep.round(3).to_string(index=False))
        df.to_csv(RESULTS_DIR / f"n0_nonresponder_{tag}_flags.csv")
        ep.to_csv(RESULTS_DIR / f"n0_nonresponder_{tag}_endpoints.csv", index=False)
        if tag == "lee":
            lee_extras(df, flags, ep, rng)
        elif tag == "iv2a":
            ranks = pd.DataFrame({f: rankdata(df[f], method="min") for f in flags}, index=df.index)
            print("IV-2a rank of each flag (1 = lowest flag value), subjects 2 and 6:")
            print(ranks.loc[[2, 6]].to_string())
            print("IV-2a R11 accuracy %:", (100 * df.r11).round(1).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
