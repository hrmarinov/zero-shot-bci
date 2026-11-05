"""X1, cross-paradigm session transfer: does our recipe (per-session Riemannian alignment + a diverse
second member) also help P300 and SSVEP decoders transfer from one session to the next? PRE-REGISTERED.

Committed BEFORE GuttmannFlury2025 is downloaded; nothing here was tuned on it. Opened once. An
implementation bug found after the run may be fixed in a separate commit that leaves the protocol unchanged.
CPU only.

DATASET (MOABB GuttmannFlury2025, Sci. Data 12:587): 31 subjects, 62 EEG channels + 1 vertical EOG, 1000 Hz.
Every session runs MI, ME, SSVEP and the P300 speller in random order. Only the 17 subjects with >= 2
sessions are used: 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 23, 25. Train on the first session,
test on the second; a third session is not used. LIMITATIONS fixed in advance: the paper does not report the
interval between sessions (they are separate recordings, not necessarily separate days); in SSVEP the
four targets sit in the four screen quadrants, so gaze direction is class-locked by design (our narrow
filter-bank bands, 9.5-13.5 Hz on occipital channels, exclude the low-frequency ocular content).

P300 (4-letter speller, flash-level Target vs NonTarget; metric ROC AUC on the test session):
  epochs 0-0.8 s after each flash, 1-24 Hz, resampled to 128 Hz, all EEG channels.
  p_ts      XdawnCovariances(4 filters, 'lwf') -> tangent space -> LR (C=1), no alignment (MOABB standard).
  p_ts_ra   the same, with each session's xDAWN covariances recentred to identity by that session's own
            Riemannian mean (label-free, per session; our recipe's alignment step).
  p_lda     xDAWN (4 filters) spatial filtering -> flattened time course -> shrinkage LDA (classic P300).
  p_ens     mean of p_ts_ra and p_lda probabilities (the "diverse second member").
SSVEP (4 classes, 10/11/12/13 Hz; metric accuracy):
  epochs 0.5-4.5 s after stimulus onset, resampled to 250 Hz, 8 occipital channels
  (PO7, PO3, POz, PO4, PO8, O1, Oz, O2).
  s_ts      filter bank 9.5-10.5 / 10.5-11.5 / 11.5-12.5 / 12.5-13.5 Hz stacked along channels (32 rows)
            -> OAS covariance -> tangent space -> LR (C=1), no alignment.
  s_ts_ra   the same with per-session Riemannian alignment of the stacked signal (src.pipeline.align_by_group).
  s_cca     CCA correlations with sine/cosine references (3 harmonics) on 8-40 Hz signals -> LR (C=1) on the
            4 correlations (training only calibrates the CCA scores).
  s_ens     mean of s_ts_ra and s_cca probabilities.

HYPOTHESES (one-sided exact sign-flip over 17 subjects, Holm over the four):
  P1 p_ts_ra - p_ts   > 0   alignment helps P300 session transfer.
  P2 p_ens   - p_ts_ra > 0  a diverse second member helps P300.
  S1 s_ts_ra - s_ts   > 0   alignment helps SSVEP session transfer.
  S2 s_ens   - s_ts_ra > 0  a diverse second member helps SSVEP.
Reported: every arm's mean, and each contrast's mean, median, up/down and 95% bootstrap CI.

CHECKS: P300 and SSVEP each give exactly two sessions per used subject (first two kept); both classes /
all four classes present in each session; --dry-run runs the whole path on MOABB FakeDatasets.

    .venv\\Scripts\\python experiments\\x1_cross_paradigm_guttmann.py --dry-run
    .venv\\Scripts\\python experiments\\x1_cross_paradigm_guttmann.py      # downloads ~35 GB (17 subjects x 2 paradigms)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from moabb.paradigms import P300, SSVEP, FilterBankSSVEP  # noqa: E402
from pyriemann.estimation import Covariances, XdawnCovariances  # noqa: E402
from pyriemann.geometry.mean import mean_riemann  # noqa: E402
from pyriemann.spatialfilters import Xdawn  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from scipy.linalg import fractional_matrix_power  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.evaluate import holm_adjust, sign_flip_permutation_test  # noqa: E402
from src.pipeline import align_by_group  # noqa: E402

mne.set_log_level("ERROR")
SUBJECTS = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 23, 25]
OCCIPITAL = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]  # names as in the loaded raw files
SSVEP_CLASSES = ["10.0", "11.0", "12.0", "13.0"]
FB = [(9.5, 10.5), (10.5, 11.5), (11.5, 12.5), (12.5, 13.5)]
HYPOTHESES = {"P1": ("p_ts_ra", "p_ts"), "P2": ("p_ens", "p_ts_ra"),
              "S1": ("s_ts_ra", "s_ts"), "S2": ("s_ens", "s_ts_ra")}
N_BOOT = 10_000
# Deviation recorded before any SSVEP result for this subject was computed: the Zenodo archive S12.zip (record
# 18978288, md5 matches Zenodo) has its Sess01 members before the start of the file (negative offsets), so the
# first SSVEP session of subject 12 cannot be read. Subject 12 is dropped from SSVEP only (n = 16 there).
SSVEP_UNAVAILABLE = {12: "hosted S12.zip is truncated: Sess01 BDF/sync unreadable"}


def first_two(meta: pd.DataFrame, tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(keep, train, sessions) masks for the first two sessions in sorted order."""
    sessions = sorted(meta.session.unique())
    if len(sessions) < 2:
        raise AssertionError(f"{tag}: needs >= 2 sessions, got {sessions}")
    sess = meta.session.to_numpy()
    keep = np.isin(sess, sessions[:2])
    return keep, sess[keep] == sessions[0], sess[keep]


def recentre(C: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Recentre each group's SPD matrices to identity with that group's own Riemannian mean."""
    out = np.empty_like(C)
    for g in np.unique(groups):
        W = fractional_matrix_power(mean_riemann(C[groups == g]), -0.5).real
        out[groups == g] = W @ C[groups == g] @ W
    return out


def ts_lr(Ct: np.ndarray, yt: np.ndarray, Ce: np.ndarray) -> np.ndarray:
    ts = TangentSpace(metric="riemann").fit(Ct)
    return LogisticRegression(C=1.0, max_iter=5000).fit(ts.transform(Ct), yt).predict_proba(ts.transform(Ce))


# ---------------------------------------------------------------- P300

def run_p300(dataset, s: int, p300_kw: dict) -> dict:
    X, labels, meta = P300(**p300_kw).get_data(dataset=dataset, subjects=[s])
    keep, train, sess = first_two(meta, f"P300 s{s}")
    X, y = X[keep], (labels[keep] == "Target").astype(int)
    for sname in np.unique(sess):
        if len(np.unique(y[sess == sname])) != 2:
            raise AssertionError(f"P300 s{s}: session {sname} lacks a class")
    xc = XdawnCovariances(nfilter=4, estimator="lwf").fit(X[train], y[train])
    C = xc.transform(X)
    p_ts = ts_lr(C[train], y[train], C[~train])[:, 1]
    Cr = recentre(C, sess)
    p_ts_ra = ts_lr(Cr[train], y[train], Cr[~train])[:, 1]
    xd = Xdawn(nfilter=4).fit(X[train], y[train])
    F = xd.transform(X).reshape(len(X), -1)
    p_lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F[train], y[train]).predict_proba(F[~train])[:, 1]
    ye = y[~train]
    return {"p_ts": roc_auc_score(ye, p_ts), "p_ts_ra": roc_auc_score(ye, p_ts_ra),
            "p_lda": roc_auc_score(ye, p_lda), "p_ens": roc_auc_score(ye, (p_ts_ra + p_lda) / 2),
            "p_n_test": int(len(ye))}


# ---------------------------------------------------------------- SSVEP

def cca_scores(X: np.ndarray, sfreq: float, n_harm: int = 3) -> np.ndarray:
    """Largest canonical correlation between each trial and each class's sine/cosine reference."""
    t = np.arange(X.shape[2]) / sfreq
    out = np.empty((len(X), len(SSVEP_CLASSES)))
    for j, f in enumerate(float(c) for c in SSVEP_CLASSES):
        Y = np.concatenate([[np.sin(2 * np.pi * h * f * t), np.cos(2 * np.pi * h * f * t)] for h in range(1, n_harm + 1)])
        Qy = np.linalg.qr((Y - Y.mean(1, keepdims=True)).T)[0]
        for i, x in enumerate(X):
            Qx = np.linalg.qr((x - x.mean(1, keepdims=True)).T)[0]
            out[i, j] = np.linalg.svd(Qx.T @ Qy, compute_uv=False)[0]
    return out


def run_ssvep(dataset, s: int, fb_kw: dict, bb_kw: dict, occ: list) -> dict:
    Xf, labels, meta = FilterBankSSVEP(filters=FB, n_classes=4, channels=occ, resample=250, **fb_kw).get_data(
        dataset=dataset, subjects=[s])
    Xb, labels_b, meta_b = SSVEP(fmin=8, fmax=40, n_classes=4, channels=occ, resample=250, **bb_kw).get_data(
        dataset=dataset, subjects=[s])
    if not (np.array_equal(labels, labels_b) and np.array_equal(meta.session.to_numpy(), meta_b.session.to_numpy())):
        raise AssertionError(f"SSVEP s{s}: trial order differs between filter bank and broadband")
    keep, train, sess = first_two(meta, f"SSVEP s{s}")
    y = np.array([SSVEP_CLASSES.index(str(l)) for l in labels[keep]])
    for sname in np.unique(sess):
        if len(np.unique(y[sess == sname])) != 4:
            raise AssertionError(f"SSVEP s{s}: session {sname} lacks a class")
    Xs = Xf[keep].transpose(0, 3, 1, 2).reshape(keep.sum(), -1, Xf.shape[2])  # stack bands along channels
    cov = Covariances(estimator="oas")
    C = cov.fit_transform(Xs)
    p_ts = ts_lr(C[train], y[train], C[~train])
    A = align_by_group(Xs, sess)
    Ca = cov.transform(A)
    p_ts_ra = ts_lr(Ca[train], y[train], Ca[~train])
    R = cca_scores(Xb[keep], 250.0)
    p_cca = LogisticRegression(C=1.0, max_iter=5000).fit(R[train], y[train]).predict_proba(R[~train])
    ye = y[~train]

    def acc(p):
        return float((p.argmax(1) == ye).mean())

    return {"s_ts": acc(p_ts), "s_ts_ra": acc(p_ts_ra), "s_cca": acc(p_cca), "s_ens": acc((p_ts_ra + p_cca) / 2),
            "s_n_test": int(len(ye))}


# ---------------------------------------------------------------- summary

def summarise(df: pd.DataFrame) -> None:
    arms = ["p_ts", "p_ts_ra", "p_lda", "p_ens", "s_ts", "s_ts_ra", "s_cca", "s_ens"]
    print(f"\n=== X1 cross-paradigm session transfer, {len(df)} subjects (P300: AUC, SSVEP: accuracy) ===")
    print((100 * df[arms].mean()).round(2).to_string())
    rng = np.random.default_rng(0)
    ps = {h: sign_flip_permutation_test(100 * (df[a] - df[b]).dropna().to_numpy(), "greater")
          for h, (a, b) in HYPOTHESES.items()}
    holm = dict(zip(ps, holm_adjust(list(ps.values()))))
    print("\nHYPOTHESES (one-sided sign-flip, Holm over four):")
    for h, (a, b) in HYPOTHESES.items():
        d = 100 * (df[a] - df[b]).dropna().to_numpy()
        boots = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ok = holm[h] < 0.05 and d.mean() > 0
        print(f"  {h} {a} - {b}: mean {d.mean():+.2f} pp [95% CI {lo:+.2f}, {hi:+.2f}], median {np.median(d):+.2f}, "
              f"{(d > 0).sum()}/{len(d)} up, {(d < 0).sum()} down, p = {ps[h]:.4f} (Holm {holm[h]:.4f}) "
              f"-> {'SUPPORTED' if ok else 'not supported'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    p300_kw = dict(fmin=1, fmax=24, tmin=0.0, tmax=0.8, resample=128)
    fb_kw, bb_kw = dict(tmin=0.5, tmax=4.5), dict(tmin=0.5, tmax=4.5)
    if args.dry_run:
        from moabb.datasets.fake import FakeDataset
        occ = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]  # FakeDataset needs standard-case names
        chans = tuple(occ + ["Cz", "Pz", "Fz", "C3", "C4"])
        p_ds = FakeDataset(event_list=("Target", "NonTarget"), paradigm="p300", n_subjects=2, n_sessions=2,
                           n_runs=1, channels=chans, sfreq=256, duration=200, n_events=120, seed=0)
        s_ds = FakeDataset(event_list=tuple(SSVEP_CLASSES), paradigm="ssvep", n_subjects=2, n_sessions=2, n_runs=1,
                           channels=chans, sfreq=250, duration=300, n_events=40, seed=1)
        fb_kw, bb_kw = dict(tmin=0.5, tmax=2.5), dict(tmin=0.5, tmax=2.5)
        subjects = p_ds.subject_list
    else:
        from moabb.datasets import GuttmannFlury2025_P300, GuttmannFlury2025_SSVEP
        import moabb.datasets.guttmann_flury2025 as gf
        eligible = sorted(s for s, n in gf._SESSIONS_PER_SUBJECT.items() if n >= 2)
        if eligible != SUBJECTS:
            raise AssertionError(f"eligible subjects {eligible} differ from the pre-registered list")
        p_ds, s_ds, subjects = GuttmannFlury2025_P300(subjects=SUBJECTS), GuttmannFlury2025_SSVEP(subjects=SUBJECTS), SUBJECTS
        occ = OCCIPITAL
    rows = []
    for s in subjects:
        ssvep = ({k: np.nan for k in ("s_ts", "s_ts_ra", "s_cca", "s_ens", "s_n_test")}
                 if (not args.dry_run and s in SSVEP_UNAVAILABLE) else run_ssvep(s_ds, s, fb_kw, bb_kw, occ))
        row = {"subject": s, **run_p300(p_ds, s, p300_kw), **ssvep}
        rows.append(row)
        print(f"  s{s}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items() if k not in ("subject",) and "n_test" not in k),
              flush=True)
    df = pd.DataFrame(rows)
    if args.dry_run:
        summarise(df) if len(df) > 1 else None
        print("DRY RUN OK (no real data touched)")
        return 0
    out = RESULTS_DIR / "x1_cross_paradigm_guttmann.csv"
    df.to_csv(out, index=False)
    summarise(df)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
