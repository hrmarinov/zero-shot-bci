"""R11, round-3 screen: ensemble members that err differently from the CNN (IV-2a + IV-2b, 18 subjects).

Ideation and scoring: reports/r11_ideation_ensemble_diversity.md. R10 showed that TS+LR adds
~+1 pp to a CNN ensemble beyond a size-matched CNN ensemble, i.e. the value is in different
errors. Round 3 asks which THIRD member adds more.

COMPARATOR (R10 best, fixed): r10 = 0.5 * CNN3 + 0.5 * TSLR, where
  CNN3 = mean softmax of cached ShallowConvNet seeds 0-2 on RA input
         (IV-2a: data/cache/r3_preds cnn_ra; IV-2b: data/cache/r7_preds teacher, 3test+4test order),
  TSLR = OAS covariance -> TangentSpace(riemann, fitted on T) -> LogisticRegression(C=1),
         8-30 Hz, 0.5-2.5 s, per-session RA (IV-2a) / per-recording RA (IV-2b). Exactly the R10 code.

ARMS (each a third member at equal weight; all settings a priori, nothing tuned on E):
  lowfreq  (CNN3 + TSLR + LF) / 3. LF = 0.5-4 Hz band, 0.5-2.5 s, each channel averaged into
           consecutive 25-sample (100 ms) bins, features z-scored per session/recording on that
           session's own trials (label-free, the time-domain analogue of per-session RA), then
           LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto") fitted on T.
  fts      (CNN3 + TSLR + FTS2) / 3. FTS2 = mean softmax of cached FTSurrogate ShallowConvNet
           seeds 0-1 (IV-2a: data/cache/r8_preds; IV-2b: data/cache/r9_preds). Two seeds so the
           member is size-matched to the control's CNN45.
  fbts     (CNN3 + TSLR + FB) / 3. FB = filter-bank TS+LR: bands (4,8) (8,12) (12,16) (16,24)
           (24,30) (30,38) Hz, 0.5-2.5 s; per band: per-session/recording RA, OAS covariance,
           TangentSpace(riemann) fitted on T; tangent vectors concatenated; LogisticRegression(C=1).

SIZE-MATCHED CONTROL (no new model family):
  ctrl     (CNN3 + TSLR + CNN45) / 3, CNN45 = mean softmax of the cached base CNN seeds 3-4.

PRIMARY CONTRASTS (pooled over 18 subjects; subject score = mean over its evaluation recordings,
ensemble argmax accuracy):
  H_lowfreq, H_fts, H_fbts: arm minus r10.
  PASS iff pooled mean >= +0.5 pp AND one-sided sign-flip p < 0.05 (n = 18, Holm over the 3 arms)
     AND the mean is > 0 in EACH dataset separately.
  SECONDARY (diversity credit): arm minus ctrl, one-sided sign-flip p (unadjusted). A PASS arm's
     gain is credited to diversity iff arm - ctrl pooled > 0 with p < 0.05; otherwise to member count.
  Reported: ctrl - r10; each member alone; among E trials the CNN3 ensemble gets wrong, the
     fraction each member gets right (descriptive only, never used for selection).
  New best: the PASS arm with the largest pooled gain; if none passes, R10 stays the best.
  Combining several passing members is NOT tested here (would be a new round).

VALIDITY:
  V1 For every loader used (8-30 Hz, wide, 0.5-4 Hz, each filter-bank band), labels (and IV-2a
     runs / IV-2b recordings) equal those of the wide CNN loader, so the cached CNN predictions
     line up trial by trial. A mismatch is an assertion.
  V2 Each cache's accuracy is recomputed and must equal the recorded value (|diff| < 1e-9):
     r3 cnn_ra per seed (results/r3_convnet_session_norm.csv); r8 ftsurrogate 3-seed single mean;
     r7 teacher 5-seed single mean per recording; r9 ftsurrogate 3-seed single mean per recording.
  V3 R10 reproduction: this run's cnn, tslr and single-seed stack per-subject accuracies equal
     results/r10_round2_stack_position.csv (|diff| < 1e-12).
  V4 Determinism: a second full run must produce a byte-identical CSV (checked after the run).

    .venv\\Scripts\\python experiments\\r11_round3_diverse_members.py     # ~20 min, CPU
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402

from experiments.r3_convnet_session_norm import WIDE, align_per_session  # noqa: E402
from experiments.r7_self_training_iv2b import CLASSES as IV2B_CLASSES  # noqa: E402
from experiments.r7_self_training_iv2b import EVAL_RECORDINGS, TRAIN_RECORDINGS  # noqa: E402
from experiments.r10_round2_stack_position import align_by_recording, tangent, ts_lr  # noqa: E402
from src.adaptation.convnet import encode_labels  # noqa: E402
from src.config import DATA_DIR, FMAX, FMIN, IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.datasets import load_iv2a_subject_window, load_iv2b_subject_window  # noqa: E402
from src.evaluate import holm_adjust, sign_flip_permutation_test  # noqa: E402

OUT_CSV = RESULTS_DIR / "r11_round3_diverse_members.csv"
CACHE = DATA_DIR / "cache"
LF_BAND = (0.5, 4.0)
LF_BIN = 25
FB_BANDS = ((4, 8), (8, 12), (12, 16), (16, 24), (24, 30), (30, 38))
GATE_PP = 0.5
ARMS = ("lowfreq", "fts", "fbts")
TOL_V2, TOL_V3 = 1e-9, 1e-12


# ---------------------------------------------------------------- members
def bin_features(X: np.ndarray) -> np.ndarray:
    nb = X.shape[2] // LF_BIN
    return X[:, :, : nb * LF_BIN].reshape(len(X), X.shape[1], nb, LF_BIN).mean(-1).reshape(len(X), -1)


def zscore_by_group(F: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.empty_like(F)
    for g in np.unique(groups):
        m = groups == g
        out[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-12)
    return out


def lowfreq_member(Xt, yt, gt, Xe, ge) -> np.ndarray:
    Ft, Fe = zscore_by_group(bin_features(Xt), gt), zscore_by_group(bin_features(Xe), ge)
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Ft, yt).predict_proba(Fe)


def fb_member(bands_t: list, yt, bands_e: list) -> np.ndarray:
    pairs = [tangent(Xt, Xe) for Xt, Xe in zip(bands_t, bands_e)]
    return ts_lr(np.hstack([p[0] for p in pairs]), yt, np.hstack([p[1] for p in pairs]))


# ---------------------------------------------------------------- scoring
def rec_acc(p: np.ndarray, y: np.ndarray, rec: np.ndarray) -> float:
    pr = p.argmax(1)
    return float(np.mean([(pr[rec == r] == y[rec == r]).mean() for r in np.unique(rec)]))


def score(rows, dataset, s, y, rec, base, fts, ts, lf, fb) -> dict:
    cnn3, cnn45, fts2 = np.mean(base[:3], 0), np.mean(base[3:5], 0), np.mean(fts[:2], 0)
    probs = {"r10": 0.5 * cnn3 + 0.5 * ts, "ctrl": (cnn3 + ts + cnn45) / 3,
             "lowfreq": (cnn3 + ts + lf) / 3, "fts": (cnn3 + ts + fts2) / 3, "fbts": (cnn3 + ts + fb) / 3,
             "m_cnn3": cnn3, "m_tslr": ts, "m_lowfreq": lf, "m_fts2": fts2, "m_fbts": fb}
    for arm, p in probs.items():
        rows.append({"dataset": dataset, "subject": s, "arm": arm, "acc": rec_acc(p, y, rec)})
    wrong = cnn3.argmax(1) != y
    for m in ("m_tslr", "m_lowfreq", "m_fts2", "m_fbts"):
        rows.append({"dataset": dataset, "subject": s, "arm": f"rescue_{m[2:]}",
                     "acc": float((probs[m].argmax(1)[wrong] == y[wrong]).mean())})
    # V3 inputs: R10's arms recomputed (seed-wise single-model means)
    return {"cnn": np.mean([rec_acc(p, y, rec) for p in base[:3]]), "tslr": rec_acc(ts, y, rec),
            "stack": np.mean([rec_acc(0.5 * p + 0.5 * ts, y, rec) for p in base[:3]])}


def check_v2(name: str, got: float, want: float) -> None:
    if abs(got - want) >= TOL_V2:
        raise AssertionError(f"V2: {name} recomputed {got:.12f} != recorded {want:.12f}")


# ---------------------------------------------------------------- datasets
def run_iv2a(s: int, rows: list, r10: list, rec_tables: dict) -> None:
    w = load_iv2a_subject_window(s, **WIDE)
    load = lambda lo, hi, t0=TMIN, t1=TMAX: load_iv2a_subject_window(s, lo, hi, t0, t1)  # noqa: E731
    d = load(FMIN, FMAX)
    lfd = load(*LF_BAND)
    fbd = [load(lo, hi) for lo, hi in FB_BANDS]
    for name, x in [("8-30", d), ("lowfreq", lfd)] + [(f"fb{b}", x) for b, x in zip(FB_BANDS, fbd)]:
        if not (np.array_equal(x.y_eval, w.y_eval) and np.array_equal(x.y_calib, w.y_calib)
                and np.array_equal(x.runs_eval, w.runs_eval) and np.array_equal(x.runs_calib, w.runs_calib)):
            raise AssertionError(f"V1: IV-2a s{s} {name} trial order differs from the wide loader")
    yt, ye = encode_labels(d.y_calib, IV2A_CLASSES), encode_labels(d.y_eval, IV2A_CLASSES)
    rec_e = np.zeros(len(ye))
    Xt, Xe = align_per_session(d.X_calib, d.X_eval)
    Ft, Fe = tangent(Xt, Xe)
    ts = ts_lr(Ft, yt, Fe)
    lf = lowfreq_member(lfd.X_calib, yt, np.zeros(len(yt)), lfd.X_eval, rec_e)
    fb_al = [align_per_session(x.X_calib, x.X_eval) for x in fbd]
    fb = fb_member([a[0] for a in fb_al], yt, [a[1] for a in fb_al])

    base = [np.load(CACHE / "r3_preds" / f"s{s}_cnn_ra_seed{k}.npy") for k in range(5)]
    fts = [np.load(CACHE / "r8_preds" / f"s{s}_ftsurrogate_seed{k}.npy") for k in range(3)]
    r3 = rec_tables["r3"]
    for k in range(5):
        want = r3[(r3.subject == s) & (r3.arm == "cnn_ra") & (r3.seed == k)].acc.item()
        check_v2(f"r3 s{s} seed{k}", (base[k].argmax(1) == ye).mean(), want)
    r8 = rec_tables["r8"]
    want = r8[(r8.subject == s) & (r8.arm == "ftsurrogate") & (r8.metric == "single_mean_acc")].value.item()
    check_v2(f"r8 s{s}", np.mean([(p.argmax(1) == ye).mean() for p in fts]), want)
    r10.append({"dataset": "IV-2a", "subject": s, **score(rows, "IV-2a", s, ye, rec_e, base, fts, ts, lf, fb)})


def run_iv2b(s: int, rows: list, r10: list, rec_tables: dict) -> None:
    w = load_iv2b_subject_window(s, **WIDE)
    load = lambda lo, hi: load_iv2b_subject_window(s, lo, hi, TMIN, TMAX)  # noqa: E731
    d, lfd = load(FMIN, FMAX), load(*LF_BAND)
    fbd = [load(lo, hi) for lo, hi in FB_BANDS]
    for name, x in [("8-30", d), ("lowfreq", lfd)] + [(f"fb{b}", x) for b, x in zip(FB_BANDS, fbd)]:
        if not (np.array_equal(x.y, w.y) and np.array_equal(x.session, w.session)):
            raise AssertionError(f"V1: IV-2b s{s} {name} label/recording order differs from the wide loader")
    y = encode_labels(d.y, IV2B_CLASSES)
    tr = np.isin(d.session, TRAIN_RECORDINGS)
    ev = np.concatenate([np.flatnonzero(d.session == r) for r in EVAL_RECORDINGS])  # the R7/R9 cache order
    yt, ye, gt, ge = y[tr], y[ev], d.session[tr], d.session[ev]

    X = align_by_recording(d.X, d.session)
    Ft, Fe = tangent(X[tr], X[ev])
    ts = ts_lr(Ft, yt, Fe)
    lf = lowfreq_member(lfd.X[tr], yt, gt, lfd.X[ev], ge)
    fb_al = [align_by_recording(x.X, x.session) for x in fbd]
    fb = fb_member([a[tr] for a in fb_al], yt, [a[ev] for a in fb_al])

    base = [np.load(CACHE / "r7_preds" / f"s{s}_teacher_seed{k}.npy") for k in range(5)]
    fts = [np.load(CACHE / "r9_preds" / f"s{s}_ftsurrogate_seed{k}.npy") for k in range(3)]
    for tab, arm, preds in (("r7", "teacher", base), ("r9", "ftsurrogate", fts)):
        t = rec_tables[tab]
        for r in EVAL_RECORDINGS:
            m = ge == r
            want = t[(t.subject == s) & (t.recording == r) & (t.arm == arm) & (t.metric == "single_mean_acc")].value.item()
            check_v2(f"{tab} s{s} {r}", np.mean([(p.argmax(1)[m] == ye[m]).mean() for p in preds]), want)
    r10.append({"dataset": "IV-2b", "subject": s, **score(rows, "IV-2b", s, ye, ge, base, fts, ts, lf, fb)})


# ---------------------------------------------------------------- analysis
def check_v3(r10_rows: list) -> None:
    ref = pd.read_csv(RESULTS_DIR / "r10_round2_stack_position.csv")
    got = pd.DataFrame(r10_rows).melt(id_vars=["dataset", "subject"], var_name="arm", value_name="acc")
    m = got.merge(ref, on=["dataset", "subject", "arm"], suffixes=("", "_ref"))
    if len(m) != 54 or (m.acc - m.acc_ref).abs().max() >= TOL_V3:
        raise AssertionError(f"V3: R10 not reproduced (n = {len(m)}, max diff {(m.acc - m.acc_ref).abs().max():.2e})")
    print("V3: R10 cnn / tslr / stack reproduced for all 18 subjects")


def contrast_line(name: str, d: pd.Series, p: float, p_holm: float | None = None) -> str:
    per = d.groupby("dataset").mean()
    holm = f" (Holm {p_holm:.4f})" if p_holm is not None else ""
    return (f"  {name:18s} pooled {d.mean():+.2f} pp, {(d > 0).sum()}/18 up, {(d < 0).sum()} down, p = {p:.4f}{holm}; "
            + ", ".join(f"{k} {v:+.2f}" for k, v in per.items()))


def summarise(df: pd.DataFrame) -> None:
    piv = 100 * df.pivot_table(index=["dataset", "subject"], columns="arm", values="acc")
    print("\n=== R11 round-3 screen (IV-2a + IV-2b, 18 subjects) ===")
    print(piv.groupby("dataset").mean().round(2).T.to_string())
    ps = {a: sign_flip_permutation_test((piv[a] - piv["r10"]).to_numpy(), "greater") for a in ARMS}
    holm = dict(zip(ps, holm_adjust(list(ps.values()))))
    print("\nPrimary (arm - r10):")
    passed = {}
    for a in ARMS:
        d = piv[a] - piv["r10"]
        ok = d.mean() >= GATE_PP and holm[a] < 0.05 and (d.groupby("dataset").mean() > 0).all()
        print(contrast_line(a, d, ps[a], holm[a]) + f"  -> {'PASS' if ok else 'killed'}")
        if ok:
            passed[a] = d.mean()
    print("Secondary (arm - ctrl, diversity credit):")
    for a in ARMS:
        d = piv[a] - piv["ctrl"]
        p = sign_flip_permutation_test(d.to_numpy(), "greater")
        print(contrast_line(a, d, p) + f"  -> {'diversity' if d.mean() > 0 and p < 0.05 else 'not credited'}")
    d = piv["ctrl"] - piv["r10"]
    print("Reported:\n" + contrast_line("ctrl - r10", d, sign_flip_permutation_test(d.to_numpy(), "greater")))
    best = max(passed, key=passed.get) if passed else "r10"
    print(f"\nNEW BEST: {best}" + ("" if passed else " (no arm passed; R10 stays)"))


def main() -> int:
    tables = {"r3": pd.read_csv(RESULTS_DIR / "r3_convnet_session_norm.csv"),
              "r7": pd.read_csv(RESULTS_DIR / "r7_self_training_iv2b.csv"),
              "r8": pd.read_csv(RESULTS_DIR / "r8_augmentation.csv"),
              "r9": pd.read_csv(RESULTS_DIR / "r9_ftsurrogate_iv2b.csv")}
    rows: list[dict] = []
    r10_rows: list[dict] = []
    for s in IV2A_SUBJECTS:
        run_iv2a(s, rows, r10_rows, tables)
        run_iv2b(s, rows, r10_rows, tables)
        print(f"  s{s} done (V1, V2 passed)", flush=True)
    check_v3(r10_rows)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summarise(df)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
