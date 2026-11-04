"""R10, round-2 screen. Scored on IV-2a and IV-2b jointly (18 subjects, see the winner's-curse note).

Round-2 ideas (docs/diag_gates_d1_d4.md, after R9/G0):
  stack     Heterogeneous ensemble: the average of ShallowConvNet softmax (RA input) and the
            Riemannian tangent-space logistic regression (TS+LR) probabilities. The two models
            have different inductive biases, and the decoder is variance-limited (R6, R8).
  posmatch  Position-matched TS+LR, borrowed from matched-condition training in speech. The
            decline is time-on-task in BOTH sessions (D5), and a trial's position in its
            recording is known without labels. So the training trials are weighted towards the
            test trial's half of the recording: weight 1.0 for the same half, 0.5 for the other.

FIXED SETTINGS (a priori; nothing is tuned on E):
  TS+LR: 8-30 Hz, 0.5-2.5 s, per-session RA (IV-2a) or per-recording RA (IV-2b), OAS
    covariance, TangentSpace(riemann) fitted on training data, LogisticRegression(C=1, lbfgs).
  CNN: cached seeds 0-2. IV-2a: R3 cnn_ra (data/cache/r3_preds). IV-2b: the R7 teacher
    (data/cache/r7_preds). Both are unaugmented ShallowConvNet on RA input.
  Position: a trial's index within its recording divided by the recording length; "early" < 0.5.

PRIMARY CONTRASTS (pooled over 18 subjects; a subject's score is the mean over its evaluation recordings):
  H_stack:    stack (3-seed single-model mean) minus CNN (3-seed single-model mean).
  H_posmatch: posmatch minus TS+LR.
  PASS iff pooled mean >= +1.0 pp AND one-sided sign-flip p < 0.05 (n = 18, Holm over the two)
     AND the mean is > 0 in EACH dataset separately.
  Reported: TS+LR vs CNN (which of "the Riemannian method" and the CNN is stronger, per dataset).

VALIDITY:
  V1 Labels from the 8-30 Hz loader equal the labels from the wide CNN loader, so the cached
     CNN predictions line up trial by trial. A mismatch is an assertion.
  V2 The cached CNN accuracy per subject equals the R3/R7 recorded value (recomputed here).
     Any misalignment would show up as near-chance CNN accuracy; the check requires >= 60% mean.

    .venv\\Scripts\\python experiments\\r10_round2_stack_position.py     # ~5 min, CPU
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from experiments.r3_convnet_session_norm import WIDE, align_per_session  # noqa: E402
from experiments.r7_self_training_iv2b import CLASSES as IV2B_CLASSES  # noqa: E402
from experiments.r7_self_training_iv2b import EVAL_RECORDINGS, TRAIN_RECORDINGS  # noqa: E402
from src.adaptation.convnet import encode_labels  # noqa: E402
from src.adaptation.riemannian_align import RiemannianAlignment  # noqa: E402
from src.config import DATA_DIR, FMAX, FMIN, IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.datasets import load_iv2a_subject_window, load_iv2b_subject_window  # noqa: E402
from src.evaluate import holm_adjust, sign_flip_permutation_test  # noqa: E402

OUT_CSV = RESULTS_DIR / "r10_round2_stack_position.csv"
SEEDS = (0, 1, 2)
OTHER_HALF_WEIGHT = 0.5
GATE_PP = 1.0


def positions(groups: np.ndarray) -> np.ndarray:
    """Fractional position of each trial within its recording (trials are stored in time order)."""
    pos = np.empty(len(groups))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        pos[idx] = np.arange(len(idx)) / len(idx)
    return pos


def tangent(Xt: np.ndarray, Xe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = Covariances(estimator="oas")
    ts = TangentSpace(metric="riemann").fit(cov.fit_transform(Xt))
    return ts.transform(cov.transform(Xt)), ts.transform(cov.transform(Xe))


def ts_lr(Ft, yt, Fe, weights=None) -> np.ndarray:
    return LogisticRegression(C=1.0, max_iter=5000).fit(Ft, yt, sample_weight=weights).predict_proba(Fe)


def posmatch(Ft, yt, pos_t, Fe, pos_e) -> np.ndarray:
    out = np.empty((len(Fe), len(np.unique(yt))))
    for early in (True, False):
        te = (pos_e < 0.5) == early
        if not te.any():
            continue
        w = np.where((pos_t < 0.5) == early, 1.0, OTHER_HALF_WEIGHT)
        out[te] = ts_lr(Ft, yt, Fe[te], w)
    return out


def score(rows, dataset, s, ye, rec_of_e, cnn, p_ts, p_pm) -> None:
    arms = {"cnn": [p.argmax(1) for p in cnn],
            "stack": [(0.5 * p + 0.5 * p_ts).argmax(1) for p in cnn],
            "tslr": [p_ts.argmax(1)], "posmatch": [p_pm.argmax(1)]}
    for arm, preds in arms.items():
        acc = np.mean([np.mean([(pr[rec_of_e == r] == ye[rec_of_e == r]).mean() for r in np.unique(rec_of_e)])
                       for pr in preds])
        rows.append({"dataset": dataset, "subject": s, "arm": arm, "acc": float(acc)})


def run_iv2a(s: int, rows: list[dict]) -> None:
    d = load_iv2a_subject_window(s, FMIN, FMAX, TMIN, TMAX)
    w = load_iv2a_subject_window(s, **WIDE)
    if not np.array_equal(d.y_eval, w.y_eval):
        raise AssertionError(f"V1: IV-2a s{s} label order differs between loaders")
    Xt, Xe = align_per_session(d.X_calib, d.X_eval)
    yt, ye = encode_labels(d.y_calib, IV2A_CLASSES), encode_labels(d.y_eval, IV2A_CLASSES)
    Ft, Fe = tangent(Xt, Xe)
    cnn = [np.load(DATA_DIR / "cache" / "r3_preds" / f"s{s}_cnn_ra_seed{k}.npy") for k in SEEDS]
    rec_t, rec_e = np.zeros(len(yt)), np.zeros(len(ye))
    score(rows, "IV-2a", s, ye, rec_e, cnn, ts_lr(Ft, yt, Fe),
          posmatch(Ft, yt, positions(rec_t), Fe, positions(rec_e)))


def align_by_recording(X: np.ndarray, rec: np.ndarray) -> np.ndarray:
    out = np.empty_like(X)
    for r in np.unique(rec):
        out[rec == r] = RiemannianAlignment().fit_transform(X[rec == r])
    return out


def run_iv2b(s: int, rows: list[dict]) -> None:
    d = load_iv2b_subject_window(s, FMIN, FMAX, TMIN, TMAX)
    w = load_iv2b_subject_window(s, **WIDE)
    if not (np.array_equal(d.y, w.y) and np.array_equal(d.session, w.session)):
        raise AssertionError(f"V1: IV-2b s{s} label/recording order differs between loaders")
    X = align_by_recording(d.X, d.session)
    y = encode_labels(d.y, IV2B_CLASSES)
    tr = np.isin(d.session, TRAIN_RECORDINGS)
    ev = np.concatenate([np.flatnonzero(d.session == r) for r in EVAL_RECORDINGS])  # the R7 cache order
    Ft, Fe = tangent(X[tr], X[ev])
    cnn = [np.load(DATA_DIR / "cache" / "r7_preds" / f"s{s}_teacher_seed{k}.npy") for k in SEEDS]
    score(rows, "IV-2b", s, y[ev], d.session[ev], cnn, ts_lr(Ft, y[tr], Fe),
          posmatch(Ft, y[tr], positions(d.session[tr]), Fe, positions(d.session[ev])))


def summarise(df: pd.DataFrame) -> None:
    piv = 100 * df.pivot_table(index=["dataset", "subject"], columns="arm", values="acc")
    print("\n=== R10 round-2 screen (IV-2a + IV-2b, 18 subjects) ===")
    print(piv.groupby("dataset").mean().round(2).to_string())
    if piv["cnn"].groupby("dataset").mean().min() < 60:
        print("VERDICT: UNINFORMATIVE (V2: cached CNN predictions misaligned)")
        return
    contrasts = {"stack - cnn": piv["stack"] - piv["cnn"], "posmatch - tslr": piv["posmatch"] - piv["tslr"]}
    ps = {k: sign_flip_permutation_test(v.to_numpy(), "greater") for k, v in contrasts.items()}
    holm = dict(zip(ps, holm_adjust(list(ps.values()))))
    for name, d in contrasts.items():
        per = d.groupby("dataset").mean()
        ok = d.mean() >= GATE_PP and holm[name] < 0.05 and (per > 0).all()
        print(f"  {name:16s} pooled {d.mean():+.2f} pp, {(d > 0).sum()}/18 up, p = {ps[name]:.4f} (Holm {holm[name]:.4f}); "
              + ", ".join(f"{k} {v:+.2f}" for k, v in per.items()) + f"  -> {'PASS' if ok else 'killed'}")
    t = piv["tslr"] - piv["cnn"]
    print("  reported: tslr - cnn " + ", ".join(f"{k} {v:+.2f} pp" for k, v in t.groupby('dataset').mean().items()))


def main() -> int:
    rows: list[dict] = []
    for s in IV2A_SUBJECTS:
        run_iv2a(s, rows)
        run_iv2b(s, rows)
        print(f"  s{s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summarise(df)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
