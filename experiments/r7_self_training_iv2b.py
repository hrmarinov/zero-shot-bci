"""R7 CONFIRMATION: R6 pseudo-label self-training, setting frozen, on BCI IV-2b.

R6 (docs/diag_gates_d1_d4.md) passed the screen on IV-2a (+1.62 pp over the teacher ensemble,
Holm 0.070). This is the confirmatory test on a dataset the method has never seen. Nothing is
tuned: same ShallowConvNet, TrainConfig (CFG), band/window (WIDE), confidence fraction 0.5 per
predicted class, 5 seeds, hard ensemble pseudo-labels, one round.

PROTOCOL (fixed before the first run):
  Data: BNCI2014_004 (3 ch C3/Cz/C4, left vs right hand), all 9 subjects. Training = MOABB
    recordings 0train+1train+2train; evaluation = 3test and 4test, each treated as its own
    deployment session (self-training sees one evaluation recording at a time).
  Alignment: Riemannian alignment fitted per RECORDING on that recording only (the 5
    recordings are not protocol-equivalent, docs/datasets.md §3).
  Teacher: ShallowConvNet(3, 2, 1001) on aligned training recordings, seeds 0-4.
  Student (st): seed k retrained on training recordings + the confident 50% (per predicted
    class, by teacher-ensemble max-probability) of ONE evaluation recording with the
    ensemble's hard labels; scored on that recording.
  Subject score = mean over the two evaluation recordings.

HYPOTHESIS (one, so no multiplicity correction):
  H1: st 5-seed ensemble accuracy > teacher 5-seed ensemble accuracy.
  CONFIRMED iff subject-mean difference > 0 AND one-sided exact sign-flip p < 0.05 (n = 9).
  Reported regardless: per-subject differences, 95% t-interval, single-model contrast.

VALIDITY (the run is UNINFORMATIVE if any fails; a failure is reported, never re-run with changes):
  V1 teacher ensemble mean >= 60% (else the teacher is too weak for pseudo-labels to mean anything)
  V2 shuffled pseudo-labels (subjects 1, 3, 7; recording 3test; seed 0) must NOT beat the
     teacher seed-0 model by >= 1 pp.
  V3 determinism: subject 1 teacher seed 0 trained twice gives identical probabilities.
  V4 every alignment self-check passes (recording mean -> identity, < 1e-8).

    .venv\\Scripts\\python experiments\\r7_self_training_iv2b.py     # ~80 min on the RTX 4060
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from experiments.r3_convnet_session_norm import CFG, WIDE, ra_self_check  # noqa: E402
from src.adaptation.convnet import ShallowConvNet, encode_labels, predict_proba, set_deterministic, train_model  # noqa: E402
from src.adaptation.riemannian_align import RiemannianAlignment  # noqa: E402
from src.adaptation.self_training import one_hot, select_confident  # noqa: E402
from src.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.datasets import IV2B_CALIB_SESSIONS, IV2B_EVAL_SESSIONS, load_iv2b_subject_window  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402

OUT_CSV = RESULTS_DIR / "r7_self_training_iv2b.csv"
PRED_DIR = DATA_DIR / "cache" / "r7_preds"
CLASSES = ["left_hand", "right_hand"]
SUBJECTS = list(range(1, 10))
SEEDS = (0, 1, 2, 3, 4)
CONF_FRAC = 0.5
N_TIMES = 1001
CONTROL_SUBJECTS = (1, 3, 7)
EVAL_RECORDINGS = sorted(IV2B_EVAL_SESSIONS)
TRAIN_RECORDINGS = sorted(IV2B_CALIB_SESSIONS)


def align_recordings(X: np.ndarray, session: np.ndarray) -> np.ndarray:
    out = np.empty_like(X)
    for rec in np.unique(session):
        idx = session == rec
        ra = RiemannianAlignment().fit(X[idx])
        dev = ra_self_check(X[idx], ra)
        if dev > 1e-8:
            raise AssertionError(f"V4: alignment self-check failed for {rec}: {dev:.2e}")
        out[idx] = ra.transform(X[idx])
    return out


def fit_predict(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, seed: int, device) -> np.ndarray:
    """Train on (X_train, probability targets Y_train); return probabilities on X_test."""
    wait_until_healthy()
    set_deterministic(seed)
    model = train_model(ShallowConvNet(3, 2, N_TIMES).to(device),
                        torch.as_tensor(X_train.astype(np.float32), device=device),
                        torch.as_tensor(Y_train.astype(np.float32), device=device), CFG, seed=seed)
    probs = predict_proba(model, torch.as_tensor(X_test.astype(np.float32), device=device))
    del model
    torch.cuda.empty_cache()
    return probs


def cached(path: Path, compute) -> np.ndarray:
    if path.exists():
        return np.load(path)
    probs = compute()
    np.save(path, probs.astype(np.float32))
    return probs


def run_subject(s: int, device, rows: list[dict]) -> None:
    d = load_iv2b_subject_window(s, **WIDE)
    if d.X.shape[1:] != (3, N_TIMES):
        raise AssertionError(f"unexpected shape {d.X.shape}")
    X = align_recordings(d.X, d.session)
    y = encode_labels(d.y, CLASSES)
    tr = np.isin(d.session, TRAIN_RECORDINGS)
    Xt, Yt = X[tr], one_hot(y[tr], 2)
    ev_idx = {rec: np.flatnonzero(d.session == rec) for rec in EVAL_RECORDINGS}
    X_eval_all = X[np.concatenate(list(ev_idx.values()))]

    teacher = [cached(PRED_DIR / f"s{s}_teacher_seed{k}.npy", lambda: fit_predict(Xt, Yt, X_eval_all, k, device))
               for k in SEEDS]
    if s == 1:
        again = fit_predict(Xt, Yt, X_eval_all, 0, device).astype(np.float32)
        if not np.array_equal(again, teacher[0]):
            raise AssertionError("V3: teacher seed 0 is not deterministic")
    offset = 0
    for rec in EVAL_RECORDINGS:
        idx = ev_idx[rec]
        sl = slice(offset, offset + len(idx))
        offset += len(idx)
        Xe, ye = X[idx], y[idx]  # ye is used for scoring only, below
        t_probs = [p[sl] for p in teacher]
        ens = np.mean(t_probs, axis=0)
        conf = select_confident(ens, CONF_FRAC)
        pseudo = one_hot(ens.argmax(1)[conf], 2)
        X_st, Y_st = np.concatenate([Xt, Xe[conf]]), np.concatenate([Yt, pseudo])
        st = [cached(PRED_DIR / f"s{s}_{rec}_st_seed{k}.npy", lambda: fit_predict(X_st, Y_st, Xe, k, device))
              for k in SEEDS]

        def add(arm, metric, value):
            rows.append({"subject": s, "recording": rec, "arm": arm, "metric": metric, "value": float(value)})

        for arm, ps in (("teacher", t_probs), ("st", st)):
            add(arm, "ens_acc", (np.mean(ps, 0).argmax(1) == ye).mean())
            add(arm, "single_mean_acc", np.mean([(p.argmax(1) == ye).mean() for p in ps]))
        add("teacher", "confident_half_acc", (ens.argmax(1)[conf] == ye[conf]).mean())
        if s in CONTROL_SUBJECTS and rec == EVAL_RECORDINGS[0]:
            shuffled = pseudo[np.random.default_rng(0).permutation(len(pseudo))]
            sh = fit_predict(X_st, np.concatenate([Yt, shuffled]), Xe, 0, device)
            add("st_shuffled", "seed0_acc", (sh.argmax(1) == ye).mean())
            add("teacher", "seed0_acc", (t_probs[0].argmax(1) == ye).mean())
        print(f"  s{s} {rec}: teacher ens {100*(ens.argmax(1) == ye).mean():.1f}%  "
              f"st ens {100*(np.mean(st, 0).argmax(1) == ye).mean():.1f}%", flush=True)


def summarise(df: pd.DataFrame) -> None:
    def per_subject(arm: str, metric: str) -> pd.Series:
        q = df[(df.arm == arm) & (df.metric == metric)]
        return 100 * q.groupby("subject").value.mean()

    t, st = per_subject("teacher", "ens_acc"), per_subject("st", "ens_acc")
    d = st - t
    p = sign_flip_permutation_test(d.to_numpy(), "greater")
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    single = per_subject("st", "single_mean_acc") - per_subject("teacher", "single_mean_acc")
    print("\n=== R7 confirmation: self-training on IV-2b (9 subjects, mean of 3test/4test) ===")
    print("per subject (st - teacher, ensemble pp): " + ", ".join(f"s{k} {v:+.1f}" for k, v in d.items()))
    print(f"teacher ensemble {t.mean():.2f}%  (confident half {per_subject('teacher', 'confident_half_acc').mean():.1f}%)")
    print(f"st ensemble      {st.mean():.2f}%")
    print(f"difference {d.mean():+.2f} pp, 95% CI [{d.mean()-half:+.2f}, {d.mean()+half:+.2f}], "
          f"{(d > 0).sum()}/9 up, one-sided sign-flip p = {p:.4f}")
    print(f"single-model mean: teacher {per_subject('teacher', 'single_mean_acc').mean():.2f}%, "
          f"st {per_subject('st', 'single_mean_acc').mean():.2f}% ({single.mean():+.2f} pp)")
    v1 = t.mean() >= 60.0
    sh = per_subject("st_shuffled", "seed0_acc") - per_subject("teacher", "seed0_acc")
    v2 = bool((sh < 1.0).all()) and len(sh) == len(CONTROL_SUBJECTS)
    print(f"V1 teacher >= 60%: {'PASS' if v1 else 'FAIL'};  V2 shuffled control "
          f"({', '.join(f's{k} {v:+.1f}pp' for k, v in sh.items())}): {'PASS' if v2 else 'FAIL'};  V3, V4: PASS (asserted)")
    if not (v1 and v2):
        print("VERDICT: UNINFORMATIVE (validity failed)")
    else:
        print(f"VERDICT: H1 {'CONFIRMED' if d.mean() > 0 and p < 0.05 else 'NOT CONFIRMED'}")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in SUBJECTS:
        print(f"=== subject {s} ===", flush=True)
        run_subject(s, device, rows)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    summarise(pd.DataFrame(rows))
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
