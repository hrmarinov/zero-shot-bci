"""R3 — subject-specific ShallowConvNet + per-session normalisation at every layer.

Pre-registered in docs/prereg_convnet_session_norm.md (arms, seeds, budget, statistics
and decision rule are fixed there; this script implements exactly that). Statistics are
computed by experiments/r3b_convnet_session_norm_analysis.py from the CSV written here.

Resumable: every (subject, arm, seed) result is appended to the CSV as soon as it exists
and skipped on a rerun, so a health-guard abort or a crash loses at most one training run.
Per-run class probabilities go to data/cache/r3_preds/ (gitignored) for McNemar and the
seed ensemble.

Leakage discipline: evaluation labels are loaded once and passed only to `score()`.
Nothing that fits anything (alignment, CSP, LDA, the convnet, AdaBN) receives them.

    .venv\\Scripts\\python experiments\\r3_convnet_session_norm.py            # ~2.5 h on the RTX 4060
    .venv\\Scripts\\python experiments\\r3_convnet_session_norm.py --subjects 1 --seeds 0   # smoke
"""

from __future__ import annotations

import argparse
import copy
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.geometry.mean import mean_riemann  # noqa: E402

from src.adaptation.convnet import (  # noqa: E402
    ShallowConvNet,
    TrainConfig,
    adapt_bn_statistics,
    encode_labels,
    predict_proba,
    predict_proba_groupwise,
    set_deterministic,
    train_model,
)
from src.adaptation.riemannian_align import RiemannianAlignment  # noqa: E402
from src.baselines import make_csp_lda  # noqa: E402
from src.config import DATA_DIR, FMAX, FMIN, IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.datasets import load_iv2a_subject_window  # noqa: E402
from src.evaluate import fit_predict  # noqa: E402
from src.gpu_health import query_gpu, wait_until_healthy  # noqa: E402
from src.perf import mem_report  # noqa: E402

WIDE = dict(fmin=4.0, fmax=38.0, tmin=0.0, tmax=4.0)
N_TIMES_WIDE = 1001
CFG = TrainConfig(epochs=300, batch_size=48, lr=1e-3, weight_decay=1e-3, max_shift=0)
SEEDS = (0, 1, 2, 3, 4)
PHASE2_CSV = RESULTS_DIR / "phase2_riemannian_alignment_iv2a.csv"
OUT_CSV = RESULTS_DIR / "r3_convnet_session_norm.csv"
PRED_DIR = DATA_DIR / "cache" / "r3_preds"
CNN_ARMS = ("cnn_none", "cnn_adabn", "cnn_ra", "cnn_ra_adabn", "cnn_ra_sat")


# --------------------------------------------------------------------------- helpers


def score(probs_or_labels: np.ndarray, y_eval_idx: np.ndarray) -> tuple[float, int]:
    """The only place evaluation labels are used."""
    pred = probs_or_labels.argmax(1) if probs_or_labels.ndim == 2 else probs_or_labels
    if pred.shape[0] != y_eval_idx.shape[0]:
        raise ValueError("prediction / label length mismatch")
    n_correct = int((pred == y_eval_idx).sum())
    return n_correct / len(y_eval_idx), n_correct


def ra_self_check(X: np.ndarray, aligner: RiemannianAlignment) -> float:
    """max |W · mean_riemann(OAS cov(X)) · W − I| — must be ~1e-12 (see ceiling_analysis §5 Q2)."""
    mean_cov = mean_riemann(Covariances(estimator=aligner.estimator).fit_transform(X))
    W = aligner.whitening_
    return float(np.abs(W @ mean_cov @ W - np.eye(W.shape[0])).max())


def align_per_session(X_calib: np.ndarray, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit one RiemannianAlignment per session, on that session only (asserted)."""
    ra_c, ra_e = RiemannianAlignment().fit(X_calib), RiemannianAlignment().fit(X_eval)
    for X, ra in ((X_calib, ra_c), (X_eval, ra_e)):
        dev = ra_self_check(X, ra)
        if dev > 1e-8:
            raise AssertionError(f"alignment self-check failed: {dev:.2e}")
    if np.allclose(ra_c.whitening_, ra_e.whitening_):
        raise AssertionError("calib and eval whitening are identical - per-session fit broken")
    return ra_c.transform(X_calib), ra_e.transform(X_eval)


def load_done() -> set[tuple[int, str, int]]:
    if not OUT_CSV.exists():
        return set()
    df = pd.read_csv(OUT_CSV)
    return set(zip(df.subject, df.arm, df.seed))


def append_row(row: dict) -> None:
    pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)


def phase2_reference(subject: int) -> float:
    df = pd.read_csv(PHASE2_CSV)
    return float(df.loc[df.subject.astype(str) == str(subject), "cross_session_aligned_acc"].iloc[0])


# --------------------------------------------------------------------------- per subject


def check_sessions(d) -> None:
    for X, y, runs in ((d.X_calib, d.y_calib, d.runs_calib), (d.X_eval, d.y_eval, d.runs_eval)):
        assert X.shape == (288, 22, N_TIMES_WIDE), X.shape
        assert set(np.unique(y)) == set(IV2A_CLASSES), np.unique(y)
        assert np.all(np.unique(y, return_counts=True)[1] == 72)
        assert np.all(np.bincount(runs) == 48)


def run_csp_controls(subject: int, wide, y_cal_str: np.ndarray, y_eval_idx: np.ndarray, done) -> None:
    narrow = load_iv2a_subject_window(subject, FMIN, FMAX, TMIN, TMAX)
    conditions = {
        "csp_ra": align_per_session(narrow.X_calib, narrow.X_eval),
        "csp_ra_wide": align_per_session(wide.X_calib, wide.X_eval),
        "csp_none_wide": (wide.X_calib, wide.X_eval),
    }
    for arm, (Xc, Xe) in conditions.items():
        if (subject, arm, -1) in done:
            continue
        pred = encode_labels(fit_predict(make_csp_lda(), Xc, y_cal_str, Xe), IV2A_CLASSES)
        acc, n_correct = score(pred, y_eval_idx)
        if arm == "csp_ra" and acc != phase2_reference(subject):
            raise AssertionError(f"csp_ra {acc!r} does not reproduce Phase 2 {phase2_reference(subject)!r}")
        np.save(PRED_DIR / f"s{subject}_{arm}_seed-1.npy", pred)
        append_row({"subject": subject, "arm": arm, "seed": -1, "acc": acc, "n_correct": n_correct,
                    "gpu_temp_c": np.nan, "train_s": np.nan})
        print(f"    {arm:14s} acc={acc:.4f}", flush=True)


def run_cnn(subject: int, inputs: dict, runs: tuple, y_cal_idx: np.ndarray, y_eval_idx: np.ndarray,
            seeds, done, device) -> None:
    def tensors(key):
        Xc, Xe = inputs[key]
        return torch.as_tensor(Xc, device=device), torch.as_tensor(Xe, device=device)

    yc = torch.as_tensor(y_cal_idx, device=device)
    runs_c, runs_e = runs

    def fresh(seed: int) -> ShallowConvNet:
        set_deterministic(seed)
        return ShallowConvNet(22, 4, N_TIMES_WIDE).to(device)

    def record(arm, seed, probs, train_s, temp):
        acc, n_correct = score(probs, y_eval_idx)
        np.save(PRED_DIR / f"s{subject}_{arm}_seed{seed}.npy", probs.astype(np.float32))
        append_row({"subject": subject, "arm": arm, "seed": seed, "acc": acc, "n_correct": n_correct,
                    "gpu_temp_c": temp, "train_s": train_s})
        return acc

    for seed in seeds:
        # (training input key, trained-arm, adabn-arm, groupwise?)
        plans = [("raw", "cnn_none", "cnn_adabn", False), ("ra", "cnn_ra", "cnn_ra_adabn", False),
                 ("ra", None, "cnn_ra_sat", True)]
        for key, arm_plain, arm_adapt, sat in plans:
            needed = [a for a in (arm_plain, arm_adapt) if a and (subject, a, seed) not in done]
            if not needed:
                continue
            status = wait_until_healthy()
            Xc, Xe = tensors(key)
            t0 = time.time()
            model = train_model(fresh(seed), Xc, yc, CFG, seed=seed, groups=runs_c if sat else None)
            train_s = time.time() - t0
            temp = status.temp_c if status else np.nan
            msg = []
            if arm_plain:
                msg.append(f"{arm_plain}={record(arm_plain, seed, predict_proba(model, Xe), train_s, temp):.4f}")
            if sat:
                probs = predict_proba_groupwise(copy.deepcopy(model), Xe, runs_e)
            else:
                probs = predict_proba(adapt_bn_statistics(copy.deepcopy(model), Xe), Xe)
            msg.append(f"{arm_adapt}={record(arm_adapt, seed, probs, train_s, temp):.4f}")
            print(f"    seed {seed} [{train_s:5.1f}s, GPU {temp}C] " + "  ".join(msg), flush=True)
            del model
            torch.cuda.empty_cache()


def run_permuted_control(subject: int, inputs: dict, y_cal_idx, y_eval_idx, done, device) -> None:
    if (subject, "ctrl_permuted", 0) in done:
        return
    wait_until_healthy()
    y_perm = np.random.default_rng(1000 + subject).permutation(y_cal_idx)
    Xc, Xe = (torch.as_tensor(a, device=device) for a in inputs["ra"])
    set_deterministic(0)
    model = train_model(ShallowConvNet(22, 4, N_TIMES_WIDE).to(device), Xc,
                        torch.as_tensor(y_perm, device=device), CFG, seed=0)
    probs = predict_proba(adapt_bn_statistics(model, Xe), Xe)
    acc, n_correct = score(probs, y_eval_idx)
    append_row({"subject": subject, "arm": "ctrl_permuted", "seed": 0, "acc": acc, "n_correct": n_correct,
                "gpu_temp_c": np.nan, "train_s": np.nan})
    print(f"    ctrl_permuted acc={acc:.4f} (must be near 0.25)", flush=True)


def run_subject(subject: int, seeds, device, skip_control: bool) -> None:
    done = load_done()
    wide = load_iv2a_subject_window(subject, **WIDE)
    check_sessions(wide)
    y_cal_idx = encode_labels(wide.y_calib, IV2A_CLASSES)
    y_eval_idx = encode_labels(wide.y_eval, IV2A_CLASSES)  # quarantined: passed to score() only

    run_csp_controls(subject, wide, wide.y_calib, y_eval_idx, done)

    scale = float(wide.X_calib.std())  # T-only constant, applied to both sessions
    Xc_ra, Xe_ra = align_per_session(wide.X_calib, wide.X_eval)
    inputs = {
        "raw": ((wide.X_calib / scale).astype(np.float32), (wide.X_eval / scale).astype(np.float32)),
        "ra": (Xc_ra.astype(np.float32), Xe_ra.astype(np.float32)),
    }
    for key, (a, b) in inputs.items():
        for arr in (a, b):
            s = float(arr.std())
            if not 0.1 <= s <= 10:
                raise AssertionError(f"input '{key}' std {s:.3g} outside [0.1, 10] - scaling bug")
    run_cnn(subject, inputs, (wide.runs_calib, wide.runs_eval), y_cal_idx, y_eval_idx, seeds, done, device)
    if not skip_control:
        run_permuted_control(subject, inputs, y_cal_idx, y_eval_idx, done, device)
    del wide, inputs
    gc.collect()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=IV2A_SUBJECTS)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--skip-control", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={device} cfg={CFG} gpu={query_gpu()}", flush=True)
    t0 = time.time()
    for subject in args.subjects:
        print(f"\n=== subject {subject} ===", flush=True)
        run_subject(subject, args.seeds, device, args.skip_control)
        mem_report(f"after s{subject}")
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
