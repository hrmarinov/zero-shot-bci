"""R5 screen: does re-centering on a reference that follows the within-session drift help?

Motivation: D5 (docs/diag_gates_d1_d4.md) found a slow, monotone, label-free drift (rising
alpha / broadband power) and accuracy falling over runs in both sessions. Session-wise RA
whitens every trial by one reference; here trial t is whitened by a reference built near t
(src/adaptation/drift_tracking.py). The same tracker is applied to T (training) and E (test),
each session on its own trials only, no labels, no other subject.

ARMS (pre-declared, one setting each, no tuning):
  static      session-wise RA (the baseline; must reproduce Phase 2 CSP and R3 cnn_ra seed 0)
  kernel      Gaussian-weighted Riemannian mean over trial index, sigma = 48 trials (one run);
              NON-causal - the most generous tracker, used as the gate
  causal_cum  inductive running mean of trials 1..t (causal control, no forgetting)
  causal_ema  running mean that becomes an EMA with half-life 48 trials (deployable tracker)
Models: CSP+LDA on the narrow window; ShallowConvNet (R3 cnn_ra config) on the wide window,
5 seeds, subject score = 5-seed mean. Static CNN = the cached R3 cnn_ra predictions.

RULES — fixed before the first run:
  G1 tracking family (gate): OPEN iff kernel - static >= +1.0 pp subject mean with two-sided
     sign-flip p < 0.10 for CSP+LDA or for the CNN. Otherwise the tracking family is CLOSED
     (no causal tracker can beat the non-causal one by construction of the screen).
  G2 deployable value: causal_ema - causal_cum >= +1.0 pp with two-sided p < 0.10 (same model),
     i.e. forgetting helps over a causal tracker without forgetting. causal_ema - static is
     reported as the price of causality (transductive static uses all of E, incl. the future).
  ADOPT (for a later confirmatory step only): G1 and G2 both hold on the same model AND
     causal_ema - static > 0 with Holm-adjusted one-sided p < 0.05 over the 4 contrasts
     {kernel, causal_ema} x {csp, cnn} vs static.
  Mechanism (exploratory): late (runs 3-5) minus early (runs 0-2) accuracy per arm - does
     tracking flatten the D5 decline?
  VALIDITY (screen is INVALID if any fails):
     V1 static CSP reproduces Phase 2 per subject exactly; static CNN seed 0 retrained here
        reproduces R3 cnn_ra seed 0 exactly.
     V2 planted drift on subjects 1, 3, 7 (CSP): E trial t is mixed by expm((t/287) S), S a
        random symmetric matrix with max |eigenvalue| 1.0 (seed 0). Static must drop >= 5 pp
        vs unplanted, and kernel must recover >= 50% of that drop.
     V3 trials are in chronological run order (runs non-decreasing) in both sessions.

    .venv\\Scripts\\python experiments\\r5_drift_tracking.py          # ~60 min (144 GPU trainings)
    .venv\\Scripts\\python experiments\\r5_drift_tracking.py --csp-only   # ~5 min
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from experiments.r3_convnet_session_norm import CFG, N_TIMES_WIDE, WIDE, align_per_session, phase2_reference  # noqa: E402
from src.adaptation.convnet import ShallowConvNet, encode_labels, predict_proba, set_deterministic, train_model  # noqa: E402
from src.adaptation.drift_tracking import causal_references, kernel_references, trial_covariances, whiten_per_trial  # noqa: E402
from src.baselines import make_csp_lda  # noqa: E402
from src.config import DATA_DIR, FMAX, FMIN, IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.datasets import load_iv2a_subject_window  # noqa: E402
from src.evaluate import holm_adjust, sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402

OUT_CSV = RESULTS_DIR / "r5_drift_tracking.csv"
R3_CSV = RESULTS_DIR / "r3_convnet_session_norm.csv"
R3_PRED_DIR = DATA_DIR / "cache" / "r3_preds"
PRED_DIR = DATA_DIR / "cache" / "r5_preds"
SIGMA = 48.0
HALF_LIFE = 48.0
SEEDS = (0, 1, 2, 3, 4)
TRACKED_ARMS = ("kernel", "causal_cum", "causal_ema")
ARMS = ("static",) + TRACKED_ARMS
PLANTED_SUBJECTS = (1, 3, 7)
PLANTED_MAX_EIG = 1.0
GATE_PP = 1.0
ROWS: list[dict] = []


def row(subject, model: str, arm: str, metric: str, value: float) -> None:
    ROWS.append({"subject": subject, "model": model, "arm": arm, "metric": metric, "value": value})


def track(X: np.ndarray, arm: str) -> np.ndarray:
    """Whiten one session (chronological) by the arm's per-trial references."""
    C = trial_covariances(X)
    refs = {
        "kernel": lambda: kernel_references(C, SIGMA),
        "causal_cum": lambda: causal_references(C),
        "causal_ema": lambda: causal_references(C, HALF_LIFE),
    }[arm]()
    return whiten_per_trial(X, refs)


def session_inputs(Xc: np.ndarray, Xe: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {"static": align_per_session(Xc, Xe)}
    for arm in TRACKED_ARMS:
        out[arm] = (track(Xc, arm), track(Xe, arm))
    return out


def record_accuracy(s: int, model: str, arm: str, correct: np.ndarray, runs: np.ndarray) -> None:
    row(s, model, arm, "acc", correct.mean())
    for r in range(6):
        row(s, model, arm, f"run{r}_acc", correct[runs == r].mean())


def check_chronological(d) -> None:
    for runs in (d.runs_calib, d.runs_eval):
        if np.any(np.diff(runs) < 0) or not np.all(np.bincount(runs) == 48):
            raise AssertionError("V3: trials are not in chronological run order")


# --------------------------------------------------------------------------- CSP+LDA


def planted_drift(X: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((X.shape[1], X.shape[1]))
    S = (A + A.T) / 2
    S *= PLANTED_MAX_EIG / np.abs(np.linalg.eigvalsh(S)).max()
    n = len(X)
    return np.stack([expm((t / (n - 1)) * S) @ X[t] for t in range(n)])


def csp_subject(s: int, nar, yc, ye) -> None:
    inputs = session_inputs(nar.X_calib, nar.X_eval)
    for arm, (Xc, Xe) in inputs.items():
        correct = make_csp_lda().fit(Xc, yc).predict(Xe) == ye
        record_accuracy(s, "csp_lda", arm, correct, nar.runs_eval)
    ref = phase2_reference(s)
    got = next(r["value"] for r in ROWS if r["subject"] == s and r["model"] == "csp_lda"
               and r["arm"] == "static" and r["metric"] == "acc")
    if got != ref:
        raise AssertionError(f"V1: s{s} static CSP {got} does not reproduce Phase 2 {ref}")
    if s in PLANTED_SUBJECTS:
        Xe_p = planted_drift(nar.X_eval)
        for arm in ("static", "kernel"):
            Xc = inputs[arm][0]
            Xe = align_per_session(nar.X_calib, Xe_p)[1] if arm == "static" else track(Xe_p, arm)
            row(s, "csp_lda", arm, "planted_acc", (make_csp_lda().fit(Xc, yc).predict(Xe) == ye).mean())


# --------------------------------------------------------------------------- CNN


def cnn_probs(Xc: np.ndarray, yc: np.ndarray, Xe: np.ndarray, seed: int, device) -> np.ndarray:
    wait_until_healthy()
    set_deterministic(seed)
    model = train_model(ShallowConvNet(22, 4, N_TIMES_WIDE).to(device),
                        torch.as_tensor(Xc.astype(np.float32), device=device),
                        torch.as_tensor(yc, device=device), CFG, seed=seed)
    probs = predict_proba(model, torch.as_tensor(Xe.astype(np.float32), device=device))
    del model
    torch.cuda.empty_cache()
    return probs


def cnn_subject(s: int, wide, yc, ye, device) -> None:
    inputs = session_inputs(wide.X_calib, wide.X_eval)
    r3 = pd.read_csv(R3_CSV)
    ref0 = r3[(r3.subject == s) & (r3.arm == "cnn_ra") & (r3.seed == 0)].acc.iloc[0]
    Xc_static, Xe_static = inputs["static"]
    check = (cnn_probs(Xc_static, yc, Xe_static, 0, device).argmax(1) == ye).mean()
    if check != ref0:
        raise AssertionError(f"V1: s{s} static CNN seed 0 {check} does not reproduce R3 {ref0}")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        correct = []
        for seed in SEEDS:
            if arm == "static":
                probs = np.load(R3_PRED_DIR / f"s{s}_cnn_ra_seed{seed}.npy")
            else:
                path = PRED_DIR / f"s{s}_{arm}_seed{seed}.npy"
                if path.exists():
                    probs = np.load(path)
                else:
                    probs = cnn_probs(inputs[arm][0], yc, inputs[arm][1], seed, device)
                    np.save(path, probs.astype(np.float32))
                print(f"  s{s} {arm} seed{seed}: {(probs.argmax(1) == ye).mean():.4f}", flush=True)
            correct.append(probs.argmax(1) == ye)
        record_accuracy(s, "cnn", arm, np.mean(correct, axis=0), wide.runs_eval)


# --------------------------------------------------------------------------- summary


def summarise(df: pd.DataFrame) -> None:
    def per_subject(model: str, arm: str, metric: str = "acc") -> pd.Series:
        q = df[(df.model == model) & (df.arm == arm) & (df.metric == metric)]
        return q.set_index("subject").value.sort_index()

    def late_minus_early(model: str, arm: str) -> pd.Series:
        early = sum(per_subject(model, arm, f"run{r}_acc") for r in (0, 1, 2)) / 3
        late = sum(per_subject(model, arm, f"run{r}_acc") for r in (3, 4, 5)) / 3
        return 100 * (late - early)

    print("\n=== R5 drift tracking: mean accuracy (pp vs static, subjects up, two-sided p) ===")
    verdict, holm_in = {}, {}
    for model in sorted(df.model.unique()):
        base = per_subject(model, "static")
        print(f"{model}: static {100*base.mean():.2f}%  (late-early {late_minus_early(model, 'static').mean():+.2f}pp)")
        for arm in TRACKED_ARMS:
            d = 100 * (per_subject(model, arm) - base)
            p = sign_flip_permutation_test(d.to_numpy(), "two-sided")
            print(f"  {arm:11s} {100*per_subject(model, arm).mean():6.2f}%  {d.mean():+5.2f}pp  "
                  f"{(d > 0).sum()}/9 up  p={p:.3f}   late-early {late_minus_early(model, arm).mean():+.2f}pp")
            if arm in ("kernel", "causal_ema"):
                holm_in[(model, arm)] = (d.mean(), sign_flip_permutation_test(d.to_numpy(), "greater"))
        g1 = 100 * (per_subject(model, "kernel") - base)
        g2 = 100 * (per_subject(model, "causal_ema") - per_subject(model, "causal_cum"))
        p1 = sign_flip_permutation_test(g1.to_numpy(), "two-sided")
        p2 = sign_flip_permutation_test(g2.to_numpy(), "two-sided")
        verdict[model] = (g1.mean() >= GATE_PP and p1 < 0.10, g2.mean() >= GATE_PP and p2 < 0.10)
        print(f"  G1 kernel-static {g1.mean():+.2f}pp p={p1:.3f} -> {'OPEN' if verdict[model][0] else 'closed'};"
              f"  G2 ema-cum {g2.mean():+.2f}pp p={p2:.3f} -> {'OPEN' if verdict[model][1] else 'closed'}")
    adj = dict(zip(holm_in, holm_adjust([v[1] for v in holm_in.values()])))
    for key, (m, _) in holm_in.items():
        print(f"  Holm one-sided {key}: {m:+.2f}pp p_adj={adj[key]:.3f}")
    g1_any = any(v[0] for v in verdict.values())
    print(f"\nG1 tracking family: {'OPEN' if g1_any else 'CLOSED'}")
    for model, (g1, g2) in verdict.items():
        adopt = g1 and g2 and adj.get((model, "causal_ema"), 1.0) < 0.05 and holm_in[(model, "causal_ema")][0] > 0
        print(f"  {model}: G1 {'OPEN' if g1 else 'closed'}, G2 {'OPEN' if g2 else 'closed'}, ADOPT {adopt}")

    pl = df[df.metric == "planted_acc"]
    if len(pl):
        print("\nV2 planted drift (CSP):")
        ok = True
        for s in PLANTED_SUBJECTS:
            unpl = per_subject("csp_lda", "static").loc[s]
            st = pl[(pl.subject == s) & (pl.arm == "static")].value.iloc[0]
            ke = pl[(pl.subject == s) & (pl.arm == "kernel")].value.iloc[0]
            drop, rec = 100 * (unpl - st), 100 * (ke - st)
            passed = drop >= 5.0 and rec >= 0.5 * drop
            ok &= passed
            print(f"  s{s}: static drop {drop:+.1f}pp, kernel recovers {rec:+.1f}pp -> {'pass' if passed else 'FAIL'}")
        print(f"  V2 {'PASS' if ok else 'FAIL - screen INVALID'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csp-only", action="store_true")
    ap.add_argument("--subjects", type=int, nargs="*", default=list(IV2A_SUBJECTS))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for s in args.subjects:
        print(f"=== subject {s} ===", flush=True)
        nar = load_iv2a_subject_window(s, FMIN, FMAX, TMIN, TMAX)
        check_chronological(nar)
        yc, ye = encode_labels(nar.y_calib, IV2A_CLASSES), encode_labels(nar.y_eval, IV2A_CLASSES)
        csp_subject(s, nar, yc, ye)
        if not args.csp_only:
            wide = load_iv2a_subject_window(s, **WIDE)
            check_chronological(wide)
            if not np.array_equal(encode_labels(wide.y_eval, IV2A_CLASSES), ye):
                raise AssertionError("narrow and wide loaders disagree on trial order")
            cnn_subject(s, wide, yc, ye, device)
        pd.DataFrame(ROWS).to_csv(OUT_CSV, index=False)
    df = pd.DataFrame(ROWS)
    if len(args.subjects) == len(IV2A_SUBJECTS):
        summarise(df)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
