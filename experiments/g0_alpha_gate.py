"""G0 gate: does a trial's own pre-cue alpha predict whether it is decoded correctly, beyond run index?

This gates both state-conditioning ideas from the ideation round
(reports/Alien space ideation within-session decline.md): A, covariate residualisation, and B,
a FiLM decoder conditioned on alpha. Neither idea is worth building unless pre-cue alpha carries
trial-level information about decoding success that "which run is this" does not already carry.

DATA: session T only, so E stays untouched for later screens. Each subject is analysed alone.
  Correctness: leave-one-run-out CSP+LDA on the RA'd T session (the D5 estimator), per trial.
  Alpha: log10 of 8-13 Hz power at the motor channels, pre-cue window -0.8..-0.3 s (the
    primary baseline in char_01). It is z-scored within the session.
  Per subject, two unpenalised logistic models are fitted for correctness:
    M0 ~ run dummies;   M1 ~ run dummies + z(alpha).
  The test statistic is the likelihood-ratio deviance difference.

RULES (fixed before the first run):
  PASS iff the combined LR statistic (sum over 9 subjects, chi2 with 9 df) gives p < 0.05
     AND the alpha coefficient is negative (more alpha means more errors) in >= 6/9 subjects.
  CONTROL (the run is UNINFORMATIVE if it fails): alpha permuted within each run (seed 0)
     must give a combined p >= 0.05.
  Reported (descriptive, not gating): the same test on E, with correctness from the T->E
     CSP+LDA model.

    .venv\\Scripts\\python experiments\\g0_alpha_gate.py     # ~2 min, CPU
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from experiments.char_01_iv2a_signal import BASELINE_INTERIOR_S, MOTOR_IDX, _band_power_trials, _window_slice  # noqa: E402
from experiments.r3_convnet_session_norm import align_per_session  # noqa: E402
from src.adaptation.riemannian_align import RiemannianAlignment  # noqa: E402
from src.baselines import make_csp_lda  # noqa: E402
from src.config import FMAX, FMIN, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.datasets import load_iv2a_subject_wide_window, load_iv2a_subject_window  # noqa: E402

OUT_CSV = RESULTS_DIR / "g0_alpha_gate.csv"
ALPHA = (8.0, 13.0)
N_RUNS = 6
EXPECTED_RUNS = np.repeat(np.arange(N_RUNS), 48)


def log_alpha(X_wide: np.ndarray) -> np.ndarray:
    a = np.log10(_band_power_trials(X_wide, ALPHA, _window_slice(BASELINE_INTERIOR_S))[:, MOTOR_IDX].mean(1))
    return (a - a.mean()) / a.std()


def loro_correct(X: np.ndarray, y: np.ndarray, runs: np.ndarray) -> np.ndarray:
    X_al = RiemannianAlignment().fit_transform(X)
    out = np.empty(len(y), dtype=bool)
    for r in range(N_RUNS):
        m = runs == r
        out[m] = make_csp_lda().fit(X_al[~m], y[~m]).predict(X_al[m]) == y[m]
    return out


def lr_test(correct: np.ndarray, alpha: np.ndarray, runs: np.ndarray) -> tuple[float, float]:
    """Deviance difference M0 (run dummies) vs M1 (+ alpha), and M1's alpha coefficient."""
    dummies = (runs[:, None] == np.arange(1, N_RUNS)[None, :]).astype(float)

    def loglik(F: np.ndarray) -> tuple[float, np.ndarray]:
        m = LogisticRegression(penalty=None, max_iter=5000, tol=1e-10).fit(F, correct)
        p = np.clip(m.predict_proba(F)[:, 1], 1e-12, 1 - 1e-12)
        return float(np.sum(correct * np.log(p) + (~correct) * np.log(1 - p))), m.coef_[0]

    ll0, _ = loglik(dummies)
    ll1, coef = loglik(np.column_stack([dummies, alpha]))
    return 2 * (ll1 - ll0), float(coef[-1])


def permute_within_runs(a: np.ndarray, runs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = a.copy()
    for r in range(N_RUNS):
        idx = np.flatnonzero(runs == r)
        out[idx] = a[rng.permutation(idx)]
    return out


def main() -> int:
    rows = []
    rng = np.random.default_rng(0)
    for s in IV2A_SUBJECTS:
        nar = load_iv2a_subject_window(s, FMIN, FMAX, TMIN, TMAX)
        wide = load_iv2a_subject_wide_window(s)
        for runs in (nar.runs_calib, nar.runs_eval):
            if not np.array_equal(runs, EXPECTED_RUNS):
                raise AssertionError("trials are not stored run by run")
        if not (np.array_equal(wide.y_calib, nar.y_calib) and np.array_equal(wide.y_eval, nar.y_eval)):
            raise AssertionError("wide and narrow loaders disagree on trial order")
        a_t, a_e = log_alpha(wide.X_calib), log_alpha(wide.X_eval)
        c_t = loro_correct(nar.X_calib, nar.y_calib, EXPECTED_RUNS)
        Xa_tr, Xa_te = align_per_session(nar.X_calib, nar.X_eval)
        c_e = make_csp_lda().fit(Xa_tr, nar.y_calib).predict(Xa_te) == nar.y_eval
        for label, c, a in (("T", c_t, a_t), ("T_control", c_t, permute_within_runs(a_t, EXPECTED_RUNS, rng)),
                            ("E_descriptive", c_e, a_e)):
            lr, coef = lr_test(c, a, EXPECTED_RUNS)
            rows.append({"subject": s, "analysis": label, "lr_stat": lr, "alpha_coef": coef, "acc": c.mean()})
        print(f"  s{s}: T lr {rows[-3]['lr_stat']:.2f} coef {rows[-3]['alpha_coef']:+.3f}  |  "
              f"E lr {rows[-1]['lr_stat']:.2f} coef {rows[-1]['alpha_coef']:+.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n=== G0: pre-cue alpha -> trial correctness, beyond run index ===")
    res = {}
    for label, q in df.groupby("analysis"):
        stat = q.lr_stat.sum()
        p = float(chi2.sf(stat, len(q)))
        neg = int((q.alpha_coef < 0).sum())
        res[label] = (p, neg)
        print(f"  {label:14s} combined LR {stat:6.2f} (chi2 df {len(q)}), p = {p:.4f}, coef < 0 in {neg}/9, "
              f"mean coef {q.alpha_coef.mean():+.3f}")
    if res["T_control"][0] < 0.05:
        print("VERDICT: UNINFORMATIVE (control failed)")
    else:
        p, neg = res["T"]
        print(f"VERDICT: G0 {'PASS' if p < 0.05 and neg >= 6 else 'FAIL'}")
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
