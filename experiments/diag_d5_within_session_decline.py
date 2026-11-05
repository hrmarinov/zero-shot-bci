"""D5 — why does accuracy fall across the evaluation session? (follow-up to D3)

D3 found cross-session accuracy declining over the 6 evaluation runs (CSP+LDA+RA 69.7% ->
62.0%, convnet 78.5% -> 69.6%) with no trend in the per-run covariance mean and with
run-wise re-centering *hurting*. Abstracted, the pattern is "performance decays with time on
task while first/second-order input statistics stay put" — the vigilance decrement of human
factors, and sensor degradation in instrumentation. Four hypotheses, each with the standard
marker from its field:

  H1 signal weakening (time-on-task): the task-related ERD shrinks over runs, so per-run
     decodability falls even when training and testing within the SAME session.
  H2 drowsiness / reduced vigilance: pre-cue (theta+alpha)/beta rises over runs (the classic
     EEG fatigue index).
  H3 sensor / artefact degradation: broadband power, high-beta/EMG-band power (25-38 Hz),
     amplitude outliers (>100 uV) or kurtosis rise over runs.
  H4 transfer-specific: same-session decodability stays flat while the T->E transfer
     declines (later runs differ from the calibration day in some other way).

INTERPRETATION RULES — fixed before the first run. For every measure: per-subject OLS slope
over runs 1-6, "first->last change" = 5 x mean slope, two-sided exact sign-flip p over the 9
subjects. A trend is "present" iff p < 0.10 (exploratory; many measures, no correction).
  H1 supported iff same-session leave-one-run-out (LORO) accuracy in E falls by >= 3 pp
     (present) — strengthened if the mu or beta ERD weakens (present, positive slope = less
     desynchronisation). Called "time-on-task, not session-specific" if LORO also falls in T.
  H2 supported iff baseline (theta+alpha)/beta rises (present) in E.
  H3 supported iff any of broadband log-power, 25-38 Hz relative power, outlier fraction or
     kurtosis rises (present) in E.
  H4 supported iff E LORO change is within +/-2 pp while T->E cross-session falls >= 3 pp.
Which marker tracks the decline is reported descriptively (within-subject correlation of
per-run marker with per-run accuracy).

POST-HOC measures (added after the first full run, exploratory, prefixed "posthoc_"):
  - directional covariance drift: Riemannian distance of each run's mean covariance from RUN 1's
    (D3 measured distance from the *session* mean, which is U-shaped under a linear drift and so
    cannot detect one — a design error in D3);
  - cue-window log power in theta 4-7 Hz and beta 15-30 Hz separately, to tell whether the
    broadband rise is only the alpha rise.

Estimators are imported from experiments/char_01_iv2a_signal.py (R1-B, audited); this script
asserts that its session-level ERD reproduces R1-B's CSV exactly before using them per run.

    .venv\\Scripts\\python experiments\\diag_d5_within_session_decline.py     # ~3 min, CPU
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
from pyriemann.geometry.distance import distance_riemann  # noqa: E402
from pyriemann.geometry.mean import mean_riemann  # noqa: E402
from scipy.stats import kurtosis, pearsonr  # noqa: E402

from experiments.char_01_iv2a_signal import (  # noqa: E402
    BASELINE_INTERIOR_S,
    BASELINE_WIDE_S,
    BETA_BAND,
    CUE_WINDOW_S,
    HAND_CLASSES,
    MOTOR_IDX,
    MU_BAND,
    _band_power_trials,
    _erd_from_trial_powers,
    _window_slice,
)
from experiments.r3_convnet_session_norm import align_per_session  # noqa: E402
from src.adaptation.riemannian_align import RiemannianAlignment  # noqa: E402
from src.baselines import make_csp_lda  # noqa: E402
from src.config import FMAX, FMIN, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.datasets import load_iv2a_subject_wide_window, load_iv2a_subject_window  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402

OUT_CSV = RESULTS_DIR / "diag_d5_within_session_decline.csv"
CHAR_CSV = RESULTS_DIR / "char_iv2a_subject_summary.csv"
N_RUNS = 6
OUTLIER_UV = 100.0
THETA, ALPHA, BETA_LOW = (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)
EMG_BAND, BROAD_BAND = (25.0, 38.0), (4.0, 38.0)
ROWS: list[dict] = []
EXPECTED_RUNS = np.repeat(np.arange(N_RUNS), 48)


def add(subject: int, session: str, measure: str, run: int, value: float) -> None:
    ROWS.append({"subject": subject, "session": session, "measure": measure, "run": run, "value": value})


# --------------------------------------------------------------------------- decodability


def loro_accuracy(X: np.ndarray, y: np.ndarray, runs: np.ndarray) -> np.ndarray:
    """Same-session decodability per run: train on the other 5 runs, test the held-out one.

    The session is re-centred once on all its own (unlabelled) trials, as in Phase 2.
    """
    X_al = RiemannianAlignment().fit_transform(X)
    return np.array([
        (make_csp_lda().fit(X_al[runs != r], y[runs != r]).predict(X_al[runs == r]) == y[runs == r]).mean()
        for r in range(N_RUNS)
    ])


def cross_accuracy(X_tr, y_tr, X_te, y_te, runs_te) -> np.ndarray:
    Xa_tr, Xa_te = align_per_session(X_tr, X_te)
    pred = make_csp_lda().fit(Xa_tr, y_tr).predict(Xa_te)
    return np.array([(pred[runs_te == r] == y_te[runs_te == r]).mean() for r in range(N_RUNS)])


# --------------------------------------------------------------------------- physiology


def erd_per_run(X: np.ndarray, y: np.ndarray, runs: np.ndarray, band) -> np.ndarray:
    """Mean ERD over motor channels and ALL four classes, per run (ratio of mean powers)."""
    cue = _band_power_trials(X, band, _window_slice(CUE_WINDOW_S))
    base = _band_power_trials(X, band, _window_slice(BASELINE_INTERIOR_S))
    out = []
    for r in range(N_RUNS):
        m = runs == r
        erd = _erd_from_trial_powers(cue[m], base[m], y[m])
        out.append(np.mean([erd[c][MOTOR_IDX].mean() for c in erd]))
    return np.array(out)


def session_hand_erd(X: np.ndarray, y: np.ndarray) -> float:
    """R1-B's erd_mu_hands_mean, recomputed — the estimator-identity check."""
    erd = _erd_from_trial_powers(_band_power_trials(X, MU_BAND, _window_slice(CUE_WINDOW_S)),
                                 _band_power_trials(X, MU_BAND, _window_slice(BASELINE_INTERIOR_S)), y)
    return float(0.5 * sum(erd[c][MOTOR_IDX].mean() for c in HAND_CLASSES))


def baseline_markers(X: np.ndarray, runs: np.ndarray) -> dict[str, np.ndarray]:
    """Pre-cue fatigue index and noise markers per run (no labels needed)."""
    base_sl, cue_sl = _window_slice(BASELINE_WIDE_S), _window_slice(CUE_WINDOW_S)
    p = {name: _band_power_trials(X, band, base_sl)[:, MOTOR_IDX].mean(1)
         for name, band in (("theta", THETA), ("alpha", ALPHA), ("beta", BETA_LOW))}
    emg = _band_power_trials(X, EMG_BAND, cue_sl).mean(1)
    broad = _band_power_trials(X, BROAD_BAND, cue_sl).mean(1)
    peak = np.abs(X).max(axis=(1, 2))
    kurt = np.median(kurtosis(X, axis=-1), axis=1)
    out = {k: [] for k in ("fatigue_index_ta_over_b", "log_baseline_alpha", "log_broadband_power",
                           "emg_band_rel_power", "outlier_frac_100uv", "median_kurtosis")}
    for r in range(N_RUNS):
        m = runs == r
        out["fatigue_index_ta_over_b"].append((p["theta"][m].mean() + p["alpha"][m].mean()) / p["beta"][m].mean())
        out["log_baseline_alpha"].append(np.log10(p["alpha"][m].mean()))
        out["log_broadband_power"].append(np.log10(broad[m].mean()))
        out["emg_band_rel_power"].append(emg[m].mean() / broad[m].mean())
        out["outlier_frac_100uv"].append((peak[m] > OUTLIER_UV).mean())
        out["median_kurtosis"].append(np.median(kurt[m]))
    return {k: np.array(v) for k, v in out.items()}


def posthoc_markers(X_narrow: np.ndarray, X_wide: np.ndarray, runs: np.ndarray) -> dict[str, np.ndarray]:
    C = Covariances(estimator="oas").fit_transform(RiemannianAlignment().fit_transform(X_narrow))
    means = [mean_riemann(C[runs == r]) for r in range(N_RUNS)]
    cue_sl = _window_slice(CUE_WINDOW_S)
    theta = _band_power_trials(X_wide, (4.0, 7.0), cue_sl)[:, MOTOR_IDX].mean(1)
    beta = _band_power_trials(X_wide, (15.0, 30.0), cue_sl)[:, MOTOR_IDX].mean(1)
    return {
        "posthoc_cov_dist_from_run1": np.array([distance_riemann(m, means[0]) for m in means]),
        "posthoc_log_cue_theta": np.array([np.log10(theta[runs == r].mean()) for r in range(N_RUNS)]),
        "posthoc_log_cue_beta": np.array([np.log10(beta[runs == r].mean()) for r in range(N_RUNS)]),
    }


# --------------------------------------------------------------------------- per subject


def run_subject(s: int, char: pd.DataFrame) -> None:
    nar = load_iv2a_subject_window(s, FMIN, FMAX, TMIN, TMAX)
    wide = load_iv2a_subject_wide_window(s)
    for runs in (nar.runs_calib, nar.runs_eval):
        if not np.array_equal(runs, EXPECTED_RUNS):
            raise AssertionError("trials are not stored run by run - per-run slicing would be wrong")
    if not (np.array_equal(wide.y_calib, nar.y_calib) and np.array_equal(wide.y_eval, nar.y_eval)):
        raise AssertionError("wide and narrow loaders disagree on trial order")
    got, ref = session_hand_erd(wide.X_calib, wide.y_calib), char.loc[s, "erd_mu_hands_mean_calib"]
    if not np.isclose(got, ref, rtol=0, atol=1e-9):
        raise AssertionError(f"s{s}: ERD estimator {got} does not reproduce R1-B {ref}")

    sessions = {"T": (nar.X_calib, nar.y_calib, wide.X_calib, wide.y_calib),
                "E": (nar.X_eval, nar.y_eval, wide.X_eval, wide.y_eval)}
    cross = {"E": cross_accuracy(nar.X_calib, nar.y_calib, nar.X_eval, nar.y_eval, EXPECTED_RUNS),
             "T": cross_accuracy(nar.X_eval, nar.y_eval, nar.X_calib, nar.y_calib, EXPECTED_RUNS)}
    for sess, (Xn, yn, Xw, yw) in sessions.items():
        measures = {
            "acc_loro_same_session": loro_accuracy(Xn, yn, EXPECTED_RUNS),
            "acc_cross_session": cross[sess],  # E: trained on T; T: trained on E
            "erd_mu": erd_per_run(Xw, yw, EXPECTED_RUNS, MU_BAND),
            "erd_beta": erd_per_run(Xw, yw, EXPECTED_RUNS, BETA_BAND),
            **baseline_markers(Xw, EXPECTED_RUNS),
            **posthoc_markers(Xn, Xw, EXPECTED_RUNS),
        }
        for name, vals in measures.items():
            for r, v in enumerate(vals):
                add(s, sess, name, r, float(v))


# --------------------------------------------------------------------------- analysis


def trend_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    runs = np.arange(N_RUNS)
    for (sess, meas), g in df.groupby(["session", "measure"]):
        piv = g.pivot(index="subject", columns="run", values="value")
        slopes = np.array([np.polyfit(runs, piv.loc[s].to_numpy(), 1)[0] for s in piv.index])
        rows.append({"session": sess, "measure": meas, "first": piv[0].mean(), "last": piv[N_RUNS - 1].mean(),
                     "change_5x_slope": 5 * slopes.mean(), "n_up": int((slopes > 0).sum()),
                     "p_signflip": sign_flip_permutation_test(slopes, "two-sided")})
    return pd.DataFrame(rows)


def tracking_table(df: pd.DataFrame, session: str = "E") -> pd.DataFrame:
    """Within-subject (per-subject demeaned) correlation of each marker with per-run accuracy."""
    g = df[df.session == session].pivot_table(index=["subject", "run"], columns="measure", values="value")
    g = g - g.groupby(level="subject").transform("mean")
    rows = []
    for acc in ("acc_cross_session", "acc_loro_same_session"):
        for m in g.columns:
            if m.startswith("acc_"):
                continue
            r, p = pearsonr(g[acc], g[m])
            rows.append({"accuracy": acc, "marker": m, "r_within": r, "p_uncorrected": p})
    return pd.DataFrame(rows)


def verdicts(t: pd.DataFrame) -> dict[str, str]:
    def get(sess, meas):
        return t[(t.session == sess) & (t.measure == meas)].iloc[0]

    def present(r) -> bool:
        return r.p_signflip < 0.10

    lo_e, lo_t, cr_e = get("E", "acc_loro_same_session"), get("T", "acc_loro_same_session"), get("E", "acc_cross_session")
    erd_weakens = any(present(get("E", m)) and get("E", m).change_5x_slope > 0 for m in ("erd_mu", "erd_beta"))
    h1 = present(lo_e) and 100 * lo_e.change_5x_slope <= -3
    h1_txt = "SUPPORTED" if h1 else "not supported"
    if h1:
        h1_txt += (" + ERD weakens" if erd_weakens else " (ERD trend not present)")
        h1_txt += ("; also in T -> time-on-task" if present(lo_t) and lo_t.change_5x_slope < 0 else "; not in T")
    fi = get("E", "fatigue_index_ta_over_b")
    noise = [m for m in ("log_broadband_power", "emg_band_rel_power", "outlier_frac_100uv", "median_kurtosis")
             if present(get("E", m)) and get("E", m).change_5x_slope > 0]
    h4 = abs(100 * lo_e.change_5x_slope) <= 2 and 100 * cr_e.change_5x_slope <= -3 and present(cr_e)
    return {
        "H1 signal weakening": h1_txt,
        "H2 drowsiness (fatigue index rises)": "SUPPORTED" if present(fi) and fi.change_5x_slope > 0 else "not supported",
        "H3 sensor/artefact degradation": f"SUPPORTED ({', '.join(noise)})" if noise else "not supported",
        "H4 transfer-specific": "SUPPORTED" if h4 else "not supported",
    }


def main() -> int:
    char = pd.read_csv(CHAR_CSV).set_index("subject")
    for s in IV2A_SUBJECTS:
        print(f"=== subject {s} ===", flush=True)
        run_subject(s, char)
    df = pd.DataFrame(ROWS)
    df.to_csv(OUT_CSV, index=False)
    print("[PASS] ERD estimator reproduces R1-B erd_mu_hands_mean_calib for all 9 subjects (atol 1e-9)")
    t = trend_table(df)
    pd.set_option("display.width", 200)
    print("\n=== trends over runs 1-6 (9 subjects; change = 5 x mean per-run slope) ===")
    show = t.copy()
    for c in ("first", "last", "change_5x_slope"):
        show[c] = show[c].round(4)
    print(show.sort_values(["measure", "session"]).to_string(index=False))
    print("\n=== which marker tracks per-run accuracy in E (within-subject r, exploratory) ===")
    print(tracking_table(df).round(3).sort_values(["accuracy", "p_uncorrected"]).to_string(index=False))
    print("\n=== VERDICTS (rules fixed in the module docstring) ===")
    for k, v in verdicts(t).items():
        print(f"  {k:38s} {v}")
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
