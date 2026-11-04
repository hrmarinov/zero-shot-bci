"""Gating diagnostics D1-D4 from reports/Cross field BCI calibration methods.md.

Each diagnostic decides whether a whole *family* of cross-field methods has headroom on
IV-2a T->E before any member of it is built. Evaluation labels are used ONLY to compute
upper bounds ("oracles"); nothing here is a deployable method or a reportable accuracy
claim. Baselines are reproduced bit-exactly inside the run (Phase 2 CSP+LDA+RA per subject;
R3 cnn_ra seed 0 per subject) so a configuration drift cannot masquerade as headroom.

GATES — fixed before the first run (a family is CLOSED unless its gate opens):
  D1 rotation family (pseudo-label RPA/TSA, EM-CMLLR, Wasserstein Procrustes, ...):
     OPEN iff the *in-sample* oracle rotation (true E class means, the most generous
     bound) gains >= +2.0 pp mean over the aligned baseline for at least one of
     {csp_lda, mdm, ts_lr, cnn_ra}. The cross-fitted (2-fold) oracle is reported as the
     realistic bound. VALIDITY: on control subjects a planted random rotation of E must
     drop the baseline >= 10 pp and the oracle must recover to within 1 pp of the
     unplanted oracle; otherwise D1 is INVALID.
  D2 class-conditional family (MAP class means, adaptive MDM, T3A prototypes,
     pseudo-label ComBat): OPEN iff in-sample oracle E class means beat the label-free
     pooled-mean shift (PMean) by >= +2.0 pp mean on CSP+LDA, or beat T means by
     >= +2.0 pp mean on MDM.
  D3 online-tracking family (Vidaurre PMean tracking, recursive re-centering, Kalman,
     exponential moving standardisation): OPEN iff |mean(late runs 3-5 - early runs 0-2)|
     >= 2.0 pp with two-sided sign-flip p < 0.10 for cnn_ra_adabn or csp_ra, OR run-wise
     re-centering of E beats session-wise re-centering by >= +1.0 pp mean (CSP+LDA).
  D4a self-training / pseudo-label / distillation family: OPEN iff the 5-seed
     cnn_ra_adabn ensemble's most-confident 50% of E trials are >= 80% correct on the
     subject mean AND >= 70% on the WEAK group {2, 5, 6}.
  D4b frequency-warping (VTLN) family: OPEN for a subject iff its motor-strip mu-peak
     frequency differs between sessions by >= 0.5 Hz (R1-B estimates).

    .venv\\Scripts\\python experiments\\diag_gates_d1_d4.py        # ~25 min (CPU + 9 short GPU trainings)
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
from pyriemann.classification import MDM  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.geometry.distance import distance_riemann  # noqa: E402
from pyriemann.geometry.mean import mean_riemann  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from scipy.stats import ortho_group, spearmanr  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from experiments.r3_convnet_session_norm import CFG, N_TIMES_WIDE, WIDE, align_per_session  # noqa: E402
from src.adaptation.convnet import ShallowConvNet, encode_labels, predict_proba, set_deterministic, train_model  # noqa: E402
from src.adaptation.riemannian_align import RiemannianAlignment  # noqa: E402
from src.adaptation.rotation_oracle import fit_orthogonal_congruence  # noqa: E402
from src.baselines import make_csp_lda  # noqa: E402
from src.config import DATA_DIR, FMAX, FMIN, IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.datasets import load_iv2a_subject_window  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402

OUT_CSV = RESULTS_DIR / "diag_gates_iv2a.csv"
R3_CSV = RESULTS_DIR / "r3_convnet_session_norm.csv"
PHASE2_CSV = RESULTS_DIR / "phase2_riemannian_alignment_iv2a.csv"
CHAR_CSV = RESULTS_DIR / "char_iv2a_subject_summary.csv"
PRED_DIR = DATA_DIR / "cache" / "r3_preds"
WEAK = (2, 5, 6)
PLANTED_CONTROL_SUBJECTS = (1, 3, 7)
GATE_PP = 2.0
ROWS: list[dict] = []


def row(diag: str, subject, model: str, condition: str, value: float) -> None:
    ROWS.append({"diag": diag, "subject": subject, "model": model, "condition": condition, "value": value})


# --------------------------------------------------------------------------- helpers


def covs(X: np.ndarray) -> np.ndarray:
    return Covariances(estimator="oas").fit_transform(X)


def class_means(C: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([mean_riemann(C[y == k]) for k in range(len(IV2A_CLASSES))])


def rotate(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.einsum("cd,ndt->nct", R, X)


def oracle_rotation(C_eval: np.ndarray, y_eval: np.ndarray, M_calib: np.ndarray) -> np.ndarray:
    """R with R · M_eval,k · R^T ≈ M_calib,k (true eval labels: an upper bound, not a method)."""
    R, _ = fit_orthogonal_congruence(class_means(C_eval, y_eval), M_calib)
    return R


def folds(y: np.ndarray):
    return list(StratifiedKFold(n_splits=2, shuffle=True, random_state=42).split(np.zeros(len(y)), y))


class LinearModels:
    """The three T-trained linear decoders, fixed; only the E input changes."""

    def __init__(self, Xc: np.ndarray, yc: np.ndarray) -> None:
        Cc = covs(Xc)
        self.csp = make_csp_lda().fit(Xc, yc)
        self.mdm = MDM(metric="riemann").fit(Cc, yc)
        self.tslr = make_pipeline(TangentSpace(metric="riemann"), LogisticRegression(max_iter=2000)).fit(Cc, yc)

    def predict(self, Xe: np.ndarray) -> dict[str, np.ndarray]:
        Ce = covs(Xe)
        return {"csp_lda": self.csp.predict(Xe), "mdm": self.mdm.predict(Ce), "ts_lr": self.tslr.predict(Ce)}


# --------------------------------------------------------------------------- D1


def d1_linear(s: int, Xc, yc, Xe, ye) -> None:
    models = LinearModels(Xc, yc)
    M_calib = class_means(covs(Xc), yc)
    Ce = covs(Xe)
    base = models.predict(Xe)
    phase2 = pd.read_csv(PHASE2_CSV).astype({"subject": str}).set_index("subject")["cross_session_aligned_acc"]
    if (base["csp_lda"] == ye).mean() != phase2.loc[str(s)]:
        raise AssertionError(f"s{s}: CSP+LDA+RA does not reproduce Phase 2")
    R_in = oracle_rotation(Ce, ye, M_calib)
    rot_in = models.predict(rotate(Xe, R_in))
    rot_cv = {k: np.empty_like(ye) for k in base}
    for a, b in folds(ye):
        R = oracle_rotation(Ce[a], ye[a], M_calib)
        for k, p in models.predict(rotate(Xe[b], R)).items():
            rot_cv[k][b] = p
    for k in base:
        row("D1", s, k, "baseline", (base[k] == ye).mean())
        row("D1", s, k, "oracle_insample", (rot_in[k] == ye).mean())
        row("D1", s, k, "oracle_2fold", (rot_cv[k] == ye).mean())
    if s in PLANTED_CONTROL_SUBJECTS:  # validity: can the tool find a rotation we planted?
        Q = ortho_group.rvs(Xe.shape[1], random_state=100 + s)
        Xp = rotate(Xe, Q)
        planted_base = models.predict(Xp)["csp_lda"]
        R_p = oracle_rotation(covs(Xp), ye, M_calib)
        row("D1ctrl", s, "csp_lda", "planted_baseline", (planted_base == ye).mean())
        row("D1ctrl", s, "csp_lda", "planted_oracle", (models.predict(rotate(Xp, R_p))["csp_lda"] == ye).mean())


def d1_cnn(s: int, wide_c, wide_e, yc, ye, device) -> None:
    wait_until_healthy()
    set_deterministic(0)
    Xc_t = torch.as_tensor(wide_c.astype(np.float32), device=device)
    model = train_model(ShallowConvNet(22, 4, N_TIMES_WIDE).to(device), Xc_t,
                        torch.as_tensor(yc, device=device), CFG, seed=0)

    def acc(X: np.ndarray) -> np.ndarray:
        return predict_proba(model, torch.as_tensor(X.astype(np.float32), device=device)).argmax(1)

    base = acc(wide_e)
    r3 = pd.read_csv(R3_CSV)
    ref = r3[(r3.subject == s) & (r3.arm == "cnn_ra") & (r3.seed == 0)].acc.iloc[0]
    if (base == ye).mean() != ref:
        raise AssertionError(f"s{s}: cnn_ra seed 0 {(base == ye).mean()} does not reproduce R3 {ref}")
    M_calib = class_means(covs(wide_c), yc)
    Ce = covs(wide_e)
    pred_in = acc(rotate(wide_e, oracle_rotation(Ce, ye, M_calib)))
    pred_cv = np.empty_like(ye)
    for a, b in folds(ye):
        pred_cv[b] = acc(rotate(wide_e[b], oracle_rotation(Ce[a], ye[a], M_calib)))
    row("D1", s, "cnn_ra", "baseline", (base == ye).mean())
    row("D1", s, "cnn_ra", "oracle_insample", (pred_in == ye).mean())
    row("D1", s, "cnn_ra", "oracle_2fold", (pred_cv == ye).mean())
    del model
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- D2


def _lda_predict(F: np.ndarray, means: np.ndarray, cov: np.ndarray, priors: np.ndarray) -> np.ndarray:
    W = np.linalg.solve(cov, means.T)  # (d, K)
    scores = F @ W - 0.5 * np.sum(means.T * W, axis=0) + np.log(priors)
    return scores.argmax(1)


def d2(s: int, Xc, yc, Xe, ye) -> None:
    csp = make_csp_lda().fit(Xc, yc).named_steps["csp"]
    Fc, Fe = csp.transform(Xc), csp.transform(Xe)
    lda = LinearDiscriminantAnalysis(solver="lsqr", store_covariance=True).fit(Fc, yc)
    cov, priors, mu_t = lda.covariance_, lda.priors_, lda.means_
    agree = (_lda_predict(Fe, mu_t, cov, priors) == make_csp_lda().fit(Xc, yc).predict(Xe)).mean()
    row("D2", s, "csp_lda", "own_lda_agreement_with_pipeline", agree)
    if agree < 0.98:
        raise AssertionError(f"s{s}: own LDA agrees with the pipeline on only {agree:.3f}")
    mu_e = np.stack([Fe[ye == k].mean(0) for k in range(4)])
    variants = {"T_means": mu_t, "pmean_shift": mu_t + (Fe.mean(0) - Fc.mean(0)), "oracle_insample": mu_e}
    for name, mu in variants.items():
        row("D2", s, "csp_lda", name, (_lda_predict(Fe, mu, cov, priors) == ye).mean())
    pred_cv = np.empty_like(ye)
    for a, b in folds(ye):
        mu_a = np.stack([Fe[a][ye[a] == k].mean(0) for k in range(4)])
        pred_cv[b] = _lda_predict(Fe[b], mu_a, cov, priors)
    row("D2", s, "csp_lda", "oracle_2fold", (pred_cv == ye).mean())

    Cc, Ce = covs(Xc), covs(Xe)
    mdm = MDM(metric="riemann").fit(Cc, yc)
    row("D2", s, "mdm", "T_means", (mdm.predict(Ce) == ye).mean())
    t_means = mdm.covmeans_.copy()
    mdm.covmeans_ = class_means(Ce, ye)
    row("D2", s, "mdm", "oracle_insample", (mdm.predict(Ce) == ye).mean())
    pred_cv = np.empty_like(ye)
    for a, b in folds(ye):
        mdm.covmeans_ = class_means(Ce[a], ye[a])
        pred_cv[b] = mdm.predict(Ce[b])
    mdm.covmeans_ = t_means
    row("D2", s, "mdm", "oracle_2fold", (pred_cv == ye).mean())


# --------------------------------------------------------------------------- D3


def d3_subject(s: int, Xc_raw, Xe_raw, yc, ye, runs_e) -> None:
    """Run-wise re-centering (label-free) and per-run covariance drift."""
    Xc_al, Xe_al = align_per_session(Xc_raw, Xe_raw)
    Xe_run = np.empty_like(Xe_raw)
    for r in np.unique(runs_e):
        idx = runs_e == r
        Xe_run[idx] = RiemannianAlignment().fit_transform(Xe_raw[idx])
    model = make_csp_lda().fit(Xc_al, yc)
    p_sess, p_run = model.predict(Xe_al), model.predict(Xe_run)
    row("D3", s, "csp_lda", "session_recentering", (p_sess == ye).mean())
    row("D3", s, "csp_lda", "runwise_recentering", (p_run == ye).mean())
    row("D3", s, "csp_lda", "runwise_pred_changed_frac", (p_sess != p_run).mean())  # no-op guard
    Ce = covs(Xe_al)
    # reference = the session mean of the SAME covariance estimates (OAS is not equivariant,
    # so the identity is a biased reference here - see docs/ceiling_analysis.md §5 Q2)
    ref = mean_riemann(Ce)
    dists = [distance_riemann(mean_riemann(Ce[runs_e == r]), ref) for r in range(6)]
    for r, d in enumerate(dists):
        row("D3", s, "cov_drift", f"run{r}_dist_to_session_mean", d)
    row("D3", s, "cov_drift", "spearman_run_vs_dist", spearmanr(range(6), dists).statistic)


def d3_runs_from_stored(s: int, ye, runs_e) -> None:
    for arm, seeds in (("cnn_ra_adabn", range(5)), ("csp_ra", (-1,))):
        accs = np.zeros(6)
        for k in seeds:
            p = np.load(PRED_DIR / f"s{s}_{arm}_seed{k}.npy")
            pred = p.argmax(1) if p.ndim == 2 else p
            accs += np.array([(pred[runs_e == r] == ye[runs_e == r]).mean() for r in range(6)])
        accs /= len(list(seeds))
        for r in range(6):
            row("D3", s, arm, f"run{r}_acc", accs[r])
        row("D3", s, arm, "late_minus_early", accs[3:].mean() - accs[:3].mean())


# --------------------------------------------------------------------------- D4


def d4_subject(s: int, ye) -> None:
    probs = np.mean([np.load(PRED_DIR / f"s{s}_cnn_ra_adabn_seed{k}.npy") for k in range(5)], axis=0)
    pred, conf = probs.argmax(1), probs.max(1)
    order = np.argsort(-conf, kind="stable")
    for q in (0.25, 0.5, 0.75, 1.0):
        top = order[: int(round(q * len(ye)))]
        row("D4", s, "cnn_ra_adabn_ens", f"top{int(q*100)}_acc", (pred[top] == ye[top]).mean())
    for k, name in enumerate(IV2A_CLASSES):
        row("D4", s, "cnn_ra_adabn_ens", f"class_acc_{name}", (pred[ye == k] == k).mean())


def d4_rasch(correct: dict[int, np.ndarray], labels: dict[int, np.ndarray]) -> None:
    """Rasch-style logistic GLM: P(correct) ~ subject ability + class easiness (diagnostic only)."""
    subjects = sorted(correct)
    X, y = [], []
    for i, s in enumerate(subjects):
        for c, k in zip(correct[s], labels[s]):
            X.append(np.r_[np.eye(len(subjects))[i], np.eye(4)[k][1:]])
            y.append(c)
    glm = LogisticRegression(C=1e6, max_iter=5000, fit_intercept=False).fit(np.array(X), np.array(y))
    for k, name in enumerate(IV2A_CLASSES[1:], start=1):
        row("D4", "all", "rasch", f"class_easiness_vs_{IV2A_CLASSES[0]}_{name}", glm.coef_[0][len(subjects) + k - 1])


# --------------------------------------------------------------------------- gates


def summarise() -> pd.DataFrame:
    df = pd.DataFrame(ROWS)
    df.to_csv(OUT_CSV, index=False)
    per = df[df.subject != "all"]

    def vals(diag, model, cond):
        sel = per[(per.diag == diag) & (per.model == model) & (per.condition == cond)]
        return sel.set_index("subject").value.sort_index()

    print("\n=== D1 oracle rotation (pp over aligned baseline; mean over 9 subjects) ===")
    d1_open = False
    for m in ("csp_lda", "mdm", "ts_lr", "cnn_ra"):
        b = vals("D1", m, "baseline")
        gin, gcv = 100 * (vals("D1", m, "oracle_insample") - b), 100 * (vals("D1", m, "oracle_2fold") - b)
        d1_open |= gin.mean() >= GATE_PP
        print(f"  {m:8s} base {100*b.mean():5.2f}%  in-sample {gin.mean():+5.2f} ({(gin>0).sum()}/9 up)  "
              f"2-fold {gcv.mean():+5.2f} ({(gcv>0).sum()}/9 up)")
    pb, po = vals("D1ctrl", "csp_lda", "planted_baseline"), vals("D1ctrl", "csp_lda", "planted_oracle")
    unpl = vals("D1", "csp_lda", "oracle_insample").loc[pb.index]
    base = vals("D1", "csp_lda", "baseline").loc[pb.index]
    valid = bool(((base - pb) >= 0.10).all() and (abs(po - unpl) <= 0.01).all())
    print(f"  planted control (s{list(pb.index)}): baseline drop {list((100*(base-pb)).round(1))} pp, "
          f"oracle vs unplanted oracle {list((100*(po-unpl)).round(2))} pp -> {'VALID' if valid else 'INVALID'}")
    verdicts = {"D1 rotation": "INVALID" if not valid else ("OPEN" if d1_open else "CLOSED")}

    print("\n=== D2 oracle class means ===")
    agree = vals("D2", "csp_lda", "own_lda_agreement_with_pipeline")
    print(f"  own-LDA agreement with pipeline: min {agree.min():.3f}")
    lda = {c: vals("D2", "csp_lda", c) for c in ("T_means", "pmean_shift", "oracle_insample", "oracle_2fold")}
    mdm = {c: vals("D2", "mdm", c) for c in ("T_means", "oracle_insample", "oracle_2fold")}
    g_lda = 100 * (lda["oracle_insample"] - lda["pmean_shift"]).mean()
    g_mdm = 100 * (mdm["oracle_insample"] - mdm["T_means"]).mean()
    print("  CSP+LDA  " + "  ".join(f"{c} {100*v.mean():.2f}" for c, v in lda.items()) + f"  | oracle-PMean {g_lda:+.2f}pp")
    print("  MDM      " + "  ".join(f"{c} {100*v.mean():.2f}" for c, v in mdm.items()) + f"  | oracle-T {g_mdm:+.2f}pp")
    pm = 100 * (lda["pmean_shift"] - lda["T_means"])
    print(f"  (label-free PMean vs T means, exploratory: {pm.mean():+.2f}pp, {(pm>0).sum()}/9 up)")
    verdicts["D2 class-conditional"] = "OPEN" if max(g_lda, g_mdm) >= GATE_PP else "CLOSED"

    print("\n=== D3 within-session drift ===")
    d3_open = False
    for arm in ("cnn_ra_adabn", "csp_ra"):
        d = 100 * vals("D3", arm, "late_minus_early")
        p = sign_flip_permutation_test(d.to_numpy(), "two-sided")
        d3_open |= abs(d.mean()) >= GATE_PP and p < 0.10
        print(f"  {arm:13s} late-early {d.mean():+.2f}pp ({(d>0).sum()}/9 later better), sign-flip p={p:.3f}")
    rw = 100 * (vals("D3", "csp_lda", "runwise_recentering") - vals("D3", "csp_lda", "session_recentering"))
    d3_open |= rw.mean() >= 1.0
    rho = vals("D3", "cov_drift", "spearman_run_vs_dist")
    changed = vals("D3", "csp_lda", "runwise_pred_changed_frac")
    print(f"  run-wise re-centering changes {100*changed.mean():.1f}% of predictions (no-op guard; must be > 0)")
    print(f"  run-wise vs session re-centering (CSP+LDA): {rw.mean():+.2f}pp ({(rw>0).sum()}/9 up)")
    print(f"  per-run covariance distance vs run index, Spearman rho: median {rho.median():+.2f}, "
          f"{(rho>0).sum()}/9 positive")
    verdicts["D3 online tracking"] = "OPEN" if d3_open else "CLOSED"

    print("\n=== D4 pseudo-label reliability (5-seed cnn_ra_adabn ensemble) ===")
    top = {q: vals("D4", "cnn_ra_adabn_ens", f"top{q}_acc") for q in (25, 50, 75, 100)}
    for q, v in top.items():
        print(f"  top {q:3d}% most confident: mean {100*v.mean():.1f}%  WEAK {100*v.loc[list(WEAK)].mean():.1f}%")
    verdicts["D4a self-training"] = ("OPEN" if top[50].mean() >= 0.80 and top[50].loc[list(WEAK)].mean() >= 0.70
                                     else "CLOSED")
    char = pd.read_csv(CHAR_CSV).set_index("subject")
    shift = (char["peak_freq_motor_eval"] - char["peak_freq_motor_calib"]).abs()
    print("  |mu-peak shift T->E| (Hz): " + ", ".join(f"s{s} {v:.2f}" for s, v in shift.items()))
    opened = [int(s) for s, v in shift.items() if v >= 0.5]
    verdicts["D4b frequency warping"] = f"OPEN for subjects {opened}" if opened else "CLOSED"
    print("  per-class accuracy (ensemble, mean over subjects): " + ", ".join(
        f"{c} {100*vals('D4', 'cnn_ra_adabn_ens', f'class_acc_{c}').mean():.1f}%" for c in IV2A_CLASSES))
    ras = df[df.model == "rasch"]
    print("  Rasch class easiness vs left_hand (log-odds): " + ", ".join(
        f"{r.condition.split('_vs_left_hand_')[1]} {r.value:+.2f}" for r in ras.itertuples()))

    print("\n=== GATE VERDICTS (thresholds fixed in the module docstring) ===")
    for k, v in verdicts.items():
        print(f"  {k:22s} {v}")
    return df


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct, labels = {}, {}
    for s in IV2A_SUBJECTS:
        print(f"=== subject {s} ===", flush=True)
        nar = load_iv2a_subject_window(s, FMIN, FMAX, TMIN, TMAX)
        yc, ye = encode_labels(nar.y_calib, IV2A_CLASSES), encode_labels(nar.y_eval, IV2A_CLASSES)
        Xc, Xe = align_per_session(nar.X_calib, nar.X_eval)
        d1_linear(s, Xc, yc, Xe, ye)
        d2(s, Xc, yc, Xe, ye)
        d3_subject(s, nar.X_calib, nar.X_eval, yc, ye, nar.runs_eval)
        wide = load_iv2a_subject_window(s, **WIDE)
        if not np.array_equal(encode_labels(wide.y_eval, IV2A_CLASSES), ye):
            raise AssertionError("narrow and wide loaders disagree on trial order")
        d1_cnn(s, *align_per_session(wide.X_calib, wide.X_eval), yc, ye, device)
        d3_runs_from_stored(s, ye, wide.runs_eval)
        d4_subject(s, ye)
        probs = np.mean([np.load(PRED_DIR / f"s{s}_cnn_ra_adabn_seed{k}.npy") for k in range(5)], axis=0)
        correct[s], labels[s] = (probs.argmax(1) == ye).astype(int), ye
    d4_rasch(correct, labels)
    summarise()
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
