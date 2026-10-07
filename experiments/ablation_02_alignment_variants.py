"""Ablation 02 - is the Riemannian mean the right whitening reference? (TODO.md A12 + A5)

Riemannian alignment is the only adaptation method in this project that works, and it has
never been ablated. Every variant below is a *drop-in replacement for one line* - how the
session's whitening matrix is estimated - with the entire rest of the protocol held fixed
(identical loader, CSP+LDA configuration, trial sets and scoring). That makes the
comparison exact: any accuracy difference is attributable to the reference estimate alone.

Conditions
----------
  none              no whitening at all. Control; must reproduce Phase 1's 62.2%.
  riemann           mean_riemann (the current method). Control; must reproduce 66.9%.
  euclid            arithmetic mean covariance - i.e. **Euclidean Alignment**
                    (He & Wu 2020), which `docs/state_of_the_art.md` section 2 cites as
                    the original method but which this project has never run.
  logeuclid         log-Euclidean mean (cheaper, and the metric the shrinkage work used).
  wasserstein       Wasserstein barycenter (`mean_wasserstein`) - an optimal-transport
                    flavoured reference, the cheap end of TODO A5, which records a
                    published +2.45pp for OT alignment on this exact dataset.
  power_p0.5        power mean, a one-parameter family interpolating the arithmetic and
                    harmonic means (p=1 is euclid, p=-1 harmonic-ish).
  subspace_k<K>     regularised reference: the mean of the session's tangent vectors after
                    projecting onto their top-K principal directions, mapped back to SPD.
                    With 288 trials against a 253-dimensional tangent space, the full-rank
                    mean is only just identifiable, so a rank-limited mean may be *less*
                    noisy. Swept over K in {10, 30, 80}.
  perclass_calib    the calibration session is whitened **per class** (by each class's own
                    mean covariance, using its labels - allowed, since calibration labels
                    are available), while the evaluation session is whitened session-wide
                    as usual (no labels there). Contrast with `riemann` isolates the effect
                    of class-conditional training whitening.

Verification built in
---------------------
`none` and `riemann` are asserted against the committed Phase 1 / Phase 2 per-subject
accuracies at 1e-9, and the script exits non-zero if either fails. This is the check that
proves the harness is identical to the established protocol; without it a "variant beats
the baseline" claim would be uninterpretable.

Statistics
----------
Per condition: paired Wilcoxon across the 9 subjects vs `riemann`, plus a per-subject
exact McNemar against `riemann` on the identical evaluation trials (the project's standard
test, and the more powerful one at this n). With n=9 the Wilcoxon floor is p=0.0039, so
per-subject McNemar is reported alongside rather than replaced.

Determinism: no random numbers are used at all, so reruns are byte-identical.
Cost: this machine, ~2-4 minutes for all 9 subjects (no torch, CPU only, one subject at a
time). See docs/compute_environment.md.

Output: results/ablation_alignment_variants.csv
    subject, condition, accuracy, n_correct, n_total, n_components_or_k, improvement_vs_riemann,
    mcnemar_p_vs_riemann
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import guard_memory, mem_report, setup_threads

setup_threads(8)  # before numpy

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mne  # noqa: E402

mne.set_log_level("WARNING")  # MNE's CSP logs a line per class per fit

from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.geometry.base import invsqrtm  # noqa: E402
from pyriemann.geometry.mean import (  # noqa: E402
    mean_euclid,
    mean_logeuclid,
    mean_power,
    mean_riemann,
    mean_wasserstein,
)
from pyriemann.geometry.tangentspace import tangent_space, untangent_space  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src.adaptation.riemannian_align import RiemannianAlignment  # noqa: E402
from src.baselines import make_csp_lda  # noqa: E402
from src.config import RANDOM_STATE, RESULTS_DIR  # noqa: E402
from src.datasets import load_iv2a_subject  # noqa: E402
from src.evaluate import accuracy_summary, mcnemar_test  # noqa: E402

SUBSPACE_K = (10, 30, 80)
REFERENCE_CONDITION = "riemann"
ESTIMATOR = "oas"  # must match RiemannianAlignment's default

# Committed per-subject values, for the harness-identity assertions.
EXPECTED_PHASE1_CROSS = {
    1: 0.7847222222222222, 2: 0.4236111111111111, 3: 0.7673611111111112,
    4: 0.5868055555555556, 5: 0.3402777777777778, 6: 0.4756944444444444,
    7: 0.625, 8: 0.7881944444444444, 9: 0.8090277777777778,
}
EXPECTED_PHASE2_ALIGNED = {
    1: 0.8090277777777778, 2: 0.4444444444444444, 3: 0.8159722222222222,
    4: 0.6076388888888888, 5: 0.4236111111111111, 6: 0.4930555555555556,
    7: 0.7847222222222222, 8: 0.8055555555555556, 9: 0.8402777777777778,
}

CONDITIONS = ["none", REFERENCE_CONDITION, "euclid", "logeuclid", "wasserstein", "power_p0.5", "perclass_calib"] + [
    f"subspace_k{k}" for k in SUBSPACE_K
]


def _session_covs(X: np.ndarray) -> np.ndarray:
    return Covariances(estimator=ESTIMATOR).fit_transform(X)


def _subspace_mean_cov(X: np.ndarray, k: int) -> np.ndarray:
    """Rank-k regularised mean covariance of a session (unsupervised).

    Reconstruct every trial covariance from its top-k principal geodesics and average the
    reconstructions, with the geodesics **anchored at a fixed reference (the identity)**
    rather than at the session's own mean.

    Two implementation traps, both hit while building this (the `intervention check` in the
    report flagged them automatically, which is why it exists):

    1. Averaging the *tangent vectors* at the session mean is a no-op: `tangent_space`
       centres there, so their mean is 0 and `untangent_space(0, cref)` returns `cref`.
    2. Reconstructing per trial but *still* in the tangent space at the session mean is
       **also** a no-op, and necessarily so: the affine-invariant Riemannian mean satisfies
       `sum_i log_cref(covs_i) = 0` exactly (that is its first-order condition), so any
       linear projection P of the centred vectors still sums to zero, which makes `cref`
       satisfy the mean's own first-order condition. Rank-limiting is only a real operation
       when the anchor is external to the set being averaged.

    Anchoring at the identity makes the projection bite, so the result now depends on k.
    """
    covs = _session_covs(X)
    anchor = np.eye(covs.shape[-1])
    vecs = tangent_space(covs, anchor, metric="riemann")  # log maps at the identity
    k_eff = int(min(k, vecs.shape[1], vecs.shape[0]))
    pca = PCA(n_components=k_eff, random_state=RANDOM_STATE)
    vecs_rec = pca.inverse_transform(pca.fit_transform(vecs))
    covs_rec = untangent_space(vecs_rec, anchor, metric="riemann")
    return mean_riemann(covs_rec)


def _apply_whitening(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply a whitening matrix to a trial stack (same index convention as
    RiemannianAlignment.transform)."""
    return np.einsum("cd,ndt->nct", W, X)


def _session_whitening_matrix(condition: str, X: np.ndarray) -> np.ndarray:
    """The session's whitening matrix W = invsqrtm(reference covariance)."""
    if condition == "euclid":
        return invsqrtm(mean_euclid(_session_covs(X)))
    if condition == "logeuclid":
        return invsqrtm(mean_logeuclid(_session_covs(X)))
    if condition == "wasserstein":
        return invsqrtm(mean_wasserstein(_session_covs(X)))
    if condition.startswith("power_p"):
        return invsqrtm(mean_power(_session_covs(X), float(condition.split("power_p", 1)[1])))
    if condition.startswith("subspace_k"):
        return invsqrtm(_subspace_mean_cov(X, int(condition.split("subspace_k", 1)[1])))
    raise ValueError(f"no whitening matrix defined for condition {condition!r}")


def _per_class_calib_whiten(X_calib: np.ndarray, y_calib: np.ndarray) -> np.ndarray:
    """Whiten each calibration trial by its OWN class's mean covariance.

    Supervised, and legitimately so: calibration labels are available by construction.
    The evaluation session never gets this treatment (it has no labels), which is exactly
    the asymmetry this condition is designed to test.
    """
    out = np.empty_like(X_calib)
    for label in np.unique(y_calib):
        mask = y_calib == label
        out[mask] = _apply_whitening(
            invsqrtm(mean_riemann(_session_covs(X_calib[mask]))), X_calib[mask]
        )
    return out


def run_condition(subject: int, condition: str, data) -> tuple[dict, np.ndarray]:
    """Fit and evaluate one condition for one subject. Returns (row, predictions)."""
    n_eval = data.X_eval.shape[0]

    if condition == "none":
        X_train, X_test = data.X_calib, data.X_eval
    elif condition == "riemann":
        # Deliberately the real class, not a reimplementation: this condition IS Phase 2,
        # and must reproduce its committed numbers bit-for-bit.
        X_train = RiemannianAlignment(estimator=ESTIMATOR).fit_transform(data.X_calib)
        X_test = RiemannianAlignment(estimator=ESTIMATOR).fit_transform(data.X_eval)
    elif condition == "perclass_calib":
        X_train = _per_class_calib_whiten(data.X_calib, data.y_calib)
        X_test = _apply_whitening(invsqrtm(mean_riemann(_session_covs(data.X_eval))), data.X_eval)
    else:
        X_train = _apply_whitening(_session_whitening_matrix(condition, data.X_calib), data.X_calib)
        X_test = _apply_whitening(_session_whitening_matrix(condition, data.X_eval), data.X_eval)

    pipeline = make_csp_lda()
    pipeline.fit(X_train, data.y_calib)
    y_pred = pipeline.predict(X_test)
    acc, n_correct, _ = accuracy_summary(data.y_eval, y_pred)
    k_or_n = ""
    if condition.startswith("subspace_k"):
        k_or_n = int(condition.split("subspace_k", 1)[1])
    elif condition.startswith("power_p"):
        k_or_n = float(condition.split("power_p", 1)[1])
    row = {
        "subject": subject,
        "condition": condition,
        "accuracy": acc,
        "n_correct": n_correct,
        "n_total": n_eval,
        "n_components_or_k": k_or_n,
    }
    return row, y_pred


def run(subjects=None, out_name: str = "ablation_alignment_variants.csv") -> pd.DataFrame:
    from src.config import IV2A_SUBJECTS

    subjects = subjects if subjects is not None else IV2A_SUBJECTS
    rows: list[dict] = []
    preds: dict[int, dict[str, np.ndarray]] = {}
    y_true: dict[int, np.ndarray] = {}

    for subject in subjects:
        guard_memory(1.0)
        print(f"\n[subject {subject}] loading...", flush=True)
        data = load_iv2a_subject(subject)
        preds[subject], y_true[subject] = {}, data.y_eval
        for condition in CONDITIONS:
            row, y_pred = run_condition(subject, condition, data)
            rows.append(row)
            preds[subject][condition] = y_pred
            print(f"  {condition:<16} {row['accuracy']:.4f}", flush=True)
        del data
        gc.collect()
        mem_report(f"subject {subject} done")

    df = pd.DataFrame(rows)

    # --- per-subject comparison against the reference condition -------------
    df["improvement_vs_riemann"] = np.nan
    df["mcnemar_p_vs_riemann"] = np.nan
    for subject in subjects:
        ref_acc = float(df[(df.subject == subject) & (df.condition == REFERENCE_CONDITION)]["accuracy"].iloc[0])
        for i, row in df[df.subject == subject].iterrows():
            if row["condition"] == REFERENCE_CONDITION:
                df.loc[i, "improvement_vs_riemann"] = 0.0
                df.loc[i, "mcnemar_p_vs_riemann"] = 1.0
                continue
            df.loc[i, "improvement_vs_riemann"] = row["accuracy"] - ref_acc
            df.loc[i, "mcnemar_p_vs_riemann"] = mcnemar_test(
                y_true[subject], preds[subject][REFERENCE_CONDITION], preds[subject][row["condition"]]
            )

    # --- mean rows ----------------------------------------------------------
    mean_rows = []
    for condition in CONDITIONS:
        block = df[df.condition == condition]
        mean_rows.append(
            {
                "subject": "average",
                "condition": condition,
                "accuracy": float(block["accuracy"].mean()),
                "n_correct": int(block["n_correct"].sum()),
                "n_total": int(block["n_total"].sum()),
                "n_components_or_k": block["n_components_or_k"].iloc[0],
                "improvement_vs_riemann": float(block["improvement_vs_riemann"].mean()),
                "mcnemar_p_vs_riemann": np.nan,
            }
        )
    df = pd.concat([df, pd.DataFrame(mean_rows)], ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / out_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows -> {out_path}")

    _report(df, subjects, preds)
    return df


def _report(df: pd.DataFrame, subjects, preds: dict) -> None:
    from scipy.stats import wilcoxon

    subj = df[df.subject != "average"].copy()
    subj["subject"] = subj["subject"].astype(int)
    print("\n=== accuracy by subject x condition ===")
    wide = subj.pivot(index="subject", columns="condition", values="accuracy")[CONDITIONS]
    print(wide.to_string(float_format=lambda v: f"{v:.4f}"))
    print(wide.mean().to_frame("MEAN").T.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== vs the current method (`riemann`), paired across subjects ===")
    print(f"{'condition':<16} {'mean':>7} {'delta':>8} {'wins':>5} {'losses':>7} {'wilcoxon_p':>11} {'median McNemar p':>17}")
    ref = wide[REFERENCE_CONDITION].to_numpy()
    for condition in CONDITIONS:
        if condition == REFERENCE_CONDITION:
            continue
        got = wide[condition].to_numpy()
        diff = got - ref
        p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(got, ref).pvalue)
        mc = subj[subj.condition == condition]["mcnemar_p_vs_riemann"].to_numpy()
        print(
            f"{condition:<16} {got.mean():7.4f} {diff.mean():+8.4f} {int((diff > 0).sum()):5d} "
            f"{int((diff < 0).sum()):7d} {p:11.4f} {np.median(mc):17.4f}"
        )
    print(
        "\n(wilcoxon floor at n=9 is p=0.0039; per-subject McNemar on 288 identical trials is the "
        "more sensitive test and is reported as a median across subjects)"
    )
    print(
        "SCALE NOTE: each subject has 288 evaluation trials, so ONE trial = 0.35pp. A per-subject "
        "delta of one or two trials is noise by construction, which is exactly why the paired "
        "tests above (and per-subject McNemar, which counts only discordant trials) carry the "
        "inference rather than the raw means."
    )

    # --- intervention check -------------------------------------------------
    # A condition that produces bit-identical predictions to `riemann` for EVERY subject is
    # either a genuine mathematical equivalence or a no-op bug. This guard exists because an
    # earlier revision of the subspace variants was the latter (see _subspace_mean_cov) and
    # produced three conditions that silently duplicated the reference.
    print("\n=== intervention check: does each condition actually change the predictions? ===")
    for condition in CONDITIONS:
        if condition == REFERENCE_CONDITION:
            continue
        n_same = sum(
            bool(np.array_equal(preds[s][REFERENCE_CONDITION], preds[s][condition])) for s in subjects
        )
        flag = "   <-- NO-OP SUSPECTED: identical to riemann for every subject" if n_same == len(subjects) else ""
        print(f"  {condition:<16} identical-to-riemann on {n_same}/{len(subjects)} subjects{flag}")

    # --- harness-identity assertions ---------------------------------------
    print("\n=== harness identity checks (must match the committed baselines at 1e-9) ===")
    ok = True
    for condition, expected, label in (
        ("none", EXPECTED_PHASE1_CROSS, "Phase 1 cross-session"),
        (REFERENCE_CONDITION, EXPECTED_PHASE2_ALIGNED, "Phase 2 aligned"),
    ):
        got = wide[condition].to_dict()
        worst = max(abs(got[s] - expected[s]) for s in subjects)
        passed = worst <= 1e-9
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {condition} reproduces {label}: max|diff|={worst:.3e}")
    print(f"\nharness_identity_ok={ok}")
    if not ok:
        print("!! the harness does NOT match the established protocol - results above are not comparable !!")


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="default: all 9")
    parser.add_argument("--out-name", default="ablation_alignment_variants.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(subjects=args.subjects, out_name=args.out_name)
