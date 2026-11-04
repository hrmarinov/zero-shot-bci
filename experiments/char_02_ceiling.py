"""Characterization 02 - how much headroom is there above 66.9%, and is the
existing evaluation leak-free?

Motivation
----------
`docs/progress_and_direction.md` ("Established, trustworthy baselines") reports
the within-session repeated-CV number (65.1%) as an "upper bound-ish" for the
cross-session numbers it is supposed to bound - yet the cross-session number
*with* Riemannian alignment (66.9%) is higher than that bound, and the
cross-session number *without* adaptation (62.2%) is lower. Nobody knows how
much headroom is actually left above 66.9%.

This script computes, on one identical protocol (same loader, same
`RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)`
splitter wherever a CV is involved, same CSP+LDA pipeline configuration), the
conditions needed to answer that:

  within_calib_cv          repeated 5-fold x10 CV on the *calibration* session
                           (the number the docs call 65.1%)
  within_eval_cv           the same CV on the *evaluation* session - the true
                           same-session ceiling for the session being tested
  cross_none               train on all calibration, test on all evaluation,
                           no adaptation (the 62.2% baseline)
  cross_aligned            same, with RiemannianAlignment fit independently on
                           each session (the 66.9% headline)
  within_eval_cv_aligned   CV within the evaluation session with alignment fit
                           on the training folds only (leak-free) - shows
                           whether alignment inflates a within-session CV
  cross_aligned_nested     aligned cross-session pipeline with CSP
                           `n_components` chosen by an inner CV *inside the
                           calibration session only* (does the fixed
                           n_components=8 cap the "ceiling"?)

Two clearly-labelled supplementary diagnostics are also written, because they
are what actually disentangles "the calibration session is harder" from "the
evaluation session is easier":

  diag_reverse_cross_none      train on evaluation, test on calibration
  diag_wrongsession_alignment  the *bug* variant: one whitening fit on the
                               calibration session and applied to both sessions
                               (Phase 2 deliberately does not do this; this row
                               quantifies what that choice is worth)

Confidence intervals - what `ci_low`/`ci_high` mean (the `ci_kind` column says
which kind each row carries)
---------------------------------------------------------------------------
* Single-pass conditions (`cross_none`, `cross_aligned`,
  `cross_aligned_nested`, `diag_*`) test one fixed model on 288 independent
  trials, so a Wilson binomial interval on (n_correct, n_total) is valid:
  `ci_kind == "wilson_binomial_pooled_trials"`, and macro (subject-mean) and
  micro (pooled-trial) accuracy coincide exactly because every subject
  contributes the same 288 trials.
* CV conditions re-evaluate the SAME 288 trials 10 times (50 folds), so the
  folds are not independent samples and a binomial interval on the pooled
  2880 fold-counts is invalid - it comes out ~3x too narrow (the magnitude is
  printed by this script and quoted in docs/ceiling_analysis.md). Those rows
  therefore carry `ci_low`/`ci_high` = a 95% t-interval on the *mean of the 50
  per-fold accuracies* (which is exactly the statistic the 65.1% headline is),
  labelled `ci_kind == "t_interval_on_50_fold_means"`, with `fold_std` giving
  the per-fold spread. Treating that interval as a population CI would be an
  overclaim; it is a descriptive interval for a correlated-fold mean.

Every CV-based condition is computed with a hand-rolled fold loop that is a
faithful re-implementation of `sklearn.model_selection.cross_val_score`
(same splitter object, `clone` + `fit` per fold), so that both the per-fold
accuracy vector (for the std) and the pooled n_correct/n_total are available
from one pass. `within_calib_cv`, `cross_none` and `cross_aligned` are
asserted against the committed Phase 1 / Phase 2 CSVs to 1e-9 when all nine
subjects are run; a failure is printed as a headline finding, not silently
swallowed (and the process exits non-zero).

Reproduction:
    .venv\\Scripts\\python experiments\\char_02_ceiling.py
    .venv\\Scripts\\python experiments\\char_02_ceiling.py --subjects 1 --no-write

Deterministic: RANDOM_STATE everywhere, no torch, no GPU, one subject at a time
(docs/compute_environment.md rules 1, 2 and 5; setup_threads(4) because another
moderate job runs concurrently).
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import guard_memory, mem_report, setup_threads

# Rule 1 of docs/compute_environment.md: pin threads *before* importing numpy.
# 4 rather than 8 because a concurrent moderate job is running.
setup_threads(4)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402
from scipy.stats import ttest_rel, wilcoxon  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import mne  # noqa: E402

# MNE's CSP logs one INFO line per class per fold ("Estimating class=...
# covariance using LEDOIT_WOLF", mne/cov.py:1316); at 50 folds x 4 classes x
# several conditions that is thousands of lines of noise per subject.
mne.set_log_level("WARNING")

from src.adaptation.riemannian_align import RiemannianAlignment  # noqa: E402
from src.baselines import make_csp_lda  # noqa: E402
from src.config import IV2A_SUBJECTS, RANDOM_STATE, RESULTS_DIR  # noqa: E402
from src.datasets import load_iv2a_subject  # noqa: E402
from src.evaluate import wilson_interval  # noqa: E402

N_SPLITS = 5
N_REPEATS = 10
INNER_CV_REPEATS = 2
N_COMPONENT_GRID = (2, 4, 6, 8, 10, 12, 16, 22)

REQUIRED_CONDITIONS = (
    "within_calib_cv",
    "within_eval_cv",
    "cross_none",
    "cross_aligned",
    "within_eval_cv_aligned",
    "cross_aligned_nested",
)
SUPPLEMENTARY_CONDITIONS = (
    "diag_reverse_cross_none",
    "diag_wrongsession_alignment",
)
CONDITIONS = REQUIRED_CONDITIONS + SUPPLEMENTARY_CONDITIONS

#: Conditions that average per-fold accuracies from a repeated CV.
CV_CONDITIONS = frozenset({"within_calib_cv", "within_eval_cv", "within_eval_cv_aligned"})

CSV_COLUMNS = [
    "subject", "condition", "accuracy", "n_correct", "n_total",
    "ci_low", "ci_high", "ci_kind", "fold_std", "detail",
]


def aligned_pipeline(n_components: int = 8):
    """`make_csp_lda`, with RiemannianAlignment prepended as step 0.

    Reuses `make_csp_lda`'s exact CSP/LDA configuration (n_components,
    reg="ledoit_wolf", log=True, norm_trace=False) so that the aligned and
    unaligned conditions differ by the alignment step and nothing else.

    As a `Pipeline`, the whole chain - alignment included - is cloned and
    refit inside every CV fold, so the alignment in a CV is fit on that fold's
    training trials only (no training-fold-to-test-fold covariance leakage).
    """
    return Pipeline([("align", RiemannianAlignment())] + list(make_csp_lda(n_components).steps))


def repeated_cv(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = RANDOM_STATE,
):
    """Repeated stratified K-fold CV: per-fold accuracies + pooled counts.

    Faithful re-implementation of `cross_val_score(clone(pipeline), X, y, cv,
    scoring="accuracy")` - same splitter, and `clone` + `fit` per fold, which
    is exactly what `cross_val_score` does internally (sklearn 1.7.2
    model_selection/_validation.py:677 delegates to `cross_validate`, which
    clones per split at line 401 before `_fit_and_score` fits its clone at
    line 859). Doing the loop by hand is what makes the pooled
    n_correct/n_total available, which `cross_val_score` cannot give.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    fold_accs = []
    n_correct = 0
    n_total = 0
    for train_idx, test_idx in cv.split(X, y):
        model = clone(pipeline)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        correct = int(np.sum(pred == y[test_idx]))
        fold_accs.append(correct / len(test_idx))
        n_correct += correct
        n_total += len(test_idx)
    return np.asarray(fold_accs), n_correct, n_total


def select_n_components(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    grid=N_COMPONENT_GRID,
    n_splits: int = N_SPLITS,
    n_repeats: int = INNER_CV_REPEATS,
    seed: int = RANDOM_STATE,
):
    """Pick CSP n_components by an inner CV *inside the calibration session only*.

    For each candidate, the inner CV evaluates the full aligned pipeline
    (alignment refit on each inner training fold), so no trial is ever scored
    by a model that saw it. Only calibration-session trials and labels take
    part; the evaluation session is not touched. Ties resolve to the smallest
    candidate (strict `>` comparison).
    """
    best_n, best_acc = grid[0], -np.inf
    scores = {}
    for n in grid:
        pipe = aligned_pipeline(n)
        fold_accs, _, _ = repeated_cv(
            pipe, X_calib, y_calib, n_splits=n_splits, n_repeats=n_repeats, seed=seed
        )
        acc = float(fold_accs.mean())
        scores[n] = acc
        if acc > best_acc + 1e-12:
            best_n, best_acc = n, acc
    return int(best_n), scores


def wilson_row(subject, condition, n_correct, n_total, accuracy=None, detail=""):
    """Row for a single-pass condition: 288 independent trials, Wilson CI valid."""
    ci_low, ci_high = wilson_interval(int(n_correct), int(n_total))
    return {
        "subject": subject,
        "condition": condition,
        "accuracy": float(n_correct / n_total if accuracy is None else accuracy),
        "n_correct": int(n_correct),
        "n_total": int(n_total),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_kind": "wilson_binomial_pooled_trials",
        "fold_std": float("nan"),
        "detail": detail,
    }


def cv_row(subject, condition, fold_accs, n_correct, n_total, detail=""):
    """Row for a repeated-CV condition.

    `ci_low`/`ci_high` are a 95% t-interval on the mean of the per-fold
    accuracies - deliberately NOT a Wilson interval on (n_correct, n_total),
    which would treat 50 folds that re-evaluate the same 288 trials as 2880
    independent draws and come out ~3x too narrow.
    """
    fold_accs = np.asarray(fold_accs, dtype=float)
    n_folds = len(fold_accs)
    mean = float(fold_accs.mean())
    half = float(t_dist.ppf(0.975, n_folds - 1) * fold_accs.std(ddof=1) / np.sqrt(n_folds))
    return {
        "subject": subject,
        "condition": condition,
        "accuracy": mean,
        "n_correct": int(n_correct),
        "n_total": int(n_total),
        "ci_low": mean - half,
        "ci_high": mean + half,
        "ci_kind": f"t_interval_on_{n_folds}_fold_means",
        "fold_std": float(fold_accs.std(ddof=1)),
        "detail": detail
        or (
            f"repeated CV mean of {n_folds} folds; n_correct/n_total pool folds "
            f"that re-evaluate the same trial set {n_folds // N_SPLITS}x, so a "
            "binomial CI on them would be ~3x too narrow and is deliberately "
            "not used as ci_low/ci_high"
        ),
    }


def subject_mean_cv_row(condition, accs, n_correct, n_total, detail=""):
    """Mean row for a CV condition: t-interval across subject accuracies."""
    accs = np.asarray(accs, dtype=float)
    n = len(accs)
    mean = float(accs.mean())
    if n > 1:
        half = float(t_dist.ppf(0.975, n - 1) * accs.std(ddof=1) / np.sqrt(n))
    else:
        half = float("nan")
    return {
        "subject": "average",
        "condition": condition,
        "accuracy": mean,
        "n_correct": int(n_correct),
        "n_total": int(n_total),
        "ci_low": mean - half,
        "ci_high": mean + half,
        "ci_kind": f"t_interval_across_{n}_subjects",
        "fold_std": float("nan"),
        "detail": detail,
    }


_FIT_LOG: list = []


class _FitCounter(ClassifierMixin, BaseEstimator):
    """Counts `fit` calls; used by the leakage audit's refit demonstration.

    `_FIT_LOG` is module-level, so every *clone* made inside sklearn's CV also
    records its fit there, while `self.n_fit` only ever increments on whichever
    object actually gets fitted.
    """

    def __init__(self):
        self.n_fit = 0
        self.classes_ = None

    def fit(self, X, y):
        _FIT_LOG.append(1)
        self.n_fit += 1
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def audit_cross_val_score_refits(rng) -> dict:
    """Demonstrate that `cross_val_score` refits the pipeline inside every fold.

    Three claims checked on synthetic data (no EEG involved):
      1. the estimator object *handed to* `cross_val_score` is never fitted
         (`n_fit` stays 0) - so `within_session_accuracy`'s outer `clone` is
         redundant, and no pre-fitted state can ever be reused across folds;
      2. the number of actual `fit` calls is exactly n_splits * n_repeats;
      3. this script's hand-rolled `repeated_cv` reproduces `cross_val_score`'s
         per-fold mean to 1e-12 - i.e. the loop really is the same protocol.
    """
    X = rng.normal(size=(120, 5))
    y = np.tile(np.arange(4), 30)
    counter = _FitCounter()

    _FIT_LOG.clear()
    sklearn_scores = cross_val_score(
        counter,
        X,
        y,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE),
    )
    sklearn_fits = len(_FIT_LOG)

    _FIT_LOG.clear()
    my_fold_accs, _, _ = repeated_cv(_FitCounter(), X, y)
    own_loop_fits = len(_FIT_LOG)

    return {
        "caller_estimator_n_fit": int(counter.n_fit),
        "sklearn_fits_measured": sklearn_fits,
        "own_loop_fits_measured": own_loop_fits,
        "expected_fits": int(5 * 10),
        "sklearn_mean": float(sklearn_scores.mean()),
        "own_loop_mean": float(my_fold_accs.mean()),
        "own_loop_matches_sklearn": bool(
            abs(sklearn_scores.mean() - my_fold_accs.mean()) < 1e-12
        ),
    }


def _mean_cov_deviation(whitening: np.ndarray, X_raw: np.ndarray) -> float:
    """max|W @ mean_riemann(OAS cov(X_raw)) @ W - I|.

    ~1e-13 when `whitening` is the inverse square root of X_raw's own mean
    covariance (i.e. that session is whitened); large when a *different*
    session's map is applied. Note this deliberately uses the same OAS
    estimator `RiemannianAlignment.fit` uses on the RAW session: OAS shrinkage
    is not equivariant under a linear transform, so re-estimating OAS on the
    already-whitened data does not return the identity even when the whitening
    is exact - that is a property of the estimator, not of the code.
    """
    from pyriemann.estimation import Covariances
    from pyriemann.geometry.mean import mean_riemann

    covs = Covariances(estimator="oas").fit_transform(X_raw)
    mean_cov = mean_riemann(covs)
    return float(np.max(np.abs(whitening @ mean_cov @ whitening - np.eye(whitening.shape[0]))))


def _emp_cov_equivariance(whitening: np.ndarray, X_raw: np.ndarray) -> float:
    """max|cov_emp(W @ X_trial) - W cov_emp(X_trial) W| for the first trial.

    Confirms `RiemannianAlignment.transform` applies exactly the linear map it
    stored (`np.einsum("cd,ndt->nct", W, X)`) and nothing else.
    """
    empirical = np.cov(X_raw[0])
    return float(
        np.max(np.abs(np.cov(whitening @ X_raw[0]) - whitening @ empirical @ whitening))
    )


def _fit_predict(pipeline, X_train, y_train, X_test):
    """Local copy of src.evaluate.fit_predict (identical semantics)."""
    model = clone(pipeline)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_subject(subject: int):
    """All conditions for one subject. Returns (rows, audit_facts)."""
    guard_memory(1.0)
    data = load_iv2a_subject(subject)
    n_calib = int(data.X_calib.shape[0])
    n_eval = int(data.X_eval.shape[0])
    assert n_calib == n_eval, (
        f"subject {subject}: unequal session sizes {n_calib} vs {n_eval} - the "
        f"macro/micro mean distinction in the docs would need restating"
    )
    rows = []
    facts = {"subject": subject, "n_calib": n_calib, "n_eval": n_eval}

    # 1. within_calib_cv -- must reproduce the committed 65.1% mean.
    fold_accs, n_correct, n_total = repeated_cv(make_csp_lda(), data.X_calib, data.y_calib)
    rows.append(cv_row(subject, "within_calib_cv", fold_accs, n_correct, n_total))

    # 2. within_eval_cv -- the ceiling that should bound cross-session accuracy.
    fold_accs_e, n_correct_e, n_total_e = repeated_cv(make_csp_lda(), data.X_eval, data.y_eval)
    rows.append(cv_row(subject, "within_eval_cv", fold_accs_e, n_correct_e, n_total_e))
    # Evidence for the CI policy: a binomial interval on the pooled 2880 fold
    # counts vs one on the same accuracy from 288 independent trials.
    pooled_low, pooled_high = wilson_interval(n_correct_e, n_total_e)
    single_low, single_high = wilson_interval(round(fold_accs_e.mean() * n_eval), n_eval)
    facts["cv_pooled_wilson_width"] = float(pooled_high - pooled_low)
    facts["cv_single_session_wilson_width"] = float(single_high - single_low)

    # 3. cross_none -- Phase 1 baseline.
    y_pred_none = _fit_predict(make_csp_lda(), data.X_calib, data.y_calib, data.X_eval)
    n_correct_none = int(np.sum(y_pred_none == data.y_eval))
    rows.append(wilson_row(subject, "cross_none", n_correct_none, n_eval))

    # Leakage audit evidence: predictions cannot depend on evaluation labels.
    # Re-running must give identical predictions (fit_predict has no y_eval
    # parameter to receive them), while scoring against permuted labels must
    # collapse to chance - i.e. y_eval's only role is scoring.
    y_eval_perm = np.random.default_rng(0).permutation(data.y_eval)
    y_pred_none_again = _fit_predict(make_csp_lda(), data.X_calib, data.y_calib, data.X_eval)
    facts["predictions_identical_on_rerun"] = bool(
        np.array_equal(y_pred_none, y_pred_none_again)
    )
    facts["accuracy_under_permuted_eval_labels"] = float(np.mean(y_pred_none == y_eval_perm))

    # 4. cross_aligned -- Phase 2 headline. Each session whitened by its OWN mean.
    #    Two separate estimator instances, each fit on exactly the array it then
    #    transforms: `align_calib` only ever sees `data.X_calib`, `align_eval`
    #    only ever sees `data.X_eval` (labels are not even a parameter of
    #    RiemannianAlignment.fit - src/adaptation/riemannian_align.py:32).
    align_calib = RiemannianAlignment()
    align_eval = RiemannianAlignment()
    X_calib_aligned = align_calib.fit_transform(data.X_calib)
    X_eval_aligned = align_eval.fit_transform(data.X_eval)
    W_calib = align_calib.whitening_
    W_eval = align_eval.whitening_
    # Measured checks of the claims above (see the two helpers' docstrings):
    facts["whiten_self_calib_dev"] = _mean_cov_deviation(W_calib, data.X_calib)
    facts["whiten_self_eval_dev"] = _mean_cov_deviation(W_eval, data.X_eval)
    facts["whiten_cross_calib_map_on_eval_dev"] = _mean_cov_deviation(W_calib, data.X_eval)
    facts["transform_is_exactly_the_linear_map_dev"] = _emp_cov_equivariance(
        W_calib, data.X_calib
    )

    y_pred_aligned = _fit_predict(make_csp_lda(), X_calib_aligned, data.y_calib, X_eval_aligned)
    n_correct_aligned = int(np.sum(y_pred_aligned == data.y_eval))
    rows.append(wilson_row(subject, "cross_aligned", n_correct_aligned, n_eval))

    # 5. within_eval_cv_aligned -- alignment inside the CV, training folds only.
    fold_accs_ea, n_correct_ea, n_total_ea = repeated_cv(
        aligned_pipeline(), data.X_eval, data.y_eval
    )
    rows.append(cv_row(subject, "within_eval_cv_aligned", fold_accs_ea, n_correct_ea, n_total_ea))

    # 6. cross_aligned_nested -- n_components chosen inside the calibration
    #    session only, then one aligned cross-session run with that value.
    best_n, inner_scores = select_n_components(data.X_calib, data.y_calib)
    y_pred_nested = _fit_predict(
        make_csp_lda(best_n), X_calib_aligned, data.y_calib, X_eval_aligned
    )
    n_correct_nested = int(np.sum(y_pred_nested == data.y_eval))
    rows.append(
        wilson_row(
            subject,
            "cross_aligned_nested",
            n_correct_nested,
            n_eval,
            detail=f"n_components selected as {best_n} (inner CV acc "
            f"{inner_scores[best_n]:.3f}; all candidates "
            + ", ".join(f"n={n}:{a:.3f}" for n, a in inner_scores.items())
            + ")",
        )
    )
    facts["nested_best_n_components"] = best_n
    facts["nested_inner_scores"] = {int(k): float(v) for k, v in inner_scores.items()}

    # diag: reverse direction, no adaptation (tests session difficulty).
    y_pred_rev = _fit_predict(make_csp_lda(), data.X_eval, data.y_eval, data.X_calib)
    rows.append(
        wilson_row(
            subject, "diag_reverse_cross_none", int(np.sum(y_pred_rev == data.y_calib)), n_calib
        )
    )

    # diag: the bug variant - ONE whitening fit on calibration, applied to both.
    y_pred_wrong = _fit_predict(
        make_csp_lda(),
        align_calib.transform(data.X_calib),
        data.y_calib,
        align_calib.transform(data.X_eval),
    )
    n_correct_wrong = int(np.sum(y_pred_wrong == data.y_eval))
    rows.append(wilson_row(subject, "diag_wrongsession_alignment", n_correct_wrong, n_eval))
    facts["wrongsession_equals_cross_aligned"] = bool(
        np.array_equal(y_pred_wrong, y_pred_aligned)
    )

    del data, X_calib_aligned, X_eval_aligned
    gc.collect()
    return rows, facts


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired Wilcoxon + t-test across subjects for a vs b."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    out = {
        "mean_diff": float(diff.mean()),
        "n_subjects_a_wins": int(np.sum(diff > 0)),
        "n_subjects_b_wins": int(np.sum(diff < 0)),
    }
    if np.allclose(diff, 0.0):
        out.update({"wilcoxon_p": 1.0, "ttest_p": 1.0})
        return out
    out["wilcoxon_p"] = float(wilcoxon(a, b).pvalue)
    out["ttest_p"] = float(ttest_rel(a, b).pvalue)
    return out


def reproduction_report(df: pd.DataFrame, subjects) -> tuple:
    """Assert this run against the committed Phase 1 / Phase 2 numbers.

    Only meaningful for the full nine-subject run; a partial run returns
    `(checks, skipped=True)` rather than shape-mismatched NaNs. Every
    per-subject comparison array is asserted to be length 9 so a silent
    reindex/NaN mismatch can never pass as "close enough".
    """
    if sorted(subjects) != sorted(IV2A_SUBJECTS):
        return [], True

    p1 = pd.read_csv(RESULTS_DIR / "phase1_baseline_iv2a.csv")
    p2 = pd.read_csv(RESULTS_DIR / "phase2_riemannian_alignment_iv2a.csv")
    # Both committed CSVs carry a final subject == "average" row
    # (phase1_baseline.py:74, phase2_riemannian_alignment.py:69); strip it for
    # the per-subject checks and keep it only for the mean-level ones.
    p1_subj = p1[p1["subject"] != "average"].copy()
    p2_subj = p2[p2["subject"] != "average"].copy()
    for frame in (p1_subj, p2_subj):
        frame["subject"] = frame["subject"].astype(int)
    subj = df[df["subject"] != "average"].copy()
    subj["subject"] = subj["subject"].astype(int)
    piv = subj.pivot(index="subject", columns="condition", values="accuracy")
    checks = []

    def add(name, got, want, tol=1e-9):
        got = np.asarray(got, dtype=float)
        want = np.asarray(want, dtype=float)
        assert len(got) == len(want) == len(IV2A_SUBJECTS), (
            f"reproduction check '{name}': expected {len(IV2A_SUBJECTS)} paired "
            f"values, got {len(got)} vs {len(want)}"
        )
        assert not np.isnan(got).any(), f"reproduction check '{name}': NaN in measured values"
        checks.append(
            {
                "check": name,
                "max_abs_diff": float(np.max(np.abs(got - want))),
                "tol": tol,
                "pass": bool(np.all(np.abs(got - want) <= tol)),
            }
        )

    add(
        "within_calib_cv vs phase1 within_session_acc",
        piv["within_calib_cv"].reindex(p1_subj["subject"]).to_numpy(),
        p1_subj["within_session_acc"].to_numpy(),
    )
    add(
        "cross_none vs phase1 cross_session_acc",
        piv["cross_none"].reindex(p1_subj["subject"]).to_numpy(),
        p1_subj["cross_session_acc"].to_numpy(),
    )
    add(
        "cross_none vs phase2 cross_session_baseline_acc",
        piv["cross_none"].reindex(p2_subj["subject"]).to_numpy(),
        p2_subj["cross_session_baseline_acc"].to_numpy(),
    )
    add(
        "cross_aligned vs phase2 cross_session_aligned_acc",
        piv["cross_aligned"].reindex(p2_subj["subject"]).to_numpy(),
        p2_subj["cross_session_aligned_acc"].to_numpy(),
    )
    means = df[df["subject"] == "average"].set_index("condition")["accuracy"]
    add(
        "MEAN within_calib_cv vs phase1 average (9 subjects)",
        [means["within_calib_cv"]] * len(IV2A_SUBJECTS),
        [p1["within_session_acc"].iloc[-1]] * len(IV2A_SUBJECTS),
    )
    add(
        "MEAN cross_none vs phase1 average (9 subjects)",
        [means["cross_none"]] * len(IV2A_SUBJECTS),
        [p1["cross_session_acc"].iloc[-1]] * len(IV2A_SUBJECTS),
    )
    add(
        "MEAN cross_aligned vs phase2 average (9 subjects)",
        [means["cross_aligned"]] * len(IV2A_SUBJECTS),
        [p2["cross_session_aligned_acc"].iloc[-1]] * len(IV2A_SUBJECTS),
    )
    return checks, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument(
        "--no-write", action="store_true", help="do not write results/char_ceiling_iv2a.csv"
    )
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else list(IV2A_SUBJECTS)
    t_start = time.time()
    mem_report("start")

    rng = np.random.default_rng(RANDOM_STATE)
    audit = audit_cross_val_score_refits(rng)
    print("\n=== LEAKAGE-AUDIT DEMONSTRATION (synthetic, no EEG) ===")
    print(
        "cross_val_score with RepeatedStratifiedKFold(5, 10): fits on the "
        f"*caller's* estimator object = {audit['caller_estimator_n_fit']} "
        "(0 => it is cloned and refit per fold, never reused)"
    )
    print(
        f"fits actually performed: cross_val_score = {audit['sklearn_fits_measured']}, "
        f"own loop = {audit['own_loop_fits_measured']} "
        f"(expected {audit['expected_fits']} = 5 folds x 10 repeats)"
    )
    print(
        f"hand-rolled loop mean = {audit['own_loop_mean']:.12f} vs "
        f"cross_val_score mean = {audit['sklearn_mean']:.12f}; "
        f"identical = {audit['own_loop_matches_sklearn']}"
    )
    if (
        audit["caller_estimator_n_fit"] != 0
        or audit["sklearn_fits_measured"] != audit["expected_fits"]
        or audit["own_loop_fits_measured"] != audit["expected_fits"]
        or not audit["own_loop_matches_sklearn"]
    ):
        print("!! AUDIT DEMONSTRATION FAILED - do not trust the audit verdicts !!")

    all_rows = []
    facts = []
    for subject in subjects:
        t0 = time.time()
        print(f"\n[subject {subject}] loading...", flush=True)
        rows, f = run_subject(subject)
        all_rows.extend(rows)
        facts.append(f)
        accs = pd.DataFrame(rows).set_index("condition")["accuracy"]
        print(
            f"[subject {subject}] "
            + " ".join(f"{c}={accs[c]:.4f}" for c in CONDITIONS)
            + f"  ({time.time() - t0:.0f}s)"
        )
        mem_report(f"subject {subject} done")

    df = pd.DataFrame(all_rows)

    # ---- per-condition mean rows -------------------------------------------
    mean_rows = []
    for condition in CONDITIONS:
        block = df[df["condition"] == condition]
        n_correct = int(block["n_correct"].sum())
        n_total = int(block["n_total"].sum())
        micro = n_correct / n_total
        between_sd = float(block["accuracy"].std(ddof=1)) if len(block) > 1 else float("nan")
        shared = (
            f"macro mean of {len(block)} subject accuracies; micro (pooled trials) "
            f"{micro:.4f}; between-subject SD {between_sd:.4f}"
        )
        if condition in CV_CONDITIONS:
            mean_rows.append(
                subject_mean_cv_row(
                    condition,
                    block["accuracy"].to_numpy(),
                    n_correct,
                    n_total,
                    detail=shared
                    + f"; each subject's n_total={block['n_total'].iloc[0]} pools "
                    f"{N_SPLITS * N_REPEATS} folds that re-evaluate the same 288 trials",
                )
            )
        else:
            mean_rows.append(
                wilson_row(
                    "average",
                    condition,
                    n_correct,
                    n_total,
                    accuracy=float(block["accuracy"].mean()),
                    detail=shared
                    + (f"; n_components picks {[f['nested_best_n_components'] for f in facts]}"
                       if condition == "cross_aligned_nested" else ""),
                )
            )
    df = pd.concat([df, pd.DataFrame(mean_rows)], ignore_index=True)[CSV_COLUMNS]

    # ---- printed summary tables -------------------------------------------
    subj_df = df[df["subject"] != "average"].copy()
    subj_df["subject"] = subj_df["subject"].astype(int)
    wide = subj_df.pivot(index="subject", columns="condition", values="accuracy")[list(CONDITIONS)]
    print("\n=== ACCURACY BY SUBJECT x CONDITION ===")
    print(wide.to_string(float_format=lambda v: f"{v:.4f}"))
    print(wide.mean().to_frame("MEAN").T.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== CONDITION SUMMARY (macro mean; ci_kind says which interval) ===")
    summary = df[df["subject"] == "average"][
        ["condition", "accuracy", "n_correct", "n_total", "ci_low", "ci_high", "ci_kind"]
    ]
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== PER-SUBJECT DETAIL (fold_std / pooled counts / ci_kind) ===")
    print(
        subj_df[["subject", "condition", "accuracy", "n_correct", "n_total",
                 "ci_low", "ci_high", "ci_kind", "fold_std"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )

    print("\n=== WHY THE CV ROWS DO NOT CARRY A WILSON CI (per subject) ===")
    for f in facts:
        print(
            f"  subject {f['subject']}: pooled-fold Wilson width "
            f"{f['cv_pooled_wilson_width']:.4f} vs single-session Wilson width "
            f"{f['cv_single_session_wilson_width']:.4f} "
            f"(ratio {f['cv_pooled_wilson_width'] / f['cv_single_session_wilson_width']:.2f}x)"
        )

    print("\n=== HEADROOM vs the eval-session within-session CV ceiling ===")
    head = pd.DataFrame(
        {
            "within_eval_cv": wide["within_eval_cv"],
            "cross_none": wide["cross_none"],
            "cross_aligned": wide["cross_aligned"],
            "headroom_above_cross_aligned": wide["within_eval_cv"] - wide["cross_aligned"],
            "headroom_above_cross_none": wide["within_eval_cv"] - wide["cross_none"],
            "reported_bound_gap": wide["within_calib_cv"] - wide["cross_aligned"],
        }
    )
    print(head.to_string(float_format=lambda v: f"{v:.4f}"))
    print("means:")
    print(head.mean().to_frame("MEAN").T.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== SESSION-DIFFICULTY DECOMPOSITION (paired across subjects) ===")
    stats = {
        "within_eval_cv - within_calib_cv (eval session easier under CV?)": paired_stats(
            wide["within_eval_cv"].to_numpy(), wide["within_calib_cv"].to_numpy()
        ),
        "cross_none - diag_reverse_cross_none (eval easier as a test session?)": paired_stats(
            wide["cross_none"].to_numpy(), wide["diag_reverse_cross_none"].to_numpy()
        ),
        "cross_aligned - cross_none (Phase 2 gain)": paired_stats(
            wide["cross_aligned"].to_numpy(), wide["cross_none"].to_numpy()
        ),
        "within_eval_cv_aligned - within_eval_cv (alignment inflates CV?)": paired_stats(
            wide["within_eval_cv_aligned"].to_numpy(), wide["within_eval_cv"].to_numpy()
        ),
        "cross_aligned_nested - cross_aligned (nested CSP helps?)": paired_stats(
            wide["cross_aligned_nested"].to_numpy(), wide["cross_aligned"].to_numpy()
        ),
    }
    for name, s in stats.items():
        print(
            f"  {name}: mean diff {s['mean_diff']:+.4f}, wins "
            f"{s['n_subjects_a_wins']}/{len(wide)} vs {s['n_subjects_b_wins']}, "
            f"wilcoxon p={s['wilcoxon_p']:.4f}, ttest p={s['ttest_p']:.4f}"
        )

    print("\n=== REPRODUCTION CHECKS vs committed CSVs (tol 1e-9) ===")
    checks, skipped = reproduction_report(df, subjects)
    if skipped:
        print("  SKIPPED: partial run (--subjects), reproduction needs all 9 subjects")
        repro_ok = True
    else:
        for c in checks:
            print(
                f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: "
                f"max|diff|={c['max_abs_diff']:.3e} (tol {c['tol']:.0e})"
            )
        repro_ok = all(c["pass"] for c in checks)

    print("\n=== ALIGNMENT / LABEL-INDEPENDENCE FACTS (per subject) ===")
    for f in facts:
        print(
            f"  subject {f['subject']}: whitening self-check dev calib="
            f"{f['whiten_self_calib_dev']:.1e} eval={f['whiten_self_eval_dev']:.1e}; "
            f"calib's map applied to eval session dev="
            f"{f['whiten_cross_calib_map_on_eval_dev']:.1e}; "
            f"transform-is-exactly-W dev={f['transform_is_exactly_the_linear_map_dev']:.1e}; "
            f"preds identical on rerun={f['predictions_identical_on_rerun']}; "
            f"acc under permuted eval labels={f['accuracy_under_permuted_eval_labels']:.3f}; "
            f"nested pick n_components={f['nested_best_n_components']}"
        )

    if not args.no_write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "char_ceiling_iv2a.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} rows to {out_path}")
    else:
        print("\n--no-write: results CSV not written")

    print(f"\nreproduction_ok={repro_ok}")
    print(f"total runtime {time.time() - t_start:.0f}s")
    mem_report("end", collect=True)
    return 0 if repro_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
