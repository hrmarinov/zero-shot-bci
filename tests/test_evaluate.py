"""Tests for the shared evaluation protocol (`src/evaluate.py`).

These guard the statistics every reported number in `results/` depends on:
accuracy on a held-out session, the confidence interval attached to it, and the
paired significance test used to decide whether one method beat another. A bug
in any of these silently invalidates the project's conclusions, so they are
checked against hand-computable reference values rather than merely asserting
that the functions run.
"""

import numpy as np
import pytest

from src.baselines import make_csp_lda
from src.evaluate import (
    accuracy_summary,
    cross_session_accuracy,
    fit_predict,
    mcnemar_test,
    wilson_interval,
    within_session_accuracy,
)


def test_wilson_interval_matches_closed_form_for_50_of_100() -> None:
    # Reference values computed by hand from the Wilson score formula
    # (z = 1.959964, phat = 0.5, n = 100).
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)


def test_wilson_interval_stays_in_bounds_at_the_extremes() -> None:
    # The whole reason this interval is used instead of the Wald approximation:
    # it must remain a valid probability interval when accuracy is at 0% or 100%.
    for n_correct, n_total in [(0, 20), (20, 20), (0, 288), (288, 288)]:
        low, high = wilson_interval(n_correct, n_total)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_narrows_with_more_trials() -> None:
    width_small = np.subtract(*reversed(wilson_interval(50, 100)))
    width_large = np.subtract(*reversed(wilson_interval(500, 1000)))
    assert width_large < width_small


def test_accuracy_summary_counts_matches() -> None:
    accuracy, n_correct, n_total = accuracy_summary(
        np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])
    )
    assert (n_correct, n_total) == (3, 4)
    assert accuracy == pytest.approx(0.75)


def test_mcnemar_is_uninformative_when_predictions_agree() -> None:
    y_true = np.array([0, 1, 0, 1])
    assert mcnemar_test(y_true, y_true.copy(), y_true.copy()) == 1.0


def test_mcnemar_detects_a_wholly_one_sided_difference() -> None:
    # Ten trials: model A right on all, model B wrong on all. The exact
    # two-sided binomial p-value for 0/10 successes at p=0.5 is 2 * 0.5**10.
    y_true = np.zeros(10, dtype=int)
    pred_a = np.zeros(10, dtype=int)
    pred_b = np.ones(10, dtype=int)
    assert mcnemar_test(y_true, pred_a, pred_b) == pytest.approx(2 * 0.5**10)


def _separable_two_class_problem(n_trials: int = 60, n_channels: int = 8, n_times: int = 64, seed: int = 0):
    """Random noise plus a class-specific oscillatory burst on disjoint channels.

    Deliberately easy to separate, so that a low accuracy here means the
    protocol itself is broken rather than the data being hard.
    """
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n_trials // 2))
    t = np.arange(n_times) / 64.0
    carrier = np.sin(2 * np.pi * 10.0 * t)
    X = rng.normal(0.0, 1.0, size=(n_trials, n_channels, n_times))
    for i, label in enumerate(y):
        channels = [0, 1] if label == 0 else [6, 7]
        X[i, channels, :] += 3.0 * carrier
    return X, y


def test_csp_lda_separates_a_separable_problem_within_session() -> None:
    X, y = _separable_two_class_problem()
    pipeline = make_csp_lda(n_components=4)
    accuracy, _ = within_session_accuracy(pipeline, X, y, n_splits=5, n_repeats=2)
    assert accuracy > 0.85


def test_cross_session_accuracy_transfers_between_independent_draws() -> None:
    X_train, y_train = _separable_two_class_problem(seed=1)
    X_test, y_test = _separable_two_class_problem(seed=2)
    accuracy, n_correct, n_total = cross_session_accuracy(
        make_csp_lda(n_components=4), X_train, y_train, X_test, y_test
    )
    assert (n_correct, n_total) == (int(round(accuracy * n_total)), n_total)
    assert accuracy > 0.85


def test_fit_predict_does_not_mutate_the_supplied_pipeline() -> None:
    # `fit_predict` clones internally; experiments reuse a single pipeline
    # object across subjects, so a leak here would silently cross-contaminate.
    from sklearn.linear_model import LogisticRegression

    pipeline = LogisticRegression()
    X_train, y_train = _separable_two_class_problem(seed=3)
    X_flat = X_train.reshape(len(y_train), -1)
    fit_predict(pipeline, X_flat, y_train, X_flat)
    assert not hasattr(pipeline, "coef_")


def test_sign_flip_permutation_test_known_values() -> None:
    from src.evaluate import sign_flip_permutation_test

    # all 9 positive: only the observed assignment is >= observed -> 1/512
    assert sign_flip_permutation_test(np.ones(9)) == pytest.approx(1 / 512)
    # two-sided doubles it (the all-negative assignment is as extreme)
    assert sign_flip_permutation_test(np.ones(9), "two-sided") == pytest.approx(2 / 512)
    # symmetric data -> mean 0 -> every assignment as extreme in the two-sided sense
    assert sign_flip_permutation_test(np.array([1.0, -1.0]), "two-sided") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        sign_flip_permutation_test(np.ones(21))


def test_holm_adjust_matches_hand_computation() -> None:
    from src.evaluate import holm_adjust

    # sorted p: 0.01*3=0.03, 0.02*2=0.04, 0.04*1=0.04 (monotone), mapped back to input order
    assert holm_adjust([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_accuracy_summary_rejects_text_vs_integer_labels() -> None:
    from src.evaluate import accuracy_summary

    with pytest.raises(TypeError):
        accuracy_summary(np.array(["left_hand", "feet"]), np.array([0, 2]))
    assert accuracy_summary(np.array(["a", "b"]), np.array(["a", "a"]))[0] == 0.5
    assert accuracy_summary(np.array([1, 2]), np.array([1, 2]))[0] == 1.0


def test_riemannian_alignment_whitens_its_own_session_only() -> None:
    """Core of both headline results: W · mean_riemann(covs) · W = I, transform = W @ X."""
    from pyriemann.estimation import Covariances
    from pyriemann.geometry.mean import mean_riemann

    from src.adaptation.riemannian_align import RiemannianAlignment

    rng = np.random.default_rng(0)
    mix_a, mix_b = (rng.standard_normal((6, 6)) + 3 * np.eye(6) for _ in range(2))
    X_a = np.einsum("cd,ndt->nct", mix_a, rng.standard_normal((40, 6, 300)))
    X_b = np.einsum("cd,ndt->nct", mix_b, rng.standard_normal((40, 6, 300)))
    ra_a, ra_b = RiemannianAlignment().fit(X_a), RiemannianAlignment().fit(X_b)
    for X, ra in ((X_a, ra_a), (X_b, ra_b)):
        mean_cov = mean_riemann(Covariances(estimator="oas").fit_transform(X))
        np.testing.assert_allclose(ra.whitening_ @ mean_cov @ ra.whitening_, np.eye(6), atol=1e-8)
        np.testing.assert_allclose(ra.transform(X), np.einsum("cd,ndt->nct", ra.whitening_, X))
    assert not np.allclose(ra_a.whitening_, ra_b.whitening_), "fit must depend on the session given"
    np.testing.assert_array_equal(RiemannianAlignment().fit(X_a, y=np.zeros(40)).whitening_, ra_a.whitening_)
