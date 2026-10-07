"""Within-session and cross-session evaluation protocols (no adaptation)."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, norm
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from src.config import RANDOM_STATE


@dataclass
class SubjectResult:
    subject: int
    within_session_acc: float
    within_session_std: float
    cross_session_acc: float
    cross_session_ci_low: float
    cross_session_ci_high: float
    cross_session_n: int

    @property
    def gap(self) -> float:
        return self.within_session_acc - self.cross_session_acc


def wilson_interval(n_correct: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal (Wald) approximation for accuracy estimates:
    stays within [0, 1] and stays well-calibrated even when accuracy is near
    chance level or n is a few hundred trials, both of which apply here.
    """
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = phat + z**2 / (2 * n_total)
    margin = z * np.sqrt(phat * (1 - phat) / n_total + z**2 / (4 * n_total**2))
    return (center - margin) / denom, (center + margin) / denom


def within_session_accuracy(
    pipeline, X: np.ndarray, y: np.ndarray, n_splits: int = 5, n_repeats: int = 10
) -> tuple[float, float]:
    """Repeated stratified K-fold CV accuracy within a single session.

    A single 5-fold split on ~288 trials has ~58-trial test folds, so its
    per-subject estimate carries substantial sampling variance. Repeating the
    split n_repeats times (each with a different fold assignment) and pooling
    all resulting scores tightens the mean estimate and, via the std, makes
    the noise floor visible instead of hiding it in a single point value.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    scores = cross_val_score(clone(pipeline), X, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def fit_predict(pipeline, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Train on one set of trials, predict labels for another."""
    model = clone(pipeline)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def accuracy_summary(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int, int]:
    """Returns (accuracy, n_correct, n_total) for a set of predictions.

    The counts let the caller build a binomial confidence interval around a
    single deterministic train/test estimate (e.g. one evaluation session),
    which can't be tightened by repeated resampling the way within-session CV
    can.
    """
    n_correct = int(np.sum(np.asarray(y_pred) == np.asarray(y_true)))
    n_total = len(y_true)
    return n_correct / n_total, n_correct, n_total


def cross_session_accuracy(
    pipeline, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[float, int, int]:
    """Train on one session, test on another, with zero adaptation."""
    y_pred = fit_predict(pipeline, X_train, y_train, X_test)
    return accuracy_summary(y_test, y_pred)


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> float:
    """Exact McNemar test p-value comparing two classifiers on the same trials.

    Only trials where the two predictions disagree on correctness are
    informative for whether accuracy actually changed, as opposed to two
    independent confidence intervals which would also react to trials both
    models get right or both get wrong. Returns 1.0 (no evidence of a
    difference) if there are no discordant trials.
    """
    correct_a = np.asarray(y_pred_a) == np.asarray(y_true)
    correct_b = np.asarray(y_pred_b) == np.asarray(y_true)
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n_discordant = a_only + b_only
    if n_discordant == 0:
        return 1.0
    return binomtest(min(a_only, b_only), n_discordant, p=0.5).pvalue
