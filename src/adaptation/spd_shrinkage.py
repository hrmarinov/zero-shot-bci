"""Empirical-Bayes shrinkage of noisy SPD covariance estimates.

Transplanted from diffusion-tensor imaging statistics (Bayesian shrinkage
estimation on the SPD manifold via Tweedie's formula, e.g. Kim et al.,
arXiv:2007.02153) — DTI produces the same mathematical object we do (a
per-subject SPD covariance/tensor), and has a mature answer for "this
per-subject SPD estimate is noisy when computed from little data; borrow
strength from a population of other subjects to correct it."

We use the log-Euclidean framework (matrix logarithm centered at identity),
which turns SPD matrices into an ordinary Euclidean vector space where
standard closed-form Gaussian empirical-Bayes shrinkage (the Tweedie/
James-Stein/Efron-Morris result) applies exactly, rather than the paper's
more general anisotropic SURE-tuned estimator — a deliberate simplification
justified by the small number of reference subjects available (single
scalar between/within variance, not a full anisotropic covariance model,
which would be unidentifiable from ~8-50 subjects).
"""

from dataclasses import dataclass

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.geometry.tangentspace import tangent_space, untangent_space


@dataclass
class SPDPopulationPrior:
    """Population statistics for empirical-Bayes shrinkage in log-Euclidean tangent space.

    grand_mean_: population-average tangent vector (the shrinkage target).
    between_subject_var_: variance of *true* per-subject tangent vectors
        around the grand mean (tau^2) — estimated from reference subjects'
        full-data (low-noise) tangent vectors.
    trial_sampling_var_: variance of a *single trial's* tangent vector
        around its own subject's true mean (sigma^2) — determines how fast
        a small-sample subject mean's sampling noise shrinks as 1/n_trials.
    subject_means_: the reference subjects' own tangent vectors (kept for
        fitting an anisotropic/eigenspace prior on top without recomputing).
    """

    reference_: np.ndarray
    grand_mean_: np.ndarray
    between_subject_var_: float
    trial_sampling_var_: float
    subject_means_: np.ndarray


@dataclass
class SPDEigenspacePrior:
    """Anisotropic (eigenvoice-style) prior: per-direction shrinkage using the
    top-k principal directions of between-subject variation (each with its
    own between-subject variance = PCA eigenvalue), instead of one isotropic
    weight shared by every tangent-space dimension. Directions outside the
    top-k are dropped entirely (fully pooled to the grand mean), the same
    "represent a new subject only within the eigenvoice space" assumption
    speaker-adaptation eigenvoices make.
    """

    reference_: np.ndarray
    grand_mean_: np.ndarray
    components_: np.ndarray
    eigenvalues_: np.ndarray
    trial_sampling_var_: float


def _trial_tangent_vectors(
    X: np.ndarray, reference: np.ndarray, estimator: str = "oas", normalize_trace: bool = False
) -> np.ndarray:
    """Per-trial covariances, mapped to log-Euclidean tangent vectors.

    normalize_trace divides each trial's covariance by its own trace before
    mapping to tangent space. Needed when combining subjects recorded on
    different hardware (different amplifier gain, reference scheme, or
    units) into one population prior: absolute EEG power scale differs for
    reasons that have nothing to do with real between-subject neurophysiology
    and would otherwise dominate the between-subject variance estimate.
    Downstream CSP+LDA accuracy is provably unaffected by this (CSP's log-
    variance features under a uniform positive rescaling of every channel
    shift by the same additive constant, which LDA's decision boundary is
    invariant to) — trace normalization only removes a confound from the
    *variance estimate*, not from the classifier's inputs.
    """
    covs = Covariances(estimator=estimator).fit_transform(X)
    if normalize_trace:
        traces = np.trace(covs, axis1=1, axis2=2)
        covs = covs / traces[:, np.newaxis, np.newaxis]
    return tangent_space(covs, reference, metric="logeuclid")


def logeuclid_mean_cov(X: np.ndarray, estimator: str = "oas") -> np.ndarray:
    """Raw (unshrunk) log-Euclidean mean covariance — the shrinkage estimator's
    fair baseline, since it isolates the effect of shrinkage from any change in
    which mean (log-Euclidean vs. affine-invariant Riemannian) is used."""
    n_channels = X.shape[1]
    reference = np.eye(n_channels)
    mean_vec = _trial_tangent_vectors(X, reference, estimator=estimator).mean(axis=0)
    return untangent_space(mean_vec[np.newaxis, :], reference, metric="logeuclid")[0]


def fit_population_prior(
    reference_subjects_X: list[np.ndarray], estimator: str = "oas", normalize_trace: bool = False
) -> SPDPopulationPrior:
    """Fit population prior from several reference subjects' full-data trials.

    reference_subjects_X: one (n_trials, n_channels, n_times) array per
    reference subject, using all of that subject's available trials (their
    own subject-level tangent vector is assumed well-estimated).
    """
    n_channels = reference_subjects_X[0].shape[1]
    reference = np.eye(n_channels)

    subject_means = []
    within_subject_devs = []
    for X in reference_subjects_X:
        trial_vecs = _trial_tangent_vectors(X, reference, estimator=estimator, normalize_trace=normalize_trace)
        subject_mean = trial_vecs.mean(axis=0)
        subject_means.append(subject_mean)
        within_subject_devs.append(trial_vecs - subject_mean)

    subject_means = np.array(subject_means)
    grand_mean = subject_means.mean(axis=0)
    between_subject_var = float(np.mean((subject_means - grand_mean) ** 2))
    trial_sampling_var = float(np.mean(np.concatenate(within_subject_devs, axis=0) ** 2))

    return SPDPopulationPrior(
        reference_=reference,
        grand_mean_=grand_mean,
        between_subject_var_=between_subject_var,
        trial_sampling_var_=trial_sampling_var,
        subject_means_=subject_means,
    )


def shrink_subject_mean(
    prior: SPDPopulationPrior, X_subject: np.ndarray, estimator: str = "oas", normalize_trace: bool = False
) -> np.ndarray:
    """Empirical-Bayes shrinkage estimate of a subject's true mean covariance.

    Uses only X_subject's own trials (e.g. a short, few-trial recording) plus
    the population prior — no labels needed. Returns an SPD matrix, drop-in
    replacement for the raw empirical mean covariance used elsewhere (e.g.
    as the recentering reference in RiemannianAlignment). normalize_trace
    must match whatever fit_population_prior used, so the target subject's
    tangent vector lives in the same (trace-normalized or not) space as the
    prior it's being shrunk toward; the *shrinkage weight* is computed in
    that normalized space (removing cross-hardware scale confounds from the
    variance estimate), but the returned covariance is rescaled back to this
    subject's own natural magnitude — leaving it on a ~unit-trace scale
    instead (orders of magnitude off real EEG covariance scale) risks
    numerical issues downstream (e.g. in CSP's own internal regularization)
    even though a pure scalar is theoretically harmless to CSP+LDA accuracy.
    """
    n_trials = X_subject.shape[0]
    trial_vecs = _trial_tangent_vectors(X_subject, prior.reference_, estimator=estimator, normalize_trace=normalize_trace)
    raw_mean = trial_vecs.mean(axis=0)

    sampling_var = prior.trial_sampling_var_ / n_trials
    weight = prior.between_subject_var_ / (prior.between_subject_var_ + sampling_var)
    shrunk = prior.grand_mean_ + weight * (raw_mean - prior.grand_mean_)
    shrunk_cov = untangent_space(shrunk[np.newaxis, :], prior.reference_, metric="logeuclid")[0]

    if normalize_trace:
        natural_covs = Covariances(estimator=estimator).fit_transform(X_subject)
        natural_trace = float(np.mean(np.trace(natural_covs, axis1=1, axis2=2)))
        shrunk_cov = shrunk_cov * natural_trace

    return shrunk_cov


def fit_eigenspace_prior(prior: SPDPopulationPrior, n_components: int) -> SPDEigenspacePrior:
    """Build an eigenvoice-style anisotropic prior from an already-fit isotropic prior.

    n_components is capped at (n_reference_subjects - 1), the maximum number
    of non-trivial directions of variation a finite reference set can define.
    """
    centered = prior.subject_means_ - prior.grand_mean_
    n_ref = centered.shape[0]
    k = min(n_components, n_ref - 1)

    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singular_values**2) / n_ref

    return SPDEigenspacePrior(
        reference_=prior.reference_,
        grand_mean_=prior.grand_mean_,
        components_=components[:k],
        eigenvalues_=eigenvalues[:k],
        trial_sampling_var_=prior.trial_sampling_var_,
    )


def shrink_subject_mean_eigenspace(
    prior: SPDEigenspacePrior, X_subject: np.ndarray, estimator: str = "oas"
) -> np.ndarray:
    """Per-direction empirical-Bayes shrinkage in the eigenvoice space.

    Same idea as shrink_subject_mean, but each principal direction gets its
    own shrinkage weight (based on that direction's own between-subject
    variance) instead of one shared isotropic weight — directions with more
    real between-subject signal are trusted more; directions outside the
    top-k eigenspace are dropped (fully pooled to the grand mean).
    """
    n_trials = X_subject.shape[0]
    trial_vecs = _trial_tangent_vectors(X_subject, prior.reference_, estimator=estimator)
    raw_mean = trial_vecs.mean(axis=0)

    centered = raw_mean - prior.grand_mean_
    coords = prior.components_ @ centered

    sampling_var = prior.trial_sampling_var_ / n_trials
    weights = prior.eigenvalues_ / (prior.eigenvalues_ + sampling_var)
    shrunk_coords = coords * weights
    shrunk = prior.grand_mean_ + shrunk_coords @ prior.components_

    return untangent_space(shrunk[np.newaxis, :], prior.reference_, metric="logeuclid")[0]
