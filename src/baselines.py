"""Non-adapted baseline classifiers."""

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE


def make_csp_lda(n_components: int = 8) -> Pipeline:
    """Standard CSP+LDA pipeline for motor-imagery classification.

    CSP learns spatial filters that maximize variance ratio between classes;
    LDA classifies the resulting log-variance features. This is the classical
    strong baseline that most cross-session adaptation methods are compared
    against.
    """
    csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    return Pipeline([("csp", csp), ("lda", lda)])
