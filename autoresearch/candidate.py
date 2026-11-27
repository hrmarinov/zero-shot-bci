"""The ONLY file the loop edits. Run 0: R11 itself."""

from __future__ import annotations

import numpy as np

from src.pipeline import SubjectData, tslr_member


def predict(own: dict, test: dict, library: list) -> np.ndarray:
    n, m = len(own["y"]), len(test["X_ts"])
    data = SubjectData(X_cnn=np.concatenate([own["X_cnn"], test["X_cnn"]]), X_tslr=np.concatenate([own["X_ts"], test["X_ts"]]),
                       y=np.r_[own["y"], np.zeros(m, int)], groups=np.r_[np.zeros(n), np.ones(m)],
                       train=np.r_[np.ones(n, bool), np.zeros(m, bool)])
    return (test["cnn3"] + tslr_member(data) + test["fts2"]) / 3
