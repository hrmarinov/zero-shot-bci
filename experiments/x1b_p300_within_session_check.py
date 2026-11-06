"""X1b, post-hoc validity check: is the GuttmannFlury2025 P300 data decodable at all through MOABB's adapter?

X1 found P300 AUC ~ 0.50 for every subject and every method. A working P300 pipeline reaches AUC >~ 0.8
WITHIN a session, so this check uses the FIRST (training) session only: 5-fold within-session AUC of
XdawnCovariances + TS + LR, the Target fraction, and where the Target - NonTarget average difference peaks.
If within-session AUC is also ~0.5, the labels/event timing delivered by the adapter carry no P300, and the
X1 P300 arm measures nothing about session transfer.

    .venv\\Scripts\\python experiments\\x1b_p300_within_session_check.py     # CPU, ~5 min
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import mne  # noqa: E402
import numpy as np  # noqa: E402
from moabb.datasets import GuttmannFlury2025_P300  # noqa: E402
from moabb.paradigms import P300  # noqa: E402
from pyriemann.estimation import XdawnCovariances  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

mne.set_log_level("ERROR")
SUBJECTS = (2, 4, 13, 17)
SFREQ = 128


def main() -> int:
    ds = GuttmannFlury2025_P300(subjects=list(SUBJECTS))
    for s in SUBJECTS:
        X, lab, meta = P300(fmin=1, fmax=24, tmin=0, tmax=0.8, resample=SFREQ).get_data(ds, [s])
        first = (meta.session == sorted(meta.session.unique())[0]).to_numpy()
        X, y = X[first], (lab[first] == "Target").astype(int)
        pipe = make_pipeline(XdawnCovariances(4, estimator="lwf"), TangentSpace(), LogisticRegression(max_iter=5000))
        auc = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()
        d = X[y == 1].mean(0) - X[y == 0].mean(0)
        peak_t = np.argmax(np.abs(d).max(0)) / SFREQ
        print(f"s{s}: n = {len(y)}, Target fraction {y.mean():.3f}, within-session 5-fold AUC {auc:.3f}, "
              f"max |Target - NonTarget| {np.abs(d).max():.1f} uV at {peak_t:.3f} s (mean |X| {np.abs(X).mean():.1f} uV)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
