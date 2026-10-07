"""Pool-based active learning for BCI calibration trial selection.

Source field: adaptive psychophysics (QUEST/QUEST+, see docs/state_of_the_art.md
Section 10.4) — instead of running a fixed-length, fixed-order calibration
block, adaptively pick which trial to run next based on how much it would
tell you about the subject, converging in fewer trials than a fixed
protocol. This module implements the offline, pool-based simulation of that
idea: the "pool" is the fixed set of already-collected calibration trials;
at deployment each reveal would instead mean asking the subject to attempt
one more imagery trial of a chosen class.
"""

import numpy as np
from sklearn.base import clone


def _predictive_entropy(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, 1e-12, 1.0)
    return -np.sum(proba * np.log(proba), axis=1)


def uncertainty_sampling_batches(
    pipeline,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    seed_idx: np.ndarray,
    batch_size: int,
) -> list[np.ndarray]:
    """Reveal order: seed first, then whichever pool trials the current
    partial model is least certain about (highest predictive entropy),
    retraining after each batch. Returns the sequence of index batches.
    """
    revealed: list[int] = list(seed_idx)
    remaining = [i for i in range(len(y_pool)) if i not in set(revealed)]
    batches = [np.array(revealed)]

    while remaining:
        model = clone(pipeline)
        model.fit(X_pool[revealed], y_pool[revealed])
        proba = model.predict_proba(X_pool[remaining])
        entropy = _predictive_entropy(proba)
        order = np.argsort(-entropy)
        take = [remaining[i] for i in order[:batch_size]]
        revealed.extend(take)
        remaining = [i for i in remaining if i not in set(take)]
        batches.append(np.array(take))

    return batches


def fixed_order_batches(order_idx: np.ndarray, seed_idx: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Reveal order determined up front (random permutation, or the trials'
    original as-collected order) — the non-adaptive baselines to compare
    uncertainty sampling against.
    """
    seed_set = set(seed_idx)
    remaining = [i for i in order_idx if i not in seed_set]
    batches = [np.array(seed_idx)]
    for start in range(0, len(remaining), batch_size):
        batches.append(np.array(remaining[start : start + batch_size]))
    return batches
