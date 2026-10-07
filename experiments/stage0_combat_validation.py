"""Stage 0 gate for ComBat harmonization (src/adaptation/combat_align.py),
before touching real IV-2a data.

The specific hypothesis to test, not just "does ComBat do something
reasonable": recenter+rescale (this project's existing, established
correction) fits ONE global mean-shift and ONE global dispersion-scale
statistic across all tangent-space dimensions - it cannot represent a
batch effect that is *heterogeneous* across dimensions (dimension 5 shifted
more than dimension 50, say). ComBat fits a *separate* location/scale
correction per dimension, regularized via empirical Bayes. So: does ComBat
recover more of a downstream classification signal than recenter+rescale
specifically when the planted batch effect is heterogeneous across
dimensions - the exact case recenter+rescale is structurally blind to?

Synthetic model: D=253 dimensions (matching IV-2a's 22-channel tangent
space exactly), K=4 class clusters (matching IV-2a's 4 classes), a KNOWN
per-dimension location shift and per-dimension scale factor applied to the
"eval" session only (calib stays clean) - checked via downstream nearest-
class-mean classification accuracy on held-out eval trials, comparing no
correction, recenter+rescale, and ComBat.

**Result found, checked across four eval trial counts (16/32/80/160), not
just one**: ComBat clearly beats no correction every time, but never beats
recenter+rescale - not a small-sample artifact (more samples didn't change
which one won). Best explanation found: this synthetic batch effect was
deliberately constructed with NO shared structure across dimensions (each
dimension's shift/scale drawn i.i.d.) - close to the worst case for
ComBat's core assumption, that per-feature batch effects share a common
distribution worth empirical-Bayes-shrinking toward. Real tangent-space
dimensions have the same property for a structural reason, not just this
test's construction: they are not independent features the way genomics
features are - all 253 of them are derived from one much lower-rank 22x22
covariance matrix, so real session drift plausibly has far fewer effective
degrees of freedom than "253 independent per-feature batch effects" - a
mismatch recenter+rescale's simpler, SPD-structure-respecting single
global correction doesn't have to fight.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.adaptation.combat_align import combat_harmonize
from src.adaptation.riemannian_icp import dispersion, recenter, rescale

D = 253  # matches IV-2a's 22-channel tangent-space dimensionality exactly
CLASSES = ["left_hand", "right_hand", "feet", "tongue"]
N_TRIALS_PER_CLASS = 40
WITHIN_CLASS_STD = 1.0
CLASS_SEPARATION = 0.8
BATCH_SHIFT_STD = 4.0  # heterogeneous per-dimension location shift magnitude
BATCH_SCALE_STD = 1.2  # heterogeneous per-dimension scale-factor spread (multiplicative, around 1.0)
NOISE_STD = 2.5
# The above (not the initially-tried 2.5/1.5/0.4/1.0) were needed to avoid a
# ceiling effect: with D=253 dimensions, per-dimension noise/shift partially
# cancels in a nearest-mean Euclidean classifier's aggregate distance (a
# blessing-of-dimensionality effect), so the milder settings gave 100%
# accuracy even with no correction at all - no room to show any method's
# effect. Tuned by sweeping, not guessed once and trusted.
RANDOM_STATE = 7


def generate_session(class_means: np.ndarray, batch_shift: np.ndarray, batch_scale: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i, c in enumerate(CLASSES):
        for _ in range(N_TRIALS_PER_CLASS):
            clean = class_means[i] + rng.normal(scale=WITHIN_CLASS_STD, size=D)
            drifted = clean * batch_scale + batch_shift + rng.normal(scale=NOISE_STD, size=D)
            X.append(drifted)
            y.append(c)
    X, y = np.array(X), np.array(y)
    order = rng.permutation(len(y))
    return X[order], y[order]


def nearest_mean_classify(calib_vecs: np.ndarray, calib_y: np.ndarray, eval_vecs: np.ndarray) -> np.ndarray:
    means = {c: calib_vecs[calib_y == c].mean(axis=0) for c in CLASSES}
    dists = np.stack([np.linalg.norm(eval_vecs - means[c], axis=1) for c in CLASSES], axis=1)
    return np.array(CLASSES)[dists.argmin(axis=1)]


def run() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    class_means = rng.normal(scale=CLASS_SEPARATION, size=(len(CLASSES), D))

    # calibration session: no batch effect (shift=0, scale=1)
    X_calib, y_calib = generate_session(class_means, np.zeros(D), np.ones(D), rng)

    # eval session: a KNOWN, per-dimension HETEROGENEOUS batch effect -
    # exactly what recenter+rescale's single global statistic can't capture.
    batch_shift = rng.normal(scale=BATCH_SHIFT_STD, size=D)
    batch_scale = 1.0 + rng.normal(scale=BATCH_SCALE_STD, size=D)
    X_eval, y_eval = generate_session(class_means, batch_shift, batch_scale, rng)

    acc_none = (nearest_mean_classify(X_calib, y_calib, X_eval) == y_eval).mean()

    calib_centered, calib_mean = recenter(X_calib)
    eval_centered, eval_mean = recenter(X_eval)
    target_disp = dispersion(calib_centered)
    eval_rescaled = rescale(eval_centered, target_disp) + calib_mean
    acc_recenter_rescale = (nearest_mean_classify(X_calib, y_calib, eval_rescaled) == y_eval).mean()

    eval_combat = combat_harmonize(X_calib, X_eval)
    acc_combat = (nearest_mean_classify(X_calib, y_calib, eval_combat) == y_eval).mean()

    print(f"  no correction:              {acc_none:.3f}")
    print(f"  recenter+rescale (global):  {acc_recenter_rescale:.3f}")
    print(f"  ComBat (per-dim, shrunk):   {acc_combat:.3f}")

    assert acc_combat > acc_none + 0.05, f"ComBat should clearly beat no correction (none={acc_none:.3f}, combat={acc_combat:.3f})"
    # NOT asserting acc_combat > acc_recenter_rescale - checked across four
    # eval trial counts (16/32/80/160) and ComBat consistently tied or
    # underperformed recenter+rescale at every one, never won. Not a small-
    # sample artifact (more samples didn't change the ordering) - see this
    # script's own module docstring for the real explanation found: this
    # synthetic batch effect was constructed as i.i.d. random per dimension,
    # with no shared cross-dimensional structure at all, which is close to
    # the worst case for ComBat's core assumption (that per-feature batch
    # effects share a common distribution worth shrinking toward). Real
    # tangent-space dimensions aren't independent features the way genomics
    # features are either - they're all derived from one much lower-rank
    # 22x22 covariance matrix - so this mismatch is plausibly not just a
    # quirk of this synthetic test, but a real structural reason to expect
    # ComBat's genomics-style independent-feature model won't beat
    # recenter+rescale's simpler, SPD-structure-respecting global
    # correction. Reported honestly rather than tuned until it wins.
    print("\n  ComBat clearly beats no correction, but does not beat recenter+rescale (checked at 4 "
          "different eval trial counts, consistently) - see module docstring for why this is plausibly a "
          "real structural mismatch (independent-per-feature assumption vs. low-rank covariance-derived "
          "features), not just an unlucky synthetic setup. Real IV-2a data will be checked for completeness, "
          "but do not expect ComBat to beat recenter+rescale there either.")


if __name__ == "__main__":
    run()
