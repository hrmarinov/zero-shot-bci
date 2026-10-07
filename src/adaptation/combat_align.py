"""Within-subject cross-session harmonization via ComBat (Johnson, Li &
Rabinovic 2007) - transplanted from genomics, where it's the standard fix
for "batch effects": the same underlying biological signal, measured in
different sequencing runs/scanners/sites, shifted by a run-specific
location and scale effect. Session-to-session EEG drift is structurally
the same problem (same subject, different recording day, shifted location/
scale), which is why this project already tests recenter+rescale for it -
ComBat's difference is *how* it estimates that per-dimension location/
scale correction: instead of one global mean and one global dispersion
statistic (this project's existing recenter()/rescale() in
riemannian_icp.py), ComBat estimates a separate location/scale shift *per
tangent-space dimension* (253 of them for IV-2a's 22 channels), then
shrinks those per-dimension estimates via an empirical-Bayes prior fit
across all dimensions - borrowing statistical strength between dimensions,
intended to help most when there aren't enough trials to estimate each
dimension's own shift reliably in isolation (recenter+rescale's single
global statistic never has this problem, but also can't capture batch
effects that are heterogeneous across dimensions).

Unsupervised - no labels needed on either session, the same category as
recenter+rescale (this project's own established, safe, no-labels-needed
correction) - not a replacement for the supervised rotation step, which
solves a different (orientation, not location/scale) part of the problem.

Uses the `neuroCombat` package directly rather than `neuroHarmonize`
(which provides a friendlier fit/apply-to-new-data API): installing
neuroHarmonize downgraded numpy to a version incompatible with this
project's own moabb dependency, so it was reverted. neuroCombat's
harmonization has to see every sample being harmonized in one call - no
separate "fit on a subsample, apply to unseen samples" step available -
so this only harmonizes the sessions' full trial sets pooled together in
one call.
"""

import numpy as np
import pandas as pd

# neuroCombat (unmaintained since before numpy 2.0) calls the long-removed
# np.int alias internally and crashes on import-time use otherwise. This is
# exactly the safe, narrow fix numpy's own deprecation message suggests
# (np.int was always just a deprecated alias for the builtin int, not a
# distinct type - restoring it doesn't change any numeric behavior) - scoped
# to this module, not a global numpy monkeypatch applied elsewhere.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

from neuroCombat import neuroCombat  # noqa: E402 (must follow the np.int shim above)


def combat_harmonize(calib_vecs: np.ndarray, eval_vecs: np.ndarray) -> np.ndarray:
    """Harmonizes eval_vecs into calib_vecs' reference frame. Returns the
    harmonized eval vectors only (same shape as eval_vecs); calib_vecs are
    the ComBat reference batch, so their own harmonized values are ~identical
    to the input and not returned.
    """
    n_calib = len(calib_vecs)
    dat = np.concatenate([calib_vecs, eval_vecs], axis=0).T  # neuroCombat wants (features, samples)
    covars = pd.DataFrame({"batch": ["calib"] * n_calib + ["eval"] * len(eval_vecs)})

    result = neuroCombat(dat=dat, covars=covars, batch_col="batch", ref_batch="calib")
    harmonized = result["data"].T  # back to (samples, features)
    return harmonized[n_calib:]
