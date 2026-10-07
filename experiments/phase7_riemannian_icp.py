"""Phase 7 - within-subject Riemannian ICP realignment, with an RPA-bridging ablation.

Per user direction: build ICP (src/adaptation/riemannian_icp.py) as the
base method, then add recentering + rescaling + rotation on top and test
whether they improve over base ICP alone - an ablation, not one combined
method. All alignment happens within one subject's own calibration <->
eval sessions - no other subject's data anywhere.

Design note (a real bug caught before running this, not a stylistic
choice): ICP's rotation is solved *about each point cloud's own mean*
(both clouds are recentered to zero before the rotation is fit) - a
correction scheme that only ever recomputes a single *mean* reference
covariance (the way RiemannianAlignment's whitening does) is structurally
blind to rotation, since a rotation about the origin leaves the origin
itself unchanged. Testing whether ICP's rotation-finding does anything at
all requires applying the alignment *per trial*, not to a single derived
mean. So classification here works directly in log-Euclidean tangent
space (LDA on tangent vectors, the same representation and evaluation
style as geometric_correction.py's Module A and prototypical_network.py),
not through CSP - CSP needs raw multi-channel signal, which per-trial
tangent-space realignment doesn't directly produce. This means the
baseline number here is "tangent-space LDA, no realignment", not CSP+LDA's
62.2% - a different feature representation, reported for a fair apples-to-
apples ablation within this experiment, not a direct comparison to
Phase 1/2's numbers.

Five conditions, each producing per-trial-aligned eval tangent vectors,
classified by an LDA trained on calibration tangent vectors:
  - none: no realignment (the fair within-experiment baseline)
  - recenter_rescale: unsupervised recentering + dispersion-matching
  - recenter_rescale_icp: adds ICP refinement on top
  - recenter_rescale_rotation: adds the few-shot supervised rotation step
    instead of ICP
  - recenter_rescale_rotation_icp: rotation first, then ICP refines further
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.adaptation.riemannian_icp import (
    dispersion,
    icp_align,
    recenter,
    rescale,
    supervised_rotation_from_labels,
    trial_tangent_vectors,
)
from src.config import IV2A_SUBJECTS, RANDOM_STATE, RESULTS_DIR
from src.datasets import load_iv2a_subject
from src.evaluate import accuracy_summary

SAMPLE_SIZES = [16, 32, 64, 128, 288]
N_REPEATS = 5
N_ROTATION_LABEL_TRIALS = 16  # few-shot labeled eval trials for the supervised rotation step
CLASSES = ["left_hand", "right_hand", "feet", "tongue"]


def align_eval_trials(
    calib_vecs: np.ndarray, calib_y: np.ndarray, eval_sub_vecs: np.ndarray, eval_sub_y: np.ndarray | None,
    all_eval_vecs: np.ndarray, method: str,
) -> np.ndarray:
    """Returns all_eval_vecs, realigned into calib's frame per the given
    method. The alignment (recenter/rescale/rotation/ICP) is *fit* using
    only eval_sub_vecs (the available unlabeled-or-few-shot-labeled
    subsample, matching every other phase's "how much eval data is
    available" framing), then *applied* to all_eval_vecs (the full eval
    session), matching phase2b/phase2d's fit-on-subsample/apply-to-all
    pattern.
    """
    if method == "none":
        return all_eval_vecs

    calib_centered, calib_mean = recenter(calib_vecs)
    eval_sub_centered, eval_sub_mean = recenter(eval_sub_vecs)
    target_disp = dispersion(calib_centered)
    scale = np.sqrt(target_disp / (dispersion(eval_sub_centered) + 1e-12))

    def apply_recenter_rescale(vecs: np.ndarray) -> np.ndarray:
        return (vecs - eval_sub_mean) * scale

    all_eval_rr = apply_recenter_rescale(all_eval_vecs)

    if method == "recenter_rescale":
        return all_eval_rr + calib_mean

    if method == "recenter_rescale_icp":
        # eval_sub_rr and calib_centered are already both ~zero-mean, so
        # icp_align's own internal recentering is close to a no-op here -
        # its rotation is what matters. Applying that rotation to the full
        # (already recenter+rescaled) eval set, then shifting into calib's
        # absolute frame, is the actual per-trial realignment.
        eval_sub_rr = apply_recenter_rescale(eval_sub_vecs)
        result = icp_align(eval_sub_rr, calib_centered, max_iters=100)
        return all_eval_rr @ result.rotation.T + calib_mean

    if method in ("recenter_rescale_rotation", "recenter_rescale_rotation_icp"):
        if eval_sub_y is None:
            raise ValueError(f"method={method!r} needs labeled eval_sub_y for the supervised rotation step")
        eval_sub_rr = apply_recenter_rescale(eval_sub_vecs)
        R = supervised_rotation_from_labels(eval_sub_rr, eval_sub_y, calib_centered, calib_y, CLASSES)
        rotated_all = all_eval_rr @ R.T

        if method == "recenter_rescale_rotation":
            return rotated_all + calib_mean

        rotated_sub = eval_sub_rr @ R.T
        result = icp_align(rotated_sub, calib_centered, max_iters=100, init_rotation=np.eye(rotated_sub.shape[1]))
        return rotated_all @ result.rotation.T + calib_mean

    raise ValueError(f"unknown method={method!r}")


def stratified_sample_idx(y: np.ndarray, n: int, classes: list[str], rng: np.random.Generator) -> np.ndarray:
    """n trials, drawn evenly across classes - not pure random pooling.

    Found necessary, not just tidier: random pooling across all trials let
    a small n (e.g. 16 trials over 4 classes) draw *zero* trials of some
    class purely by chance, producing a NaN class mean and crashing the
    supervised rotation step's SVD. Every class-mean-based method here
    needs every class actually represented in the sample it's given.
    """
    if n < len(classes):
        raise ValueError(f"n={n} is smaller than the number of classes ({len(classes)}) - cannot stratify")
    per_class = n // len(classes)
    remainder = n - per_class * len(classes)
    idx_parts = []
    for i, c in enumerate(classes):
        class_idx = np.flatnonzero(y == c)
        take = per_class + (1 if i < remainder else 0)
        take = min(take, len(class_idx))
        idx_parts.append(rng.choice(class_idx, size=take, replace=False))
    return np.concatenate(idx_parts)


def run(subjects: list[int] = IV2A_SUBJECTS, out_name: str = "phase7_riemannian_icp.csv") -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    methods = ["none", "recenter_rescale", "recenter_rescale_icp", "recenter_rescale_rotation", "recenter_rescale_rotation_icp"]

    for subject in subjects:
        print(f"\n[subject {subject}] loading...")
        data = load_iv2a_subject(subject)

        calib_vecs = trial_tangent_vectors(data.X_calib)
        all_eval_vecs = trial_tangent_vectors(data.X_eval)

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(calib_vecs, data.y_calib)

        for n in SAMPLE_SIZES:
            n_repeats = 1 if n == data.X_eval.shape[0] else N_REPEATS
            for repeat in range(n_repeats):
                idx = stratified_sample_idx(data.y_eval, n, CLASSES, rng)
                eval_sub_vecs = all_eval_vecs[idx]
                eval_sub_y = data.y_eval[idx] if n >= N_ROTATION_LABEL_TRIALS else None

                for method in methods:
                    if method.startswith("recenter_rescale_rotation") and eval_sub_y is None:
                        continue
                    aligned = align_eval_trials(calib_vecs, data.y_calib, eval_sub_vecs, eval_sub_y, all_eval_vecs, method)
                    y_pred = lda.predict(aligned)
                    acc, _, _ = accuracy_summary(data.y_eval, y_pred)
                    rows.append({"subject": subject, "n_trials": n, "repeat": repeat, "method": method, "accuracy": acc})

        print(f"[subject {subject}] done")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / out_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    summary = df.groupby(["method", "n_trials"])["accuracy"].agg(["mean", "std"]).reset_index()
    print(summary.to_string(index=False))
    return df


if __name__ == "__main__":
    run()
