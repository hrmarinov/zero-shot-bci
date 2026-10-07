"""Phase 6 - prototypical-network meta-learning: PhysioNet base, IV-2a target.

The real-data test of the mechanism described in docs/progress_and_direction.md's
"External literature check" (MetaWearS, Amirshahi et al., Commun Med 2026).
Encoder meta-trained via episodic training on PhysioNet (108 subjects,
classes: left_hand/right_hand/feet/hands), with disjoint support/query
subject pools per episode - never fine-tuned on IV-2a. At IV-2a deployment,
per-class prototypes are computed fresh from each subject's own calibration
trials (note IV-2a's classes are left_hand/right_hand/feet/*tongue* - a
class PhysioNet never has; this is fine and expected for a prototypical
network, whose whole point is generalizing to novel classes via a learned
metric, not memorizing specific class identities), and eval trials are
classified by nearest prototype.

Structurally cannot fail the way Module A failed (PhysioNet's population
covariance shape being far from IV-2a's): the classification decision
never depends on that shape match, only on the target subject's own
support trials and the learned metric's ability to separate them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.adaptation.prototypical_network import (
    SubjectTrials,
    compute_prototypes,
    predict_via_prototypes,
    train_prototypical_network,
)
from src.config import IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import PHYSIONET_EXCLUDED_SUBJECTS, PHYSIONET_MI_EVENTS, load_iv2a_subject, load_physionet_reference_subject_labeled
from src.evaluate import accuracy_summary, wilson_interval

RANDOM_STATE = 42


def build_physionet_pool(n_subjects: int) -> list[SubjectTrials]:
    subject_ids = [s for s in range(1, 110) if s not in PHYSIONET_EXCLUDED_SUBJECTS][:n_subjects]
    pool = []
    for s in subject_ids:
        X, y = load_physionet_reference_subject_labeled(s)
        X_by_class = {label: X[y == label] for label in PHYSIONET_MI_EVENTS}
        pool.append(SubjectTrials(X_by_class=X_by_class))
        print(f"  PhysioNet subject {s}: " + ", ".join(f"{l}={len(X_by_class[l])}" for l in PHYSIONET_MI_EVENTS))
    return pool


def run(n_physionet_subjects: int = 108, iv2a_subjects: list[int] = IV2A_SUBJECTS, out_name: str = "phase6_prototypical.csv") -> pd.DataFrame:
    print(f"Loading {n_physionet_subjects} PhysioNet subjects...")
    physionet_pool = build_physionet_pool(n_physionet_subjects)

    print("\nMeta-training encoder via episodic training on PhysioNet...")
    encoder = train_prototypical_network(
        train_subjects=physionet_pool, classes=PHYSIONET_MI_EVENTS,
        random_state=RANDOM_STATE, verbose=True,
    )

    rows = []
    for subject in iv2a_subjects:
        print(f"\n[IV-2a subject {subject}] loading...")
        data = load_iv2a_subject(subject)
        calib_by_class = {label: data.X_calib[data.y_calib == label] for label in IV2A_CLASSES}

        prototypes = compute_prototypes(encoder, calib_by_class)
        y_pred = predict_via_prototypes(encoder, prototypes, data.X_eval)

        acc, n_correct, n_total = accuracy_summary(data.y_eval, y_pred)
        ci_low, ci_high = wilson_interval(n_correct, n_total)
        print(f"[IV-2a subject {subject}] cross-session accuracy: {acc:.3f} ({n_correct}/{n_total})  "
              f"95% CI [{ci_low:.3f}, {ci_high:.3f}]")

        rows.append({
            "subject": subject, "accuracy": acc, "n_correct": n_correct, "n_total": n_total,
            "ci_low": ci_low, "ci_high": ci_high,
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / out_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(f"\nMean cross-session accuracy: {df['accuracy'].mean():.3f} (std {df['accuracy'].std():.3f})")
    print("Comparison: Phase 4b (wavelet calibration) = 31.4%, Phase 1 (raw CSP+LDA) = 62.2%, "
          "Phase 2 (Riemannian alignment) = 66.9%, chance level = 25.0%")
    return df


if __name__ == "__main__":
    run()
