"""Stage 0 gate for the prototypical-network mechanism (src/adaptation/prototypical_network.py).

Same discipline as every other mechanism in this project: validate on
synthetic data with known ground truth before spending real compute
(PhysioNet meta-training) on it. Generates many synthetic "subjects" (each
with its own randomized ERD/ERS parameters - a known, personalized ground-
truth signature per subject, exactly matching what "personalization" needs
to be checked against), meta-trains an encoder via episodic training on
most of them, then checks whether nearest-prototype classification works
on *held-out* synthetic subjects never seen during meta-training - the
same "generalizes to a genuinely new subject" property the real IV-2a
deployment needs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.adaptation.prototypical_network import (
    SubjectTrials,
    compute_prototypes,
    predict_via_prototypes,
    sample_episode,
    train_prototypical_network,
)
from src.datasets import IV2A_CHANNELS
from src.evaluate import accuracy_summary
from src.synthetic_eeg import generate_synthetic_subject

SFREQ = 250.0
DURATION_S = 3.0
SNR_DB = 0.0
N_TRAIN_SUBJECTS = 60
N_TEST_SUBJECTS = 15
N_TRIALS_PER_CLASS = 40  # 80 trials/subject - enough for several support/query draws
CLASSES = ["left_hand", "right_hand"]
RANDOM_STATE = 11


def generate_pool(n_subjects: int, seed_offset: int) -> list[SubjectTrials]:
    pool = []
    for i in range(n_subjects):
        rng = np.random.default_rng(RANDOM_STATE + seed_offset + i)
        ds = generate_synthetic_subject(
            IV2A_CHANNELS, sfreq=SFREQ, n_trials_per_class=N_TRIALS_PER_CLASS,
            duration_s=DURATION_S, snr_db=SNR_DB, rng=rng,
        )
        X_by_class = {label: ds.X[ds.y == label] for label in CLASSES}
        pool.append(SubjectTrials(X_by_class=X_by_class))
    return pool


def run() -> None:
    print(f"Generating {N_TRAIN_SUBJECTS} synthetic meta-train subjects...")
    train_pool = generate_pool(N_TRAIN_SUBJECTS, seed_offset=0)
    print(f"Generating {N_TEST_SUBJECTS} held-out synthetic meta-test subjects (never used in training)...")
    test_pool = generate_pool(N_TEST_SUBJECTS, seed_offset=10_000)

    print("\nMeta-training encoder via episodic training on synthetic subjects...")
    encoder = train_prototypical_network(
        train_subjects=train_pool, classes=CLASSES,
        n_support_subjects=8, n_query_subjects=8, k_shot=5, m_query=5,
        max_episodes=2000, eval_every=50, patience_evals=10,
        random_state=RANDOM_STATE, verbose=True,
    )

    print(f"\nEvaluating on {N_TEST_SUBJECTS} held-out synthetic subjects "
          f"(prototypes from a few of each subject's own trials, classify the rest)...")
    rng = np.random.default_rng(RANDOM_STATE + 99)
    k_shot = 10
    all_y_true, all_y_pred = [], []
    for subject in test_pool:
        support_X, support_y, query_X, query_y = {}, [], [], []
        for label in CLASSES:
            trials = subject.X_by_class[label]
            idx = rng.permutation(len(trials))
            support_idx, query_idx = idx[:k_shot], idx[k_shot:]
            support_X[label] = trials[support_idx]
            query_X.append(trials[query_idx])
            query_y += [label] * len(query_idx)

        prototypes = compute_prototypes(encoder, support_X)
        y_pred = predict_via_prototypes(encoder, prototypes, np.concatenate(query_X, axis=0))
        all_y_true += query_y
        all_y_pred += list(y_pred)

    acc, n_correct, n_total = accuracy_summary(np.array(all_y_true), np.array(all_y_pred))
    print(f"\nHeld-out synthetic-subject nearest-prototype accuracy: {acc:.3f} ({n_correct}/{n_total})")
    print(f"Chance level (2-class): 0.500")

    if acc < 0.75:
        raise AssertionError(
            f"prototypical-network mechanism only reached {acc:.3f} accuracy on held-out synthetic "
            "subjects (expected clearly above chance, ideally >0.75) - the mechanism is not working, "
            "do not proceed to real PhysioNet/IV-2a data"
        )

    print("\nPrototypical-network mechanism validated on synthetic ground truth - "
          "safe to proceed to real PhysioNet/IV-2a data.")


if __name__ == "__main__":
    run()
