"""Pre-fetch and validate the full PhysioNet reference pool.

Downloads/caches all 108 usable subjects (1-109, excluding subject 88's
different sampling rate) via load_physionet_reference_subject, logging
per-subject progress and trial counts. Runs the whole pool through the same
loader Module A's training script will use, so any subject-specific quirks
beyond subject 88 (missing files, unusual event structure, etc.) surface
here - as a validation pass, not just a bulk download - rather than mid-way
through a training run.

A failure on one subject (after the loader's own retry logic is exhausted)
is logged and skipped, not fatal to the whole run - 108 independent
subjects, no reason one bad subject should lose progress on the other 107.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import PHYSIONET_EXCLUDED_SUBJECTS, load_physionet_reference_subject

ALL_SUBJECTS = [s for s in range(1, 110) if s not in PHYSIONET_EXCLUDED_SUBJECTS]


def run() -> None:
    succeeded = []
    failed = []

    for subject in ALL_SUBJECTS:
        try:
            X = load_physionet_reference_subject(subject)
            print(f"[subject {subject}] OK - {X.shape[0]} trials, shape {X.shape}")
            succeeded.append(subject)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, one bad subject must not kill the batch
            print(f"[subject {subject}] FAILED: {exc!r}")
            failed.append(subject)

    print(f"\nDone. {len(succeeded)}/{len(ALL_SUBJECTS)} subjects succeeded.")
    if failed:
        print(f"Failed subjects: {failed}")


if __name__ == "__main__":
    run()
