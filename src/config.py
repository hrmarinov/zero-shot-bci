"""Shared constants for the cross-session EEG personalization experiments."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"

# MOABB/MNE sanitize ":" out of download destination paths, which corrupts
# Windows absolute paths (e.g. "C:\...") into relative-looking ones and
# causes downloads to land nested under the process cwd instead of DATA_DIR.
# A cwd-relative path with no drive letter/colon sidesteps the bug and keeps
# the MOABB cache inside this repo's data/ directory as intended.
os.environ.setdefault("MNE_DATA", os.path.relpath(DATA_DIR, Path.cwd()))
os.environ.setdefault("MNE_DATASETS_BNCI_PATH", os.path.relpath(DATA_DIR, Path.cwd()))

# BCI Competition IV-2a: 9 subjects, motor imagery, 4 classes.
IV2A_SUBJECTS = list(range(1, 10))
IV2A_CLASSES = ["left_hand", "right_hand", "feet", "tongue"]
IV2A_CHANCE_LEVEL = 1 / len(IV2A_CLASSES)

# BCI Competition IV-2b: 9 subjects, motor imagery, 2 classes, 3 channels.
IV2B_SUBJECTS = list(range(1, 10))

# Band-pass filter used for CSP (standard mu/beta motor-imagery band).
FMIN, FMAX = 8.0, 30.0

# Trial epoch window relative to cue onset (seconds), standard for IV-2a.
TMIN, TMAX = 0.5, 2.5

RANDOM_STATE = 42
