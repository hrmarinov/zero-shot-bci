"""Pytest configuration.

Makes the repository root importable so tests can `import src.*` regardless of
the working directory pytest is invoked from. Every experiment script does the
equivalent `sys.path.insert` itself; this is the test-side counterpart.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
