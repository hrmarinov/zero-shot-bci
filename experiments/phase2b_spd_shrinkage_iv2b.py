"""Phase 2b generalization check - same SPD shrinkage prototype on IV-2b.

IV-2b has a very different tangent-space dimensionality (3 channels -> 6-dim,
vs. IV-2a's 22 channels -> 253-dim) while still only offering 8 reference
subjects per leave-one-out fold. If the IV-2a null result (see
phase2b_spd_shrinkage.py) is driven by too few reference subjects relative to
tangent-space dimensionality (curse of dimensionality on the between-subject
variance estimate), IV-2b's much lower dimensionality should make shrinkage
noticeably more effective with the same subject count - a direct test of that
hypothesis, not just a "does it also work here" replication.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.phase2b_spd_shrinkage import run
from src.config import IV2B_SUBJECTS
from src.datasets import load_iv2b_subject

if __name__ == "__main__":
    run(
        subjects=IV2B_SUBJECTS,
        loader=load_iv2b_subject,
        sample_sizes=[8, 16, 32, 64, 128, 288],
        out_name="phase2b_spd_shrinkage_iv2b.csv",
        csp_components=3,
    )
