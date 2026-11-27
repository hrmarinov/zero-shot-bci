"""Known-before candidate (N8-like): R11 plus a pooled fourth member with own weight 20. Never edited by the loop."""

import candidate

_REF = {"w_pool": 1.0, "pool_w": 20.0}


def predict(own: dict, test: dict, library: list):
    saved, candidate.CFG = candidate.CFG, _REF
    try:
        return candidate.predict(own, test, library)
    finally:
        candidate.CFG = saved
