"""Experiment scripts, one per research phase.

An explicit package marker so cross-script imports resolve deterministically.
`phase2b_spd_shrinkage_iv2b.py` imports `run` from `phase2b_spd_shrinkage`,
which otherwise relies on PEP 420 implicit namespace packages and only works
when the repository root happens to be on `sys.path`.

Scripts are still meant to be *run* directly (each inserts the repository root
into `sys.path` itself and writes a CSV into `results/`), not imported as a
library. The reusable code lives in `src/`.
"""
