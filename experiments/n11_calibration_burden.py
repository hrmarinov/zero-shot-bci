"""N11, calibration burden with a population prior: how few own trials does the TS+LR member need? PRE-REGISTERED.

Committed BEFORE any arm is scored. N10 showed that a zero-shot pooled decoder (no own labels) beats R11's own TS+LR member on
Lee2019 even with 8 library subjects, and that the gain does not grow with library size, which suggests the prior helps when the
subject's own data is scarce (100 trials per session here). N11 measures the TS+LR MEMBER ALONE (not R11: the cached CNN members
are fitted on all 100 trials and cannot be subsampled without new fits) on session 2 of Lee2019, n = 54, leave-one-subject-out,
features as N7 (per-session alignment, OAS, tangent space at I; the held-out session-2 trials are aligned on themselves as always).

ARMS at n_own in {5, 10, 20, 40, 70, 100} own labelled session-1 trials (stratified random subsample of the subject's session 1,
3 draws per (subject, n_own), seeds 1000 * s + draw; n_own = 100 is the full session and uses one draw):
  own    LR (C = 1) on the subject's n_own trials only (needs two classes in the draw; guaranteed by stratification).
  pool0  zero-shot: LR on the 53 other subjects' trials (both sessions), constant in n_own.
  poolw  the pool plus the n_own own trials at sample weight w = 20 (fixed here, not tuned; the middle of N8's grid).
Accuracy per subject = mean over draws. own at n_own = 100 must equal results/n7_drift_forecast_lee2019.csv column ts_base_I.
ENDPOINTS. For each n_own: mean accuracy of the three arms; poolw - own and pool0 - own with one-sided sign-flip p (as D2); Holm over
the 5 sample sizes below 100 for poolw - own. DERIVED: n_own* = smallest n_own at which own's mean accuracy reaches pool0's mean
accuracy (the burden a subject-only decoder needs to match a calibration-free one); n_pool = smallest n_own at which poolw's mean
accuracy reaches own's accuracy at n_own = 100 minus 0.5 pp, and the burden reduction 100 / n_pool.
DECISION. CALIBRATION-FREE VIABLE iff pool0 >= own(100) - 0.5 pp. BURDEN CLAIM iff n_pool <= 50 (a reduction of at least 2x) AND poolw
- own at n_own = n_pool has Holm p < 0.05. If the claim holds, README "Next" and TODO get a short note (the plan changes: the
deliverable becomes calibration burden). Otherwise the line is closed and nothing is added to the plan. TS+LR member only; a claim
about the full R11 pipeline would need new CNN fits and is not made here. IV-2a is not tested (N9: no library effect there).

    .venv\Scripts\python experiments\n11_calibration_burden.py
"""
