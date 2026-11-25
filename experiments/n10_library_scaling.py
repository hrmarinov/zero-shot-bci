"""N10, library-size scaling of the pooled cross-subject decoder. PRE-REGISTERED.

Committed BEFORE any arm is scored. N8 found a zero-shot pooled decoder (53 other Lee2019 subjects, none of the subject's own labels)
at R11 76.83 % against base_I 74.48 %; N9 found no such gain on IV-2a with a library of 8 subjects and 4 classes. Library size,
class count and montage are confounded there. N10 separates library size on Lee2019 itself. CPU only, cached; Zhou2020 sealed.
Same features, leave-one-subject-out protocol and R11 stack as N7 / N8 (src: experiments/n7_drift_forecast_screen.py).

ARMS (R11 with the TS+LR member swapped for a pooled LR, w = 0 i.e. zero-shot, C = 1, max_iter 2000). For each held-out subject s the
library is K other subjects drawn at random (numpy default_rng(1000 * s + draw), without replacement, from the 53 others), both of
their labelled sessions; K in {8, 13, 27, 53}; 3 independent draws per (s, K) for K < 53 (K = 53 is the whole library, one fit);
the subject's accuracy at K is the mean over its draws. base_I is recomputed and must equal results/n7_drift_forecast_lee2019.csv;
the K = 53 arm must equal N8's pool_w0 exactly.
ENDPOINT. gain(K) = R11[pool, K] - R11[base_I] per subject, Lee2019 n = 54; one-sided sign-flip Monte Carlo (as D2) for each K, and
the paired contrast gain(53) - gain(8).
DECISION (the branch the next step follows). SCALES iff gain(53) - gain(8) >= +1.0 pp with one-sided p < 0.05. FLAT otherwise.
  SCALES: IV-2a's null in N9 is attributed to the small library; the next step is a large EXTERNAL library (other datasets'
          left/right-hand trials, if cached or cheaply obtainable) tested on IV-2a / IV-2b as a binary task.
  FLAT:   the cross-subject line is closed as Lee2019-specific and the project moves to the calibration-burden reframe on cached data only.
REPORTED, NOT FOR DECISION: gain(K) mean, up/down, the transfer-limited subjects and the bottom third by csp_lda at each K; the
within-subject spread across draws.

    .venv\Scripts\python experiments\n10_library_scaling.py
"""
