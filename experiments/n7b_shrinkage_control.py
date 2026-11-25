"""N7b, shrinkage control for the N7 drift forecast. PRE-REGISTERED.

Committed BEFORE any control arm is scored. N7 (experiments/n7_drift_forecast_screen.py) found R11[ridge] - R11[base_I] = +1.69 pp
on Lee2019 and ridge - const = +1.15 pp, but registered no control that separates "forecasting a subject's drift" from "pulling
the subject's class means toward a population prior" (or from any 53-subject ridge fit that ignores the subject's own pairing).
Same data, features, leave-one-subject-out protocol, LR (C = 1) and R11 stack as N7; CPU only; Zhou2020 sealed, BNCI2015_001 unused.
N7's ridge, base_I and const numbers are recomputed here and must equal N7's results/n7_drift_forecast_lee2019.csv exactly.

CONTROL ARMS (same translation of session-1 trials by delta_c per class as N7; only delta differs):
  shrink2  delta = (1 - lam) m1_s + lam mu2 - m1_s, mu2 = mean over the 53 other subjects of their session-2 class means:
           pulls the subject toward the population session-2 class means. lam in {0, 0.1, ..., 1} chosen by leave-one-out over
           the 53 others (each other subject's m2 predicted from its own m1 and mu2 of the remaining 52; squared error).
  shrink1  delta = lam (mu1 - m1_s), mu1 = population mean of session-1 class means (pure regularisation toward the population
           with no drift term); lam chosen the same way, on the error of predicting m2 from the shrunk m1.
  ridge_perm  as N7 ridge, but the 53 others' session-2 targets are cyclically shifted by one subject relative to their session-1
           inputs (pairing destroyed; sorted subject order, fixed): a ridge that cannot know subject-specific drift.
CONTRASTS (R11 with the arm's TS+LR member, Lee2019 n = 54, one-sided sign-flip Monte Carlo as D2, Holm over the 3):
  ridge - shrink2, ridge - shrink1, ridge - ridge_perm.
DECISION. The N7 gain is attributed to SUBJECT-SPECIFIC DRIFT FORECASTING only if ridge beats ALL THREE controls by >= +0.5 pp with
Holm p < 0.05. If not: the mechanism is a population prior / generic shrinkage, and the candidate carried forward is the SIMPLEST
control arm whose gain over base_I is >= +0.5 pp with one-sided p < 0.05 (ties on significance are broken by fewer parameters:
shrink1 < shrink2 < ridge_perm), because a one-parameter shrinkage deploys with no subject library beyond two population means.
REPORTED, NOT FOR DECISION: every arm's mean R11, gain over base_I, up/down, D3 transfer-limited and bottom-third means.

    .venv\Scripts\python experiments\n7b_shrinkage_control.py
"""
