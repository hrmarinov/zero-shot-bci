"""N7c, strength of the population shrinkage, chosen by accuracy on the OTHER subjects. PRE-REGISTERED.

Committed BEFORE any arm is scored. N7b (experiments/n7b_shrinkage_control.py) found that shrink1 (translate the subject's
session-1 trials by lam * (mu1 - m1_s), mu1 = population mean of the session-1 class-mean tangent features) gives R11 +1.09 pp
over base_I on Lee2019 with lam chosen by forecast error (median 0.4), while a ridge with shuffled pairing gave +1.94 pp, which
suggests that the accuracy-optimal pull is stronger. Choosing lam by session-2 accuracy of the SAME 54 subjects would be selection
on test labels; here lam is chosen for each held-out subject s ONLY from the other 53 subjects' session-2 accuracy (leave-one-
subject-out, nested). Same features, protocol, LR (C = 1) and R11 stack as N7; CPU only; Zhou2020 sealed, BNCI2015_001 unused.
N7b's base_I and shrink1 R11 accuracies are recomputed and must equal results/n7b_shrinkage_control_lee2019.csv exactly.

GRID (fixed here, not tuned): lam in {0, 0.2, 0.4, 0.6, 0.8, 1.0}; lam = 1 sets the subject's session-1 class means to the population's.
ARMS:
  fixed_lam  shrink1 with each lam of the grid for every subject (descriptive: the accuracy-vs-lam curve; NOT for selection).
  auto       lam(s) = argmax over the grid of the mean R11 session-2 accuracy of the other 53 subjects o, where each o is shrunk
             toward the population mean of the 52 subjects other than o and s (ties: the smaller lam). Applied to s with the
             population mean of the 53 others, exactly as N7b's shrink1.
PRIMARY ENDPOINT. R11[auto] - R11[shrink1 of N7b], Lee2019 n = 54, per-subject, one-sided sign-flip Monte Carlo (as D2); a single
contrast, no Holm. SECONDARY (reported): R11[auto] - base_I, the D3 transfer-limited subjects, the bottom third by csp_lda, share
>= 70 %, up/down, the distribution of the chosen lam, and the oracle-in-hindsight fixed lam (reported to show the optimism of picking
it on the cohort, never used).
DECISION. auto replaces shrink1 as the candidate only if R11[auto] - R11[shrink1] >= +0.5 pp with one-sided p < 0.05; otherwise
shrink1 (lam by forecast error) stays, and the lane is closed with the finding "strength tuning adds nothing reliable". Either
way the result remains a screen on the development cohort: a Zhou2020 opening is justified only for a candidate gain >= +3 pp
over base_I (R11[auto] - base_I), which is not expected.

    .venv\Scripts\python experiments\n7c_shrinkage_strength.py
"""
