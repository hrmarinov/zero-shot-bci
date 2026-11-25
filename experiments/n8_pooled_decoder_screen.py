"""N8, pooled cross-subject decoder as the TS+LR member of R11. PRE-REGISTERED.

Committed BEFORE any arm is scored. N7c found that the accuracy-optimal pull of a subject's session-1 class means toward the
population is 0.8-1.0 (R11 +1.91 pp over base_I), i.e. the TS+LR member under-uses cross-subject information. N8 tests the direct
version: a logistic regression trained on the other subjects' labelled trials. Same features (per-session Riemannian alignment,
OAS covariance, tangent space at the fixed reference I, 210-d), leave-one-subject-out protocol and R11 stack as N7/N7c; CPU only;
Zhou2020 sealed, BNCI2015_001 unused. The held-out subject's session-2 labels are used only to score; the other 53 subjects
contribute both labelled sessions (session 1 and 2) as training trials.
N7's base_I and N7c's auto accuracies are recomputed here and must equal results/n7_drift_forecast_lee2019.csv and
results/n7c_shrinkage_strength_lee2019.csv exactly.

ARMS (TS+LR member of R11 replaced; CNN3 and FTS2 cached as in D2; LR with C = 1, max_iter 2000):
  pool_w  LR fitted on the 53 other subjects' trials (weight 1 each) plus the subject's own labelled session-1 trials with
          sample weight w, w in {0, 1, 5, 20, 100} (w = 0 is zero-shot: no data of the subject's own enters). Fixed grid, not tuned.
  auto_w  w(s) = argmax over the grid of the mean R11 session-2 accuracy over 10 FIXED validator subjects (the first 10 of
          numpy default_rng(0).permutation(1..54) that are not s), each validator o scored with a pool of the 52 subjects other
          than o and s plus o's own session-1 trials at weight w (ties: the smaller w). Applied to s with the pool of the 53 others.
  ens4    (CNN3 + TS+LR[N7c auto] + FTS2 + pool[auto_w]) / 4: the pooled decoder as an additional member next to N7c's candidate.
  The comparison arm n7c_auto is N7c's candidate (lam chosen on the other subjects) recomputed here.
PRIMARY ENDPOINT. R11 session-2 accuracy, Lee2019 n = 54, per subject. Contrasts against n7c_auto: pool[auto_w] swapped in for TS+LR,
and ens4; one-sided sign-flip Monte Carlo (as D2), Holm over the 2.
SECONDARY (reported): the contrasts against base_I; every fixed-w arm (descriptive, not for selection); the chosen-w histogram; the
D3 transfer-limited subjects, the bottom third by csp_lda, share >= 70 %, up/down.
DECISION. A pooled arm replaces n7c_auto as the candidate only if its gain over n7c_auto is >= +0.5 pp with Holm p < 0.05. If neither
passes, the cross-subject lane is closed at N7c's candidate with the finding "the pooled decoder adds nothing over the population
class-mean prior". Either way a screen on the development cohort: a Zhou2020 opening is justified only for a candidate gain >= +3 pp
over base_I, and IV-2a / IV-2b cannot replicate it (8-9 subjects, a different montage).

    .venv\Scripts\python experiments\n8_pooled_decoder_screen.py
"""
