"""N9, equal-size control and IV-2a replication for the N8 four-member ensemble. PRE-REGISTERED.

Committed BEFORE any arm of this script is scored. N8 (experiments/n8_pooled_decoder_screen.py) found ens4 = (CNN3 + TS+LR[N7c] +
FTS2 + pool[auto_w]) / 4 at R11 +2.94 pp over base_I on Lee2019 (+1.04 pp over N7c auto, Holm p 0.021), but registered no
equal-size control (the gain may be member count / diversity in general) and no replication. CPU only, cached members only;
Zhou2020 sealed, BNCI2015_001 unused. Same features, leave-one-subject-out protocol and R11 stack as N7 / N7c / N8.

PART A, Lee2019 n = 54 (inference). ENS4_CTRL = (CNN3 + TS+LR[N7c auto] + FTS2 + CNN45) / 4, CNN45 = mean softmax of the cached base
CNN seeds 3-4 (a fourth member that is subject-specific and equally large). lam (N7c) and w (N8) are chosen exactly as there; ens4 and
n7c_auto must reproduce results/n8_pooled_decoder_lee2019.csv exactly or the script stops.
  PRIMARY: ens4 - ens4_ctrl, one-sided sign-flip Monte Carlo (as D2), a single contrast. The pooled member's gain is credited to
  CROSS-SUBJECT INFORMATION only if the mean is >= +0.5 pp with p < 0.05; otherwise to member count / diversity in general.
  REPORTED: ens4_ctrl - n7c_auto (the effect of a fourth member of the old kind), ens4 - base_I, transfer-limited and bottom third.
PART B, IV-2a n = 9 (descriptive replication, NO inference: 9 subjects, 4 classes, 22 channels). Library = the 8 other subjects (both
labelled sessions, 576 trials each); features as in N7 (per-session alignment, OAS, tangent space at I, 253-d); lam grid {0, .2, .4,
.6, .8, 1} chosen on the 8 others (nested), w grid {0, 1, 5, 20, 100} chosen on the 7 other validators (each o scored with the pool of
the subjects other than o and s); arms base_I, n7c_auto, pool[auto_w], ens4, ens4_ctrl with CNN45 = cached r3_preds cnn_ra seeds 3-4;
R11 base must reproduce the stored IV-2a R11 accuracy exactly. IV-2b is not tested (a different recording design and 3 channels).
  REPLICATION CONSISTENT iff ens4 - base_I mean >= +0.5 pp AND ens4 > base_I for >= 7 of the 9 subjects AND ens4 - ens4_ctrl mean > 0.
DECISION. ens4 stays the candidate (and is worth packaging as a pipeline option) only if Part A credits cross-subject information and
Part B is consistent. If A fails, the N8 gain is generic member count and ens4 is dropped; if only B fails, the result is Lee2019-
specific and stays a development-cohort screen. A Zhou2020 opening still needs a candidate gain >= +3 pp over base_I.

    .venv\Scripts\python experiments\n9_ens4_controls.py
"""
