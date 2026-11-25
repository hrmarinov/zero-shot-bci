"""N7, drift forecast: learn how a session-1 decoder drifts into session 2 from the OTHER subjects. PRE-REGISTERED.

Committed BEFORE any arm is scored. Motivation: N0 found no flag that helps, and D3 found that 5 of the 12 weakest Lee2019
subjects are "transfer-limited" (a decoder fitted inside session 2 reaches ~76 %, but session 1 -> 2 reaches 61 %): signal
exists, it moves between days. The 54 subjects each have two labelled sessions, so the day-to-day movement can be learned
across subjects, which no single subject can supply. CPU only, cached data only; Zhou2020 stays sealed, BNCI2015_001 unused.

FEATURES. Per-session Riemannian alignment (src.pipeline.align_by_group, as R11), OAS covariances, tangent space at the
FIXED reference I (both sessions of every subject are aligned to I, so the feature space is common to all subjects), 210-d.
m1 = (mean feature of class 0, mean of class 1) in session 1, m2 the same in session 2; delta = m2 - m1 (2 x 210).

ARMS (TS+LR member of R11 only; CNN3 and FTS2 cached as in D2; LR with C = 1 as R11):
  base    R11's own TS+LR (subject-fitted tangent space), the D2 reference; must reproduce D2 exactly or stop.
  base_I  TS+LR on the fixed-I tangent features, no shift. The PAIRED control for the two arms below.
  const   session-1 trials of class c are translated by the mean delta_c of the 53 other subjects, then LR is fitted.
  ridge   as const, but with a subject-specific forecast: delta_hat = Ridge(m1) - m1 fitted on the 53 other subjects
          (m1 -> m2, 420 -> 420, alpha from {1, 1e1 ... 1e5} by leave-one-subject-out inside those 53, RidgeCV).
Leave-one-subject-out: the held-out subject contributes only its own labelled session 1 and its unlabelled session-2 trials
(aligned on themselves, as R11). Its session-2 labels are used only to score.

PRIMARY ENDPOINT. R11 with its TS+LR member replaced by the arm's, Lee2019 n = 54, per-subject accuracy; contrasts ridge -
base_I and const - base_I, one-sided sign-flip Monte Carlo (as D2), Holm over the 2. SECONDARY (reported): the same on the TS+LR
member alone; ridge - const; vs base; the D3 "transfer_limited" subjects (categories fixed in results/d3_weak_subject_diagnosis.csv),
the bottom third by csp_lda, share >= 70 %, up/down counts. DIAGNOSTIC: leave-one-subject-out forecast skill of delta,
R2 = 1 - sum|delta - delta_ridge|^2 / sum|delta - delta_const|^2 over subjects.
IV-2a / IV-2b: NOT applicable (8-9 subjects, a different montage, nothing to learn the drift from); the replication a pass
needs is therefore a Zhou2020 opening, justified only for an effect >= +3 pp.

KILL RULE. If the forecast skill R2 <= 0, drift is not forecastable and the track ends (ridge is then not scored for adoption).
ADOPT ridge if ALL hold: R11[ridge] - R11[base_I] >= +0.5 pp, Holm p < 0.05, AND ridge - const > 0 with one-sided p < 0.05
(the gain comes from subject-specific drift, not from a population shift). Otherwise the drift-forecast lane is closed.

    .venv\Scripts\python experiments\n7_drift_forecast_screen.py
"""
