# Benefit against complexity: review of the work so far, and the plan that follows from it

**Date:** 2026-11-24. **Rule applied** (README, Method rules, item 7): before a test, write down the expected gain
(shrunk for winner's curse) and the cost (GPU time per user, code, extra models). Skip tests with a poor ratio.
Sources: `docs/diag_gates_d1_d4.md`, `docs/project_audit.md`, `TODO.md`, `reports/t1_council.md`, `results/`.
Gains below are accuracy on IV-2a unless stated. "Estimated" means my judgment, not a measurement.

## 1. Where the accuracy came from

| Step | Result | Change | Cost |
|---|---|---|---|
| CSP+LDA, no adaptation (Phase 1) | 62.2 % | | tiny |
| + per-session Riemannian alignment (Phase 2) | 66.9 % | +4.7 | tiny, CPU |
| ShallowConvNet on aligned input (R3) | 74.7 % | +7.8 | 1 GPU fit, seconds |
| + CNN seeds and TS+LR stack (R10) | 79.0 % | +4.3 | 4 fits |
| + FTSurrogate members (R11) | 82.5 % | +3.5 | 5 fits, 0.5 min per user |

**Three ideas gave about 90 % of the gain:** alignment, a CNN on aligned input, and a diverse ensemble. All three are
cheap and use only the target session. Everything after Phase 2 that did not build on them returned a null or a loss.

## 2. Large speculative chases, scored

| Chase | Cost | Result | Verdict |
|---|---|---|---|
| Wavelet personalisation (Phase 4) | days | 31.4 % (below chance-plus-baseline) | wasted |
| Prototypical-network meta-learning (Phase 6) | hours per run, PhysioNet 108 subjects | 49.0 % | wasted; the "same-subject episodes" fix was clever but ended below CSP+LDA+RA |
| SPD shrinkage, eigenvoice, Module A and its bounded variant | weeks of design, 3 datasets | null or 3-15× worse | wasted; the failure was predictable from the first null |
| ICP, CPD, ComBat, flow fields, state-space flow, latency alignment | one script each | null | wasted, each small |
| Drift trackers and self-training (D3, R5-R7) | about 5 rounds | closed or not confirmed | wasted |
| 14 rounds of ensemble screens on the same 18 sessions | many GPU hours | real gain at R10/R11, then 2 rounds with none | useful, then overfit: about 60 configurations on the same test sessions |
| Cross-paradigm X1 (35 GB download) | days | SSVEP supports the recipe; P300 unusable | off the goal |
| A1 CTNet / TCFormer as ensemble member | days | not confirmed on the lockbox | wasted |
| T1 (36 TCFormer fits, 16 GPU hours) | 1.5 days | inconclusive on the bar; found the ocular null on IV-2a and the complementarity lead | partly useful |
| T2 Lee2019 blend (stopped, 21 of 54 fits) | 1.5 hours | +1.5 pp for 8-28× the training time | stopped correctly |

**Pattern.** Ideas were chosen for novelty or for "what does the literature say is the bar", not for expected gain
per unit of cost. Nulls were reported honestly, which is why the record is clean, but the effort went to the
speculative lane while small items sat on the list with no status.

## 3. Small, cheap fixes that were missed

These have been on `TODO.md` since the early phases (cost marked S) or appeared in this session. None needs a new
dataset or a new architecture. Most run on cached predictions or features in CPU time.

| # | Fix | Why it is cheap | Evidence it is worth a test | Estimated gain |
|---|---|---|---|---|
| S1 | **Class-balance correction.** Rescale test probabilities so that predicted class counts match the known equal counts | about 10 lines, CPU, zero per-user cost | **Measured post hoc this session** on R11 probabilities: Lee2019 +0.70 pp (25 up / 11 down, one-sided p = 0.012), IV-2a +0.27 pp (4 up / 2 down). Subject 2 of IV-2a **got worse** (-3.1 pp): its R11 error is not only a count bias | +0.3 to +0.7 |
| S2 | **Ensemble averaging rule.** Log-mean, rank-mean or per-member temperature scaling before the average (CNN and TS+LR are not equally confident) | CPU on cached member probabilities | never tested; members differ in calibration | 0 to +1 |
| S3 | **Tune TS+LR.** Inner-CV for C and for the covariance estimator (OAS, Tyler, Huber; pyRiemann 0.12 has them) | CPU, library calls | C = 1 was fixed once and never tuned (A10) | 0 to +1 |
| S4 | **Per-member weights from out-of-fold training predictions**, shrunk to equal (A16) | CPU on cached members | audit: equal weights cost 0.5-1 pp where TS+LR is weak | +0.5 to +1 |
| S5 | **Alignment-estimator variants** (A12): Euclidean against Riemannian mean, pooled reference | CPU, one flag | listed since the first phases, cost S | 0 to +1 |
| S6 | **CNN training recipe.** The T1 lesson is that the recipe, not the architecture, separates TCFormer from our trainer (A1 74.5 % against T1 84.4 %). Try on the ShallowConvNet members: cosine schedule with warm-up, interaug, Mixup, per-channel z-score | small GPU screen with `fast=True`; per-user time must not grow beyond about 1 min | CNN was never tuned (audit, "never tuned the CNN itself") | 0 to +2, estimated |
| S7 | **FBCSP baseline** (A9) | MOABB ships it | a missing standard baseline; validity, not accuracy | 0 |
| S8 | **Store class order in every prediction cache** | a metadata field | T1 cache order [feet, left, right, tongue] against ours [left, right, feet, tongue] cost the reviewers a trap | validity |
| S9 | **Within-subject shrinkage** (A8) | S cost | the untested sibling of a proven null | 0 to +1 |

Stack S1-S5 into **one** pre-registered hypothesis, as the alien-space reviewer advised: each effect is below the 1.9 pp
detectable limit of the 54-subject cohort, but the stack may not be.

## 4. What stays closed, and what to stop

- Closed for good (Part D of `TODO.md`): every cross-subject covariance comparison, naive feature concatenation, PCA on
  windowed features, ICP, CPD, ComBat, foundation-model fine-tuning for motor imagery.
- Do not add TCFormer to R11 (T2). Keep the IV-2a blend lead as an observation only.
- No further large downloads or new datasets unless the test cannot be run without them.
- No more than **two arms per round**, one size-matched control, and a rule written before the run (already the project
  habit; keep it).
- Screen on Lee2019 (54 subjects). The lockbox on BNCI2015_001 is spent. Zhou2020 is opened once, and only for a stacked
  claim of at least +3 pp.

## 5. The plan, in order

Each step states the gain and the cost, and what it decides.

1. **Pre-register the stack S1-S5 (0 h).** One commit with the rules before any run. Primary: pooled gain at least +1.0 pp
   over R11 on Lee2019 with Holm p < 0.05; secondary: IV-2a and IV-2b not worse. Decides: whether the cheap stack is real.
2. **Run S1-S5 on cached predictions (about 1 day, CPU, no new fits).** Cost per user stays at zero. Decides: adopt, or
   close the cheap lane.
3. **S6 recipe pieces on the CNN members (about 1 GPU-day).** Adopt only a piece that adds at least +1 pp and no more than
   1 min of training per user. Decides whether the TCFormer advantage is recoverable for free.
4. **S7, S8 (a few hours).** Baseline and cache hygiene.
5. **One Zhou2020 opening**, only if steps 2-3 together reach +3 pp.
6. **Preprint track in parallel, capped.** The ocular-confound paper is the goal that is not accuracy. Its missing pieces
   (TCFormer on IV-2b, a shuffled-EOG control, a training against feedback comparison, an equivalence bound for IV-2a)
   are worth their cost because the paper is the output. Budget: about 8 GPU hours in total.
7. **Live and burden claims.** They are done and confirmed. The only open item is a closed-loop session, which needs
   hardware; do not plan it.

**Gate for every new idea.** Write: expected gain (shrunk), cost (GPU minutes per user, lines of code, extra models), the
detectable effect at the dataset size, and a kill rule. If gain divided by cost is poor, add it to the closed list
instead of testing it.

## 6. Update: the cheap CPU stack was run (S1 screen, 2026-11-24)

Result: **none of S1, S2, S3a, S3b, S5 was adopted** (`docs/diag_gates_d1_d4.md`, section "S1"). Class balancing was the nearest
(+0.70 pp, Holm p = 0.060). The cheap lane on cached members is closed. Remaining plan items: S6 (CNN training recipe, needs
fits, at most +1 min per user), S4 (needs out-of-fold fits), the preprint track, and S7/S8 (hygiene). The ratio rule now points
at S6 first, and only if a pre-registered one-day screen on Lee2019 with `fast=True` can show a gain of at least +1 pp.

## 7. Update: S6 recipe screen run (2026-11-25)

Result: **neither interaug (+0.26 pp) nor Mixup (+0.31 pp) was adopted** (Holm p = 0.29; `docs/diag_gates_d1_d4.md`, section "S6").
The recipe lane is closed. Remaining plan items: S4 (member weights, needs out-of-fold fits; expected gain small), the
preprint track, S7/S8 hygiene. No accuracy lever with a good ratio is left on the list.
