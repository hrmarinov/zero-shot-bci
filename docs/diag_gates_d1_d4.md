# Gating diagnostics D1–D4 — which cross-field method families have headroom

Source: [`experiments/diag_gates_d1_d4.py`](../experiments/diag_gates_d1_d4.py) →
[`results/diag_gates_iv2a.csv`](../results/diag_gates_iv2a.csv). Plan:
[`reports/Cross field BCI calibration methods.md`](../reports/Cross%20field%20BCI%20calibration%20methods.md).
Gate thresholds were fixed in the script's docstring before the first full run. Evaluation
labels are used **only** to compute upper bounds; nothing here is a deployable method.

```powershell
.venv\Scripts\python experiments\diag_gates_d1_d4.py     # ~20 min, exit 0
```

**Validity checks (all passed):** CSP+LDA+RA reproduces Phase 2 per subject bit-exactly;
`cnn_ra` seed 0 reproduces R3 per subject bit-exactly; own-LDA agrees with the pipeline on
100% of trials; planted-rotation control on subjects 1/3/7: a random rotation of E drops the
baseline by 41.7/54.9/50.0 pp and the oracle recovers to within ±0.35 pp of the unplanted
oracle; run-wise re-centering changes 8.3% of predictions (not a no-op).

**Tool fix found on the way:** `pyriemann`'s private rotation solver stalls on 22-channel
planted rotations (max residual 0.13–0.23 even with restarts), which would have understated
the D1 bound. Replaced by `src/adaptation/rotation_oracle.py` (batched Adam over
expm-parametrised rotations, both determinant signs, L-BFGS polish): planted 22-channel
rotations recovered to ~1e-6; unit-tested.

## Verdicts

| Gate | Letter of the pre-declared rule | Read with the cross-fitted bound | Action |
|---|---|---|---|
| **D1 rotation** | **CLOSED** | closed | drop pseudo-label RPA/TSA, EM-CMLLR, Wasserstein Procrustes, anchor TSA |
| **D2 class-conditional** | OPEN | **effectively closed** | deprioritise MAP class means / adaptive MDM / prototypes |
| **D3 online tracking** | OPEN | **closed for statistic-tracking; a real within-session decline exists** | drop re-centering trackers; the decline is a new, different problem |
| **D4a self-training** | **OPEN** | open | self-training, distillation, entropy/IM TTA are viable |
| **D4b frequency warping** | **CLOSED** | closed | drop VTLN / peak-anchored warping |

### D1 — even an oracle rotation does not help

Mean gain of the best class-mean-matching rotation over the aligned baseline:

| Model | Baseline | In-sample oracle | 2-fold oracle |
|---|---|---|---|
| CSP+LDA | 66.94% | −0.96 pp (3/9 up) | −2.74 pp |
| MDM | 64.31% | +0.12 pp (7/9 up) | −1.85 pp |
| TS+LR | 69.25% | +0.27 pp (6/9 up) | −3.05 pp |
| ShallowConvNet (`cnn_ra`, seed 0) | 75.31% | −2.35 pp (1/9 up) | −4.44 pp |

With true labels, the rotation that best matches the two sessions' class-mean covariances
gains nothing and usually *hurts*. After per-session whitening, matching class means is not
the right objective; the residual between sessions is not a rotation of the class geometry.
Scope of the claim: this closes every method whose rotation is fitted by matching class means
or anchors (that is the whole family listed). It does not bound an arbitrary rotation chosen
to maximise accuracy, which no label-free method can target.

### D2 — open by the letter, closed in substance

| Model | T means | Label-free PMean | Oracle E means, in-sample | Oracle, 2-fold |
|---|---|---|---|---|
| CSP+LDA | 66.94% | 67.79% | 70.10% (+2.31 over PMean) | 67.40% (−0.39 vs PMean) |
| MDM | 64.31% | — | 77.35% (+13.04) | 65.62% (+1.31) |

The pre-declared gate used the in-sample bound and opens. That gate was **too generous for
D2**, a design error I am recording rather than re-deciding silently: re-estimating class means
on the very trials being scored is close to within-session training (MDM in-sample 77.4% vs
2-fold 65.6%). Cross-fitted, a class-conditional correction with *perfect* labels for half of E
does not beat the label-free pooled-mean shift on CSP+LDA, and gives MDM +1.3 pp — below the
+2 pp a pseudo-label version would need. Lesson for future gates: use the cross-fitted oracle
whenever the oracle re-estimates a quantity from the scored trials.

By-product (exploratory, not a result): label-free pooled-mean shift (PMean) on CSP+LDA
+0.85 pp, 6/9 up — below the +1 pp screening kill.

### D3 — accuracy decays within the session, but not through a trackable statistic

Mean accuracy per evaluation run (runs are chronological, 48 trials each):

| Run | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| CSP+LDA+RA | 69.7 | 69.7 | 69.0 | 66.4 | 64.8 | **62.0** |
| ShallowConvNet+RA+AdaBN (5 seeds) | 78.5 | 76.2 | 74.0 | 76.0 | 74.0 | **69.6** |

Late minus early runs: CSP −5.0 pp (8/9 subjects worse, sign-flip p = 0.062), convnet −3.0 pp
(p = 0.17). The gate opens on the CSP decline. But the two direct tests of *tracking* say no:
the per-run mean covariance shows no trend over the session (median Spearman ρ = −0.03, 4/9
positive), and re-centering each run on its own statistics is −0.62 pp (1/9 up). So the decline
is real but is not a drift in the second-order statistics that alignment trackers follow.
Candidate explanations (untested): fatigue or disengagement weakening the task modulation
itself, or within-session electrode changes that alter signal-to-noise rather than covariance
geometry. **Recentering/Kalman/EMS trackers: closed. The within-session decline: logged as a new
open question** — it is ~7–9 pp from first to last run, larger than any method gain still on the
table.

### D4 — pseudo-labels are reliable enough; there is no peak shift to warp

Accuracy of the 5-seed ensemble's most confident E trials:

| Top-confidence fraction | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| All subjects | 93.2% | **89.8%** | 85.1% | 78.3% |
| WEAK {2, 5, 6} | 86.1% | **78.0%** | 71.3% | 63.2% |

Gate (≥ 80% all, ≥ 70% WEAK at top-50%) opens: confidence is well calibrated, and even the weak
subjects' confident half is ~78% correct (s2 75.7, s5 80.6, s6 77.8). Self-training, ensemble
distillation and entropy/information-maximisation TTA are viable — with confidence thresholds.

Mu-peak shift between sessions: 0.00–0.24 Hz for every subject (R1-B estimates, Welch bin
≈ 0.12 Hz, i.e. ≤ 2 bins) — no frequency displacement for VTLN-style warping to correct.

Class structure (ensemble; Rasch-style GLM log-odds vs left hand): right hand −0.19, feet −0.51,
tongue −0.48. Feet and tongue are the hard classes for every group; no subject-specific
class-pair failure was found that a targeted fix could exploit.

## What this leaves on the screening list

From the ranked catalogue, after gating: **(1) augmentation priors, (2) run-wise epoch
selection + SWA, (3) label smoothing + entropy/IM TTA, (7) self-training / label spreading,
(10) ensemble distillation on unlabelled E** — plus the non-gated linear-pipeline screens
(4 per-session score standardisation, 9 CORAL, 11 TLScale, 13 RUV). Dropped by gates:
rotation family (8, 15-adjacent), class-conditional family (6), frequency warping (12),
statistic trackers. New open question: the within-session accuracy decline.

## D5 — why accuracy declines within a session (follow-up, `experiments/diag_d5_within_session_decline.py`)

Estimator identity: session-level ERD reproduces R1-B's CSV for all 9 subjects (atol 1e-9).
Rules fixed in the script docstring; `posthoc_` measures added after the first run (exploratory).

| Finding | Numbers (first→last run, 9 subjects) |
|---|---|
| Later runs are harder in **both** sessions, **both** directions | T→E −7.9 pp, E→T −7.2 pp (each p = 0.027). E→T rules out "distance from the calibration day": later T runs are *closer* to E and still worse → time on task |
| Same-session (leave-one-run-out) decodability | E −4.5 pp (p = 0.020, 8/9); T −1.5 pp (n.s.) |
| Pre-cue alpha rises (vigilance-decrement marker) | T 8/9 (p = 0.039), E 7/9 (p = 0.086); **best tracker of per-run accuracy, within-subject r = −0.44, p = 0.001** |
| Pre-declared fatigue index (θ+α)/β | not significant → H2 **not supported by the letter**; the alpha evidence is exploratory |
| Broadband power rises | T 9/9 (p = 0.004), E 7/9 (p = 0.066); theta and beta alone do not → largely the alpha rise |
| ERD magnitude | no trend (mu, beta) |
| Directional covariance drift (posthoc; distance from run 1, trend over runs 2–6 so the zero self-distance cannot bias it) | T 8/9 (p = 0.008), E 7/9 (p = 0.047); tracks accuracy r = −0.34 |

**Correction to D3 (my design error):** D3 measured per-run distance from the *session mean*,
which is U-shaped under a linear drift and therefore cannot detect one; its "no covariance
trend" conclusion is withdrawn. There **is** a slow, monotone, label-free-measurable drift
(rising alpha/broadband power). D3's only valid evidence against trackers is run-wise
re-centering (−0.62 pp), which uses noisy 48-trial estimates. **The online-tracking family is
re-opened as "uncertain"**: a smooth causal tracker (EMA/Kalman re-centering, or alpha-power
normalisation) is now a well-motivated screen, not a closed one.

Working interpretation (exploratory): a vigilance decrement — rising pre-cue alpha over time on
task, with later runs harder to decode in both sessions — not electrode/artefact degradation
(no rise in outliers, kurtosis, EMG band).

## R5 — smooth drift trackers screened and closed (`experiments/r5_drift_tracking.py`)

D5 re-opened the tracking family, so it was screened: trial *t* is whitened by a reference
built near *t* instead of one per session (`src/adaptation/drift_tracking.py`), applied to T and E
alike, per session, label-free. Arms and gates were fixed in the script docstring before the run;
one setting each (σ = 48 trials, half-life 48 trials), no tuning. CNN = R3 `cnn_ra` config, 5 seeds.

```powershell
.venv\Scripts\python experiments\r5_drift_tracking.py     # ~70 min, exit 0
```

**Validity (all passed):** static CSP reproduces Phase 2 and static CNN seed 0 reproduces R3
`cnn_ra` per subject bit-exactly; trials are in chronological run order; **planted drift** on
subjects 1/3/7 (E mixed by expm((t/287)·S), max |eig S| = 1) drops static CSP by 31.9/16.3/19.4 pp
and the kernel tracker recovers 27.8/16.3/19.1 pp. The tracker works when a covariance drift is
the cause.

| Arm (vs static) | CSP+LDA | ShallowConvNet | late − early runs (CSP / CNN) |
|---|---|---|---|
| static (session RA) | 66.94% | 74.64% | −5.0 / −2.6 pp |
| kernel, non-causal | −0.66 pp (3/9 up, p = 0.29) | +0.21 pp (5/9, p = 0.41) | −4.2 / −2.2 pp |
| causal_cum (running mean) | −0.89 pp (1/9, p = 0.047) | −0.92 pp (0/9, p = 0.004) | −3.6 / −2.5 pp |
| causal_ema (half-life 48) | −0.73 pp (3/9, p = 0.13) | −0.75 pp (2/9, p = 0.043) | −3.1 / −1.9 pp |

**Verdict: G1 closed on both models, so the tracking family is CLOSED** (with a valid positive
control this time, unlike D3). G2 (forgetting vs no forgetting) +0.15 / +0.17 pp, closed. Causality
costs ~0.8 pp against the transductive session mean.

Reading: the drift D5 measured is real but is not what costs accuracy. Following it removes a
planted drift of similar or larger size, yet barely flattens the real decline (CSP −5.0 → −4.2 pp).
The decline therefore sits in the task-related signal itself (consistent with D5's vigilance
reading and the drop in same-session decodability), not in a second-order shift a reference can
undo. Remaining levers for it are on the training side (time-on-task-aware augmentation,
run-weighted training) or self-training on E (D4a open), not on re-centering.

## R6 — self-training on the unlabelled session: screen OPEN (`experiments/r6_self_training.py`)

D4a left this family open, so three label-free uses of E were screened on ShallowConvNet
(`src/adaptation/self_training.py`). Teacher = R3 `cnn_ra`, 5 seeds. Arms, gates and controls
were fixed in the script docstring before the run, one setting each:
**st** retrains on T + the confident 50% of E (per predicted class) with hard ensemble
pseudo-labels; **distill** retrains on T + all E with the ensemble's soft probabilities;
**im** tunes only BN affine parameters by information maximisation (SHOT-IM / Tent).
The primary contrast is **5-seed ensemble vs the teacher's 5-seed ensemble**, so ensembling
alone does not count as a gain.

```powershell
.venv\Scripts\python experiments\r6_self_training.py     # ~95 min, exit 0
```

**Validity (all passed):** every retrained teacher reproduces the cached R3 `cnn_ra` predictions
(45/45 subject-seeds). Shuffled-pseudo-label control (subjects 1/3/7): −46.9/−51.0/−49.7 pp vs
the teacher, i.e. the student follows its pseudo-labels and no path from true E labels exists.

| Arm | Ensemble | vs teacher ensemble (77.58%) | WEAK {2,5,6} | Unconfident half of E | Single-model mean |
|---|---|---|---|---|---|
| teacher | 77.58% | — | — | 65.4% | 74.64% |
| **st** | **79.21%** | **+1.62 pp, 7/9 up, p = 0.047** | +1.27 pp | **+3.26 pp** | **78.12%** |
| distill | 77.85% | +0.27 pp, 5/9, p = 0.13 | +0.58 pp | +0.54 pp | 77.70% |
| im | 78.20% | +0.62 pp, 4/9, p = 0.37 | +0.46 pp | +1.17 pp | 74.93% |

**Verdict:** **st OPEN** by the screening gate; distill and im closed. ADOPT not reached: Holm
one-sided p_adj = 0.070 for st (needs < 0.05), so this is a screen pass, not a result.

Reading: the gain comes from where it must, the teacher's *unconfident* half (+3.3 pp); the
confident half is ~90% right already. A single self-trained model (78.1%) beats the 5-seed teacher
ensemble (77.6%) and the R3 single-model headline (74.7%) by +3.4 pp. Distillation is a clean
by-product: one distilled model matches the ensemble (77.7% vs 77.6%) at 1/5 of the inference cost.
IM-TTA adds little on top of RA, as AdaBN did.

Caveats: transductive (E is used unlabelled before it is scored, like session-wise RA itself);
one round, one confidence fraction; IV-2a only. Next is confirmation, not tuning: a fresh
pre-registered run on another dataset (IV-2b / Stieger2021), with the setting frozen as is.

## R7 — confirmation on IV-2b: NOT CONFIRMED (`experiments/r7_self_training_iv2b.py`)

The R6 setting was frozen and the protocol pre-registered in the script docstring and committed
(`5434be7`) before the run. Data: BNCI2014_004, 9 subjects, 3 channels, left vs right hand;
train on recordings 0train–2train, self-train on each of 3test/4test separately; Riemannian
alignment per recording. One hypothesis, no multiplicity.

```powershell
.venv\Scripts\python experiments\r7_self_training_iv2b.py     # ~80 min, exit 0
```

**Validity (all passed):** teacher ensemble 79.1% (≥ 60%); shuffled pseudo-labels drop
36.2/5.6/23.1 pp on subjects 1/3/7; teacher training deterministic; every alignment self-check passed.

| | Teacher ensemble | Self-trained ensemble | Difference |
|---|---|---|---|
| IV-2b (mean of 3test/4test) | 79.07% | 79.29% | **+0.22 pp, 95% CI [−0.72, +1.16], 3/9 up, one-sided p = 0.30** |

Per subject (pp): s1 +1.9, s2 −1.1, s3 +1.9, s4 0.0, s5 −0.6, s6 +1.6, s7 −0.6, s8 −0.9, s9 0.0.
Single model: 78.62% → 79.09% (+0.47 pp).

**Verdict: H1 NOT CONFIRMED.** Pseudo-label self-training is **not adopted**. The IV-2a gain
(+1.62 pp) may be specific to that dataset or a screening false positive (Holm 0.070 already
signalled that). It stays in the record as "IV-2a screen pass, failed external replication".

Possible reasons (post-hoc, untested, and NOT grounds for a modified re-run on this data):
2 classes leave less room on the unconfident half; three subjects are near ceiling (s4/s5/s8
≥ 89%) and two near chance (s2/s3, 50–56%), where pseudo-labels carry little information; 3
channels give far less spatial information to re-learn. Any follow-up would need a new
pre-registration on a third dataset.

## R8 — augmentation screen: FTSurrogate PROMOTE (`experiments/r8_augmentation.py`)

Pre-registered in the script docstring and committed (`a156220`) before the run. Settings were fixed
a priori from Rommel et al. 2022 (`src/adaptation/augmentation.py`), never tuned on E. Each
augmentation was applied alone with p = 0.5 per trial, to ShallowConvNet on per-session RA input,
with seeds 0–2. The comparator is R3 `cnn_ra` with seeds 0–2 (seed-matched).

```powershell
.venv\Scripts\python experiments\r8_augmentation.py     # ~75 min, exit 0
```

**Validity:**
- V1: the seed-0 baseline retrained in this run reproduced R3 `cnn_ra` seed 0 exactly for every subject (asserted).
- V2: the augmentation unit tests pass.
- With `augment=None`, `train_model` is bit-identical to before.

Baseline: single model 75.21%, 3-seed ensemble 78.09%.

| Arm | Single (3-seed mean) | Δ vs baseline | Up | p (2-sided) | Subjects losing > 2 pp | WEAK {2,5,6} | Ensemble Δ | Verdict |
|---|---|---|---|---|---|---|---|---|
| **ftsurrogate** | **79.32%** | **+4.12 pp** | 8/9 | 0.012 | 0 | +6.10 | +2.39 | **PROMOTE** |
| freqshift | 75.40% | +0.19 | 6/9 | 0.76 | 2 | +0.73 | −0.46 | killed |
| timemask | 74.97% | −0.23 | 3/9 | 0.72 | 1 | −1.58 | −0.96 | killed |
| timereverse | 74.92% | −0.28 | 4/9 | 0.84 | 4 | −0.54 | −0.77 | killed |
| chandrop | 72.67% | −2.53 | 1/9 | 0.027 | 6 | −1.39 | −1.62 | killed |

Holm-adjusted one-sided p: ftsurrogate 0.029; all other arms 1.000.

FTSurrogate gain per subject (pp): s1 +3.7, s2 −1.0, s3 +2.0, s4 +4.2, s5 +15.4, s6 +3.9,
s7 +6.2, s8 +2.3, s9 +0.3. It also meets the catalogue's stricter promotion bar: at least 7/9
non-negative and no subject below −5 pp.

**Reading:**
- This is the largest single gain in the project.
- It is largest where it is most needed: weak subjects gain +6.1 pp on average, and s5 gains +15.4 pp.
- This fits the "variance-limited" reading from R6 (the ensemble adds ~3 pp). FTSurrogate randomises
  phase while keeping each channel's amplitude spectrum, which tells the network that only band
  power carries the class. The squaring/log-variance layer of ShallowConvNet is built to read
  exactly that.
- FrequencyShift does nothing. This makes the subject-5 "peak shift" story less likely; that
  subject's gain comes from FTSurrogate instead.
- Channel dropout hurts on RA input.

**Post-hoc descriptive check (not a gate):** I recomputed accuracy per E run from the cached
predictions, as the 3-seed mean over 9 subjects.

| E run | 1 | 2 | 3 | 4 | 5 | 6 | runs 4–6 minus runs 1–3 |
|---|---|---|---|---|---|---|---|
| baseline | 78.5 | 76.7 | 73.4 | 77.4 | 74.5 | 70.7 | −2.0 |
| ftsurrogate | 82.9 | 80.0 | 77.4 | 81.4 | 78.6 | 75.5 | −1.6 |

The gain is a level shift of +3.3 to +4.9 pp in every run. It does not remove the
within-session decline (D5), so the decline remains an open, separate problem.

**Next (pre-registered rule):** PROMOTE means one confirmatory run with the setting frozen as is,
on a second dataset (IV-2b, the same harness as R7). It is not adopted until then. R6/R7 showed
that an IV-2a screen pass can fail to replicate.

## R9 — FTSurrogate confirmation on IV-2b: NOT CONFIRMED (`experiments/r9_ftsurrogate_iv2b.py`)

Pre-registered and committed (`c84de96`) before the run. The R8 setting was frozen. The run uses
the R7 harness: train on 0train–2train, evaluate on 3test and 4test, RA per recording, and the R7
teacher predictions as the seed-matched baseline.

**Validity:** V1 passed — the seed-0 baseline reproduced the R7 teacher exactly for every subject.
Baseline 78.58% (≥ 60%). Alignment checks passed.

| | Baseline | FTSurrogate | Difference |
|---|---|---|---|
| Single model (3-seed mean) | 78.58% | 79.28% | **+0.70 pp, 95% CI [−0.40, +1.81], 7/9 up, one-sided p = 0.096** |
| 3-seed ensemble | 78.81% | 79.59% | +0.79 pp |

Per subject (pp): s1 −0.9, s2 +0.1, s3 +1.5, s4 −1.6, s5 +0.6, s6 +0.1, s7 +1.6, s8 +2.8, s9 +2.2.

**Verdict: NOT CONFIRMED.** FTSurrogate is not adopted. The direction is the same, but the effect
is about one-sixth of the IV-2a screen (+4.12 pp).

## G0 — pre-cue alpha gate: FAIL (`experiments/g0_alpha_gate.py`)

Pre-registered and committed (`b78ef26`). Session T only.
- **Question:** does a trial's own pre-cue alpha predict whether leave-one-run-out CSP+LDA decodes
  it correctly, beyond run index?
- **Test:** combined likelihood-ratio test over the 9 subjects, chi2 with 9 df.

| Analysis | Combined LR | p | Alpha coefficient < 0 |
|---|---|---|---|
| **T (gate)** | 9.73 | **0.37** | 5/9 |
| T control (alpha permuted within each run) | 1.74 | 0.99 (control valid) | 6/9 |
| E (descriptive) | 8.50 | 0.49 | 5/9 |

**Verdict: FAIL.** The alpha–accuracy link from D5 exists only at the level of runs: both alpha
and errors rise with time on task. Within a run, a trial's alpha says nothing about whether it
will be decoded correctly. Ideas A (alpha residualisation) and B (alpha-conditioned FiLM) are
closed without being built.

## What R6–R9 and G0 say about the search itself

Two methods passed the IV-2a screen and then shrank sharply on IV-2b:

| Method | IV-2a screen | IV-2b result |
|---|---|---|
| Self-training | +1.62 pp | +0.22 pp |
| FTSurrogate | +4.12 pp | +0.70 pp |

This is the winner's curse. Screening many ideas on the same 9 E sessions and keeping the best one
inflates the estimate. An "idea-space descent" that uses IV-2a E as its loss would mostly fit noise
in those 9 sessions.

From round 2 onwards:
1. Every screen is scored on **both** datasets at once (18 subjects).
2. A screen passes only if the pooled gain passes **and** the sign is positive on each dataset.
3. Effects are reported as pooled estimates, not as the best dataset's number.

## R10 — round 2 (IV-2a + IV-2b jointly): the CNN + TS-LR stack PASSES (`experiments/r10_round2_stack_position.py`)

Pre-registered and committed (`95525d3`). This is the first screen under the joint rule:
18 subjects, Holm-adjusted, and the sign must be positive on each dataset. It runs on CPU in
about 5 minutes and uses the cached CNN predictions.

| Dataset | CNN (single) | TS+LR | Stack (CNN seed + TS+LR) | Position-matched TS+LR |
|---|---|---|---|---|
| IV-2a | 75.21 | 69.25 | **77.88** | 69.44 |
| IV-2b | 78.58 | 77.37 | **79.63** | 77.72 |

| Contrast | Pooled | Up | p (Holm) | IV-2a | IV-2b | Verdict |
|---|---|---|---|---|---|---|
| stack − CNN | **+1.86 pp** | 17/18 | < 0.001 | +2.67 | +1.05 | **PASS** |
| posmatch − TS+LR | +0.27 pp | 10/18 | 0.082 | +0.19 | +0.35 | killed |

TS+LR alone (the classic Riemannian decoder) is weaker than the CNN: −5.95 pp on IV-2a and
−1.20 pp on IV-2b.

**Post-hoc control (descriptive): is the gain just from ensembling?**

| | IV-2a | IV-2b |
|---|---|---|
| CNN, 2-seed ensemble (mean of 3 pairs) | 76.99 | 78.69 |
| Stack: 1 CNN seed + TS+LR | 77.88 | 79.63 |
| CNN, 3-seed ensemble | 78.09 | 78.81 |
| **CNN, 3-seed ensemble + TS+LR** (0.5/0.5) | **78.97** | **80.05** |

- Stack minus the 2-seed CNN ensemble: +0.91 pp pooled, 12/18 up, p = 0.008.
- The 3-seed ensemble plus TS+LR minus the 3-seed ensemble alone: +1.06 pp pooled, 15/18 up,
  p = 0.002. This is positive on both datasets (+0.89 and +1.24).
- So about +1 pp comes from **model diversity**, not from the number of members. A weaker
  Riemannian model still fixes some of the CNN's errors.

**Status and caveats:**
- The stack is the first method in the project whose gain holds on both datasets. It is the new
  best pipeline: 3 CNN seeds plus TS+LR gives 79.0% on IV-2a and 80.1% on IV-2b, against the R3
  single model at 75.2% and 78.6%.
- The 0.5/0.5 weight was fixed a priori.
- The pre-registered test is stack versus a single CNN. The diversity-versus-size comparison is
  post-hoc.
- No untouched third two-session dataset is available locally (Lee2019 exposes only one day
  through MOABB, see docs/datasets.md §4). Any further steps that select more ensemble members
  must therefore be scored on both datasets, with this R10 result as the bar to beat.

## R11 — round 3: a third ensemble member; FTSurrogate CNNs PASS, low-frequency member INVALID (ocular) (`experiments/r11_round3_diverse_members.py`)

Ideation and scoring: `reports/r11_ideation_ensemble_diversity.md` (F = 0.5·coherence +
0.5·unavailability). Pre-registered and committed (`cf41e0c`) before the run. CPU only; no new
CNNs were trained (cached r3/r7/r8/r9 predictions).

- **Comparator (R10 best):** r10 = 0.5·CNN3 + 0.5·TS+LR (CNN3 = base ShallowConvNet seeds 0–2).
- **Arms (third member, equal 1/3 weights, all settings a priori):**
  - `lowfreq`: 0.5–4 Hz time course, 100 ms bins, per-session z-scoring, shrinkage LDA;
  - `fts`: FTSurrogate ShallowConvNet seeds 0–1;
  - `fbts`: filter-bank TS+LR over six bands from 4 to 38 Hz.
- **Size-matched control:** `ctrl` = (CNN3 + TS+LR + base CNN seeds 3–4) / 3.
- **Gate:** pooled ≥ +0.5 pp, Holm one-sided sign-flip p < 0.05 over the 3 arms, and > 0 on each dataset.

```powershell
.venv\Scripts\python experiments\r11_round3_diverse_members.py     # ~20 min, exit 0
```

**Validity (all passed):**
- V1: every loader's labels, runs and recordings match the wide CNN loader.
- V2: every cache's recomputed accuracy equals its recorded r3/r7/r8/r9 value (< 1e-9).
- V3: R10's cnn, tslr and stack accuracies are reproduced exactly for all 18 subjects.
- V4: a second full run gives a byte-identical CSV.

| Dataset | r10 | ctrl | **fts** | fbts | lowfreq | member alone: FTS2 / FB / LF |
|---|---|---|---|---|---|---|
| IV-2a | 78.97 | 79.24 | **82.45** | 79.63 | 80.63 | 79.98 / 71.03 / 46.57 |
| IV-2b | 80.05 | 79.44 | **80.86** | 80.31 | 82.50 | 79.51 / 77.87 / 67.43 |

| Contrast | Pooled | Up / down | p (Holm) | IV-2a | IV-2b | Verdict |
|---|---|---|---|---|---|---|
| **fts − r10** | **+2.14 pp** | 15 / 3 | 0.0009 (0.0018) | +3.47 | +0.81 | **PASS** |
| lowfreq − r10 | +2.06 pp | 14 / 4 | 0.0006 (0.0018) | +1.66 | +2.45 | PASS by the letter → **INVALID** (see R11b) |
| fbts − r10 | +0.46 pp | 10 / 7 | 0.19 (0.19) | +0.66 | +0.27 | killed |
| fts − ctrl (diversity) | +2.31 pp | 16 / 0 | < 0.001 | +3.20 | +1.41 | credited to diversity |
| fbts − ctrl | +0.63 pp | 13 / 4 | 0.13 | +0.39 | +0.87 | not credited |
| ctrl − r10 (member count) | −0.17 pp | 7 / 10 | 0.66 | +0.27 | −0.60 | adding CNN seeds does nothing |

**R11b — post-hoc ocular check (`experiments/r11b_lowfreq_eog_check.py`, descriptive, not pre-registered).**
The low-frequency member decoded far above chance on its own, which is the signature of eye
movements towards the cue arrow (or, in IV-2b, following the feedback smiley). We rebuilt the
member exactly on the EEG channels (reproduces R11, max |diff| 0.000 pp) and on the **3 EOG
channels only**:

| | IV-2a | IV-2b | As third member, − r10 |
|---|---|---|---|
| LF member on EEG | 46.57 | 67.43 | +2.06 pp, 14/18 up |
| LF member on **EOG only** | 44.60 | 69.56 | **+3.28 pp**, 13/18 up |

EOG alone carries the same class information, and its gain as a member is even larger. **So the
`lowfreq` gain is ocular, not motor imagery. It is invalid as a BCI result and is not adopted.**
This is a legitimate finding about the datasets: the cue-locked < 4 Hz band of IV-2a/2b is
dominated by class-dependent eye movement. Any < 4 Hz method on them needs an EOG control.

**Reading of `fts`:**
- FTSurrogate CNNs do not replicate as a *replacement* for the base CNN on IV-2b (R9, +0.70 pp).
  As an additional *member* they pass on both datasets.
- Against the size-matched control the gain is 16/18 up and 0 down, with +1.41 pp on IV-2b.
  Adding two more base seeds instead does nothing (−0.17 pp). The gain is therefore
  augmentation-induced diversity, not member count.

**Caveats:**
- The IV-2a part is inflated by s5 (+14.2 pp); FTSurrogate itself was selected on IV-2a E in R8.
  Without s5 the pooled gain is +1.43 pp. IV-2b (+0.81 pp; s1 −1.6, s7 −0.9) is the cleaner estimate.
- The 1/3 weights were fixed a priori. No weights were fitted.
- Combining `fts` with other members is not tested here.

**New best pipeline:** (CNN3 + TS+LR + FTS2) / 3, giving **82.45% on IV-2a and 80.86% on IV-2b**.
The previous best (R10) was 78.97% and 80.05%.

## R12 — round 4: a fourth member; NO IMPROVEMENT, the loop stops (`experiments/r12_round4_fourth_member.py`)

Ideation and scoring: `reports/r12_ideation_fourth_member.md`. Pre-registered and committed
(`2a88087`) before the run. Comparator B = the R11 best, (CNN3 + TS+LR + FTS2) / 3.

- **Arms (fourth member, weight 1/4 each):**
  - `zoo`: the mean of ShallowConvNet seed 0 trained with each of FrequencyShift, SmoothTimeMask
    and TimeReverse. IV-2a predictions come from the R8 cache; the 27 IV-2b fits were trained here
    exactly as in R9.
  - `late`: TS+LR on 2.0–4.0 s.
- **Control:** `ctrl`, whose fourth member is base CNN seeds 3–4 plus FTSurrogate seed 2.
- **Gate:** the same as R11, with Holm over 2 arms.

**Validity (all passed):**
- V1: trial order checked.
- V2: all caches reproduce their recorded accuracies.
- V3: B reproduces the R11 `fts` arm exactly.
- V4: on the GPU path, the IV-2b seed-0 baseline reproduces the R7 teacher for all 9 subjects, and
  subject 1's FTSurrogate seed 0 reproduces R9.
- V5: a second run gives a byte-identical CSV.

| Dataset | B | ctrl | zoo | late | members alone: zoo / late TS+LR / more seeds |
|---|---|---|---|---|---|
| IV-2a | 82.45 | 82.37 | 82.72 | 82.48 | 80.02 / 59.80 / 79.01 |
| IV-2b | 80.86 | 80.57 | 81.38 | 81.30 | 80.41 / 71.26 / 79.84 |

| Contrast | Pooled | Up / down | p (Holm) | IV-2a | IV-2b | Verdict |
|---|---|---|---|---|---|---|
| zoo − B | +0.40 pp | 9 / 5 | 0.095 (0.19) | +0.27 | +0.52 | killed |
| late − B | +0.24 pp | 9 / 8 | 0.19 (0.19) | +0.04 | +0.44 | killed |
| zoo − ctrl | +0.58 pp | 11 / 5 | 0.019 | +0.35 | +0.81 | some diversity, below the gate |
| ctrl − B | −0.18 pp | 6 / 9 | 0.85 | −0.08 | −0.29 | more seeds of the existing families do nothing |

**Reading:**
- The member-addition axis has saturated. Per round, the gain over the previous best was:

  | Round | Gain |
  |---|---|
  | R10, 2nd member | +1.86 pp |
  | R11, 3rd member | +2.14 pp |
  | R12, 4th member | +0.40 pp (n.s.) |

- The zoo shows some real diversity against the control (+0.58 pp, p = 0.019), but too little to
  clear +0.5 pp over B under Holm.
- The three spectrum-preserving augmentations make errors much closer to the base CNN's than
  FTSurrogate does. That fits R11's reading: phase randomisation is what changes the CNN's errors.

**Loop stopping decision.** This round brought no improvement, so the ideation → screen loop
stops here.
- The remaining candidates from rounds 3–4 all score F ≤ 0.43, or need every CNN to be retrained
  so its weights are saved: test-time FTSurrogate, and leave-one-run-out weights or stacking on the
  training session.
- Re-opening the loop should start from a new axis rather than another member, for example
  test-time augmentation. It should be pre-registered against B.

**Final best pipeline (R11):** mean probabilities of 3 base ShallowConvNet seeds, TS+LR, and 2
FTSurrogate ShallowConvNet seeds, at weights 1/3 each. This gives **82.45% on IV-2a and 80.86% on
IV-2b**, against the R3 single model at 75.21% and 78.58%.
- Caveat: the IV-2a part of R11's gain is inflated by subject 5, and FTSurrogate was itself
  selected on IV-2a's evaluation session.
- No untouched dataset confirms this pipeline yet.

## R13 — round 5, a new axis: test-time FTSurrogate; NO IMPROVEMENT, the loop stops (`experiments/r13_tta_ftsurrogate.py`)

Ideation: `reports/r13_ideation_test_time_ftsurrogate.md`. Pre-registered and committed (`077bd57`)
before the run.
- **Bug fix during the run:** the first launch stopped at V2 on subject 1. The model was built
  before seeding, which R8/R9 do not do. It was fixed in `211ea66` with the protocol unchanged and
  no result computed.
- **Retraining:** all 90 CNN members were retrained (base seeds 0–2 and FTSurrogate seeds 0–1, on
  both datasets).
- **TTA:** each member's test prediction is the mean of the clean softmax and K = 8 FTSurrogate
  copies (probability 1, phase noise 0.9).
- **Arms:** `tta_fts` applies TTA to the FTSurrogate members only; `tta_all` applies it to every CNN.
  Both are compared against B, the R11 best.

**Validity (all passed):**
- V1: trial order checked.
- V2: every one of the 90 retrained models reproduces its cached prediction (identical argmax,
  max |Δp| < 1e-5).
- V3: B reproduces R11 exactly.
- V4: rerunning with warm caches gives a byte-identical CSV. The 5 IV-2b subject-1 models were
  deleted and retrained, and the recomputed files are byte-identical.

| Dataset | B | tta_fts | tta_all | single base: clean → TTA | single FTS: clean → TTA |
|---|---|---|---|---|---|
| IV-2a | 82.45 | 81.91 | 79.86 | 75.21 → 71.31 | 79.51 → 77.30 |
| IV-2b | 80.86 | 81.53 | 81.26 | 78.58 → 79.29 | 79.49 → 78.77 |

| Contrast | Pooled | Up / down | p (Holm) | IV-2a | IV-2b | Verdict |
|---|---|---|---|---|---|---|
| tta_fts − B | +0.07 pp | 10 / 7 | 0.43 (0.86) | −0.54 | +0.67 | killed |
| tta_all − B | −1.09 pp | 5 / 10 | 0.95 (0.95) | −2.58 | +0.41 | killed |
| single FTS, TTA − clean | −1.47 pp | 4 / 14 | — | −2.22 | −0.72 | TTA hurts |

**Reading:**
- The hypothesis was that FTSurrogate-trained models are phase-invariant, so averaging over phase
  surrogates would only remove noise. That is false: their accuracy drops on surrogate copies.
- So FTSurrogate works during *training* as a regulariser, not because it teaches invariance.
  Its member diversity (R11) comes from a different solution, not from phase invariance.
- On the base CNN, TTA hurts IV-2a (22 channels, −3.9 pp) and helps IV-2b slightly (3 channels,
  +0.7 pp). The sign flip between datasets is one more example of why the joint rule is needed.

**Loop stopping decision (final).** Two consecutive rounds (R12 and R13), on two different axes,
brought no improvement, so the ideation → screen loop stops.
- **The best pipeline stays R11:** (3 base ShallowConvNet seeds + TS+LR + 2 FTSurrogate
  ShallowConvNet seeds) / 3. It reaches **82.45% on IV-2a and 80.86% on IV-2b**.
- **Before more screening:** the next informative step is a confirmation on an untouched
  two-session dataset, not another screen on these 18 subjects.

## R14 — EOG validity gate on the R11 best: PASS, with a measurable ocular share (`experiments/r14_eog_gate.py`)

The council review (`reports/r14_council_and_ideation.md`) flagged that no member of the R11 best has an
EOG control. Its first recommendation was to measure how much of B depends on ocular signal. The gate was
pre-registered and committed (`e986e2a`) before the run.
- **Fix during the run:** the self-check that the regression residual is uncorrelated with EOG used a
  1e-6 tolerance. Round-off reached 1.2e-6 on IV-2a subject 4. The tolerance was loosened to 1e-4 in
  `cbe9228`; the protocol was unchanged and no result had been computed.

**Design.**
- Per subject and per band, the EEG is regressed on the 3 EOG channels by least squares. The coefficients
  are fitted on the training session only (IV-2b: recordings 0–2train) and the cleaned EEG is
  EEG − EOG·B.
- Every member is then retrained on the cleaned EEG with its unchanged code path and seeds:
  3 base CNNs, 2 FTSurrogate CNNs and TS+LR.
- Members built on the 3 EOG channels only: TS+LR at 8–30 Hz, TS+LR at 4–38 Hz on the CNN's window,
  and a ShallowConvNet.
- **Gate:** B − B_clean ≥ +1.0 pp pooled with one-sided p < 0.05 counts as an ocular-material share.
  The regression also removes any neural signal that reaches the EOG electrodes, so the drop is an
  *upper bound* on the ocular share.

**Validity (all passed):**
- V1: the raw epochs cut here equal the harness arrays exactly (max |diff| = 0) for both bands, with the
  same labels and recordings.
- V2: TS+LR on the raw EEG reproduces R11 per subject, and the regression residual is orthogonal to EOG.
- V3: B reproduces R11 exactly.
- V4: a rerun with the 6 IV-2b subject-1 models deleted and retrained gives byte-identical model files
  and a byte-identical CSV.

| Dataset | B | B_clean | r10 | r10_clean | CNN3 → clean | FTS2 → clean | TS+LR → clean | EOG-only: TS+LR 8–30 / TS+LR 4–38 / CNN |
|---|---|---|---|---|---|---|---|---|
| IV-2a | 82.45 | 81.98 | 78.97 | 78.90 | 78.09 → 77.66 | 79.98 → 81.06 | 69.25 → 68.87 | 34.65 / 41.13 / **52.31** (chance 25) |
| IV-2b | 80.86 | 79.81 | 80.05 | 78.38 | 78.81 → 77.79 | 79.51 → 78.69 | 77.37 → 77.42 | 60.84 / 66.76 / **69.65** (chance 50) |

| Contrast | Pooled | Up / down | p | IV-2a | IV-2b | Verdict |
|---|---|---|---|---|---|---|
| **B − B_clean** (primary) | **+0.75 pp** | 13 / 3 | 0.017 | +0.46 | +1.04 | **PASS** (below the 1.0 pp line) |
| B_clean − r10_clean | +2.26 pp | 13 / 3 | 0.001 | +3.09 | +1.44 | FTSurrogate diversity survives cleaning |
| EOG-only CNN − chance | +23.5 pp | 18 / 0 | < 0.001 | +27.3 | +19.7 | EOG carries strong class information |
| excess CNN/EOG error agreement | +3.4 pp | 13 / 5 | 0.020 | +3.4 | +3.5 | CNN errors partly share the ocular drive |
| CNN3 raw − clean | +0.72 pp | 10 / 5 | 0.055 | +0.42 | +1.02 | |
| TS+LR raw − clean | +0.17 pp | 10 / 8 | 0.35 | +0.39 | −0.05 | TS+LR is essentially EOG-free |
| FTS2 raw − clean | −0.13 pp | 10 / 7 | 0.60 | −1.08 | +0.82 | |

The regression removes 33% of the EEG variance on IV-2a E (22 channels, frontal ones included) and 7.4% on
IV-2b (C3/Cz/C4 only).

**Reading:**
- The R11 headline is mostly neural. Removing EOG costs at most about 0.75 pp, and the pre-registered
  decision keeps B as the reference.
- The ocular share is still real and significant (p = 0.017). It is concentrated in the plain CNNs and in
  IV-2b (+1.04 pp), which fits IV-2b's feedback sessions where gaze follows the class-dependent smiley.
- On their own, the 3 EOG channels decode almost as well as early-2000s EEG pipelines (CNN 52% 4-class,
  70% 2-class). This is a dataset property any IV-2a/2b result inherits.
- TS+LR (8–30 Hz, 0.5–2.5 s) is unaffected by EOG removal. The CNN's wider band and window are where
  ocular leakage enters.
- The diversity result from R11 holds on clean data: the FTSurrogate member adds +2.26 pp over the clean
  R10 stack, on both datasets.
- **EOG-clean reference numbers:** 81.98% on IV-2a and 79.81% on IV-2b. Any lockbox confirmation should
  report the clean pipeline next to the raw one.

## Consolidation: the R11 pipeline as one entry point (`src/pipeline.py`)

This was step 2 of the council's sequence (`reports/r14_council_and_ideation.md` §7). No new science.

- **What it is.** `src.pipeline.fit_predict(data, n_classes, cfg)` fits every member on the training
  trials and returns test probabilities per family plus the ensemble.
  - Families are the plain CNN, TS+LR and the FTSurrogate CNN. Seeds are averaged within a family, and
    families are averaged with equal weight.
  - `load_subject_data("IV-2a" | "IV-2b", s)` builds the input, with the trial-order check.
- **Configurations.** `PipelineConfig(n_cnn, n_fts, use_tslr)` sets the member counts. Three are named:
  - `R11`: 3 base CNN seeds + TS+LR + 2 FTSurrogate seeds, 5 GPU fits;
  - `R10`: 3 base CNN seeds + TS+LR;
  - `LITE`: 1 FTSurrogate CNN + TS+LR, 1 GPU fit.
- **Alignment.** One `align_by_group` replaces the three per-script variants (per session, per
  recording, and the TS+LR one), and it self-checks that each aligned group has an identity mean.
- **Reproduction check** (`experiments/check_pipeline_reproduces_r11.py`; 90 GPU fits, passed):
  - every CNN member equals its cached R3/R7/R8/R9 prediction (identical argmax, max |Δp| < 1e-5);
  - per subject, TS+LR, R10 and R11 give exactly the accuracies recorded in
    `results/r11_round3_diverse_members.csv`.

| Config | GPU fits / subject | IV-2a | IV-2b |
|---|---|---|---|
| R11 | 5 | 82.45 | 80.86 |
| R10 | 3 | 78.97 | 80.05 |
| LITE (FTS seed 0 + TS+LR) | 1 | 80.86 | 81.40 |

LITE is descriptive only: it was not screened, and seed 0 is a favourable draw. The seed-averaged
council ablation gives 80.79 / 80.84. It needs a pre-registered non-inferiority test, ideally on the
lockbox dataset, before it can replace R11.

## D6 — the IV-2b gap to the published ShallowConvNet is eye movement (`experiments/d6_iv2b_gap.py`)

**Question.** CTNet's leak-free reimplementation of ShallowConvNet scores 85.13% on IV-2b; ours scores
78.58% (`docs/state_of_the_art.md` §0). The gap sits *below* any ensembling. The diagnostic was
pre-registered and committed (`e75f782`) before the run.

**Design.** A single ShallowConvNet on IV-2b, seeds 0–2, with one change at a time towards the published
data path (EEG-Conformer / CTNet):
- the 0.5–100 Hz band;
- a global z-score instead of Riemannian alignment;
- 1000 epochs instead of 300;
- all three together ("recipe");
- the recipe on EOG-regressed EEG (R14's regression, coefficients fitted on the training recordings).

**Fix during the run.** The regression's self-check used a centred correlation. Least squares without an
intercept only guarantees *uncentred* orthogonality, and the 0.5 Hz high-pass leaves an offset, so the
check fired (1.6e-4) on subject 5. It now checks the uncentred cosine (`99cbedd`). A warm rerun of R14
gives a byte-identical CSV, so R14's results are unchanged.

**Validity (all passed):**
- V1: the 4–38 Hz epochs equal the harness arrays.
- V2: the base cell retrained here reproduces the cached R7 teacher.
- V3: a warm rerun gives a byte-identical CSV.

| Cell (single model, mean of 3 seeds) | IV-2b | − base | Up / down | p |
|---|---|---|---|---|
| base: RA, 4–38 Hz, 300 epochs (ours) | 78.58 | — | | |
| global z-score, no RA | 76.63 | −1.94 | 3 / 6 | 0.95 |
| **0.5–100 Hz band** | **82.93** | **+4.35** | 6 / 3 | 0.045 |
| 1000 epochs | 77.82 | −0.76 | 3 / 5 | 0.94 |
| recipe (0.5–100 Hz, z-score, 1000 epochs) | 82.33 | +3.75 | 5 / 4 | 0.12 |
| **recipe on EOG-regressed EEG** | **77.76** | −0.82 | 4 / 5 | 0.65 |
| *recipe − recipe on EOG-regressed EEG (ocular share)* | | **+4.57** | **8 / 1** | **0.004** |

The 3-seed ensembles give the same picture: the ocular share is +4.26 pp, with 9 subjects up and none
down.

**Reading:**
- The whole IV-2b gap comes from **the band**: the 0.5–4 Hz content. Neither alignment (z-score is
  −1.9 pp) nor the longer training (−0.8 pp) contributes.
- That gain is **ocular**. Regressing the 3 EOG channels out of the broadband EEG removes all of it and
  more (82.3 → 77.8, 8 of 9 subjects down). The gainers lose the most: s3 69.8 → 59.9, s5 95.7 → 84.5,
  s7 88.1 → 78.4.
- This matches R11b. IV-2b's test recordings are feedback sessions in which gaze follows the
  class-dependent smiley.
- **Implication for the state of the art.** Published IV-2b numbers (85–90%) come from the broadband
  data path, so they very likely include about 4–5 pp of eye-movement decoding. On EOG-clean input, the
  published recipe (77.8%) is no better than ours (EOG-clean 77.8% in R14).
- Our 4–38 Hz band already excludes most of the ocular content: R14 measured about 1 pp. So our 78.6% is
  close to the honest neural ceiling of this architecture on IV-2b.
- **Decision:** do *not* widen our band to chase the published IV-2b number. Report the comparison with
  this caveat. The fair bar on IV-2b is an EOG-controlled one, which no published paper reports.
- Segment-and-recombine augmentation, the remaining recipe element, was not tested.

## L1 — LOCKBOX: R11 replicates on the untouched BNCI2015_001 (`experiments/l1_lockbox_bnci2015_001.py`)

**Protocol.** Pre-registered and committed (`c6762b9`) *before* the dataset was downloaded. A dry run on
MOABB's FakeDataset exercised the code first. The dataset was opened once, with no reruns.
- **Dataset:** BNCI2015_001, 12 subjects, 13 channels, right hand vs feet.
- **Split:** train on session A, test on session B, with 200 test trials per subject.
- **Pipeline:** `src/pipeline.py` unchanged; input resampled to 250 Hz.
- **No EOG control:** the dataset has no EOG channels.

| Arm | Accuracy |
|---|---|
| CSP+LDA+RA | 76.96 |
| TS+LR | 80.88 |
| Single ShallowConvNet (mean of 5 seeds) | 84.19 |
| R10 (CNN3 + TS+LR) | 86.00 |
| Size-matched control (CNN3 + TS+LR + 2 more base seeds) | 86.58 |
| LITE (1 FTSurrogate CNN + TS+LR) | 85.42 |
| **R11 (CNN3 + TS+LR + FTS2)** | **87.88** |

| Hypothesis | Mean [95% CI] | Up / down | p (Holm) | Verdict |
|---|---|---|---|---|
| **H1** R11 − single CNN | **+3.68 pp** [+1.89, +5.69] | 10 / 2 | 0.0020 | **SUPPORTED: R11 replicates** |
| H2 R11 − size-matched control | +1.29 pp [−0.29, +3.00] | 7 / 4 | 0.085 | not supported |
| **H3** R11 − CSP+LDA+RA | **+10.92 pp** [+6.58, +15.88] | **12 / 0** | 0.0007 | **SUPPORTED** |
| R10 − single CNN (secondary) | +1.81 pp | 10 / 1 | 0.001 | |
| R11 − R10 (secondary) | +1.88 pp | 9 / 3 | 0.027 | |
| LITE − R11, non-inferiority at −1 pp (secondary) | −2.46 pp, lower bound −3.58 | 0 / 10 | | **not non-inferior** |

**Reading:**
- **The pipeline is confirmed on unseen data.** The gain over a single CNN (+3.68 pp) almost exactly
  matches the council's shrunk prediction of about +3.5 pp. The gain over CSP+LDA+RA is large and holds
  for every one of the 12 subjects.
- **The FTSurrogate-diversity mechanism is only partly confirmed.** R11 still beats R10 (+1.88 pp,
  p = 0.027). Against the size-matched control the effect has the predicted size (+1.29 pp; the
  council's shrunk estimate was about +1.2) but does not reach significance with 12 subjects.
- **LITE does not replace R11.** It loses 2.5 pp, with 10 of 12 subjects down. The extra fits are worth
  their cost.
- **R11 stays the reference pipeline,** now confirmed outside the development data.

## X1 — cross-paradigm session transfer (GuttmannFlury2025): SSVEP supports the recipe; the P300 data is not decodable through MOABB (`experiments/x1_cross_paradigm_guttmann.py`)

**Question.** Do the two ingredients of our motor-imagery recipe also help other paradigms transfer from
one session to the next?
- the ingredients: per-session Riemannian alignment, and a diverse second model in the ensemble;
- the test: train on the first session, test on the second, for the 17 subjects with two or more
  sessions;
- this is CPU only.

The study was pre-registered and committed before download (`c5ee6bc`), with a dry run on FakeDatasets.

**Fixes during the run** (each made before the affected subject was scored; protocol unchanged):
- `1d6837e`: channel-name case (`POz`/`Oz`).
- `0470efe`: subject 12 is dropped from SSVEP. Its Zenodo archive indexes the first session's EEG before
  the start of the file (the md5 matches Zenodo). A label-free integrity check of all 34 archives found
  no other problem.

**Limitations fixed in advance:**
- The paper does not report the interval between sessions.
- In SSVEP, gaze direction is class-locked by design: the four targets sit in the four screen quadrants.

| Paradigm | No alignment | + per-session alignment | Second model alone | Ensemble |
|---|---|---|---|---|
| SSVEP, accuracy, n = 16 (chance 25%) | 50.3 | **60.2** | CCA + LR: **84.1** | 65.2 |
| P300, AUC, n = 17 | 49.8 | 49.7 | xDAWN + LDA: 50.0 | 49.9 |

| Hypothesis | Mean [95% CI] | Up / down | p (Holm) | Verdict |
|---|---|---|---|---|
| S1 SSVEP: alignment − none | **+9.84 pp** [+2.50, +18.59] | 10 / 4 | 0.034 | **SUPPORTED** |
| S2 SSVEP: ensemble − aligned TS+LR | **+5.00 pp** [+2.34, +7.97] | 11 / 1 | 0.005 | **SUPPORTED** |
| P1 P300: alignment − none | −0.17 pp | 4 / 13 | 0.94 | not supported (**invalid**, see X1b) |
| P2 P300: ensemble − aligned TS+LR | +0.25 pp | 11 / 6 | 0.68 | not supported (**invalid**) |

**X1b, post-hoc validity check** (`experiments/x1b_p300_within_session_check.py`, first session only).
Even *within* one session, the P300 pipeline is at chance:

| Subject | Within-session 5-fold AUC | max \|Target − NonTarget\| | At |
|---|---|---|---|
| 2 | 0.463 | 571 µV | 0.000 s |
| 4 | 0.504 | 849 µV | 0.008 s |
| 13 | 0.506 | 1451 µV | 0.000 s |
| 17 | 0.510 | 62995 µV | 0.461 s |

The Target fraction is 1/6, as the speller design implies. The "difference" sits at flash onset with
artefact-sized amplitude, not at a ~300 ms P300. So the flash labels or event timing that the MOABB
adapter produces carry no P300. **The P300 arm measures nothing about session transfer.** It is
reported as invalid, not as a negative result. Repairing the adapter's event decoding from the sync and
E-Prime files would be a separate task.

**Reading (SSVEP):**
- Per-session alignment rescues the trained Riemannian decoder across sessions (+9.8 pp). Without
  alignment, 5 of 16 subjects collapse to exactly chance (25%), a constant prediction after the
  session shift.
- A diverse second model helps the aligned decoder (+5.0 pp, 11 up, 1 down), as in MI.
- **However, the training-free CCA decoder alone (84.1%) beats every trained pipeline,** including the
  ensemble (65.2%). Equal-weight averaging with the weaker aligned TS+LR drags CCA down.
- So the recipe's *direction* generalises: alignment helps, and diversity helps. For SSVEP, though,
  the best session-transfer strategy is a decoder that needs no session-specific training at all, so
  the recipe is not the best choice there.

## B1 — calibration burden: R11 halves the labelled trials needed to match a single CNN (`experiments/b1_r11_calibration_burden.py`)

**Question.** This project's goal is less calibration. Does the ensemble reduce how many labelled
training trials a user needs?

**Design.** Pre-registered and committed (`60fb56a`) before the run; A29 conventions.
- **Budgets:** 48, 96 and 144 labelled training trials, stratified by class, 3 seeded draws each.
- **Training side:** alignment is fitted on the drawn trials only.
- **Test:** the full evaluation session.
- **FULL** is the whole training session: 288 trials on IV-2a and 400–440 on IV-2b. It reproduces R11
  exactly.

| Budget (pooled, 18 subjects) | Single CNN | TS+LR | CSP+LDA+RA | R10 | **R11** |
|---|---|---|---|---|---|
| 48 | 60.51 | 68.75 | 67.06 | 66.42 | **69.88** |
| 96 | 67.50 | 71.19 | 70.08 | 72.10 | **74.72** |
| 144 | 71.00 | 72.09 | 70.60 | 75.03 | **77.48** |
| FULL | 76.89 | 73.31 | 72.31 | 79.51 | **81.65** |

| Budget | IV-2a: single CNN → R11 | IV-2b: single CNN → R11 |
|---|---|---|
| 48 | 49.49 → 63.50 | 71.54 → 76.25 |
| 96 | 58.79 → 71.30 | 76.21 → 78.15 |
| 144 | 64.95 → 75.68 | 77.06 → 79.27 |
| FULL | 75.21 → 82.45 | 78.58 → 80.86 |

**Parity with a single CNN on the full session** (R11 at n minus single CNN at FULL; one-sided 95%
bootstrap lower bound, non-inferiority margin −1 pp):

| n | Mean | Lower bound | Up | Verdict |
|---|---|---|---|---|
| 48 | −7.01 | −9.35 | 3/18 | not shown |
| **96 (primary H1)** | **−2.17** | **−3.18** | 3/18 | **not shown: H1 not supported** |
| 144 (secondary) | +0.59 | −0.21 | 11/18 | **non-inferior** |

**R11's gain over the single CNN *grows* as the budget shrinks:**
- +4.8 pp at FULL, +6.5 pp at 144, +7.2 pp at 96 and +9.4 pp at 48;
- at FULL and at 48 every one of the 18 subjects improves;
- p < 0.0001 at every budget.

Over CSP+LDA+RA the gain is +2.8 / +4.6 / +6.9 / +9.3 pp, from 48 trials up to FULL.

**Reading:**
- The pre-registered parity at 96 trials fails: R11 at 96 is 2.2 pp below a single CNN on the full
  session. But **R11 at 144 trials matches it** (+0.6 pp, non-inferior). That is about half the
  labelled data on IV-2a (144 vs 288) and about a third on IV-2b (144 vs ~420).
- **The ensemble matters most when labels are scarce.** At 48 trials a single CNN is near 50% on 4-class
  IV-2a, while R11 reaches 63.5%. The Riemannian members (TS+LR, and alignment on the training side)
  carry the low-budget regime, and the CNNs take over as data grows. This is the "built-in floor"
  property the project's lessons call for.
- R10 (CNN3 + TS+LR) does not reach parity before FULL. The FTSurrogate members are part of the
  low-budget gain.
- **The project-level claim is now:** with the R11 ensemble, a new session needs about half the
  labelled calibration trials to match the accuracy a single CNN reaches with the full calibration
  session. This is on development data, so a lockbox confirmation of the *burden* claim is still
  needed.

## L2 — LOCKBOX: the calibration-burden claim CONFIRMS on BNCI2015_001 (`experiments/l2_lockbox_burden.py`)

**Protocol.** Pre-registered and committed (`473f78d`) before any reduced-budget result on BNCI2015_001
existed; opened once.
- **Design:** B1's design on the lockbox, using L1's data path and caches.
- **Training pool:** session A, 200 trials, 100 per class.
- **Budgets:** 48, 100 and 144 trials, 3 stratified draws each.
- **Test:** all of session B.
- **Primary:** R11 trained on **half** the session (100 trials) is non-inferior (margin −1 pp) to a
  single CNN trained on the full 200.
- **Caveat stated in advance:** L1 had already used this dataset's full-budget results.
- **Check:** FULL reproduces L1 for all 12 subjects.

| Budget | Single CNN | TS+LR | CSP+LDA+RA | R10 | **R11** |
|---|---|---|---|---|---|
| 48 | 76.08 | 79.33 | 75.86 | 80.22 | **82.03** |
| 100 | 79.44 | 80.18 | 74.60 | 82.94 | **84.56** |
| 144 | 83.41 | 80.69 | 77.40 | 85.28 | **86.83** |
| FULL (200) | 84.18 | 80.88 | 76.96 | 86.00 | **87.88** |

| R11 at n − single CNN at FULL | Mean | One-sided 95% lower bound | Up | Verdict |
|---|---|---|---|---|
| 48 | −2.15 | −3.46 | 4/12 | not shown |
| **100 (primary H1)** | **+0.38** | **−0.74** | 8/12 | **NON-INFERIOR: CONFIRMED** |
| 144 | +2.65 | +1.36 | 10/12 | non-inferior (and better) |

**R11's gain over a single CNN at the same budget:**
- +6.0 pp at 48 trials, +5.1 at 100, +3.4 at 144 and +3.7 at FULL;
- every subject improves at 48, 100 and 144, and 11 of 12 at FULL;
- p ≤ 0.0005 at every budget.

Over CSP+LDA+RA the gain is +6.2 to +10.9 pp.

**Reading:**
- **The burden claim replicates outside the development data.** On BNCI2015_001, R11 trained on half the
  labelled calibration trials matches a single ShallowConvNet trained on all of them.
- R11 with 144 trials beats the full-session single CNN outright, by +2.7 pp with the lower bound
  above zero.
- As on IV-2a/2b, the ensemble's advantage is largest when labels are scarce.
- **Combined with L1, the project's two headline claims both hold on an untouched dataset:**
  1. R11 beats a single CNN (+3.7 pp) and CSP+LDA+RA (+10.9 pp).
  2. R11 halves the calibration trials needed to reach single-CNN accuracy.
