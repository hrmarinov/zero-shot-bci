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
