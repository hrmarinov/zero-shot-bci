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
