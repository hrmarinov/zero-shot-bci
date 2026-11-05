# R14: council review of R3–R13, and ideation on how to use and simplify the pipeline

**Date:** 2026-11-05. **Inputs:** `docs/diag_gates_d1_d4.md` (R10–R13), `results/r10–r13_*.csv`, and the cached
per-trial predictions.

**Council.** Four independent reviewers:
- a statistical skeptic;
- a pipeline simplifier, who ran ablations from cached predictions on CPU;
- a BCI domain expert, who compared against the literature;
- a research strategist, who looked at datasets and the project goal.

**Status of the numbers.** All new numbers here are **post-hoc and descriptive** on E sessions that have already been
searched. None of them is a selection. The scripts are in the session scratchpad, not in the repo.

## 1. What is real (skeptic)

| Contrast | Pooled | Median | Without s5 | IV-2a [95% CI] | IV-2b [95% CI] |
|---|---|---|---|---|---|
| R11 − R3 single CNN | +4.76 | +3.59 | +3.85 | +7.24 [4.8, 10.9] | +2.28 [1.3, 3.5], 9/9 up |
| R11 − R10 | +2.14 | +1.25 | +1.43 | +3.47 [1.4, 6.4] | +0.81 [−0.21, 1.92], p = 0.11 |
| R11 − size-matched control | +2.31 | +1.65 | — | +3.20 | +1.41 [0.90, 1.97], 16/0 |

- **The most robust fact in the project.** A diverse ensemble (CNN seeds + TS+LR + FTSurrogate CNNs) beats a
  ensemble with the same number of members (16 subjects up, 0 down, positive on both datasets). The gain comes
  from diversity, not from adding members.
- **Shrunk estimates.** The empirical-Bayes prior over the 9 pre-registered R10–R13 contrasts is
  +0.42 ± 0.97 pp. Under it, the true gain of R11 is about **+1.2 pp over R10** (plausible +0.4 to +2.0) and
  about **+3.5 pp over R3 single** (+2.5 to +4.5). Roughly 1.5 pp of that is seed ensembling, 1 pp is TS+LR
  and 1 pp is FTSurrogate.
- **Selection history.** About 60 configurations have been screened on these E sessions. IV-2a's E is no longer
  clean. IV-2b's E is the less-selected of the two. Every R-number here is transductive: per-session RA uses the
  whole unlabelled E session.
- **Not safe to claim:**
  - 82.45% on IV-2a as an unbiased estimate;
  - that FTSurrogate improves single models;
  - generalisation to new subjects or sessions;
  - that the gain is *motor imagery only*;
  - single-model literature comparisons against a 6-member ensemble.

## 2. Where it sits (domain)

| Method, subject-specific T→E, no E labels | IV-2a | IV-2b |
|---|---|---|
| FBCSP (competition winners, κ at best time point) | ≈67.8 | ≈80 |
| ShallowConvNet (Schirrmeister 2017) | 73.7 | – |
| FBCNet | 76.2 | – |
| EEG-Conformer* | 78.66 | 84.63 |
| ATCNet* / CTNet* | 85.38 / 82.52 | – / 88.49 |
| **Ours, R11** | **82.45** | **80.86** |

\*The public code may select checkpoints on the E session. The domain reviewer recalled this but did not
re-check it.

- **IV-2a:** at the top of the leak-free numbers.
- **IV-2b:** the weak spot. It is level with 2008's FBCSP and 4–8 pp below the transformer results. With
  3 channels, ShallowConvNet's spatial layer has almost nothing to work with, and RA does little on a 3×3
  covariance.
- **Red flags:**
  - **Ocular contamination of the *existing* members.** R11b showed class-dependent EOG below 4 Hz. The CNN's
    4–38 Hz band includes the cue/saccade window and frontal channels. IV-2b's E sessions are feedback sessions
    with a moving smiley, so gaze is class-locked by design. No member has an EOG control.
  - **Tongue class.** Glossokinetic and EMG activity falls in 25–38 Hz.

## 3. What can be simplified (simplifier, cached predictions, descriptive)

| Pipeline | GPU fits / subject | IV-2a | IV-2b | Pooled | Share of R11's gain over CNN1 |
|---|---|---|---|---|---|
| R11 (CNN3 + TS+LR + FTS2)/3 | 5 | 82.45 | 80.86 | 81.65 | 100% |
| (CNN2 + FTS2 + TS+LR)/3 | 4 | 82.37 | 80.79 | 81.58 | 98% |
| (CNN1 + FTS1 + TS+LR)/3, mean of 15 seed pairs | 2 | 81.22 | 80.79 | 81.00 | 86% |
| **½FTS1 + ½TS+LR, mean of 3 seeds** | **1** | 80.79 | 80.84 | 80.81 | **82%** |
| R10 (½CNN3 + ½TS+LR) | 3 | 78.97 | 80.05 | 79.51 | 53% |
| FTS1 alone, mean of 3 seeds | 1 | 79.32 | 79.28 | 79.30 | 49% |
| CNN1 (seed 0) | 1 | 75.31 | 78.87 | 77.09 | 0% |

- **One FTSurrogate CNN plus TS+LR** (1 GPU fit) beats R10 (3 fits). The gain has two sources: FTSurrogate as
  a training regulariser, and TS+LR diversity (+1.5 pp on top of FTSurrogate).
- **Extra seeds buy little.** Going from 2 fits to 5 adds about 0.65 pp, which is about the size of seed noise
  (±0.5 pp).
- **Weighting.** Equal-weight arithmetic averaging is as good as any alternative tried. The geometric mean and
  ½ TS+LR weighting are both slightly worse.
- **Code.** There is no single entry point. The final pipeline is spread across r3/r7/r8/r9/r10, and the two
  datasets use different `fit_predict` paths. Distillation into one CNN is judged implausible: the teachers
  memorise T.

## 4. Does it serve the project goal? (strategist)

**Only partly.** The goal is calibration-free or low-calibration cross-session decoding. Under TODO Part E #7,
every method must show a calibration-burden curve. So far, only CSP+LDA has one (alignment @96 trials ≈ none
@288). No CNN or ensemble has a burden curve. "R11 @ 96 trials ≥ CNN @ 288" would be a stronger result than
+pp at the full 288 trials.

**Process.** Idea-space descent did its job, and its round gains (+1.86, +2.14, +0.40, −1.09) show it has
saturated. More screening on these 18 E sessions mostly fits noise.

## 5. Where the council disagrees, and the resolution

| Issue | Position | Resolution |
|---|---|---|
| More members | **Domain:** architecture-diverse members (EEGNet, EEG-TCNet) are the untested lever, the same mechanism as FTSurrogate. **Strategist:** pause member ideation. | Architecture members are allowed only as a *dev* screen. They are budgeted (≤ 2 arms, ≤ 4 GPU-h) and only *after* a lockbox dataset is reserved. They are not the next move. |
| Headline baseline | **Skeptic:** compare against a size-matched CNN ensemble, not R3 single. **Others:** no objection. | Adopted. All future headlines report R11 − size-matched control. |

## 6. Ideation (F = 0.5·coherence + 0.5·unavailability; simplicity and cost noted separately)

| # | Idea | Mechanism / why now | Coh. | Unav. | F | Cost |
|---|---|---|---|---|---|---|
| **V** | **EOG validity gate on the existing members** (prerequisite, not scored) | TS+LR and ShallowConvNet trained on the 3 EOG channels only, at 8–30 and 4–38 Hz. EOG-regressed TS+LR, with the regression fitted on T, rerun through R10/R11. Then a conditional agreement test between the CNN's errors and the EOG-only member. If EOG-only is well above chance in the members' bands, part of 82/81 is ocular. | — | — | gate | CPU ~30 min; GPU follow-up |
| 1 | **Lockbox confirmation**: three pipelines frozen together (R11, lite ½FTS1+½TS+LR, and the size-matched CNN ensemble) on an untouched two-session dataset | This is the only real fix for the winner's curse. **Primary: BNCI2015_001** (12 subjects, 13 channels, 2 classes, 2–3 sessions; already in MOABB, small). **Larger/4-class: Zhou2020** (20 subjects, 7 sessions, 4 classes, EOG, about 17.6 GB; the strategist verified only the record page). Contrasts: R11 vs the control (primary), and lite vs R11 non-inferiority within 1 pp. Opened once, no reruns. | 0.8 | 0.6 | **0.70** | ~0.5 day of loader and pre-registration work; ~1.5 GPU-h |
| 2 | **Calibration-burden curve for R11 vs its members** at {48, 96, 144, 288} T trials | TS+LR and RA degrade gracefully at small n; CNNs may not. The ensemble may move the parity point (pre-register R11 @ 96 ≥ CNN @ 288 − 1 pp). This ties the work back to the project goal. | 0.7 | 0.6 | **0.65** | ~6–9 GPU-h (the TS+LR part is CPU) |
| 3 | **Lite pipeline as the default**: ½FTS1 + ½TS+LR, consolidated in `src/pipeline.py` with `n_cnn` and `n_fts` as parameters | It keeps about 82% of the gain with 20% of the GPU cost. The question is non-inferiority, not superiority. The consolidation is checked by reproducing the cached R11 numbers exactly. | 0.7 | 0.3 | 0.50 | CPU and engineering; the simplification itself |
| 4 | **Multi-session personalisation on IV-2b** (earlier recordings + k new-session trials), later on Zhou2020 sessions 2–7 | This is the project's actual cross-session question and reuses existing data. The longevity metric (A20/A31) comes almost for free. | 0.6 | 0.5 | 0.55 | ~3 GPU-h |
| 5 | **Architecture-diverse member** (EEGNet or EEG-TCNet on RA input, replacing base seeds; dev only) | It is the same error-decorrelation lever as FTSurrogate, and seeds are saturated. Risk: these architectures share ShallowConvNet's spectral priors. | 0.55 | 0.4 | 0.48 | ~3 GPU-h |
| 6 | **IV-2b-specific member**: an FBCNet-style band-variance CNN | Band diversity can stand in for the spatial diversity that 3 channels lack. fbts as a TS member was +0.27 on IV-2b, so the CNN form is untested. | 0.45 | 0.4 | 0.43 | ~2 GPU-h |
| 7 | Cropped training with test-time crop averaging | The shifted crops stay on the data manifold, so this is not the R13 failure mode. It is known to help ShallowConvNet. | 0.45 | 0.3 | 0.38 | ~3 GPU-h |
| 8 | Label smoothing or mixup on CNN members | Better-calibrated softmax should improve probability averaging. | 0.35 | 0.3 | 0.33 | ~3 GPU-h |
| ✗ | Distillation into one CNN; more seeds; re-weighting; Euclidean alignment vs RA | These are implausible, already measured as null, or a near-null simplification. | | | dropped | |

## 7. Recommended sequence

1. **V, the EOG gate** (CPU, now). If it fails, every later number needs EOG regression first. Cheap, and it
   protects everything downstream.
2. **#3, consolidation** into `src/pipeline.py` (no new science). This makes R11, lite and the control each a
   one-line configuration, and must reproduce R11 exactly.
3. **#1, the lockbox.** Pre-register R11, lite and the control *before* downloading. BNCI2015_001 is the
   primary. If it is ambiguous, Zhou2020 is the 4-class replicate.
4. **#2, the burden curve.** This is the project-goal story. It is pre-registered on the dev datasets and
   confirmed on the lockbox at the same time.
5. Only then, if wanted: **#5 or #6** as budgeted dev screens.

**Process changes adopted:**
- IV-2a and IV-2b are now *dev* datasets. A lockbox is opened once per frozen pipeline.
- Each round is capped at ≤ 2 arms and a declared GPU budget.
- Every headline is reported with s5 excluded and against the size-matched ensemble.
- Any feature below 8 Hz, or any frontal-heavy feature, gets an EOG control.
