# Ideation round 4: a fourth ensemble member

**Date:** 2026-11-05. **Scoring:** F = 0.5·coherence + 0.5·unavailability, as in R11.

**Current best (R11):** B = (CNN3 + TS+LR + FTS2) / 3. That gives 82.45% on IV-2a and 80.86% on IV-2b.

## Facts added by R11

| # | Fact |
|---|---|
| R11a | A CNN of the same architecture trained with a different augmentation (FTSurrogate) is a strongly diverse member: vs the size-matched control it is +2.31 pp, with 16 subjects up and none down. More base seeds add nothing (−0.17 pp). |
| R11b | Below 4 Hz, the cue-locked signal is mostly ocular, so it is excluded as a source of members. |
| R11c | A finer spectral split of the same 0.5–2.5 s window (filter-bank TS+LR) adds little (+0.46 pp). Spectral diversity *within the same window* is nearly exhausted. |

## Candidates

| Idea | Mechanism | Coherence | Unavailability | F |
|---|---|---|---|---|
| **Augmentation zoo** (diverse deep ensembles in ML; "different invariances, different errors") | 4th member = mean of ShallowConvNets trained with FrequencyShift, SmoothTimeMask and TimeReverse (seed 0 each). These three were neutral as *single* models in R8, which is irrelevant if what matters is that they err differently. ChannelsDropout is excluded a priori because it lost −2.5 pp alone. | 0.6. Direct extension of R11a. Risk: the three are closer to the base CNN than FTSurrogate is, because they leave the spectrum intact. | 0.5. In vision, augmentation-diverse ensembles are known; for EEG/MI we found none (searched in R11). | **0.55** |
| **Late-window TS+LR** (temporal diversity; multi-window FBRTS lineage) | 4th member = TS+LR on 2.0–4.0 s (TS+LR uses 0.5–2.5 s). The window is changed only. | 0.5. R11c says same-window spectral splitting is exhausted, so the time axis is the untouched one. Late MI is weaker and more fatigue-prone (D5). | 0.3. Multi-time-window Riemannian methods exist (FBRTS). | **0.40** |
| Re-weighting (e.g. 0.5 on FTS) | Choose weights a priori. | 0.3. No fact supports any particular weight. | 0.1. | 0.20 |
| Test-time FTSurrogate (TTA) | Average predictions over phase-randomised copies of each test trial. | 0.45. | 0.4. | 0.43, but the CNN weights were not saved; every model would have to be retrained (deferred). |
| Laplacian / CSD-input CNN | Spatial diversity. | 0.4. | 0.3. | 0.35, and impossible on IV-2b with 3 channels. |

## Chosen for R12

1. The augmentation zoo (needs 27 new IV-2b fits; IV-2a is cached from R8).
2. Late-window TS+LR.

Each is a 4th member at weight 1/4. The size-matched control's 4th member is the mean of the
existing families' unused seeds: base CNN seeds 3 and 4, and FTSurrogate seed 2.
