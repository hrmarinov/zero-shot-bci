# Ideation round 3: ensemble members that err differently from the CNN

**Date:** 2026-11-05. **Scoring:** F = 0.5·coherence + 0.5·unavailability (same rule as
`reports/Alien space ideation within-session decline.md`). Coherence comes from this project's
measured facts; unavailability from a literature search for the idea.

**Target.** R10: averaging TS+LR into the CNN ensemble adds ~+1 pp *beyond* a size-matched CNN
ensemble, so the second member's value is that it errs on different trials. The question for
round 3 is which extra member carries information the CNN (and TS+LR) do not use.

## Facts used for coherence

| # | Fact | Source |
|---|---|---|
| R10a | A weaker but differently-biased member (TS+LR, −6.0/−1.2 pp vs CNN alone) still adds +1.06 pp to a 3-seed CNN ensemble, 15/18 up. | R10 |
| R10b | Adding CNN seeds saturates fast: 2 → 3 seeds adds +1.1 pp (IV-2a) and +0.1 pp (IV-2b). | R10 |
| F7 | The decoder is variance-limited (ensemble ~+3 pp; FTSurrogate +4 pp on IV-2a). | R6, R8 |
| R9 | FTSurrogate's single-model gain shrinks to +0.7 pp on IV-2b. | R9 |
| IN | The CNN sees 4–38 Hz and its squaring/log layer discards phase; TS+LR sees 8–30 Hz covariance. Nothing in the stack sees < 4 Hz or the time course. | design |

## Candidates

| Idea (source field) | Mechanism | Coherence | Unavailability (evidence) | F |
|---|---|---|---|---|
| **Low-frequency time-course member** (ERP decoding in BCI; MRCP work in rehab) | 0.5–4 Hz waveform, 0.5–2.5 s, averaged into 100 ms bins, per-session z-scored, shrinkage LDA. Uses information **no current member can see** (phase-locked < 4 Hz activity). | 0.55. IN + R10a predict maximal error decorrelation; unknown whether < 4 Hz carries class signal in cued MI. Risk: eye movements toward the cue arrow leak into frontal channels (a stimulus confound, flagged). | 0.65. MRCP is studied for executed/attempted movement (e.g. [O'Keeffe et al.](https://par.nsf.gov/servlets/purl/10540502)); no paper found that fuses a δ-band time-domain decoder with ERD decoders on IV-2a/2b as a late-fusion member. Time-domain statistics (Hjorth etc.) on IV-2a exist ([Frontiers 2024](https://frontiersin.org/articles/10.3389/fnhum.2024.1525139/full)) but are band-power-like. | **0.60** |
| **FTSurrogate CNN seeds as a member** (augmentation-diverse ensembles) | Add a 2-seed FTSurrogate ShallowConvNet ensemble as a third member. Same architecture, different invariance. | 0.6. F7 and R8 support; R9 says the single-model gain is small on IV-2b, but member *diversity* is a different question. | 0.4. Augmentation-diverse deep ensembles are known in ML; not specifically tested in MI stacks. | **0.50** |
| **Filter-bank TS+LR** (FBCSP lineage) | Six sub-bands 4–38 Hz, per-band RA + tangent space, concatenated, LR. Finer spectral resolution than 8–30 Hz TS+LR. | 0.55. Covers 4–8 and 30–38 Hz that TS+LR ignores; but the CNN already covers 4–38 Hz, so diversity vs the CNN is partial. | 0.3. FBRTS ([Essex repo](https://repository.essex.ac.uk/34091/1/Feature_Extraction_Method_Based_on_Filter_Banks_and_Riemannian_Tangent_Space_in_Motor-Imagery_BCI.pdf)), DMFBTSM ([Frontiers 2021](https://www.frontiersin.org/articles/10.3389/fnhum.2021.595723)), DFBRTS ([arXiv 2310.19198](https://arxiv.org/pdf/2310.19198v1.pdf)). | **0.43** |
| TS+LR on wide 4–38 Hz | Same model on the CNN's band. | 0.35. Moves TS+LR *towards* the CNN's information, so it should lower diversity. | 0.2. Standard. | 0.28 |
| FgMDM / MDM | Riemannian minimum distance. | 0.3. Same covariance information as TS+LR, usually weaker; errors correlated with TS+LR. | 0.1. pyriemann standard. | 0.20 |
| CSP+LDA | Log-variance of spatial filters. | 0.2. This *is* ShallowConvNet's inductive bias (ShallowConvNet was designed as a learned FBCSP), so its errors should track the CNN's. | 0.1. | 0.15 |
| Weights by T leave-one-run-out | Choose weights on T only. | 0.3. Needs CNN LORO probabilities (≈ 80 extra GPU fits); with 2–3 members the weight surface is flat near 0.5 and T-chosen weights tend to over-trust the member that fits T best. | 0.2. Standard stacking. | 0.25 — deferred |
| Stacking meta-learner on T LORO | LR on member log-probs. | 0.3. Same cost; 240 trials per subject for a meta-learner is little. | 0.2. | 0.25 — deferred |

## Chosen for R11

The three top-F members, each as a **third equally weighted member** (1/3 CNN-ensemble, 1/3 TS+LR,
1/3 new member), against the R10 best, with one size-matched control: 1/3 CNN(seeds 0–2),
1/3 TS+LR, 1/3 CNN(seeds 3–4). Combination rules that need training-session CNN probabilities
are deferred: their coherence is low and their cost is the GPU budget of a full round.
