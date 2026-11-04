# Ideation round: the within-session decline

**Date:** 2026-11-05.

**Method.** This round follows the "Alien Space of Science" scoring rule (arXiv 2603.01092):
F = (1 − β)·coherence + β·unavailability, with β = 0.5. We change one thing. Coherence comes from
this project's own measured facts, not from an LLM judge. Unavailability comes from searching the
EEG/BCI literature for the idea.

All earlier screens scored coherence only (β = 0). That steered them towards ideas the field had
already tried.

**Target problem.** Within the evaluation session, accuracy falls run by run (D5).
- R3 baseline: runs 4–6 are 2.0 pp below runs 1–3.
- The FTSurrogate model from R8 shows the same −1.6 pp decline. Its +4 pp gain is a level shift, not a fix.

## Coherence facts (measured here)

| # | Fact | Source |
|---|---|---|
| F1 | Per-session whitening (RA) works. BN on top of it adds nothing. | R3, AdaBN null |
| F2 | The oracle rotation hurts. Cross-fitted class means have no headroom. | D1, D2 |
| F3 | Smooth covariance trackers fail even though the planted-drift control passes. **So the decline is not a covariance drift.** | R5 |
| F4 | ERD (event-related desynchronisation) shows no trend, yet same-session decodability falls late (E −4.5 pp). | D5 |
| F5 | Pre-cue alpha rises over the session and tracks accuracy (r = −0.44). | D5 |
| F6 | Confidence is calibrated. Pseudo-labels pass on IV-2a but do not replicate on IV-2b. | R6, R7 |
| F7 | The model is variance-limited: the ensemble adds ~3 pp and FTSurrogate adds ~4 pp. | R6, R8 |
| F8 | Mu-peak shifts are ≤ 0.24 Hz. FrequencyShift does nothing. | D4b, R8 |
| F9 | IV-2a runs are class-balanced by design (12 trials per class per run). | dataset |

## Candidates, scored

Coherence and unavailability are judged on a 0–1 scale. F is computed at β = 0.5.

| | Idea (source field) | Mechanism | Coherence | Unavailability (evidence) | F |
|---|---|---|---|---|---|
| **B** | **State-conditioned decoder.** Sources: gain scheduling in control; Lombard-effect conditioning in speech. | Feed the trial's own pre-cue alpha (a scalar) into the network through FiLM (feature-wise scale and shift). The decoder then changes its operating point with the user's state, instead of the input being re-centred. | 0.65. F3 and F4 say the change is in how the task signal is expressed, not in the covariance, so conditioning beats re-centring. F5 supplies the state variable. | 0.8. In EEG, FiLM is used only to condition on subject ID ([arXiv 2509.23247](https://arxiv.org/html/2509.23247v1)). The closest work, [arXiv 2511.07891](https://arxiv.org/abs/2511.07891) (verified), uses the alpha/theta ratio only to **remove** trials (+1.3 pp for ShallowConvNet on IV-2b); its decoder is not conditioned. | **0.73** |
| **A** | **Covariate residualisation.** Sources: genomics surrogate-variable analysis; clinical covariate adjustment. | Within a session, regress the log-variance features on pre-cue alpha and decode the residuals. | 0.6. Same support as B. The risk is that alpha lies inside the 8–30 Hz band, so residualising may remove class signal. | 0.8. No MI paper found. Alpha is known only as a fatigue/attention confound ([Zhou et al. 2021](https://www.frontiersin.org/articles/10.3389/fnhum.2021.701091)). | **0.70** |
| C | **Per-trial noise whitening.** Source: the radar adaptive matched filter. | Whiten each trial with its own pre-cue covariance. | 0.5. It handles state at the level of single trials, but F3 says covariance-type corrections have failed. | 0.7. Rare. One SBCI paper notes that baselining is rarely used in BCIs. | 0.60 |
| E | **Noisy-label-robust training.** Source: ML on noisy labels. | Down-weight disengaged training trials, e.g. with symmetric CE or co-teaching. | 0.5. F7 supports it. | 0.2. Already published for IV-2a: symmetric CE, co-teaching ([arXiv 2112.09243](https://ar5iv.labs.arxiv.org/html/2112.09243)), dataset refinement ([arXiv 2411.10450](https://arxiv.org/html/2411.10450v1)). | 0.35 |
| D | **Label-shift EM per run.** Sources: the Saerens method; criterion shift in signal detection theory. | Re-estimate the class priors in each run. | 0.3. F9: the true priors never shift. Only prediction bias could be corrected, and F6 calibration leaves little of it. | 0.3. SPDIM ([arXiv 2411.07249](https://arxiv.org/pdf/2411.07249)) and others. | 0.30 |
| — | Volatility targeting (finance) | Rescale by a running variance. | 0.1. This is a tracker, which F3 already rejects. | — | rejected |
| — | Selective classification (abstaining) | Abstain on low-confidence trials. | — | — | out of scope: it changes the metric |

**Effect of the scoring rule.** With β = 0, E would rank level with A and B, because it is
coherent, cheap and known to work. Weighting unavailability moves the two state-conditioning ideas
to the top. Both are also the only ideas aimed at F4/F5, the facts that explain the decline.

## Recommendation

1. **G0 gate (cheap, CPU, no E labels).** On session T, using cross-fitted predictions from
   leave-one-run-out:
   - **Question:** does the trial's pre-cue alpha predict whether that trial is classified
     correctly, *beyond run index*?
   - **Test:** compare a logistic model of correctness on (run index, alpha) against one on
     run index only, with a likelihood-ratio test.
   - **If it fails:** A and B are both closed before anything is built.
2. **If G0 passes, screen B.** Use FTSurrogate-augmented ShallowConvNet (the R8 PROMOTE setting) as
   the new base, with one FiLM layer after the temporal convolution driven by a z-scored pre-cue alpha
   scalar. Use 3 seeds and the standard bars (OPEN at ≥ +1 pp, p < 0.10, fewer than 3 subjects
   losing more than 2 pp). The pre-registration must also require that the **late-minus-early run
   gap shrinks**, because that is the claim.
3. **Screen A alongside B on CSP+LDA.** It costs minutes and is the linear control for B.
   - A passes and B fails: a feature correction is enough.
   - B passes and A fails: the interaction matters.

C is held in reserve. D and E are not pursued.
