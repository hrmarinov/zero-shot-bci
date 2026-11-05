# Ideation round 5: a new axis — test-time FTSurrogate

**Date:** 2026-11-05. **Why a new axis:** R12 showed that adding ensemble members has saturated
(+0.40 pp, n.s.; more seeds −0.18 pp). R12 named the one candidate left with a decent score:
test-time augmentation (TTA). It needs the CNNs retrained with their weights in memory, so this
round retrains them.

**Idea (TTA from computer vision; phase surrogates from nonlinear time-series analysis).** For each
test trial, predict on the original plus K = 8 FTSurrogate copies of it (phase noise 0.9, the
training setting, applied with probability 1). Average the 9 softmax outputs.

| | Coherence | Unavailability | F |
|---|---|---|---|
| TTA on the FTSurrogate members | 0.6. These models were trained to be invariant to phase randomisation, so averaging over phase surrogates removes the trial-specific phase "noise" they still respond to. F7 says the decoder is variance-limited. R11 says phase randomisation changes the errors. | 0.5. TTA appears in EEG for sleep and seizure work; we found no MI study that uses surrogate-phase TTA with a cross-session ensemble. | **0.55** |
| TTA on all CNN members (base too) | 0.45. The base models never saw surrogates, so the copies are out of distribution for them. If it still helps, the effect is variance reduction, not learned invariance. | 0.5. | 0.48 |

Rejected: TTA with FrequencyShift or TimeMask copies, because those transforms produced members that
were not diverse in R12.

**Arms.** `tta_fts` (TTA on the FTSurrogate members only) and `tta_all` (TTA on every CNN member),
both against B, the R11 best. The member set does not change, so a member-count control does not
apply. The "more members" control was already measured in R12 (−0.18 pp).
