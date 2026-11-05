# Domain-neutral abstraction of BCI calibration / "illiteracy" (evidence-informed, 2026-11-04)

ONE SENTENCE: Classify K=4 hidden intents from a C=22-channel noisy multivariate time
series (~1000 samples/trial, information mostly in band-limited power and its spatial
covariance), when each individual passes the signal through an unknown transformation
that differs between individuals and drifts between recording days; only ~288 labelled
examples from day 1 exist, day 2 is unlabelled (all its trials available at once, or
streaming), and for some individuals the intent signal is weak.

WHAT OUR OWN EVIDENCE ALREADY SAYS (constraints on any lead)
- Per-day self-normalisation of second-order statistics (whiten each day by its own mean
  covariance) works: +4.7pp linear model, +11.4pp convnet. Transferring statistics from
  OTHER individuals has failed six times.
- After that whitening, extra per-layer normalisation adds ~0 (the residual day-shift is
  not a per-band gain). What remains unexplained: an orthogonal "rotation" ambiguity left
  by whitening, class-conditional shifts, and within-day drift.
- Best model: subject-specific small convnet 74.7% (5-seed ensemble 78.3%), linear
  CSP+LDA 66.9%. Weak individuals 2/5/6 reach 57–60% with the convnet.
- n=9 individuals → only large effects are detectable; cheap screening must precede any
  confirmatory run.

SUB-PROBLEMS (each phrased without BCI jargon)
P1. Unknown per-device/per-day transform between two unlabelled clouds of covariance
    matrices from the same process — recover it (beyond whitening: rotation/congruence).
P2. Label efficiency — learn from ~100 labels plus a large unlabelled pool from the same
    and a shifted distribution (semi-supervised / transductive / active).
P3. Slow drift within a session — track a non-stationary mapping online without labels.
P4. Weak responders — predict them from baseline data and/or find the representation that
    rescues them (per-individual model selection with tiny data).
P5. Model robustness with tiny data — augmentation/regularisation priors that encode known
    invariances of the signal (time shift, frequency scaling, channel noise).
(P6 co-adaptation of a learning human and an adaptive decoder: not testable offline.)

SEARCH-READY PHRASES
- "calibration transfer between instruments with unknown per-device distortion, unlabelled"
- "unsupervised alignment of two sets of covariance matrices related by unknown congruence"
- "speaker adaptation with no transcribed data from the new speaker"
- "blind equalization of an unknown mixing channel using second/higher-order statistics"
- "batch effect removal when the biological signal of interest is unknown in the new batch"
- "transductive classification under covariate shift with all test inputs available"
- "tracking a slowly drifting linear mapping without labels"
- "predicting non-responders from baseline measurements"
