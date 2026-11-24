> Raw reviewer note. Written by a subagent. Citations were checked only where the text says so; treat the rest as unverified.

# Council: alien-space lens (8-10 leads from fields that solved the same abstract problem)

Verification legend. VERIFIED-LOCAL = found in the repo's .venv (braindecode, pyriemann 0.12). UNVERIFIED = from my
memory of the literature, no web check done; confirm before building. Nothing here is a number I measured.
Bars: Lee2019 n=54 MDE ~1.9 pp (project figure); IV-2a n=9 MDE ~3 pp. Run screens on Lee2019 first, so
effects of 2 pp are visible. Every test: paired per-subject, same seeds as R11 baseline, params fixed from source, never tuned on eval.

Already closed, not repeated: SPD shrinkage, ComBat, CPD/ICP, flow, wavelets, protos, label-shift EM, rotation family,
Tent/T3A/CoTTA, AdaBN, TTA, distillation, hard pseudo-label self-training (did not replicate on IV-2b), SWA/run-wise epoch selection (screened).

Pattern across the closures: all are "move the distribution". The D4 finding (+3 pp per doubling, other-day trials worth little)
says the bottleneck is estimator variance from few same-session trials. So most leads below attack variance, not shift.

## 1. Dawid-Skene member aggregation (crowdsourcing / ASR ROVER system combination)
- Abstract problem (d): combine diverse noisy voters without equal weights. A16 (stacking) and per-subject member weights are listed untouched in the audit.
- Library: crowd-kit `DawidSkene` (Toloka) UNVERIFIED; trivial to hand-code (EM on K x K confusion matrices, ~40 lines numpy).
- Mapping: members (SCN, FTS-SCN, TS-LR, and TCFormer as a 4th, strong on subj 2,4,6, weak on 5,7) are annotators; fit a per-subject confusion matrix per member with a Dirichlet prior pulled to a pooled-within-subject-runs prior, using leave-one-run-out cross-fitted calibration predictions. Combine by log-likelihood, not by averaging probabilities.
- Differs: no cross-subject comparison; the weights are class-conditional (feet/tongue are the hard classes, D4), not a scalar weight.
- Cost: CPU minutes once cross-fitted member outputs exist; TCFormer member = one extra training (~hours on a 4060; it is a small model).
- Test: Lee2019 (members: R11 three + TCFormer if run; else R11 three). Metric: mean accuracy vs R11 equal-weight. Rule: adopt if >= +1.0 pp and >= 34/54 up (sign test p<0.05); kill if < +0.5 pp. MDE 1.9 pp is a screen bar; confirm on IV-2b.
- Learn: if it passes, the ensemble gap is weighting; if not, the members are redundant and TCFormer's wins on 2,4,6 need a different explanation (recipe, not diversity).

## 2. Robust covariance (Tyler/Huber/Student M-estimators) (radar/sonar, compound-Gaussian clutter)
- Abstract (a, c): heavy-tailed, non-stationary noise corrupts a few-sample covariance estimate; radar uses scale-invariant M-estimators (Tyler) so loud bursts cannot dominate.
- Library: `pyriemann.estimation.Covariances(estimator='hub'|'stu'|'tyl')`; VERIFIED-LOCAL (pyriemann/geometry/covariance.py wraps hub, stu, tyl).
- Mapping: EEG trials contain EMG/ocular bursts; the per-trial SCM and per-session alignment mean inherit that. Replace both by Tyler (trial) and a Tyler-style session reference. Only the TS-LR member and alignment reference change, so it is also a cheap input fix for the CNNs (whitening matrix from robust mean).
- Differs: within-subject estimator, not a shrinkage toward a population (that is what failed). A12 (alignment-estimator variants) is untested per the audit.
- Cost: CPU, ~1-2 h for 54 subjects.
- Test: Lee2019, TS-LR + RA pipeline, SCM vs Tyler vs Huber (3 arms, Holm). Kill: < +0.5 pp for the best arm. Adopt: >= +1 pp and >= 70 % subjects non-negative. Also report the answer in the weak-third.
- Learn: whether artefact-heavy trials, not shift, cost accuracy; if yes, it also tells us TCFormer's broadband recipe is robust to that.

## 3. FixMatch / UDA consistency training on the unlabelled eval session (semi-supervised vision, speech noisy-student)
- Abstract (a, c): few labels, plenty of unlabelled data from the same distribution; use invariance, not hard labels.
- Library: no dependency; ~60 lines. Weak view = identity, strong view = braindecode FTSurrogate + SmoothTimeMask + FrequencyShift (VERIFIED-LOCAL classes); loss = KL on trials whose confidence > tau (FixMatch, Sohn et al. 2020 UNVERIFIED id; UDA Xie et al. 2020 UNVERIFIED).
- Differs: hard pseudo-label refits failed to replicate (R6/R7). Consistency uses the augmentation prior that already works (+4 pp, R8) as a regulariser on the target trials; no label is ever committed.
- Cost: ~2x one ShallowConvNet training per subject per seed; Lee2019 54 subj x 3 seeds ~ one overnight on the 4060.
- Test: Lee2019 session 2 unlabelled used transductively (same status as RA). Compare FTS-SCN +/- consistency, single model and 3-seed. Adopt: >= +1.0 pp single model and >= +0.7 pp on the ensemble; kill otherwise. Also check the late-minus-early run gap (D5).
- Learn: whether unlabelled same-session trials can substitute for labelled ones (D4 says same-session count is the lever).

## 4. V-REx / GroupDRO with runs as environments (causal-invariant learning, IRM family)
- Abstract (b, e): find predictors whose risk is stable across environments; nuisance correlated with the label is down-weighted.
- Library: DomainBed (facebookresearch) implements IRM, VREx, GroupDRO UNVERIFIED; the VREx loss is ~5 lines: loss = mean(R_e) + beta * var(R_e).
- Mapping: environments = the 6 runs of the calibration session within ONE subject; penalise variance of per-run risk. Targets F4/F5 (within-session decline with a state variable) without conditioning on it.
- Differs: never compares subjects; not a tracker; uses an invariance penalty instead of re-centring.
- Cost: negligible over baseline training.
- Test: IV-2a (9) is under-powered; use Lee2019 sessions-split into 5-6 pseudo-runs per session (check run structure exists; if not, use contiguous 40-trial blocks). Metric: mean accuracy plus late-minus-early gap. Adopt: >= +1 pp. MDE caveat as above.
- Learn: whether the within-session decline is learnable-invariance or irreducible; a null closes the "state" family cheaply.

## 5. Self-supervised pretraining on the subject's own continuous inter-trial EEG (wav2vec 2.0 / masked modelling, ASR)
- Abstract (a): labelled samples are few but unlabelled in-domain signal is plentiful; the trials are only ~(4 s x 288) of a recording that also contains rests, cue-free segments and a full session of continuous data.
- Library: braindecode `MaskEncoding` VERIFIED-LOCAL (masking transform); the SSL objective (masked reconstruction or CPC) hand-written ~80 lines. wav2vec 2.0 (Baevski et al. 2020) UNVERIFIED id.
- Differs: TODO A30 is "untouched" but the audit notes the field-level "joint-embedding" closure was about cross-subject foundation models. This is per-subject, uses windows of the continuous stream, no other subject.
- Cost: moderate. ShallowConvNet-sized encoder, ~30 min per subject pretraining; Lee2019 54 subjects ~ overnight x1.5.
- Test: pretrain on calibration-session continuous EEG minus the labelled trials' labels (no label use), then fine-tune on the same trials; Lee2019 n=54; arm vs FTS-SCN. Adopt >= +1.5 pp at full trials AND the 48-trial calibration burden curve moves >= +3 pp (that is where SSL should help).
- Learn: whether in-session unlabelled data are worth anything beyond the labelled trials; an informative result for D4 either way.

## 6. Mixup / manifold mixup within class and across class (vision/speech)
- Abstract (c): variance reduction through vicinal risk, strongest at tiny sample counts.
- Library: braindecode.augmentation.Mixup VERIFIED-LOCAL. Manifold mixup (Verma et al. 2019) UNVERIFIED id.
- Status: TODO 526 lists mixup as a proposed augmentation; I found no recorded result in docs for it (R7/R8 killed ChannelsDropout/SmoothTimeMask etc.; not mixup). Check the R log before running.
- Mapping: mix aligned trials (post-RA whitening makes covariances comparable, mixed trial is then a physically sensible 'in-between').
- Cost: one extra transform; trivial.
- Test: Lee2019, FTS-SCN +/- Mixup(alpha=0.4 fixed), 3 seeds. Adopt: >= +1 pp; kill at < +0.5 pp or if >= 8 subjects lose > 2 pp. A hard negative would show augmentation is saturated with FTSurrogate.
- Learn: whether augmentations stack or whether FTSurrogate already took the variance reduction.

## 7. Heterogeneous-diversity ensembles (M4/N-BEATS forecasting, FFORMA)
- Abstract (d): M4-competition winners diversified by lookback window, loss, and base model, not by seed (N-BEATS: Oreshkin et al. 2020 UNVERIFIED id).
- Mapping: R11's gain is mostly diversity (+2.9 pp vs a same-size CNN ensemble, Lee2019). Generate more diversity at the input: members with different time windows (0.5-2.5 s, 1-3 s, 2-4 s after cue), different band-passes (4-40, 8-30, 0.5-100 broadband like TCFormer), different loss (CE vs label smoothing).
- Differs: R11 diversifies by model family, not by view. Cheap and orthogonal to #1.
- Cost: 3x member count training; use IV-2a first (9 subjects, 5 seeds ~ hours).
- Test: IV-2a + IV-2b + Lee2019 subset of 18: compare R11 vs R11 + view-diverse members with the same total member count (not more members). Adopt: >= +1 pp on Lee2019-18 and non-negative on IV-2a/b. MDE for 18 subjects ~2.3 pp, so call a +1 pp result only a screen and escalate to n=54.
- Learn: is the broadband view what helps TCFormer on 2,4,6 (answers the T1 open question about why).

## 8. Concept erasure (LEACE / INLP) of the EOG direction (fairness / NLP nuisance removal)
- Abstract (e): nuisance correlated with the label; remove the linear subspace that predicts the nuisance while preserving the task.
- Library: `concept-erasure` (EleutherAI, Belrose et al. 2023) UNVERIFIED; closed-form LEACE is ~30 lines.
- Differs: regressing EOG channels out (T1 eog arm) removes the recorded ocular signal; LEACE removes the learned feature directions that predict an EOG-derived label (e.g. gaze proxy) inside the representation, including cortical leakage.
- Cost: CPU on penultimate features. Only helpful on IV-2b / datasets with EOG.
- Test: IV-2b (n=9 only, MDE 3 pp, so diagnostic): does LEACE on TCFormer features drop its 4-5 pp ocular edge (D6) to R11 level while R11 is unchanged? Metric: TCFormer-with-erasure minus TCFormer. Decision: diagnostic, not accuracy. Learn: how much of any published number is ocular, with a method that does not require EOG channels at test time.

## 9. Multitaper / Welch cross-spectral covariance (seismology, Thomson multitaper; MNE)
- Abstract (a, c): spectral/covariance estimates from short windows have high variance; seismologists use Slepian multitapers.
- Library: `mne.time_frequency.csd_multitaper` and pyriemann `CospCovariances` (the latter pyriemann; VERIFIED-LOCAL module present in pyriemann.estimation, the former UNVERIFIED in this env).
- Mapping: per-trial covariance in the mu/beta band from a multitaper cross-spectrum instead of a band-passed time-domain SCM; lowers estimator variance per trial.
- Differs: does not touch alignment; pure input estimator.
- Cost: CPU ~1 h on Lee2019.
- Test: Lee2019 TS-LR, SCM-bandpass vs multitaper-cospectra (fixed NW=2, K=3). Adopt >= +1 pp; kill at < +0.5 pp. Honest expectation: small, because trial count not window variance is the bottleneck.
- Learn: separates within-trial from between-trial variance in the sample limit.

## 10. Sharpness-Aware Minimisation (SAM) (vision small-data generalisation)
- Abstract (c): flat minima reduce variance when n is small.
- Library: davda54/sam or `torch_optimizer` SAM UNVERIFIED; ~15 lines hand-code.
- Status: SWA was screened and closed (D4/diag_gates note); SAM is the sibling and is not recorded as tried.
- Cost: 2x per-step compute; fine for ShallowConvNet.
- Test: Lee2019, FTS-SCN +/- SAM (rho=0.05 fixed), 3 seeds. Adopt >= +1 pp single model AND >= +0.5 pp inside the full ensemble; kill otherwise.
- Learn: whether the CNN is optimisation-limited or data-limited (D4 predicts the latter; a null confirms).

## Ranking for a first sprint (cost / expected value)
1. #2 Tyler/Huber (CPU, hours; can catch an artefact-driven part of the weak third). 
2. #1 Dawid-Skene weights incl. TCFormer (CPU after members exist; also tests the 'ensemble of TCFormer + R11' question the T1 left open).
3. #6 Mixup and #10 SAM (one-day screens on the same Lee2019 runner).
4. #3 FixMatch-style consistency (the one that most directly exploits unlabelled same-session data).
5. #7 and #5 are GPU-night leads. #4, #8, #9 are diagnostics/low prior.

## What I would NOT do from this lens
Active learning (cued labels are fixed, offline pointless); meta-learning that pools subjects (already fails to converge); domain-adversarial (published null); seismic interferometry (B25 already listed); anything that compares covariances across subjects.
Biggest caveat: most of these are small effects (+0.5 to +2 pp) near the MDE even at n=54. A combined stack (#2 + #1 + #6) as ONE pre-registered hypothesis is the only way a few 1 pp effects become visible; do not confirm each separately.
