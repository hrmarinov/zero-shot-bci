# P4 weak responders and P5 robust learning from tiny per-individual datasets: cross-field methods for MI-BCI (BCI IV-2a)

Scope note: this covers precision medicine (responder prediction), psychometrics (IRT), audiology/device fitting, and ML augmentation/regularisation, applied to our setup: 9 subjects, 288 labelled trials from session T, session E unlabelled, subject-specific ShallowConvNet at 74.7% (weak subjects 2/5/6 at 57-60%), 46k params, 300 epochs, no augmentation.
Notation: **[V]** = verified from a fetched or searched source in this session. **[I]** = inferred or from prior knowledge and not re-verified here. "Cross-subject" flags: **NONE** = one subject's data only; **POP-HP** = population used only to set a hyperparameter or train a predictor (allowed, must be flagged); **POP-CORR** = population used as a correction (disallowed).

## Q1. Responder prediction (SMR predictor, other baseline predictors, IRT, precision-medicine responder models). Can any be evaluated with n=9?

### Takeaway
The standard resting-state SMR predictor (Blankertz 2010) explains only ~28% of variance (r=0.53, N=80). With n=9 a correlation can't be estimated usefully: the 95% CI on r=0.53 at n=9 spans roughly -0.2 to 0.88 [I]. Responder prediction is therefore **not screenable as a predictor** here. Its useful offline role is **diagnostic**: use a per-subject spectral peak (e.g. subject 5's 7.3 Hz mu) to choose a representation for that subject. That counts as individualisation (Q2), not prediction.

### Cited Findings
- Blankertz et al. 2010 (NeuroImage 51(4):1303-1309): the SMR predictor is computed from a 2-min eyes-open resting recording over two Laplacian channels (C3, C4). It correlates with feedback performance at r=0.53 in N=80 BCI-naive participants **[V]** — [Korea Univ. Pure record](https://pure.korea.ac.kr/en/publications/neurophysiological-predictor-of-smr-based-bci-performance/); [Fraunhofer record](https://publica.fraunhofer.de/handle/publica/228017)
- A conference poster from the same group gave R²=37%. The journal r=0.53 (≈28% variance) is the number to cite **[V]** — [BMC Neurosci 10(S1):P84](https://bmcneurosci.biomedcentral.com/track/pdf/10.1186/1471-2202-10-S1-P84)
- A later study of 33 novices found psychological predictors explaining ~20% of SMR-BCI variance, and its neurophysiological predictor was not significant. Replication of the SMR predictor is therefore mixed **[V, but whether it tested the exact Blankertz predictor is unconfirmed]** — [Frontiers Hum Neurosci 2014](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2014.00574/pdf)
- About 10-30% of users cannot produce classifiable SMR modulation ("BCI illiteracy", attributed to Blankertz 2010). Another estimate puts roughly 10-20% of subjects as unable to operate a BCI **[V]** — [Frontiers Hum Neurosci 2018](https://www.frontiersin.org/articles/10.3389/fnhum.2018.00059); [MPI Empirical Inference 2011](https://is.mpg.de/ei/publications/6806)
- Ahn & Jun 2015 (J Neurosci Methods 243:103-110) group MI-BCI predictors into personal information, psychological, anatomical and physiological categories. They argue that low performers have a "less-developed brain network" for MI, and that psychological and physiological state drives within-subject variation **[V via secondary citations]** — [PMC5826359](https://pmc.ncbi.nlm.nih.gov/articles/PMC5826359); [Frontiers 2018](https://www.frontiersin.org/articles/10.3389/fnhum.2018.00059)
- Lower resting alpha relative power, and higher theta, are reported in poor MI performers. In alpha up-training neurofeedback, subjects who raised alpha more improved more **[V, snippet level]** — [Frontiers Hum Neurosci 2021 (fnhum.2021.701091)](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.701091/pdf); [PMC7775663](https://pmc.ncbi.nlm.nih.gov/articles/PMC7775663)
- IRT in BCI: Bauer & Gharabaghi combine item response theory with cognitive-load theory. They separate a subject's self-regulation **ability** (an item function) from classifier performance (a rate function). This is the closest existing BCI use of IRT **[V]** — [PMC4329795](https://pmc.ncbi.nlm.nih.gov/articles/PMC4329795)
- Hill, Häuser & Schalk 2014 (J Neural Eng 11:026018) adapt task difficulty along one axis with a staircase to assess BCI performance across the full ability range. This is the psychometric approach to measuring ability without floor or ceiling effects **[V]** — [neurotechcenter.org PDF](https://www.neurotechcenter.org/sites/default/files/misc/A%20general%20method%20for%20assessing%20brain%E2%80%93computer%20interface%20performance%20and%20its%20limitations.pdf)
- An IRT model with ability, item difficulty, discrimination and guessing parameters has been applied to fMRI task design. This shows the formalism ports to neuro tasks **[V]** — [Thomas et al. 2013](https://surfer.nmr.mgh.harvard.edu/ftp/articles/2013/2013_-_Thomas_et_al._-_Psychiatry_Research_Neuroimaging.pdf)

### Method cards
| Method | Field | Mechanism | EEG mapping | Labels (eval) | Cross-subject? | Python | Effort on IV-2a | Kill criterion |
|---|---|---|---|---|---|---|---|---|
| SMR predictor (Blankertz 2010) | BCI / precision-medicine "baseline biomarker" | Height of the resting-spectrum mu/beta peak above a 1/f fit at C3/C4 | 2a has a ~2-min eyes-open/closed EOG calibration block per session [I]; otherwise use the inter-trial/fixation period | none | Correlation across subjects = **POP-HP** (descriptive only) | numpy/scipy Welch + 1/f fit (~50 lines); FOOOF/`specparam` for peak fitting [I] | <1 h | n=9 → not confirmable. Report only. Kill if peak height doesn't rank subjects 2/5/6 in the bottom 4 |
| Per-subject spectral-peak diagnostic | precision medicine "companion diagnostic" | Find the subject's own mu peak frequency and SNR from **their own** T-session data | Choose band edges and filter bank per subject (see Q2) | none on E; T labels optional | **NONE** | `specparam`/scipy | <1 h | Kill if peak-anchored bands don't beat fixed 8-30 Hz in inner CV on T |
| IRT on trials (Rasch / 2PL) | psychometrics | P(correct)=σ(θ_subject − b_trial or b_class) separates subject ability from class/trial difficulty | Treat decoder correctness per trial as a binary response. Estimate class difficulty b_class (e.g. tongue vs feet) and subject ability θ | Needs correctness → uses T-session CV predictions only | Fitting θ jointly across 9 subjects is a population model; use = **POP-HP** (analysis only, not a correction) | `py-irt`, `girth`, or a 10-line logistic GLM in statsmodels [I] | 1-2 h | Analysis tool, not an accuracy lever. Kill if class difficulty doesn't differ across weak/strong subjects |
| Responder classification (precision-medicine ML: baseline features → responder yes/no) | precision medicine | Supervised predictor of outcome from baseline | — | — | **POP-HP** | sklearn | trivial | **Not evaluable with n=9** (LOSO with 9 points, 3 positives). Kill immediately unless pooled with external datasets (e.g. OpenBMI 54 subj, Lee 2019 [I]) |

### Inferences
- **[I]** With 9 subjects and 3 "weak", any responder classifier has 3 positive examples. Leave-one-subject-out accuracy would be noise. Only an external cohort (OpenBMI/Lee2019 54 subjects, Cho2017 52 subjects, both in MOABB) could test a predictor, and that falls outside the "cheap" screen.
- **[I]** IRT's real value here is separating "subject 5 is hard" from "class X (e.g. feet/tongue) is hard for everyone". If weak subjects fail mostly on one class pair, a 2-class or class-weighted rescue becomes possible. It is a cheap analysis from existing CV predictions.
- **[I]** The precision-medicine analogue that **does** transfer is the "companion diagnostic → stratified treatment" pattern. A per-subject spectral measurement (peak frequency, peak SNR, lateralisation) picks the model variant. That is per-individual, NONE cross-subject, and testable with inner CV.

### Gaps
- I could not verify whether BCI IV-2a's EOG/rest blocks are usable as clean eyes-open rest for the SMR predictor. Check the 2a description PDF.
- I found no study applying a formal Rasch/2PL IRT model to MI-BCI trial-level outcomes. This appears **genuinely untried** (low confidence: one search).
- I found no published SMR-predictor values for the 9 IV-2a subjects.

## Q2. Per-individual model/representation selection with tiny data: inner CV, nested-selection risk, IAF bands, frequency warping, filter banks (FBCSP, FBCNet)

### Takeaway
Filter-bank representations are the established way to absorb per-subject band differences. FBCNet reports 76.2% on 2a, and secondary sources say it degrades far less than DeepConvNet/EEGNet when data shrinks. IAF- or peak-anchored bands make physiological sense, but I found **no published evidence that they rescue poor performers specifically**. Per-subject selection must be done by inner CV on session T only. With 288 trials, the inner-CV standard error is ~±3pp [I], so choose among ≤3-5 candidates to limit winner's-curse optimism.

### Cited Findings
- FBCNet (Mane et al. 2021, arXiv:2104.01233): multi-view (filter-bank) CNN with variance-layer temporal aggregation. Reports 76.20% 4-class accuracy on BCIC-IV-2a. Code at github.com/ravikiran-mane/FBCNet **[V headline]** — [arXiv 2104.01233](https://arxiv.org/abs/2104.01233v1)
- A benchmark aggregator lists FBCNet at 71.45% on 2a, which conflicts with the 76.20% headline. The protocol difference is unknown **[V conflict]** — [sotaverified](https://sotaverified.org/papers/fbcnet-a-multi-view-convolutional-neural)
- A secondary summary says DeepConvNet/EEGNet lost ~35% accuracy when training data was cut to 20%, while FBCNet degraded about as much as FBCSP-SVM, i.e. much less. **Unverified against the paper itself** (the PDF fetch failed) — [alphaxiv overview](https://alphaxiv.org/overview/2104.01233v1)
- FBCNet's per-subject analysis across "varying BCI aptitudes" is shown on OpenBMI, not 2a **[V via summary]** — [alphaxiv overview](https://alphaxiv.org/overview/2104.01233v1)
- Schirrmeister 2017 (Human Brain Mapping 38(11):5391-5420), BCI IV-2a, session-1 train / session-2 test. FBCSP got 68.0% (0-38 Hz) and 67.8% (4-38 Hz). ShallowConvNet scored +5.7pp (→73.7%, p<0.05) and +4.1pp (→71.9%, n.s.). DeepConvNet scored +2.9pp / +2.3pp. Including the <4 Hz band helped the ConvNets **[V]** — [Schirrmeister PhD thesis, Movement Decoding chapter](https://robintibor.github.io/eeg-deep-learning-phd-thesis/MovementDecoding.html)
- Subject-specific optimal frequency and temporal windows are documented. Many studies nevertheless use fixed 8-30 Hz **[V snippet]** — [Frontiers Hum Neurosci 2022 (fnhum.2022.831995)](https://www.frontiersin.org/articles/10.3389/fnhum.2022.831995)
- An IAF-defined high-alpha/low-beta study exists across sessions, but its result was not retrieved **[V existence]** — [Frontiers 2021](https://www.frontiersin.org/articles/10.3389/fnhum.2021.701091/pdf)

### Method cards
| Method | Field | Mechanism | EEG mapping | Labels (eval) | Cross-subject? | Python | Effort | Kill criterion |
|---|---|---|---|---|---|---|---|---|
| Peak-anchored bands (IAF/individual mu frequency) | neurophysiology / audiology "fit to the individual's audiogram" | Shift the band set by (f_peak − 10 Hz), e.g. subject 5: 7.3 Hz peak → mu band ≈ 5.3-11.3 Hz | Change bandpass of input (CSP) or initialise ShallowConvNet temporal filters at the shifted band | none | **NONE** (the shift comes from the subject's own T-session spectrum) | scipy.signal + `specparam` | <2 h (CSP+LDA), ~3 h (convnet ×9) | Kill if subject 5 doesn't gain ≥5pp on E with CSP+LDA, or inner-CV on T doesn't prefer it |
| VTLN-style frequency warping | speech recognition (vocal-tract length normalisation) | Rescale the frequency axis by α so the subject's peak lands at a canonical frequency | Resample in time by α (time-stretch ≡ frequency scale), or warp a filter-bank center frequency. Pick α per subject by maximising T-session inner-CV (as VTLN picks α by likelihood) | none on E (α can even be picked from E's **unlabelled** spectrum peak) | **NONE** | `scipy.signal.resample_poly` | 2-3 h | **Genuinely untried in MI-BCI as far as I found [I]**. Kill if α≠1 doesn't win inner-CV for ≥1 of 2/5/6 |
| FBCSP (+ mutual-info feature selection) | BCI | 9 bands × CSP, MIBIF selects bands per subject | Built-in per-subject band selection | none | **NONE** | MNE `CSP` + sklearn; MOABB has FBCSP pipelines [I] | <1 h | Baseline/reference; compare against our 66.9% CSP+LDA |
| FBCNet | BCI DL | Filter bank → depthwise spatial conv → variance layer | Drop-in subject-specific model | none | **NONE** | official repo (PyTorch); also in `torcheeg` [I] | 2-4 h on RTX 4060 | Kill if weak-subject mean doesn't beat ShallowConvNet's 57-60% by ≥3pp |
| Per-subject model selection by inner CV on T | ML / precision medicine "N-of-1" | k-fold (stratified, time-blocked) on T picks among {ShallowConvNet, FBCNet, CSP+LDA, band variants} | — | none | **NONE** | sklearn `GridSearchCV` | cheap once the models exist | Kill if the selected-per-subject pipeline (nested evaluation) isn't ≥ the best single fixed pipeline on E |

### Inferences
- **[I]** Nested-selection risk: with 288 trials, a 5-fold inner-CV accuracy has SE ≈ sqrt(0.6·0.4/288) ≈ 2.9pp. Picking the max of K candidates inflates the inner score by ~σ·E[max of K normals] (≈1.16σ for K=5). Keep K ≤ 5. Report E-session accuracy of the **selected** pipeline, not the best-on-E pipeline. Use time-blocked folds: 2a T-session has 6 runs, so leave-one-run-out gives 6 folds and respects within-session drift.
- **[I]** Since 2a session E is fully available unlabelled, a subject's mu peak can be re-estimated on E to set α/band shift. This is label-free and per-subject. It also targets the drift of peak frequency between days, which is unknown in size.
- **[I]** The Schirrmeister result that including 0-4 Hz helps ConvNets suggests checking our input band. If our pipeline uses 4-38 Hz, try 0.5-38 Hz.
- **[I]** ShallowConvNet's first temporal conv (25 taps at 250 Hz) can learn a 7 Hz filter in principle, so the band shift matters more for CSP-type pipelines than for the convnet. Test on CSP+LDA first (fast) as a mechanistic check.

### Gaps
- Per-subject FBCNet accuracies on 2a, especially subjects 2/5/6, were not retrieved (the PDF was not parseable). Check Table in arXiv 2104.01233 or rerun the official code.
- I found no evidence that IAF-anchored band individualisation specifically helps poor performers. VTLN-style warping for EEG MI was not found in the literature (one search only), so treat it as untried.
- The original FBCSP 2a competition result (commonly cited kappa ≈ 0.57) was not verified here.

## Q3. Data-augmentation and regularisation priors for EEG, with numbers; cropped training, ensembling/SWA, distillation; which are cheap?

### Takeaway
Rommel et al. 2022 is the key directly applicable evidence. It uses **the same protocol as ours** (within-subject, 2a session 1 → session 2, Braindecode convnet). **FTSurrogate** (best overall), **TimeReverse** (best low-data), **SmoothTimeMask**, **ChannelsDropout** and **FrequencyShift** help. **ChannelsShuffle/ChannelsSymmetry** hurt, and **SignFlip** is a no-op for ShallowConvNet because of its squaring layer. All are one-line `braindecode.augmentation` transforms, so a 9-subject × ~6-augmentation screen fits in hours on an RTX 4060. This is the cheapest and best-evidenced lever for P5.

### Cited Findings
- Rommel, Paillard, Moreau, Gramfort 2022 (J Neural Eng, DOI 10.1088/1741-2552/aca220) compare 13 augmentations on 2 tasks. "No single best augmentation" exists. Up to 45% accuracy improvement in low-data regimes **[V]** — [arXiv 2206.14483](https://arxiv.org/abs/2206.14483)
- Protocol on BCI IV-2a: within-subject, train on session 1 (or a fraction), test on session 2. The model is a Braindecode generic convnet described as CSP-inspired; ShallowConvNet is likely but not confirmed in the text **[V]** — [arXiv 2206.14483 HTML](https://arxiv.org/html/2206.14483v2)
- Selected hyperparameters and effects on 2a **[V]** — [arXiv 2206.14483 HTML](https://arxiv.org/html/2206.14483v2):
  - FTSurrogate (Δφ_max = 9π/10) is the best frequency augmentation. It gives 45% relative gain at a training subset 2⁻³ of the full set. Half the data + FTSurrogate beat the full-data baseline. Gains held at all training sizes.
  - TimeReverse is the best time augmentation in low-data regimes, up to +25% at fraction 2⁻⁴.
  - SmoothTimeMask (mask 1.6 s) is the best time augmentation with larger sets, up to 20% relative gain.
  - ChannelsDropout (selected p_drop = 1 as reported, possibly a parameterisation quirk) is the best spatial augmentation on larger sets, up to 20-25% (the paper is internally inconsistent).
  - FrequencyShift (Δf_max = 2.7 Hz) is sometimes comparable to FTSurrogate. BandstopFilter (0.4 Hz bandwidth) gives up to 17% in low data.
  - GaussianNoise (σ = 0.16) brought no robust gain. SensorsRotation was mostly n.s., though Z-axis helped in ≥75% of per-class cases.
  - SignFlip equals the baseline (squaring layer is sign-invariant). ChannelsSymmetry hurts L/R hand classes. ChannelsShuffle is consistently harmful.
  - BCI benefits more from augmentation than sleep staging does. Gains are reported relative to the no-augmentation baseline on balanced accuracy, mostly at sub-full training fractions.
- Mixup was **not** tested by Rommel on 2a. The only MI mixup paper found (Alwasiti & Yusoff 2022) uses 120 trials per subject with no baseline, which is not evidence **[V]** — [search summary of PMC9788676 / arXiv 2206.14483](https://pmc.ncbi.nlm.nih.gov/articles/PMC9788676)
- CovMix (Zoumpourlis & Patras 2022) mixes covariance structure as a regulariser for MI decoding. It is an alternative to mixup that respects the spatial-covariance nature of MI **[V existence, numbers not retrieved]** — [QMUL accepted manuscript](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/77747/Zoumpourlis%20CovMix:%20Covariance%20Mixing%20Regularization%20for%20Motor%20Imagery%20Decoding%202022%20Accepted.pdf?sequence=2)
- Amplitude-perturbation augmentation on ShallowConvNet/2a reproduced ~74% for the baseline, consistent with the original 73.7% **[V]** — [arXiv 1811.02353](https://arxiv.org/pdf/1811.02353)
- Segmentation and recombination (Lotte 2015) builds artificial trials by concatenating time segments (or time-frequency segments) drawn from different trials of the **same class**, keeping temporal order. It has been applied to MI with small data **[V, secondary]** — [arXiv 2511.00369](https://arxiv.org/pdf/2511.00369)
- Channel-level brain-area recombination (Pei et al. 2021) swaps left/right hemisphere channel groups between same-class trials. EEGNet trained this way gained up to 8.3% over CSP-SVM (p<0.01) **[V, snippet]** — [UiTM record](https://openaccess.library.uitm.edu.my/Record/doaj-8ce8f7360dff42258fd4fc42a6aabdb7/Description)
- EMD-based artificial trials from few calibration trials improve LDA/logistic regression **[V existence]** — [PMC8824327](https://rcastoragev2.blob.core.windows.net/2ac6391eabb6eff790036caf2c5caa7a/PMC8824327.pdf)
- Cropped training (Schirrmeister 2017): all 2a ConvNet results in the main table use cropped training. Cropped vs trial-wise was significant only for DeepConvNet on High-Gamma. For ShallowConvNet, differences were n.s. and highly subject-variable **[V]** — [Schirrmeister thesis](https://robintibor.github.io/eeg-deep-learning-phd-thesis/MovementDecoding.html)
- SWA: no EEG/MI paper using stochastic weight averaging was found. Deep ensembles for MI exist (cross-subject homogeneous ensemble, 64.75% on 2a cross-subject), which is not our within-subject setting **[V]** — [IEEE Access 2025 record](https://library.kab.ac.ug/Record/doaj-art-080e7b0acf5347afa443e33448ed1e19/Description)

### Method cards (all within-subject, labels only from T, **NONE** cross-subject unless noted)
| Method | Field | Mechanism / invariance encoded | Python | Est. cost (9 subj, 1 seed, RTX 4060) [I] | Published 2a evidence | Kill criterion |
|---|---|---|---|---|---|---|
| FTSurrogate | ML aug / surrogate-data stats | Randomise Fourier phases, keep the power spectrum. The class info is in band power | `braindecode.augmentation.FTSurrogate(probability=0.5, phase_noise_magnitude=0.9, channel_indep=False)` | ~1-2 h | Best frequency aug; 45% rel. at 2⁻³ data **[V]** | <+2pp mean on E at full data, or no gain on 2/5/6 |
| TimeReverse | ML aug | Band power is time-direction invariant | `TimeReverse(0.5)` | ~1-2 h | +25% at 2⁻⁴ **[V]** | same |
| SmoothTimeMask | ML aug (SpecAugment-like) | Robust to transient artifacts | `SmoothTimeMask(0.5, mask_len_samples=400)` (1.6 s at 250 Hz) | ~1-2 h | ≤20% rel. **[V]** | same |
| FrequencyShift | ML aug ≈ VTLN prior | Small spectral-peak shifts are label-preserving. **Directly addresses subject 5's 7.3 Hz peak and day-to-day peak drift [I]** | `FrequencyShift(0.5, sfreq=250, max_delta_freq=2.7)` | ~1-2 h | comparable to FTSurrogate **[V]** | same |
| ChannelsDropout | ML aug | Robust to bad electrodes and spatial noise | `ChannelsDropout(0.5, p_drop=0.2)` [I: Rommel's "p_drop=1" looks like a scaled-parameter quirk] | ~1-2 h | best spatial on larger sets **[V]** | same |
| BandstopFilter | ML aug | Not over-relying on one narrow band | `BandstopFilter(0.5, sfreq=250, bandwidth=0.4)` | ~1-2 h | ≤17% low data **[V]** | same |
| SignFlip | ML aug | — | — | skip | **no-op for ShallowConvNet [V]** | skip |
| ChannelsShuffle / ChannelsSymmetry | ML aug | — | — | skip | harmful on 2a **[V]** | skip |
| Mixup / CovMix | ML regularisation | Interpolate trials and labels (Mixup) or covariances (CovMix) | `braindecode.augmentation.Mixup` + `mixup_criterion` [I] | ~1-2 h | **no clean 2a within-subject number found** | Screen only after the above |
| Segmentation & recombination (Lotte 2015) | BCI aug | Same-class segment shuffling across trials at fixed time positions | ~30 lines numpy; `braindecode.augmentation.SegmentationReconstruction` exists in recent braindecode [I] | ~1-2 h | MI gains reported, no 2a ShallowConvNet number **[V partial]** | same |
| Cropped training | DL training scheme | Many shifted windows per trial (time-shift invariance), with cropped decoding | `braindecode` `EEGClassifier(cropped=True)` + `CroppedLoss` [I] | ~2-3 h | Used for the 73.7% result; n.s. vs trial-wise for Shallow **[V]** | Kill if not ≥ trial-wise +1pp mean |
| Seed ensemble (already 78.3% for 5 seeds) | ML | Variance reduction | existing | done | our own: +3.6pp | — |
| SWA | ML | Flat-minimum averaging over the last epochs, single model | `torch.optim.swa_utils.AveragedModel` + `SWALR` + `update_bn` (BN stats need recomputing, which **interacts with per-session BN**) [I] | ~1 h, ≈ free | **none found in EEG** → untried | Kill if < half the gain of the 5-seed ensemble |
| Distillation from ensemble | ML | Single student matches the ensemble's soft labels. Can also use **unlabelled E trials as transfer set** (transductive, NONE cross-subject) | ~40 lines PyTorch (KL on temperature-softened logits) [I] | ~2 h | none found for within-subject MI → untried | Kill if student < ensemble − 2pp |
| Early stopping / fewer epochs / weight decay | ML regularisation | 300 epochs on 288 trials overfits | inner run-wise CV on T to choose epochs | ~1 h | — | — |

### Inferences
- **[I]** Rommel's gains are mostly reported at reduced training fractions. At full 288 trials the gain is smaller but non-negative for FTSurrogate ("gains held at all sizes"). Realistic expectation at full data is +1-5pp mean. Weak subjects (high noise, low SNR) plausibly benefit more, but that is unverified.
- **[I]** FrequencyShift is the augmentation analogue of VTLN/IAF individualisation. Training with ±2-3 Hz shifts makes the convnet tolerant to subject 5's low peak and to between-session peak drift. Pair it with Q2's peak-anchored bands as two routes to the same hypothesis: test both and see whether they are redundant.
- **[I]** Recommended screen (cheap, ~1 day GPU): baseline vs {FTSurrogate, TimeReverse, SmoothTimeMask, FrequencyShift, ChannelsDropout, FTSurrogate+FrequencyShift+SmoothTimeMask}. Use 9 subjects × 2-3 seeds, Rommel's hyperparameters fixed a priori (no tuning on E), and keep our per-session RA/BN pipeline. Pre-register: promote only if mean ΔE ≥ +2pp **and** ≥6/9 subjects non-negative. Report subjects 2/5/6 separately.
- **[I]** SWA and BN: our model uses per-session BatchNorm. After SWA, BN statistics must be re-estimated (`update_bn`) on the T session for T and on E for E. Otherwise SWA can silently break the per-session normalisation that gave +11.4pp.
- **[I]** Distilling the 5-seed ensemble (78.3%) into one student using the unlabelled E trials is a transductive self-training step. It overlaps P2 (semi-supervised), so coordinate with that note.
- **[I]** Rommel's reported "per-class" augmentation effects mean an augmentation can help tongue/feet while hurting L/R hands (e.g. ChannelsSymmetry). Combine with Q1's IRT class-difficulty analysis to see whether weak subjects fail on specific class pairs.

### Gaps
- No verified number for Mixup, SWA, or ensemble distillation in within-subject 2a.
- Rommel's absolute accuracies per fraction and per subject were not extracted. The text only gives relative gains, and the learning-curve figures need reading. The last ~9k characters of the HTML were not read.
- Whether Rommel's network is exactly ShallowConvNet is unconfirmed.
- braindecode class names and signatures listed above are from prior knowledge, not checked against current docs. Verify against the installed braindecode version (repo commit 7ced960 made braindecode importable).

## Q4. Cochlear-implant / hearing-aid fitting analogues: closed-loop parameter fitting for poor responders, anything transferable offline?

### Takeaway
CI fitting's main transferable idea is **outcome-targeted, rule- or learning-based adjustment of a per-individual parameter map**. FOX (Fitting to Outcomes eXpert) measures outcomes, compares them to targets, and edits the map. It raised targets achieved by 24% in 255 recipients, and showed mixed but hopeful results in 2 poor performers. Offline, this reduces to **per-subject hyperparameter/representation search against a held-out-run objective on session T**, i.e. Q2's inner-CV selection. The genuinely closed-loop part (human adapts to the map) maps to P6 and is not testable offline.

### Cited Findings
- FOX (computer-assisted CI fitting): a retrospective study of 255 recipients reported a significant 24% (p<0.001) increase in outcome targets achieved. Automated starting maps already hit 57% of 22 targets. Mostly single-centre (Antwerp), and the developer is affiliated (Otoconsult) **[V]** — [PMC3976838](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3976838/)
- In a 2019 case report of 2 poor-performing adults, one had no open-set word recognition after 19 manual fitting sessions. AI-assisted fitting (probabilistic, self-learning FOX) gave mixed results that were judged improvement-capable. Hypothesis-generating only **[V]** — [search summary; UCLouvain repository](https://research.dial.uclouvain.be/bitstreams/7b62cae1-0a73-4ca3-9650-6ff7f344dd07/download)
- EU projects OPTI-FOX/Hearing Minds funded this work and claim automated fitting is faster than manual. These are project-reported claims **[V]** — [CORDIS 262266](https://cordis.europa.eu/project/id/262266/reporting); [CORDIS article 92129](https://www.cordis.europa.eu/article/id/92129-optimising-cochlear-implant-functionality)

### Method cards
| Method | Field | Mechanism | EEG mapping | Labels (eval) | Cross-subject? | Effort | Kill criterion |
|---|---|---|---|---|---|---|---|
| Outcome-targeted map fitting (FOX-style) | audiology | Measure outcome → apply rules/learned policy → adjust per-individual map → re-measure | Per-subject loop over a small "map" (band edges, time window e.g. 0.5-2.5 s vs 1-4 s, spatial filter count, aug set). Objective = leave-one-run-out accuracy on T. Stop when no improvement | none (T only) | **NONE**. A rule set learned from other subjects ("if peak <8 Hz → shift bands") = **POP-HP**, flag it | 2-4 h for CSP-based, more for convnet | Must beat the fixed pipeline on E for weak subjects with selection done blind to E |
| Population-derived "starting map" | audiology (FOX starting maps) | Population prior as initialisation, then individual fitting | Default hyperparameters from the other 8 subjects (LOSO), then per-subject refinement | none | **POP-HP** (allowed, flag) | cheap | Kill if the starting map + refinement isn't > refinement from a generic default |

### Inferences
- **[I]** The "19 manual sessions failed" poor-performer CI case parallels subject 2/5/6. Audiology's answer was to search the parameter space more systematically against measured outcomes, not to change the patient. The offline BCI equivalent is a disciplined per-subject search with nested CV, and its limit is the 288-trial noise floor (Q2 inference).
- **[I]** Hearing-aid fitting prescriptions (NAL, DSL) map an audiogram to gains through a formula. The BCI analogue is a formula from the subject's resting/T-session spectrum (peak frequency, peak SNR) to band edges and filter-bank weights. This is cheap, deterministic, NONE cross-subject when the formula is physiology-based, and appears untried as an explicit "prescription" in MI-BCI [I].

### Gaps
- No controlled trial of automated CI fitting in poor performers was found.
- I found no direct BCI literature framing decoder hyperparameter fitting as a "prescription formula" from baseline physiology.

## Q5. Established in EEG vs genuinely untried (summary for prioritisation)

### Takeaway
Augmentation (Rommel), filter banks (FBCSP/FBCNet) and seed ensembling are **established** with 2a numbers. Peak-anchored bands are physiologically motivated but **unproven for poor performers**. VTLN-style frequency warping, SWA, ensemble-to-student distillation using unlabelled E trials, IRT decomposition of trial outcomes, and "fitting prescriptions" are **untried in within-subject MI as far as these searches found**. Responder prediction is established but **cannot be validated with n=9**.

### Cited Findings
- Established with 2a numbers: augmentation within-subject session1→session2 **[V]** — [Rommel 2022](https://arxiv.org/html/2206.14483v2). ShallowConvNet 73.7% vs FBCSP 68.0% **[V]** — [Schirrmeister thesis](https://robintibor.github.io/eeg-deep-learning-phd-thesis/MovementDecoding.html). FBCNet 76.2% **[V]** — [arXiv 2104.01233](https://arxiv.org/abs/2104.01233v1)
- Established predictor, mixed replication: SMR predictor r=0.53 **[V]** — [Pure record](https://pure.korea.ac.kr/en/publications/neurophysiological-predictor-of-smr-based-bci-performance/); [Frontiers 2014](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2014.00574/pdf)
- IRT in BCI exists only in a neurofeedback ability framework, not as trial-level decoding analysis **[V]** — [PMC4329795](https://pmc.ncbi.nlm.nih.gov/articles/PMC4329795)

### Inferences (prioritised cheap screen, all **NONE** cross-subject unless flagged) [I]
1. **Augmentation screen** (FTSurrogate, TimeReverse, SmoothTimeMask, FrequencyShift, ChannelsDropout, combo). Best evidence, one-line code, ~1 day GPU. Expected +1-5pp mean. Watch 2/5/6.
2. **Training-length / early-stopping via leave-one-run-out on T**, plus **SWA**: near free, addresses the documented overfitting (300 epochs on 288 trials).
3. **Peak-anchored bands / VTLN warping for subject 5** (7.3 Hz peak): <2 h on CSP+LDA as a mechanistic test, then the convnet. Also check including 0-4 Hz.
4. **FBCNet** subject-specific with existing code: 2-4 h.
5. **Ensemble → single student distillation on unlabelled E**: overlaps P2, ~2 h.
6. **IRT/class-difficulty analysis** of existing CV predictions: 1 h, diagnostic only.
7. **Responder prediction (SMR predictor)**: compute and report descriptively. Do not claim predictive validity with n=9 (**POP-HP** if correlated across subjects).

### Gaps
- Whether augmentation gains concentrate in weak subjects is unknown. Rommel reports averages; per-subject breakdowns were not retrieved.
- Search coverage was limited (~15 queries). The "untried" labels mean "not found in these searches", not an exhaustive literature claim.
