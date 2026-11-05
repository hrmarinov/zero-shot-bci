# Speech speaker adaptation and neuroimaging/genomics harmonization: which methods transfer to zero-label cross-session EEG motor-imagery decoding

Scope: the BCI IV-2a setup. Session T is labelled (288 trials, 4 classes). Session E is unlabelled. Methods are single-subject only: no single computation may pool two humans. Already closed in this project: per-session whitening (works), AdaBN after whitening (adds about 0), ComBat on tangent features (tied/null), population shrinkage.

Marking used throughout:
- [V] = verified this session from a fetched or searched source.
- [K] = standard literature fact recalled from background knowledge. The URL/DOI is given but the page was not fetched this session, so the report writer should treat it as citation-grade but not re-checked.
- [I] = my inference.

## Q1. Speech adaptation (fMLLR/CMLLR, MAP, VTLN, HEQ, SAT, i/x-vectors, LHUC, confidence-filtered pseudo-labels): EEG analogues and numbers

### Takeaway
The EEG literature has re-invented only the unsupervised, model-free half of speech adaptation: global covariance re-colouring and LDA bias/mean updates. These match CMN/CMVN and global-CMLLR-without-classes, and this project has already exhausted them. Speech has two families that EEG has never tried in their real form:
- Model-based ML adaptation driven by first-pass hypotheses: CMLLR/fMLLR with a full affine transform, estimated with class posteriors under the frozen source model, iterated (EM).
- Low-dimensional per-speaker re-weighting of a trained network: LHUC.

Both are cheap, and both fit the project's residual problem: the rotation/class-conditional shift left after whitening.

### Cited Findings

**Speech-side mechanisms (primary sources)**
- **CMLLR/fMLLR (Gales 1998)** [K]. A single affine transform x' = A x + b per speaker is shared by the means and covariances of all Gaussians. That is equivalent to transforming the features. It is estimated by EM to maximise the likelihood of the adaptation data under the speaker-independent model. In unsupervised mode, the Gaussian occupation posteriors come from first-pass recognition hypotheses. Source: [Gales, "Maximum likelihood linear transformations for HMM-based speech recognition", Computer Speech & Language 12(2), 1998](https://doi.org/10.1006/csla.1998.0043).
- **VTLN (Lee & Rose 1998)** [K]. A single per-speaker frequency-axis warping factor α is chosen by grid search, usually α in about 0.88–1.12 in 0.02 steps. The criterion is maximum likelihood of the warped features under the model, given first-pass transcripts. Source: [Lee & Rose, IEEE Trans. Speech Audio Proc. 6(1), 1998](https://doi.org/10.1109/89.668817).
- **VTLN as one parameter** [V]. VTLN is "a transformation of a spectrum frequency axis" controlled by "one Speaker Dependent parameter … called warping factor". The warp is normally chosen by maximum likelihood because vocal tract length cannot be measured directly. Sources: [Pitz et al., Eurospeech 2001](https://www.isca-archive.org/eurospeech_2001/pitz01_eurospeech.html); [Dognin & El-Jaroudi, Eurospeech 2003](https://www.isca-archive.org/eurospeech_2003/dognin03_eurospeech.pdf).
- **LHUC (Swietojanski, Li & Renals 2016)** [V]. LHUC "linearly re-combines hidden units in a speaker- or environment-dependent manner using small amounts of unsupervised adaptation data". The per-unit amplitude is r = 2·sigmoid(θ), learned per speaker while all other weights are frozen [K]. Findings [V]:
  - LHUC is robust to over-fitting.
  - Test-only LHUC and SAT-LHUC differ by only about 0.1% absolute WER.
  - SAT-LHUC trains speaker-independent and speaker-dependent parts jointly.
  - Bayesian LHUC (BLHUC) is a variant designed against over-fitting.

  Sources: [arXiv 1601.02828](https://arxiv.org/pdf/1601.02828); [BLHUC, SigPort](https://sigport.org/documents/blhuc-bayesian-learning-hidden-unit-contributions-deep-neural-network-speaker-adaptation).
- **MAP adaptation (Gauvain & Lee 1994)** [K]. Each class-conditional Gaussian mean is interpolated between its prior (speaker-independent) value and the adaptation-data mean, weighted by occupation count. Source: [IEEE TSAP 2(2), 1994](https://doi.org/10.1109/89.279278).
- **SAT (Anastasakos et al. 1996)** [K]. SAT trains the canonical model jointly with per-speaker transforms, so the model learns "transform-normalised" data. Source: [ICSLP 1996](https://doi.org/10.1109/ICSLP.1996.607807).
- **Histogram equalisation (de la Torre et al. 2005)** [K]. A per-dimension CDF matching maps each test feature's empirical CDF onto a reference CDF. It is a non-linear generalisation of CMVN. Source: [IEEE TSAP 13(3), 2005](https://doi.org/10.1109/TSA.2005.845805).

**EEG analogues found**
- **Unsupervised EEG Data-Space Adaptation (EEG-DSA), Arvaneh, Guan, Ang & Quek, Neural Computation 2013** [V]. This is the closest EEG analogue of global CMLLR.
  - Mechanism: it minimises KL[N(0, VᵀΣ̄_tV) ‖ N(0, Σ̄_s)] between session-average covariances.
  - Closed form: V_unsup = Σ̄_t^(-1/2) Σ̄_s^(1/2). It is "computed without using any predicted labels" and the authors note it is "similar to what Tomioka et al. (2006) proposed as the normalizing approach". It is therefore whitening-plus-recolouring, which is what this project already does.
  - Supervised variant: matches class-wise average covariances and needs eval labels.
  - Setup: 2-class subset of BCI IV-2a plus a 16-subject dataset; 8–35 Hz, CSP (3+3 filters) + LDA. The first 20 eval trials were used for adaptation, either once ("single") or as a sliding 20-trial window ("continuous").
  - Pooled results across both datasets, accuracy % (single / continuous):

    | Method | All subjects | Subjects with baseline error >30% |
    |---|---|---|
    | Unsupervised EEG-DSA | 72.1 / 73.1 | 65.8 / 66.7 |
    | Supervised EEG-DSA | 72.9 / 73.6 | 65.7 / 66.9 |
    | PMean (LDA bias adaptation) | 71.7 / 72.2 | 63.9 / 64.2 |

  - Significance: every EEG-DSA version beat no-adaptation (p = 0.018, 0.004, 0.006, 0.002). Supervised vs unsupervised EEG-DSA was not significant (p = 0.10 / 0.189). Unsupervised EEG-DSA vs PMean over all subjects was not significant (p = 0.65 / 0.34), but it was significant in the high-error group (p = 0.03).
  - On dataset 1 (IIa), PMean beat unsupervised EEG-DSA on average.

  Source: [Arvaneh et al. 2013 PDF](https://personal.ntu.edu.sg/ctguan/Publications/2013_Mahnaz_Neural_Comp.pdf).
- **Unsupervised LDA adaptation (Vidaurre, Kawanabe, von Bünau, Blanchard & Müller, IEEE TBME 2011)** [V]. This is the analogue of MLLR mean/bias adaptation. It adapts the LDA (bias via the global feature mean, "PMean") to counter "nonclass-related nonstationarities". It was tested offline on 19 datasets and online, and was reported to beat a supervised adaptation approach. Sources: [Tecnalia record](https://dsp.tecnalia.com/items/fb7aa4de-8670-4dd2-b091-ce16f6c57053/full); [Fraunhofer record](https://publica.fraunhofer.de/handle/publica/226727).
- **Feature normalisation across sessions (Krauledat et al., NeurIPS 2006)** [V]. CSP features were normalised and "prototypical" features derived across sessions to cut calibration time. This is a CMVN analogue. Source: [NeurIPS 2006 metadata](https://papers.neurips.cc/paper_files/paper/2006/file/275d7fb2fd45098ad5c3ece2ed4a2824-Metadata.json) (via search snippet).
- **Normalisation choice (Apicella et al. 2023)** [V]. Normalisation choice alone rivals domain adaptation. On BCI IV-2a, proper normalisation reached 68.52 ± 11.35% cross-subject and 67.82 ± 12.48% cross-session. Source: [arXiv 2210.01081](https://arxiv.org/pdf/2210.01081).
- **Confidence-filtered pseudo-labels** [V]. This is the analogue of unsupervised adaptation with first-pass hypotheses.
  - A smooth conditional domain-adversarial framework with self-iterative pseudo-labels and outlier removal reports 80.67% (IV-2a) and 86.17% (IV-2b) cross-session.
  - BLSAN uses confidence-based pseudo-labels on IV-2a/2b with unlabelled calibration data.
  - SSTDA applies synchronized self-training for inter-session IV-2a; no number was available in the snippet.

  No threshold values were found in the snippets. Sources: [IEEE 10684550](https://ieeexplore.ieee.org/document/10684550); [BLSAN](https://pure.bit.edu.cn/en/publications/blsan-a-brain-lateralization-guided-subject-adaptive-network-for-/); [SSTDA record](https://vufind.katalog.k.utb.cz/SummonRecord/FETCH-LOGICAL-c274t-e99e579bfe990061bf052bc508de19033ef1bff5989ddef135fd95f2e9f02e7f3).
- **Riemannian Procrustes Analysis, RPA (Rodrigues, Jutten & Congedo, TBME 2019)** [V]. This is the Riemannian analogue of a full CMLLR rotation: re-centre → stretch → rotate on the SPD manifold.
  - pyRiemann's `TLRotate` "rotates the matrices from each source domain so to match its class means with those from the target domain", using `metric` in {"euclid", "riemann"} and per-class `weights`.
  - The rotation therefore needs target class means, which means labels or pseudo-labels. This is an inference [I]: the docs imply it but do not state it outright.

  Sources: [pyRiemann TLRotate docs](https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.transfer.TLRotate.html); [Rodrigues et al. PDF](https://plcrodrigues.github.io/files/paper-ieee-tbme-2018.pdf).
- **Subject-conditioning layers (LHUC / x-vector-conditioning analogue)** [V]. A 2025 arXiv "Subject-Conditioned Layer" splits weights into a shared part and a per-subject low-rank correction. This is cross-subject and supervised. Source: [arXiv 2510.08059](https://arxiv.org/abs/2510.08059v1).
- **VTLN analogue** [V]. Searches found no EEG frequency warping. The nearest work is subject-specific band selection:
  - A static fixed band is beaten by adaptive tracking of discriminative frequency components (AWSSP).
  - pyRiemann examples show per-subject optimal bands of 9–15 Hz vs 9–17 Hz.

  Sources: [HKUST AWSSP](https://researchportal.hkust.edu.hk/en/publications/adaptive-tracking-of-discriminative-frequency-components-in-elect/); [pyRiemann band selection](https://pyriemann.readthedocs.io/en/v0.8/auto_examples/biosignal-mi/plot_frequency_band_selection.html).

### Inferences

**Method catalogue, speech side.** Screening refers to IV-2a T→E, offline, 9 subjects.

**S1. Model-based fMLLR / "pseudo-label rotation" (EM-CMLLR)** [I, untried in EEG as far as I found]
- Field: speech.
- Mechanism:
  1. After per-session whitening, fit class-conditional Gaussians on session-T tangent vectors. Alternatively use the CSP+LDA or convnet posteriors.
  2. On session E, alternate: (E-step) compute posteriors γ_ik under the frozen model; (M-step) solve for an orthogonal R, or a full A with a log|A| term as in CMLLR. The objective maximises Σ γ_ik log N(R·x_i; μ_k, Σ_k). With the orthogonal constraint this is a weighted Procrustes problem (SVD of Σ_k μ_k (Σ_i γ_ik x_i)ᵀ).
  3. Iterate 3–5 times.
- Labels: none on E. It uses soft posteriors, exactly like unsupervised fMLLR.
- Cross-subject: none.
- EEG mapping:
  - (a) CSP/tangent features: transform the whitened covariances by C → R C Rᵀ, or rotate tangent vectors.
  - (b) Convnet: transform the whitened raw signal x → R x (channel-space rotation), then apply the frozen net; posteriors come from the net.
- Implementation: no library does this directly. `pyriemann.transfer.TLRotate` with target y = argmax pseudo-labels gives the hard-label version in about 10 lines (`encode_domains`, `TLCenter`, `TLStretch`, `TLRotate`). The soft-EM version is about 40 lines of numpy using scipy.linalg.orthogonal_procrustes.
- Screening effort: about half a day.
- Kill criterion: below +2pp mean over no-rotation on IV-2a E, or a loss in ≥4/9 subjects. Risk: with 4 classes the rotation can permute classes when first-pass accuracy is low (subjects 2/5/6). Guard: allow only small rotations (regularise R toward I), and accept the update only if E-set log-likelihood increases.

**S2. LHUC on the convnet** [I, untried for cross-session EEG]
- Field: speech.
- Mechanism: one scalar per hidden unit (per conv channel), r = 2σ(θ) with θ initialised to 0 so r = 1. All else is frozen. Train on E with an unsupervised loss, either confidence-filtered pseudo-label cross-entropy (exactly as in the LHUC paper's unsupervised mode) or entropy minimisation.
- How it differs from AdaBN: it learns gains through the classifier objective rather than matching moments, so it can reweight discriminative filters, not just renormalise them.
- Labels: none on E (pseudo-labels from a first pass).
- Cross-subject: none.
- Implementation: PyTorch; about 20 lines of forward hooks or `nn.Parameter` scaling after each conv/BN in braindecode ShallowFBCSPNet. For the TTA variant, Tent / SHOT code could be reused.
- Screening effort: about half a day (40 hidden units in Shallow → 40 parameters).
- Kill criterion: below +1.5pp mean, or variance across 5 seeds larger than the gain. Close the method if it equals plain BN-affine (γ/β) Tent; that would mean the extra parameterisation adds nothing.

**S3. Confidence-filtered pseudo-label self-training** [V for EEG, numbers above]
- Mechanism: the generic speech unsupervised loop: decode → keep high-confidence → retrain or adapt.
- Labels: none on E.
- Cross-subject: none if the source is the same subject's T only.
- Implementation: sklearn `SelfTrainingClassifier(threshold=…)` around LDA/tangent-space LR; for the net, braindecode with a fine-tune loop.
- Screening effort: 2 hours for the linear version.
- Kill criterion: below +1.5pp mean, or a loss in weak subjects. Weak subjects should be checked because their first-pass accuracy is about 57–60%, near where pseudo-label noise dominates.

**S4. MAP mean adaptation of LDA / Gaussian classes** [I]
- Mechanism: class means on E are estimated by soft-posterior-weighted MAP interpolation toward the T means, with τ ≈ 10–50 trials of prior weight. This generalises PMean, which shifts only the global mean, to class-conditional shifts. This is one of the named residuals.
- Labels: none.
- Implementation: about 20 lines on top of sklearn LDA / `pyriemann.classification.MDM` (update the `covmeans_`).
- Screening effort: 2 hours.
- Kill criterion: below +1pp over PMean.

**S5. VTLN-analogue: per-session spectral warp α** [I, untried]
- Mechanism: resample each E trial's band edges by α in [0.9, 1.1] (equivalently, shift the filter-bank centre frequencies). Choose α by maximum likelihood of E features under the T class-mixture, or by minimum posterior entropy, with no labels.
- Physiological rationale: within-subject peak frequencies of mu/beta drift across days (state, fatigue). This is not verified; I found no source quantifying day-to-day drift of the mu peak frequency in MI subjects.
- Cross-subject: none.
- Implementation: scipy.signal filters inside a grid loop; MNE `filter_data`.
- Screening effort: half a day.
- Kill criterion: chosen α equals 1.0 for most subjects, or below +1pp.

**S6. HEQ / quantile mapping per feature** [I]
- Mechanism: per-dimension CDF matching of E features to T features. It is non-linear, so it goes beyond mean/variance.
- Implementation: `sklearn.preprocessing.QuantileTransformer` fitted per session (each session mapped to a normal) or `np.interp` of quantiles.
- Screening effort: 1 hour.
- Relevance: it changes only marginal shapes, while the known residual is a rotation (joint structure). Prior expectation is small.
- Kill criterion: below +1pp.

**S7. SAT** [I]
- Mechanism: train on T with per-session normalisation inside the loop, so the model is trained on normalised data. The project's whitening-before-training is already the SAT analogue for second moments.
- SAT-LHUC needs ≥2 labelled sessions/blocks to learn speaker-independent vs speaker-dependent parts. Within-subject this could use the 6 runs of session T as pseudo-"speakers" (per-run LHUC during training). It is cheap but speculative.

**S8. i-vector/x-vector conditioning** [I]
- Requires a population-trained extractor, so it is disallowed as a cross-subject correction. With one subject and two sessions, a "session embedding" reduces to the session statistics already used in whitening. Excluded.

### Gaps
- I found no published EEG application of model-based CMLLR with posterior-weighted EM, of LHUC, or of VTLN-style frequency warping. The searches "MLLR EEG", "LHUC EEG" and "frequency warping EEG" returned only speech papers or unrelated subject-conditioning work.
- RPA's original IV-2a numbers were not retrieved because the PDF was not fetched.
- Whether TLRotate tolerates pseudo-labels well on 4-class IV-2a has no published number found.
- No pseudo-label confidence thresholds were found for IV-2a.
- The 80.67% cross-session figure (IEEE 10684550) comes from a domain-adversarial deep method. Whether it pools other subjects as source was not verified.

## Q2. Harmonization (ComBat variants, CovBat, longitudinal ComBat, RUV, quantile normalisation, RAVEL, neuroHarmonize): which work within one subject across two sessions

### Takeaway
Every ComBat-family method is formally a per-batch location/scale (plus, for CovBat, covariance) model with empirical-Bayes pooling across features. Its value-add over CMVN comes from borrowing strength across features and from protecting a biological covariate, and that covariate is unknown (unlabelled) in session E. Within one subject with trials as samples, two-batch ComBat is mathematically allowed. However, because the class covariate cannot be supplied for E, it either reduces to per-feature mean/variance matching (already closed) or must be run with pseudo-labels.

CovBat is the only member that goes beyond second moments of features. Its PCA-residual covariance step is a cousin of the per-session whitening the project already does, so expected gain is small.

RUV is the one genuinely new idea. It needs negative controls: features known not to carry class information (e.g. non-motor channels/bands, or pre-cue baseline windows). That is plausible in EEG and untried.

### Cited Findings
- **ComBat / neuroCombat (Fortin et al.)** [V for API, K for paper].
  - API: `neuroCombat(dat, covars, batch_col, categorical_cols, eb, parametric=True, mean_only=False, ref_batch=None)`. `dat` is features × scans and `covars` holds one row per scan with batch and biological covariates to preserve. It returns `data`, `estimates`, `info`. `ref_batch` defaults to None, so the reference is the average across scans.
  - Paper: [Fortin et al., NeuroImage 167, 2018](https://doi.org/10.1016/j.neuroimage.2017.11.024) [K].

  Source: [neuroCombat GitHub](https://github.com/Jfortin1/neuroCombat).
- **CovBat (Chen et al., Human Brain Mapping 43(4):1179–1195, 2022)** [V]. It corrects site effects "in mean, variance, and covariance".
  - After harmonisation, ML could not identify scanner manufacturer, while disease-group prediction was retained.
  - R and Python implementations exist; the Python version is a modification of brentp/combat.py. An FC-CovBat variant for functional connectivity exists in FCHarmony.
  - Algorithm [K]: ComBat → PCA of the residuals → ComBat on the PC scores → back-project.

  Sources: [PubMed 34904312](https://www.pubmed.gov/34904312/); [CovBat GitHub](https://github.com/andy1764/CovBat_Harmonization); [FCHarmony](https://github.com/andy1764/FCHarmony); [Penn DBEI summary](https://dbei.med.upenn.edu/?p=1452).
- **ComBat on resting-state EEG (Jaramillo-Jiménez et al., Clin. Neurophysiol. 2024)** [V]. Several ComBat variants were applied to sensor-space oscillatory and aperiodic features. All reduced batch effects; HarmonizR and OPNested-GMM ComBat performed best. This is multisite and cross-subject, not BCI cross-session. No ComBat application to cross-session MI decoding was found. Source: [record](https://ouci.dntb.gov.ua/works/4NpPvdv9).
- **RUV (Gagnon-Bartsch & Speed 2012; RUV\*, Gerard & Stephens)** [V].
  - Negative controls are features "known a priori not to be differentially expressed", so their variation is unwanted.
  - In brain imaging, white matter, CSF, or V1 were suggested as negative controls.
  - The R package `ruv` implements RUV-2, RUV-4, RUV-inv and RUV-III.

  Sources: [Berkeley tech report](https://statistics.berkeley.edu/sites/default/files/tech-reports/ruv.pdf); [arXiv 1705.08393](https://arxiv.org/pdf/1705.08393); [CRAN ruv](https://cran.r-universe.dev/ruv/doc/manual.html).
- **Euclidean Alignment (Wu 2025 revisit)** [V]. EA, the whitening the project already uses, reduces inter-subject and inter-session differences and is "effective and efficient" across 13 paradigms. Source: [arXiv 2502.09203](https://arxiv.org/pdf/2502.09203).
- **Other harmonisation methods** [K]:
  - neuroHarmonize (Pomponio et al. 2020) = ComBat-GAM, with non-linear age covariates and `harmonizationLearn` / `harmonizationApply`: [GitHub rpomponio/neuroHarmonize](https://github.com/rpomponio/neuroHarmonize).
  - Longitudinal ComBat (Beer et al. 2020) adds a subject random effect, which requires multiple subjects: [NeuroImage 220, 2020](https://doi.org/10.1016/j.neuroimage.2020.117129).
  - RAVEL (Fortin et al. 2016) uses control-region voxels (CSF) as RUV-style negative controls for MRI intensity: [NeuroImage 132, 2016](https://doi.org/10.1016/j.neuroimage.2016.02.036).
  - Python ComBat: `pycombat` / `inmoose.pycombat.pycombat_norm`.

### Inferences
Single-subject admissibility, with trials as samples and sessions as the two batches [I]:

| Method | Usable within one subject? | Why / what it reduces to | Beyond mean/variance? | Screening effort | Verdict |
|---|---|---|---|---|---|
| ComBat (neuroCombat, pycombat), `ref_batch=T` | Yes, mechanically | The class covariate is unknown on E, so it cannot be "preserved"; without it this is per-feature location/scale with EB shrinkage | No | Closed in project (tied/null) | Skip |
| ComBat with pseudo-label covariate | Yes | Iterate: predict E → ComBat with class covariate → re-predict. Removes the batch effect estimated *conditional on class*, so it targets class-conditional shift | Partly | 2 h | Cheap screen; kill if below +1pp over plain ComBat |
| CovBat, `ref_batch=T` | Yes | ComBat, then PCA on pooled residuals, then per-batch rescaling of PC scores. Harmonises the feature covariance (for CSP log-var/tangent features: covariance of features, i.e. 4th-order signal stats) | Yes (feature covariance) | 2–3 h (Python port of combat.py) | Moderately promising for tangent features. Kill if below +1pp, or if equal to whitening+ComBat |
| ComBat-GAM / neuroHarmonize | Yes, trivially | The non-linear covariate (age) is irrelevant within subject; a non-linear time-in-session covariate could model within-session drift (P3) | Drift only | 3 h | Low priority |
| Longitudinal ComBat | No | Needs a subject random effect across many subjects | — | — | Disallowed |
| RUV-2/RUV-4/RUV-III | Yes | Estimate the unwanted factor W from negative-control features (pre-cue baseline −2..0 s covariances/log-powers, non-sensorimotor bands, e.g. 1–4 Hz or >40 Hz, or frontal/occipital channels), then regress W out of all features. Unlike moment matching it removes *specific directions* of session variation, including rotation-like ones, without needing class labels | Yes | Half a day (no Python package; RUV-2 ≈ 15 lines numpy: SVD of control block → W → OLS residualise) | Most novel lead in this section. Kill if below +1pp, or if baseline-window controls carry class info (check that T-session decoding from controls is at chance) |
| RAVEL | Yes (RUV with control region) | Same as RUV using "control channels" | Yes | Same as RUV | Merge with RUV |
| Quantile normalisation / HEQ | Yes | Per-feature marginal CDF matching | Marginal shape only | 1 h | Low prior |

Further inferences [I]:
- The EB step of ComBat pools across features, not subjects, so it is allowed. With only 2 batches, the batch-variance hyperprior is poorly identified, so `eb=False` vs `True` should not matter much.
- RUV's pre-cue baseline negative control matches the speech practice of estimating the channel from silence or non-speech frames, i.e. noise-adaptive CMLLR. In 2a, each trial has a 2 s fixation period before the cue, so controls are available at zero label cost.

### Gaps
- No EEG/BCI paper applying CovBat, RUV, or RAVEL was found.
- The ComBat-in-EEG evidence is multisite resting-state, not decoding transfer.
- The CovBat Python function name and signature were not retrieved; the repo page points to a Python subdirectory without documenting it.
- No source was found on whether 2-batch CovBat with about 288 samples per batch is stable.

## Q3. Cheap transfer to (a) CSP log-variance / tangent features and (b) convnet input/features; already-applied vs untried

### Takeaway
The best ratio of novelty to cost for this project is:
1. Pseudo-label EM rotation (CMLLR/RPA-rotate with soft posteriors) on whitened covariances or tangent vectors, serving both (a) and the convnet's input via x → R x.
2. LHUC/Tent-style per-unit gain adaptation with confidence-filtered pseudo-labels for (b).
3. RUV with pre-cue negative controls for (a).

Already applied in EEG, and either exhausted here or marginal: global covariance re-colouring (EEG-DSA unsupervised = whitening), PMean bias adaptation, CMVN-style normalisation, and multisite ComBat.

### Cited Findings
- **Unsupervised EEG-DSA** reduces to the normalising transform (Σ̄_t^(-1/2)Σ̄_s^(1/2)). Its gain over PMean was +0.4–0.9pp overall (not significant) and +1.9–2.5pp in poor performers (p = 0.03) [V]. Source: [Arvaneh et al. 2013](https://personal.ntu.edu.sg/ctguan/Publications/2013_Mahnaz_Neural_Comp.pdf).
- **Continuous (sliding-window) re-estimation** beat single estimation: 73.1 vs 72.1 unsupervised, p = 0.067. This suggests within-session drift is a real but small residual [V]. Same source.
- **pyRiemann `TLRotate`** aligns class means, so it needs target class labels or pseudo-labels [V docs; I for the labels conclusion]. Source: [pyRiemann](https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.transfer.TLRotate.html).
- **Pseudo-label self-training** for cross-session IV-2a is published with deep models (80.67% reported) [V]. Source: [IEEE 10684550](https://ieeexplore.ieee.org/document/10684550).

### Inferences
Summary table [I unless marked]:

| # | Method (field) | Feature target | E labels | Cross-subject op? | Code | Effort | Published EEG? | Kill criterion |
|---|---|---|---|---|---|---|---|---|
| 1 | EM-CMLLR / soft pseudo-label Procrustes rotation (speech) | (a) whitened cov / tangent; (b) raw x → Rx | none (posteriors) | no | scipy `orthogonal_procrustes`; pyRiemann `TLRotate` with pseudo-y | 0.5 d | RPA rotation exists but uses labels; soft-EM version untried [I] | below +2pp mean, or loss in ≥4/9 |
| 2 | LHUC unsupervised (speech) | (b) per-channel gains | none (pseudo-labels/entropy) | no | PyTorch, ~20 lines | 0.5 d | none found | below +1.5pp, or no better than BN-affine Tent |
| 3 | Confidence pseudo-label self-training (speech) | (a)+(b) | none | no | sklearn `SelfTrainingClassifier` | 2 h | yes (deep, IV-2a) [V] | below +1.5pp, or harm in subjects 2/5/6 |
| 4 | MAP class-mean adaptation (speech) | (a) LDA/MDM means | none (soft posteriors) | no | ~20 lines | 2 h | PMean = global-mean only [V] | below +1pp over PMean |
| 5 | RUV-2 with pre-cue controls (genomics) | (a) | none | no | numpy | 0.5 d | none found | below +1pp; controls decode class above chance |
| 6 | CovBat, ref=T (neuroimaging) | (a) tangent features | none | no | CovBat Python | 2–3 h | none in BCI | below +1pp, or equal to ComBat |
| 7 | Pseudo-label-covariate ComBat | (a) | none | no | neuroCombat | 2 h | none | below +1pp over plain ComBat |
| 8 | VTLN-style spectral warp α (speech) | (a)+(b) at filtering | none (ML/entropy) | no | scipy/MNE | 0.5 d | none found | α* ≈ 1 for most subjects, or below +1pp |
| 9 | HEQ / quantile mapping (speech) | (a) | none | no | sklearn `QuantileTransformer` | 1 h | none specific | below +1pp |
| 10 | SAT-LHUC with T's 6 runs as "speakers" | (b) | none on E | no | PyTorch | 1 d | none | below +1pp over plain training |
| — | x-vector conditioning, longitudinal ComBat, population priors | — | — | yes | — | — | — | Disallowed |

Priority order for screening: 3 → 1 → 4 → 2 → 5 → 6. Rationale:
- Items 3 and 4 are 2-hour baselines that also produce the posteriors items 1 and 2 need.
- Item 1 directly targets the documented residual (rotation after whitening).
- Items 5 and 6 are the only harmonisation ideas that go beyond moment matching without labels.

With n = 9, a pre-registered screen should use a paired sign test across subjects plus mean difference, and promote a method only if it gains ≥ +2pp with ≥7/9 subjects non-negative. This threshold is a suggestion [I], not sourced.

### Gaps
- No published numbers exist for items 1 (soft-EM version), 2, 5, 6, 7, 8 or 10 on IV-2a cross-session. These are untried, as far as the searches show.
- Whether whitening already removes most of what CovBat's PC rescaling would fix is untested.
- The size of within-subject day-to-day mu/beta peak-frequency drift, which justifies VTLN, is unquantified in the sources found.
