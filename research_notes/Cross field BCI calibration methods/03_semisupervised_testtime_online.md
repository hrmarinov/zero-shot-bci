# P2 + P3: Label-efficient (semi-supervised / active) and test-time / online-drift methods for EEG MI

Scope: methods for (P2) learning from ~100-288 labelled session-1 trials plus an unlabelled, shifted session-2 pool, and (P3) tracking within-session drift online without labels. AdaBN/per-layer BN after alignment is excluded (already null here). Constraint: the eval session stays unlabelled, and no computation may combine two different subjects.

Evidence tags: **[V]** means verified this session from the fetched primary source. **[K]** means prior knowledge of a canonical paper or library; the URL is given but the page was not fetched this session, so check it before relying on it. **[I]** means my inference.

Research budget note: 15 tool calls. The Wimpff 2024 paper, T-TIME, the eeg-otta README and the skada API were read in full or near-full. Vidaurre 2011, SPDIM, OSPDIM and Kumar 2019 were seen only as abstracts.

---

## Q1. Test-time adaptation (TENT, SHOT, T3A, EATA, CoTTA, MEMO, NOTE/RoTTA, LAME) and Wimpff et al. 2024

### Takeaway
The only within-subject cross-session TTA result on IV-2a (Wimpff 2024) gives these numbers for a BaseNet convnet trained on session 1:
- source model: 76.2%
- online RA + AdaBN: 78.3%
- adding entropy minimisation (EM): 79.7%

The EM gain (+1.45pp) only appears when the source model was trained with heavy label smoothing (δ=0.5). Without smoothing, EM adds about 0.

In cross-subject EEG benchmarks:
- Tent adds about 0 over BN-adapt.
- T3A and CoTTA hurt.
- Information-maximisation variants (SHOT-IM-like T-TIME, SAR) give +1 to +2pp.

So the realistic ceiling for TTA on top of our pipeline is about +1-2pp. The cheapest screens are "label smoothing + entropy/IM" and LAME (output-only).

### Cited Findings

**Wimpff, Döbler, Yang 2024 ("Calibration-free online test-time adaptation for EEG MI decoding", IEEE BCI Winter Conf 2024, arXiv 2311.18520)**
- Protocol [V]:
  - Model is BaseNet (an EEGNet/ShallowNet hybrid), 1000 epochs, 5 seeds. Preprocessing is only a 40 Hz low-pass.
  - Datasets: BCI IV-2a (4 classes, 22 channels, 288 trials/session) and IV-2b.
  - Cross-session setting: train on session 1 and adapt online on session 2 of the same subject. This is within-subject, so it is compatible with our no-cross-subject constraint.
  - Alignment is online EA (arithmetic mean) or RA (geometric mean) computed from a FIFO buffer of 32 trials and recomputed for each incoming trial. Linear or EMA weighting (momentum 0.1) is applied.
  - AdaBN variants: BN-1, BN-0.5 (mixed with source statistics) and BN-EMA.
  - EM uses Adam at lr 5e-4, with one loss step per full buffer renewal.
  - Tent, T3A, SHOT and NOTE are not baselines.
  - — [arXiv HTML v2](https://arxiv.org/html/2311.18520v2)
- IV-2a cross-session numbers (mean±std over 5 seeds × 9 subjects) [V] — [arXiv HTML v2](https://arxiv.org/html/2311.18520v2):

  | Configuration | Accuracy |
  |---|---|
  | Source | 76.16 |
  | EA | 75.43 (EA hurts cross-session) |
  | RA | 76.91 |
  | BN-1 | 77.72 |
  | RA(EMA)-BN-1 | 78.29 |
  | RA(EMA)-BN-0.5 | 78.33 |
  | RA(EMA)-BN-1 + EM, δ=0 | 78.22 |
  | RA(EMA)-BN-1 + EM, δ=0.5 | **79.74±0.57** |

- IV-2a cross-subject [V]: source 57.38, RA-BN 65.4-65.6, + EM with δ=0.4 67.31, continual 69.01. — [arXiv HTML v2](https://arxiv.org/html/2311.18520v2)
- Failure modes reported [V] — [arXiv HTML v2](https://arxiv.org/html/2311.18520v2):
  - EM needs label smoothing. The authors argue that sharp, overconfident source models make entropy minimisation ineffective.
  - BN-1 is unreliable with small buffers.
  - Continual adaptation hurts when session conditions differ (2b).
  - Class imbalance is not studied.
  - There is a text/table mismatch: 65.75 in the text vs 65.31 in the table.
- Code [V]:
  - Repository: [github.com/martinwimpff/eeg-otta](https://www.github.com/martinwimpff/eeg-otta), MIT licence, built on mariodoebler/test-time-adaptation.
  - Workflow: `train_source_model.py` then `run_adaptation.py --config tta_*`.
  - Pretrained `_within` (cross-session) and `_loso` checkpoints are included.

**T-TIME (Li/Wu group, arXiv 2412.07228)**
- Setting [V]:
  - Cross-subject (LOSO), trials arriving one by one, TTA batch size 8.
  - EA aligned incrementally, then an ensemble of EEGNets combined by a spectral meta-learner.
  - Update loss is conditional entropy plus an adaptive marginal (diversity) regulariser, i.e. information maximisation.
  - — [arXiv 2412.07228](https://arxiv.org/html/2412.07228v1)
- BNCI2014001 (= IV-2a) average accuracy [V] — [arXiv 2412.07228](https://arxiv.org/html/2412.07228v1):

  | Method | Accuracy |
  |---|---|
  | BN-adapt | 74.54 |
  | Tent | 74.32 |
  | T3A | 68.23 |
  | CoTTA | 72.04 |
  | SAR | 74.80 |
  | T-TIME | 76.44 |
  | T-TIME, 5-model ensemble | 79.30 |

  - The ~74% level for cross-subject suggests a 2-class (left/right) subset [I; class count not confirmed].
- Further results [V] — [arXiv 2412.07228](https://arxiv.org/html/2412.07228v1):
  - SHOT, SHOT-IM and ISFDA are reported offline.
  - Pure conditional-entropy minimisation "may collapse all test data into one class". The marginal regulariser is added for this reason.
  - Under 2:1 simulated test class imbalance, MCC was worst.
  - "Overly complicated" methods such as CoTTA hurt with limited data.
  - Code: [sylyoung/DeepTransferEEG](https://github.com/sylyoung/DeepTransferEEG). It contains EEG implementations of Tent/T3A/SHOT/SAR baselines [I from the paper's baseline list].

**SHOT on EEG**
- AEA-SHOT (Essex) builds on SHOT for source-free EEG adaptation and reports 81.4% average on PhysioNet, not IV-2a. — [Essex repository](https://repository.essex.ac.uk/39235/1/Lightweight_Source-Free_Domain_Adaptation_based_on_Adaptive_Euclidean_Alignment_for_Brain-Computer_Interfaces.pdf) (via search snippet)
- No published SHOT result for IV-2a within-subject cross-session was found.

**SPDIM (Li, Kawanabe, Kobler, ICLR 2025, arXiv 2411.07249)** — [arXiv](https://arxiv.org/abs/2411.07249) [V abstract]
- Source-free.
- Learns one SPD-manifold-constrained parameter per target domain by information maximisation.
- Argues that Riemannian re-centering (RA) is biased under label (class-prior) shift.
- Evaluated on MI and sleep data in cross-session and cross-subject settings.
- Exact BNCI2014001 numbers were not retrieved.

**OSPDIM (Chu, Li, Kawanabe, Kobler, arXiv 2608.05315)** — [arXiv](https://arxiv.org/abs/2608.05315) [V abstract]
- Online version of SPDIM.
- Adds a manifold-constrained bias to the tangent-space mapping, optimised by information maximisation on a stream.
- Mechanism: RCT/online re-centering implicitly assumes balanced class priors and misaligns data under imbalanced streams.
- Per the search summary it evaluates on BNCI2014001 with imbalance ratios of 0.1-0.4 under LOSO — [search snippet](https://arxiv.org/pdf/2608.05315).

**Mechanisms of the generic vision TTA methods [K]**
- TENT: entropy minimisation updating only BN affine parameters — [Wang et al. ICLR 2021, arXiv 2006.10726](https://arxiv.org/abs/2006.10726).
- SHOT: freeze the source classifier, train the feature extractor with information maximisation (low per-sample entropy + high batch-marginal entropy) plus k-means-centroid pseudo-labels — [Liang et al. ICML 2020, arXiv 2002.08546](https://arxiv.org/abs/2002.08546).
- T3A: replaces the last linear layer with prototypes built from confident test features; no back-propagation — [Iwasawa & Matsuo NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html).
- EATA: entropy filter + Fisher anti-forgetting — [Niu et al. ICML 2022, arXiv 2204.02610](https://arxiv.org/abs/2204.02610).
- CoTTA: mean-teacher + augmentation averaging + stochastic restore — [Wang et al. CVPR 2022, arXiv 2203.13591](https://arxiv.org/abs/2203.13591).
- MEMO: marginal entropy over augmentations of a single sample — [Zhang et al. NeurIPS 2022, arXiv 2110.09506](https://arxiv.org/abs/2110.09506).
- NOTE: instance-aware BN + prediction-balanced reservoir for temporally correlated / non-i.i.d. streams — [Gong et al. NeurIPS 2022, arXiv 2208.05117](https://arxiv.org/abs/2208.05117).
- RoTTA: robust BN + category-balanced memory + time-aware reweighting — [Yuan et al. CVPR 2023, arXiv 2303.13899](https://arxiv.org/abs/2303.13899).
- LAME: output-only. Adjusts the source softmax outputs with a Laplacian (kNN-affinity) regulariser, solved by a concave-convex procedure; the network is never updated — [Boudiaf et al. CVPR 2022, arXiv 2201.05718](https://arxiv.org/abs/2201.05718).
- Döbler's test-time-adaptation repo (the basis of eeg-otta) implements Tent, EATA, CoTTA, RoTTA, LAME, SAR, NOTE, T3A, MEMO and others [K: verify at github.com/mariodoebler/test-time-adaptation].

**Method catalogue (all within-subject; source = own session 1, target = own session 2 unlabelled)**

| Method | Maps onto | Labels | Cross-subject? | Python | Screen effort | EEG evidence | Kill criterion | Failure modes |
|---|---|---|---|---|---|---|---|---|
| Label smoothing + entropy min (Wimpff) | ShallowConvNet | none at eval | no | eeg-otta `run_adaptation.py`; or ~30 lines torch | 0.5 day (retrain source with δ=0.5) | +1.45pp over RA-BN, IV-2a cross-session [V] | <+1pp mean, or any subject down >3pp | Collapse without smoothing; buffer-size sensitive |
| SHOT-IM (entropy + marginal-diversity) | ShallowConvNet features | none | no | DeepTransferEEG baselines; or torch | 0.5-1 day | T-TIME (IM-type) +~2pp over BN-adapt cross-subject [V]; SHOT itself on IV-2a cross-session not found | as above | Diversity term assumes balanced classes (harmful under prior shift) [I] |
| Tent (BN-affine only) | ShallowConvNet | none | no | Döbler repo | 0.25 day | 74.32 vs BN-adapt 74.54 (≈0) [V] | expect null; low priority | Overconfidence / collapse |
| T3A (prototype last layer) | ShallowConvNet; also LDA-like | none | no | Döbler repo | 0.25 day | −6pp vs BN-adapt [V] | expect negative | Prototype drift on weak subjects |
| CoTTA | convnet | none | no | Döbler repo | 0.5 day | −2.5pp [V] | expect negative | Too many parameters for tiny data |
| EATA / SAR | convnet | none | no | Döbler repo | 0.5 day | SAR +0.3pp vs BN [V] | <+1pp | — |
| NOTE / RoTTA | convnet | none | no | Döbler repo | 0.5 day | no EEG numbers found | only relevant if class prior per buffer is skewed | — |
| LAME | any probabilistic output (LDA, MDM, convnet softmax) | none | no | Döbler repo / ~50 lines numpy | ≤0.25 day, no retraining | none found on EEG [gap] | <+0.5pp | kNN graph meaningless if features are not cluster-structured |
| SPDIM / OSPDIM | tangent-space classifier after RA | none | no | code not located | 1-2 days | claimed to beat RCT under label shift [V abstract] | — | Needs a differentiable TS pipeline |

### Inferences
- Wimpff's +2.1pp from RA+BN over source is larger than our ~0 AdaBN gain. Our ShallowConvNet is already trained on RA input with per-session BN, which already captures what BN-1 buys them [I].
- The transferable increment is the EM step: +1.45pp only with label smoothing δ=0.5 [I from V numbers].
- Our 5-seed ensemble (78.3%) already matches their best non-EM configuration. TTA's gain may partly overlap with ensembling, which also smooths outputs [I].
- LAME is the cheapest untried screen: post-hoc and output-only. It also works on CSP+LDA and MDM posteriors and has no collapse risk because no weights change [I].
- With all of session 2 available offline (transductive), the "online" constraint can be relaxed for screening. Running offline SHOT-IM over all 288 eval trials upper-bounds the online variant [I].

### Gaps
- No IV-2a within-subject cross-session results were found for SHOT, T3A, EATA, NOTE, RoTTA, MEMO or LAME. T-TIME's numbers are cross-subject and its class count was not confirmed.
- SPDIM/OSPDIM numbers and code were not retrieved.
- Per-subject (weak 2/5/6) effects of EM in Wimpff were not reported in the parts read.

---

## Q2. Semi-supervised learning (self-training, label propagation, TSVM, co-training, EM-LDA, Riemannian pseudo-labels)

### Takeaway
Semi-supervised MI decoding is old and established (self-training CSP+SVM/LDA loops, EM-style LDA, co-training). Its dominant failure is confirmation bias from wrong pseudo-labels, which compounds through the supervised CSP filter. No modern FixMatch-for-EEG-MI result on IV-2a cross-session was found.

The most promising cheap screens:
- (a) tangent-space features + `sklearn.semi_supervised.LabelSpreading` / self-training on session-1 labels + session-2 unlabelled;
- (b) skada OT label propagation / JDOT / DASVM on tangent vectors;
- (c) pseudo-label-refit MDM class means.

All are within-subject, so they are allowed.

### Cited Findings
- **Iterative pseudo-labelled CSP:** CSP is supervised, so with few labels the spatial filter is unreliable. Established work iterates: fit CSP+SVM on labelled data, pseudo-label the unlabelled data, then refit both CSP and the classifier — [PMC6925734 (IST-TSVM paper)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6925734) (via search snippet).
- **Confirmation bias:** self-training and co-training accuracy decreases when unlabelled data receive wrong labels. Selection, i.e. confidence-filtered pseudo-labelling, is needed — [search snippet summarising PMC6925734 / related](https://pmc.ncbi.nlm.nih.gov/articles/PMC6925734).
- **Base learners and session-wise adaptation:**
  - LDA and Bayesian LDA are the common self-training base learners.
  - One work adds a nonparametric local filter to LDA self-training.
  - An earlier "small training set" paper adapts session-by-session by adding previously classified samples to the training pool.
  - — [arXiv 1306.3474](https://arxiv.org/pdf/1306.3474) (via search snippet)
- **skada implements** [V] — [skada API](https://scikit-adaptation.github.io/all.html):
  - Label propagation: `OTLabelProp`, `JCPOTLabelProp` (target/label shift).
  - `JDOTClassifier`, `DASVMClassifier` (a transductive SVM-style DA classifier).
  - Mapping: `CORAL`, `LinearOTMapping`, `EntropicOTMapping`, `ClassRegularizerOTMapping`, `SubspaceAlignment`, TCA.
  - Label-shift reweighting: `MMDTarSReweight`.
  - Unsupervised model-selection scorers: `PredictionEntropyScorer`, `DeepEmbeddedValidation`, `SoftNeighborhoodDensity`, `CircularValidation`, `MixValScorer`, `MaNoScorer`.
  - Pipeline glue: `make_da_pipeline`, `source_target_split`.
  - No self-training class.
- **sklearn** [K] — [sklearn semi_supervised docs](https://scikit-learn.org/stable/modules/semi_supervised.html):
  - `LabelPropagation`, `LabelSpreading` (kNN/RBF graph, unlabelled = −1).
  - `SelfTrainingClassifier(base, threshold=…, criterion='threshold'|'k_best')`.
  - These wrap pyriemann `TangentSpace` + LDA or LogisticRegression directly.
- **pyriemann** [K] — [pyriemann API](https://pyriemann.readthedocs.io/en/latest/api.html):
  - `MDM`, `TangentSpace`, `FgMDM`.
  - Transfer tools: `pyriemann.transfer.TLCenter` (re-centering), `TLStretch`, `TLRotate` (Procrustes rotation; uses class means, so it needs labels or pseudo-labels in the target), `MDWM`.
  - Pseudo-label MDM, i.e. refitting class means with confident target trials, is a few lines on top of `MDM`.
- **EM for LDA / unsupervised class-mean re-estimation:** see Vidaurre / Kumar in Q3.

**Method catalogue**

| Method | Maps onto | Python | Screen effort | EEG evidence | Kill criterion | Failure modes |
|---|---|---|---|---|---|---|
| Confidence-threshold self-training (tangent-space + LR/LDA) | TS classifier | `SelfTrainingClassifier(LogisticRegression(), threshold=0.8)` on `TangentSpace` of RA-aligned covs | 1-2 h | established for CSP+LDA; IV-2a cross-session numbers not found | <+1pp at 288 labels, and no shift of the 96-label curve | confirmation bias; weak subjects (57-60%) produce mostly wrong pseudo-labels → collapse |
| Self-training ShallowConvNet (FixMatch-style: weak/strong augmentation consistency, threshold τ) | convnet | torch; braindecode augmentations (`braindecode.augmentation`: FTSurrogate, ChannelsDropout, FrequencyShift, SmoothTimeMask) [K] | 1-2 days | not found for IV-2a cross-session | <+1pp | needs good augmentations; threshold miscalibration |
| Label spreading on tangent vectors | TS | `LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.2)` | 1 h | none found on IV-2a | <+1pp | graph dominated by session shift unless RA applied first |
| OT label propagation / JDOT / DASVM | TS vectors | skada `OTLabelProp`, `JDOTClassifier`, `DASVMClassifier` | 2-4 h | no IV-2a cross-session numbers found | <+1pp | OT assumes matched class proportions (use `JCPOTLabelProp` for prior shift) |
| Pseudo-label MDM / CSP refit | MDM, CSP+LDA | pyriemann `MDM` + loop | 2 h | iterative CSP refit established [V snippet] | <+1pp | CSP refit on wrong labels propagates errors |
| Co-training (two views, e.g. CSP-band-power vs TS) | linear | hand-rolled | 0.5 day | established generically; no IV-2a numbers | — | views not conditionally independent |

### Inferences
- Because our session-2 data are already RA-aligned, the two sessions overlap better. That is the regime where label spreading or self-training on tangent vectors has a chance; without alignment the graph would follow the session, not the class [I].
- On the calibration-burden question ("aligned CSP needs 96 labels to match unaligned 288"), semi-supervised methods should be screened as a learning curve:
  - labelled n ∈ {48, 96, 144, 288} from session 1;
  - unlabelled = remainder of session 1 + all of session 2;
  - test = session 2.
  - This is transductive but legitimate as long as session-2 labels are never used for fitting or model selection [I].
- Use skada's `PredictionEntropyScorer` / `DeepEmbeddedValidation` for unsupervised hyper-parameter selection. Otherwise threshold choice silently leaks eval labels [I].

### Gaps
- No modern (2022+) IV-2a within-subject cross-session semi-supervised numbers were found; the snippets did not carry the tables.
- No retrieved paper tests label spreading or TSVM on Riemannian tangent features for IV-2a.

---

## Q3. Online drift tracking (adaptive LDA, Kalman, recursive Riemannian mean, EMS)

### Takeaway
Unsupervised online tracking of the global mean or bias is the best-established P3 tool in EEG:
- Vidaurre 2011 (pooled-mean / bias adaptation of LDA): beats the supervised state of the art; 80 users + 4 SCI patients; first online use.
- Riemannian equivalents: online re-centering / recursive geometric mean (Wimpff online RA; Kumar et al.; Zanini 2018).

Online RA with EMA gives only +0.75pp over source in Wimpff's cross-session IV-2a. Its value comes mostly from between-session centering, which we already do per session (batch). The marginal gain from within-session recursive re-centering on IV-2a is therefore likely small, but it costs about one hour to screen.

Main known risk: re-centering on the unconditional mean is biased under class-prior shift (SPDIM/OSPDIM).

### Cited Findings
- **Vidaurre, Kawanabe, von Bünau, Blankertz, Müller**, "Toward unsupervised adaptation of LDA for BCIs", IEEE TBME 58(3):587-597, 2011, DOI 10.1109/TBME.2010.2093133 [V abstract] — [Korea Univ. record](https://pure.korea.ac.kr/en/publications/toward-unsupervised-adaptation-of-lda-for-brain-computer-interfac/):
  - Three unsupervised LDA adaptation procedures counteract non-class-related nonstationarity.
  - They were tested offline on 19 datasets; one was chosen and tested on 80 healthy users + 4 high-SCI patients.
  - First online unsupervised-classifier BCI experiment; significantly better than the supervised state of the art.
- The specific procedures were not shown on the page read [gap]. From prior knowledge [K]:
  - They are bias-only adaptation via the pooled (class-unconditional) mean;
  - the pooled mean plus an inverse-covariance update (Sherman-Morrison);
  - and a combination.
  - Bias adaptation, b = −wᵀμ_pooled with μ updated by EMA (update coefficient ~0.05 per trial), was preferred.
  - Verify against the paper.
- **Kumar, Yger, Lotte**, "Towards adaptive classification using Riemannian geometry approaches in BCI" [V abstract]:
  - Supervised, unsupervised and hybrid adaptive Riemannian classifiers (MDM).
  - Hybrid adaptation improves 3-class mental-imagery accuracy by "as high as ~17%" over a non-adaptive baseline.
  - — [HAL/typeset copy](https://typeset.io/pdf/towards-adaptive-classification-using-riemannian-geometry-9xvfvhwcg9.pdf) (via search snippet)
- **Related adaptive Riemannian classifiers** — [Frontiers / search snippets](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.635653/pdf):
  - an unsupervised adaptive MDM that retrains after each prediction;
  - another that only re-centers;
  - Benaroch, Lotte et al. 2021 used an unsupervised, incrementally updated Riemannian classifier with a tetraplegic user over long-term training, capturing within- and between-session non-stationarity.
- **Online RA / EA in Wimpff 2024** [V]:
  - Buffer-based online RA with EMA momentum 0.1 gives 76.77 vs source 76.16 vs batch RA 76.91 on IV-2a cross-session.
  - Online RA is about as good as batch RA, but both add <1pp over source for a convnet trained without alignment.
  - — [arXiv HTML v2](https://arxiv.org/html/2311.18520v2)
- **Zanini et al. 2018** [K]: "Transfer learning: a Riemannian geometry framework with applications to BCI", IEEE TBME 65(5):1107-1116 — affine re-centering of each session/subject to identity by its own Riemannian mean, including an online/incremental variant. — [DOI 10.1109/TBME.2017.2742541](https://doi.org/10.1109/TBME.2017.2742541)
- **Exponential moving standardisation** [K]: `braindecode.preprocessing.exponential_moving_standardize(data, factor_new=1e-3, init_block_size=1000)`, used in Schirrmeister et al. 2017 for ShallowConvNet on IV-2a. — [braindecode API](https://braindecode.org/stable/api.html)
- **Class-prior bias of re-centering** [V abstract]: standard centering (RCT) assumes balanced class priors and misaligns under imbalanced streams. OSPDIM corrects this with an information-maximised bias. — [arXiv 2608.05315](https://arxiv.org/abs/2608.05315)
- **Kalman-filtered covariance/mean tracking:** no EEG-MI-specific paper retrieved this session [gap]. Kalman-adaptive LDA (supervised, Vidaurre 2006/2007) exists in prior knowledge [K], but it needs labels.

**Method catalogue**

| Method | Maps onto | Labels | Python | Effort | Evidence | Kill criterion | Failure |
|---|---|---|---|---|---|---|---|
| Pooled-mean bias adaptation (Vidaurre) | CSP+LDA; also TS+LDA | none | ~15 lines: μ ← (1−η)μ + ηx; b = −wᵀμ | 1 h | 19 + 80 users, beats supervised [V abstract] | <+0.5pp across η ∈ {0.01, 0.05, 0.1} | class-prior drift in a window shifts the bias; mis-tuned η |
| Pooled-mean + inverse-covariance (Sherman-Morrison) | LDA | none | numpy | 2 h | Vidaurre's other procedures [K] | — | instability with 22-channel × bands features |
| Recursive Riemannian mean re-centering (online RA) | MDM / TS / convnet input | none | pyriemann `mean_riemann` on a buffer, or geodesic EMA: M ← M #_η C (`pyriemann.utils.geodesic.geodesic_riemann(M, C, η)`) [K] | 1-2 h | +0.75pp online, ≈ batch RA [V] | no gain over per-session batch RA | label-shift bias (OSPDIM); start-up phase with few trials |
| Unsupervised adaptive MDM (refit class means with predicted labels) | MDM | none | pyriemann `MDM` + loop | 2 h | Kumar et al. up to ~17% in hybrid 3-class [V abstract] | <+0.5pp | confirmation bias; mean collapse on weak subjects |
| EMS (braindecode) | convnet input | none | `exponential_moving_standardize` | 0.5 h (retrain) | standard in Schirrmeister pipeline [K] | — | interacts with RA; likely redundant |
| OSPDIM bias | TS | none | code not found | 2+ days | abstract only | — | — |

### Inferences
- P3 on IV-2a is probably small in magnitude. IV-2a sessions are ~1 hour with 6 runs, and Wimpff's online vs batch RA differ by 0.1pp. Our per-session batch RA already uses all 288 eval trials, i.e. it is "future-peeking" but legal [I].
- A cheap diagnostic before any P3 method: split eval-session predictions by run (6 runs × 48 trials) and test for an accuracy trend over runs. A flat trend kills P3 for this dataset [I].
- The Vidaurre bias update composes trivially with our CSP+LDA and TS+LDA pipelines. It is the canonical "already established" P3 baseline and should be run as a control, not as novelty [I].

### Gaps
- Exact Vidaurre 2011 accuracy numbers and procedure details were not retrieved.
- No Kalman-based unsupervised covariance tracking results were found for MI.
- Zanini 2018 numbers were not fetched.

---

## Q4. Active learning / optimal experimental design for calibration trials

### Takeaway
Active learning for MI calibration exists but is a poor fit for our offline protocol. Session-1 trials are recorded with fixed, cued labels, so "choosing which to label" only means choosing a subset of an already-labelled pool, i.e. retrospective subset selection.

Published approaches:
- query-by-committee with LDA (Deakin);
- uncertainty + representativeness (She et al. 2020, ELM);
- batch-mode nearest-average-class-distance (Hossain et al.).

No IV-2a cross-session numbers were retrieved. Expected value for our burden question is low to moderate.

### Cited Findings
- Query-by-committee with three feature extractors + LDA for MI calibration-time reduction (Deakin, batch-mode AL) — [Deakin figshare](https://deakin.figshare.com/articles/conference_contribution/Calibration_time_reduction_for_motor_imagery-based_BCI_using_batch_mode_active_learning/20782612) (via search snippet).
- Uncertainty + representativeness inside an extreme learning machine, reported to match or exceed state-of-the-art AL algorithms — [PubMed 32256550](https://www.pubmed.gov/32256550/) (via search snippet).
- Batch-mode AL based on nearest average class distance (NACD) for multiclass BCI — [Global-Sci](https://global-sci.com/article/86688/a-batch-mode-active-learning-method-based-on-the-nearest-average-class-distance-nacd-for-multiclass-brain-computer-interfaces).
- Azab et al. select trials near the new subject's informative trials. This is cross-subject, so it is **disallowed** here — [White Rose](https://eprints.whiterose.ac.uk/147288/1/Final_version.pdf) (via search snippet).
- Tools [K]: `modAL` (`modAL.uncertainty.uncertainty_sampling`, `margin_sampling`, `modAL.disagreement` for QBC); `skactiveml`.

### Inferences
- An offline screen is possible but somewhat artificial: from the 288 labelled session-1 trials, pick 96 by uncertainty/representativeness and compare with 96 random or first-96 chronological trials.
- In a real cued paradigm the label is fixed by the cue, so AL could at best choose *which class to cue next* (adaptive design). That is not testable offline [I].
- A cheaper OED-like alternative: choose the labelled subset to cover the class-conditional covariance spread, e.g. k-means/kernel-herding on tangent vectors within each class [I].

### Gaps
- No IV-2a numbers for any AL method were found.
- Whether AL beats chronological-first-n (the realistic baseline, since early trials are what you have) is untested in the retrieved literature.

---

## Q5. What is established in EEG vs genuinely untried (for this setup)

### Takeaway
- **Established:**
  - unsupervised mean/bias adaptation (Vidaurre 2011);
  - online/batch Riemannian re-centering (Zanini 2018, Wimpff 2024);
  - AdaBN + entropy minimisation for convnets (Wimpff 2024);
  - IM-based cross-subject TTA (T-TIME);
  - iterative pseudo-label CSP self-training;
  - unsupervised adaptive MDM (Kumar/Lotte).
- **Apparently untried or unreported on IV-2a within-subject cross-session:**
  - LAME (output-only Laplacian adjustment);
  - label spreading / OT label propagation on RA-aligned tangent vectors;
  - FixMatch-style consistency self-training of ShallowConvNet;
  - SHOT-IM within-subject;
  - prior-shift-robust TTA (NOTE/RoTTA/OSPDIM) on IV-2a cross-session;
  - semi-supervised learning as a means to shift the labelled-trials learning curve (96 → fewer).

### Cited Findings
- See Q1-Q3 citations. Wimpff cross-session IV-2a is the only within-subject TTA benchmark found, and it tests only alignment, BN and EM — [arXiv HTML v2](https://arxiv.org/html/2311.18520v2).
- T-TIME tests Tent/T3A/CoTTA/SAR/SHOT only cross-subject — [arXiv 2412.07228](https://arxiv.org/html/2412.07228v1).

### Inferences — recommended screening order (cheap → expensive; all within-subject, eval labels used only for final scoring)
1. **Run-wise drift diagnostic.** Accuracy per eval run; 10 min. Gates all of P3.
2. **LAME** on ShallowConvNet softmax and CSP+LDA posteriors. No retraining; 0.25 day; kill <+0.5pp.
3. **Vidaurre pooled-mean bias adaptation** on CSP+LDA / TS+LDA; 1 h; established control.
4. **Self-training / LabelSpreading on RA tangent vectors** (sklearn) as a learning curve over n_labelled ∈ {48, 96, 144, 288}. This is the direct attack on the 96-trial burden; kill if the curve does not move ≥2pp at n=96.
5. **Label smoothing (δ=0.5) + entropy/IM TTA** on ShallowConvNet, offline-transductive over all eval trials and online with a 32-trial buffer. eeg-otta code is reusable. Kill <+1pp mean or any weak subject −3pp.
6. Only if 4 or 5 is positive: FixMatch-style consistency or SPDIM-style label-shift correction.

Caveats:
- With n=9, only ≥~3pp mean effects are reliably detectable (from the problem brief). Most published increments here are 1-2pp, so expect nulls and pre-register the kill thresholds.
- Confirmation bias is worst on the weak subjects 2/5/6 (57-60%). For a 4-class problem, ~40% of pseudo-labels will be wrong at those accuracies, so report per-subject deltas and specifically weak-subject deltas [I].

### Gaps
- Effects on weak subjects specifically are unreported in every source retrieved.
- Code for SPDIM/OSPDIM was not located.
- The [K] items (TTA mechanisms, sklearn/pyriemann/braindecode function names, Vidaurre procedure details, Zanini 2018) should be spot-checked before implementation.
