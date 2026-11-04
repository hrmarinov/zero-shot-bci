# P1: Recovering an unknown per-session transform without new-session labels: calibration transfer (chemometrics), blind equalisation/BSS (communications), closed-form UDA (vision), mapped to EEG MI (BCI IV-2a, train-session -> eval-session)

Legend: **[V]** = verified this session from a primary source (paper full text, or by inspecting installed library source code). **[S]** = seen only in a search-engine abstract or snippet. **[I]** = inference or prior knowledge, not verified this session. Constraint shorthand: **L0** = needs zero eval-session labels. **L-cal** = uses only the calibration (train) session's labels. **XS** = some computation compares two different subjects. XS methods are disallowed.

Project baseline to beat (from 00_problem_abstraction.md): per-session Riemannian re-centering is already in place. It gives CSP+LDA 66.9% and ShallowConvNet 74.7% (4-class, IV-2a, session T -> session E). After re-centering, the remaining shift is an orthogonal rotation U (C -> U C U^T), plus class-conditional shifts.

---

## Q1. Chemometrics calibration transfer (DS, PDS, slope/bias, OSC, TCA, di-PLS, standard-free): which have a label-free form usable here, and what Python exists?

### Takeaway
Classical calibration transfer (DS, PDS) needs **paired transfer standards**: the same physical samples measured on both instruments. EEG has no paired trials across sessions, so DS and PDS do not apply directly. The only analogue is to treat session-start resting or eyes-open epochs as "standards" [I]. The usable chemometrics ideas are the label-free ones. di-PLS needs source X,y plus target X only, and is pip-installable (`diPLSlib`). Standard-free methods such as CTAI and MSC/SNV reduce to per-session affine standardisation, which the project's re-centering largely already does.

### Cited Findings
- di-PLS (Nikzad-Langerodi et al., Anal. Chem. 2018, 90(11):6693-6701) is packaged as `diPLSlib`. It offers a scikit-learn-style API: `diPLSlib.models.DIPLS`, `GCTPLS` (graph-based calibration transfer), `KDAPLS` (kernel domain-adaptive PLS), and `EDPLS`, with the call pattern `m.fit(X, y, X_source, X_target)`. The latest PyPI version is 2.5.0 [S] — [PyPI diPLSlib](https://pypi.org/project/diPLSlib/), [docs](https://di-pls.readthedocs.io)
- di-PLS aligns source and target feature distributions in the PLS latent space. The search summary says target-domain labels are not needed for DIPLS, but GCT-PLS uses **paired** samples across domains [S] — [PyPI diPLSlib](https://pypi.org/project/diPLSlib/); a third-party method page also exists: [nirs4all di_pls](https://nirs4all-methods.readthedocs.io/en/latest/methods/di_pls.html)
- Calibration-transfer methods fall into two groups: those that need transfer standards and those that do not. CTAI (Calibration Transfer based on Affine Invariance) needs no transfer standards and outperformed five other methods on two NIR datasets [S] — [Zhao et al. 2019, Molecules, PMC6539942](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6539942/)
- Multiplicative scatter correction needs no standards but "is difficult to handle complex situations" [S] — [PMC6539942](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6539942/)
- Slope/bias correction is "the oldest and ever used method" of calibration transfer. With about 10 samples and low r², the slope correction may be unjustified because its confidence interval is wide. It is applied to **predictions**, and estimating it requires reference values (labels) on the slave instrument [S] — [Dardenne group review 2004](https://www.cra.wallonie.be/uploads/2004/01/U15-0202-dardenne-2004.pdf)
- FIR-filter standard-free transfer was generally successful but less effective than PDS when a master-slave bias was present [S] — [USPTO 6441388 citing Blank et al.](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6441388)

### Inferences
- **DS/PDS** [I]: these fit a (banded) linear map F with X_slave F ≈ X_master on paired standards. In EEG, a whole-session-to-whole-session channel-space map F (C×C) is exactly the congruence the project wants. Without pairing, F is unidentifiable beyond second moments, and second-moment matching is already re-centering. PDS's "piecewise/banded" prior maps to a **spatial locality prior**: F restricted to neighbouring electrodes, e.g. a sparse Laplacian-banded matrix. This prior is genuinely new for EEG, but any fit would still need a pairing signal. Candidate pairing signals are class-conditional means using source labels plus target pseudo-labels, or rest epochs. Verdict: L0 only with pseudo-labels; not XS; no off-the-shelf EEG implementation.
- **Slope/bias correction** [I]: label-free version = per-session standardisation of classifier outputs or features (z-scoring the eval session's LDA scores). This equals the "domain-specific standardisation (DSS)" baseline. FBCSP+DSS+LDA scored 71.3% inter-session on BNCI2014001 (see Q4) [V]. That makes it a cheap, already-proven sibling of re-centering. It is worth adding on top of CSP+LDA (minutes of effort).
- **OSC** (orthogonal signal correction) [I]: removes X variance orthogonal to y. It uses only calibration labels (L-cal). Its EEG analogue is removing tangent-space directions that carry no class information but large session variance. A domain-indicator variant (y = session id) is close to Stationary Subspace Analysis (see Q3). It can be done in minutes with sklearn PLS; no dedicated maintained OSC package was verified (the `nippy` package is NIR preprocessing (SNV/MSC/derivatives), not transfer [I, not verified]).
- **di-PLS on tangent vectors** [I]: X_s = train-session tangent vectors (253-dim for 22 channels), y = one-hot classes (PLS-DA), X_t = eval-session tangent vectors (unlabelled). It is L-cal + L0 and not XS. Effort is about 2-4 h, including wrapping PLS-DA into an argmax classifier. Published EEG/BCI use: none found [I].
- `pycaltransfer`: I could not verify that this package exists or what its API is. Treat as unverified.

### Gaps
- No source found applying di-PLS, PDS, or CTAI to EEG/BCI. Absence of evidence after one search pass, not proof.
- I could not confirm from the diPLSlib docs whether the target X must be paired with the source X for DIPLS. The search summary says unpaired target X is fine for DIPLS and pairing is needed for GCT-PLS. Check the docstring before use.
- Mishra et al. 2021 and Feudale et al. 2002 reviews were not fetched.

---

## Q2. Riemannian Procrustes (stretch + rotation), unsupervised rotation estimation, and what pyriemann 0.12 `pyriemann.transfer` actually provides

### Takeaway
pyriemann 0.12 (installed in this repo's `.venv`) provides `TLCenter`, `TLScale` (the "stretch" step; there is **no `TLStretch`** class), `TLRotate`, `MDWM`, `TLClassifier`/`TLEstimator`/`TLRegressor`, `TLDummy`, `TLSplitter`, and `encode_domains`/`decode_domains` [V]. `TLCenter` and `TLScale` are label-free. **`TLRotate` requires class labels in BOTH domains**, because it matches class means or class-wise PCA-cluster anchors [V]. A label-free rotation therefore needs either (a) pseudo-labels on the eval session or (b) unsupervised anchors (Bleuzé's TSA notes this is possible but fragile) or (c) Wasserstein-Procrustes / OT-coupled rotation. Published evidence on IV-2a inter-session suggests rotation-type alignment gives little or no gain over well-normalised baselines.

### Cited Findings
- `pyriemann.transfer` in pyriemann 0.12 exports: `MDWM, TLCenter, TLClassifier, TLDummy, TLEstimator, TLRegressor, TLRotate, TLScale, TLSplitter, decode_domains, encode_domains` [V, by introspecting the installed package] — [pyriemann API docs](https://pyriemann.readthedocs.io/en/latest/api.html)
- `TLCenter(target_domain, metric='riemann')` re-centers each domain's matrices to identity (whitening). On tangent vectors it re-centers to zero. `fit_transform` should be applied on the training set and `transform` on the target test domain [V, docstring] — [pyriemann docs](https://pyriemann.readthedocs.io/en/latest/api.html)
- `TLScale(target_domain, final_dispersion=1.0, centered_data=False, metric='riemann')` stretches each domain around its mean to a fixed dispersion. On tangent vectors it rescales so the mean norm is 1. It is label-free, and properly defined only for the riemann metric [V, docstring].
- `TLRotate(target_domain, weights=None, metric='euclid', n_jobs=1, expl_var=0.999, n_components=1, n_clusters=3, tol_step=1e-9, maxiter=10000)` rotates each *source* domain to match its **class means** with the target domain's class means. On the manifold it uses the RPA loss (Rodrigues 2019) optimised with manifold optimisation (Boumal). In tangent space it follows Bleuzé et al. 2022. Inputs must be centered first [V, docstring + source].
- In `TLRotate._fit_tangentspace`, target labels `y_tgt` define `n_classes` and the per-class anchors. A source domain whose class count differs raises a ValueError. Target labels are therefore mandatory in the shipped implementation [V, source `pyriemann/transfer/_estimators.py` lines ~706-760].
- `MDWM(domain_tradeoff, target_domain, metric='riemann')` builds class centroids as a weighted mix of source and target class means. It therefore needs **target labels** [V, docstring].
- Original RPA is semi-supervised. It needs at least one labelled target sample per class, and "the class label information is only needed in computing U" (the rotation) [S] — [Rodrigues et al., IEEE TBME 2019, author PDF](https://plcrodrigues.github.io/files/paper-ieee-tbme-2018.pdf). Kremer et al. 2024 likewise describe RPA's centering and stretching as unsupervised and its rotation as supervised [S] — [arXiv 2403.15415](https://arxiv.org/pdf/2403.15415)
- TSA (Bleuzé, Mattout, Congedo, Front. Hum. Neurosci. 2022) does re-centering, stretching, and a rotation via Procrustes (SVD of a cross-product matrix of anchor points) in tangent space. Anchors are class means, or class-wise PCA-split cluster means (3 clusters per class in their results). The authors say this "is also possible with unlabeled data" for unsupervised TL. In that case at least two PCA components are needed, and the reduced representation "may be as well totally inaccurate, depending on the data, especially for unsupervised TL" [V, full text] — [Bleuzé et al. 2022](https://www.frontiersin.org/articles/10.3389/fnhum.2022.1049985)
- TSA was tested on 18 databases (349 subjects; ERP, MI, SSVEP) with an SVM. The evaluation was **cross-subject pairs**, i.e. XS. TSA improved ERP and showed "no deterioration" for MI and SSVEP versus train-test. It gave +2.7% globally over RPA, and significantly beat RPA in 11/18 databases. BNCI2014001 (22 ch, 288 trials, 2 sessions, 9 subjects) is among the MI databases [V, full text].
- SPDAlign (Li, …, Kawanabe, Kawato, Ding; arXiv 2610.06315, Oct 2026) aligns domain means and "corrects global rotations across domains" using **Wasserstein Procrustes**, framed as unsupervised and without labelled calibration data. Datasets and numbers were not visible in the abstract [S] — [papers.cool/arxiv/2610.06315](https://papers.cool/arxiv/2610.06315)
- Mixing-model Stiefel Adaptation (MSA; Collas, Flamary, Gramfort, arXiv 2402.03345) learns two Stiefel matrices to align source and target covariances jointly with a predictor. It is **weakly supervised**: a small labelled target fraction plus abundant unlabelled data. It was shown on MEG brain-age regression (Cam-CAN), not BCI [S] — [arXiv 2402.03345](https://arxiv.org/abs/2402.03345)
- Kobler et al. (NeurIPS 2022) give BNCI2014001 **inter-session** balanced accuracy for URPA+MDM (unsupervised RPA) of **59.5% (±2.7)**. This is below non-adapted FB+TSM+LR at 69.8% and below TSMNet at 69.0% [V, Table 5 full text] — [arXiv 2206.01323](https://arxiv.org/abs/2206.01323)

### Inferences
- **Pseudo-label RPA/TSA (self-training rotation)** [I]: (1) Re-center both sessions (TLCenter) and optionally apply TLScale. (2) Train the classifier on the train session and predict the eval session. (3) Call `TLRotate` with the eval session as one domain with *pseudo-labels*. Which domain plays "target" matters: pyriemann rotates *sources toward target*. To keep the trained classifier fixed, make the labelled train session the target and rotate the eval session as a "source" with pseudo-labels. (4) Iterate 2-3 times, optionally keeping only confident trials. Properties: L-cal + L0, not XS. Effort is about 3-5 h with the existing CSP+LDA/MDM pipeline. Risk: confirmation bias. At 67-75% accuracy, a quarter or more of pseudo-labels are wrong, so the rotation may lock in errors.
- **Unsupervised anchor TSA** [I]: anchors come from k-means clusters or PCA-split clusters of the *unlabelled* eval tangent vectors, matched to the train session's clusters. Matching needs a permutation, e.g. Hungarian on anchor norms or on a sorted spectrum. Properties: L0, not XS, about 4 h. Bleuzé themselves flag it as unreliable [V].
- **Eigenbasis matching / Wasserstein Procrustes** [I]: in tangent space after centering, estimate U by alternating (i) an OT coupling (Sinkhorn, POT `ot.sinkhorn`) between eval and train tangent clouds and (ii) orthogonal Procrustes (`scipy.linalg.orthogonal_procrustes`) on the coupled pairs. This is Grave et al. 2019's word-embedding alignment algorithm (NLP field), now used by SPDAlign. A channel-space U (22×22) acting by congruence is more constrained, and better posed, than a free rotation of the 253-dim tangent space. Restrict to congruence by optimising U in O(22) via the RPA loss with OT-weighted pairs. Properties: L0, not XS, about 1 day for a correct congruence-constrained version, about 4 h for the free tangent-space version.
- **Headroom warning** [I, based on V numbers]: on BNCI2014001 inter-session, unsupervised RPA (59.5%) was **worse** than no adaptation with FB+TSM+LR (69.8%). Spatial covariance alignment beyond re-centering has not shown a large gain within one subject across sessions. Likely reason: the same cap and the same person give a small rotation, so estimation noise dominates. A cheap diagnostic is worth running *before* any rotation method. Use the oracle rotation: fit TLRotate with the TRUE eval labels and see how much accuracy it buys. If the oracle gain is under about 2pp, kill the whole rotation family.

### Gaps
- No primary source found for a fully label-free rotation step evaluated on IV-2a **within-subject cross-session**.
- The SPDAlign full text (datasets, IV-2a numbers, code) was not fetched.
- The Rodrigues 2019 RPA paper was not fetched in full; the per-dataset numbers for its cross-session experiments are unverified.

---

## Q3. Blind equalisation / blind source separation (ICA, SOBI, AJD): can two sessions' sources be aligned by matching components?

### Takeaway
pyriemann ships `AJDC` (approximate joint diagonalisation of Fourier cospectra) and the AJD solvers `rjd`, `ajd_pham`, `uwedge` [V]. AJDC explicitly supports multi-condition and "group BSS" use [V]. Running AJD jointly on **one subject's two sessions** (allowed) yields a shared demixing matrix whose sources can be compared across sessions. This is a genuinely under-tested route for IV-2a cross-session. Stationary Subspace Analysis (SSA) is the closest published BCI analogue. Its unsupervised form needs no labels; the cross-subject variant (ssCSP) is XS and disallowed.

### Cited Findings
- `pyriemann.spatialfilters.AJDC(window=128, overlap=0.5, fmin=None, fmax=None, fs=None, dim_red=None)` does "approximate joint diagonalization of Fourier cospectral matrices" for second-order-statistics BSS. It can be applied to a single subject, to several subjects (group BSS), or to "several experimental conditions (for eg, baseline versus task)". `pyriemann.utils.ajd` (deprecated in favour of `pyriemann.geometry.ajd`, to be removed in 0.14) provides `ajd`, `ajd_pham`, `rjd`, `uwedge` [V, local introspection of pyriemann 0.12] — [pyriemann docs](https://pyriemann.readthedocs.io/en/latest/api.html)
- SSA (von Bünau et al., 2009) is an unsupervised split of a multivariate time series into stationary and non-stationary subspaces. Applied to EEG (2010), restricting the BCI to stationary sources "can significantly increase the performance" [S] — [SSA record](https://pure.korea.ac.kr/en/publications/stationary-subspace-analysis/), search summary
- Samek, Meinecke, Müller 2013 (ssCSP) remove the most non-stationary directions estimated from **other users'** data. This is a cross-subject (XS) design [S] — [arXiv 1209.4115](https://ar5iv.arxiv.org/html/1209.4115)

### Inferences
- **Two-session AJD (shared demixing)** [I]: (1) Stack band-limited covariances, or cospectra (AJDC), from both sessions of one subject. (2) Jointly diagonalise them to get a single unmixing B. (3) In source space, apply per-session per-source variance normalisation, i.e. diagonal re-scaling. The diagonal model is exactly what blind equalisation assumes: a fixed mixing with per-source gain changes. If the residual transform after whitening is "same sources, re-weighted", this recovers it with no labels. Properties: L0, not XS, 3-6 h with `AJDC` or `ajd_pham`. Kill criterion: if the per-source variance ratios between sessions are about 1 after re-centering (no structure), or if accuracy is not above re-centering + CSP+LDA by at least 2pp in the screen.
- **Per-session ICA + component matching** [I]: run ICA (MNE `mne.preprocessing.ICA` with picard/fastica, or sklearn `FastICA`) per session. Then match components across sessions by spatial-pattern correlation using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), similar to EEGLAB's CORRMAP template matching (recalled, not verified). The matched permutation-plus-scaling gives an explicit session-to-session channel transform B_E^{-1} P D B_T. Properties: L0, not XS, about 4-8 h. Risk: ICA on 22-channel, about 288-trial MI data is unstable, and sign/permutation errors give catastrophic failures. Kill criterion: median matched-pattern |correlation| < 0.8 for the top-10 components.
- **Unsupervised SSA between the two sessions of one subject** [I]: treat each session (or each run) as an epoch set, find the stationary subspace across runs/sessions, and train CSP inside it. Properties: L0 (labels only for the downstream CSP on train), not XS. No maintained Python SSA package is known to me; it is about 50 lines from the 2009 paper's objective (KL between epoch Gaussians) or via pymanopt. About 1 day.

### Gaps
- No primary source found reporting ICA/AJD source matching for **within-subject cross-session** MI on IV-2a.
- SSA numbers on IV-2a cross-session were not found.

---

## Q4. Vision/ML closed-form UDA (CORAL, Subspace Alignment, GFK, TCA, optimal transport incl. class-regularised OT and JDOT): libraries, and published EEG MI cross-session numbers

### Takeaway
OT-based DA has been tested on MI cross-session. It was **not better than simple Euclidean Alignment** (Peterson et al. TBME 2022). SPD-manifold OT on BNCI2014001 inter-session (66.8%) did not beat non-adapted FB+TSM+LR (69.8%) or domain-specific-standardised FBCSP (71.3%) (Kobler 2022 table). CORAL, SA, and TCA are all one-liners in skada/ADAPT/POT, so screening is cheap. Prior published evidence predicts small gains on top of the re-centering the project already does.

### Cited Findings
- Peterson et al. (IEEE TBME 2022) use **backward OTDA**: they transport new-session data to the calibration session so the classifier is not retrained. Features are CSP + LDA; the method uses POT and Sinkhorn. BOTDA-S is plain entropic Sinkhorn (no labels). BOTDA-GL adds a group-lasso class regulariser using source labels; in the online scenario it also uses current-trial labels. Code: github.com/vpeterson/otda-mibci [V, full text] — [Peterson et al. 2022 PDF](https://sinc.unl.edu.ar/sinc-publications/2022/PNWGLMS22/sinc_PNWGLMS22.pdf)
- Peterson et al.: block-wise cross-session accuracy, averaged across runs [V, Tables 1-2]. The paper's statistical tests show BOTDA-S "competitive" with RPA/EA (p > 0.07).

  | Method | Dataset-1 (own 10-subject set) | Dataset-2 (BNCI2015001) |
  |---|---|---|
  | No adaptation (SC) | 68.4% | 75.6% |
  | EA | 73.6% | 79.9% |
  | RPA | 72.1% | 78.6% |
  | BOTDA-S | 71.5% | 77.6% |
  | BOTDA-GL | 71.1% | 76.2% |
  | FOTDA-S | 71.4% | 77.4% |
  | Retraining (SR) | 72.9% | 80.4% |

  The headline 90.23% is the sample-wise scenario with current-trial labels. **BCI IV-2a was not used.**
- Kobler et al. (NeurIPS 2022; code github.com/rkobler/TSMNet) report BNCI2014001 = IV-2a, 4-class, **inter-session**, balanced accuracy, models given only the same subject's data (not XS) [V, Table 5]. Their ShallowConvNet baseline is much weaker than this project's 74.7%, so absolute numbers are protocol-dependent.

  | Method | BNCI2014001 inter-session |
  |---|---|
  | FBCSP+SVM | 60.6% |
  | FB+TSM+LR | 69.8% |
  | EEGNet | 41.8% |
  | ShConvNet | 51.3% |
  | FBCSP+DSS+LDA | **71.3%** |
  | URPA+MDM | 59.5% |
  | SPDOT+TSM+SVM (OT on SPD manifold, Yair et al.) | 66.8% |
  | EEGNet+DANN | 50.0% |
  | ShConvNet+DANN | 51.6% |
  | TSMNet (SPD domain-specific momentum BN) | 69.0% |

  Grand average over 5 MI datasets inter-session: TSMNet 54.7%. Ablation: removing domain-specific BN costs 3.9%.
- Deep OT on SPD manifolds (Ju & Guan, arXiv 2201.05745) evaluates unsupervised and semi-supervised cross-session MI on KU, BNCI2014001, and BNCI2015001. Numbers were not extracted [S] — [arXiv 2201.05745](https://arxiv.org/pdf/2201.05745)
- A multi-source OT framework (J Neural Eng 2024) reports 73.69% on IV-2a and 70.83% on IV-2b without target labels. It aligns *multiple source domains* using resting-state data. The search summary does not confirm whether the sources are other subjects; if they are, it is **XS → disallowed** [S] — [PubMed/BVS record 38232381](https://pesquisa.bvsalud.org/bolivia/resource/es/mdl-38232381)
- A Siamese deep DA framework with channel normalisation + Euclidean Alignment reports 82.01% on IV-2a (IIA) and 87.52% on IIB, cross-session [S, abstract only] — [arXiv 2202.09559](https://arxiv.org/abs/2202.09559)
- A deep DA framework with correlation alignment (DDAF-CORAL) reports mean kappa 0.761 cross-session MI. The dataset is not stated in the snippet [S] — [HIT scholar record](https://scholar.hit.edu.cn/en/publications/a-deep-domain-adaptation-framework-with-correlation-alignment-for/)
- Euclidean Alignment (He & Wu) is fully unsupervised and was evaluated on IV-2a among others [S] — [IEEE Brain newsletter 2020](https://brain.ieee.org/newsletter/2020-issue-1/transfer-learning-for-brain-computer-interfaces-euclidean-alignment-and-label-alignment/); review: [Wu et al., arXiv 2004.06286](https://arxiv.org/pdf/2004.06286)
- skada (scikit-adaptation) has a benchmark, SKADA-Bench, published in TMLR (07/2025) [S] — [skada-bench GitHub](https://github.com/scikit-adaptation/skada-bench)

### Inferences
Library class names below come from my knowledge of POT/skada/ADAPT and were **not verified this session** [I]. The skada API page returned 404. None of POT/skada/ADAPT is installed in `.venv`; `pip install pot skada adapt` would be required.
- **CORAL** (Sun et al. 2016): whitens source features and re-colours them with target feature covariance. Implementations: `skada.CORAL` / `skada.CORALAdapter`, `adapt.feature_based.CORAL`. EEG mapping: on tangent vectors or CSP log-variance features. On channel covariances it is essentially re-centering (EA is CORAL-like in channel space), so expect about 0 gain there. Properties: L0, not XS, under 1 h.
- **Subspace Alignment** (Fernando et al. 2013): PCA bases P_s and P_t, with source projected through P_s P_s^T P_t. Implementation: `skada.SubspaceAlignment`. This is the closest vision analogue of "recover the rotation": it estimates an orthogonal-ish alignment between principal subspaces. Properties: L0, not XS, under 1 h on tangent vectors (choose d about 10-30).
- **GFK** (Gong et al. 2012): no maintained Python package known to me; skip (low priority).
- **TCA**: `skada.TransferComponentAnalysis`, `adapt.feature_based.TCA`. Properties: L0, not XS, about 1 h.
- **OT-DA**:
  - `ot.da.SinkhornTransport` (L0), `ot.da.EMDTransport` (L0).
  - `ot.da.SinkhornLpl1Transport` / `SinkhornL1l2Transport`: class-regularised using source labels only (L-cal).
  - `ot.da.LinearTransport`: closed-form Gaussian/Bures map. A symmetric map, so it **cannot represent a rotation**; it is CORAL-like.
  - skada equivalents: `OTMapping`, `EntropicOTMapping`, `ClassRegularizerOTMapping`, `LinearOTMapping`.
  - **JDOT**: `skada.JDOTClassifier`/`JDOTRegressor`, `ot.da` examples. L-cal + L0.
  - Effort: about 1-2 h each on tangent vectors or CSP features. Backward mode (transport eval → train, keep the classifier fixed) is the Peterson recipe.
- On the published evidence above, OT/CORAL/SA on top of an aligned (EA/re-centered) pipeline are each **expected to move accuracy by about 0-2pp** on IV-2a cross-session. They are cheap to screen as a batch. Kill criterion: below +2pp mean over 9 subjects versus re-centering + CSP+LDA (66.9%), or any subject losing more than 5pp.

### Gaps
- No primary-source IV-2a **within-subject T→E** numbers were found for plain CORAL, SA, TCA, or JDOT on shallow (non-deep) pipelines.
- The skada API listing could not be retrieved (404), so exact class names are unverified.
- The Ju & Guan per-dataset numbers were not extracted.

---

## Q5. Already applied to EEG/BCI vs genuinely untried, and a consolidated screening catalogue

### Takeaway
Already applied to EEG MI with numbers: EA, RPA/URPA, TSA, OT-DA (forward/backward, Sinkhorn/group-lasso), SPD-OT, SPD domain-specific BN, DANN, CORAL-in-deep-nets, and SSA/ssCSP. Apparently **untried** for within-subject cross-session MI, with a label-free form that is not XS: di-PLS on tangent features, banded/local PDS-style congruence, two-session AJD with per-source gain equalisation, and congruence-constrained Wasserstein Procrustes. SPDAlign (Oct 2026) may already cover the last one. The strongest prior evidence says IV-2a cross-session alignment headroom beyond per-session normalisation is small. The first experiment should be an **oracle-rotation headroom test**.

### Cited Findings
- Unsupervised RPA lost to the non-adapted tangent-space baseline on BNCI2014001 inter-session (59.5% vs 69.8%). The best shallow aligned method was domain-specific standardisation (71.3%) [V] — [Kobler et al. 2022](https://arxiv.org/abs/2206.01323)
- OT-DA variants did not beat EA in block-wise cross-session MI [V] — [Peterson et al. 2022](https://sinc.unl.edu.ar/sinc-publications/2022/PNWGLMS22/sinc_PNWGLMS22.pdf)
- TSA's unsupervised anchor variant is described as possible but unreliable [V] — [Bleuzé et al. 2022](https://www.frontiersin.org/articles/10.3389/fnhum.2022.1049985)
- Wasserstein-Procrustes rotation correction for EEG SPD alignment appears in SPDAlign (2026) [S] — [arXiv 2610.06315](https://papers.cool/arxiv/2610.06315)

### Inferences: screening catalogue
Effort is hours to screen on IV-2a T→E with the existing pipeline. Unless noted, every method below compares one subject's two sessions only, so none is XS. All rows are [I] except where a [V] number is cited.

**Tier 1: screen first (cheap, L0 or L-cal)**

0. **Oracle-rotation headroom test** (diagnostic)
   - Field: none (control).
   - Mechanism: fit `TLRotate` using TRUE eval labels and measure the accuracy gain.
   - Space: SPD or tangent.
   - Labels: uses eval labels, so it is an upper bound only.
   - Python: `pyriemann.transfer.TLRotate` [V].
   - Effort: about 1 h.
   - Kill: an oracle gain under 2pp kills rows 2, 5, and 9.

1. **Feature/score standardisation per session (slope-bias, DSS)**
   - Field: chemometrics (slope/bias); BCI (DSS).
   - Mechanism: z-score features or LDA scores within each session.
   - Space: feature space.
   - Labels: L0.
   - Python: sklearn `StandardScaler` per session.
   - Effort: under 1 h.
   - Published: FBCSP+DSS+LDA 71.3% on IV-2a inter-session [V].
   - Kill: below +1pp over re-centering + CSP+LDA.

2. **Pseudo-label RPA/TSA rotation (self-training)**
   - Field: BCI (Rodrigues; Bleuzé) + self-training.
   - Mechanism: rotate the eval session onto train-session class means estimated from pseudo-labels; iterate.
   - Space: SPD or tangent.
   - Labels: L-cal + L0.
   - Python: `pyriemann.transfer.TLCenter`, `TLScale`, `TLRotate` [V].
   - Effort: 3-5 h.
   - Published: TSA +2.7% over RPA, cross-subject only [V]. URPA 59.5% on IV-2a inter-session [V].
   - Kill: below +2pp, or fails the oracle test.

3. **TLScale (stretch) after centering**
   - Field: BCI (RPA step 2).
   - Mechanism: equalise per-session dispersion.
   - Space: SPD.
   - Labels: L0.
   - Python: `pyriemann.transfer.TLScale` [V].
   - Effort: under 1 h.
   - Kill: below +1pp.

4. **Subspace Alignment / CORAL / TCA on tangent or CSP features**
   - Field: vision UDA.
   - Mechanism: align PCA bases (SA), second moments (CORAL), or an MMD-minimising kernel projection (TCA).
   - Space: feature space.
   - Labels: L0.
   - Python: skada `SubspaceAlignment`, `CORAL`, `TransferComponentAnalysis`; ADAPT `CORAL`, `TCA` [I].
   - Effort: 1-2 h total.
   - Published: CORAL only inside deep nets [S].
   - Kill: below +2pp.

5. **OT-DA, backward (Sinkhorn, class-regularised)**
   - Field: ML/vision (Courty) → BCI (Peterson).
   - Mechanism: entropic OT coupling, then barycentric map of eval features onto the train session; the classifier stays fixed.
   - Space: CSP or tangent features.
   - Labels: L0 (Sinkhorn); L-cal (group-lasso).
   - Python: POT `ot.da.SinkhornTransport`, `SinkhornLpl1Transport`; code github.com/vpeterson/otda-mibci [V].
   - Effort: 2 h.
   - Published: BOTDA-S 71.5% / 77.6% vs EA 73.6% / 79.9% (not IV-2a) [V]; SPDOT 66.8% on IV-2a [V].
   - Kill: below EA/re-centering.

6. **JDOT**
   - Field: ML (Courty 2017).
   - Mechanism: OT on the joint feature-label distribution, with target labels inferred by the classifier.
   - Space: feature space.
   - Labels: L-cal + L0.
   - Python: skada `JDOTClassifier` [I].
   - Effort: 2 h.
   - Published: none found for IV-2a.
   - Kill: below +2pp.

**Tier 2: untried or half-tried, moderate effort**

7. **di-PLS (PLS-DA) on tangent vectors**
   - Field: chemometrics.
   - Mechanism: PLS latent space regularised so that source and target latent covariances match.
   - Space: tangent features.
   - Labels: L-cal + L0.
   - Python: `diPLSlib.models.DIPLS` [S].
   - Effort: 2-4 h.
   - Published: none found.
   - Kill: below plain PLS-DA/LDA + re-centering.

8. **Two-session AJD + per-source gain equalisation (blind equalisation)**
   - Field: communications / BSS.
   - Mechanism: a shared unmixing from joint diagonalisation of both sessions' covariances or cospectra; per-session diagonal gains are re-normalised.
   - Space: input signal or SPD.
   - Labels: L0.
   - Python: `pyriemann.spatialfilters.AJDC`, `pyriemann.utils.ajd.ajd_pham`/`rjd` [V].
   - Effort: 3-6 h.
   - Published: none found for within-subject cross-session MI.
   - Kill: below +2pp, or off-diagonality after AJD stays above 0.3.

9. **Congruence-constrained Wasserstein Procrustes**
   - Field: NLP embedding alignment (Grave 2019) → SPDAlign.
   - Mechanism: alternate a Sinkhorn coupling of trials with orthogonal Procrustes for U in O(22) acting by C → U C U^T.
   - Space: SPD (congruence).
   - Labels: L0.
   - Python: POT `ot.sinkhorn` + `scipy.linalg.orthogonal_procrustes` (free version); pymanopt for the congruence version [I].
   - Effort: 4 h (free) to 1 day.
   - Published: SPDAlign abstract only [S].
   - Kill: fails the oracle test, or below +2pp.

10. **ICA per session + Hungarian component matching**
    - Field: communications / BSS + EEG practice.
    - Mechanism: explicit transform B_E^{-1} P D B_T from matched independent components.
    - Space: input signal.
    - Labels: L0.
    - Python: `mne.preprocessing.ICA`, `scipy.optimize.linear_sum_assignment`.
    - Effort: 4-8 h.
    - Published: none found.
    - Kill: matched |r| < 0.8, or below +2pp.

11. **Unsupervised SSA across runs/sessions**
    - Field: ML / BCI (Berlin group).
    - Mechanism: project out the non-stationary subspace, then run CSP.
    - Space: input or SPD.
    - Labels: L0 (CSP uses L-cal).
    - Python: none maintained; custom implementation.
    - Effort: 1 day.
    - Published: EEG gains reported [S]; no IV-2a T→E number found.
    - Kill: below +2pp.

12. **OSC with session-indicator target**
    - Field: chemometrics.
    - Mechanism: remove tangent directions that predict session id but not class.
    - Space: tangent features.
    - Labels: L-cal + session id.
    - Python: custom, on sklearn PLS.
    - Effort: 2 h.
    - Published: none found.
    - Kill: below +1pp.

**Excluded as inapplicable**

- **DS/PDS proper** (chemometrics): needs paired transfer standards, and none exist across EEG sessions. The only proxy would be session-start rest epochs, whose class-free covariance is already used by re-centering.
- **Multi-source OT using other subjects** (JNE 2024, 73.69%): XS, so disallowed.
- **MDWM**: needs eval labels.

Statistical note: with n = 9 subjects, only effects of roughly +4pp or more with a consistent sign are detectable. Screen all Tier-1 rows on CSP+LDA and MDM/TS+LR first, carry only those above +2pp to the ShallowConvNet pipeline, and pre-register.

### Gaps
- No source verified for `pycaltransfer`.
- Not verified: `nippy` scope, and the exact skada and ADAPT class names.
- Peterson BOTDA was not evaluated on IV-2a, and the Ju & Guan IV-2a numbers were not extracted.
- Kobler's "inter-session" numbers use a randomised leave-sessions-out CV within subject. For a 2-session dataset that is effectively T↔E, but it may differ slightly from the official T→E protocol.
- It is unknown whether the residual post-whitening rotation in IV-2a within-subject is large enough to matter. Only the proposed oracle test can settle this.
