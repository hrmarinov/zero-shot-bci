# Ready-made Python DA / transfer-learning implementations for cross-session EEG (verified inventory, 2026-11-04)

Harness assumed: `predict(X_calib, y_calib, X_eval) -> y_pred` with day-1 labelled (~288 trials, 22 ch, 4 classes) and day-2 fully unlabelled (all trials available at once). "LOCAL" = verified by read-only introspection of `.venv` (pyriemann 0.12, braindecode 1.7.0, moabb 1.5.0). Nothing was installed. Glue-line estimates are my own estimates (not sourced) and assume covariances / tangent vectors already computed by the existing harness.

## Q1. pyriemann 0.12 `pyriemann.transfer` and rotation utilities

### Takeaway
pyriemann 0.12 ships `TLDummy, TLCenter, TLScale, TLRotate, TLEstimator, TLClassifier, TLRegressor, TLSplitter, MDWM, encode_domains, decode_domains` (no `TLStretch`; it is now `TLScale`). Only TLCenter and TLScale are label-free; **TLRotate requires class labels in BOTH domains** (it matches per-class means), so it cannot address the post-whitening rotation with an unlabelled day 2 except via pseudo-labels.

### Cited Findings
- LOCAL: `dir(pyriemann.transfer)` = `['MDWM','TLCenter','TLClassifier','TLDummy','TLEstimator','TLRegressor','TLRotate','TLScale','TLSplitter','decode_domains','encode_domains']`; `TLStretch` does not exist in 0.12. Source file: `.venv/Lib/site-packages/pyriemann/transfer/_estimators.py`, `_tools.py`.
- LOCAL signatures: `TLCenter(target_domain, metric='riemann')`; `TLScale(target_domain, final_dispersion=1.0, centered_data=False, metric='riemann')`; `TLRotate(target_domain, weights=None, metric='euclid', n_jobs=1, expl_var=0.999, n_components=1, n_clusters=3, tol_step=1e-09, maxiter=10000)`; `TLClassifier(target_domain, estimator, domain_weight=None)`; `MDWM(domain_tradeoff, target_domain, metric='riemann', n_jobs=1)`; `TLSplitter(target_domain, cv)`; `encode_domains(X, y, domain)` / `decode_domains(X_enc, y_enc)`.
- LOCAL `encode_domains`: domain is encoded into the label string, e.g. `domain_01/left_hand`; integer classes become strings. All TL transformers take `(X, y_enc)` with X either SPD matrices `(n, C, C)` or tangent vectors `(n, n_ts)` (matrix and tangent-space variants since v0.8).
- LOCAL `TLCenter`: "recenters the matrices from each domain to the identity" (= per-domain whitening; refs Zanini 2017, He & Wu 2019 EA). `fit` only uses the decoded *domains*, never class labels (it fits a `pyriemann.preprocessing.Whitening` per domain) -> unlabelled-safe. Caveat in source: `.transform()` (test time) recenters with the **stored target-domain centre**, whereas `.fit_transform()` recenters each domain with its own mean. So to self-centre an unlabelled day 2, it must be included as its own domain in `fit_transform` (or call `Whitening().fit_transform` directly).
- LOCAL `TLScale`: "stretches the matrices from each domain around their mean so that the dispersion ... is equal to one" (Rodrigues 2018 RPA step 2); domain-only, unlabelled-safe. Note docstring says param `dispersion` but signature is `final_dispersion` (docstring/signature mismatch in 0.12).
- LOCAL `TLRotate._fit_manifold` source: computes `gmean(X_target[y_target == label])` for each target class and the same for each source domain, then `_get_rotation_manifold(M_source, M_target, ...)` -> **needs target labels**. Rotates *source* domains onto the target; target is left unchanged. Docstring: "inputs from each domain must have been centered" first. Tangent-space variant (`_fit_tangentspace`, Bleuzé 2022 TSA) also uses `y_tgt` per class (PCA anchors per class, `n_clusters`), and raises if class counts differ -> also needs target labels.
- LOCAL rotation solvers (private): `pyriemann.optimization.grassmann._get_rotation_manifold(X_source, X_target, weights, metric, ...)` minimises `sum_i w_i d^2(X_target_i, Q X_source_i Q^T)` (Riemannian Procrustes, Boumal manifold optimisation); `_get_rotation_tangentspace(X_source, X_target, expl_var)` is orthogonal Procrustes `||X_s Q - X_t||_F`. Both are usable directly with ANY paired matrices (e.g. pseudo-label class means of day 2), but are underscore-private (API not stable).
- LOCAL unsupervised building blocks relevant to the "rotation ambiguity": `pyriemann.geometry.ajd` has `rjd`, `ajd_pham`, `uwedge`, `ajd(X, method=...)` (approximate joint diagonalisation, label-free) and `pyriemann.spatialfilters.AJDC`; `pyriemann.clustering` has `Kmeans`, `GaussianMixture`, `MeanShift`, `KmeansPerClassTransform` (SPD clustering usable for pseudo-labels). No unsupervised `TLRotate` variant exists in 0.12 (searched `_estimators.py`; none found).
- LOCAL `TLClassifier/TLEstimator.fit` decodes and trains on **all** rows weighted by `domain_weight` -> unlabelled target rows would need a dummy class label (would be seen as an extra class), so for unlabelled day 2 do not route it through TLClassifier; train the classifier on day-1 only.
- LOCAL `MDWM`: class centroids = weighted mean of source class means and target class means (`domain_tradeoff`) -> **needs labelled target**; not applicable to unlabelled day 2.
- LOCAL `pyriemann.utils` in 0.12 is a lazy shim re-exporting `pyriemann.geometry.*` (means, etc.); geometry code now lives in `pyriemann.geometry`.
- RPA original paper: [Rodrigues et al. 2018, Riemannian Procrustes analysis](https://hal.archives-ouvertes.fr/hal-01971856) (cited in local docstring).

### Inferences
- Unlabelled options in pyriemann: recentre (TLCenter / Whitening) + stretch (TLScale). This is what the project already does (RA); TLScale is the only untried pyriemann step (~10 lines glue).
- For the rotation: "TLRotate with pseudo-labels" = predict day 2 with the day-1 model, compute per-class day-2 means from pseudo-labels (or from `Kmeans` clusters matched to classes), call `_get_rotation_manifold`, rotate day-1 (or the inverse on day 2), refit/iterate. ~30-40 lines glue. Risk: errors of the pseudo-labels for weak subjects reinforce themselves.
- Glue estimates: TLCenter+TLScale ~10-15 lines; pseudo-label RPA ~35 lines; MDWM n/a.

### Gaps
- Not verified whether TLScale with `centered_data=True` on a single domain behaves as expected numerically (not run, per constraints).

## Q2. skada (scikit-adaptation)

### Takeaway
skada 0.6.0 (PyPI, 2026-06-17) is the most complete sklearn-compatible DA toolbox: ~25 shallow methods (reweighting, mapping incl. CORAL/OT, subspace incl. SA/TCA/TJM/TSL, JDOT, DASVM, OT label-prop) + 9 deep methods (torch/skorch). Target is unlabelled by design. NOT installed locally. SKADA-Bench includes a BCI-IV-2a tangent-space task where CORAL/LinOT gave +6-7pp over no-DA and most others hurt.

### Cited Findings
- Version 0.6.0 released Jun 17 2026, Python >=3.9; extras `subspace`, `deep`, `test`, `doc` — [PyPI skada](https://pypi.org/project/skada/)
- Full API (exact names), 0.6.0 — [skada API](https://scikit-adaptation.github.io/all.html):
  - Reweighting: `DensityReweight, GaussianReweight, DiscriminatorReweight, KLIEPReweight, NearestNeighborReweight, MMDTarSReweight, KMMReweight` (+ `*Adapter` versions)
  - Mapping: `CORAL, OTMapping, EntropicOTMapping, ClassRegularizerOTMapping, LinearOTMapping, MMDLSConSMapping, MultiLinearMongeAlignment` (+ `*Adapter`)
  - Subspace: `SubspaceAlignment, TransferComponentAnalysis, TransferJointMatching, TransferSubspaceLearning` (+ `*Adapter`)
  - Other: `JDOTClassifier, JDOTRegressor, DASVMClassifier, OTLabelProp, JCPOTLabelProp`
  - Pipeline: `make_da_pipeline, Shared, PerDomain, SelectSource, SelectTarget, SelectSourceTarget`
  - Deep (`skada.deep`): `DeepCoral, DeepJDOT, DAN, DANN, CDAN, MCC, CAN, MDD, SPA` (+ `*Loss`, `skada.deep.losses.*`)
  - Unsupervised validation (`skada.metrics`): `SupervisedScorer, ImportanceWeightedScorer, PredictionEntropyScorer, DeepEmbeddedValidation, SoftNeighborhoodDensity, CircularValidation, MixValScorer, MaNoScorer`
  - Model selection: `SourceTargetShuffleSplit, DomainShuffleSplit, StratifiedDomainShuffleSplit, LeaveOneDomainOut`; datasets `DomainAwareDataset, make_shifted_datasets, ...`; utils `source_target_split, per_domain_split, source_target_merge, check_X_y_domain`.
- Conventions: `sample_domain` is "integer >=0 for source and <0 for target"; target labels "are not used when fitting"; predict on target by default via `clf.predict(Xt)` (no `sample_domain` needed except with `PerDomain`); `make_da_pipeline(StandardScaler(), CORALAdapter(), SVC())`; "The DA pipeline can be used with any estimator and any adapter"; reweighting adapters need `.set_fit_request(sample_weight=True)` on the final estimator — [How to use SKADA](https://scikit-adaptation.github.io/auto_examples/plot_how_to_use_skada.html)
- Packing with masked target labels: `ds.pack(as_sources=["s"], as_targets=["t"], mask_target_labels=True)` then `clf.fit(X, y, sample_domain=sample_domain)` — [method comparison example](https://scikit-adaptation.github.io/auto_examples/plot_method_comparison.html). (The exact mask value, believed to be -1, was NOT stated on the pages fetched — unverified.)
- Example gallery (23 examples) has no EEG/MOABB/braindecode example — [skada examples](https://scikit-adaptation.github.io/auto_examples/index.html)
- SKADA-Bench BCI task: BCI Competition IV data, Cov+TangentSpace features (253 dims), 288 samples, 4 classes, 9 adaptation tasks; no-DA 0.55, train-on-target 0.64; CORAL 0.62, LinOT 0.61; Disc.RW 0.56; Dens.RW/KLIEP/MMDTarS ~0.55; MapOT and JDOT 0.47; Gauss.RW 0.25, TCA 0.26, TSL 0.26; DASVM n/a. Deep (ShallowFBCSPNet): no-DA 0.54, DeepCORAL/DeepJDOT 0.54, DAN/MCC 0.53, DANN/SPA 0.41, MDD 0.40. Across datasets CircularValidation correlated best with target accuracy (rho 0.71), IW 0.56, MixVal 0.34 — [SKADA-Bench, arXiv 2407.11676](https://arxiv.org/html/2407.11676); code [skada-bench](https://github.com/scikit-adaptation/skada-bench)
- Same group: Spatio-Temporal Monge Alignment (STMA, test-time / multi-source DA on multivariate signals, PSD alignment) — [arXiv 2407.14303](https://arxiv.org/abs/2407.14303v1); PSDNorm test-time temporal normalisation — [arXiv 2503.04582](https://arxiv.org/abs/2503.04582v1). skada's `MultiLinearMongeAlignment` is listed in the API; whether it is exactly STMA was not verified.

### Inferences
- SKADA-Bench's BCI task is exactly this project's dataset and feature type (paper does not state whether domains are sessions or subjects — gap). Its result says: on tangent vectors, only second-order mapping (CORAL / linear OT, i.e. affine Gaussian maps) helps; nonlinear OT, subspace and reweighting methods collapse. Because RA/recentring already matches first/second moments, expected extra gain of CORAL/LinOT *after* RA is small; worth a 1-hour screen.
- Glue: ~15-20 lines (stack calib+eval, build `sample_domain = [1]*n_cal + [-2]*n_eval`, `y_eval=-1` mask, `make_da_pipeline(TangentSpace-output, XAdapter(), LogisticRegression())`, fit, predict X_eval). Requires `pip install skada` (+POT dependency); deep methods need `skada[deep]` (skorch).
- Hyper-parameter selection without day-2 labels: skada provides `CircularValidation` etc. — directly useful for P4-style per-subject selection.

### Gaps
- Exact target-label mask value (-1) and skada's dependency pins (POT, sklearn version compatibility with sklearn 1.7.2) not verified.
- Whether SKADA-Bench BCI domains are sessions or subjects not stated in the text read.

## Q3. POT `ot.da`

### Takeaway
POT 0.9.7 provides sklearn-style transport classes with `fit(Xs, ys, Xt, yt)`; `yt` optional, unlabelled target marked `-1`. Class-regularised variants use *source* labels only. Out-of-sample transform supported. No class named JDOT in `ot.da` (JDOT lives in skada). Not installed locally.

### Cited Findings
- POT docs version 0.9.7 — [POT docs](https://pythonot.github.io/all.html)
- `ot.da` classes: `BaseTransport, EMDTransport, SinkhornTransport, SinkhornLpl1Transport` (non-convex LpL1 class regularisation), `SinkhornL1l2Transport` (group-lasso class reg.), `EMDLaplaceTransport` (Laplacian reg.), `MappingTransport` (joint coupling + linear or Gaussian-kernel map), `LinearTransport` (closed-form affine Gaussian OT map), `LinearGWTransport`, `UnbalancedSinkhornTransport`, `NearestBrenierPotential` (needs CVXPY), `JCPOTTransport` (multi-source target shift); functions `sinkhorn_lpl1_mm, sinkhorn_l1l2_gl, emd_laplace, distribution_estimation_uniform` — [ot.da](https://pythonot.github.io/gen_modules/ot.da.html)
- `BaseTransport.fit(Xs=None, ys=None, Xt=None, yt=None)`; unlabelled target entries filled with `-1` ("-1 cannot be used as a class label"); `transform` supports out-of-sample via `out_of_sample_map` and `batch_size` — [ot.da](https://pythonot.github.io/gen_modules/ot.da.html)
- `SinkhornTransport` was not listed in the class list returned on that page fetch's first segment; `SinkhornLaplaceTransport` not found (may be removed/renamed) — [ot.da](https://pythonot.github.io/gen_modules/ot.da.html)

### Inferences
- Typical use: map day-1 tangent vectors to day-2 (`ot_obj.fit(Xs=T_cal, ys=y_cal, Xt=T_eval)`, `Xs_mapped = ot_obj.transform(T_cal)`), train classifier on mapped source, predict day 2. ~12-15 lines. `LinearTransport` ~= skada `LinearOTMapping` (the SKADA-Bench winner). With n=288 vs 253 dims, entropic/class-reg OT will need dimension reduction first (PCA) — inference.

### Gaps
- Page only read up to 100k chars; per-class parameter defaults not captured.

## Q4. ADAPT (adapt-python)

### Takeaway
ADAPT (latest setup.py version 0.4.5) has 37 methods but hard-depends on **TensorFlow** (+ cvxopt, scikeras) for install even for shallow methods; skada covers the same shallow methods without TF, so ADAPT is low priority for a torch project.

### Cited Findings
- Methods — feature-based: `PRED, FA, CORAL, SA, TCA, fMMD, DeepCORAL, DANN, ADDA, WDGRL, CDAN, MCD, MDD, CCSA`; instance-based: `LDM, KLIEP, KMM, ULSIF, RULSIF, NearestNeighborsWeighting, IWC, IWN, BalancedWeighting, TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2, WANN`; parameter-based: `LinInt, RegularTransferLR, RegularTransferLC, RegularTransferNN, RegularTransferGP, FineTuning, TransferTreeClassifier, TransferForestClassifier, TransferTreeSelector, TransferForestSelector`; metrics incl. `make_uda_scorer, cov_distance, reverse_validation` — [ADAPT map](https://adapt-python.github.io/adapt/map.html)
- `setup.py`: version 0.4.5, install_requires `numpy, scipy, tensorflow, scikit-learn, cvxopt, scikeras` (no version pins) — [setup.py](https://raw.githubusercontent.com/adapt-python/adapt/master/setup.py)

### Inferences
- From method families (standard definitions, not verified per-class): FA, TrAdaBoost, RegularTransfer*, LinInt, CCSA, FineTuning, Transfer*Tree need labelled target; CORAL/SA/TCA/fMMD/KMM/KLIEP/ULSIF/NNW/DANN/ADDA/MDD/MCD/CDAN/WDGRL are unsupervised-target. The parameter-based ones (`RegularTransferLR/LC`) would only matter if a few day-2 labels were allowed.
- Glue ~15 lines, but installing TF into the torch venv is a real cost.

### Gaps
- Per-class supervision requirements not verified on individual doc pages; PyPI page failed to load (release date unknown).

## Q5. braindecode 1.7 and moabb 1.5 helpers

### Takeaway
braindecode 1.7.0 provides 23 augmentation transforms, cropped-decoding losses, and `exponential_moving_standardize/demean`, but **no domain-adaptation or test-time-adaptation utilities**. moabb 1.5.0 provides `CrossSessionEvaluation/CrossSessionSplitter` (plain train-on-other-session) and no DA/unlabelled-target evaluation mode.

### Cited Findings
- LOCAL `braindecode.augmentation`: `Transform, IdentityTransform, Compose, AugmentedDataLoader, TimeReverse, SignFlip, FTSurrogate, ChannelsShuffle, ChannelsDropout, GaussianNoise, ChannelsSymmetry, SmoothTimeMask, BandstopFilter, FrequencyShift, SensorsRotation, SensorsZRotation, SensorsYRotation, SensorsXRotation, Mixup, SegmentationReconstruction, MaskEncoding, AmplitudeScale, BandRotation, ChannelsReref, functional`.
- LOCAL `braindecode.preprocessing.exponential_moving_standardize(data, factor_new=0.001, init_block_size=None, eps=0.0001)` and `exponential_moving_demean` (per-channel online EMA normalisation; relevant to P3 drift).
- LOCAL `braindecode.training`: `CroppedLoss, CroppedTrialEpochScoring, CroppedTimeSeriesEpochScoring, TimeSeriesLoss, predict_trials, trial_preds_from_window_preds, mixup_criterion, DanceLoss` (`DANCE` is an event-set-prediction model, not domain adaptation, per local docstring).
- LOCAL `braindecode.modules` includes `SubjectLayers` and `ChannelInterpolationLayer`; no AdaBN/TTA module found (grep for alignment/adapt classes found nothing).
- LOCAL `moabb.evaluations`: `CrossSessionEvaluation, CrossSessionSplitter(cv_class=LeaveOneGroupOut, ...), CrossSubjectEvaluation, WithinSessionEvaluation, LearningCurveSplitter, CrossDatasetSplitter, ...`; grep for `calibration_size|unlabel|adaptation` in moabb code found only dataset metadata (BEETL, Kumar2024 "Generic Recentering"), no DA evaluation code. A web search summary claimed MOABB docs describe `calibration_size` and unlabelled-target adaptation modes — **contradicted locally for 1.5.0 and not confirmed on any primary page** (possibly a newer/dev version) — [search result, unverified](https://github.com/NeuroTechX/moabb).

### Inferences
- Test-time adaptation for the convnet (AdaBN = recompute BN stats on day 2, Tent, entropy minimisation) must be hand-written: AdaBN ~10 lines (`model.train()` on BN layers only, forward over X_eval, `eval()`); Tent ~30-40 lines. Augmentations are drop-in via `AugmentedDataLoader` (~10 lines).

### Gaps
- None material beyond the unverified MOABB claim.

## Q6. EEG-specific DA / TTA reference implementations

### Takeaway
The most directly reusable EEG-specific code is Wimpff et al.'s **eeg-otta** (MIT; online alignment EA/RA + adaptive BN + entropy minimisation on BCI IV 2a, incl. cross-session). It reports small cross-session gains (76.2% -> 79.7%) over a strong source model, mainly from BN adaptation + entropy minimisation, not alignment.

### Cited Findings
- Repo [martinwimpff/eeg-otta](https://www.github.com/martinwimpff/eeg-otta): MIT, `train_source_model.py` / `run_adaptation.py --config tta*`, settings "cross-session or cross-subject / cross-subject continual", based on mariodoebler/test-time-adaptation; installs via `pip install .` / Poetry; model BaseNet.
- Methods evaluated: EA and RA with linear or EMA weighting, adaptive BN (BN-1, BN-0.5, BN-EMA), entropy minimisation with label smoothing; buffer of 32 unlabelled test samples, no source data at adaptation time. BCI IV 2a **cross-session**: source 76.16%, EA 75.43%, RA 76.91%, BN-1 77.72%, RA(EMA)-BN-1 78.29%, +EM 79.74 ± 0.57%. Cross-subject: 57.38% -> 67.31% (continual 69.01%) — [Wimpff et al., arXiv 2311.18520](https://arxiv.org/html/2311.18520)
- skada-bench (above) and STMA/PSDNorm papers from the skada group — [skada-bench](https://github.com/scikit-adaptation/skada-bench), [STMA](https://arxiv.org/abs/2407.14303v1)

### Inferences
- The project's per-session RA + per-session BN result (+11.4pp convnet) already corresponds to eeg-otta's RA + BN-1; the remaining eeg-otta ingredient is entropy minimisation (+~1.5pp cross-session in their numbers). Porting EM/Tent into the existing torch harness is ~40 lines and avoids adopting their framework (hydra-style configs, BaseNet).
- Fastest screening order (cheapest glue first): (1) pyriemann TLScale after recentring (~10 lines), (2) skada CORAL / LinearOTMapping on tangent vectors (~20 lines + install), (3) pseudo-label RPA rotation with pyriemann private Procrustes (~35 lines), (4) entropy-minimisation TTA on the convnet (~40 lines), (5) skada OTLabelProp / JDOT / DASVM (~20 lines, SKADA-Bench suggests these hurt on BCI).

### Gaps
- Did not find verified repos named "eegtta" or "TL-for-BCI"; did not inspect eeg-otta source (method list from paper, not code). The 2025 Wimpff/Aristimunha/Chevallier continual fine-tuning code link was not found.
- No verification of other EEG DA collections (e.g. Dongrui Wu's group code, TL4BCI) within the tool budget.
