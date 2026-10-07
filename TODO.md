# TO-DO — untested openings that could surpass Riemannian alignment

**Bar to beat: 66.9%** mean cross-session accuracy (9 IV-2a subjects, 4 classes, chance
25.0%) — `results/phase2_riemannian_alignment_iv2a.csv`, train on the `0train`
calibration session, test on `1test`, zero labelled eval data.

This file is a prioritised list of things the project's own evidence *points at* but that
have never been run, plus methodology that could be transplanted from other fields. It is
deliberately not a wish list: every item states what would have to be true for it to work,
the cheapest experiment that would falsify it, and a cost estimate.

Companion reading: [`docs/progress_and_direction.md`](docs/progress_and_direction.md)
(what was already tried and closed) and [`docs/state_of_the_art.md`](docs/state_of_the_art.md)
(the literature survey, §1–§11).

---

## 0. Ground rules — apply these before adding anything to the list

**The design filter (this project's single most expensive lesson):** the failure mode is
never "population data", it is *any single computation that directly compares or aligns two
different subjects' covariance structure*. Six independent attempts confirm it
(`docs/progress_and_direction.md` §"The recurring negative result"). Every candidate below
is tagged with how it fares against that filter:

| Tag | Meaning |
|---|---|
| `[within-subject]` | Uses only the target subject's own sessions. No headwind. |
| `[label-free]` | No labels at all at deployment. No headwind, on-theme for "zero-shot". |
| `[pop-data, safe]` | Population data trains a *skill* or sets a *hyperparameter*; no cross-subject comparison inside a loss or correction. |
| `[cross-subject ⚠]` | Requires comparing subject A to subject B inside one computation. Expect a headwind regardless of sophistication. |

**The evidence ladder** (from the project's own methodology lessons — nothing has been
trustworthy without it):

1. Synthetic ground-truth gate (`experiments/stage0_*.py` pattern) — prove the mechanism
   recovers a *known* answer from noisy input.
2. Real data, full 9 subjects, same protocol as Phase 1/2.
3. Paired test on the same trials (McNemar per subject) **and** paired Wilcoxon across
   subjects.
4. Multiple-comparison control, or an explicit statement that the result is exploratory.
5. Per-subject breakdown reported, never just the mean.

**Statistical reality check to internalise before planning anything:** with n=9 subjects the
smallest achievable two-sided Wilcoxon p is **0.0039**. Any single-subject-level effect
smaller than roughly +2pp average is undetectable here. This is why Tier 1 below favours
*structural* wins (new information, new data, larger ceilings) over parameter tweaks.

**Cost scale:** `S` = under an hour · `M` = half a day · `L` = multi-day or hours of GPU.

---

## 1. The strategic finding that should shape the whole plan

**Stop optimising the mean. The mean is being held down by three subjects.**

Phase 2 per subject (aligned), from `results/phase2_riemannian_alignment_iv2a.csv`:

| Subject | 9 | 3 | 1 | 8 | 7 | 4 | 6 | 2 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 84.0% | 81.6% | 80.9% | 80.6% | 78.5% | 60.8% | 49.3% | 44.4% | 42.4% |

Five subjects are already at 78–84%. Three (5, 2, 6) sit at 42–49% — barely above chance,
and they are exactly the "illiterate" users this project set out to help. Lifting *only*
those three to 65% would take the mean from 66.9% to **73.5%** — a larger gain than any
global method in the literature is likely to deliver uniformly, and it is the gain that
actually matters for the stated goal.

Consequence for this TO-DO: **Tier 1 leads with per-subject failure analysis and
subject-specific mechanisms**, not with a better global alignment.

**Open question nobody has asked:** are subjects 2/5/6 bad because their signal is
genuinely weak (in which case no method fixes them, and the honest answer is "predict and
route them elsewhere"), or because the pipeline's fixed choices (8–30 Hz band, 0.5–2.5 s
window, 8 CSP components) are wrong *for them*? The project measured the mean and used
paired tests, but never ran a per-subject diagnostic. That is A1 below.

---

## 2. Priority table

| # | Item | Tag | Cost | Upside | Tier |
|---|---|---|---|---|---|
| A1 | Per-subject failure analysis on the bottom 3 (and a real ceiling estimate) | `[within-subject]` | S–M | 0 to +6.6pp, and it gates everything else | **1** |
| A2 | RPA's stretch + rotation step (published +2.7% over recentering; only recentering implemented) | `[within-subject]` | M | +1–3pp, literature-backed | **1** |
| A3 | Online / transductive test-time adaptation (published 76.2→79.7% on IV-2a cross-session) | `[label-free]` | M–L | +2–4pp | **1** |
| A4 | Per-class prototype alignment on *unlabeled* eval trials (PMANet reports 79.3% on 4-class IV-2a) | `[label-free]` | M–L | +3–8pp if it replicates | **1** |
| A5 | Optimal transport / Wasserstein alignment (documented +2.45pp on this exact dataset, never built) | `[within-subject]` | M | +2pp | **1** |
| A6 | Per-subject hyperparameters (band, window, CSP components) chosen by inner CV | `[within-subject]` | M | +1–4pp, directly tests A1's hypothesis | **1** |
| A7 | Pre-cue baseline as the alignment reference (truly zero-shot) | `[label-free]` | M | 0 to +3pp, plus a genuine zero-shot result | **2** |
| A8 | Within-subject shrinkage: shrink the eval estimate toward *the same subject's* calibration covariance | `[within-subject]` | S | 0 to +2pp; the untested sibling of a proven null | **2** |
| A9 | Filter-bank CSP (FBCSP — standard strong IV-2a baseline, never tried) | `[within-subject]` | M | +2–5pp (within-session benchmark says single-band CSP is not SOTA) | **2** |
| A10 | Canonical Riemannian classifier (MDM / tangent-space + tuned regularisation) *on aligned data* | `[within-subject]` | S–M | +0–2pp, cheap | **2** |
| A11 | Self-training with eval pseudo-labels (closed loop with A2's rotation) | `[label-free]` | M | +1–3pp | **2** |
| A12 | Alignment-estimator variants: Euclidean vs Riemannian mean, per-class alignment, pooled calib+eval reference, geodesic-subspace alignment | `[within-subject]` | S | +0–2pp | **2** |
| A13 | Spatial preprocessing never varied: CSD/Laplacian re-reference, EOG regression instead of ICA, shared vs per-session cleaning | `[within-subject]` | S–M | +0–3pp | **2** |
| A14 | Calibration-trial quality weighting + data augmentation of the 288 trials | `[within-subject]` | M | +1–3pp | **2** |
| A15 | Ensembling (RA + tangent + prototypical) and test-time augmentation | `[within-subject]` | M | +1–3pp | **2** |
| A16 | Out-of-fold stacking (flagged twice in the docs, never built) | `[within-subject]` | M | +1–2pp | **2** |
| A17 | Prototypical embeddings + CSP combination (gated on Phase 6 being "strong enough" — it now is) | `[pop-data, safe]` | M | +1–3pp | **2** |
| A18 | Similarity-selected recentering target (the documented open gap #1) | `[cross-subject ⚠]` | M | 0 to +2pp; informative either way | **3** |
| A19 | Illiteracy predictor wired into the decoder as a conditioning input (open gap #2) | `[pop-data, safe]` | L | 0 to +2pp, high novelty | **3** |
| A20 | Multi-session drift modelling — needs a dataset with >2 sessions (Won et al. 5-day: cited, never downloaded) | `[within-subject]` | L | Unknown, structurally new capability | **3** |
| A21 | Cross-paradigm fingerprint (OpenBMI has MI + ERP + SSVEP for the same 54 subjects) | `[label-free]` | L | Research-grade novelty | **3** |
| A22 | PhysioNet "rest" trials as interleaved drift probes — makes §10.3's "guide star" testable after all | `[label-free]` | M | Speculative | **3** |
| A23 | Small conv nets (EEGNet / ShallowConvNet) with cross-session fine-tuning — current conclusion is literature-only, never tested locally | `[pop-data, safe]` | L | Unknown; benchmark says deep nets can beat CSP within-session | **3** |
| A24 | Per-subject / per-class jitter tolerance instead of a fixed 0.15 s Gaussian (already flagged) | `[within-subject]` | S | +0–1pp | **3** |
| A25 | RIDE multi-component decomposition (the one latency-alignment variant left untried) | `[within-subject]` | M | +0–1pp | **3** |
| A26 | Prototypical open items: subject-5 outlier, `episodes_per_step` tuning, significance vs Phase 2 | `[pop-data, safe]` | M | Closes out the best non-Riemannian result | **3** |
| A27 | Measurement hygiene: multiple-comparison control, power statement, pre-registration | — | S | Not accuracy — validity | **1** |
| A28 | Align the fitted *dynamics*, not the statistics (systems-neuroscience answer, primitives already in the repo) | `[within-subject]` `[label-free]` | M | +0–3pp and a genuinely different axis from RA | **1** |
| A29 | Make **calibration burden** the headline metric — accuracy vs number of labelled trials, not at one fixed budget | `[within-subject]` | S–M | Likely a *better story* than +4.7pp; reframes the whole project | **1** |
| A30 | Per-subject self-supervised pretraining on the target's own unlabeled data (masked channel/time objective) | `[label-free]` | L | The honest test of "wrong idea" vs "wrong data" | **2** |
| A31 | Longevity: accuracy decay over sessions, and "sessions until below X" (blocked on A20) | `[within-subject]` | L | New capability, not a mean shift | **3** |
| A32 | State-space / Mamba-style sequence encoder | `[pop-data, safe]` | L | Unknown; architecture bets are expensive on 9 subjects | **3** |

Cross-disciplinary transplants are in [Part B](#part-b--cross-disciplinary-transplants-bonus),
scored separately.

---

## 3. External evidence that reshaped this list (Neuralink, Oct 2026)

Neuralink's report *Pretraining on 50,000 Hours of Unlabeled Brain Data* is not an EEG
paper, and none of its numbers transfer. It is included because it **independently
corroborates this project's central finding at ~5 orders of magnitude more data**, and
because three of its design choices become new experiments here (A28–A30).
[neuralink.com/updates/pretraining-on-50000-hours](https://neuralink.com/updates/pretraining-on-50000-hours/)

**What they did:** self-supervised pretraining (spatially masked auto-Poisson regression —
mask half the channels, predict the other half) over 50,000 hours of unlabeled
intracortical recordings from clinical-trial participants, on a Mamba / state-space
backbone. One participant alone contributed 9,000 hours / 22.4 billion spikes.

**The corroboration that matters most.** All of their live results come from
*participant-specific* encoders pretrained on that participant's **own** recordings. Pooling
across participants was tried, and their verdict is explicit: *"decoders built from
multi-participant models perform no better than their single-participant counterparts."*
That is precisely the null this project hit six times (SPD shrinkage, eigenvoice, Module A,
and cross-subject meta-training episodes) — now reproduced by a different group, in a
different modality, with four orders of magnitude more data and a foundation-model
architecture. **This is the strongest external evidence yet that the finding is about the
structure of the problem rather than about this project's data being small**, and it is
worth citing whenever the negative result is presented.

**The three transfers:**

| From their report | Becomes |
|---|---|
| *"Intention lives in a stable subspace"* — day-specific variation is not dominant mid-encoder; the **dynamics** stay stable across sessions while raw statistics drift (they cite Karpowicz et al., Nat Commun 2025; Gallego et al., Nat Neurosci 2020) | **A28** — align the fitted dynamical system, not the covariance. This project's `state_space_flow.py` already fits one and has never been used for alignment |
| Data-efficiency as the headline: *"30 seconds of labeled embeddings ≈ 3.5 minutes of raw spikes"*; calibration burden cut from ~10 min/day to ~10 min/week | **A29** — report accuracy against number of labelled trials. The project has no such curve, and may be sitting on an unusually strong version of it |
| Decoder **longevity** (some decoders stable >3 weeks; one usable 20 months after calibration) | **A31** (+A20) — accuracy decay over sessions, which IV-2a's two sessions cannot measure |

**What does not transfer, and why it matters for reading §11 of the survey:** intracortical
arrays give a far higher-SNR, higher-channel-count signal than 22 volume-conducted EEG
channels, and their 50,000 hours are *longitudinal per participant* — a data shape no public
MI dataset has. So this does not contradict the survey's finding that EEG foundation models
lose on motor imagery; it explains it. The revised reading: **self-supervised pretraining
does not lose on MI because it is self-supervised, but because of public-EEG scale, public-EEG
SNR, and cross-subject pooling.** The per-subject and alignment-oriented uses of
self-supervision are exactly the parts that survive that re-reading (A28, A30).

**Caveat on the source:** a company engineering blog, not peer-reviewed, with no independent
replication and a modality-specific metric (bits per second in closed-loop control) that is
not comparable to offline 4-class accuracy. Cite it for *direction*, never for numbers.

---

## Part A — Oversights in the existing work

### A1. No per-subject failure analysis, and no trustworthy ceiling `[within-subject]` **Tier 1**

**The oversight.** Two numbers in the committed results contradict each other and nobody
noticed:

- Phase 1 within-session CV (calibration session): **65.1%**
- Phase 2 aligned cross-session (calib → eval): **66.9%**

The "upper-bound-ish" within-session number is *below* the cross-session number it is
supposed to bound. Either the CV estimate is pessimistic (288 trials, 60-trial folds, CSP
fit inside each fold), or the eval session is simply easier. Nobody knows, which means the
project has no idea how much headroom is left: it cannot say whether 66.9% is 3pp from the
ceiling or 15pp from it. (The one direct measurement on cleaned data — subject 1 at
76.5%/76.6% within-session on either session — suggests the real ceiling is well above
65.1%.)

**Why it could beat 66.9%.** Not directly — it *gates* everything. If the ceiling is ~70%,
Tier-1 method work is pointless. If it is ~85%, there is 18pp on the table and the
per-subject gap is the place to find it.

**First experiment.** For all 9 subjects: within-session repeated CV on the *eval* session
(the relevant ceiling for train-on-calib/test-on-eval), plus a nested-CV estimate that
selects hyperparameters honestly. Then a per-subject breakdown of where Phase 2's errors
sit: class confusion matrix per subject, alpha/mu power and SNR per subject, correlation
between subject-level SNR and subject-level accuracy.

**Cost:** S–M. **Kill criterion:** if subject-level accuracy correlates strongly with a
simple SNR proxy, then subjects 2/5/6 are hardware/signal-limited and only *prediction and
routing* (A19) can help them — which would redirect the entire roadmap.

---

### A2. RPA's stretch and rotation steps `[within-subject]` **Tier 1**

**The oversight.** `docs/state_of_the_art.md` §2 documents Riemannian Procrustes Analysis
as recentering **+ stretching + rotation**, and explicitly states *"we only implemented
recentering, the first and cheapest step"* — with a published **further 2.7% improvement**
over prior Riemannian methods. Only recentering was ever wired into the Phase 2 pipeline.

What *was* built, in the separate tangent-space branch (`src/adaptation/riemannian_icp.py`,
Phase 7): recenter+rescale (66.6% vs its own 65.9% tangent-LDA baseline) and a
**supervised** rotation that used the eval session's *true labels* at n=288 — an oracle
upper bound (68.4%, p=0.25). The deployable version — Procrustes rotation from *matched
class means*, using calibration labels plus **eval pseudo-labels** — is written in the
docstring (`supervised_rotation_from_labels`, lines 173–177: *"Needs labels (or
pseudo-labels) on both source and target"*) and **never called by any experiment**. The
primitives already exist and are unit-validated (Procrustes recovers an exact rotation to
1e-14).

**Why it could beat 66.9%.** It is the canonical, published generalisation of the exact
method that already works here, it is within-subject, and it needs no population data. The
2.7pp claim is on the recentering baseline, not on a tangent-LDA baseline.

**First experiment.** On top of the Phase 2 pipeline: recenter → rescale (match eval
dispersion to calibration dispersion) → rotation via Procrustes on calibration class means
mapped through a first-pass classifier to eval class means (pseudo-labels, EM-style, 2–3
iterations). Report the ablation: +rescale alone, +rotation with *true* eval labels (oracle
reference), +rotation with pseudo-labels (deployable). Cost: M.

**Headwind to check first:** in Phase 7 the *oracle* rotation at realistic few-shot sizes
actively hurt. Verify RPA's rotation behaves differently at full-calibration-sample count
before investing further.

---

### A3. Online / transductive test-time adaptation `[label-free]` **Tier 1**

**The oversight.** `docs/state_of_the_art.md` §3 documents the "hero" option from the
project plan — *Calibration-free online test-time adaptation* (arXiv 2311.18520) — and
reports **IV-2a cross-session 76.2% → 79.7%** using Riemannian Alignment plus running
BatchNorm statistic updates plus entropy minimisation, evaluated online per trial with no
labels. §9 further notes the project's pipeline is *"strictly offline, two-stage,
calibrate-once-deploy-once"*. Nothing online was ever built: the grep for
entropy-minimisation / BN-adaptation / self-training in the codebase returns nothing.

**Honest caveat** (the survey says it, and it should not be glossed over): that paper's
76.2% baseline comes from a deep net pretrained across many subjects, so it is not the same
starting point as a single-subject CSP+LDA. The **+3.6pp is the transferable part**, not the
absolute number.

**Why it could beat 66.9%.** Riemannian alignment is a *one-shot* estimate from all 288
eval trials at once. Online adaptation can (a) refine as trials accumulate, (b) exploit the
eval session's own structure sequentially, and (c) pair with A2/A11. No population data
required.

**First experiment.** Replace LDA with a differentiable logistic regression (or keep CSP for
features and adapt online): run the eval session in trial order, after each trial update
(a) the running whitening/alignment estimate, (b) the classifier's normalisation statistics,
(c) optionally an entropy-minimisation step on the last *k* trials. Predict **before**
updating (no leakage). Compare against the identical pipeline with batch alignment. Cost:
M–L (needs a leakage-safe harness).

---

### A4. Per-class prototype alignment on unlabeled eval trials `[label-free]` **Tier 1**

**The oversight.** §6 documents **PMANet**, reporting **79.32% on 4-class IV-2a
cross-session** — the closest same-dataset, same-protocol number found anywhere in the
survey to this project's 66.9%. Its mechanism: per-**class** prototypes derived from the
source-session classifier, with *unlabeled target-session trials softly aligned* to them.

Phase 6 built per-**subject** prototypes from the target's own *labelled* calibration trials
(49.0%) — a different mechanism. Nothing in the repo aligns unlabeled target trials to class
prototypes. This is a distinct, literature-backed, label-free gap.

**Caveat to state honestly:** the survey could not confirm whether PMANet's training pool is
single-subject or pooled across subjects (paywalled full text). Treat 79.3% as directional
until reproduced; reproducing it *is* the experiment.

**Why it could beat 66.9%.** It uses the eval session's 288 unlabeled trials as evidence,
which the current pipeline only uses for a second-order whitening statistic. Class-level
structure is more information than a single mean covariance.

**First experiment.** CSP+LDA+RA first pass → soft-assign eval trials to classes →
recompute class prototypes (per-class mean covariance / tangent mean) → re-fit or re-align →
iterate 2–3 times. Validate on synthetic data first, where the true class structure is
known, so a null is attributable to the mechanism rather than the implementation. Cost: M–L.

---

### A5. Optimal transport / Wasserstein alignment `[within-subject]` **Tier 1**

**The oversight.** `docs/progress_and_direction.md` records, in its own literature sweep for
Riemannian-framework refinements: *"optimal transport/sliced-Wasserstein distribution
alignment (a real, published **+2.45pp improvement on this exact dataset** — not yet built
here)"*. It was flagged and then dropped in favour of CPD and ComBat, both of which closed
out as nulls. The flagged lead itself was never built. The grep confirms: no Sinkhorn, no
Wasserstein code anywhere.

**Why it could beat 66.9%.** It is the *same family* as the method that works (Riemannian
alignment): a label-free, within-session distribution-matching step. Published on IV-2a. It
is also strictly more general — recentering matches first moments (mean covariance);
OT/Wasserstein can match the full distribution.

**First experiment.** Sliced-Wasserstein / Sinkhorn alignment of the calibration and eval
trial-covariance distributions as a drop-in alternative to `RiemannianAlignment`, same
downstream CSP+LDA, same 9-subject protocol. Cost: M. Also worth including the free variant:
**Wasserstein barycenter of calibration + eval** as a shared reference (see A12).

---

### A6. Per-subject hyperparameters chosen by inner CV `[within-subject]` **Tier 1**

**The oversight.** Every pipeline choice is a single global constant:
`FMIN, FMAX = 8.0, 30.0`; `TMIN, TMAX = 0.5, 2.5`; CSP `n_components=8`;
`reg="ledoit_wolf"`; plain (unshrunk) LDA; `estimator="oas"` for covariances. Nothing was
ever tuned per subject — and the project's own synthetic generator explicitly encodes that
*peak frequency varies by subject* (9–12 Hz) and that some subjects show ERS instead of ERD.

**Why it could beat 66.9%.** This is A1's hypothesis made executable. If subjects 2/5/6 fail
because their sensorimotor rhythm sits outside 8–30 Hz, or their ERD peaks late, or they
need fewer CSP components, a per-subject inner-CV selection recovers them. It is also the
cheapest possible test of "the pipeline's fixed constants are the problem".

**First experiment.** Per subject, inner 5-fold CV *within the calibration session only*
(no eval leakage), selecting over: band (e.g. 6–35 / 8–30 / 8–24 Hz), window
(0.5–2.5 / 0.75–3.5 s), CSP components (4/6/8/12), and covariance estimator
(oas / ledoit-wolf / empirical). Report both the inner-CV choice and the eval-session
accuracy, plus a nested-CV honest estimate.

**Risk to control:** with 9 subjects and a ~50-configuration grid, selection noise is the
main enemy — nested CV or a held-out subject split is mandatory, otherwise this becomes a
false-positive generator (see A27).

---

### A7. The pre-cue baseline as an alignment reference `[label-free]` **Tier 2**

**The oversight.** `src/datasets.py` already provides `load_iv2a_subject_wide_window`
(-1 s to +4 s post-cue), built specifically for the wavelet work, and it retains the pre-cue
baseline that the standard loader discards. Nothing uses the pre-cue segment as the
alignment reference.

**Why it could beat 66.9%.** It is *strictly more zero-shot* than the current method:
Riemannian alignment today consumes the eval session's **task trials** (unlabeled, but
still task data — you must run the task to get them). Recentring on the **pre-cue baseline**
requires no task trials at all — genuinely calibration-free, and it is the "resting-state
fingerprint" idea the survey's §7 gap (2) points at. It is also available *before* the
session starts, which changes what the system can promise.

**First experiment.** Estimate the eval session's whitening matrix from pre-cue segments
only; compare (a) pre-cue-only, (b) task-trials-only (current), (c) both pooled. Cost: M.
**Prediction to state in advance:** pre-cue-only will underperform, because the task
covariance differs from rest; the interesting question is whether *pooling* beats either.

---

### A8. Within-subject shrinkage — the untested sibling of a proven null `[within-subject]` **Tier 2**

**The oversight.** The project's flagship negative result is SPD shrinkage: shrinking a
subject's few-trial covariance toward a **population** prior. It failed at 8 subjects, again
at 20 OpenBMI subjects, and the eigenvoice variant failed worse. The diagnosed reason —
population mean has the wrong shape for heterogeneous subjects — kills the *population*
prior. It says nothing about shrinking toward **the same subject's own calibration-session
covariance**, which is a within-subject prior, requires no other subject, and was never
tested.

This is possibly the cheapest unrun experiment in this document: `shrink_subject_mean`
already takes an arbitrary `prior`, and `fit_population_prior` would simply be replaced by
one built from the target subject's own calibration trials.

**Why it could beat 66.9%.** It directly addresses the low-data regime (short eval-session
recordings, n=8–32 trials) where the current estimate is noisiest, using exactly the kind of
prior the project has already proven it can compute. The failure mode that killed the
population version (wrong shape from other subjects) cannot occur.

**First experiment.** Re-run the `phase2b` sweep with `prior` = the subject's own
calibration-session covariance (and a second variant: prior = mean of that subject's
calibration *and* the eval trials seen so far). Cost: S.

---

### A9. Filter-bank CSP (FBCSP) `[within-subject]` **Tier 2**

**The oversight.** The pipeline uses one fixed band (8–30 Hz) for CSP. FBCSP — multiple
bands, CSP per band, mutual-information feature selection — is the standard strong baseline
for IV-2a and is entirely absent (grep confirms no filter-bank code). The project's own
survey records the MOABB within-session benchmark where single-band CSP+LDA is 82.3% while
tuned deep nets reach 86.2% — i.e. plain CSP is *known* not to be the within-session
ceiling, and no multi-band variant was ever tried.

**Why it could beat 66.9%.** More bands = more chances that *this* subject's mu/beta is
captured well, which is A1/A6's hypothesis with a standard, non-overfit-prone
implementation. It also composes with Riemannian alignment (align, then FBCSP, or FBCSP
features then align).

**First cost of admission:** the 440-feature Phase 4c lesson — added features cost
sample efficiency (0.65 samples/feature was a wash). FBCSP needs aggressive feature
selection (its own MI-based selection step) and shrinkage LDA, and the
samples-per-feature ratio must be reported. Cost: M.

---

### A10. The canonical Riemannian classifier on aligned data `[within-subject]` **Tier 2**

**The oversight.** After Riemannian alignment the pipeline hands off to **CSP+LDA**.
The canonical Riemannian pipeline in the literature is alignment **+ MDM (minimum distance
to mean)** or alignment + **tangent-space features + regularised linear classifier**. The
survey's benchmark table lists MDM at 81.7% within-session (vs CSP+LDA 82.3%) — i.e. a
peer, not a weakling.

Phase 7 did use tangent-LDA, but as its own baseline in the ICP branch, not composed with
Riemannian alignment and not tuned. `pyriemann` is already a dependency.

**First experiment.** `RiemannianAlignment` → tangent-space projection → {LDA, logistic
regression with CV-chosen C, MDM, SVM} vs the current CSP+LDA, all on the identical
9-subject protocol. Cost: S–M. This is the cheapest way to check whether the classifier
head, not the alignment, is the limiter.

---

### A11. Self-training with eval pseudo-labels `[label-free]` **Tier 2**

**The oversight.** Nothing in the repo uses the classifier's own confident predictions on
the eval session to improve the eval-session model. `active_calibration.py` does
uncertainty sampling for *calibration trial ordering* (§10.4, tested, narrow), which is the
inverse problem.

**Why it could beat 66.9%.** It is the missing half of A2's rotation (which needs pseudo-
labels) and the natural partner of A4. It is label-free at deployment and within-subject.

**First experiment.** Fit on calibration → predict eval with confidence → keep the top-q%
most confident eval trials (q swept 25/50/75) → re-fit/re-align → re-predict. Report
accuracy on the *non-selected* eval trials separately, otherwise the result is inflated by
construction. Cost: M.

---

### A12. Alignment-estimator variants `[within-subject]` **Tier 2**

Cheap ablations of the one method that works; all unrun, all `S`:

- **Euclidean Alignment** vs Riemannian recentering (arithmetic mean covariance instead of
  the Riemannian mean). He & Wu 2020, cited in §2, trivially switchable via `pyriemann`.
- **Per-class recentering** instead of per-session: whiten each class's trials by that
  class's own mean — removes class-mean shift while preserving class structure.
- **Pooled calibration+eval reference**: estimate one whitening matrix from **all 576**
  available unlabeled trials (both sessions) instead of 288, then… which session do you map
  to? Worth testing explicitly; the naive answer (map both to the pooled identity) is not
  obviously right.
- **Geodesic-subspace alignment**: estimate the whitening from only the top-k principal
  geodesics of the trial covariances, discarding the rest — a regularised alignment
  (the CPD-PCA lesson, applied to the one method that works).
- **Robust mean** (geodesic median instead of mean) to resist outlier trials.

**Cost:** S each. Any single one is unlikely to be decisive; as a group they may find 1–2pp
and they are nearly free.

---

### A13. Spatial preprocessing was never varied `[within-subject]` **Tier 2**

**The oversight.** The pipeline consumes MOABB's default-referenced signal, bandpasses, and
(depending on the branch) either uses raw data or the ICA-cleaned cache. Untested:

- **Current Source Density / surface Laplacian re-referencing** (per session, label-free).
  Standard in MI decoding for suppressing volume conduction; a pure spatial high-pass that
  is *identical in form* across sessions and therefore cannot itself add cross-session
  variability the way independently-fitted ICA did.
- **EOG regression** instead of ICA component removal. The cleaned-pipeline result (61.3%
  vs 66.9%; worse and noisier) was attributed to ICA removing genuine discriminative
  signal. EOG regression is a strictly gentler, deterministic alternative — and it is
  *session-independent in form*, which directly addresses the diagnosed cause (each
  session's ICA being fit independently applying a session-specific spatial transform).
- **A shared cleaning transform** (fit ICA/artifact model once on calibration, apply the
  same weights to eval) rather than two independent fits.
- **Explicit re-referencing choices**: common average, CAR, REST, or a fixed channel.

**Why it could beat 66.9%.** The cleaned-data experiment is the one place this project
*found* a concrete, diagnosed cross-session hazard of its own making (session-specific
spatial transforms). Fixing that hazard with a deterministic, session-invariant transform
is a targeted, testable repair — and the cleaned pipeline's subject 9 reached 80.9%,
showing the ceiling is there for some subjects. Cost: S–M.

---

### A14. Calibration-trial quality weighting and augmentation `[within-subject]` **Tier 2**

**The oversight.** All 288 calibration trials are weighted equally when fitting CSP and
LDA, and the pipeline has no augmentation. Two cheap, orthogonal levers, both untried:

- **Quality weighting**: down-weight trials with high EOG/muscle contamination or low
  class-separability under a preliminary model (a soft, label-aware analogue of artifact
  rejection).
- **Augmentation**: sliding sub-windows (increases effective n for CSP covariance
  estimation), Gaussian noise injection at matched SNR, channel dropout, and **mixup**
  within a class. Cross-disciplinary: mixup and noise injection come from vision; the
  idea of trading label precision for sample count is standard in audio.

**Why it could beat 66.9%.** CSP quality is limited by covariance estimation error from 288
trials — the single most-cited practical limitation of CSP. Augmentation attacks exactly
that. Cost: M.

---

### A15. Ensembling and test-time augmentation `[within-subject]` **Tier 2**

**The oversight.** Every result in the repo is a single pipeline. Untried: averaging
predictions across RA + tangent-LDA + CSP variants + the prototypical encoder; and
**test-time augmentation** (average predictions over time-shifted / band-shifted copies of
each eval trial — a standard, near-free trick from vision).

**Why it could beat 66.9%.** Ensembling decorrelated methods routinely buys 1–3pp, and this
project already has several methods with *different error patterns* (Riemannian 66.9%,
prototypical 49.0% but 59–67% on four subjects, flow-field weak but independent). Cost: M.
Note this is the honest, non-naive successor to the Phase-5 concatenation failure, which
was fit once on full calibration rather than combined out-of-fold (see A16).

---

### A16. Out-of-fold stacking `[within-subject]` **Tier 2**

**The oversight.** Flagged explicitly and twice in `docs/progress_and_direction.md` as the
correct fix for Phase 5's failure, and never built: *"A meta-classifier trained on honest,
non-leaked out-of-fold predictions might learn to trust CSP heavily and only lean on
wavelet-calibration where it adds real value, rather than diluting uniformly."*

**Why it could beat 66.9%.** Phase 5's failure (62.2% → 54.8%) is a *combiner* failure, not
necessarily an information failure. Cost: M (each base learner refit per fold).

---

### A17. Prototypical embeddings + CSP `[pop-data, safe]` **Tier 2**

**The oversight.** Explicitly gated in the docs: *"Do not revisit naive combination until
the standalone wavelet-calibration mechanism is substantially stronger"* and, for
prototypical networks, *"now that the standalone mechanism is much stronger… worth
revisiting"*. Phase 6 reached 49.0% with four subjects at 59–67%; the gate it was waiting
for is arguably met. Untested either as concatenation **with** out-of-fold stacking or as an
ensemble.

---

### A18. Similarity-selected recentering target `[cross-subject ⚠]` **Tier 3**

**The oversight.** `docs/state_of_the_art.md` §"Where this leaves the fingerprint idea"
names this as the **sharpest and cheapest** of the two genuinely open gaps: *"Nobody selects
the recentering/transfer target based on subject similarity"* — Kim et al. (2023) use one
arbitrarily-picked expert for everyone on IV-2a; Kumar et al. (2024) use one fixed expert
for all 18 subjects. Neither asks which reference is closest to *this* subject. The proposed
mechanism — compute each candidate expert's mean covariance once, take the Riemannian
distance from the new subject's short recording, recenter toward the nearest instead of a
fixed reference — is a small extension of `RiemannianAlignment` and has two named baselines
to beat.

**Honest headwind:** this *is* a cross-subject comparison, so the project's central finding
predicts trouble. Two reasons to run it anyway: (a) it is a *selection* problem, not a
correction injected into one computation — the decision variable is a scalar distance
between two subjects, and the outcome is "use subject X's session statistics as a target",
which may be benign in the way that wavelet calibration is benign; (b) it is the only
documented, sourced, still-open gap in the literature review, so a well-powered null is
itself a publishable answer.

**First experiment.** Leave-one-subject-out over the 9 IV-2a subjects: for each target,
rank the other 8 by Riemannian distance from the target's *short* (n=16/32) recording;
recenter toward nearest / farthest / random / population mean of the k nearest. Controls
included by construction. Cost: M.

---

### A19. The illiteracy predictor as a conditioning input `[pop-data, safe]` **Tier 3**

**The oversight.** Documented open gap #2: *"Every resting-state predictor stops at a
binary screening label. Every personalization method starts from calibration-session data,
not a pre-session predictor. Wiring these together… still hasn't turned up."*

**Why it matters even with no accuracy gain.** Given A1's likely outcome (some subjects are
signal-limited), a *predictor* that says "this subject will not reach usable accuracy, use a
different paradigm" is a clinically meaningful deliverable and does not require beating
66.9% at all. That reframing — from "raise the mean" to "predict and route" — is the
project's own thesis ("it's the system, not the user") turned into a product.

**First experiment.** From the pre-cue baseline only (A7): predict subject-level Phase-2
accuracy from band power, alpha peak frequency, and covariance structure; report
leave-one-subject-out correlation. Cost: L.

---

### A20. A dataset with more than two sessions `[within-subject]` **Tier 3**

**The oversight.** The survey's §1 leads with **Won et al. 2022** — 25 subjects, **5
sessions**, 2-class, within-session 68.8% / cross-session 53.7% / with adaptation 78.9% —
and it is never downloaded or used. Neither is Cho2017 (52 subjects) nor BEETL. The entire
project rests on IV-2a's *two* sessions per subject.

**Why it could beat 66.9% — structurally.** With ≥3 sessions per subject you can learn a
*drift trajectory* rather than one offline correction: predict session k's whitening from
sessions 1..k-1. The current pipeline cannot even express that question. This also unlocks
A-control (iterative learning control, B11) and the classic co-adaptive literature (§9)
that §9 says was never implemented.

**First experiment.** Download Won et al., reproduce its published within/cross-session
numbers as a sanity check, then test "predict next session's alignment from the previous
session's alignment" against plain recentering. Cost: L (download + new loader).

---

### A21. Cross-paradigm fingerprint `[label-free]` **Tier 3**

**The oversight.** OpenBMI (Lee2019) contains **MI, ERP and SSVEP recordings for the same 54
subjects**, and this project already loads OpenBMI subjects for the population pool — but
only the MI part. The cross-paradigm structure is untouched, and the survey's §8 notes that
ERP/SSVEP illiteracy rates are 10–11% vs MI's 53.7%.

**Why it could beat 66.9%.** A subject's *paradigm-independent* fingerprint (resting-state
or SSVEP-evoked) is far better identified than their MI fingerprint, and it is available
before any MI calibration. This is a genuinely novel, label-free route to A18/A19 that uses
a dataset already partially on disk. Cost: L.

---

### A22. PhysioNet's excluded "rest" trials as interleaved drift probes `[label-free]` **Tier 3**

**The oversight — and a correction to the project's own conclusion.** `docs/state_of_the_art.md`
§10.3 concludes the adaptive-optics "guide star" idea is *"genuinely untestable with IV-2a/
OpenBMI as they stand (no dataset has interleaved task-irrelevant probes)"*.

That conclusion is scoped to *"IV-2a/OpenBMI"* — but PhysioNet is a third dataset, already
fully downloaded here (108 subjects), and it appears to have exactly the missing property.
`src/datasets.py` deliberately *excludes* PhysioNet's `rest` event type — *"~48% of returned
trials"* — precisely because a covariance prior should not be contaminated with non-task
baseline trials. Those rest trials are an **interleaved, task-irrelevant probe**: recorded
alongside the task, carrying no class information, and therefore usable for tracking drift
without touching labels. (Verify first that they are distributed across the session rather
than confined to block boundaries; `src/datasets.py` already uses the explicit `events=`
route that would be needed to load them.)

**Why it could beat 66.9%.** A probe interleaved with the task measures *drift* rather than
*class structure*, so it can be used to track slow changes within a session and correct for
them without ever touching task labels — the adaptive-optics mechanism, made testable. Cost:
M. This is the highest-novelty item in Part A relative to the project's own stated
conclusions, and the cheapest to falsify.

---

### A23. Small conv nets with cross-session fine-tuning `[pop-data, safe]` **Tier 3**

**The oversight.** The deep-learning conclusion in this project is **literature-only**: §11
rules out JEPA/foundation models for MI (S-JEPA 65% vs Riemannian 84.7%; LaBraM/LUNA/CBraMod/
Laya at 0.47–0.51 balanced accuracy under frozen linear probing), and §4 notes EEGNet
zero-shot at 43%±7%. But the MOABB benchmark in the same document shows **ShallowConvNet
86.2% vs CSP+LDA 82.3%** *within-session* — so a small conv net is not inherently weak; the
losses come from cross-subject zero-shot transfer.

**Untested here:** a small conv net (EEGNet / ShallowConvNet / EEG-TCNet) trained
*per-subject* on calibration and fine-tuned on the eval session's **unlabeled** trials
(unsupervised) — which is A3's mechanism with a stronger backbone. Cost: L. **Practical
blocker:** `braindecode` does not import in this environment (torchaudio/torch version
mismatch, see `requirements.txt`), so this needs either a torchaudio fix or a ~200-line
EEGNet implementation.

---

### A24–A26. Already-logged small items `[within-subject] / [pop-data, safe]` **Tier 3**

Recorded in `docs/progress_and_direction.md` and not repeated at length here:

- **A24** Per-subject and per-class jitter tolerance instead of a fixed 0.15 s Gaussian
  smoothing width (`S`, +0–1pp).
- **A25** RIDE's multi-component decomposition — the literature *specifically* claims it
  handles low-SNR cases better than plain Woody, and only Woody was tested (`M`, +0–1pp).
  Note: run it *combined with* Riemannian alignment, not as a competitor — the project's
  finding was "real but modest", not "wrong".
- **A26** Prototypical-network open items: understand the subject-5 chance-level outlier,
  tune `episodes_per_step`, and compute a proper paired significance test against Phase 2
  (49.0% vs 66.9%, std 16.1% — a real gap, but the second-best result in the project and
  the only non-Riemannian mechanism with four subjects in the 59–67% band) (`M`).

---

### A27. Measurement hygiene `[—]` **Tier 1 (gates everything)**

**The oversight.** Roughly 40+ conditions have been evaluated on 9 subjects, most reported
with per-condition p-values and no multiple-comparison control. Several "significant"
results (phase2b's single nominally-significant n opposite in sign to another n; idea10_4's
p<0.05 in one trial-count window) are already documented as *most likely* multiple-
comparisons noise. That instinct is right — but it needs to become a rule, not a footnote.

**Actions (all cheap, all protective):**

- Adopt an explicit decision rule: Holm–Bonferroni within a family of comparisons, or
  Benjamini–Hochberg for exploratory screens.
- State power up front: with n=9, Wilcoxon cannot go below p=0.0039; anything smaller than
  ~+2pp average is undetectable. Do not claim otherwise.
- Pre-register the primary endpoint for each new experiment (eval-session accuracy on all 9
  subjects, paired Wilcoxon + per-subject McNemar) *before* running it.
- Never report a mean without the per-subject row (A1).
- Fix the leakage-adjacent traps the project already found once each: in-sample sanity
  checks before blaming generalization; selected-subset reporting (A11); selection noise in
  per-subject tuning (A6).

---

### A28. Align the fitted dynamics, not the statistics `[within-subject]` `[label-free]` **Tier 1** *(new — see §3)*

**The oversight.** Every alignment method in this repo matches *second-order statistics*:
`RiemannianAlignment` whitens the session's mean covariance, shrinkage interpolates
covariances, tangent-space methods rotate point clouds of covariance vectors. Systems
neuroscience tackled the same multi-day instability problem for intracortical BCIs and
arrived somewhere else: the raw activity drifts across days, but the **latent dynamical
system** generating it is stable (Gallego et al., *Nat Neurosci* 2020), so decoders can be
stabilised by aligning *latent dynamics* rather than distributions (Karpowicz et al.,
*Nat Commun* 2025). Neuralink's pretraining leans on the same premise. Neither reference
appears anywhere in `docs/state_of_the_art.md`.

**What already exists here, unused for this purpose.** `src/adaptation/state_space_flow.py`
fits a skew-symmetric linear dynamical system to trial trajectories
(`fit_skew_symmetric_flow_debiased`, `fit_pooled_flow_debiased`) and extracts rotation
frequency and plane. It was tested **only as a feature extractor** — `phase10` reached 25.8%,
a clean null — and never as an *alignment target*. `riemannian_icp.procrustes_rotation` is a
validated, closed-form similarity-transform solver. The two have never been pointed at each
other.

**Why it could beat 66.9%.** It is within-subject, label-free, and it corrects something
whitening provably cannot. Whitening fixes the *marginal* covariance of the trial
distribution; if the drift is a rotation or expansion of the *state space* — which is
exactly the structure the project's synthetic drift experiments inject — then matching the
marginal leaves the dynamics mismatched. This composes with Riemannian alignment rather than
competing with it.

**First experiment.** Fit a linear dynamical system per session in a shared low-dimensional
subspace (PCA on pooled calibration trials, fixed across sessions to make `A` comparable);
estimate the similarity transform `T` mapping session B's `A` onto session A's; apply `T` to
the eval trials; run the standard CSP+LDA. **Gate it on synthetic data where the planted
drift is known by construction** — the project already has the generators
(`stage0_state_space_flow_eeg_validation.py` plants 8 known channel-mixing drifts).

**Cost:** M. **Kill criterion:** if, on synthetic data with a *known* planted drift, the
per-session `A` estimates are not measurably closer after alignment, the mechanism is wrong
for this signal and should be closed out like CPD — do not carry it to real data.

**Related caveat to respect:** `phase10`'s null showed real IV-2a trials do not exhibit the
rotational structure the synthetic generator guarantees. So the synthetic gate passing is
*necessary but not sufficient*; treat this exactly like Module A (passed its gate, failed on
real data) and budget for that possibility.

---

### A29. Make calibration burden the headline metric `[within-subject]` **Tier 1** *(new — see §3)*

**The oversight.** Every result in this repo is reported at one fixed calibration budget:
all 288 labelled calibration trials. The question *"how few labelled trials are needed to
reach accuracy X?"* is never asked. `phase2b_spd_shrinkage.py` sweeps sample size — but only
of the **eval-side unlabeled reference**, never of the **training/labelled** data.

**Why this may matter more than any accuracy gain.** It is the metric the field and industry
actually optimise (Neuralink's headline result is a *calibration-burden* reduction — 10
min/day → 10 min/week — not a peak-accuracy number). And this project may be sitting on an
unusually strong version of it: Riemannian alignment uses **zero labels** from the eval
session, so its advantage over non-adapted CSP should *grow* as the calibration budget
shrinks. If that holds, the project's real claim is not "66.9% vs 62.2%" but something like
*"reaches 62% with half the calibration"* — a more useful and more defensible result, and
one that reframes the whole negative-results narrative as a story about label efficiency.

**First experiment.** Sweep labelled calibration trials n ∈ {32, 64, 96, 144, 192, 288} ×
{no alignment, RA, RA+RPA (A2), tangent-LDA}, and report two things: accuracy vs n, and
**trials-to-threshold** (n needed to reach 60% / 62% / 65%). All data is cached locally, so
this is refits only.

**Cost:** S–M. **Deliverable:** one figure and one table — the most presentable result this
project could produce, and it requires no new mechanism to succeed. Do this early even if
every method-level item in this file stalls.

---

### A30. Per-subject self-supervised pretraining on the target's own unlabeled data `[label-free]` **Tier 2** *(new — see §3)*

**Where it comes from.** Neuralink's live results use *participant-specific* encoders
pretrained on that participant's own recordings, with masked channel prediction as the
objective — self-supervision used **within** a subject, not to transfer **across** subjects.
The survey's §11 dismissal of self-supervised pretraining for MI is about the cross-subject,
public-corpus version. The within-subject version has never been tried here.

**The transplant.** Pretrain a small encoder on the target's own data — 288 calibration + 288
eval trials, optionally plus the continuous recordings already cached under `data/cleaned/` —
with a masked-channel / masked-time reconstruction objective, then use its embeddings
downstream (classifier head, or prototypes as in Phase 6).

**Honest headwind — state it before running.** IV-2a gives a subject ~3–4 orders of magnitude
less data than the Neuralink report, and the project's own evidence says high-capacity
learned corrections fail when reference data is small (Methodology lesson 3). Expect a null.
The value is that it makes the question *falsifiable*: **is the foundation-model idea wrong
for MI, or is the public data wrong for foundation models?**

**Route through the gates:** synthetic generator (arbitrarily many trials for a *known*
subject) → one PhysioNet subject's long recording → IV-2a. **Cost:** L.

---

### A31. Longevity: accuracy decay over sessions `[within-subject]` **Tier 3** *(new — see §3)*

**The oversight.** The project has no longitudinal metric at all, because IV-2a has two
sessions. Neuralink's most striking results are longevity results: some decoders stable for
>3 weeks, one still usable **20 months** after calibration, recalibration dropping from
10 min/day to 10 min/week.

**Why it matters.** "Sessions until accuracy drops below X" is the real-world question, and
it is orthogonal to mean accuracy: a method that is 2pp worse on average but decays half as
fast is the better product. Nothing in the current evaluation can express that.

**Blocked on A20** (Won et al. 5-session dataset — cited in the survey's §1, never
downloaded). **Cost:** L, after A20.

---

### A32. State-space / Mamba-style sequence encoder `[pop-data, safe]` **Tier 3** *(new — see §3)*

Neuralink's first architectural bet is that neural activity is best modelled as a dynamical
system, motivating a Mamba/SSM backbone with constant-latency inference. The analogous EEG
choice has never been tested here — the project's only learned encoder is a small MLP over
tangent vectors (`prototypical_network.py`).

**Honest assessment:** low priority. Architecture changes are expensive to evaluate honestly
on 9 subjects, and everything this project has learned says the *alignment* step, not encoder
capacity, is where the leverage sits. Listed for completeness, and because A28 is the
cheap part of the same dynamical-systems insight. **Cost:** L.

---

## Part B — Cross-disciplinary transplants (bonus)

Each card: **field → borrowed mechanism → transplant → fit against the central finding →
cheapest first test → cost**. Ordering within tiers is by expected value.

`NEW` = not in `docs/state_of_the_art.md` §10. `EXTENDS §10.x` = a variant of something
already surveyed there.

### Tier B1 — Strong fits: cheap, label-free or within-subject, unclaimed

**B27. Latent-dynamics alignment — systems neuroscience / latent variable models `[within-subject]` `[label-free]` `NEW`**
*(Added after the Oct 2026 Neuralink update; numbered out of sequence to avoid renumbering
the cards below. Same experiment as A28 — listed here for its cross-disciplinary provenance.)*
Systems neuroscience faced this project's exact problem in a harder setting — intracortical
recordings that drift across days — and solved it by showing that the **latent dynamics** are
stable across days even when spike statistics are not (Gallego et al., *Nat Neurosci* 2020),
then aligning those dynamics instead of the data (Karpowicz et al., *Nat Commun* 2025; the
premise Neuralink's pretraining rests on). This project matches *marginals* (covariances);
this matches the *transition operator* — an orthogonal correction that composes with
Riemannian alignment instead of competing with it. **Fit:** within-subject, label-free, and
all the primitives already exist (`state_space_flow.fit_skew_symmetric_flow_debiased` fits
the system; `riemannian_icp.procrustes_rotation` solves for the transform between two of
them). **Test:** see A28. **Cost:** M. **Why it is the strongest transplant in this list:** the
home domain's drift structure (slow, structured, within-subject) matches this project's drift
structure far more closely than any of the other fields surveyed.

**B1. Adaptive noise cancellation — radar / sonar / audio (LMS, RLS) `[label-free]` `NEW`**
The classic problem in radar and audio: a target signal is corrupted by an additive
interference that can be *measured separately* via a reference channel; an adaptive filter
(LMS/RLS) estimates the interference and subtracts it. Transplant: treat **session drift** as
the interference, and a task-irrelevant reference — the pre-cue baseline (A7), PhysioNet
rest trials (A22), or an EOG/occipital channel — as the reference input. Estimate the
per-channel transfer function from reference to task signal in the *calibration* session,
then apply the inverse to the eval session. **Fit:** fully label-free, within-subject, no
cross-subject comparison, and it targets gain/drift directly rather than the covariance
envelope. **Test:** least-squares/LMS estimate of drift from pre-cue to task in calibration;
apply to eval; compare against RA. **Cost:** M.

**B2. RUV — remove unwanted variation — genomics / transcriptomics `[label-free]` `NEW`**
RUV (Risso et al.) removes batch effects using **negative-control genes** known to be
unaffected by the biological signal of interest, estimating the unwanted factors from them
alone. This is precisely ComBat's problem statement with a structurally better idea.
Transplant: designate **task-irrelevant channels/bands** (occipital alpha, frontal channels
for MI, or the pre-cue baseline) as negative controls; estimate the session drift factor
from them; regress it out of the task channels. **Fit:** within-subject, label-free, and it
does not force all 253 tangent dimensions to be treated as exchangeable features — which is
exactly why the project's own post-mortem judged ComBat's assumption violated. **Test:**
same protocol as `phase7c_combat.py` for a direct comparison, with the negative-control set
chosen anatomically in advance. **Cost:** M. **This is my single highest-rated transplant**
— it explains the ComBat null mechanically and offers a repaired version.

**B3. Quantile mapping / bias correction — climate science `[label-free]` `NEW`**
Climate model output is corrected to match observed distributions by mapping quantiles of
the modelled series onto quantiles of the observed series, per variable and per location.
Transplant: per channel (and per band), map the eval session's amplitude distribution onto
the calibration session's, using *all unlabeled eval trials* — a monotone, label-free,
one-dimensional warp. It is the marginal-distribution analogue of Riemannian alignment,
which only matches second-order structure. **Fit:** within-subject, label-free, and O(n)
per channel to compute. **Test:** apply after (or instead of) RA; also worth trying on
log-band-power features rather than raw signal. **Cost:** S–M.

**B4. CMVN + RASTA — speech recognition `[label-free]` `NEW`**
Cepstral mean and variance normalisation is the standard fix for exactly this problem in
speech: the *same* utterance recorded on a different microphone/channel has a different
feature mean, and CMVN removes it per utterance, without labels. RASTA filtering additionally
suppresses slow channel-induced spectral modulation while preserving faster speech dynamics.
Transplant: per-session, per-channel mean/variance normalisation of band-power features
(and/or a RASTA-style bandpass along the frequency axis to suppress slow spectral tilt).
**Fit:** trivially label-free and per-session; the direct analogue of Riemannian alignment
in the *feature* domain instead of the covariance domain. **Test:** cheap ablation on the
existing feature pipelines; report whether it is subsumed by RA. **Cost:** S.

**B5. VTLN — vocal tract length normalisation — speech `[`within-subject`] `NEW`**
Speaker differences in speech are largely a *frequency-axis warp* (vocal tract length);
VTLN estimates one warp parameter per speaker by maximising likelihood, then resamples the
frequency axis. Transplant: the project's own synthetic generator encodes that ERD/ERS **peak
frequency varies by subject (9–12 Hz)**, and the pipeline's fixed band ignores this. Estimate
one frequency-warp parameter per session (or per subject, from calibration) that aligns the
mu/beta peak to a canonical position, then resample. **Fit:** within-subject, one parameter,
physiologically motivated, and it makes the fixed 8–30 Hz band defensible by construction.
**Cost:** S–M. **High value per line of code.**

**B6. MLLR / MAP speaker adaptation — speech `[within-subject]` `NEW`**
Before deep learning, ASR adapted a speaker-independent model to a new speaker from a few
seconds of speech using **maximum-likelihood linear regression** on the model's means, or
Bayesian MAP adaptation — closed-form, few parameters, designed for exactly the "a little
data from one new user" regime. Transplant: closed-form MAP/MLLR adaptation of the LDA (or
logistic-regression) class means and shared covariance from the target's own calibration
trials, instead of refitting from scratch. **Fit:** within-subject; also the *correct*
Bayesian answer to A8's within-subject shrinkage. **Cost:** S–M.

**B7. Hierarchical Bayesian partial pooling — biostatistics / epidemiology `[pop-data, safe]` `NEW`**
The project diagnosed its own flagship null precisely: *"a mean-shrinkage model is the wrong
shape for genuinely heterogeneous subjects, not a data-scarcity problem."* That is the exact
problem hierarchical models were invented for. Instead of one global shrinkage weight toward
one population mean, model each subject's covariance as drawn from a population
distribution with **subject-level random effects**, and let partial pooling set the amount
per subject — including ~zero for genuine outliers. **Fit:** note this *is* population data
used to correct an individual, so honesty requires flagging it — but the comparison being
made is between the *population distribution* and the subject, not between subject A and
subject B directly. It deserves a test on that distinction alone, since it is the natural
repair of the single most-investigated null in the project. **Test:** synthetic gate first,
with the generator's known heterogeneous subjects (the project already has
`sample_subject_params`), then the OpenBMI-20 pool where the −3.5pp/+4.2pp per-subject
cancellation was found. **Cost:** M–L.

**B8. CORAL / subspace alignment / geodesic flow kernel — computer-vision domain adaptation `[label-free]` `NEW`**
A family of classic unsupervised DA methods built specifically for "same task, shifted
distribution, no target labels": CORAL aligns second-order statistics of source and target
features; Subspace Alignment rotates the source PCA subspace onto the target's; GFK
interpolates infinitely many subspaces along the Grassmann geodesic between them. All are
closed-form and label-free. **Fit:** Riemannian alignment already *is* the covariance-space
member of this family; these are the **feature-space** members, which have never been tried
here, and GFK's geodesic interpolation is close in spirit to the project's tangent-space
machinery. **Test:** CORAL on CSP features as the minimal version — a ~20-line experiment —
then GFK if CORAL shows anything. **Cost:** S (CORAL) / M (GFK).

**B9. Optimal transport / Wasserstein barycentres `[within-subject]` `EXTENDS (§"refinements" sweep)`**
Already the subject of A5; listed here because the *cross-disciplinary* content is real:
Sinkhorn divergence and Wasserstein barycentres come from optimal transport (Monge,
Kantorovich; now standard in ML). Two transplants rather than one: (a) align eval to calib
via sliced-Wasserstein; (b) use the **Wasserstein barycenter** of the two sessions'
covariance distributions as a shared reference — a middle ground that neither the
calibration-only nor eval-only estimate provides. **Cost:** M.

**B10. Kalman filtering / state-space tracking — control theory `[label-free]` `NEW`**
Treat session drift as a slowly-varying latent state observed noisily through each trial's
covariance. A Kalman filter gives the *recursively optimal* online estimate of that state —
which is exactly what A3's "running statistic update" does ad hoc, and it comes with a
principled gain (how much to trust the new trial vs the running estimate) that adapts to
noise automatically. **Fit:** label-free, online, within-subject; the project already owns
a state-space module (`state_space_flow.py`) for the linear-dynamics machinery, though that
one was used for rotational structure rather than drift tracking. **Test:** Kalman-filtered
running covariance estimate vs plain exponential moving average, in the sequential harness
A3 requires. **Cost:** M.

**B11. Iterative learning control — control theory `[within-subject]` `NEW`**
ILC improves a repeated task by using the *previous repetition's error* to pre-compensate
the next one — designed for exactly the case of "same system, repeated trials, slowly
varying disturbance". Transplant: across sessions of the same subject, use session k's
measured drift to pre-compensate session k+1. **Blocker:** needs ≥3 sessions (A20, Won et
al.). **Cost:** L (after A20).

**B12. Adaptive beamforming (MVDR / LCMV) — radar & sonar array processing `[within-subject]` `NEW`**
Array processing solved "extract a weak source with a known spatial signature while nulling
interference" decades before CSP: MVDR (Capon) beamforming uses the *data* covariance to
place nulls adaptively, LCMV adds explicit linear constraints. CSP is a related but cruder
instrument (generalised eigen-decomposition without explicit interference nulling).
Transplant: per class, an MVDR/LCMV spatial filter built from the calibration covariance,
with the *other classes* treated as interference to be nulled. **Fit:** within-subject;
directly targets the fact that 4-class MI means the three non-target classes are competing
sources — a framing the current one-vs-rest CSP handles only implicitly. **Test:** replace
CSP with an LCMV filter bank, same downstream LDA. **Cost:** M. **Under-appreciated idea:**
this is the strongest structural alternative to CSP in the whole list.

**B13. Generalised Procrustes analysis & shape statistics — morphometrics `[within-subject]` `EXTENDS §10.3-adjacent`**
RPA (A2) is a two-set Procrustes. GPA aligns *many* configurations simultaneously to a
consensus by iteratively estimating the mean shape and each configuration's transform.
Transplant: align all sessions and/or all classes jointly to a consensus reference rather
than pairwise eval→calib — which is exactly the multi-session setting of A20, and the
principled way to choose a reference when you have more than two sessions. **Cost:** M.

**B14. Cryo-EM / crystallography ab initio orientation alignment — structural biology `[within-subject]` `NEW`**
Cryo-EM determines particle orientations **without labels** by common-lines and
moment-based methods: the second-order statistics of projections constrain the relative
rotation, so orientation can be recovered from the data's own geometry. Transplant: this is
the missing *unsupervised* rotation for A2 — recovering the eval→calib rotation from
covariance eigenstructure (matched principal axes, with sign/permutation ambiguity resolved
by a consistency criterion) instead of from pseudo-labels. **Fit:** within-subject,
label-free. Phase 7's ICP failed as the correspondence-discovery method; eigenstructure
matching is a different and more robust route to the same rotation. **Cost:** M.

**B15. Differential photometry — astronomy `[label-free]` `NEW`**
Variable stars are measured by comparing the target against **reference stars** in the same
frame, which cancels atmospheric and instrumental drift common to all of them. Transplant:
pick channels (or bands) with no expected MI signal as *reference* channels, in *both*
sessions; estimate the per-session gain/offset from them; normalise the task channels by it.
**Fit:** label-free, within-subject, and it is the simplest possible version of B2/RUV —
worth doing as B2's baseline. **Cost:** S. (This is the same family as §10.3's guide star,
but with the reference *already present in the data* rather than requiring new probes —
which is why it is testable and §10.3's original framing was not.)

**B16. Domain randomisation — robotics sim-to-real `[pop-data, safe]` `NEW`**
Policies trained across a *distribution* of randomised simulated dynamics transfer to the
real system far better than policies trained on one. Transplant: the project's synthetic
generator already randomises per-subject ERD/ERS parameters (`sample_subject_params`); extend
it to randomise **session drift** (channel gains, mixing matrices, noise spectra, peak-frequency
shifts) and train the prototypical encoder (or Module A) across that distribution so it is
robust to *any* plausible drift rather than the specific drift in the training pool. **Fit:**
`[pop-data, safe]` — population data teaches robustness, no cross-subject comparison inside
a loss. It is also the natural successor to Module A's failure (train for invariance, not
for prediction of a specific population shape). **Cost:** M–L.

### Tier B2 — Plausible, more speculative, or with a known headwind

**B17. Collaborative filtering / matrix factorisation — recommender systems `[cross-subject ⚠]` `NEW`**
Users and items share latent factors; a new user is placed by the few ratings they give.
Transplant: subjects × (session/condition) matrix of decoding accuracies, factorised to place
a new subject in a latent space and select which prior sessions/references to transfer from —
a learned version of A18's distance-based selection. **Headwind:** explicitly cross-subject.
**Cost:** M. Worth it mainly because it turns A18 from a heuristic into a model with a
principled evaluation.

**B18. EEG microstates & functional-connectivity fingerprinting — neuroscience `[label-free]` `EXTENDS §7/§8`**
The connectomics literature identifies individuals from short resting-state recordings with
high reliability ("brain fingerprints"). Transplant: build the subject fingerprint from
resting-state/pre-cue connectivity (A7) rather than task covariance, and use it for A18/A19.
The survey already flags that this predictor never becomes the conditioning input (gap 2);
this is the concrete mechanism. **Cost:** M–L.

**B19. Test-time entropy minimisation & BN adaptation — vision TTA `[label-free]` `EXTENDS §3`**
TENT and friends adapt a trained model at test time by minimising prediction entropy on the
test batch, updating only normalisation statistics. This is the mechanism behind §3's
published IV-2a 76.2→79.7%; it belongs in A3. Listed separately to note the *cross-disciplinary*
provenance and the transferable insight: **adapt only the normalisation statistics, keep the
decision boundary frozen** — a directly implementable rule for a CSP+LDA pipeline where the
"normalisation" is the whitening step. **Cost:** M.

**B20. Propensity-score matching — epidemiology `[within-subject]` `NEW`**
Causal inference matches treated and control units on covariates to make groups comparable,
isolating the effect of interest. Transplant: the cross-session problem is a covariate-shift
problem — match eval trials to calibration trials on *nuisance* covariates (alpha power,
amplitude, drift proxies) so the classifier is evaluated and trained on comparable
subpopulations, rather than on the whole shifted distributions. **Cost:** M. Unusual, and
plausibly useful mainly as a diagnostic for *which* trials the classifier fails on.

**B21. PARAFAC / multivariate curve resolution — chemometrics `[pop-data, safe]` `NEW`**
Multi-way decomposition separates overlapping contributions in tensor data (subjects ×
sessions × channels) into chemically meaningful factors without labels. Transplant: decompose
the subject × session × covariance tensor into a "subject" factor, a "session/drift" factor
and noise; reconstruct with the drift factor removed. **Fit:** the decomposition is
unsupervised and the *subject* factor is retained, so the classifier never has to compare
subjects explicitly. **Headwind:** interpretation and identifiability in 253-dim covariances
is non-trivial. **Cost:** L.

**B22. Cointegration / vector error correction — econometrics `[within-subject]` `NEW`**
Two non-stationary series can be *cointegrated*: a stable long-run relationship exists even
though each drifts. The error-correction term measures deviation from that equilibrium and
is used to forecast. Transplant: model calibration and eval covariance trajectories as
cointegrated series; use the deviation-from-equilibrium as the drift signal to correct.
**Cost:** L. Elegant, but the project's state-space-flow null suggests the drift may not have
the smooth trackable structure this assumes (a caveat the project already articulated).

**B23. Dynamic time warping / elastic alignment — speech & bioinformatics `[within-subject]` `NEW`**
DTW aligns two sequences that differ by a non-linear time warp, without needing equal
lengths or a shared clock — standard in speech recognition and gene-sequence alignment.
Transplant: the project found a *real, load-bearing* trial-to-trial latency jitter (0.1 s
jitter dropped synthetic accuracy 75.0→67.5%) and handled it with a fixed-width Gaussian
blur. DTW aligns each eval trial's temporal profile elastically to the calibration template
instead — the non-parametric version of A24/The latency-alignment work. **Caveat:** DTW is
the multiple-comparisons trap the project already fell into once ("search over candidate
lags" made things worse via peak-picking bias); a constrained DTW (Sakoe-Chiba band, fixed
warp budget) is the disciplined version. **Cost:** M.

**B24. Item response theory & optimal experimental design — psychometrics `[within-subject]` `EXTENDS §10.4`**
IRT models each item's discriminative power and each subject's ability, then selects the
next item to maximise information. §10.4's active-calibration test used plain uncertainty
sampling and found a real but narrow effect (p=0.38 on the headline). IRT's contribution is
a *per-trial* information criterion rather than a per-trial uncertainty criterion, and it
separates item difficulty from subject ability. **Cost:** M. Modest upside, but it is the
principled version of a mechanism that already showed a real (if narrow) signal.

**B25. Seismic interferometry / Green's-function retrieval — geophysics `[label-free]` `NEW`**
Cross-correlating ambient noise between two receivers retrieves the impulse response between
them — you characterise a medium you cannot probe directly, using only signals that are
already there. Transplant: cross-correlate channel pairs *between sessions* to measure the
session-to-session transfer function directly, and invert it — a signal-level (rather than
covariance-level) session alignment. **Headwind:** this is close in spirit to §10.5's
closure-phase idea, which the project showed does not survive full-matrix distortion.
**Cost:** L, low priority. Included because it is the natural signal-domain analogue of the
one cross-disciplinary idea that was already falsified, and it should be falsified the same
way (test under scalar-gain corruption and full-matrix mixing separately).

**B26. Gauge fixing — physics (already partly explored as §10.5) `[within-subject]` `EXTENDS §10.5`**
The closure-phase result was: a real EEG analogue of gain-invariance exists (~1e-15
precision) but it does not survive full-matrix distortion. The untried remainder is a
*gauge-fixing* formulation: choose per-session channel gains as the gauge freedom and fix
them by a constraint (e.g. unit total power per band) rather than trying to construct
gauge-invariant features. Cheap to test alongside A13's re-referencing ablations. **Cost:** S–M.

---

## Part C — Recommended execution order (the next six experiments)

Chosen for information value per unit cost, and because each one gates or feeds the next.

1. **A27 + A1 (S–M).** Adopt the decision rule and power statement; then measure the
   per-subject ceiling (eval-session within-session CV) and per-subject SNR, and do the
   bottom-three failure analysis. *Outcome that changes everything:* if subjects 2/5/6 are
   signal-limited, stop pushing global methods and pivot to A19 (predict and route) plus
   A6 (per-subject configuration) — the honest answer, and the project's own thesis.
2. **A29 + A2 + A8 + A12 (S–M).** The cheap within-subject work, plus the deliverable.
   A29 (accuracy vs number of labelled trials, and trials-to-threshold) costs nothing but
   refits, produces the project's most presentable figure, and may be the strongest result
   available without inventing anything. Alongside it: RPA stretch+rotation with
   pseudo-labels (A2), within-subject shrinkage priors (A8), and the alignment-estimator
   ablations (A12). All use existing, unit-validated primitives. *This is where I would
   expect the first genuine improvement over 66.9%.*
3. **B2/B15 (S–M).** RUV-style negative-control drift removal, with differential photometry
   as its baseline. Targets the cleaned-pipeline/ComBat failure mechanically and is
   label-free.
4. **A28 + B27 (M).** Latent-dynamics alignment, gated on synthetic data with a *known*
   planted drift. The strongest new lead in this file: the only approach that corrects a
   quantity Riemannian alignment provably cannot reach, backed by a sibling modality that
   solved the same problem, using code that already exists.
5. **A3 + B19 + B10 (M–L).** The leakage-safe *sequential* evaluation harness, which is a
   prerequisite for online adaptation, Kalman-filtered running estimates, entropy
   minimisation, and self-training (A11). Once built, four hypotheses become cheap to test.
6. **A4 + A5 (M).** Prototype alignment on unlabeled eval trials and OT/Wasserstein
   alignment. Both are literature-backed with same-dataset numbers (79.3% and +2.45pp), and
   both are label-free and within-subject — the two best-evidenced ways to actually beat
   66.9% rather than nibble at it.

Then, in parallel and decoupled from the accuracy race: **A20 + A31** (a dataset with ≥3
sessions per subject, and the longevity metric it unlocks — a structural capability the
project currently lacks), **B5** (per-subject frequency warping — the highest
expected-value-per-line item in Part B), and **A30** (per-subject self-supervised
pretraining — the honest test of whether the foundation-model idea or the public data is at
fault).

---

## Part D — Do not retry (closed out; from `docs/progress_and_direction.md`)

Listed so this file does not silently re-propose them:

- Population covariance correction in any form: SPD shrinkage (8-subject and OpenBMI-20),
  eigenvoice/anisotropic shrinkage (worse at k=3 and k=7), Module A's learned correction
  (3–15× worse than doing nothing on cross-dataset transfer), and its safety-bounded
  variant (still net negative). More reference subjects did not help; the limit was not
  scale.
- Meta-training on episodes that pool different subjects into one loss — it does not merely
  transfer badly, it fails to converge.
- Naive feature concatenation (CSP + wavelet + MiniRocket: 62.2% → 54.8%).
- Dimensionality reduction on windowed wavelet features (PCA decisively worse; L1 mixed).
- Hand-restricted motor-cortex channel sets (central-17 and narrow-9 both worse than all 22).
- ICP for within-subject realignment (real-data null, 60.4% vs 65.9% baseline).
- CPD at D=253 (documented curse-of-dimensionality failure, not a tunable bug).
- ComBat on tangent-space features (synthetic null, statistically tied on real data).
- Joint-embedding/foundation models for MI, per the literature (§11) — revisit only if the
  published MI gap closes.
- The adaptive-optics "guide star" in its original form for IV-2a/OpenBMI (no interleaved
  probes) — **but see A22: PhysioNet's rest trials may make it testable after all. That is a
  correction to this list, not an exception to it.**

---

## Part E — What "beating 66.9%" has to mean

To count as a real improvement, a candidate must clear all of these:

1. Same protocol: train on `0train`, test on `1test`, all 9 subjects, 4 classes, no labelled
   eval data.
2. Paired McNemar per subject on identical trials, and paired Wilcoxon across subjects, with
   p reported **and** the effect size in pp.
3. Per-subject rows shown, not just the mean (A1/A27).
4. Correction for the number of comparisons in the family, or an explicit
   "exploratory, not confirmed" label.
5. A synthetic ground-truth gate where the mechanism has a known right answer, **before**
   the real-data claim.
6. Reproduction of the current baseline (62.2% / 66.9%) inside the same run, so a
   configuration drift cannot masquerade as a gain.
7. The calibration-burden curve for the method (A29), not just its accuracy at n=288. A
   method that wins at the full 288 calibration trials but loses at 96 has not improved the
   product, whatever the mean says.

A +1pp result that clears all seven is worth more than a +4pp result that clears none.
