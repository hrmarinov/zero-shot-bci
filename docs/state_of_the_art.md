# State of the Art: Cross-Session/Cross-Subject EEG-BCI Adaptation

Living reference doc to cross-check our own results against the literature and
spot where we're reproducing known methods vs. entering new territory. Update
it as we read more papers or land new results.

## Read this caveat before comparing any number below to our own

**Almost none of these numbers are directly comparable to each other, or to
our Phase 1/2 results, without checking the protocol.** The three things that
silently change the ceiling:

1. **Number of classes.** 2-class (left/right hand) is much easier than our
   4-class IV-2a setup (left hand, right hand, feet, tongue). A lot of the
   highest-looking numbers below (80-90%+) are 2-class.
2. **Training data pool.** Most deep-learning papers pretrain on *many*
   subjects (sometimes the whole dataset minus the target) before adapting to
   one — our Phase 1/2 baseline trains on **one subject's own 288 calibration
   trials only**, nothing else. That's a harder, more data-starved setting by
   design (it matches "you just sat down, no other subjects' data available"),
   so a method that pretrains across 50+ subjects and then fine-tunes should
   *not* be read as directly beating our number.
3. **Paradigm.** Some subject-personalization papers below are on P300/ERP
   data, not motor imagery. The method ideas still transfer, the numbers don't.

Where known, each entry below notes classes/pool/paradigm so you can judge
comparability at a glance.

## Is "illiteracy" even the right frame? (read before pitching this)

Worth flagging early since it changes how the whole project should be
positioned: a real thread of the literature argues the term itself is
methodologically weak.

- **Thompson (2018)**, "Critiquing the Concept of BCI Illiteracy" — argues
  "illiteracy" wrongly locates the failure in the user rather than in system/
  algorithm design choices, an ethically loaded framing for something that
  may just be a fixable engineering gap.
  [DOI:10.1007/s11948-018-0061-1](https://doi.org/10.1007/s11948-018-0061-1)
- **Shuqfa & Lakas (2024)** propose "MI-BCI unfamiliarity" instead, but note
  the alternative framing hasn't itself been validated.
  [DOI:10.1109/bdcat63179.2024.00050](https://doi.org/10.1109/bdcat63179.2024.00050)
- **Becker et al. (2022)**, "BCI Illiteracy: It's Us, Not Them. Optimizing
  BCIs for Individual Brains" — same argument from a different angle: treat
  the "illiterate" subject as a signal that the *system* needs to adapt to
  that brain, not evidence the person can't use a BCI.
  [DOI:10.1109/bci53720.2022.9735007](https://doi.org/10.1109/bci53720.2022.9735007)
- Separately, **definitions of illiteracy aren't standardized** — thresholds,
  trial counts, and criteria vary across studies, and static vs.
  subject-adaptive decoding methodology changes who even counts as illiterate
  (Lee et al., 2019; Allison & Neuper, 2010,
  [DOI:10.1007/978-1-84996-272-8_3](https://doi.org/10.1007/978-1-84996-272-8_3)).
  Reported illiteracy rates in the literature range roughly **15-30%** of
  users depending on paradigm and threshold used.

**Practical takeaway:** this project's framing — a personalization/adaptation
method that shrinks the illiterate-subject count — already sits on the "it's
the system's fault, not the user's" side of this debate, which is the more
defensible position in the literature. Worth explicitly saying so in any
pitch, rather than leaning on "illiteracy" as if it's an uncontested,
well-defined clinical category.

## Our results so far (for reference)

| Phase | Method | Setting | Cross-session accuracy | Notes |
|---|---|---|---|---|
| 1 | CSP+LDA, no adaptation | 4-class, IV-2a, N=1 subject only | 62.2% avg (34.0-80.9% range) | [results/phase1_baseline_iv2a.csv](../results/phase1_baseline_iv2a.csv) |
| 2 | + Riemannian alignment (per-session recentering) | 4-class, IV-2a, N=1 subject only | 66.9% avg, +4.7pp, 3/9 subjects significant (McNemar) | [results/phase2_riemannian_alignment_iv2a.csv](../results/phase2_riemannian_alignment_iv2a.csv) |

---

## 1. Published dataset baselines (the numbers to beat, from the dataset papers themselves)

These are the reference points the project overview specifically calls out.

- **Won et al. 2022** (the "5-day cross-session dataset" — 25 subjects, 2-class
  L/R hand, 5 sessions 2-3 days apart): within-session (WS) **68.8%**,
  cross-session no adaptation (CS) **53.7%**, cross-session *with* their
  adaptation method (CSA) **78.9%** — CSA actually exceeds WS, which is a
  signal their CSA leverages pooled multi-subject data, not just a same-subject
  recalibration. [Nature Scientific Data](https://www.nature.com/articles/s41597-022-01647-1)
- **Lee et al. 2019** (OpenBMI, 54 subjects, 3 paradigms): average decoding
  accuracy 71.1% across MI/ERP/SSVEP combined; **motor imagery illiteracy rate
  53.7%** (fraction of subjects who don't reach usable accuracy) vs. 11.1% for
  ERP and 10.2% for SSVEP — motor imagery is the hardest paradigm for
  illiteracy specifically, which matches why this project targets MI.
  [GigaScience](https://academic.oup.com/gigascience/article/8/5/giz002/5304369)
- **MOABB within-session benchmark** (2-class BNCI2014_001/IV-2a, 5-fold CV,
  not cross-session): CSP+LDA 82.3%, MDM (Riemannian) 81.7%, ShallowConvNet
  86.2%, EEGNet 77.2%. [MOABB benchmark results](https://moabb.neurotechx.com/docs/paper_results.html) /
  [MOABB paper](https://arxiv.org/abs/2404.15319)

## 2. Riemannian alignment / recentering (what we implemented in Phase 2)

- **Euclidean Alignment** (He & Wu, 2020) — the original method: whiten each
  subject/session's trials by their own mean covariance, unsupervised, no
  labels needed. Validated across 13 BCI paradigms.
  [IEEE TBME paper](https://arxiv.org/pdf/1808.05464)
- **Riemannian Procrustes Analysis** (Rodrigues, Jutten & Congedo, 2019) — adds
  stretching + rotation on top of recentering (we only implemented recentering,
  the first and cheapest step). Reports a further 2.7% improvement over prior
  Riemannian methods. [IEEE paper](https://ieeexplore.ieee.org/document/8588384/)
- **Systematic evaluation of EA + deep learning** (Junqueira et al., 2024) —
  EA improves cross-subject deep-learning transfer by **+4.33%** (offline) and
  cuts training convergence time by **>70%**.
  [J. Neural Eng. paper](https://arxiv.org/pdf/2401.10746)
- **"Overcoming the BCI Calibration Bottleneck"** (RA + Stochastic Weight
  Averaging, 2026) — 2-class BNCI2014-001. Ablation: raw baseline 57.6% →
  bandpass only 76.7% → **+ Riemannian Alignment 83.1%** (+6.4pp from RA
  alone) → full pipeline (+ ICA + SWA) 91.0% stable / 93.75% peak. Directly
  shows RA's marginal contribution isolated from other tricks — useful
  apples-to-apples number for RA specifically, modulo the 2-class vs. our
  4-class gap. [arXiv:2607.16225](https://arxiv.org/html/2607.16225)

**Read on our own number:** our Phase 2 average (+4.7pp, 62.2%→66.9%) lands
right in the range these papers report for alignment alone (+4-6pp), on a
harder 4-class, single-subject-only setup. That's a good sign we implemented
it correctly and aren't leaving obvious accuracy on the table for this
specific technique — the "known baseline" is genuinely reproduced, not
under- or over-performing versus literature.

**Important extension we haven't implemented — recentering *across*
subjects, not just within one subject's own sessions.** Kumar et al. (2024,
PNAS Nexus), "Transfer learning promotes acquisition of individual BCI
skills": trains a decoder on **one expert subject only**, then deploys it to
brand-new naïve users via two recentering variants —
- *Generic Recentering (GR)*: unsupervised, matches the naive subject's data
  distribution to the expert's decoder in real time — this is our
  `RiemannianAlignment` applied cross-subject instead of cross-session, no
  conceptual change needed.
- *Personally Assisted Recentering (PAR)*: GR plus a supervised recalibration
  step using labeled data from the very first session.

Validated in **18 real BCI-naïve subjects, 5 online sessions each** (not
offline replay — actual closed-loop control), two tasks. Both variants gave
naive users usable control from session 1 and significant skill growth over
sessions (GR bar-task kappa 0.264→0.469, P=0.02; PAR 0.405→0.680, P=0.001).
This is the strongest real-world (online, human-in-the-loop) evidence in this
whole doc that recentering-style methods work outside offline benchmarks.
[DOI:10.1093/pnasnexus/pgae076](https://doi.org/10.1093/pnasnexus/pgae076)

Note this uses a single, fixed, arbitrarily-chosen expert as the recentering
target for every naive subject — nobody in what we've read *selects or
weights* which reference to recenter toward based on how similar the target
subject's own signature is to the available experts. See the fingerprint
conclusion at the bottom for why that gap matters.

## 3. Test-time / online adaptation (calibration-free, the Phase 2 "hero" option from the project plan)

- **Calibration-free online test-time adaptation** (2311.18520) — combines
  Riemannian Alignment + running Batch-Norm stat updates + entropy
  minimization, evaluated online per-trial with no labels at all at test time.
  IV-2a: cross-session 76.2%→**79.7%** (+3.6pp), cross-subject 57.4%→**67.3%**
  (+9.9pp). IV-2b: cross-session 84.8%→86.6%, cross-subject 78.8%→83.5%. Note
  the much higher baseline (76% vs. our 62%) — this method pretrains a deep
  net across many subjects before online-adapting, so it's not the same
  starting point as our single-subject CSP+LDA.
  [arXiv:2311.18520](https://arxiv.org/html/2311.18520)

## 4. Deep learning architecture improvements (not adaptation per se, context for ceiling numbers)

Numbers here vary a lot by protocol (within-session vs. cross-session vs.
cross-subject often isn't stated consistently across papers) — treat these as
rough ceiling indicators, not a clean comparison table:

- MS-AFM (multi-scale CNN, inception+residual): 86.0% on IV-2a (protocol
  unclear from abstract).
- EEGNet zero-shot (no fine-tuning) in a meta-learning cross-subject
  evaluation: **43% ± 7%** on IV-2a — a useful reminder that plain deep nets
  without any adaptation can do *worse* than classical CSP+LDA cross-subject,
  since they need more data than a single calibration session provides.
  [EEG-TCNet paper](https://arxiv.org/pdf/2006.00622)

## 5. Meta-learning / few-shot personalization

- **MAML for EEG-MI decoding** (2103.08664) — pretrain across source
  subjects, fine-tune with a handful of labeled trials from the new subject.
  Known failure mode noted in follow-up work: few-shot fine-tuning on very
  few samples can overfit and *hurt* target performance.
- **EEG-Reptile** (2412.19725) — Reptile-based meta-learning for few-shot
  subject adaptation, reports outperforming from-scratch and standard
  transfer-learning baselines on PhysioNet + BCI Competition data (paper
  doesn't give a single clean headline number extractable from the abstract).
  [arXiv:2412.19725](https://arxiv.org/pdf/2412.19725)

## 6. Prototype-based domain adaptation

- **PMANet** (2025) — multi-scale spatio-temporal feature extractor +
  prototype alignment for cross-session MI decoding. Important distinction:
  the "prototypes" here are **per-class** (4 class-prototype vectors, one per
  MI class, derived from the source-session classifier's weights), *not*
  per-subject — unlabeled target-session trials are softly aligned to these
  class prototypes. Reports **79.32%** on 4-class IV-2a cross-session and
  86.36% on 2-class IV-2b cross-session, "outperforming SOTA" per their claim
  — this is the closest same-dataset, same-protocol-family number we've
  found to compare against our 66.9%, though we couldn't confirm whether
  their training pool is single-subject or pooled across subjects (ACM page
  blocked full-text fetch, so treat this number as directionally useful but
  unverified on that point).
  [ACM DL page](https://dl.acm.org/doi/full/10.1145/3777577.3777654)

## 7. Subject signature / embedding / "fingerprint" approaches — closest prior art to your idea

This is the section most relevant to what you're proposing. **The general
idea — compute a compact per-subject representation, use it to personalize or
fast-fine-tune a shared decoder — is an active 2025 research direction with
several independent implementations already published.** None of them use the
word "fingerprint," but they're the same concept:

- **Kim, Shin & Kam (2023)**, "Bridging the BCI illiteracy gap: a
  subject-to-subject semantic style transfer for EEG-based motor imagery
  classification" — **the closest match found, on our exact dataset.**
  9 subjects, 2 sessions, 4-class **BCI Competition IV-2a** — literally the
  dataset our Phase 1/2 experiments use. Method: pick one high-performing
  "expert" subject (they use subject 3) as the style source; a generator
  network transforms a lower-performing subject's trials to match the
  expert's feature distribution (style loss via Gram matrices + KL
  divergence) while a content loss preserves the target's own class
  information; predictions are then ensembled from both the expert and
  target classifiers. This is *not* a compact vector fingerprint — it's a
  learned generative transform between a specific pair of subjects — but it
  targets exactly our project's stated worst-case subjects. Reported
  results: their two worst-performing ("illiterate") subjects improved
  53.54%→58.33% and 60.90%→67.01%; overall mean across all 9 subjects rose
  76.37%→80.66% (protocol/session-averaging details differ from our strict
  train-session-1/test-session-2 split, so treat the absolute numbers as
  directional, not directly comparable to our 62.2%/66.9% — but the *subject
  2 and 6 worst-case pattern* is worth checking against our own per-subject
  results). [DOI:10.3389/fnhum.2023.1194751](https://doi.org/10.3389/fnhum.2023.1194751)

  **Cross-check against our own results:** subjects 2 and 6 are *also* among
  our weakest in Phase 1 (cross-session baseline 42.4% and 47.6%, both well
  below the 62.2% average) — two independent analyses of the same public IV-2a
  dataset flag the same subjects as hardest. That's reassuring: it suggests
  subject 2/6 difficulty is a real property of that data, not an artifact of
  either analysis's pipeline, and makes them the natural test subjects if we
  prototype anything from this section.
- **Explicit modelling of subject dependency in BCI decoding** (2509.23247,
  2025) — the closest match among the *learned-embedding* methods
  specifically (Kim et al. above is closer overall, but uses a generative
  transform rather than a compact vector). Learns a per-subject embedding
  table (one unit-normalized vector per subject), tests two conditioning
  mechanisms:
  - *Projection*: scales feature magnitude by cosine similarity to the
    subject's embedding vector.
  - *FiLM*: splits the subject embedding into per-feature scale (γ) and bias
    (β) applied to the network's internal features.
  For a new/unseen subject, only the subject-conditioning layer is
  fine-tuned (rest of the network stays frozen) using incremental batches of
  calibration data. Reports: baseline 0.609 MCC (zero-shot) → 0.689 MCC
  (4-batch fine-tune) on the projection variant; FiLM underperformed the
  non-conditioned baseline in their tests. **Caveat: evaluated on ERP (P300)
  data, not motor imagery** — the mechanism should transfer, the numbers
  won't directly. [arXiv:2509.23247](https://arxiv.org/html/2509.23247)
- **Stacked LoRA for Subject-Adaptive EEG Foundation Models** (2607.03094,
  2025) — explicitly calls the per-subject low-rank adapter weights "subject-
  specific neural signatures." Splits each adapted layer's update into a
  Global adapter (shared, trained jointly across all subjects) and a
  Subject-Specific adapter (absorbs individual variability), combined at
  inference. Tested on IV-2a, PhysioNet, and clinical datasets across
  multiple backbones; claims best accuracy in most backbone/dataset
  combinations vs. subject-specific-only or global-only LoRA (exact numbers
  behind the abstract, worth a follow-up read).
  [arXiv:2607.03094](https://arxiv.org/abs/2607.03094)
- **TCPL** (task-conditioned prompt learning, 2025) — encodes "subject
  individuality" as prompt tokens fed into a TCN+Transformer backbone under a
  meta-learning framework, for few-shot cross-subject MI decoding on
  GigaScience, PhysioNet, and IV-2a. Ablation: removing the prompt module
  drops accuracy to 74.6%, implying the subject-token mechanism is
  responsible for a meaningful chunk of their reported gain (exact
  with-module number not extracted, worth reading in full).
  [Frontiers in Neuroscience](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1689286/full)
- **Riemannian mean covariance as a literal geometric subject fingerprint**
  — the classical-methods analogue, and directly relevant since we already
  compute this in [`RiemannianAlignment`](../src/adaptation/riemannian_align.py):
  a subject/session's mean covariance matrix (a point on the SPD manifold) is
  used in several papers as a compact signature to measure distance between
  subjects and select which *source* subjects are most similar to a new
  target before pooling their data for transfer learning ("training
  accuracy-based subject selection", log-Euclidean data alignment + subject
  selection). We currently only use each subject's own mean covariance to
  recenter *their own* data — we don't yet use it to compare *across*
  subjects. [Selective Cross-Subject Transfer Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC8595943/) /
  [Hybrid Riemannian+Euclidean subject selection](https://www.researchgate.net/publication/348165658_Transfer_Learning_Based_on_Hybrid_Riemannian_and_Euclidean_Space_Data_Alignment_and_Subject_Selection_in_Brain-Computer_Interfaces)
- **EEG brain fingerprinting** (aperiodic 1/f power-spectrum component,
  functional-connectivity fingerprinting) — a different goal (subject
  *identification*/biometrics, not decoder personalization) but the same
  underlying idea of a compact, stable per-subject signature. Worth reading
  for signal-processing ideas (what makes a signature stable and
  discriminative) even though the application differs.
  [EEG fingerprinting paper](https://arxiv.org/pdf/2001.09424)

## 8. Predicting illiteracy *before* a full session (checked this specifically — see conclusion below)

This is close to one of the differentiators floated below, so worth stating up
front: **it's also already an active area, going back to 2009.**

- **Predicting BCI performance to study BCI illiteracy** (2009) — the
  foundational paper: a 2-minute "relax, eyes open" resting-state recording
  from 3 Laplacian channels predicts subsequent MI-BCI performance.
  [Springer](https://link.springer.com/article/10.1186/1471-2202-10-S1-P84)
- **Laterality index + cortical activation strength** predictors — identify
  BCI-inefficient users from pre-session features with 88.2% sensitivity /
  85.7% specificity (2-class) and 100% / 87.5% (brain-switch BCI).
- **Resting-state connectivity between literate/illiterate groups** (2023-24,
  bioRxiv/PubMed) — alpha-band and Granger-causality connectivity metrics
  differ between groups; 2-minute, 3-channel resting recordings get
  classifier accuracy up to ~60-62% at predicting the eventual literate/
  illiterate split. [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.25.568691v1.full)
- **Resting-state microstate analysis** (2023) — AUC 0.83 predicting MI-BCI
  performance from microstate features, outperforming spectral-entropy
  predictors. [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10526389/)
- **Ahn, Cho, Ahn & Jun (2013)**, "High Theta and Low Alpha Powers May Be
  Indicative of BCI-Illiteracy in Motor Imagery" — the most concrete
  mechanistic biomarker found in this search, with actual numbers. 52
  subjects (primary), validated on the 9 BCI Competition 2008 subjects
  (i.e. IV-2a/2b family). Illiterate subjects show **high theta (4-8Hz,
  frontal + posterior-parietal) and low alpha (8-13Hz, nearly whole-scalp)
  power**, consistent across resting, non-task, and motor-imagery states —
  i.e. this shows up even at rest, before any task starts, which is exactly
  what a pre-session predictor needs. Literate-vs-illiterate classification
  from these features: **82.35% accuracy** (72.22% without outlier
  exclusion); performance-prediction correlation **r=0.59 (r²=0.34, N=61)**,
  improving to **r=0.70 (r²=0.50, N=54)** after removing 7 outliers.
  [DOI:10.1371/journal.pone.0080886](https://doi.org/10.1371/journal.pone.0080886)

None of these frames their resting-state predictor as a "fingerprint" used to
*condition or fine-tune a decoder* — they stop at classifying literate vs.
illiterate, they don't feed the predictor into personalizing the model. That
gap (predictor → actual conditioning signal, not just a screening label) is
narrower and more technical than "predict illiteracy" as a category, but it's
real and it's what's left after this search.

## 9. Co-adaptive / zero-training calibration (classic literature, updates *during* the session)

A distinct family from everything above: instead of a two-stage
calibrate-then-deploy pipeline (which is what Phase 1/2 and most of §1-§8
are), these methods update the classifier **online, during the feedback
phase itself**, with the user in the loop from trial one. Foundational, still
frequently cited as the baseline to beat for "zero training":

- **Krauledat, Tangermann, Blankertz & Müller (2008)**, "Towards Zero
  Training for Brain-Computer Interfacing" — the original zero-training
  proposal: reuse spatial filters/patterns from a database of prior sessions
  instead of collecting new calibration data.
  [DOI:10.1371/journal.pone.0002967](https://doi.org/10.1371/journal.pone.0002967)
- **Vidaurre & Blankertz (2009)**, "Towards a Cure for BCI Illiteracy" and
  **Vidaurre, Sannelli, Müller & Blankertz (2011)**, "Machine-Learning-Based
  Coadaptive Calibration" — the classifier adapts continuously during
  feedback rather than freezing after a fixed calibration block; explicitly
  framed as an illiteracy countermeasure, from the same lineage as the
  "illiteracy" term itself.
  [DOI:10.1007/s10548-009-0121-6](https://doi.org/10.1007/s10548-009-0121-6) /
  [DOI:10.1162/neco_a_00089](https://doi.org/10.1162/neco_a_00089)
- **Kindermans, Schreuder, Schrauwen, Müller & Tangermann (2014)**, "True
  Zero-Training Brain-Computer Interfacing" — validated online (not just
  offline replay), unsupervised classifier initialization with no
  calibration block at all.
  [DOI:10.1371/journal.pone.0102504](https://doi.org/10.1371/journal.pone.0102504)
- **Thielen, Marsman, Farquhar & Desain (2021)** — the same zero-training
  idea applied to code-modulated VEP (cVEP), showing the concept generalizes
  beyond motor imagery.
  [DOI:10.1088/1741-2552/abecef](https://doi.org/10.1088/1741-2552/abecef)
- **Gao et al. (2023)** — CNN + a large EEG database eliminates/shortens
  P300 calibration, validated in a true **online** study (most zero-training
  work up to this point was offline simulation of online use).
  [DOI:10.1109/tnsre.2023.3259991](https://doi.org/10.1109/tnsre.2023.3259991)

**Relevance to us:** none of this is what we built in Phase 1/2 (ours is
strictly offline, two-stage, calibrate-once-deploy-once) or what the Phase 2
"hero" option (test-time adaptation, §3) would be either, since TTA there
still starts from a pretrained-elsewhere model rather than adapting live from
scratch. If we want a genuinely calibration-free pitch (not just
calibration-*light*), this online co-adaptive family — not just Riemannian
alignment or embeddings — is the comparison class to be measured against, and
it's been worked on since 2008 without (per this search) ever reporting
100% elimination of illiteracy — a ceiling worth knowing about before
promising to "solve" it outright.

---

## Where this leaves the fingerprint idea

**It's not new territory as a general concept, and the gap is narrower than
it looked on the first pass.** Round one (§7 minus Kim et al.) found
subject-embedding/LoRA/prompt personalization as a general 2025 trend. This
second round found something more specific and closer to home: **Kim et al.
(2023) do subject-to-subject transfer targeting illiterate users on our exact
dataset (IV-2a), and Kumar et al. (2024) do cross-subject recentering — the
same math as our Phase 2 code, just applied subject-to-subject instead of
session-to-session — validated in real online sessions with 18 people.** If
the pitch is "we invented per-subject signatures for BCI personalization,"
or even "we invented using recentering to fix illiteracy," neither survives
this literature check anymore.

**What's still genuinely open**, now narrowed to two specific, sourced gaps
rather than the broader claims from the first pass:

1. **Nobody selects the recentering/transfer target based on subject
   similarity.** Kim et al. use one arbitrarily-picked expert (subject 3) as
   the style-transfer source for everyone. Kumar et al. use one fixed expert
   as the recentering target for all 18 naive subjects. Neither paper asks
   "which reference is closest to *this* new subject" — they pick one
   reference and apply it universally. A fingerprint (their own Riemannian
   mean covariance, which Phase 2 already computes, or a resting-state
   feature vector per §8/Ahn et al.) used to **select or weight which
   expert/source subject to recenter or transfer toward, per new subject**,
   is a specific, sourced, and still-open combination — not "personalization"
   in the abstract, but this particular missing step in two concrete,
   already-published pipelines.
2. **The illiteracy *predictor* (§8/§9) never becomes the *conditioning
   input*.** Every resting-state predictor (Ahn et al.'s theta/alpha, the
   connectivity/microstate work) stops at a binary screening label. Every
   personalization method (§7) starts from calibration-session data, not a
   pre-session predictor. Wiring these together — the same short unlabeled
   recording that flags a subject as likely-illiterate *also* becomes the
   fingerprint that selects their reference subject or conditions their
   decoder (i.e., (1) computed from resting-state data instead of, or in
   addition to, calibration-session data) — still hasn't turned up.

Recommendation: (1) is the sharper and cheaper of the two to prototype —
it's a direct, small extension of Phase 2's existing `RiemannianAlignment`
(compute each candidate expert's mean covariance once, measure Riemannian
distance from a new subject's short recording to each, recenter toward the
nearest one instead of a fixed reference) and it has two concrete named
baselines (Kim et al., Kumar et al.) to beat rather than a vague "personalize
better" target. (2) is the more ambitious combination and depends on (1)
working first. Treat all of this as a literature-search-depth conclusion
(search snippets + full-text fetches where available, several paywalled), not
a systematic review — read Kim et al. and Kumar et al. in full before
committing engineering time, since they're now the two most important papers
in this whole document for deciding what to build next.

---

## 10. Cross-disciplinary transplant ideas (creative synthesis, not a literature survey)

Everything above surveys what BCI researchers have already tried. This
section is different in kind: other fields solved *structurally similar*
problems — extract a weak, individual-specific, noisy signal and either
amplify it or correct for its drift — decades ago, using tools BCI research
mostly hasn't imported. For each one I checked (via targeted search, not
exhaustive) whether someone already made the transplant into EEG/BCI, so
"unexplored" here means "didn't turn up," not "guaranteed novel" — treat as a
lead to verify further, not a confirmed gap the way §1-§9 are.

These are ranked by how concrete/cheap they are to actually prototype against
data we already have (IV-2a, OpenBMI), not by how interesting they sound.

### 10.1 Shrink the fingerprint itself, using math already published for a sibling imaging modality — strongest, most buildable lead

**Source field: diffusion MRI (DTI) statistics.** DTI produces the *exact
same mathematical object* our Riemannian alignment does — a symmetric
positive-definite (SPD) covariance/diffusion-tensor matrix per subject, per
brain region — and DTI researchers have a mature, peer-reviewed answer for
"this per-subject SPD matrix is noisy, especially from short scans; how do
we get a better estimate?" **Empirical Bayes shrinkage on the SPD manifold**
(Tweedie's formula + Stein's Unbiased Risk Estimate, adapted to the
geometry): shrink a noisy individual SPD estimate toward a population mean
SPD matrix, with the shrinkage *amount* determined automatically from the
data (not a hand-tuned hyperparameter) — validated on real diffusion-tensor
data from Parkinson's patients' motor tracts.
[arXiv:2007.02153](https://arxiv.org/pdf/2007.02153)

**I checked — this has not been applied to EEG/BCI.** It's sitting, fully
worked out, in a neighboring neuroimaging field, using the identical SPD
manifold our `RiemannianAlignment` already operates on.

**The transplant:** right now, each subject's "fingerprint" (mean covariance
in `RiemannianAlignment`) is the raw empirical mean of whatever trials we
feed it — noisy by construction when computed from a short, calibration-free
recording, which is exactly the regime the fingerprint idea needs to work in
(§7 conclusion). Apply this Tweedie/SURE shrinkage estimator instead: shrink
each new subject's short-recording covariance toward the *population* mean
covariance (computed once from OpenBMI's 54 subjects), with the shrinkage
strength set by the data itself. This should make fingerprints computed from
very little data dramatically more reliable, is a small, well-specified
addition to code that already exists, and has a hard theoretical guarantee
(dominates the naive per-subject estimator under squared error loss) rather
than "seems to help" — the kind of claim a reviewer can't easily poke holes
in.

**Why a big institution probably wouldn't chase this specifically:** it's
"boring old-school statistics," not a new architecture or a bigger model —
unlikely to be the centerpiece of a flashy deep-learning paper, and it
requires noticing a cross-field connection (DTI stats ↔ EEG covariance) that
sits between two literatures that don't usually cite each other. Cheap:
implementable in an afternoon on top of Phase 2 code, no new data collection,
no GPU needed.

#### Prototyped — honest result: null on real data, worth knowing why

Built and rigorously tested (not just implemented on faith). Summary; full
numbers below.

- **Estimator validated on synthetic data.** With a correctly-constructed
  synthetic population (subjects genuinely drawn around a shared mean),
  shrinkage beats the raw estimate at every sample size, gap shrinking to
  zero as `n` grows — exactly the textbook Efron-Morris/Tweedie behavior.
  The math is implemented correctly.
  [src/adaptation/spd_shrinkage.py](../src/adaptation/spd_shrinkage.py)
- **On real IV-2a data (leave-one-subject-out, 8 reference subjects):
  no significant benefit at any sample size.** Paired Wilcoxon test,
  shrunk vs. raw log-Euclidean mean, n=8..288 trials, 9 subjects: p > 0.3
  throughout, wins split roughly 50/50, no monotonic trend with `n` (the
  signature a real effect should show). Two subjects (5, 9) were actively
  *hurt* by shrinkage at small n (−1 to −3.4pp).
  [results/phase2b_spd_shrinkage_iv2a.csv](../results/phase2b_spd_shrinkage_iv2a.csv)
- **Generalization check on IV-2b (3 channels → 6-dim tangent space, vs.
  IV-2a's 253-dim): same null pattern.** This is informative on its own —
  if "too few reference subjects relative to tangent-space dimensionality"
  were the whole story, IV-2b's much lower dimensionality (8 subjects is
  plenty for 6 dimensions) should have shown a clear benefit. It didn't:
  one nominally significant n out of six tested (p=0.02) with the opposite
  sign at another n — consistent with multiple-comparisons noise, not a
  real effect. [results/phase2b_spd_shrinkage_iv2b.csv](../results/phase2b_spd_shrinkage_iv2b.csv)
- **Eigenvoice-stacked (anisotropic) shrinkage: clearly *worse*, not
  better, at both k=7 and k=3 components.** Tested per your instruction to
  iterate toward eigenvoices depending on results. On IV-2a, both component
  counts underperform plain raw/isotropic-shrunk at every sample size
  (~62% vs. ~64-67%) — and k=3 doing about as badly as k=7 rules out "too
  many components, overfitting the 8 reference subjects" as the sole
  explanation. The more likely read: an 8-subject reference pool can't
  define a low-rank subspace that captures real between-subject signal
  *for this downstream classification task* at any rank — forcing the
  fingerprint into that subspace throws away real, subject-specific
  structure that the full-dimensional raw estimate preserves.
  [results/phase2c_eigenspace_k7_iv2a.csv](../results/phase2c_eigenspace_k7_iv2a.csv) /
  [results/phase2c_eigenspace_k3_iv2a.csv](../results/phase2c_eigenspace_k3_iv2a.csv)

**Diagnosis (initial, at 8 reference subjects):** the estimator is correct;
8-9 subjects is enough to validate that the math works, not enough to
reliably estimate between-subject variance in a 253-dimensional tangent
space (or, per the eigenvoice result, to define a useful low-rank subspace
at all) — this looked like the likely bottleneck. **Update: tested directly
at 20 reference subjects below, and this was not the primary explanation**
— see "Tested at OpenBMI scale" further down for the full, revised
diagnosis (real subject-level heterogeneity, not reference-pool size).

**Channel-montage compatibility check (was "untested" above — now checked):**
IV-2a's 22 EEG channels and OpenBMI's 62 EEG channels were compared by exact
name (both use standard 10-20/10-10 nomenclature). **21 of IV-2a's 22
channels are present, identically named, in OpenBMI's montage** — the only
miss is `FCz`, which sits at the center of four OpenBMI channels that are
all present (`FC1`, `FC2`, `Fz`, `Cz`), so it's a natural spherical-spline
interpolation candidate if an exact 22-channel match is needed, or the
shared 21 channels can be used directly. So **OpenBMI is a viable
reference pool for this specific bottleneck** — 54 subjects vs. the current
8, which is the fix this diagnosis calls for. One more preprocessing detail:
OpenBMI is sampled at 1000Hz vs. IV-2a's 250Hz, needing a resample step
before combining (MOABB's paradigm handles this, not a blocker). Note this
check cost ~1.2GB of downloaded data for one OpenBMI subject (two ~600MB
session files) just to read channel names — the full 54-subject dataset
would be a substantially larger download if this gets pursued (verified via
one-off inspection, not saved as a script/result file).

#### Tested at OpenBMI scale (20 subjects) — the "reference pool too small" diagnosis was wrong

Built the actual cross-dataset pipeline: 21 shared channels (restricted and
reordered to match by name), trace-normalization to remove absolute-scale
differences between the two EEG systems from the variance estimate (kept
provably harmless to downstream CSP+LDA accuracy — see the `normalize_trace`
docstring in [src/adaptation/spd_shrinkage.py](../src/adaptation/spd_shrinkage.py)),
20 OpenBMI subjects as an external reference pool (no leave-one-out needed
this time, since the pool no longer overlaps the IV-2a target subjects).
[experiments/phase2d_openbmi_shrinkage.py](../experiments/phase2d_openbmi_shrinkage.py) /
[results/phase2d_openbmi20_shrinkage_iv2a.csv](../results/phase2d_openbmi20_shrinkage_iv2a.csv)

One real bug surfaced and got fixed on the way: the first full run collapsed
to exact chance-level accuracy (25.0%, ~zero variance) from a scale mismatch
between the trace-normalized shrunk covariance and the natural-scale raw
baseline it was being compared against — fixed by rescaling the shrunk
result back to the target's own natural magnitude after computing the
shrinkage weight in normalized space. Also hit two transient network
failures mid-download (`ChunkedEncodingError`, connection dropped by the
host) requiring a retry loop added to the loader — not a code bug, the host
serving this dataset is measurably flaky for sustained multi-hundred-MB
transfers.

**Result: still no significant benefit.** With the between-subject variance
now reliably estimated (20 real subjects, not 8: `between_subject_var =
0.045`, `trial_sampling_var = 0.032`, a stable, sensible ratio — not the
noisy small-sample estimate from before), the shrinkage weight comes out
high (~0.92 even at n=8, →1 as n grows), meaning the data itself says not to
shrink aggressively even from few trials. Paired Wilcoxon test, shrunk vs.
raw log-Euclidean baseline, across all 9 IV-2a subjects: p=0.43-1.0 at every
sample size, no trend. Also not significantly different from the
affine-invariant baseline (p=0.10-0.98).

**This refutes the earlier diagnosis as the primary explanation.** Going
from 8 to 20 reference subjects gave a properly-determined between-subject
variance estimate (vs. the earlier noisy one), but the qualitative
conclusion didn't change — there's no reason to expect the full 54-subject
OpenBMI pool would either, so it wasn't pulled. What the 20-subject run adds
that the 8-subject one didn't have the power to show: **real, opposite-
signed subject-level effects that cancel in the average.** Subject 2 is
consistently and substantially *hurt* by shrinkage (-3.5 to -6.2pp across
every sample size); subject 9 is consistently and substantially *helped*
(+1.9 to +4.2pp across every sample size). Both patterns are large and
stable across n, meaning they're real, individual properties — the aggregate
null hides a genuine subject-level split. Shrinkage toward *one shared*
population mean helps subjects who are close to that mean and actively hurts
subjects who are genuine outliers, exactly the effect a formally
heterogeneous (mixture-of-populations, not single-population) model would
predict — this points toward the source-selection framing from the §7
conclusion (pick or weight *which* reference to shrink toward, per subject,
rather than one global mean) as the more promising next refinement, not more
reference subjects.

**Verdict: closing this out.** The transplant works as advertised
mathematically (validated on synthetic data), reproduces cleanly across two
independent datasets and two reference-pool scales, and the honest answer at
every stage was "no significant average benefit," with the real finding
being *why* — a mean-shrinkage model is the wrong shape for data with real
subject-level heterogeneity, not a data-scarcity problem fixable by more
subjects. Not spending the ~40-50GB and further compute to test all 54
OpenBMI subjects, since nothing in this evidence suggests scale was the
limiting factor.

The other transplant ideas in §10.2-10.5 have since been resolved one way or
another, and each section below carries its own result subsection: §10.2
(eigenvoices) was tested as part of §10.1's work and came out *worse* than
plain isotropic shrinkage, §10.3 turned out to be genuinely untestable with the
datasets available, and §10.4 and §10.5 were both built and tested (a real but
narrow effect, and a real invariance that solves a narrower problem than this
project needs, respectively).

### 10.2 Eigenvoices: compress "who this subject is" into ~10-30 numbers — tested, and worse than isotropic shrinkage

**Status: built and tested.** Implemented as the anisotropic (eigenvoice)
variant of §10.1's shrinkage estimator and evaluated on IV-2a at k=3 and k=7
components: clearly *worse* than plain isotropic shrinkage at both ranks, and
k=3 doing as badly as k=7 rules out overfitting the small reference pool as the
explanation. See §10.1's result subsection
([results/phase2c_eigenspace_k3_iv2a.csv](../results/phase2c_eigenspace_k3_iv2a.csv),
[k7](../results/phase2c_eigenspace_k7_iv2a.csv)) for the numbers and the
diagnosis. The description below is kept as the original proposal.

**Source field: speech recognition, speaker adaptation (Kuhn et al.,
1998-2000).** Before deep speaker embeddings existed, ASR solved fast
speaker adaptation by building an "eigenvoice space": run factor analysis /
PCA over many training speakers' acoustic models, keep the top ~20-30
components, and represent any new speaker — even from a few seconds of
speech — as a point in that low-dimensional space, estimated via a
closed-form MAP projection.
[Kuhn et al., Eigenvoices for Speaker Adaptation](https://www.isca-archive.org/icslp_1998/kuhn98_icslp.pdf)

**Checked — no EEG/BCI use of this specific technique turned up** (search
only returned pure speech-recognition literature). The closest things in §7
(subject embedding tables, LoRA "signatures") are deep-learning
re-inventions of the same idea with far more parameters and no closed-form
projection — eigenvoice's whole appeal is that it's classical, interpretable,
and needs almost no data to place a new subject.

**The transplant:** build the eigenspace from Riemannian tangent-space
representations of many OpenBMI subjects' covariance matrices (54 subjects
is enough to get a meaningful low-rank basis), then a new subject's
"fingerprint" *is* their coordinates in this space — literally the compact
signature you originally proposed, computed in closed form rather than
learned via backprop. Combine naturally with 10.1: build the eigenspace on
shrinkage-corrected covariances instead of raw ones.

**Why a big institution probably wouldn't chase this:** it's 25-year-old
speech tech, not novel-sounding enough for a grant pitch, and doesn't
produce a publishable "new deep architecture" — despite likely being cheaper
to validate and easier to explain to a non-ML reviewer than any embedding
network. Cheap: PCA-scale compute, no GPU.

### 10.3 A decoupled "guide star" for continuous drift correction

**Source field: astronomical adaptive optics.** Telescopes correct for
atmospheric turbulence — a distortion that's essentially unpredictable and
changes within milliseconds, a much harder drift problem than session-to-
session EEG shift — using a *reference beacon* (a bright star, or an
artificial laser guide star when no natural one is bright enough) that's
continuously measured and used to drive a deformable mirror in a closed
loop, correcting the *shared* distortion in real time without needing to
know anything about the actual science target.

**Checked — no direct combination of this concept with EEG-BCI drift
correction turned up.** What exists in BCI (closed-loop adaptive/co-adaptive
calibration, §9) updates the decoder using the *task* signal itself — labels
or task-related feedback are both the thing being decoded and the thing
driving adaptation. The guide-star idea is different: a cheap,
**task-irrelevant, repeatable reference probe** (e.g., a fixed short
eyes-open baseline, or a non-task auditory click) interleaved every few
minutes *during* a session, used purely to continuously re-estimate the
session's current covariance drift — decoupling "what corrects for drift"
from "what the subject is trying to do," rather than conflating them the
way one-shot alignment (ours) and task-driven co-adaptation (§9) both do.

**Why a big institution probably wouldn't chase this:** it requires changing
the experimental protocol (interleaved probes), not just the algorithm — a
harder sell for a grant built around reanalyzing existing public datasets,
and "add idle time to the protocol to measure nothing task-relevant" reads
as wasteful unless you already believe the payoff. **Honest limitation: this
can't be validated on IV-2a or OpenBMI as they stand** — neither has
periodic task-irrelevant probes recorded during a session. It needs either
new data collection or finding/checking a dataset that happens to have
interleaved rest periods we could repurpose as a "guide star" signal.

#### Assessed — the core claim genuinely can't be tested with data we have, but a diagnostic is informative

Confirmed rather than assumed: checked IV-2a's actual session structure
(6 runs of 48 trials each within the eval session) for anything that could
stand in for a decoupled, task-irrelevant probe. There isn't one — every
recorded segment is a labeled motor-imagery trial, part of the task itself.
No amount of clever reslicing of IV-2a or OpenBMI substitutes for a
genuinely task-irrelevant reference signal; building a real test needs new
data collection with interleaved non-task probes, which is out of scope
here. Did not force a mislabeled proxy experiment.

**What *is* checkable with existing data: does within-session covariance
drift enough for continuous correction to matter at all?** Computed each
run's mean covariance (Riemannian) and its distance from the whole-session
mean and from adjacent runs, for a few subjects. Run-to-run distances are
substantial — 0.55 to 2.8, comparable to or larger than each run's distance
from the whole-session average (0.4-1.8) — so within-session covariance is
*not* flat or stable. But the pattern doesn't look like a smooth, trackable
drift the way atmospheric turbulence is (which is what makes continuous
correction pay off in adaptive optics) — it looks more like noisy
fluctuation, jumping around between consecutive runs rather than moving
steadily in one direction (e.g. one subject's run0→run1 distance was 2.8,
then run1→run2 dropped back to 1.1, no consistent trend across the 3
subjects checked).

**Verdict:** genuinely untestable in its true form here — no honest
shortcut around needing new data with actual interleaved probes. The
diagnostic tempers the idea's core premise, though: adaptive optics pays off
because atmospheric distortion is smooth and predictable enough for
continuous tracking to beat a single snapshot. If EEG's within-session
covariance change is closer to noise than trackable drift (this quick check
suggests it might be, though only 3 subjects were examined — not
conclusive), a "guide star" mechanism would have less to correct for than
the analogy implies, at least at run-level (~2-3 minute) granularity. This
doesn't rule out slower, smoother *cross-session* (day-to-day) drift being
more trackable if the right data existed — that's a separate, still-open
question this check doesn't answer.

### 10.4 Bayesian-optimal calibration protocols, not fixed trial counts

**Source field: psychophysics (QUEST/QUEST+, Watson & Pelli 1983 onward).**
Vision and hearing science stopped using fixed-length threshold tests
decades ago in favor of adaptive procedures that pick each next stimulus to
maximize expected information gain about the subject's unknown threshold,
converging in far fewer trials than a fixed protocol.
[QUEST+](https://jov.arvojournals.org/article.aspx?articleid=2611972)

**Checked — partially precedented, be honest about this.** Generic active
learning (query-by-committee, uncertainty sampling) for reducing MI-BCI
calibration trial *count* is already published (IEEE 2018 and others) — so
"use active learning to need fewer trials" isn't new. What's narrower and
still looks open: QUEST-style approaches don't just pick *which already-
collected trial to label next*, they adaptively choose the next *stimulus/
task parameters themselves* (intensity, in vision) to most efficiently pin
down a parametric model of the subject. The MI-BCI analogue — adaptively
choosing which *class* to probe next (or task-difficulty/cue-vividness
parameters) based on a live Bayesian model of "how separable are this
subject's classes so far," rather than a fixed 288-trial block executed in a
fixed order — is a narrower, more specific gap than "active learning for
calibration" as a category.

**Why a big institution probably wouldn't chase this specifically:** it's
an unglamorous protocol-engineering change, not a new model, and the
literature that would inform it (psychophysics) isn't in most BCI-ML
researchers' citation graph. Moderate cost: needs a live/simulated online
protocol to test properly, though an offline "what if we'd stopped after N
adaptively-chosen trials" simulation on IV-2a is doable now.

#### Prototyped — real but narrow effect, and the headline metric doesn't survive scrutiny

Built the offline pool-based simulation: uncertainty sampling (reveal
whichever remaining calibration trial the current partial CSP+LDA model is
least certain about, retrain, repeat) vs. random order vs. the trials'
original as-collected order, tracking cross-session accuracy as a function
of trials revealed, on all 9 IV-2a subjects.
[experiments/idea10_4_active_calibration.py](../experiments/idea10_4_active_calibration.py) /
[results/idea10_4_active_calibration_iv2a.csv](../results/idea10_4_active_calibration_iv2a.csv)

- **A real, statistically significant effect exists, but only in a specific
  window.** At n=160-192 revealed trials (roughly 55-65% of the way through
  a full 288-trial calibration), active sampling beats random order by a
  consistent +1.6 to +2.6pp, paired Wilcoxon p=0.01-0.04 at three
  consecutive checkpoints, 8/9 subjects agreeing each time. Outside that
  window, differences are not statistically significant either direction.
- **At small n (64-128 trials) active trends *worse* than random**
  (non-significant, p>0.4), the opposite of what the "fewer trials needed"
  pitch would predict. Plausible cause: uncertainty sampling early on, when
  the partial model is still bad, can pick class-imbalanced batches that
  destabilize CSP's per-class covariance estimates — a known failure mode of
  naive uncertainty sampling for CSP-based classifiers specifically (CSP
  needs reasonably balanced per-class data to estimate stable spatial
  filters; committee-based or class-balance-aware active learning variants
  in the literature exist partly to address this and weren't tested here).
- **The practically appealing summary — "fewer trials needed to reach 95% of
  full accuracy" — does not hold up.** Active needed 112 trials on average
  vs. random's 144, which sounds like a real ~22% reduction, but the paired
  per-subject test is not significant (p=0.38, only 5/9 subjects favor
  active) — driven by high subject-to-subject variance (random ranged from
  64 to 272 trials-to-target across subjects). This is the headline number a
  less careful writeup would have led with, and it doesn't survive a proper
  test.

**Verdict:** genuine signal, not nothing, but narrower and less clean than
hoped — a real mid-calibration advantage that doesn't translate into a
robust "fewer trials overall" claim at this subject count (9), and a
concrete early-calibration failure mode worth fixing (class-balance-aware
selection) before this is worth pursuing further.

### 10.5 Closure-phase-style invariants for label-free self-calibration — the long shot

**Source field: radio interferometry.** Arrays of many imperfect antennas
solve calibration *without any external reference at all* using "closure
phase" — a specific combination of phases across antenna triplets that
mathematically cancels out each antenna's own unknown, individual gain/phase
error, leaving only the true signal. This is what let interferometry image
astronomical sources through unpredictable, antenna-specific atmospheric and
instrumental corruption before any calibrator source was available.
[Closure phase, Wikipedia overview](https://en.wikipedia.org/wiki/Closure_phase)

**Checked — no EEG analogue found, and I'm not aware of one existing.** This
is the most speculative entry here: it's an open *mathematical* question,
not a ready-to-build feature. The question worth asking: is there a
combination of EEG channels/electrode-pairs (or trials, or CSP components)
whose combined statistic is invariant to per-channel or per-subject unknown
"gain" factors (impedance, electrode placement, skull conductivity) the way
closure phase is invariant to per-antenna phase error? If such an invariant
existed, it would give genuinely label-free, calibration-free per-subject
correction — not "needs less calibration data" but "provably doesn't need
an external reference at all" for that specific corruption. Worth a
literature-informed theory pass (start from what "element-based error" means
for interferometry vs. what the analogous EEG corruption model would need to
look like) before writing any code — this one could easily turn out not to
have a clean EEG analogue, unlike 10.1-10.2 which are directly transplantable
as-is.

**Why a big institution probably wouldn't chase this:** genuinely high risk
of a dead end (the math might just not carry over), no guaranteed
publishable result even if it works, and it requires fluency in a field
(interferometric calibration theory) with essentially zero overlap with
typical BCI-lab training — exactly the kind of bet a grant committee
optimizing for safe, incremental progress would decline to fund, and exactly
the kind of bet that's cheap for one person to spend a week thinking about
before committing real time.

#### Resolved — a real EEG analogue exists, but for a narrower problem than this project needs

Worked the math and verified it numerically on real IV-2a data rather than
leaving it as speculation.

**The invariant is real.** If channel `i`'s recorded signal is an unknown
real scalar gain `g_i` times its true signal (impedance mismatch, amplifier
drift — a genuine per-channel EEG nuisance), covariance under linear
rescaling gives *exactly* `Cov(g_i x_i, g_j x_j) = g_i g_j Cov(x_i, x_j)` —
the same multiplicative-per-node structure closure phase exploits. So for
four distinct channels, `ratio = (Σ_ij · Σ_kl) / (Σ_ik · Σ_jl)` is exactly
invariant to `g_i, g_j, g_k, g_l`, since each gain appears once in the
numerator and once in the denominator. Verified on real subject-1 IV-2a
covariances: the ratio matches to ~1e-15 (floating-point precision) under
injected random per-channel gains 0.2x-3x, while the underlying covariance
entries themselves visibly change (e.g. one entry roughly doubled).
[experiments/idea10_5_closure_amplitude_check.py](../experiments/idea10_5_closure_amplitude_check.py)

**But it does not survive the corruption that actually causes cross-session
drift.** Cross-session/cross-subject EEG differences aren't well-modeled as
a diagonal per-channel scalar gain — they're closer to a full mixing-matrix
difference (volume conduction, electrode placement, source geometry), which
doesn't have the "each unknown factor appears once with each sign around a
loop" structure the cancellation depends on. Verified this too: under an
injected *full-matrix* (non-diagonal) distortion, the same ratio moves by
36-155% (even changes sign in one case) instead of staying invariant — the
exact opposite of the scalar-gain result, using the same script.

**Verdict:** this is real math, not a dead end, but it solves a different
(narrower, real) problem than the one motivating this project — automatic
detection/correction of bad or attenuated electrode channels, a data-quality
utility, not a cross-session-drift or illiteracy fix. It would *not* be a
substitute for Riemannian alignment or the shrinkage work in §10.1. Not
pursuing further under this project's goal, but noting it as a
legitimately-verified, separate, smaller idea (automatic per-channel
gain/impedance calibration with no reference recording needed) if that ever
becomes independently useful.

---

## 11. Joint-embedding / JEPA-style self-supervised foundation models — already tried, and they currently lose to Riemannian geometry on motor imagery specifically

Not a transplant from another field this time — checking whether the
mainstream 2024-2026 deep-learning direction (self-supervised joint-embedding
pretraining, à la LeCun's JEPA) has already been applied here, and how it
actually performs, before treating it as an unexamined "obviously better"
option.

**Yes, multiple papers have done this:**

- **S-JEPA** (2024) — masks spatial blocks of EEG channels and predicts the
  *embedding* of the masked region (not the raw signal) using an EMA target
  encoder, evaluated on OpenBMI (54 subjects) across MI, ERP, and SSVEP.
  [arXiv:2403.11772](https://arxiv.org/html/2403.11772)
- **Laya / LeJEPA** (2026) — same latent-prediction-over-reconstruction idea,
  compared directly against four other reconstruction-based EEG foundation
  models (LaBraM, LUNA, CBraMod, REVE) on left-vs-right-hand MI.
  [arXiv:2603.16281](https://arxiv.org/html/2603.16281v2)
- Also: EEG-VJEPA (video-JEPA adapted to EEG), LaBraM, LUNA, CBraMod, REVE —
  a genuinely active, crowded 2024-2026 foundation-model space, not a gap.

**The result is decisive and consistent across both papers checked in
detail: motor imagery is specifically where these models struggle.**

- S-JEPA's own numbers: 97% (ERP, ties Riemannian SOTA), 94% (SSVEP,
  *beats* Riemannian SOTA's 89.4%), but only **65% on MI vs. Riemannian
  SOTA's 84.7%** — a 20-point gap, on the model's own home dataset. (Caveat
  the authors themselves flag: their MI evaluation used only the 7
  hardest-to-classify subjects vs. SOTA's full 54 — the gap is likely
  somewhat inflated by that, but 20 points is too large to be explained away
  entirely by it.) Also worth flagging: despite the paper's title emphasizing
  "seamless cross-dataset transfer," the actual evaluation never leaves a
  single dataset (Lee2019/OpenBMI, split by subject) — the title overclaims
  relative to what was tested, worth knowing before citing it as cross-
  dataset evidence.
- Laya's comparison table is the more damning number: on left-vs-right-hand
  MI under frozen linear probing, **LaBraM, LUNA, CBraMod, and Laya itself
  all score 0.47-0.51 balanced accuracy — at or barely above chance (0.50)
  for a 2-class task.** Every current EEG foundation model checked here is
  essentially non-functional for MI specifically under this evaluation
  protocol, despite being legitimate, well-cited architectures.

**Why MI specifically, when the same architectures do fine or better on
SSVEP/ERP (my synthesis, not stated this way in either paper):** SSVEP has a
strong, stereotyped periodic driven response and ERP has a strong
time-locked component — both are the kind of high-SNR, consistent structure
a generic masked/latent-prediction objective can pick up on without much
task-specific guidance. Motor imagery's signal (event-related desync/
resync in sensorimotor rhythms) is weaker, more subject-idiosyncratic, and
easily dominated by other structure (alpha rhythm, artifacts) that a
generic self-supervised objective has no reason to ignore. Riemannian/CSP
methods encode a strong domain-specific prior (spatial covariance structure
tied to ERD/ERS) that generic SSL pretraining would have to learn from
data — and current EEG pretraining corpora (LaBraM: ~2500 hours across ~20
datasets) are tiny next to vision/language foundation-model scale, with MI
likely a small fraction of that mix.

**Answer to "would this bring leverage against SOTA": not currently, for
motor imagery specifically — the opposite. On the one paradigm this whole
project targets, joint-embedding foundation models are reported *behind*
Riemannian geometry by a wide margin, in some evaluations barely better than
chance. This isn't a niche finding — it recurs across independent papers and
model families (S-JEPA, LaBraM, LUNA, CBraMod, Laya), which makes it a
real, checkable line in the sand rather than one bad result. Building on top
of a joint-embedding foundation model would very likely regress from where
Phase 1/2's classical Riemannian pipeline already stands (62-67% cross-
session on 4-class IV-2a), not improve on it — the literature says the
opposite direction (Riemannian geometry, which is what §10.1's shrinkage
work already builds on) is where the actual motor-imagery leverage is,
at least as of the papers checked here. Worth re-checking periodically since
this is a fast-moving space and today's gap could close.
