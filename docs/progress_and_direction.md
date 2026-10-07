# Progress and Direction: What We've Done, What Worked, What Didn't

**Purpose of this file:** consult this *before* starting new work. It exists so
we don't re-litigate settled questions, re-attempt approaches already shown
not to work, or lose track of *why* the project is shaped the way it is.
`docs/state_of_the_art.md` is the literature survey; this is the project's
own execution log and decision history. Update it whenever a phase produces
a real result (positive or negative) - a stale version of this file is worse
than none, because it would misdirect the next round of work.

## Executive summary (read this first)

- **The refined core finding of this project, now confirmed across six
  attempts:** the failure mode is never really "population data" per se -
  it's specifically requiring a computation (a correction, or a training
  gradient) to directly *compare or align two different subjects'*
  covariance structure. Every method that does this (shrinkage, eigenvoice,
  Module A's learned correction, and - a new twist - even just *training*
  an encoder on episodes that pool different subjects together) is null-
  to-harmful, up to and including breaking training convergence itself,
  not just deployment transfer. Every method that avoids it, even while
  still using population data extensively, works: Riemannian alignment
  (per-session only), wavelet-calibration (per-session only), and now
  prototypical-network meta-learning (population data trains a general
  *skill* via same-subject-only episodes, deployment always grounds in the
  target's own data). See "The recurring negative result" and
  "Prototypical-network meta-learning" below for the full evidence trail.
- **Current best classifier for this task, full stop: CSP+LDA** (62.2%
  cross-session, non-adapted) improved by Riemannian alignment (66.9%).
- **Second-best, and the most promising active direction: prototypical-
  network meta-learning, 49.0% mean, built and fixed this session.** First
  attempt (following the published MetaWearS design exactly - disjoint
  support/query subjects) stalled completely on real data. Fixed by
  redesigning episodes to be same-subject-only (support and query from one
  subject's own trials, matching the real calibration -> eval task
  exactly) - a user-driven insight, not something in the source paper.
  Four of nine subjects land in the 59-67% range, directly competitive
  with CSP+LDA/Riemannian alignment. See "Prototypical-network meta-
  learning" below - this is now the highest-leverage place to keep
  pushing (understand the subject-5 chance-level outlier, tune
  episodes_per_step, consider combining with CSP now that the standalone
  mechanism is finally strong enough for that to be worth trying again).
- **Wavelet-calibration remains a live, slower-improving alternative**
  (31.4%, up from 25.0%), still per-session-only with zero population
  data - a different, complementary bet, not superseded by the
  prototypical-network result. Its "richer features" iteration (Phase 4c)
  was a validated synthetic-data win that turned out to be a real-data
  wash (sample-efficiency cost) - the lesson there was to watch the
  samples-per-feature ratio, not to stop pushing on the mechanism.
- **Naive feature-level combination (wavelet-calibration + CSP + MiniRocket,
  Phase 5) failed** (hurt CSP's own accuracy, statistically significant).
  Worth retrying now that prototypical-network gives a much stronger
  standalone signal to combine, ideally via proper out-of-fold stacking
  rather than the naive concatenation that failed before.

## External literature check

Log external papers checked for relevance here, with a concrete verdict -
not just "interesting," but *what specifically transfers to this project
and why*, so a future skim doesn't have to re-derive the connection.

### MetaWearS (Amirshahi, Toosi et al., Communications Medicine 2026)

Prototypical-Networks-based few-shot meta-learning for wearable health
monitoring (epileptic seizure detection from EEG, atrial fibrillation
detection from ECG). Encoder meta-trained via episodic training on a large
*base* dataset (e.g. TUSZ EEG); at deployment on a genuinely different
*target* dataset/site/device (e.g. Siena EEG), classification is done by
computing per-class prototypes (mean embeddings) from a handful of the
target patient's own labeled examples and classifying by nearest
prototype - no fine-tuning of the encoder itself. Updating later means
recomputing prototypes from new shots only (cheap, no backprop); the
paper's headline 418-456x energy savings is specific to their wearable-
hardware deployment constraint and isn't relevant to us.

**Why it's relevant here:** it's a third point in the "how should
population data inform a new subject's correction" design space, distinct
from both things we've tried. Shrinkage/Module A used population data to
produce a *fixed correction*, applied zero-shot - this is exactly what
failed, because it assumes the target resembles the base population's
shape. Wavelet-calibration uses *no* population data at all. MetaWearS's
middle path - population data teaches the encoder a good discriminative
*metric space* (a transferable skill, "what makes examples of the same
class look similar"), while the actual decision boundary (the prototype)
is always freshly computed from the target subject's own data - never
requires the target to resemble the base population's shape at all, only
that the *learned metric* generalizes. That's a structurally different
claim than anything we've tested, and it directly targets Module A's
diagnosed failure mode rather than repeating it.

**Direct mapping to this project:** PhysioNet (108 subjects, downloaded,
otherwise unused since Module A was deprioritized) as the base dataset;
IV-2a as the target. Episodic meta-training on PhysioNet subjects (sample
k support + m query trials per class per episode, extract features,
compute prototypes from support, classify query by nearest prototype,
backprop into the encoder); at IV-2a deployment, compute prototypes from
each subject's own calibration trials, classify eval trials by nearest
prototype, no fine-tuning.

**Honest caveats, before trusting this uncritically:**
- The paper's own results show degraded performance from single-shot
  updates ("low robustness of randomly selecting a single shot") - this
  method still needs a handful of labeled examples, not zero, similar to
  our existing calibration-session-dependent methods.
- The paper's own domain-shift discussion notes larger base-target
  discrepancy correlates with worse relative performance - not immune to
  domain shift, just more robust to it *structurally* than a fixed
  correction. Our own measured PhysioNet-to-IV-2a domain gap (Module A's
  diagnostic: population grand-mean shape as far from a target subject as
  two random target subjects are from each other) looked more severe than
  this paper's within-modality shifts (different EEG hospital vs.
  different EEG hospital, not different task/species/instrument) - don't
  assume the gap here is small enough for this to work without checking.
- **Built, diagnosed, fixed, and now the second-best method in the whole
  project (Phase 6, 49.0% mean).** The first design (disjoint support/query
  subjects, matching the published MetaWearS algorithm) stalled completely
  on real data. The user's own reframing - train and test are always
  *same-subject* (calibration -> eval), never cross-subject, so episodes
  should match that - fixed it decisively. See "Prototypical-network
  meta-learning" for the full story.

### Optical flow on EEG topography - checked after Phase 8b/9 were already built and validated

Retroactive literature check (web search, not a formal review) after building
`flow_field.py` and running Phase 8b/9 - done to place the result honestly
relative to prior art, not to gate the build (which was already synthetic-
validated before this check happened).

**Direct precedent for the core mechanism exists, predates this project:**
- Bashivan et al. 2016 (ICLR, arXiv:1511.06448) - azimuthal equidistant
  projection of electrodes into a topology-preserving 2D image sequence
  ("EEG movies"), fed to a CNN+LSTM. This is the same core idea as
  `electrode_positions_2d`/`interpolate_topomap` here (different
  projection formula, same purpose) - independently arrived at, not
  copied, but not novel as a projection technique either.
- Tan, Sun, Zhang, Chen, Liu (ICONIP 2017, arXiv:1807.10641; a closely
  related ICASSP 2018 paper, arXiv:1808.01752, applies the same idea in a
  BCI/stroke-rehabilitation context) - extends Bashivan's image sequence
  with an explicit "EEG optical flow" step (frame-to-frame flow of the
  topographic images, HSV-encoded direction/magnitude), fed into a CNN+RNN
  alongside the raw image stream. **This is the closest documented use of
  literal optical flow as an EEG feature for BCI-style classification
  found** - the core idea (topography sequence -> optical flow -> features)
  is not novel to this project. Neither paper's abstract/summary specifies
  exact dataset details or head-to-head accuracy against a non-flow
  baseline in what could be retrieved (PDF text extraction failed in this
  environment - poppler/pdftoppm unavailable - so this is based on
  abstracts and secondary summaries, not the full paper; treat the
  methodological precedent as confirmed, the missing numbers as
  genuinely missing, not "checked and found absent").

**Adjacent precedent, different purpose (characterizing dynamics, not classifying):**
- Lefèvre & Baillet (2007 MICCAI chapter; 2008 journal version) - optical
  flow rigorously extended to curved cortical/scalp manifolds (2-Riemannian
  manifolds), used to image propagating MEG/EEG source-activation streams.
  Analytical/descriptive, not a BCI classifier.
- Volpert, Xu, Tchechmedjiev, Harispe, Aksenov, Mesnildrey, Beuter (Math
  Biosci Eng, 2023) - explicitly applies Horn-Schunck (among other optical
  flow variants) to EEG topography, characterizing sources/sinks/spirals/
  saddles during a picture-naming (language) task. Same primitive
  (Horn-Schunck on EEG topography) as `flow_field.py`, different task
  (language, not motor imagery) and different goal (describing dynamics,
  not extracting classifier features).

**Related but not flow-based, motor imagery specifically:**
- Fukushima et al. (ACCV 2024, arXiv:2403.04353) - topological-map *image
  sequences* + spatiotemporal pooling (a PoolFormer/InternImage-style
  learned pooling, not literal velocity-field computation) for motor
  imagery classification on PhysioNet, reporting 88.57%/80.65%/70.17% for
  2/3/4-class. Shows topography-*sequence* representations (the same
  temporal-spatial information flow-field features are trying to capture)
  can be competitive with a much heavier model - a different, deep-
  learning-first way of using the same underlying information this
  project's 6-scalar engineered features tried to extract cheaply.

**What's actually novel here, as far as this search found:** not the
mechanism (optical flow on EEG topography is documented, in both the
BCI-classification and dynamics-characterization literature) but the
*application* - none of the found work evaluates this on the cross-
session calibration-transfer question this project is actually about.
Every result found is within-session classification (Bashivan, Tan,
Fukushima) or descriptive characterization (Lefèvre/Baillet, Volpert et
al.), not "does a flow-derived feature or embedding generalize from one
day's calibration session to another." That gap is either a real
opportunity or a sign the idea doesn't survive contact with the harder
problem - Phase 9's own real-data result (weak standalone signal, hurts
combined with CSP) is more consistent with the latter, though it's a
different specific instantiation (engineered scalar summaries + LDA, not
a CNN+RNN) than anything found in the literature, so it doesn't settle
the question for the deep-learning version of the idea.

### Correction: "flow field" meant something different - a second, unrelated literature check

After the above was written, the user clarified (pointing to a video on
motor-cortex population dynamics) that "flow field" meant something
entirely different from what `flow_field.py` implements: not optical flow
of a *spatial* scalp topography, but the vector field of a *dynamical
system* over an abstract **neural state space**, where each axis is one
neuron's/channel's activity and the arrow at each point is the population's
instantaneous rate of change (ẋ = f(x)). This is the Churchland/Shenoy/
Sussillo "neural population dynamics" framework (jPCA, rotational
dynamics; Kaufman/Churchland/Ryu/Shenoy 2014's null-space/potent-space
account of movement preparation - a fixed point held in a subspace the
muscles structurally cannot read, released by a "go" cue). **Everything in
`flow_field.py` and the Phase 8b/9 result above answers a different
question and should not be cited as evidence for or against this
different idea.**

That foundational work is built almost entirely on invasive multi-
electrode-array recordings of single neurons in monkey motor cortex, not
EEG. Checked what exists for EEG specifically:

- **Zhang et al. 2016, "Low-Rank Linear Dynamical Systems for Motor
  Imagery EEG," *Computational Intelligence and Neuroscience*** - the one
  genuinely close EEG precedent found. Fits a linear state-space model
  directly to raw EEG channels (x_{t+1}=Ax_t+Bv_t, y_t=Cx_t+ω, i.e.
  literally the video's ẋ=f(x) picture in discrete time, fit per trial),
  classifies via Martin distance (a Riemannian-geometry metric on
  principal angles between the fitted systems' observability subspaces) -
  no CSP anywhere in the pipeline. **Evaluated on BCI Competition IV-2a,
  this project's own dataset**: 80.32% mean vs. 77.24% for CSP (9
  subjects, 25 channels). Also tested on BCI III dataset IVa: 82.44% vs.
  77.92% CSP. **Important gap, same as everything else checked**: this
  looks like within-session cross-validation, not cross-session
  (different-day) evaluation - the paper does not test whether a fitted
  system stays stable across sessions, which is the actual question this
  project cares about. Not directly comparable to this project's 62-67%
  cross-session numbers for that reason.
- **Wu, Shi & Lu (ICONIP 2011), "Removing Unrelated Features Based on
  Linear Dynamical System for Motor-Imagery-Based BCI"** - read in full
  (user provided the paper directly) after an earlier draft of this note
  wrongly guessed it was "likely a precursor to the Zhang paper, possibly
  by the same authors" - **that guess was wrong and is corrected here,
  not left standing**. This paper's "LDS" is a much simpler, different
  technique: standard CSP first (8-20Hz, six 2Hz sub-bands), producing a
  scalar variance feature every 0.2s across a trial; the "LDS" step is a
  **Kalman smoother** applied to that already-scalar feature sequence
  (state transition matrix A fixed at 1, not fit - a random-walk
  denoising prior), not a state-space fit to the raw multichannel signal
  and nothing with rotational/flow structure. Dataset: BCI Competition
  **III-IIIa** (3 subjects, not IV-2a). Result: a small, consistent ~2pp
  average gain from smoothing (e.g. 81.1%→83.3%), within-session CV only,
  no cross-session test. Shares the name "LDS" with Zhang et al. 2016 but
  is not evidence about the same claim - don't conflate the two.
- One more nuance from the broader (non-EEG) rotational-dynamics
  literature: there's active, unresolved debate about whether "rotational
  flow field" findings in the original invasive recordings reflect a
  genuine abstract dynamical system or are better explained by simple
  **traveling waves** across the electrode array (a 2024 Scientific
  Reports paper argues the latter). If that holds, it's a real bridge back
  to what `flow_field.py` *actually* measures (spatial propagation) -
  worth remembering if this direction gets built, since the two "flow
  field" concepts may not be as unrelated as their different math suggests.

**Net finding, same shape as the optical-flow check above**: the mechanism
(fitting a linear dynamical system to EEG channels) is documented and
beats CSP on this project's own dataset - but only within-session. Whether
a fitted system is more *session-stable* than CSP/covariance features for
one subject across days - this project's actual question - appears
untested anywhere found.

### State-space flow field (jPCA-style): Stage 0 built, validated, one real bug found and fixed, one real limitation found and documented

`src/adaptation/state_space_flow.py` - fits a linear dynamical system to a
(dimensionality-reduced) state trajectory: xdot ~= Mx via least squares,
M constrained skew-symmetric (jPCA's own constraint - guarantees purely
imaginary eigenvalues, i.e. pure rotation with no growth/decay). Distinct
from `flow_field.py`: this fits a flow field in abstract *state space*
(each axis a channel/PCA component), not physical scalp positions - see
the correction above for why these are different mechanisms despite both
being called "flow field."

**`stage0_state_space_flow_validation.py`** (four checks, same
plant-a-known-answer-and-recover-it discipline as every other Stage 0 gate
here):

1. **Plane recovery** (plain least-squares fit): a planted 2D rotation,
   embedded into 22 channels (matching IV-2a) plus harsh -3dB SNR noise,
   is recovered essentially exactly (subspace cosine 0.98-1.00) across two
   different planted frequencies and both an exact-rank fit (d_fit=2) and
   a deliberately over-complete one (d_fit=6, since real EEG's true
   dynamical rank is never known in advance). **Passed, robustly.**
2. **Frequency recovery, found broken then fixed**: the plain fit's
   recovered *frequency* was systematically ~17% low at -3dB SNR - not
   noisy scatter, a consistent shortfall that got worse with more noise
   and vanished as SNR rose (checked across SNR=-3 to 40dB, monotonic).
   Diagnosed as errors-in-variables attenuation bias (the noisy observed
   state is used as its own regressor - a textbook effect, not a coding
   mistake) and fixed with a known-noise-variance correction
   (`fit_skew_symmetric_flow_debiased`), verified to recover frequency
   within ~2-3% at the same harsh SNR - confirming the diagnosis was
   complete (fixing exactly this mechanism fixed the number). **Passed,
   at the true rank, after a real fix - not a loosened tolerance.**
3. **A second, real limitation found while building the fix for #2**:
   applying that same debiasing correction to the over-complete fit
   (d_fit=6) is unsafe - it made both plane recovery (cosine 0.98 -> 0.64)
   and frequency recovery (ratio 0.83 -> 0.68) *worse* than the plain fit,
   diagnosed as the correction going ill-conditioned on the "extra" fitted
   dimensions once they carry almost no true signal above the noise floor.
   **Not claimed to work - documented and printed on every run, not
   gated as passing.** Real use of debiasing needs either a good rank
   estimate or a more robust correction (regularized, or a full EM-based
   joint fit of dynamics and noise covariance, matching Zhang et al.
   2016's actual approach) - not built here.
4. **Spurious-rotation-in-noise guard**, directly testing a documented
   critique from the literature review (jPCA can find "rotational
   dynamics" in data that has none): the identical plain-fit procedure on
   pure noise explained ~0% of variance (R^2=0.000) vs. 16-47% for the
   true-signal conditions in check 1 - a clean, wide separation. **Passed.**

### Synthetic-EEG session-drift test (`stage0_state_space_flow_eeg_validation.py`): frequency is reliably drift-robust, plane/covariance features are not

The specific question this project cares about - is a flow-field-derived
feature more *session-stable* than CSP/covariance features for one subject
across days - untested anywhere in the literature review, tested here for
the first time. Two-class synthetic data (reusing the two frequencies
already validated above, 0.15 vs. 0.35 rad/step): each class has its own
true rotational frequency *and* its own random 22-channel embedding plane,
40 calibration + 40 eval trials per class, T=500 samples/trial (matches
real IV-2a's 0.5-2.5s @ 250Hz window). Three classifiers compared on
identical data under an identical planted drift:
- **frequency-only**: per-trial *local* PCA + debiased flow fit (uses only
  that one trial's own data, no calibration-fixed projection at all), then
  nearest-class by frequency to each class's calibration-derived template.
- **plane-residual**: an eval trial projected through *calibration's*
  per-class PCA, scored against that class's calibration-fit flow model.
- **tangent-space LDA** (`riemannian_icp.py`'s `trial_tangent_vectors` +
  LDA): this project's own established no-realignment baseline.

The theoretical motivation: a linear channel-mixing drift changes the
*embedding* through which a low-dimensional rotational state is observed,
but leaves the underlying dynamics' *eigenvalues* - hence the recovered
frequency - mathematically unchanged, as long as the true signal subspace
is re-identified from the drifted trial's own data. Plane- and covariance-
based features have no such guarantee, since both depend on a trial's
representation still lining up with a calibration-fixed observation-space
template, which drift directly disturbs.

**Two real bugs caught before trusting any of this, not before the numbers looked good:**
- A first pass compared classes by raw MSE residual against each
  candidate's flow model. Wrong: class_b rotates faster, so its natural
  derivative magnitude - and thus its residual "floor," even for a
  *perfect* fit - is inherently larger than class_a's, which silently
  biased the argmin toward class_a regardless of the true label (caught by
  inspecting the actual residual values per true class, not just the
  accuracy number). Fixed by normalizing each residual to a (1 - R^2) style
  score relative to that trial's own variance under that specific
  projection, removing the scale confound.
- A first single-drift-instance run showed both flow-field classifiers at
  a perfect 100% -> 100%, which looked almost too clean. Repeating with
  more independent random drift matrices (calibration/eval data held
  fixed) showed that result was a lucky draw for plane-residual - across 8
  independent drifts it ranged from exact chance (50.0%) to 100%, no more
  reliable than tangent-LDA. Only frequency-only stayed at 100% on every
  single drift tested. Trusting a single random instance would have
  overclaimed the entire finding.
- Also corrected in passing: DRIFT_ANGLE_SCALE=0.35 was first described
  here as "modest," copied uncritically from `stage0_icp_eeg_validation.py`'s
  use of the same parameter value. Checked directly (subspace cosine
  between the clean and drifted signal plane): as low as 0.07-0.34 - a
  near-total reorientation, not a small perturbation, in this specific
  22-channel setting. The description was fixed to match the measured
  effect rather than assumed from a different script's use of the same
  number.

**Result across 8 independent drift matrices** (clean baseline: all three
100%):

| Classifier | Mean accuracy under drift | Std across 8 drifts |
|---|---|---|
| frequency-only | **100.0%** | **0.000** |
| plane-residual | 71.7% | 0.232 |
| tangent-space LDA (baseline) | 47.2% | 0.351 (one instance at 0.0% - confidently wrong, not just noisy) |

**Frequency-only classification was perfectly, reliably robust to every
single planted drift tested - the only one of the three that was.**
Plane-residual and tangent-LDA are both inconsistent: sometimes barely
affected, sometimes at or below chance, depending on the specific random
drift instance - i.e. neither is a *reliable* property, unlike
frequency-only's clean 0.000 std. This is the strongest, most cleanly
theoretically-grounded positive finding of this whole "flow field"
investigation (not the scalp-optical-flow one - the state-space dynamical
one) - but it is still a synthetic, purpose-built, orthogonal-channel-
mixing drift model, not real IV-2a data, and not yet a full 4-class or
multi-subject test. **Not yet done**: real IV-2a data (does frequency
recovered per-trial actually separate IV-2a's 4 motor-imagery classes at
all - the earlier Stage 0 gates never established that real EEG trials
even *have* cleanly-recoverable rotational structure the way this
synthetic generator does by construction), and a non-orthogonal drift
model (real session drift - electrode impedance changes, gain differences
- may not be a pure rotation the way `expm(skew)` guarantees; the
frequency-invariance argument specifically relies on the mixing preserving
isotropic noise and not degenerating the true signal subspace's rank,
which holds for any orthogonal transform but is not guaranteed for a
general one).

### Real IV-2a data (`phase10_state_space_flow_frequency.py`): clean null - real trials don't show the rotational structure the synthetic generator guaranteed

The honest first look flagged as missing above: per-trial local PCA +
plain flow fit (debiasing skipped here - it needs a known noise variance,
which real data doesn't hand over the way the synthetic validation's
controlled generator did), nearest-template-frequency classification, 4
classes, all 9 subjects, same no-cross-subject-data calibration->eval
protocol as every other phase.

**Result: mean accuracy 25.8% (std 1.7%), chance is 25.0% - indistinguishable
from chance, no subject individually significant** (best p=0.15, one
subject actually below chance at p=0.93). The per-class template
frequencies tell the real story: within every single subject, all four
classes' template omegas cluster tightly together (e.g. subject 1:
left_hand 0.068, right_hand 0.078, feet 0.068, tongue 0.074 - a spread of
0.01 rad/step against noise-driven per-trial variation far larger than
that). Real IV-2a motor-imagery trials simply do not carry a class-
differentiated rotational frequency the way this synthetic generator
guaranteed by construction - a real, honest limitation flagged in this
section from the start (the foundational jPCA/rotational-dynamics
literature comes from *invasive*, single-neuron recordings during *actual
reaching movements*, not scalp EEG during *imagined* movement; there was
never a guarantee that finding would transfer to this modality and task).

**Bottom line on the whole state-space flow-field idea**: the mechanism
itself is real, validated, and produced this project's most theoretically
clean positive finding (frequency's reliable, near-perfect drift
invariance under a severe planted channel-mixing, confirmed across 8
independent drift instances after two genuine bugs were caught and fixed -
a raw-MSE scale-confound and a misleading single-seed test). But real
IV-2a trials don't give it anything to work with: there is no detectable
class-discriminative rotational frequency to *be* robust about. Closing
this out the same way as Phase 5, Phase 7's ICP component, and Phase 9's
combination test: a real, validated mechanism, not a deployable one here -
the negative result is trustworthy specifically *because* the synthetic
validation was this thorough, not despite it.

## Latency-jitter deconvolution (Woody's filter / RIDE family): built, one bug fixed, one honest ceiling found

User's idea: frame cross-session drift as a deconvolution problem - recover
each trial's own onset/latency and a clean underlying template jointly (a
readiness-potential-style "template convolved with a per-trial start time"
model), regularized to separate genuine signal from cross-session noise,
then compare recovered structure across sessions; if promising, either an
algorithmic solution or an NN trained with progressively harder synthetic
jitter/noise to generalize zero-shot. Explicitly scoped by the user as
within-subject, cross-session only - never across subjects, consistent
with this project's central established constraint.

**Literature check** (done before building, not after): this maps onto a
real, well-established research area - Woody's adaptive filter (1967,
classic alternating template/latency estimation, the same alternating-
optimization structure as this project's own ICP), RIDE (Ouyang et al.
2011/2015, the modern standard - jointly separates latency-variable ERP
components and their per-trial latencies), and more recent methods
(ReSync, CWT-AWF) benchmarked against Woody's filter directly. None of
them are built for cross-session/cross-subject *calibration transfer*
specifically - same gap pattern as every other mechanism checked this
session. One important caveat raised before building anything: Schurger et
al.'s accumulator-model reanalysis of the *readiness potential specifically*
(PNAS 2012) argues its "start" may not be an independent event separate
from the template's own shape at all (the RP could be largely an artifact
of backward-averaging noisy threshold-crossing trajectories) - a reason to
target this project's existing ERD/ERS envelope signal, not assume the
classical RP model applies unmodified.

### Built: `src/adaptation/latency_alignment.py` - Woody's filter (cross-correlate, shift, re-average, iterate)

`stage0_latency_alignment_validation.py` validates it against known planted
per-trial jitter (reusing the exact ground-truth mechanism
`synthetic_eeg.py` already has for this - `onset_jitter_std_s` - built
earlier in this project specifically because Phase 4's wavelet-calibration
diagnostics suggested real trial-to-trial jitter was smearing its
trial-averaged fit).

**A real bug found and fixed, not just a tolerance loosened**: a first pass
failed a zero-true-jitter control (deliberately re-testing whether this
mechanism falls into the *exact* pitfall already found once in this
project - Phase 4's max-over-lags classifier scoring, a multiple-
comparisons/peak-picking bias that made accuracy worse even with no real
jitter). It did: alignment degraded the template's fidelity even with
nothing to correct (naive-vs-true-shape correlation 0.587, woody-aligned
only 0.398), with a suspiciously wide spread of "recovered" lags (std
230ms) despite no true jitter. Diagnosed properly, not assumed: raising
additive SNR from -3dB to 60dB (near-noiseless) left that spurious spread
essentially unchanged (~150ms throughout), ruling out background noise as
the cause. The actual culprit is the oscillatory carrier's own *intrinsic*
envelope randomness - a narrowband random process has a natural envelope
correlation time of roughly 1/bandwidth (~250ms here), independent of any
added noise - which Woody's filter mistook for genuine latency structure.
Fixed with `smooth_signal` (a 0.5s moving average, matched to the genuine
ERD/ERS onset-to-trough timescale, not the oscillation's own period),
verified to bring the zero-jitter control back in line (naive 0.785 vs.
woody 0.767).

### Honest ceiling found after the fix, via the multi-seed discipline this project's own drift-robustness test already established

A single run under real (nonzero) planted jitter looked promising (recovered-
lag-vs-true-jitter correlation 0.364) - but a 20-repeat sweep (same lesson
as the state-space flow-field drift test: never trust one random draw)
showed that was a lucky instance, not a reliable property: **mean lag
recovery correlation 0.063 (std 0.102, values ranging -0.114 to +0.244,
no consistent sign)** at this project's standard harsh -3dB SNR - the
mechanism does not reliably recover per-trial jitter at all once properly
tested across repeats. Template fidelity told the same story: aligned
(0.754 mean) vs. naive-unaligned (0.758 mean) - indistinguishable, woody
beat naive in only 5/20 repeats, worse than a coin flip.

Checked whether this is fixable with better SNR (matching this project's
practice of raising SNR before concluding a mechanism doesn't work at all):
at a very generous, unrealistic-for-single-trial-EEG SNR of 20dB, mean lag
recovery correlation improved to only 0.278 (std 0.130) - real and
consistent, but still modest, not strong. **This points at a genuine
structural ceiling, not just an SNR problem**: the same smoothing window
needed to suppress the carrier's intrinsic envelope randomness (~0.5s)
is comparable to or larger than the jitter magnitude being detected
(0.15s std) - the fix for the false-positive problem and the sensitivity
needed to detect real jitter are in direct tension for this specific
representation (single-channel Hilbert envelope + simple cross-
correlation), not simultaneously achievable by tuning the smoothing window
alone.

### Refinement (check 3): estimate on smoothed, apply to raw - a real, if modest, improvement

The user asked the natural next question directly: since smoothing was
only needed to make the *estimate* well-behaved, is it meaningful to
estimate the lag on the smoothed envelope but *apply* that lag to the
original, unsmoothed trial for whatever comes next, rather than smoothing
throughout? Tested, not just reasoned about - the answer is yes, but
modestly. Applying the smoothed-estimated lag to each trial's raw envelope
and averaging the *raw* shifted trials beats both doing nothing and
smoothing throughout: mean correlation with the true (unsmoothed) shape
improved from 0.531 (naive) to 0.548 (shifted), won in 37/50 independent
repeats, Wilcoxon p=1.0e-5 in the confirming run (p=5.97e-4 with the
script's own fixed seed) - a real, statistically robust effect, not noise.

Why this helps despite the underlying lag *estimate* itself remaining just
as weakly correlated with true jitter as in check 2 (~0.06-0.07,
unchanged): separating estimation from application doesn't create a
stronger signal than the weak one already found - it just stops throwing
away resolution unnecessarily. Smoothing before averaging discards fine
detail from *every* trial regardless of whether its own lag estimate was
any good; applying even a weakly-informative shift to the full-resolution
trial recovers a little more of whatever real signal that weak estimate
does carry, without also degrading everything else. A real, useful
refinement to remember - but it does not overturn check 2's core finding:
the lag *estimate* itself is still weak, so the improvement this refinement
buys is correspondingly modest (a ~3% relative gain), not transformative.

**Bottom line**: the user's specific proposed mechanism (regularized
alignment via best convolution overlap) was built faithfully, caught a
real bug (re-confirming a pitfall this project already knew about, in a
new context - useful in itself), and the regularization needed to fix that
bug (smoothing) caps the mechanism's ability to detect real jitter at a
realistic magnitude and SNR - a genuine ceiling. Separating "estimate on
smoothed" from "apply to raw" (the user's own follow-up) recovers a real
but modest improvement on top of that ceiling, not a reversal of it. This
closes out the *Woody-filter-specific* instantiation of the idea as
real-but-modest before ever reaching the cross-session comparison or
NN-generalization steps the user proposed next - building either on top
of a within-session foundation this weak would not be well-grounded. Not
yet tried: RIDE's own more sophisticated multi-component decomposition
(rather than Woody's single-template alternation), which the literature
suggests handles low-SNR cases specifically better than the classic Woody
filter alone.

## Refinements inside the Riemannian/tangent-space framework: CPD (real, unfixed dimensionality limit) and ComBat (validated null on synthetic, tied on real data)

After the flow-field, state-space-flow, and latency-alignment investigations
all closed out as genuine nulls (see their own sections above), the user
asked for research specifically on refinements *inside* Riemannian
alignment's own geometric framework rather than more imports competing
with it, since Riemannian alignment is precisely matched to the dominant,
well-characterized failure mode (session covariance drift) in a way
nothing imported so far was. Literature check surfaced (with proper
citations, see below): parallel transport (confirmed to already be what
recenter+rescale does), optimal transport/sliced-Wasserstein distribution
alignment (a real, published +2.45pp improvement on this exact dataset -
not yet built here), Coherent Point Drift (a documented, more-robust
successor to ICP specifically), and ComBat (genomics batch-effect
correction, structurally similar to session drift). User asked to build
CPD (replacing ICP) and ComBat (replacing recenter+rescale).

### CPD (src/adaptation/coherent_point_drift.py): validated at D=10, found to fundamentally fail at D=253 - not fixed

A from-scratch, dimension-general implementation of rigid Coherent Point
Drift (Myronenko & Song 2010) - `pycpd` exists but hard-restricts to 2D/3D
point clouds (an implementation choice for its computer-vision use cases,
not a mathematical limit of CPD itself), unusable for this project's
253-dimensional tangent vectors.

**Stage 0 at D=10 passed cleanly, including CPD's actual documented
advantage over ICP**: `stage0_cpd_validation.py`'s benign case matched ICP
(error 0.0035 vs 0.0023); its outlier-contamination case (25% of target
points with no true correspondence at all) showed CPD's soft, probabilistic
assignment genuinely beating ICP's hard nearest-neighbor assignment (error
0.0505 vs 0.0552) - exactly the failure mode ICP hit in Phase 7 ("noisy,
real trial clouds" breaking its hard correspondence).

**Then found to fail completely at D=253** (this project's actual
tangent-space dimensionality) - not caught by the D=10 tests, caught by
running on real data and getting a rotation stuck at exactly the identity
matrix. Three distinct problems diagnosed in sequence, each confirmed
before moving to the next (not papered over):
1. The Gaussian kernel's raw exponent underflows at D=253 (squared
   distances scale with D; "close" and "far" pairs end up differing by
   ~10^40 in raw likelihood) - fixed by normalizing the squared distance
   by D, consistent with sigma2 already being a per-dimension average.
2. That alone wasn't enough: the outlier term's own normalization constant,
   (2*pi*sigma2_total)^(D/2), is *itself* numerically catastrophic at this
   dimensionality (~1e230 in one traced case) - fixed with the standard
   remedy, computing the whole E-step in log-space via log-sum-exp.
3. estimate_scale=True let scale drift smoothly to zero given few source
   points (16 points cannot constrain a ~32,000-degree-of-freedom rotation)
   - fixed by defaulting estimate_scale=False, matching how
   `procrustes_rotation` never estimates scale either (recenter+rescale
   already matches dispersion beforehand).
4. **After all three fixes, CPD still doesn't work at D=253 - confirmed
   to be a real, dimension-driven limitation, not a residual bug**: the
   *exact same toy point-cloud construction* that passed at D=10 (rotation
   error 0.002) was re-run at D=253 with nothing else changed and gave
   rotation error 1.43 - essentially no correspondence recovered, and
   reproduced on a clean synthetic point cloud, ruling out anything EEG-
   specific. Traced to its root cause: the outlier hypothesis's log-weight
   ((D/2)*log(2*pi*sigma2_total)) is genuinely, mathematically hundreds of
   nats larger than any real match's log-likelihood at this dimensionality
   - a real instance of the curse of dimensionality (concentration of
   measure makes all "typical" pairwise distances converge to a similar
   value in ~253 dimensions, so no single correspondence can look
   distinctively better than pure noise under a literal Gaussian-density
   comparison). Not something tuning outlier_weight can fix (its
   contribution to the log-weight is negligible next to the (D/2)*log(...)
   term). A real fix would need a materially different approach -
   dimensionality reduction before computing correspondences (the same
   move state_space_flow.py's PCA step makes), or a non-Gaussian-density
   soft-correspondence scheme (optimal-transport/Sinkhorn-based assignment
   was flagged in the literature review as a documented alternative) - not
   attempted. **CPD is not used in any real-data phase as a result** -
   `stage0_cpd_validation.py`'s own D=253 check documents this on every run
   rather than letting it be discovered once and forgotten.

### ComBat (src/adaptation/combat_align.py): validated null on synthetic, statistically tied on real data

Empirical-Bayes batch-effect harmonization (Johnson, Li & Rabinovic 2007),
transplanted from genomics, applied with "batch" = session. Used
`neuroCombat` directly rather than `neuroHarmonize` (which offers a
friendlier fit/apply-to-new-data API) - installing neuroHarmonize
downgraded numpy to a version incompatible with this project's own moabb
dependency, reverted immediately, full environment re-verified against the
established CSP+LDA sanity check before proceeding. `neuroCombat` itself
needed one small local fix - a `np.int` alias the package still calls
internally, removed from numpy for several major versions; scoped to
`combat_align.py`'s own import, not a global numpy monkeypatch.

**Stage 0** (`stage0_combat_validation.py`, D=253, matching IV-2a exactly):
tested the specific hypothesis that ComBat's per-dimension, empirical-
Bayes-shrunk correction should beat recenter+rescale's single global
statistic when the planted batch effect is heterogeneous across
dimensions - recenter+rescale's structural blind spot. Found the opposite,
consistently, across four different eval trial counts (16/32/80/160): ComBat
clearly beat no correction every time, but never beat recenter+rescale.
Diagnosed, not left as a mystery: the synthetic batch effect was
constructed with no shared structure across dimensions (i.i.d. per
dimension) - close to the worst case for ComBat's core assumption, that
per-feature effects share a common distribution worth shrinking toward.
Plausibly a real structural mismatch, not just this test's construction:
log-Euclidean tangent-space dimensions are not independent features the
way genomics features are - all 253 of them are derived from one much
lower-rank 22x22 covariance matrix, so real session drift likely has far
fewer effective degrees of freedom than "253 independent batch effects."

**Real IV-2a data, checked for completeness despite the synthetic null**:
mean accuracy 66.9% vs. recenter+rescale's established 66.6% (paired,
n=288, same 9 subjects) - combat wins in 5/9 subjects, Wilcoxon p=0.445.
**Statistically indistinguishable from recenter+rescale** - the synthetic
test's specific "no shared structure" construction may simply not match
how real session drift is actually structured across tangent-space
dimensions, but on real data neither method has a real edge over the
other. Not a case for switching to ComBat, but not the clear loss the
synthetic test predicted either - an honest, both-sides-reported result.

### Follow-up on CPD's dimensionality limit: PCA-then-CPD, and full+reduced feature concatenation

Two natural follow-ups to CPD's D=253 failure, both tested rather than
assumed.

**PCA-then-CPD** (`cpd_align_via_pca_subspace` in coherent_point_drift.py):
fit CPD in a PCA subspace of the target cloud, then lift the found d x d
rotation back to a full (253, 253) transform (identity on everything
orthogonal to the subspace - R_full = I + V(R_d - I)V^T - a well-defined
orthogonal transform of the full space, not an approximation that drops
the other 253-d dimensions). Validated (`stage0_cpd_pca_validation.py`)
against the comparison that actually matters - full-D ICP, what Phase 7
already tried on real data (60.4%, worse than baseline) - not just "better
than nothing." First attempt used the same clustered-sample generator as
every other CPD/ICP toy test and found the reduced-space CPD collapsing to
the identity even at d=20; traced to the generator itself, not a new bug -
that generator spreads its cluster-separating signal roughly evenly across
all 253 dimensions, so PCA's top components don't concentrate anything
more useful than the full space already has. Redesigned to embed the true
cluster structure in a genuinely low-rank d_true=20 subspace (matching this
project's own reasoning for why ComBat failed - real tangent vectors are
lower-rank than their 253 nominal dimensions) - a fairer test, but the
result was still negative: full-D ICP recovered the planted rotation
almost exactly (MSE 0.03) while PCA(10)+CPD's best case only partially did
(MSE 67.9, much closer to no correction's 101.8 than to ICP's result).
**Honest caveat, not swept under the rug**: this synthetic setup never
reproduced whatever actually breaks ICP on *real* IV-2a data (Phase 7's
60.4% real-data null) - ICP succeeding cleanly here means this test doesn't
actually probe the regime where an ICP replacement would matter, so it's
inconclusive rather than a clean rejection of PCA+CPD. Not pursued further
given the length of this investigation already.

**Full + PCA-reduced feature concatenation** (`phase7d_pca_concat.py`):
a different idea - not using PCA to fix CPD, but giving the *classifier*
both the full-253-dim recenter+rescaled representation and a d=20 PCA-
reduced summary of it, concatenated, on the reasoning that a coarser view
might be more robust to overfitting even if it discards detail. Tested
with the same regularized combiner (LogisticRegressionCV) Phase 5 found
least-bad for its own feature combination, given this project's repeated
finding that naive concatenation dilutes a strong feature. Real IV-2a
result: mean 67.2% vs. full-alone's 66.6% (paired, n=288) - a small,
real-looking improvement, concat wins in 6/9 subjects, but **not
significant** (Wilcoxon p=0.367). Same character as Phase 7's own
supervised-rotation finding: intriguing, not yet solid evidence, and not
the double-digit gain that would justify switching to it on its own.

## Prototypical-network meta-learning: stalled, then fixed by matching the episode design to the real task

`src/adaptation/prototypical_network.py` - encoder (small MLP on each
trial's own log-Euclidean tangent vector, the same covariance/tangent-
space representation used throughout this project, *not* wavelet-
calibration's mechanism) meta-trained via episodic training on PhysioNet,
deployed on IV-2a by computing prototypes fresh from each subject's own
calibration trials - no fine-tuning, no population-derived fixed
correction. Two episode-sampling designs were tried; the first is kept in
the code for comparison, the second is now the default.

### First attempt: disjoint support/query subjects - stalled

Following MetaWearS's own published design (support and query subjects
non-overlapping per episode, mimicking "generalize to a patient never seen
before"):

- **Synthetic gate: clean pass** (100% on 15 held-out synthetic subjects).
  Confirmed the implementation itself was correct.
- **Real PhysioNet meta-training did not converge at all.** Train loss sat
  exactly at ln(4) ~= 1.386 (uniform 4-class guessing) for 200 episodes,
  zero measurable improvement; validation episode accuracy stayed at
  chance. A materially different failure than Module A's - Module A
  learned fine but transferred badly; this didn't even learn its own
  training objective.
- **Root-caused via two controls:** (1) the encoder *can* fit real
  PhysioNet data - it overfits one fixed episode to 100% in 60 steps,
  ruling out an implementation bug; (2) a *fixed* (non-rotating) 8-subject
  pool showed real, if weak, learning (loss 1.386 -> ~1.2-1.3) while the
  full rotating 20-subject pool showed none. Diagnosis: pooling many
  different real subjects into one episode's prototype computation
  produces a training signal dominated by between-subject covariance
  heterogeneity - the same central finding as everywhere else in this
  project (Module A, shrinkage), now breaking *training convergence*
  itself rather than just deployment-time transfer.

### The fix: same-subject episodes (`sample_same_subject_episode`)

The user's insight: our actual deployment task is never cross-subject at
all - IV-2a's calibration -> eval is always the *same* subject, a
different session. So training episodes should match that exactly:
support and query both drawn from *one* subject's own trials (a different
split), never pooling different subjects into a shared prototype. Cross-
subject generalization still happens - across many episodes, each
sampling a different random PhysioNet subject - but never *within* one
episode's loss, which is exactly the computation that was breaking. This
also happens to be a more honest match to the real task than the
published disjoint-subject design, not just a workaround.

- **Synthetic gate: still a clean 100% pass** (confirms the fix didn't
  break the mechanism itself).
- **Real PhysioNet meta-training converged cleanly this time**: train loss
  showed a consistent downward trend (1.34 -> ~0.92-1.2 depending on run
  scale) instead of sitting flat; validation episode accuracy reached
  40-55%, clearly and repeatably above the 25% chance level.
- **Full real result (Phase 6, 108 PhysioNet subjects, all 9 IV-2a
  subjects, `experiments/phase6_prototypical_meta_learning.py`): mean
  49.0% (std 16.1%)** - by far the best result from any mechanism built
  *this session*, and the second-best in the whole project after
  Riemannian alignment:

  | Subject | Accuracy | | Subject | Accuracy |
  |---|---|---|---|---|
  | 1 | 67.0% | | 6 | 35.8% |
  | 2 | 33.0% | | 7 | 53.5% |
  | 3 | 67.4% | | 8 | 59.4% |
  | 4 | 38.2% | | 9 | 61.8% |
  | 5 | 25.3% (chance) | | | |

  Four subjects (1, 3, 8, 9) land in the 59-67% range, genuinely
  competitive with CSP+LDA (62.2%) and Riemannian alignment (66.9%) for
  those specific subjects, using a mechanism that never bakes in a fixed
  PhysioNet-derived correction. Subject 5 sitting exactly at chance is the
  one clear outlier, not yet investigated - worth checking whether it's a
  genuinely hard subject (consistent with CSP/wavelet results for the same
  subject) or something specific to this mechanism.
- **Why this succeeded where Module A and disjoint-subject training
  failed**: the common failure mode across every population-based method
  in this project is requiring the correction/training signal to directly
  compare or align *different subjects'* covariance structure. Same-
  subject episodes never do that - the only thing that crosses subjects is
  which *episode* (i.e. which task instance) gets sampled next, not
  anything inside a single loss computation. This is a genuinely different
  point in the design space from everything tried before it, not just a
  smaller tweak.

### Still open / worth checking before trusting this further

- Subject 5's chance-level result - understand before treating 49.0% as
  representative rather than partly driven by one outlier.
- Whether `episodes_per_step` (currently 8, averaging several independent
  single-subject episodes per gradient step for a smoother gradient) is
  well-tuned, or whether more/fewer would help.
- Statistical significance vs. Phase 2 (Riemannian alignment) hasn't been
  computed yet - 49.0% vs 66.9% is a real gap, but per-subject variance is
  high (std 16.1%) and worth a proper paired test before concluding how
  much of the remaining gap is real vs. noise.
- Whether combining this encoder's embeddings with CSP (the combination
  attempts that failed for wavelet-calibration in Phase 5) fares better,
  now that the standalone mechanism is much stronger - worth revisiting
  given the "wait until standalone is much stronger" condition from
  Phase 5's postmortem may now be met.

## Within-subject Riemannian ICP + RPA bridging steps: built, validated, honest null on the practical version

`src/adaptation/riemannian_icp.py` - Iterative Closest Point (Besl & McKay
1992, transplanted from robotics/3D scanning - proposed as a genuinely
novel-to-this-project, cross-disciplinary technique, distinct from RPA
which is already published for EEG specifically) for within-subject
cross-session realignment, plus the recentering/rescaling/rotation steps
requested as a bridging ablation on top of it. All in log-Euclidean
tangent space; all strictly within one subject's own calibration <-> eval
sessions.

### Synthetic validation (both parts passed)

- **Geometric primitives** (`stage0_icp_validation.py`): closed-form
  Procrustes recovers an exact injected rotation to machine precision
  (1e-14) from matched pairs. Full ICP (unknown correspondence, shuffled
  points) recovers a *modest*, realistic-scale injected rotation cleanly
  (rotation error 0.2%) - but a first attempt with a fully random, large
  SO(10) rotation from an identity start failed badly (a well-known ICP
  limitation: it needs a reasonably close starting alignment, not a
  mechanism-implementation bug - confirmed by the modest-rotation retest
  passing cleanly). This directly validates the user's own request to add
  recenter+rescale as ICP's initialization, rather than running it cold.
- **Synthetic EEG session drift** (`stage0_icp_eeg_validation.py`): same
  synthetic subject, two sessions, a known modest channel-mixing drift
  injected into the second. Monotonic, consistent improvement at each
  bridging stage: baseline 48.7% -> recenter+rescale 51.2% -> +ICP 53.7%
  (deliberately harsh test - low SNR, 2-class chance is 50%). Confirmed
  the mechanism doesn't hurt and does help, in the right order, before
  spending real-data compute.

### Real data (Phase 7, `phase7_riemannian_icp.py`): a genuinely mixed, honest result

Classification here is tangent-space LDA (not CSP+LDA - per-trial
realignment doesn't produce something CSP's raw-signal covariance
whitening can consume; see the module's own docstring for the design
reasoning, including a real bug caught before running anything: ICP's
rotation is fit *about each point cloud's own mean*, so correcting only a
single derived mean reference - the natural way to plug into CSP+LDA's
existing whitening pattern - would have made rotation provably invisible
to the classifier. Working per-trial in tangent space was the fix). The
"none" baseline here (65.9% mean) is therefore its own reference point,
not directly comparable to Phase 1's 62.2% - a different feature
representation, though a similarly strong one.

Full 9-subject ablation, all four conditions on top of the tangent-space
LDA baseline:

| Method | Mean accuracy (n=288) | vs. baseline |
|---|---|---|
| none (baseline) | 65.9% | — |
| recenter_rescale | 66.6% | +0.7pp, small and consistent |
| **recenter_rescale_icp** | **60.4%** | **-5.5pp, consistently worse at every sample size** |
| recenter_rescale_rotation | 68.4% | +2.5pp, but **not significant** (Wilcoxon p=0.25, 6/9 subjects) |
| recenter_rescale_rotation_icp | 60.1% | -8.3pp vs. rotation alone - ICP degrades a good result |

- **ICP itself (the actual novel ask) doesn't work on real data, even
  starting from a good recenter+rescale initialization.** Consistently
  worse than doing nothing, at every sample size, across all 9 subjects.
  Unsupervised nearest-neighbor correspondence-finding across real,
  noisy, 4-class EEG trial clouds is evidently not reliable enough for
  ICP's iterative refinement to converge somewhere useful - a genuine,
  well-tested negative result, not a bug (the same mechanism passed
  cleanly on synthetic data with a comparable-scale rotation).
- **Supervised rotation only shows a real (if not statistically
  significant) benefit at n=288** - which means using almost the entire
  eval session's *true labels* to compute the rotation. That's not a
  calibration-free or realistic few-shot deployment scenario; it's closer
  to an oracle upper bound on what rotation *could* do with full
  supervision. At realistic few-shot sizes (n=16-64) it actively hurts.
  Don't present the n=288 number as a practical result without this
  caveat front and center.
- **Stacking ICP after a good supervised rotation makes things worse, not
  better, every time** - consistent with ICP's own standalone weakness:
  its unsupervised correspondence step apparently can't be trusted to
  only ever improve an alignment, even a genuinely good starting one.
- **Bottom line: the "novel, cross-disciplinary transplant" bet on ICP
  didn't pay off here**, tested properly (synthetic-validated, real-data-
  ablated, statistically checked) rather than either oversold or
  dismissed without evidence. Recenter+rescale is a safe, mildly-positive
  addition on its own. Supervised rotation is an intriguing geometric
  finding worth remembering, but not yet a deployable method.

## Phase 8/9: motor-cortex channel focus (closed, negative) + scalp flow-field features (closed, validated mechanism / weak standalone signal / negative combined)

User's idea: IV-2a's 22-channel montage isn't full-scalp - lean harder into
the motor cortex, either by restricting to the electrodes actually over it,
or by extracting features from the *spatial flow* of the signal (how the
ERD/ERS topography moves across the scalp during a trial), optionally via a
learned embedding. Two separate sub-ideas, tested/built separately.

### 8a. Channel restriction (`phase8_motor_cortex_channels.py`): tested, negative

Three nested channel sets - full 22, central 17 (drops only Fz + P1/Pz/P2/POz),
narrow 9 (C5-C6 row + FCz/CPz only) - each run through both CSP+LDA (Phase 1)
and CSP+LDA+Riemannian-alignment (Phase 2), compared back to full-22 via
paired McNemar per subject and Wilcoxon across subjects.

| Channel set | acc (no alignment) | acc (Riemannian alignment) |
|---|---|---|
| full_22 | 62.2% | 66.9% |
| central_17 | 61.2% | 64.4% |
| narrow_9 | 56.1% | 59.7% |

Monotonic and significant in the wrong direction: central_17 loses to
full_22 in 7/9 subjects (Wilcoxon p=0.039), narrow_9 loses in **9/9**
subjects (p=0.004). **Restricting channels hurts, consistently.** This
makes sense in hindsight - CSP already *is* a data-driven spatial filter
that learns per-subject which channel combinations are discriminative; it
doesn't need to be told "focus on C3/C4" because it already converges
toward something like that where the data supports it, while also legally
using whatever smaller complementary signal exists in the frontal/parietal
channels (movement-related artifacts, broader network involvement) that a
hard restriction throws away. **This half of the idea is closed out** -
don't hand-restrict channels for this task; if anything, more channels case
generally helped, not fewer.

### 8b. Scalp flow-field features (`src/adaptation/flow_field.py`): primitives built and validated, then tested on real data (weak standalone, negative combined - see 8c/9 below)

The more novel half - not yet tried anywhere in this project. Per trial:
mu/beta Hilbert envelope -> short sliding-window frames -> each frame
interpolated (thin-plate RBF) from the 22 sparse electrode positions onto a
dense 2D grid (electrodes projected via the standard EEG topomap polar
projection: azimuth preserved, radius = (pi/2 - elevation)/(pi/2), verified
Cz lands near the origin and Fz/P1/Pz/P2/POz - the very channels 8a found
unhelpful to drop - land farthest from it) -> Horn & Schunck (1981) dense
optical flow between every consecutive frame pair, borrowed from computer
vision the same way ICP was borrowed from robotics. Output: a per-trial
sequence of (u, v) velocity fields describing how the ERD/ERS topography
moves across the scalp over the trial - a temporal-spatial trajectory
feature no prior phase here has used (CSP and Riemannian alignment both
collapse a trial to one covariance matrix and discard this entirely).

**Stage 0 primitive validation** (`stage0_flow_field_validation.py`, both
checks passed):
- Electrode projection topology sanity (Cz central, rim channels peripheral) - passed.
- Horn-Schunck recovery of a known planted translation on a synthetic
  Gaussian blob - **failed on the first pass** (recovered only ~43% of the
  true velocity, flat across a 6x range of velocity magnitudes tested) and
  was diagnosed, not just loosened past: the flat ratio (not degrading with
  larger displacement) ruled out the usual "large motion breaks the
  linearization" explanation and pointed at regularization instead. Horn &
  Schunck's classic alpha=1.0 default is tuned for ~8-bit-scale image
  intensities; this pipeline's topomaps are normalized envelope values in
  ~[0, 1] with correspondingly smaller gradients, so alpha=1.0 let the
  alpha**2 term dominate the flow update's denominator and shrank every
  recovered flow toward zero regardless of the true magnitude. Fixed by
  lowering the default to alpha=0.1 (empirically verified: recovers the
  planted velocity to within 4.4% at that setting). Both checks pass
  cleanly now.

**Synthetic-EEG spatial-drift test** (`stage0_flow_field_eeg_validation.py`,
same role `stage0_icp_eeg_validation.py` played before ICP touched real
data): a purpose-built generator (fixed spatial pattern won't do here -
needed a Gaussian ERD hotspot whose *center* linearly migrates between two
real electrode positions, in true 3D anatomical distance, over the
onset-to-trough window, then holds) plants a known drift; the pipeline's
recovered flow direction is compared against that drift projected into the
pipeline's own 2D coordinate system via the identical `project_3d_to_2d`
the real electrodes use. Tested three planted directions plus an exact
reversal check, not just one - the same lesson as the ICP toy test's
isotropic-cloud bug, a test that "passes" for the wrong reason is worse
than no test. First pass (80 trial-averaged repeats, -3dB SNR) found a
genuine, physically-grounded anisotropy: a left-right drift (C3<->C1, along
IV-2a's densely electrode-sampled central row) resolved cleanly, but an
anterior-posterior drift (FC3->CP3, no equivalent dense electrode chain)
failed (cosine similarity 0.06). Diagnosed, not papered over: raising SNR
alone fixed FC3->CP3 (0.81 at +15dB), and the failure was direction-
specific rather than uniform - both rule out a coordinate bug and point at
a real cause, IV-2a's electrode geometry genuinely carries less spatial
information off the central row. **Fix**: 200 trial-averaged repeats
(still -3dB, same harsh SNR, no per-direction tuning) resolves all three
planted directions (C3->C1: 0.986, C1->C3: 0.975, FC3->CP3: 0.578) with a
near-exact reversal check (C3->C1 vs. C1->C3 recovered directions: cosine
-0.998, ruling out a fixed-artifact explanation). **Passed - the mechanism
is validated and its real limitation (directionally uneven reliability,
worse off the densely-sampled central row) is documented in
`flow_field.py`'s own docstring**, not discovered later on real data.

### 8c/9. Real IV-2a data (`phase9_flow_field_features.py`, `phase9b_flow_field_plus_csp.py`): weak but real standalone signal, hurts when combined with CSP

Route (a) from the plan above - engineered summary features (mean and std
of u, v, and speed: 6 scalars, pooled per trial over its Horn-Schunck flow
sequence), classified per-subject by an LDA trained on that subject's own
calibration session, no cross-subject data anywhere. Two spatial-pooling
conditions: whole_grid (every grid cell) and central_roi (radius 0.6,
~central_17's span - not because 8a's raw-channel restriction worked, it
didn't, but restricting *where the flow is pooled from* is a different
question and worth its own honest test).

| Feature set | Mean accuracy (chance 25%) | Wilcoxon vs. chance |
|---|---|---|
| whole_grid | 28.7% | p=0.020 (7/9 subjects above chance) |
| central_roi | 29.3% | p=0.035 (6/9 subjects above chance) |

Real, if weak: single-trial flow-field features carry statistically
significant signal above chance, confirming the mechanism validated on
synthetic data does transfer to real EEG rather than being swamped by
noise entirely - but at 28-29%, nowhere close to competitive with CSP alone
(62.2%) or Riemannian alignment (66.9%) as a standalone classifier.
central_roi vs. whole_grid: no meaningful difference (paired Wilcoxon
p=1.0, 5/9 subjects either way) - unlike 8a's raw-channel restriction,
restricting *where the flow feature is pooled from* is close to a wash,
neither helping nor hurting.

**Combined with CSP** (concatenate flow features onto CSP's 8 log-variance
features, `LogisticRegressionCV` combiner - the same combiner Phase 5 found
least-bad for its own wavelet+CSP combination): **hurts**, consistently.
Mean drops from 62.2% (CSP alone) to 59.4% (combined), worse in 8/9
subjects, Wilcoxon p=0.043. Same failure shape as Phase 5's combination
result: a much weaker standalone feature source dilutes CSP's strong,
data-efficient signal when naively concatenated, even through a regularized
combiner. The "flow features see something CSP structurally can't (temporal
evolution, not just a static covariance)" argument for hoping combination
would help was a real, principled reason to test it - but it didn't survive
contact with real data, so it's reported as a genuine null, not silently
dropped.

**Bottom line on the whole flow-field idea**: the mechanism is real (three
separate synthetic gates passed, including catching and fixing a real
regularization bug), and it recovers a genuine, statistically significant,
non-chance signal from real single-trial EEG - a legitimately novel
positive finding, since nothing in this project has extracted usable signal
from within-trial spatial-temporal dynamics before. But it is not currently
a useful classifier on its own, and naive combination with the strong
existing baseline actively hurts rather than helps. Closing this out at the
same status as Phase 5 and Phase 7's ICP component: a validated, working,
honestly-negative-for-practical-use mechanism, not a deployable method.
Untried and worth flagging for later, not attempted here: proper
out-of-fold stacking (rather than naive concatenation) - the same open item
Phase 5's section below already lists, now with a second, independent
feature source that hits the identical combination-failure pattern.

## The problem, restated

BCI ("brain-computer interface") systems decode imagined movement from EEG
to drive a classifier. "BCI illiteracy" - the observation that ~15-30% of
users never get usable accuracy - has historically been treated as a fact
about some people's brains. This project's working hypothesis is that a
meaningful fraction of that "illiteracy" is actually *miscalibration*:
classifiers trained on one session's data degrade on a later session
because the EEG signal characteristics genuinely drift (electrode
placement, skin conductivity, alertness), not because the underlying task-
relevant signal became unreadable. If true, the fix is better *cross-
session correction*, not writing off the affected users.

## Established, trustworthy baselines

These are the two results everything else in this project is measured
against. Both are real, both are independently re-derivable from
`experiments/phase1_baseline.py` and `experiments/phase2_riemannian_alignment.py`.

| Method | Cross-session accuracy (9 IV-2a subjects, 4-class) | Notes |
|---|---|---|
| Within-session (upper bound-ish) | 65.1% | Same-session train/test, repeated CV |
| **Phase 1 - non-adapted CSP+LDA** | **62.2%** | Train on calibration session, test on eval session, zero correction |
| **Phase 2 - Riemannian alignment** | **66.9%** | Per-session recentering (whiten each session's own trials to identity) before CSP+LDA. +4.7pp over Phase 1, statistically significant for 3/9 subjects individually (McNemar), real and reproducible |

Riemannian alignment needs **no other subject's data at all** - it only
uses the target session's own trials. This is the first hint of the
pattern that repeats throughout the project: *self-contained, per-session*
correction works; *borrowing from a population* doesn't.

## The recurring negative result: population-based personalization

Four different implementations of "use other subjects' data to correct
this subject's estimate," across two very different dataset scales and two
very different modeling approaches, all produced the same qualitative
outcome: null or harmful.

### 1. SPD shrinkage (empirical-Bayes, closed-form)

`src/adaptation/spd_shrinkage.py` - log-Euclidean empirical-Bayes shrinkage
transplanted from DTI statistics literature. Shrink a few-trial subject
estimate toward a population grand mean, weight = `between_var / (between_var
+ within_var/n)`.

- **8-subject IV-2a leave-one-out pool** (`phase2b_spd_shrinkage.py`): no
  benefit. Diagnosed cause: too few reference subjects to estimate
  between-subject variance reliably in a 253-dim tangent space.
- **20-subject OpenBMI external pool** (`phase2d_openbmi_shrinkage.py`,
  trace-normalized for cross-hardware compatibility): still null on
  average, but with a much more interesting per-subject finding - it
  *helped* subjects close to the population average and *hurt* genuine
  outliers (subject 2: -3.5 to -6.2pp; subject 9: +1.9 to +4.2pp, same
  method). This is the finding that most directly foreshadowed everything
  that came later: a method that assumes "you're probably like everyone
  else" structurally fails the subjects who are the whole reason
  personalization matters.
- **Eigenvoice extension** (anisotropic, PCA-subspace shrinkage,
  `phase2c_eigenspace_*`): still no better than isotropic shrinkage. More
  parameters didn't help - the reference pool, not the estimator family,
  was the bottleneck.

### 2. Module A - learned geometric correction network

`src/adaptation/geometric_correction.py` - replaced the 2-parameter
closed-form shrinkage with a small MLP, trained to predict a subject's
full-data Riemannian mean from a noisy few-trial subsample. Distillation
target: the already-validated Riemannian alignment output.

- **Synthetic ground truth (Stage 0 gate):** passed cleanly - the network
  demonstrably recovers known covariances from noisy input, beating raw
  estimates by ~2.5-3x. The *mechanism* works.
- **Real cross-dataset transfer (PhysioNet-trained, 108-subject pool,
  evaluated zero-shot on IV-2a):** catastrophic failure. The learned
  correction was **3-15x farther** from a held-out gold covariance than
  doing nothing at all. Root-caused via three independent controls: (a) a
  smaller/more-regularized network made it *worse*, ruling out simple
  overfitting; (b) explicit `1/n_trials` conditioning didn't change
  anything; (c) direct measurement showed PhysioNet's population grand-mean
  covariance *shape* is about as far from a given IV-2a subject's true
  shape as two *different* IV-2a subjects are from each other - i.e. the
  reference population carries almost no transferable shape information
  for this target population, and unlike shrinkage's mathematically capped
  weight (∈[0,1], can only interpolate toward a bad prior), the
  unconstrained MLP could extrapolate *past* a bad prior into worse
  territory.
- **Fix attempted:** a sigmoid-gated, bounded correction (`pred = (1-α)*raw
  + α*learned_correction`, α∈[0,1], initialized near 0) - this bounded the
  worst-case damage from "3-15x worse" down to "roughly 1.5-5x worse for
  2/3 subjects, comparable-or-better for 1/3" - a real improvement in
  *safety*, but still not a net positive. **Population-based correction,
  even learned and safety-bounded, does not currently produce value on
  this task.**

**Takeaway that should generalize:** if a future idea's mechanism is "use
data from other subjects/sessions/datasets to inform this subject's
correction," budget serious skepticism and demand the synthetic-ground-
truth + real-cross-population-transfer double validation this section
describes before trusting it - this project has now disproven that
approach four times at increasing sophistication.

## The pivot: per-session wavelet calibration (current active direction)

Motivated directly by the pattern above: instead of borrowing from a
population, fit a small number of parameters *directly against one
session's own trials*, using a known/textbook-plausible target shape as
the reference - like an eye exam using a known letter chart to measure
*your* specific deviation, not the average patient's prescription. No
population data anywhere in the correction step.

`src/adaptation/wavelet_calibration.py`: a per-channel learnable complex
Morlet wavelet bank plus a shared, parametric ERD/ERS envelope shape
(reusing `src/synthetic_eeg.py`'s Pfurtscheller-style envelope function),
fit via gradient descent to match a session's own cue-locked trials to that
plausible shape.

### What's been validated

- **Stage 0 synthetic gate** (`stage0_wavelet_calibration_validation.py`):
  passes cleanly - recovers known onset/trough timing (within ~0.01-0.1s),
  center frequency (within ~1-2Hz), and spatial topology (contralateral
  channel correctly identified with 3-5x higher gain than distant
  channels) from noisy synthetic trials with known ground truth.
- **Real IV-2a qualitative check:** mixed but informative. 3/12 (subject,
  condition) fits show excellent, textbook-clean contralateral
  identification; 6/12 show poor discrimination. Consistent with genuine
  per-subject signal-quality heterogeneity (i.e. consistent with the
  project's own "BCI illiteracy" premise), not obviously a bug.
- **Onset-boundary bug, found and fixed:** fitted onset kept hugging its
  lower parameter bound on real data. Diagnosed via synthetic injection of
  *known* trial-to-trial onset jitter (`diagnose_jitter_hypothesis.py`) -
  a jitter std of just 0.1s reproduced the exact real-data symptom and
  dropped classification accuracy 75.0%→67.5%. Confirms real EEG's trial-
  to-trial timing variability is a real, load-bearing effect this
  mechanism has to handle, not a nuisance to ignore.
- **First classifier attempt (Phase 4, nearest-template on a single pooled
  score per class): exactly chance (25.0% ± 1.7%),** including *in-sample*
  (classifying the calibration trials the templates were fit on) - proof
  the scoring rule itself was broken, not a generalization problem.
  Root-caused to two sequential metric bugs: (1) raw MSE confounds a
  trial's own noise scale with true class match (fixed via correlation);
  (2) raw correlation lets "loud" templates (larger effect size) win
  almost universally regardless of true match quality (fixed via RMS
  normalization / cosine similarity).
- **Jitter-tolerance, done right (Phase 4b):** first attempt (search over
  candidate time-lags per trial, keep the best) made things *worse* even
  at zero jitter (75.0%→58.1%) - a multiple-comparisons/peak-picking bias
  from searching ~251 candidate alignments. Fixed by blurring the
  *template* by the expected jitter amount instead (a single deterministic
  transform, no per-trial search) - validated on synthetic data (75.0%→
  86.3% at zero jitter, stays ahead of the old approach at every jitter
  level 0.1-0.3s tested).
- **Per-channel features instead of pooled scores:** letting a regularized
  classifier (shrinkage LDA) learn per-channel weighting instead of a
  hand-picked pooling rule was the single biggest lever found so far.
- **Real result (Phase 4b, full 9-subject cross-session):** mean accuracy
  **25.0% → 31.4%**, Wilcoxon p=0.0195 vs. the old pooled approach (8/9
  subjects improved), binomial p<0.000001 vs. chance pooling all 2,592
  eval trials. Real, statistically validated, still well below CSP+LDA.
- **Time-window breakdown (Phase 4c), a mixed/honest non-result.** Natural
  next increment: break each per-channel score into 5 phase-based windows
  (baseline/onset/trough/rebound/recovery, using each template's own
  fitted boundaries) instead of pooling over the whole trial. On synthetic
  data with known jitter this was unambiguously good - slightly behind
  per-channel-only at zero jitter (0.844 vs 0.863, expected: more features
  without more signal costs a little) but consistently *ahead* at every
  realistic jitter level tested (+3.7 to +6.3pp at 0.1-0.3s). Real IV-2a
  data (confirmed to have jitter) did **not** show the same gain: Phase 4c
  mean 31.9% vs. Phase 4b's 31.4%, Wilcoxon p=0.99 (5/9 subjects better,
  4/9 worse - indistinguishable from a coin flip). Most likely
  explanation: 440 features (4 classes x 22 channels x 5 windows) from
  ~288 calibration trials (0.65 samples/feature) costs about as much in
  estimation variance, even under shrinkage LDA, as it gains from the
  extra temporal resolution - real EEG noise is evidently messier/more
  multi-dimensional than the single controlled Gaussian-jitter source the
  synthetic test used. **Lesson: a synthetic-data win under one specific,
  controlled confound (here, timing jitter) doesn't guarantee a real-data
  win if the fix also costs sample efficiency - check the feature-to-
  sample ratio, not just the mechanism, before expecting the synthetic
  gain to transfer.** Phase 4b (fewer features, same real-data accuracy)
  remains the simpler, equally-good choice - keep it as the reference
  implementation unless a future change specifically addresses the sample-
  efficiency cost (e.g. dimensionality reduction on the windowed features,
  or more calibration trials).

### What's been tried and failed for this mechanism specifically

- **Combining with CSP + MiniRocket features (Phase 5):** naive feature
  concatenation into one LDA actively *hurt* CSP's own performance (0.622→
  0.548 mean, 8/9 subjects worse, Wilcoxon p=0.008) - CSP alone is strong
  enough and data-efficient enough that adding weaker, noisier feature
  blocks dilutes rather than helps, even after standardizing each block
  and using a properly regularized (L2 logistic regression) combiner,
  which recovered some but not all of the loss (0.548→0.548 mean, still
  below CSP alone). **Do not revisit naive combination until the
  standalone wavelet-calibration mechanism is substantially stronger, or
  until proper out-of-fold stacking is tried (see below).**
- **Time-window breakdown (Phase 4c):** see above - synthetic win, real-
  data wash. Not a failure of the underlying idea so much as a reminder
  that added feature richness has a sample-efficiency cost that must be
  paid for by real signal, and real data may not have as much extra signal
  along that axis as the synthetic test (with its single, clean jitter
  source) suggested.

## Current numbers (single source of truth - update when a phase reruns)

All cross-session, 9 IV-2a subjects, 4-class, unless noted.

| Method | Mean accuracy | Source |
|---|---|---|
| Chance level | 25.0% | — |
| Phase 4 - wavelet calibration, pooled score (broken) | 25.0% | `results/phase4_wavelet_calibration.csv` |
| **Phase 4b - wavelet calibration, per-channel + jitter-tolerant + LDA (current reference)** | **31.4%** | `results/phase4b_wavelet_per_channel.csv` |
| Phase 4c - wavelet calibration, + time-window breakdown (no real gain, more features) | 31.9% (not significant vs 4b) | `results/phase4c_wavelet_windowed.csv` |
| Phase 5 - wavelet-only (pooled, LDA) | 29.2% | `results/phase5_combined_features.csv` |
| Phase 5 - MiniRocket-only | 46.0% | `results/phase5_combined_features.csv` |
| Phase 5 - CSP + wavelet + MiniRocket combined (logreg) | 54.8% | `results/phase5_combined_features.csv` |
| Phase 6 (disjoint-subject episodes, broken) | ~chance | not saved - stalled at 20-subject smoke-test scale, never run to completion |
| **Phase 6 - prototypical-network meta-learning, same-subject episodes (second-best overall)** | **49.0%** (std 16.1%) | `results/phase6_prototypical.csv` |
| Phase 1 - CSP+LDA, non-adapted | 62.2% | `results/phase1_baseline_iv2a.csv` |
| **Phase 2 - CSP+LDA + Riemannian alignment (best overall)** | **66.9%** | `results/phase2_riemannian_alignment_iv2a.csv` |
| Phase 5 - CSP-only (within that run) | 62.2% | matches Phase 1 almost exactly - good consistency check |
| Phase 7 - tangent-LDA, no realignment (own baseline, not directly comparable to Phase 1) | 65.9% | `results/phase7_riemannian_icp.csv` |
| Phase 7 - + recenter/rescale | 66.6% | `results/phase7_riemannian_icp.csv` |
| Phase 7 - + ICP (novel ask, real-data null) | 60.4% (worse than baseline) | `results/phase7_riemannian_icp.csv` |
| Phase 7 - + supervised rotation (n=288, near-oracle, not significant) | 68.4% (p=0.25) | `results/phase7_riemannian_icp.csv` |
| Phase 7 - + rotation + ICP | 60.1% (worse than rotation alone) | `results/phase7_riemannian_icp.csv` |
| Phase 8a - CSP+LDA, central-17 motor channels only (worse than full 22) | 61.2% (no align) / 64.4% (aligned) | `results/phase8_motor_cortex_channels.csv` |
| Phase 8a - CSP+LDA, narrow-9 motor channels only (worse than full 22, 9/9 subjects) | 56.1% (no align) / 59.7% (aligned) | `results/phase8_motor_cortex_channels.csv` |
| Phase 9 - flow-field features standalone, whole grid (real signal, weak) | 28.7% (p=0.020 vs chance) | `results/phase9_flow_field_features.csv` |
| Phase 9 - flow-field features standalone, central ROI (~same as whole grid) | 29.3% (p=0.035 vs chance) | `results/phase9_flow_field_features.csv` |
| Phase 9b - flow-field features + CSP combined (worse than CSP alone, 8/9 subjects) | 59.4% (p=0.043 vs CSP's 62.2%) | `results/phase9b_flow_field_plus_csp.csv` |
| Phase 10 - state-space flow-field frequency, nearest-template (real-data null) | 25.8% (chance 25.0%, no subject significant) | `results/phase10_state_space_flow_frequency.csv` |
| ICA-cleaned data + CSP+LDA, no realignment (broken without alignment - see below) | 42.8% (3/9 subjects at exact chance) | `results/cleaned_riemannian_check.csv` |
| ICA-cleaned data + CSP+LDA + Riemannian alignment | 61.3% (worse than uncleaned's 66.9%, std 0.160 - noisier too) | `results/cleaned_riemannian_check.csv` |
| Phase 7c - ComBat (n=288, replacing recenter+rescale) | 66.9% (tied with recenter_rescale's 66.6%, p=0.445) | `results/phase7c_combat.csv` |
| Phase 7c - CPD (replacing ICP) - NOT run on real data | n/a - fails at D=253 in Stage 0, see write-up above | `experiments/stage0_cpd_validation.py` |
| Phase 7d - full (253-dim) + PCA-reduced (d=20) feature concatenation | 67.2% (full alone 66.6%, p=0.367, not significant) | `results/phase7d_pca_concat.csv` |

## Artifact-cleaned data pipeline: built, one real bug fixed, first validation is a mixed/negative result

User's request: stop repeating "constant data cleaning" per-experiment -
build a reusable, cached, artifact-cleaned (eye blinks, other EOG-
correlated artifacts, muscle) version of IV-2a, with notch + bandpass
filtering, so every future experiment can load clean data directly instead
of the raw MOABB pull every prior phase has used.

**Built**: `src/preprocessing.py` (`clean_raw`, `get_cleaned_session_raw`)
and `src/datasets.py`'s `load_iv2a_subject_cleaned` - a drop-in replacement
for `load_iv2a_subject` (identical output shape/scale, same 22-channel
order). Pipeline, applied once per subject/session to the *continuous*
concatenated-across-runs recording and cached under `data/cleaned/` as
MNE `.fif` files: notch (50Hz + 100Hz harmonic - this dataset was recorded
in Graz, Austria, European mains) -> general-purpose wide bandpass
(1-40Hz, not baked to any one analysis's narrower band) -> ICA fit on the
EEG channels -> automatic exclusion of components correlated with the 3
EOG channels (`ica.find_bads_eog`) or showing a muscle-artifact spectral
signature (`ica.find_bads_muscle`) -> reconstruction. All 9 subjects
cached (~25-35s each, one-time cost).

**Checked in passing, in case future work wants it**: the original BCI IV-2a
GDF files include a per-run "N/48 trials flagged as artifact" count in
their metadata (confirmed: `raw.info['description']` shows this, e.g.
"Artifacts: 2/48 trials"), but MOABB's wrapper does not expose *which*
specific trials that count refers to as an accessible annotation or event
code - only the aggregate count survives, in free text. Recovering the
original per-trial flags would need bypassing MOABB and parsing the raw
GDF files directly; not pursued here.

**A real bug found and fixed before trusting anything downstream**: the
first pass used `n_components=0.99` (a variance-fraction) for ICA, which
looked like a reasonable default but was badly wrong for this data - one
single dominant component captured 85.7% of total variance alone (checked
directly via the eigenvalue spectrum), so 0.99 kept only **7** components
total. `find_bads_eog`/`find_bads_muscle` score each component as an
*outlier relative to the others* - with only 7 components there isn't
enough of a distribution for that to work at all, and both returned zero
detections even for a component correlating at r=0.81 with an EOG channel
(confirmed by computing that correlation directly, bypassing the
detector). Fixed with a fixed component count close to full rank
(`n_components=20`), verified directly to find 2 genuine EOG components
and 1 muscle component on the same data the broken version found nothing
in.

**First validation, honest and mixed - not yet a case for using this data**:
naive CSP+LDA on cleaned data collapsed to near/exact chance for several
subjects (mean 42.8%, 3/9 subjects at exactly 25.0%) - diagnosed, not
assumed broken: within-session cross-validation on *either* session alone
(calibration: 76.5%, eval: 76.6%, subject 1) showed the cleaned data
itself is healthy and well-behaved, ruling out "cleaning destroyed the
signal." The collapse is cross-session-specific: each session's ICA is
fit *independently* (different data, different excluded components), so
cleaning applies a session-specific spatial transform on top of whatever
cross-session drift already existed - confirmed directly: adding this
project's own established fix, Riemannian alignment, on top recovered
subject 1 to 78.5% (better than the uncleaned pipeline). **But across all
9 subjects, cleaned+aligned still underperforms the established uncleaned+
aligned baseline** (61.3% vs. 66.9%) and is noisier (std 0.160, some
subjects clearly better - subject 9 at 80.9% - others clearly worse -
subject 5 at 42.0%). This suggests the automatic EOG/muscle detectors are
likely removing some genuine, class-discriminative signal along with (or
instead of) real artifacts for at least some subjects - a known risk of
automatic ICA-based rejection, not yet diagnosed further (untried:
checking whether restricting to EOG-only removal, without muscle
detection, changes this; inspecting which specific components get
excluded per subject for physiological plausibility).

**Bottom line**: the infrastructure is real, cached, and ready to use -
but is not yet a validated improvement over the existing raw-data
pipeline, and must never be evaluated without a realignment step on top
(the uncleaned pipeline's own Phase 1 "no realignment" result isn't a fair
comparison point for cleaned data, since cleaning demonstrably adds its
own cross-session variability that the raw data's Phase 1 baseline never
had to contend with). Do not switch other phases over to
`load_iv2a_subject_cleaned` based on this first result alone.

## Methodology lessons (apply these before trusting any new result)

1. **Always validate on synthetic ground truth before touching real data.**
   Every real-data bug this session that got caught before wasting
   compute was caught this way: the SNR/spatial-falloff bug in the
   synthetic generator itself, Module A's cross-dataset failure, the
   jitter hypothesis, the max-over-lags peak-picking bug. This is not
   optional process overhead - it is the mechanism that has made this
   project's negative results trustworthy instead of just discouraging.
2. **Scale/normalization confounds are the single most common bug class
   found this session.** MSE's noise-scale confound, raw correlation's
   template-magnitude confound, per-channel-normalization's noise-
   amplification-for-uninformative-channels confound - three variants of
   the same root issue (a metric that looks reasonable but secretly
   depends on something other than true match quality). When a new metric
   or score is introduced, explicitly ask "what happens to this score for
   a trial/template with no real signal, just noise, at different noise
   scales?" before trusting it.
3. **A "smarter"/higher-capacity method is not automatically better than a
   simpler one when the reference data is small.** Module A (a neural
   network) failed where 2-parameter shrinkage merely underperformed,
   because the network had more room to extrapolate confidently into
   wrong territory. When in doubt, prefer methods with a built-in
   "do-nothing" floor (bounded interpolation, not unconstrained
   extrapolation).
4. **"Obvious" fixes for a diagnosed problem can make it worse - test the
   fix, don't just deploy it.** The max-over-lags jitter fix is the
   clearest example: it was the natural, standard-sounding fix for the
   correctly-diagnosed jitter problem, and it made things worse via a
   different bug (peak-picking bias) that only showed up empirically.
5. **In-sample (or same-session) checks are cheap and catch scoring-rule
   bugs before blaming generalization.** Phase 4's chance-level result
   could have been misdiagnosed as "the mechanism doesn't generalize
   cross-session" - checking in-sample accuracy first (also chance) proved
   the scoring rule itself was broken, saving a lot of wasted iteration on
   the wrong hypothesis.
6. **CSP+LDA is a genuinely strong, hard-to-beat baseline for this
   specific task family.** Don't expect to beat it by accident or via a
   method not specifically designed with its strengths in mind.
7. **A synthetic-data win doesn't automatically transfer if the fix also
   costs sample efficiency.** Phase 4c's per-phase-window features were a
   clean, consistent win on synthetic data (which has abundant trials and
   one clean, controlled confound to fix) but a wash on real data (finite
   ~288 calibration trials, presumably messier/higher-dimensional real
   noise). When a fix adds feature dimensionality, check the resulting
   samples-per-feature ratio, not just whether the synthetic gate passed.
8. **The real failure condition isn't "population data," it's "directly
   aligning two different subjects inside one computation."** Refined via
   the prototypical-network fix: population data used to teach a *general
   skill* via same-subject-only training tasks (never comparing subject A
   to subject B inside one loss) worked (49.0%), where population data
   used to produce a direct cross-subject correction failed every time
   (shrinkage, Module A) - and even *training* with cross-subject-pooled
   episodes failed to converge at all, before the fix. When designing a
   population-informed method, ask specifically: does any single
   computation (a loss, a correction, a distance) ever require comparing
   two different subjects' data directly? If yes, expect it to struggle
   regardless of how sophisticated the method otherwise is.

## Tried and closed out: dimensionality reduction on windowed features

Phase 4c follow-up, tested on the same synthetic jitter-sweep harness
(`diagnose_jitter_hypothesis.py`) before touching real data again - both
closed out as dead ends, not worth revisiting without a genuinely different
approach:

- **PCA down to 10/20/40 components:** decisively *worse* than the raw
  220-dim windowed features at every component count and every jitter
  level tested (e.g. 0.775-0.781 vs. 0.844 at zero jitter). PCA is
  unsupervised - it keeps high-variance directions regardless of whether
  they separate the classes, and most of a matched-filter feature space's
  variance is plausibly noise, not signal. Shrinkage LDA's own built-in
  regularization already handles the many-features-few-samples regime
  better than PCA's blind variance-based prefiltering.
- **L1-penalized logistic regression** (supervised feature selection via
  the labels, instead of PCA's unsupervised variance criterion): mixed,
  not a clear win - slightly worse than plain shrinkage LDA on the raw
  windowed features at low jitter (0.794 vs. 0.844 at zero jitter), only
  marginally better at high jitter (+3-4pp at 0.2-0.3s). Not enough of a
  gain to expect it to flip Phase 4c's real-data null result.

**Conclusion: the sample-efficiency cost of the windowed features isn't a
fixable-by-standard-dimensionality-reduction problem.** If time-window
features are revisited, it likely needs either substantially more
calibration trials than IV-2a provides, or a different feature-selection
principle entirely (e.g. informed by which windows a given template's own
fit trusts most, not a generic statistical criterion) - not flagged as a
promising near-term lever anymore.

## Untried, worth doing (identified but not yet built; items since completed are annotated in place)

- **Prototypical-network-style meta-learning (new idea, from literature -
  see "External literature check" below).** Structurally different from
  everything tried so far: instead of population data producing a *fixed*
  correction (Module A's failure mode) or no population data at all
  (wavelet-calibration), population data would only train an *encoder* to
  produce a good discriminative embedding space; the actual per-subject
  classification decision (a prototype = mean embedding of that subject's
  own few labeled calibration trials) is always computed fresh from target
  data, never baked in from the base population. This directly sidesteps
  Module A's diagnosed failure mode (population covariance shape not
  matching target shape) since the final decision never depends on that
  match. PhysioNet's 108-subject pool (downloaded, otherwise unused) would
  be the natural base dataset, IV-2a the target. Needs the same synthetic-
  first validation discipline as everything else here before trusting it
  on real data. **Status: since built - and it stalled at first, then was
  fixed by redesigning the episodes to be same-subject-only.** See the
  "Prototypical-network meta-learning" section below for the fix and the
  49.0% result; this bullet is kept only as the original proposal.
- **Proper out-of-fold stacking** for combining wavelet-calibration with
  CSP: Phase 5's combination was fit-once-on-full-calibration-session, not
  k-fold cross-validated base-learner outputs. A meta-classifier trained
  on honest, non-leaked out-of-fold predictions might learn to trust CSP
  heavily and only lean on wavelet-calibration where it adds real value,
  rather than diluting uniformly. Bigger compute cost (each base learner
  needs re-fitting per fold), not yet attempted.
- **Per-subject/per-class fitted jitter tolerance** instead of the current
  fixed 0.15s Gaussian smoothing width - real jitter presumably varies by
  subject and possibly by class.
- **PhysioNet's 108-subject pool sits fully downloaded and unused** for
  anything except Module A's now-deprioritized population approach. If a
  future method genuinely needs population data (unlike wavelet-
  calibration), it's ready to go without re-downloading.

## Open questions genuinely unresolved

- Whether wavelet-calibration can ever close the gap to CSP+LDA as a
  *standalone* classifier, or whether its real value turns out to be as a
  *diagnostic*/*personalization-fingerprint* tool rather than a
  classification feature source in its own right.
- Whether the erd_magnitude sign ambiguity found during Module A's
  synthetic gate (flipping sign of both erd_magnitude and channel_gain
  gives a near-identical fit) has any practical consequence for wavelet-
  calibration's real-data fits - not yet checked specifically for this
  mechanism.
- Whether real per-subject "poor fit" cases (6/12 in the qualitative
  check) reflect genuine weak signal or fitting-procedure limitations
  still to be found - not yet disambiguated.
