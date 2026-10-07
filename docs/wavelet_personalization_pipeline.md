# Pipeline Design: Fingerprint-Conditioned Personalization with Riemannian Distillation

Working design doc for the pipeline discussed after `state_of_the_art.md` §11
(joint-embedding models) and the matched-wavelet follow-up. This is a plan to
work from, not a finished result — update it as stages get built and tested.

## 0. Why this pipeline, in one paragraph

Every purely statistical/geometric personalization method tried so far
(§10.1's shrinkage, §10.1's eigenvoice extension) hit the same wall: too few
reference subjects to estimate population structure reliably, and real
subject-level heterogeneity that a single shared model (population mean,
shared eigenspace) can't represent — it helps subjects close to the "average"
and hurts genuine outliers (concretely: OpenBMI-20 test, subject 2 hurt
-3.5..-6.2pp, subject 9 helped +1.9..+4.2pp by the *same* shrinkage). This
pipeline tries a *learned* rather than purely statistical mechanism, on the
theory that a network can capture structure a 2-parameter shrinkage estimator
can't — while staying honest that a learned model has more parameters to fit
and could need a reference pool this project hasn't had access to before.

**Core idea:** distill the already-validated Riemannian alignment (Phase 2,
+4.7pp over non-adapted baseline) into a personalizable, learnable form,
rather than trying to learn cross-session correction from scratch against a
weak classification-label signal.

## 1. Architecture overview

Two complementary modules, buildable and testable independently before
combining:

```
                    ┌─────────────────────────────────────┐
raw EEG trials ────>│ Module A: Geometric correction net   │──> corrected
(few, noisy)        │ (Riemannian tangent-space distiller) │    covariance
                    └───────────────┬───────────────────────┘    (whitening
                                    │ fingerprint (tangent vector)  reference)
                                    v
                    ┌─────────────────────────────────────┐
                    │ Module B: Personalized wavelet front │──> spectral
raw EEG trials ────>│ (hypernetwork predicts wavelet params│    features
                    │  from Module A's fingerprint)        │    (auxiliary,
                    └─────────────────────────────────────┘     complementary
                                                                  to CSP)
```

Both feed into the existing CSP+LDA evaluation harness from Phase 1/2/2b-d,
so every stage below is a drop-in replacement/addition, comparable via the
same paired-Wilcoxon methodology already used throughout this project.

## 2. Module A — geometric correction network (Riemannian distillation)

**Goal:** replace the closed-form shrinkage formula (`shrink_subject_mean` in
`src/adaptation/spd_shrinkage.py`) with a *learned* mapping from a noisy
few-trial covariance estimate to a prediction of what the full-data
Riemannian-aligned covariance would be.

**Training data construction** (reuses the exact simulation already built for
Phase 2b/2c/2d — this is the same "how many trials would you have in a real
short recording" harness, just repurposed to produce training pairs instead
of an evaluation sweep):

- For each reference subject/session with abundant trials (target: 100+
  subjects, see §7 for candidates):
  - **Target** `y`: the full-data Riemannian mean covariance (what
    `RiemannianAlignment.fit()` already computes from all available trials).
  - **Input** `x`: the empirical mean covariance (or raw tangent vector) from
    a random n-trial subsample, n swept the same way as
    `experiments/phase2b_spd_shrinkage.py` (8, 16, 32, 64, 128...).
- Loss: Riemannian (affine-invariant or log-Euclidean) distance between
  predicted and target covariance — not plain MSE, to respect the SPD
  manifold's geometry, consistent with how every other estimator in this
  project has been built.

**Architecture (keep it small — this is the load-bearing lesson from §10.1):**
a small MLP or a handful of residual blocks operating directly on the
log-Euclidean tangent vector (not a large image-style CNN) — the whole point
is a low-parameter-count model that a ~100-subject pool can plausibly train
without repeating the "not enough data for the model's capacity" failure.

**Evaluation:** identical harness to `phase2b_spd_shrinkage.py` — same
sample-size sweep, same `raw_ai`/`raw_le` baselines, add a `learned` column,
same paired Wilcoxon test across the 9 IV-2a subjects. This makes the result
directly, quantitatively comparable to the already-reported shrinkage null
result — a real apples-to-apples upgrade path, not a fresh unvalidated claim.

## 3. Module B — personalized wavelet front-end

**Goal:** a small hypernetwork that maps a subject fingerprint (Module A's
tangent vector, or the raw few-trial tangent vector if Module A isn't ready
yet) to the parameters of a personalized parametric wavelet — center
frequency and bandwidth per channel (Morlet-style), *not* a large free-form
learnable filter bank, for the same low-parameter-count reasoning as Module
A.

- **Conditioning input:** subject fingerprint (Module A's output, or the raw
  tangent vector as a fallback).
- **Output:** wavelet parameters (small, e.g. 2 numbers x n_channels).
- **Forward pass at deployment:** one pass through the hypernetwork, no
  gradient steps needed — genuinely calibration-free, same spirit as
  Riemannian alignment itself.
- **Downstream use:** CWT features using the personalized wavelet, fed either
  as an auxiliary input alongside CSP features, or compared head-to-head
  against CSP features to see whether the personalized spectral view adds
  information CSP's covariance-based view doesn't already capture.

**Honest expectation, from the literature check already done:** RatioWaveNet
(a *globally*-learned wavelet, not personalized) only gained +0.17 to
+2.54pp over its non-adaptive backbone. Personalization might add more, but
treat this as a moderate refinement to combine with Module A/Riemannian
alignment, not a standalone breakthrough.

## 4. Stage 2 — noise-robustness curriculum (diffusion-style, conditional on Module A working)

Train Module A (and/or B) across a *schedule* of corruption severities
instead of one fixed condition:

- Vary n_trials the same way as the sample-size sweep already does (this is
  free — it's the same axis Phase 2b/2c/2d already varies).
- Optionally inject additional synthetic corruption on top (see §6) at
  varying severity, conditioning the network on a severity indicator the way
  diffusion models condition on the timestep.
- Goal: robustness across the deployment spectrum (few-trial calibration-free
  case included), not just the one severity level happened to be sampled
  during training.

**Build this only after Module A shows a real, statistically significant
edge over the existing shrinkage baseline** — no point adding robustness
training to a base method that doesn't yet beat the null result.

## 5. Stage 3 — explicit noise-residual prediction (conditional on Stage 2)

Two-headed output: predict the clean (Riemannian-aligned) target *and*
predict `noise = raw − aligned` as an explicit residual. Matches the DDPM
finding that noise-prediction (epsilon-parameterization) often trains more
stably than direct clean-signal prediction — the mechanism is the same idea,
applied here instead of to natural images.

## 6. Stage 0 — synthetic ERP/ERD validation harness (build this *first*, before Module A)

**The user's proposal, and a good one:** construct synthetic multi-channel
EEG with a known, textbook-canonical evoked-response/ERD-ERS temporal
envelope, inject controllable synthetic noise, and check whether the pipeline
recovers the known ground truth — the same validation pattern already used
successfully to catch bugs in the SPD shrinkage estimator (the corrected
synthetic test in §10.1's build) and the closure-amplitude check (§10.5).
Real EEG never gives ground truth; synthetic data does, which is exactly what
makes it possible to tell "the model is broken" apart from "the model is
fine, real EEG is just this hard" — a distinction Phase 2b's early debugging
(chance-level collapse, the scale-mismatch bug) needed and didn't have until
diagnosed by hand.

**Concrete construction:**
- Canonical ERD/ERS envelope: power decrease in mu (8-13Hz)/beta (13-30Hz)
  starting ~0.5s post-cue, trough ~1.5-2s, partial post-movement rebound
  (ERS) — standard textbook Pfurtscheller-style time course.
- Known spatial topology: e.g. contralateral C3/C4 maximum for left/right
  hand MI, matching real MI's known (if individually-variable) spatial
  pattern.
- Inject the envelope as a time-varying gain on band-limited oscillatory
  "carrier" activity at a chosen (and variable, across synthetic "subjects")
  peak frequency, at a chosen channel topology.
- Add controllable noise: white noise, 1/f pink noise (realistic EEG
  background), and structured artifacts (blink-like broadband transients,
  muscle-like high-frequency bursts) at swept SNR levels.
- **Critically, also vary the envelope's timing/frequency/topology across
  synthetic "subjects"**, matching the real, confirmed finding that ERD/ERS
  onset latency, peak frequency, and even ERD-vs-ERS direction vary between
  real people — a synthetic benchmark that's identical across all synthetic
  subjects would understate the real personalization challenge and give a
  falsely optimistic validation.

**What this validates before touching real data:**
1. Does Module A's correction network actually recover the known clean
   covariance from noisy/few-trial input, at a known SNR?
2. Does Module B's hypernetwork correctly predict *different* wavelet
   parameters for synthetic subjects with genuinely different injected
   ERD/ERS characteristics (the personalization claim, directly testable
   here since we know the ground truth per synthetic subject)?
3. Does accuracy degrade gracefully as synthetic SNR worsens, the property
   Stage 2's curriculum is meant to build?

## 7. Data sources — researched, not assumed

| Dataset | Subjects | Channel match to IV-2a (22ch) | Size | MOABB class | Notes |
|---|---|---|---|---|---|
| OpenBMI (already used, §10.1) | 54 | 21/22 (missing FCz) | ~1.2GB/subject (~65GB full) | `Lee2019_MI` | Already tested; 20-subject pool showed no benefit for the *statistical* shrinkage estimator specifically |
| **PhysioNet EEG Motor Movement/Imagery** | **109** | **22/22 exact match** | **~15-36MB/subject (~2-4GB full)** | `PhysionetMI` | Recommended primary pool for this pipeline — more subjects, full channel match, ~20x smaller download than OpenBMI. Caveat: original curation drops ~10 subjects (4 for trial-count variability, 6 for recording anomalies) — a companion curation paper exists ("Increasing accessibility to a large BCI dataset," ScienceDirect 2024) worth reading before treating all 109 as clean. 160Hz native rate (vs. IV-2a's 250Hz) — same "different rate, same bandpass filter" non-blocker as OpenBMI's 1000Hz. |
| Cho2017 | 52 | not yet checked | not yet checked | `Cho2017` | Secondary candidate if PhysioNet + OpenBMI pooled still isn't enough |
| BEETL (NeurIPS 2021 challenge) | combines multiple MI datasets | not yet checked | not yet checked | not in MOABB directly | Purpose-built for cross-subject/cross-dataset MI transfer — exactly this problem — but exact composition and subject count need checking before relying on it; flagged as a further avenue, not yet verified |
| TUEG (used by CBraMod/LUNA pretraining) | ~15,000 | clinical EEG, not MI-labeled | ~21,000+ hours | not MI-specific | Context only — shows the scale foundation models actually use (2-3 orders of magnitude beyond anything MI-specific and public), not a direct candidate since it isn't task-labeled for MI |

**Recommendation:** PhysioNet (109 subjects, full channel match, ~2-4GB) as
the primary reference pool for Module A/B training — a strict upgrade over
OpenBMI on subject count, channel match, *and* download cost simultaneously.
Combine with the already-downloaded OpenBMI-20 pool for ~160+ combined
reference subjects if that's not enough on its own. Still an order of
magnitude below what LaBraM/CBraMod-scale foundation models use, so the
"not enough data" risk from §9 below isn't fully retired by this — just
meaningfully improved from the 20 subjects already tried.

## 8. Evaluation protocol

Reuse everything already built, don't invent new evaluation methodology:

- Same paired Wilcoxon test, same 9 IV-2a target subjects, same sample-size
  sweep (8, 16, 32, 64, 128, 288) as `phase2b_spd_shrinkage.py`.
- Same accuracy-summary / significance-reporting pattern from every prior
  experiment this session.
- Report against the *same* baselines already established: Phase 1
  (non-adapted, 62.2%), Phase 2 (Riemannian alignment, 66.9%), Phase 2b/2d
  (shrinkage, null result) — so a new method's result is immediately
  legible against everything already tried, not a number in isolation.

## 9. Risks — read before building

1. **Reference pool size, again.** PhysioNet's 109 (or 109+OpenBMI's 20
   pooled) is a real improvement over 20, but still far below foundation-
   model scale (LaBraM ~500 participants, CBraMod/LUNA ~15,000). A learned
   network has more parameters than the 2-parameter shrinkage estimator that
   already failed at 20 subjects — there's a real chance this pool still
   isn't enough, especially for Module B's hypernetwork. Keep both modules
   deliberately small (§2/§3) specifically to reduce this risk, and treat
   Stage 0's synthetic validation as the gate before spending real compute
   on the real-data version.
2. **Subject-level heterogeneity may persist.** Nothing about switching from
   a statistical to a learned estimator guarantees the "helps average
   subjects, hurts outliers" pattern goes away — a network trained to
   minimize average loss across a population has the same structural
   incentive to fit the average subject well. Check per-subject results (not
   just the average) the same way OpenBMI-20's per-subject breakdown was
   checked — that's what caught the real finding there.
3. **PhysioNet data-quality subset.** Read the curation paper before
   treating all 109 subjects as uniformly clean reference data.
4. **Riemannian-aligned targets are themselves imperfect.** Distilling
   toward Phase 2's Riemannian alignment output caps this pipeline's ceiling
   at "as good as Riemannian alignment, plus whatever the learned model adds
   on top" — it can't exceed the teacher by construction unless combined
   with the classification-loss signal too (worth considering a combined
   loss: distillation + downstream classification, not distillation alone).

## 10. Build order

1. Stage 0 (synthetic validation harness) — no real data needed, validates
   the mechanism works at all before spending real-data compute.
2. Module A alone, real data (PhysioNet primary pool), evaluated against the
   existing shrinkage baseline via the existing harness.
3. Module B alone (wavelet + hypernetwork), conditioned on Module A's
   fingerprint once Module A is validated.
4. Combine A+B, re-evaluate.
5. Stage 2 (noise curriculum) — only if 2-4 show a real, significant edge.
6. Stage 3 (explicit noise-residual head) — only if Stage 2 helps.

## 11. Open questions

- Exact Module A architecture (MLP depth/width) — start minimal, only grow
  if underfitting, not overfitting, is observed.
- Whether to train Module A with a pure distillation loss or a combined
  distillation + downstream-classification loss (see risk #4).
- BEETL's exact dataset composition and subject count — worth a follow-up
  check if PhysioNet + OpenBMI pooled isn't enough.
- Cho2017's channel layout — not yet checked, listed as a secondary
  candidate pending verification.
