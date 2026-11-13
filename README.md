# zero-shot-bci

**Toward calibration-free cross-session EEG decoding.** Research code, results, and a
full negative-results log for closing the calibration → live-session accuracy gap in
motor-imagery brain–computer interfaces (BCIs).

The name states the goal, not the current state. The endpoint is a decoder that works on
a new session — ideally a new user — with **no labelled calibration data at all**. What
this repository contains is the measured path toward it: what works, what provably does
not, and the evidence for each.

**Where it stands.** The best decoder is **R11** ([`src/pipeline.py`](src/pipeline.py)). It is an
equal-weight ensemble of ShallowConvNets, FTSurrogate-trained ShallowConvNets and tangent-space
logistic regression, all on per-session Riemannian-aligned input.
- **On the development data:** 82.5% cross-session on 4-class BCI Competition IV-2a and 80.9% on
  2-class IV-2b, against 66.9% and 77.7% for CSP+LDA with Riemannian alignment.
- **On an untouched lockbox dataset** (BNCI2015_001, opened once under pre-registration):
  - 87.9%, against 84.2% for a single ShallowConvNet and 77.0% for CSP+LDA;
  - it needs only **half the labelled calibration trials** to match a single ShallowConvNet trained on
    the full session.
- **Still required:** a (shorter) calibration session. The fully calibration-free alternatives are
  far behind and are documented as such.

---

## The problem

A BCI decodes imagined movement from EEG. In practice, a classifier trained on one
session degrades badly on the next one, days later. The literature calls the tail of this
distribution "BCI illiteracy": users who never reach usable accuracy. Reported rates run
~15–30% overall and as high as **53.7% for motor imagery specifically**, the hardest
paradigm (Lee et al., 2019, OpenBMI, 54 subjects).

That framing locates the failure in the user. This project's working hypothesis is that a
large part of it is *miscalibration* instead: EEG statistics genuinely drift between
sessions (electrode placement, skin conductivity, alertness), so the fix is better
cross-session correction rather than writing off users. That position is the more
defensible one in the literature (Becker et al., 2022; Thompson, 2018) — see
[`docs/state_of_the_art.md`](docs/state_of_the_art.md) §"Is illiteracy even the right frame?".

**Everything here is evaluated as:** train on the calibration session, test on the
evaluation session, 9 subjects, 4 classes, chance = 25.0% (IV-2a). From R10 onward every
result is also scored on BCI IV-2b (9 subjects, 2 classes, 3 channels, chance = 50%; train on
recordings 1–3, test on 4–5), and a method passes only if it gains on both datasets. The population-data setting is
deliberately hard — a method gets *one subject's own 288 calibration trials*, nothing else
— because that matches "the user just sat down."

Claims that survive this development loop are then confirmed **once** on a held-back lockbox
dataset (BNCI2015_001: 12 subjects, right hand vs feet, session A → B), pre-registered before
download. Evaluation-session labels are never used for fitting, model selection or early stopping.

## Headline results

| Method | IV-2a cross-session | Δ vs. baseline | IV-2b | Source |
|---|---|---|---|---|
| **R11 ensemble: 3 ShallowConvNet seeds + TS+LR + 2 FTSurrogate ShallowConvNet seeds, 1/3 per family** — best overall, confirmed on the lockbox | **82.5%** | +20.3pp | **80.9%** | [`results/r11_round3_diverse_members.csv`](results/r11_round3_diverse_members.csv) |
| R11 with the EOG channels regressed out of the EEG (R14) | 82.0% | +19.8pp | 79.8% | [`results/r14_eog_gate.csv`](results/r14_eog_gate.csv) |
| R10 stack: 3 ShallowConvNet seeds + TS+LR, 1/2 each | 79.0% | +16.8pp | 80.1% | [`results/r10_round2_stack_position.csv`](results/r10_round2_stack_position.csv) |
| ShallowConvNet on per-session Riemannian-aligned input (R3) | 74.7% | +12.5pp | 78.6% | [`results/r3_convnet_session_norm_summary.csv`](results/r3_convnet_session_norm_summary.csv) |
| CSP+LDA + Riemannian alignment (Phase 2) | 66.9% | +4.7pp | 77.7% | [`results/compare_r11_vs_csp_lda.csv`](results/compare_r11_vs_csp_lda.csv) |
| CSP+LDA, no adaptation (Phase 1) | 62.2% | — | — | [`results/phase1_baseline_iv2a.csv`](results/phase1_baseline_iv2a.csv) |
| Prototypical-network meta-learning (Phase 6) | 49.0% (std 16.1) | −13.2pp | — | [`results/phase6_prototypical.csv`](results/phase6_prototypical.csv) |
| Wavelet per-session calibration (Phase 4b) | 31.4% | −30.8pp | — | [`results/phase4b_wavelet_per_channel.csv`](results/phase4b_wavelet_per_channel.csv) |
| Chance | 25.0% | — | 50.0% | — |

**R11 vs CSP+LDA + Riemannian alignment, paired per subject**
([`experiments/compare_r11_vs_csp_lda.py`](experiments/compare_r11_vs_csp_lda.py)):
- R11 is +9.3 pp pooled over 18 subjects (16 up, 2 down, one-sided sign-flip p = 0.0001).
- On IV-2a it is +15.5 pp, with all 9 subjects up.
- On IV-2b it is +3.2 pp, with 7 subjects up. Subjects 2 and 8 are down 4.0 and 2.2 pp.
- With the EOG channels regressed out, the gain is still +8.6 pp (p = 0.0005).

**How much of that to believe.** R11 is the end of a long search on these same evaluation
sessions. After shrinking for that search, the council review
([`reports/r14_council_and_ideation.md`](reports/r14_council_and_ideation.md)) estimates the
true gain at about +3.5 pp over a single ShallowConvNet and about +1.2 pp over R10. Its most robust
fact is narrower: the mixed ensemble beats a CNN-only ensemble with the same number of members
by +2.3 pp, with 16 subjects up and 0 down, on both datasets. **R11 has since been confirmed on an
untouched dataset** (BNCI2015_001, see Status): +3.7 pp over a single CNN, as the shrunk estimate
predicted.

**Against the published state of the art** ([`docs/state_of_the_art.md`](docs/state_of_the_art.md) §0;
same subject-specific session 1 → session 2 split):
- The strongest leak-free published result is TCFormer (2025): 84.8% on IV-2a and 87.7% on IV-2b.
- R11 is 2.3 pp below it on IV-2a.
- On IV-2b the gap is mostly *eye movement*. Published IV-2b pipelines use 0.5–100 Hz input. Moving
  our ShallowConvNet to that input gains +4.4 pp, but regressing out the EOG channels removes all of
  it (D6).
- So published IV-2b numbers likely include ~4–5 pp of ocular decoding. Our 4–38 Hz input keeps that
  share to about 1 pp (R14).

Riemannian alignment (per-session recentering, which whitens each session's own trials to
identity) is the adaptation step every successful pipeline here relies on, and it needs **no other
subject's data**. Every method that borrows another subject's data to correct this subject's
estimate has failed. The full number table is the "Current numbers" section of
[`docs/progress_and_direction.md`](docs/progress_and_direction.md#current-numbers-single-source-of-truth---update-when-a-phase-reruns).
The complete log of every gate, screen and confirmation since R3 is
[`docs/diag_gates_d1_d4.md`](docs/diag_gates_d1_d4.md).

## The finding that matters most

> **The failure mode is not "using population data." It is requiring a single computation
> to directly compare or align two different subjects' covariance structure.**

This was refined across six separate attempts, each more sophisticated than the last:

| Attempt | What it did | Outcome |
|---|---|---|
| SPD shrinkage (`src/adaptation/spd_shrinkage.py`) | Empirical-Bayes shrinkage of a subject's covariance toward a population mean, transplanted from diffusion-MRI statistics | Null on IV-2a (8 reference subjects, too few to estimate between-subject variance in 253 dims); still null at OpenBMI scale (20 external subjects, properly estimated) — with large *opposite-signed* per-subject effects that cancel in the average |
| Eigenvoice extension (`phase2c_*`) | Anisotropic, PCA-subspace shrinkage | Clearly *worse* than isotropic at both k=3 and k=7 — more parameters didn't help, and k=3 doing as badly as k=7 rules out overfitting the small pool |
| Module A (`src/adaptation/geometric_correction.py`) | Small MLP learned to predict a subject's full-data Riemannian mean from a noisy few-trial subsample; distilled from Phase 2 | Passed the synthetic ground-truth gate (recovers known covariances ~2.5–3× better than raw), then **failed catastrophically** in cross-dataset transfer: 3–15× *further* from a held-out gold covariance than doing nothing. Root-caused with three independent controls, including direct measurement that PhysioNet's population covariance *shape* is about as far from a given IV-2a subject as two IV-2a subjects are from each other |
| Same, safety-bounded | Sigmoid-gated interpolation `(1−α)·raw + α·learned`, α ∈ [0,1], init ≈ 0 | Bounded the damage from "3–15× worse" to "~1.5–5× worse for 2/3 subjects" — better *safety*, still not net positive |
| Prototypical network, cross-subject episodes (Phase 6, attempt 1) | Meta-learning following the published MetaWearS design (disjoint support/query subjects) | Stalled completely on real data — even *training convergence* broke, not just transfer |
| Prototypical network, same-subject episodes (Phase 6, attempt 2) | Population data trains a general *skill*; support and query always come from one subject's own trials, matching the real calibration → eval task exactly | **49.0%**, converging fine, with 4/9 subjects at 59–67% — competitive with CSP+LDA. The fix was an episode-design insight, not a modelling one |

The pattern is consistent: methods that *avoid* cross-subject comparison — even while
using population data extensively — work. Riemannian alignment and wavelet calibration use
the target session only. The fixed prototypical network uses a 108-subject pool to teach a
skill, but its final decision is always grounded in the target subject's own labels.

The practical version of this rule, for anyone designing the next method: **ask whether any
single computation (a loss, a correction, a distance) ever requires directly comparing two
different subjects' data. If yes, expect it to struggle regardless of how sophisticated
the method otherwise is.**

## What's in this repository

```
src/                      Reusable library (imported by every experiment)
  pipeline.py             The R11 decoder as one entry point: fit_predict() with R11 / R10 / LITE configs
  config.py               Repo paths + shared constants; MNE/MOABB cache-path fix
  datasets.py             MOABB loaders: IV-2a, IV-2b, OpenBMI, PhysioNet
  gpu_health.py           Waits for a healthy GPU before each fit (long unattended runs)
  preprocessing.py        Reusable artifact-cleaning pipeline (notch, bandpass, ICA EOG/muscle)
  baselines.py            CSP+LDA reference pipeline
  evaluate.py             Within/cross-session protocols, Wilson CIs, exact McNemar
  synthetic_eeg.py        Ground-truth ERD/ERS generator (the Stage 0 validation gate)
  adaptation/             Every adaptation/personalization method (see below)
experiments/              One script per phase. `stage0_*` = synthetic ground-truth gates;
                          r3_..r14_ = pre-registered screens; d6_ = diagnostics; l1_/l2_ = lockbox;
                          b1_ = calibration burden; x1_ = cross-paradigm
results/                  Committed CSVs, one per experiment run (+ smoketests/ scratch)
tests/                    Unit tests for the protocol maths, loaders, generator and pipeline
docs/                     Literature survey, execution log, gate/screen log, script index
reports/                  Ideation rounds and the R14 council review
conftest.py               Makes `src` importable under pytest
```

`src/adaptation/` holds the methods: `riemannian_align.py` (the alignment every working pipeline
uses), `convnet.py` (ShallowConvNet + deterministic trainer), `augmentation.py` (FTSurrogate and
the other screened augmentations),
`spd_shrinkage.py`, `geometric_correction.py`, `prototypical_network.py`,
`wavelet_calibration.py`, `active_calibration.py`, `coherent_point_drift.py`,
`combat_align.py`, `riemannian_icp.py`, `latency_alignment.py`, `flow_field.py`,
`state_space_flow.py`.

Every experiment script is runnable standalone, prints per-subject progress, and writes
its CSV into `results/`.

### Read the work in this order

| Document | What it is |
|---|---|
| [`docs/progress_and_direction.md`](docs/progress_and_direction.md) | **Start here.** The execution log and decision history: what was tried, what worked, what didn't, and why. Contains the single-source-of-truth results table and the methodology rules |
| [`docs/diag_gates_d1_d4.md`](docs/diag_gates_d1_d4.md) | The log from R4 on: gating diagnostics, every pre-registered screen (R5–R14), the IV-2b eye-movement diagnosis (D6), the lockbox confirmations (L1, L2), the calibration-burden curve (B1) and the cross-paradigm study (X1) |
| [`reports/r14_council_and_ideation.md`](reports/r14_council_and_ideation.md) | Four-reviewer critique of R3–R13: shrunk effect sizes, simplification ablations, literature position, next moves |
| [`docs/state_of_the_art.md`](docs/state_of_the_art.md) | Literature survey: the leak-checked accuracy to beat (§0), published baselines, Riemannian alignment prior art, what is genuinely still open, and cross-disciplinary transplant ideas with verdicts |
| [`docs/wavelet_personalization_pipeline.md`](docs/wavelet_personalization_pipeline.md) | Design doc for the planned fingerprint-conditioned pipeline (learned geometric correction + personalized wavelet front end), including its risks |
| [`docs/experiments.md`](docs/experiments.md) | Index of all 43 experiment scripts: what each does, what it writes, roughly how long it takes |
| [`results/README.md`](results/README.md) | Provenance of every committed CSV — including where an artifact is missing or was reconstructed |

## Setup

Requires **Python 3.12**.

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt           # Windows
# source .venv/bin/activate && pip install -r requirements.txt    # Linux/macOS
.venv\Scripts\python -m pip install -r requirements-dev.txt       # optional: pytest
```

`requirements.txt` is pinned to the exact versions this work was produced and validated
with. Two things worth knowing before you install:

- **`torch`**: plain PyPI gives you the CPU build. This work used the CUDA 12.4 build; for
  GPU, install torch from the PyTorch index first (see the comment in `requirements.txt`).
- **`braindecode` (in `requirements-dev.txt`) is required for R11 and every result from R8 on**,
  because the FTSurrogate augmentation comes from it. It only imports when `torchaudio` matches
  torch exactly: with torch 2.6.0+cu124, install `torchaudio==2.6.0` from the PyTorch cu124 index
  with `--no-deps` *before* the dev file (the command is in `requirements-dev.txt`). Our own
  `ShallowConvNet` is tested to give identical outputs to braindecode's `ShallowFBCSPNet` under the
  same weights.
- **A CUDA GPU** is needed for the CNN experiments; the ShallowConvNet fits take seconds to a
  minute each. Training is deterministic, and reruns are byte-identical on the same machine.

Verify the install with the cheapest assertion-gated check, which needs no dataset and no
download:

```bash
.venv\Scripts\python experiments\stage0_icp_validation.py     # ~3 s, exits non-zero on failure
.venv\Scripts\python -m pytest -q                             # ~10 s, unit tests for the loaders/protocols/generator
```

Two runtime rules that will otherwise cost you a confusing debugging session:

- **Always run scripts from the repository root.** `src/config.py` deliberately sets
  `MNE_DATA` to a *cwd-relative* path to work around a MOABB/MNE bug on Windows (see
  below); launching a script from another directory silently sends the dataset cache
  somewhere else.
- The first run of most `phase*` scripts downloads data (see below).

## Data

Nothing is downloaded until an experiment needs it; MOABB/MNE caches everything under
`data/`, which is **gitignored** (it is multi-gigabyte).

| Dataset | Role | Size |
|---|---|---|
| BCI Competition IV-2a (`BNCI2014_001`) | Primary target: 9 subjects, 4-class MI, 2 sessions | ~300 MB |
| BCI Competition IV-2b (`BNCI2014_004`) | Generalization check: 9 subjects, 2-class, 3 channels | ~300 MB |
| OpenBMI / Lee2019 (`Lee2019_MI`) | External population pool (20 of 54 subjects used) | ~600 MB/subject |
| PhysioNet MI (`PhysionetMI`) | Main population pool (108 subjects), reuse for Phase 2d/Module A/Phase 6 | ~2–4 GB total |
| BNCI2015_001 | **Lockbox** (L1, L2): 12 subjects, right hand vs feet, 2–3 sessions | ~1 GB |
| GuttmannFlury2025 (P300 + SSVEP) | Cross-paradigm study (X1), the 17 subjects with ≥ 2 sessions | ~35 GB (archives include eye-tracking and video) |

```bash
.venv\Scripts\python experiments\prefetch_physionet.py   # optional: pre-fetch + validate the 108-subject pool
```

Two data-handling details are load-bearing and already handled in code:

- **Subject 88 of PhysioNet is excluded** — it was recorded at 128 Hz while every other
  subject is at 160 Hz, so mixing it in silently corrupts any sampling-rate-dependent
  processing (`src/datasets.py`).
- **MOABB sanitizes `:` out of download paths**, which corrupts Windows absolute paths
  (`C:\...`) and makes downloads land nested under the process cwd. `src/config.py`
  sidesteps this by setting a cwd-relative `MNE_DATA`, which keeps the cache inside
  `data/`. (An early run still left an ignored `C-/` artifact directory behind; it is safe
  to delete.)
- **GuttmannFlury2025, subject 12, SSVEP:** the hosted archive `S12.zip` indexes its first
  session before the start of the file (its md5 matches Zenodo), so that session is unreadable.
  X1 excludes subject 12 from SSVEP. Separately, MOABB's loader for this dataset's P300 speller
  yields labels with no P300 in them, even within one session (X1b).

## Reproducing the results

Run everything from the repository root. Each script re-derives its own numbers end to end
and overwrites its CSV in `results/` (provenance for every file, including the gaps, is in
[`results/README.md`](results/README.md)); [`docs/experiments.md`](docs/experiments.md)
indexes all 43 scripts. The commands below cover the headline numbers; almost all of them
want the datasets cached first, and runtimes are rough.

```bash
# Baseline and the one method that works
.venv\Scripts\python experiments\phase1_baseline.py                       # 62.2%
.venv\Scripts\python experiments\phase2_riemannian_alignment.py           # 66.9% (+4.7pp)

# The population-data negative result, at three scales / dimensionalities
.venv\Scripts\python experiments\phase2b_spd_shrinkage.py                 # IV-2a, 8 leave-one-out reference subjects
.venv\Scripts\python experiments\phase2b_spd_shrinkage_iv2b.py            # IV-2b (6 tangent dims) - dimensionality check
.venv\Scripts\python experiments\phase2d_openbmi_shrinkage.py --n-subjects 20
.venv\Scripts\python experiments\phase2b_spd_shrinkage.py --eigen-components 3 --out-name phase2c_eigenspace_k3_iv2a.csv
.venv\Scripts\python experiments\phase2b_spd_shrinkage.py --eigen-components 7 --out-name phase2c_eigenspace_k7_iv2a.csv

# Calibration-free / per-session-only alternatives
.venv\Scripts\python experiments\phase4b_wavelet_per_channel.py           # 31.4%
.venv\Scripts\python experiments\phase6_prototypical_meta_learning.py     # 49.0% (hours: needs the 108-subject PhysioNet pool)

# Other closed-out directions
.venv\Scripts\python experiments\phase5_combined_features.py              # naive feature concatenation hurts CSP
.venv\Scripts\python experiments\phase7_riemannian_icp.py                 # ICP: real-data null
.venv\Scripts\python experiments\phase9_flow_field_features.py            # weak standalone signal
.venv\Scripts\python experiments\cleaned_riemannian_check.py              # artifact-cleaned pipeline: worse (uses the data/cleaned/ cache)

# Cross-disciplinary idea tests
.venv\Scripts\python experiments\idea10_4_active_calibration.py            # real but narrow effect
.venv\Scripts\python experiments\idea10_5_closure_amplitude_check.py       # invariance real, wrong problem (seconds)

# The current best (R11) and its confirmations (GPU)
.venv\Scripts\python experiments\check_pipeline_reproduces_r11.py          # retrains R11 via src/pipeline.py, checks it bit for bit (~30 min)
.venv\Scripts\python experiments\compare_r11_vs_csp_lda.py                 # R11 vs CSP+LDA+RA, paired per subject (CPU)
.venv\Scripts\python experiments\r14_eog_gate.py                           # EOG-regressed retrain of every member (~40 min)
.venv\Scripts\python experiments\d6_iv2b_gap.py                            # why published IV-2b numbers are higher: eye movement (~1 h)
.venv\Scripts\python experiments\b1_r11_calibration_burden.py              # accuracy vs labelled trials, IV-2a + IV-2b (~1.5 h)
.venv\Scripts\python experiments\l1_lockbox_bnci2015_001.py                # lockbox: R11 on BNCI2015_001 (~40 min)
.venv\Scripts\python experiments\l2_lockbox_burden.py                      # lockbox: calibration burden (~1 h)
.venv\Scripts\python experiments\x1_cross_paradigm_guttmann.py             # SSVEP/P300 session transfer (CPU, ~35 GB download)
```

Using the decoder directly:

```python
from src.pipeline import R11, fit_predict, load_subject_data

data, n_classes = load_subject_data("IV-2a", subject=1)   # or "IV-2b"
probs = fit_predict(data, n_classes, R11)["ensemble"]       # test-session class probabilities
```

Synthetic ground-truth gates (`experiments/stage0_*.py`) need **no dataset download** and
are the cheapest way to see the machinery work. The project's own rule is that nothing
touches real data before it passes one of these — that discipline is what caught most of
the real bugs documented in the execution log.

## Methodology rules this project holds itself to

From [`docs/progress_and_direction.md`](docs/progress_and_direction.md#methodology-lessons-apply-these-before-trusting-any-new-result),
applied before trusting any new number:

1. **Validate on synthetic ground truth before touching real data.** Every real-data bug
   caught cheaply was caught this way. It is the mechanism that makes the negative results
   here trustworthy rather than merely discouraging.
2. **Scale/normalization confounds are the most common bug class.** Three separate scoring
   bugs were variants of one root issue: a metric that looks reasonable but secretly
   depends on something other than true match quality.
3. **A higher-capacity method is not automatically better when reference data is small.**
   Prefer methods with a built-in "do-nothing" floor (bounded interpolation, not
   unconstrained extrapolation).
4. **A synthetic-data win doesn't transfer if the fix also costs sample efficiency.**
   Check the samples-per-feature ratio, not just whether the synthetic gate passed.
5. **In-sample checks are cheap and catch scoring-rule bugs** before you blame
   generalization.
6. **CSP+LDA is a genuinely strong, hard-to-beat baseline** for this task family. It took a
   diverse convnet + Riemannian ensemble to beat it convincingly (R11, +9.3 pp over 18 subjects).
7. **Screen on two datasets at once.** Two methods that won on IV-2a alone (self-training,
   FTSurrogate as a replacement) shrank to near zero on IV-2b. This is the winner's curse.
8. **Check for eye movements.** On IV-2a/2b the three EOG channels alone decode the class far
   above chance (a CNN reaches 52% on 4 classes and 70% on 2). Any low-frequency or frontal
   feature needs an EOG control.

## Status

**Classical baseline:** CSP+LDA + Riemannian alignment, 66.9% cross-session (IV-2a).

**Best: the R11 ensemble, confirmed on the lockbox.**
- It reaches 82.5% on IV-2a and 80.9% on IV-2b, and beats CSP+LDA+RA by +9.3 pp pooled.
- It is one entry point, [`src/pipeline.py`](src/pipeline.py), and a full retrain through it
  reproduces R11 exactly.
- How it was reached ([`docs/diag_gates_d1_d4.md`](docs/diag_gates_d1_d4.md)):
  - pre-registered screens R3–R13, scored jointly on IV-2a and IV-2b from R10 onward;
  - an EOG validity gate (R14), which passed: removing eye signal costs 0.75 pp at most;
  - a four-reviewer council ([`reports/r14_council_and_ideation.md`](reports/r14_council_and_ideation.md)).
- The idea loop stopped after two rounds without gain (R12, R13). More screening on these 18
  evaluation sessions would mostly fit noise.

**Confirmed on unseen data (L1 lockbox, BNCI2015_001).** The test was pre-registered before
download, and the data were opened once (12 subjects, right hand vs feet):
- R11 reaches 87.9%, against 84.2% for a single ShallowConvNet (+3.7 pp, 10/12 up, Holm p = 0.002).
- It beats CSP+LDA+RA (77.0%) by +10.9 pp, with all 12 subjects up.
- Its edge over a same-size CNN ensemble (+1.3 pp) has the predicted size but is not significant
  with 12 subjects.
- The 1-fit LITE variant is not a substitute (−2.5 pp).

**Beyond motor imagery (X1, GuttmannFlury2025).**
- In SSVEP, both recipe ingredients help a trained decoder transfer between sessions:
  - per-session alignment adds +9.8 pp;
  - a diverse second model adds +5.0 pp.
- A training-free CCA decoder (84%) still beats every trained SSVEP pipeline.
- The dataset's P300 recordings are not decodable through MOABB's loader, even within one session, so
  P300 is untested.

**Calibration burden (B1).** The project's goal is fewer calibration trials. With only **144
labelled trials**, R11 matches a single ShallowConvNet trained on the full session (+0.6 pp,
non-inferior): about half the labels on IV-2a (288) and a third on IV-2b (~420). The pre-registered
96-trial target was missed by 2.2 pp. R11's lead over a single CNN grows as labels get scarce:
+4.8 pp with the full session, +9.4 pp with 48 trials, and every subject improves at both ends.

**Burden claim confirmed on the lockbox (L2, BNCI2015_001), pre-registered.** R11 trained on
**half** the calibration session (100 of 200 trials) matches a single ShallowConvNet trained on all
200 (84.6% vs 84.2%, non-inferior). With 144 trials it beats it (+2.7 pp). At every budget R11 beats
a single CNN trained on the same trials, and at 48, 100 and 144 trials every subject improves.

**Faster training (F1).** `PipelineConfig(fast=True)` trains R11's five ShallowConvNets as one stacked
model. Accuracy is unchanged on all 18 subjects (0.00 pp; 99.99% of member predictions identical), and it
is 4.1× faster on IV-2a (2.2 → 0.5 min per user). IV-2b, with 3 channels, gains only 1.15×. The default
stays the exact path, so every cached result still reproduces.

**Known limits:**
- R11's alignment uses the whole unlabelled test session up front, so it is not yet usable live, one
  trial at a time.
- It costs 5 small GPU fits per user (about half a minute with `fast=True`).
- On IV-2a it is 2.3 pp below the best published leak-free result.
- P300 transfer is untested, because of the dataset loader problem above.

**Next, in order:**
1. ~~Faster training~~ done (F1); mixed precision gave no gain on this GPU.
2. A different architecture (TCFormer / EEGNet / CTNet) as an ensemble member instead of extra
   ShallowConvNet seeds. This is the most likely accuracy gain; it will be pre-registered and checked
   on the lockbox.
3. Causal (online) alignment, so the decoder works live.
4. Optionally, a port of the recipe to the FALCON intracortical benchmark.

**Deliberately closed:** population-based covariance correction (four independent
failures); naive feature concatenation (actively harmful); dimensionality reduction on
windowed features (decisively worse).

**Checked and ruled out by the literature:** joint-embedding / JEPA-style EEG foundation
models for motor imagery specifically. S-JEPA reports 65% on MI vs. Riemannian SOTA's
84.7%; under frozen linear probing, LaBraM, LUNA, CBraMod and Laya all sit at 0.47–0.51
balanced accuracy on 2-class left-vs-right-hand MI — at or barely above chance. Building
on that stack would likely regress from the classical Riemannian pipeline, not improve it.

Untried-but-identified directions, open questions, and the artifact-cleaned data pipeline's
mixed first validation are all logged in
[`docs/progress_and_direction.md`](docs/progress_and_direction.md).

A prioritised list of everything still untested — including cross-disciplinary method
transplants, the per-subject failure analysis the mean has been hiding, and what "beating
66.9%" would have to mean — is in [`TODO.md`](TODO.md).

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

Datasets: BCI Competition IV (Graz University of Technology), OpenBMI/Lee2019 (GigaScience),
and PhysioNet EEG Motor Movement/Imagery. Tooling rests on
[MOABB](https://github.com/NeuroTechX/moabb), [MNE-Python](https://mne.tools),
[pyriemann](https://github.com/pyRiemann/pyRiemann) and scikit-learn. Literature references
with DOIs are listed inline in [`docs/state_of_the_art.md`](docs/state_of_the_art.md).
