# zero-shot-bci

**Toward calibration-free cross-session EEG decoding.** Research code, results, and a
full negative-results log for closing the calibration → live-session accuracy gap in
motor-imagery brain–computer interfaces (BCIs).

The name states the goal, not the current state. The endpoint is a decoder that works on
a new session — ideally a new user — with **no labelled calibration data at all**. What
this repository contains is the measured path toward it: what works, what provably does
not, and the evidence for each. The best result so far (66.9% cross-session on 4-class
BCI Competition IV-2a) still requires a calibration session; the calibration-free
alternatives are far behind and are documented as such.

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
evaluation session, 9 subjects, 4 classes, chance = 25.0%. The population-data setting is
deliberately hard — a method gets *one subject's own 288 calibration trials*, nothing else
— because that matches "the user just sat down."

## Headline results

| Method | Cross-session accuracy | Δ vs. baseline | Source |
|---|---|---|---|
| **CSP+LDA + Riemannian alignment** (Phase 2) — best overall | **66.9%** | +4.7pp | [`results/phase2_riemannian_alignment_iv2a.csv`](results/phase2_riemannian_alignment_iv2a.csv) |
| CSP+LDA, no adaptation (Phase 1) | 62.2% | — | [`results/phase1_baseline_iv2a.csv`](results/phase1_baseline_iv2a.csv) |
| Prototypical-network meta-learning (Phase 6) | 49.0% (std 16.1) | −13.2pp | [`results/phase6_prototypical.csv`](results/phase6_prototypical.csv) |
| Wavelet per-session calibration (Phase 4b) | 31.4% | −30.8pp | [`results/phase4b_wavelet_per_channel.csv`](results/phase4b_wavelet_per_channel.csv) |
| Chance | 25.0% | — | — |

Riemannian alignment (per-session recentering — whitening each session's own trials to
identity) is the one adaptation method that reliably helps, and it needs **no other
subject's data**. It is statistically significant for 3/9 subjects individually
(McNemar). Every method that borrows another subject's data to correct this subject's
estimate has failed. The full, always-current number table is the "Current numbers"
section of [`docs/progress_and_direction.md`](docs/progress_and_direction.md#current-numbers-single-source-of-truth---update-when-a-phase-reruns).

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
  config.py               Repo paths + shared constants; MNE/MOABB cache-path fix
  datasets.py             MOABB loaders: IV-2a, IV-2b, OpenBMI, PhysioNet
  preprocessing.py        Reusable artifact-cleaning pipeline (notch, bandpass, ICA EOG/muscle)
  baselines.py            CSP+LDA reference pipeline
  evaluate.py             Within/cross-session protocols, Wilson CIs, exact McNemar
  synthetic_eeg.py        Ground-truth ERD/ERS generator (the Stage 0 validation gate)
  adaptation/             Every adaptation/personalization method (see below)
experiments/              One script per phase. `stage0_*` = synthetic ground-truth gates
results/                  Committed CSVs, one per experiment run (+ smoketests/ scratch)
tests/                    Unit tests for the protocol maths, loaders and generator
docs/                     Literature survey, execution log, pipeline design doc, script index
conftest.py               Makes `src` importable under pytest
```

`src/adaptation/` holds the methods: `riemannian_align.py` (the winner),
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
| [`docs/state_of_the_art.md`](docs/state_of_the_art.md) | Literature survey (~950 lines): published baselines, Riemannian alignment prior art, what is genuinely still open, and five cross-disciplinary transplant ideas with verdicts on each |
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
- **`braindecode` is deliberately absent.** It was in the original working environment but
  is not imported by any code here, and version 1.7.0 currently fails to import against
  torch 2.6 (it needs a matching `torchaudio`). Don't add it without checking that first.

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
6. **CSP+LDA is a genuinely strong, hard-to-beat baseline** for this task family.

## Status

**Established:** CSP+LDA + Riemannian alignment, 66.9% cross-session.

**Most promising active direction:** prototypical-network meta-learning with same-subject
episodes (49.0%, four subjects at 59–67%). Open questions: the subject-5 chance-level
outlier, `episodes_per_step` tuning, and whether proper out-of-fold stacking with CSP now
adds value.

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
