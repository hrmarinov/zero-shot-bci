# zero-shot-bci

Research code for motor-imagery EEG decoding without a long calibration session.

A brain–computer interface (BCI) decodes imagined movement from EEG. A classifier that works in
one session loses accuracy in the next session. This repository measures how to reduce that loss.
It contains the code, the results, and a log of the methods that failed.

The goal is a decoder that needs **no labelled calibration data** for a new session or a new user.
This goal is not reached. The repository shows the measured path toward it.

## Results

The best decoder is **R11** ([`src/pipeline.py`](src/pipeline.py)). It averages the output of
three models. All three use input that is aligned per session (Riemannian alignment):

- ShallowConvNets
- ShallowConvNets trained with FTSurrogate augmentation
- tangent-space logistic regression

| Dataset | Task | R11 | CSP+LDA with alignment | Chance |
|---|---|---|---|---|
| BCI Competition IV-2a (9 subjects) | 4 classes | 82.5% | 66.9% | 25% |
| BCI Competition IV-2b (9 subjects) | 2 classes | 80.9% | 77.7% | 50% |
| Lee2019 (54 subjects) | 2 classes | 74.5% | — | 50% |
| BNCI2015_001 (12 subjects, lockbox) | 2 classes | 87.9% | 77.0% | 50% |

Train on the calibration session. Test on the evaluation session. The method never uses
evaluation labels for fitting, model selection or early stopping.

- **Lockbox.** BNCI2015_001 was opened once, after pre-registration. A single ShallowConvNet
  reaches 84.2% there. R11 gains 3.7 pp, which matches the shrunk estimate of 3.5 pp from the
  development data.
- **Fewer labels.** With 100 of 200 labelled trials, R11 equals a single ShallowConvNet that uses
  all 200 trials. If the networks are first trained on other subjects, 48 trials are enough.
- **Live use.** With causal alignment, R11 classifies each trial from that trial and earlier ones.
  The cost is 0.1–0.3 pp. A new user needs about 30 seconds of GPU training.
- **Still required.** A short calibration session. The methods without calibration are far less
  accurate.

### What to trust

R11 is the end of a long search on the same evaluation sessions, so its development gains are
inflated. A four-reviewer review
([`reports/r14_council_and_ideation.md`](reports/r14_council_and_ideation.md)) estimates the true
gain at about +3.5 pp over one ShallowConvNet. The lockbox result agrees with this.

The published TCFormer reaches 84.8% on IV-2a, which is 2.3 pp above R11. We ran its recipe on IV-2a (T1)
and reproduced it: 84.4%. With the EOG channels removed it scores 83.9%, which is 1.4 pp above R11. Only 5
of 9 subjects are up, so the pre-registered test is inconclusive. Unlike IV-2b, the IV-2a result is not
mostly eye movement (0.5 pp). See [`docs/diag_gates_d1_d4.md`](docs/diag_gates_d1_d4.md).

### Eye movement in IV-2b

Published IV-2b results use 0.5–100 Hz input. The three EOG channels alone decode the class at
70% (chance 50%). When we move our ShallowConvNet to broadband input, it gains 4.4 pp. Removing the
EOG signal removes the whole gain (experiment D6). Published IV-2b numbers therefore probably
contain 4–5 pp of eye-movement decoding. Our 4–38 Hz input keeps this share near 1 pp (R14).

Always test low-frequency and frontal features with an EOG control.

## What failed

Every method that compares or aligns the covariance of **two different subjects** failed. This
holds even for methods that use population data. Six attempts are logged, from covariance
shrinkage to meta-learning. Methods that use only the target session work. Riemannian alignment is
the main example, and it needs no data from other subjects.

Use this rule for new methods: if one computation (a loss, a correction or a distance) needs two
subjects' data side by side, expect it to fail.

Details: [`docs/diag_gates_d1_d4.md`](docs/diag_gates_d1_d4.md) and
[`docs/progress_and_direction.md`](docs/progress_and_direction.md).

## Repository layout

```
src/            Library code used by every experiment
  pipeline.py     R11 decoder: fit_predict() with the R11, R10, LITE and R11_PT configs
  live.py         LiveDecoder: calibrate once, then decode one trial at a time
  pretrain.py     Pretraining on other subjects, and zero-shot prediction
  config.py       Paths and shared constants
  datasets.py     MOABB loaders for IV-2a, IV-2b, OpenBMI and PhysioNet
  gpu_health.py   Waits for a healthy GPU before each fit
  preprocessing.py, baselines.py, evaluate.py, synthetic_eeg.py
  adaptation/     All adaptation methods, including the failed ones
  third_party/    Vendored TCFormer model (MIT)
experiments/    One script per experiment (see docs/experiments.md)
results/        One CSV per experiment run (provenance: results/README.md)
tests/          Unit tests
docs/           Literature survey, execution log, gate log, project audit
reports/        Ideation rounds and the council review
```

Prefixes in `experiments/`: `stage0_` = synthetic-data checks, `r3_`…`r14_` = pre-registered
screens, `d` = diagnostics, `l1_`/`l2_` = lockbox, `b1_` = calibration burden, `x1_` =
cross-paradigm, `t1_` = TCFormer head-to-head.

### Documents

| Document | Content |
|---|---|
| [`docs/progress_and_direction.md`](docs/progress_and_direction.md) | **Start here.** Decision history and the table of current numbers |
| [`docs/diag_gates_d1_d4.md`](docs/diag_gates_d1_d4.md) | Every diagnostic, screen and confirmation since R4 |
| [`docs/project_audit.md`](docs/project_audit.md) | Claims against evidence, protocol weaknesses and open tests |
| [`reports/r14_council_and_ideation.md`](reports/r14_council_and_ideation.md) | Four-reviewer critique of R3–R13 |
| [`docs/state_of_the_art.md`](docs/state_of_the_art.md) | Literature survey and the published results to beat |
| [`docs/experiments.md`](docs/experiments.md) | Index of all experiment scripts |
| [`results/README.md`](results/README.md) | Provenance of every committed CSV |
| [`TODO.md`](TODO.md) | Ranked list of untested ideas |

## Setup

Python 3.12 is required.

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-dev.txt   # optional: pytest and braindecode
```

On Linux or macOS, use `source .venv/bin/activate && pip install -r requirements.txt`.

- **torch.** PyPI gives the CPU build. This work used torch 2.6.0 with CUDA 12.4. For a GPU, install
  torch from the PyTorch index first (see the comment in `requirements.txt`).
- **braindecode.** R11 and every result from R8 onward need it for FTSurrogate. It imports only if
  `torchaudio` matches torch exactly. The install command is in `requirements-dev.txt`.
- **GPU.** The CNN experiments need a CUDA GPU. One fit takes seconds to one minute. Training is
  deterministic, so a rerun on the same machine gives identical results.

Check the install. Neither command downloads data:

```bash
.venv\Scripts\python experiments\stage0_icp_validation.py   # about 3 s
.venv\Scripts\python -m pytest -q                           # about 10 s
```

**Run all scripts from the repository root.** `src/config.py` sets `MNE_DATA` to a path relative to
the working directory. This avoids a MOABB bug on Windows. A script started elsewhere saves its
data cache in the wrong place.

## Data

MOABB and MNE download each dataset when an experiment first needs it. Everything is saved in
`data/`, which git ignores.

| Dataset | Use | Size |
|---|---|---|
| BCI Competition IV-2a (`BNCI2014_001`) | Main target: 9 subjects, 4 classes, 2 sessions | ~300 MB |
| BCI Competition IV-2b (`BNCI2014_004`) | Second check: 9 subjects, 2 classes, 3 channels | ~300 MB |
| OpenBMI / Lee2019 (`Lee2019_MI`) | 54-subject development cohort and external pool | ~600 MB per subject |
| PhysioNet MI (`PhysionetMI`) | Pretraining pool of 108 subjects | 2–4 GB |
| BNCI2015_001 | Lockbox: 12 subjects, right hand against feet | ~1 GB |
| GuttmannFlury2025 | Cross-paradigm study (X1): SSVEP and P300 | ~35 GB |

The code handles these data problems:

- PhysioNet subject 88 is excluded. It was recorded at 128 Hz and all other subjects at 160 Hz.
- MOABB removes `:` from download paths. This breaks Windows paths such as `C:\...`. The
  relative `MNE_DATA` path avoids this. An early run left an ignored `C-/` directory. You can
  delete it.
- GuttmannFlury2025, SSVEP subject 12: the archive `S12.zip` has a damaged index for its first
  session. X1 excludes this subject.
- MOABB's P300 loader for GuttmannFlury2025 returns labels with no P300 in them. P300 transfer is
  therefore untested.

## Reproduce

Run from the repository root. Each script recomputes its numbers and overwrites its CSV in
`results/`. Most scripts need the datasets cached first. Times are approximate.

```bash
# Baseline and alignment
.venv\Scripts\python experiments\phase1_baseline.py                  # 62.2%
.venv\Scripts\python experiments\phase2_riemannian_alignment.py      # 66.9%

# Cross-subject methods that failed
.venv\Scripts\python experiments\phase2b_spd_shrinkage.py
.venv\Scripts\python experiments\phase2d_openbmi_shrinkage.py --n-subjects 20
.venv\Scripts\python experiments\phase6_prototypical_meta_learning.py   # hours: needs PhysioNet

# R11 and its confirmations (GPU)
.venv\Scripts\python experiments\check_pipeline_reproduces_r11.py    # bit-exact retrain, ~30 min
.venv\Scripts\python experiments\compare_r11_vs_csp_lda.py           # paired comparison, CPU
.venv\Scripts\python experiments\r14_eog_gate.py                     # EOG regressed out, ~40 min
.venv\Scripts\python experiments\d6_iv2b_gap.py                      # eye movement in IV-2b, ~1 h
.venv\Scripts\python experiments\b1_r11_calibration_burden.py        # accuracy against label count, ~1.5 h
.venv\Scripts\python experiments\l1_lockbox_bnci2015_001.py          # lockbox, ~40 min
.venv\Scripts\python experiments\l2_lockbox_burden.py                # lockbox burden, ~1 h
.venv\Scripts\python experiments\x1_cross_paradigm_guttmann.py       # CPU, ~35 GB download
```

[`docs/experiments.md`](docs/experiments.md) lists the other scripts.

Use the decoder directly:

```python
from src.pipeline import R11, fit_predict, load_subject_data

data, n_classes = load_subject_data("IV-2a", subject=1)   # or "IV-2b"
probs = fit_predict(data, n_classes, R11)["ensemble"]     # class probabilities for the test session
```

## Method rules

1. Test every method on synthetic data with known ground truth before you use real data. This
   found most of our bugs (`experiments/stage0_*.py`, no download needed).
2. Check for scale and normalization confounds. Three scoring bugs had this one cause.
3. With small reference data, prefer a method that can do nothing (for example a bounded
   interpolation) over one that can extrapolate freely.
4. Screen on two datasets. Two methods that won on IV-2a alone gained almost nothing on IV-2b.
5. Use CSP+LDA as the baseline. It is hard to beat.
6. Run an EOG control (see above).

## Status

**Done and confirmed**

- R11 on IV-2a, IV-2b and the lockbox (see Results).
- Calibration burden (B1, L2): 100 of 200 trials match one CNN trained on all 200.
- Population pretraining (P1): 48 trials match one CNN trained on all 200 (lockbox). It brings
  no gain with the full session. Without any calibration, the pretrained networks reach 77% on
  IV-2b, 65% on the lockbox and 55% on IV-2a.
- Fast training (F1): five networks as one stacked model. Accuracy is identical. IV-2a trains 4.1×
  faster.
- Causal alignment (C1) and a live decoder ([`src/live.py`](src/live.py)). The decoder takes
  about 6 ms per trial and matches causal R11 on all 18 recorded sessions.
- 54-subject cohort (D2). Weak-subject diagnosis (D3): R11 is at the within-session ceiling on
  average. D4: the ceiling depends on the number of same-session trials. E1: surface Laplacian
  and 62 channels do not help.
- SSVEP transfer (X1): alignment adds 9.8 pp and a second model type adds 5.0 pp. A CCA decoder
  without training (84%) beats every trained pipeline.

**Not confirmed:** adding CTNet as a fourth model family (A1). It passed the development screen
(+0.9 pp) and failed on the lockbox (+0.8 pp, 6 of 12 subjects up).

**Closed:** population-based covariance correction, naive feature concatenation, dimensionality
reduction on windowed features, and EEG foundation models for motor imagery (published results are
near chance).

**Known limits**

- Live use was measured on recorded sessions. No closed-loop session was run.
- R11 needs 5 small GPU fits per user.
- On IV-2a, R11 is 2.3 pp below the best published leak-free result.
- The TCFormer head-to-head (T1) is inconclusive on IV-2a and not run on IV-2b or Lee2019.
- The 54-subject cohort has a minimum detectable effect of 1.9 pp. Smaller effects cannot be shown.

**Next**

1. Run T1 on IV-2b and Lee2019, and find which part of TCFormer's recipe helps subjects 2, 4 and 6.
2. Test variance reduction on Lee2019: a larger pretraining pool, L2-SP fine-tuning and
   per-subject member weights.
3. Optionally, port the recipe to the FALCON intracortical benchmark.

## Citation

See [`CITATION.cff`](CITATION.cff).

## AI assistance

Hristo Marinov ([ORCID 0009-0002-0542-8279](https://orcid.org/0009-0002-0542-8279)) designed the project, chose the protocols and takes responsibility for all claims.
Claude (Anthropic) helped write code, tests and documentation. Commits from 4–6 November 2026 carry
a `Co-Authored-By: Claude` trailer. The experiments are pre-registered and every result is in
`results/`, so you can check each number by rerunning the script.

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgements

Datasets: BCI Competition IV (Graz University of Technology), OpenBMI (GigaScience) and PhysioNet.
Tools: [MOABB](https://github.com/NeuroTechX/moabb), [MNE-Python](https://mne.tools),
[pyRiemann](https://github.com/pyRiemann/pyRiemann), [braindecode](https://braindecode.org) and
scikit-learn. Literature references are in [`docs/state_of_the_art.md`](docs/state_of_the_art.md).
