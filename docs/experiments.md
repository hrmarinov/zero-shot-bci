# Experiment scripts

All 43 scripts live in [`experiments/`](../experiments) and are run **from the
repository root** (e.g. `python experiments\phase1_baseline.py`) — each inserts
the repository root into `sys.path` itself, and `src/config.py` resolves the
dataset cache relative to the current working directory, so the root must be
the cwd.

Conventions:

- **`phase*`** — a numbered research phase. Most write a CSV into
  [`results/`](../results); see that directory's README for the artifact each
  one produces.
- **`stage0_*validation*`** — synthetic ground-truth gates. No dataset needed,
  run in seconds to minutes, pass/fail via exit code or printed verdict. The
  project's rule is that nothing touches real data before it passes one of
  these.
- **`diagnose_*`** — one-off investigations of a specific anomaly, usually
  synthetic-first, printing rather than saving.
- **`idea10_*`** — the cross-disciplinary transplant tests from
  [`docs/state_of_the_art.md`](state_of_the_art.md) §10.

Runtime classes below are from reading the code (loop bounds, epochs, repeats),
not from a timed run of every script on this machine.

## Baselines and the methods that work

| Script | Purpose | Writes |
|---|---|---|
| `phase1_baseline.py` | Within-session CV + cross-session CSP+LDA, no adaptation, per IV-2a subject (slow) | `phase1_baseline_iv2a.csv` |
| `phase2_riemannian_alignment.py` | Per-session Riemannian recentering vs Phase 1, paired McNemar (slow) | `phase2_riemannian_alignment_iv2a.csv` |
| `phase6_prototypical_meta_learning.py` | PhysioNet-meta-trained encoder; IV-2a prototypes from the subject's own calibration trials (slow) | `phase6_prototypical.csv` |

## Population-data personalization (the recurring negative result)

| Script | Purpose | Writes |
|---|---|---|
| `phase2b_spd_shrinkage.py` | Low-data eval-side reference sweep (`raw_ai`/`raw_le`/`shrunk`, optional `eigen`), leave-one-out prior. `--out-name`, `--eigen-components`, `--n-repeats`, `--sample-sizes` (slow) | `phase2b_spd_shrinkage_iv2a.csv` |
| `phase2b_spd_shrinkage_iv2b.py` | Same sweep on IV-2b (3 channels, 6 tangent dims) as a dimensionality test (slow) | `phase2b_spd_shrinkage_iv2b.csv` |
| `phase2d_openbmi_shrinkage.py` | Shrinkage with an external OpenBMI pool on the 21 shared channels (`--n-subjects`, `--out-name`) (slow) | `phase2d_openbmi20_shrinkage_iv2a.csv` |
| `phase3_module_a.py` | "Module A": PhysioNet-trained learned geometric correction, 4-way comparison on IV-2a (`--n-subjects`, `--out-name`) (slow) | `phase3_module_a_iv2a.csv` (not committed — see `results/README.md`) |
| `stage0_module_a_validation.py` | Gate: the network must beat raw baselines at n=8/16 on 10 held-out synthetic subjects (slow) | none (returns a DataFrame) |
| `diagnose_module_a_real.py` | Measures how far raw/learned few-trial covariances sit from a full-data gold covariance (`--n-reference-subjects`, `--n-eval-subjects`) (slow) | none |

## Calibration-free / per-session-only alternatives

| Script | Purpose | Writes |
|---|---|---|
| `phase4_wavelet_calibration.py` | Per-class wavelet templates + pooled `score_trials` nearest-template (slow) | `phase4_wavelet_calibration.csv` |
| `phase4b_wavelet_per_channel.py` | Per-channel, jitter-tolerant match-filter features + shrinkage LDA (slow) | `phase4b_wavelet_per_channel.csv` |
| `phase4c_wavelet_windowed.py` | Per-channel × 5-phase-window features (440-dim) + LDA (slow) | `phase4c_wavelet_windowed.csv` |
| `stage0_wavelet_calibration_validation.py` | Gate: recovers known onset/trough/frequency/topology from synthetic trials (slow) | none |
| `check_wavelet_calibration_real.py` | Prints fitted wavelet parameters for subjects 1–6, qualitative (slow) | none |
| `diagnose_wavelet_onset.py` | Fixed-10 Hz envelope inspection + onset-initialization sensitivity, subjects 1 and 3 | none |
| `diagnose_template_matching.py` | In-sample nearest-template vs matched-filter vs LDA-on-scores, subject 1 | none |
| `diagnose_jitter_hypothesis.py` | Synthetic jitter sweep (0.0/0.1/0.2/0.3 s), fixed vs jitter-tolerant features (slow) | none |
| `stage0_synthetic_validation.py` | 4 assert-gated checks of the synthetic generator itself (slow) | none |
| `idea10_4_active_calibration.py` | Uncertainty sampling vs random vs as-collected calibration reveal order (slow) | `idea10_4_active_calibration_iv2a.csv` |
| `idea10_5_closure_amplitude_check.py` | Closure-amplitude invariance under scalar gains vs full-matrix mixing (fast) | none |

## Other alignment, feature and combination attempts

| Script | Purpose | Writes |
|---|---|---|
| `phase5_combined_features.py` | CSP + wavelet + MiniRocket blocks, combined LDA and logistic regression (slow) | `phase5_combined_features.csv` |
| `phase7_riemannian_icp.py` | 5-condition ablation: none / recenter+rescale / +ICP / +rotation / +rotation+ICP (slow) | `phase7_riemannian_icp.csv` |
| `phase7c_combat.py` | ComBat harmonization vs no correction at n=288 (slow) | `phase7c_combat.csv` |
| `phase7d_pca_concat.py` | Full 253-dim vs PCA(d=20) vs concatenated (slow-moderate) | `phase7d_pca_concat.csv` |
| `phase8_motor_cortex_channels.py` | Full-22 vs central-17 vs narrow-9 channels, ± alignment (slow) | `phase8_motor_cortex_channels.csv` |
| `phase9_flow_field_features.py` | Per-trial Horn-Schunck scalp flow summary features + LDA (slow) | `phase9_flow_field_features.csv` |
| `phase9b_flow_field_plus_csp.py` | Flow features concatenated onto CSP (slow) | `phase9b_flow_field_plus_csp.csv` |
| `phase10_state_space_flow_frequency.py` | Per-trial local PCA(2) + skew-symmetric flow frequency, nearest-template (slow) | `phase10_state_space_flow_frequency.csv` |
| `stage0_icp_validation.py` | Gate: exact Procrustes recovery + toy point-cloud ICP (**fast, ~3 s**) | none |
| `stage0_icp_eeg_validation.py` | Gate: ICP on synthetic EEG drift with a known mixing matrix (moderate) | none |
| `stage0_flow_field_validation.py` | Gate: electrode projection topology + planted-translation recovery (fast) | none |
| `stage0_flow_field_eeg_validation.py` | Gate: full flow pipeline recovers 3 planted spatial drifts (slow) | none |
| `stage0_state_space_flow_validation.py` | Gate: plane recovery, debiased frequency, over-complete limitation, noise guard (moderate) | none |
| `stage0_state_space_flow_eeg_validation.py` | Gate: frequency vs plane-residual vs tangent-LDA under 8 planted drifts (slow) | none |
| `stage0_latency_alignment_validation.py` | Gate: zero-jitter control + jitter recovery, estimate-on-smoothed/apply-to-raw (slow) | none |
| `stage0_ride_validation.py` | Gate: RIDE 2-component separation vs naive Woody (moderate; prints its verdict rather than asserting) | none |
| `stage0_combat_validation.py` | Gate: ComBat vs recenter+rescale vs none under a heterogeneous D=253 batch effect (fast) | none |
| `stage0_cpd_validation.py` | Gate: benign parity vs ICP at D=10, outlier robustness, documented D=253 failure (fast-moderate) | none |
| `stage0_cpd_pca_validation.py` | Gate: PCA-subspace CPD lifted to D=253 vs no correction and full-D ICP (fast-moderate) | none |

## Pipeline checks and data prefetch

| Script | Purpose | Writes |
|---|---|---|
| `cleaned_riemannian_check.py` | CSP+LDA on the ICA-cleaned pipeline, before and after alignment (slow; uses the cache in `data/cleaned/`) | `cleaned_riemannian_check.csv` |
| `prefetch_physionet.py` | Downloads/validates all 108 usable PhysioNet subjects, logging per-subject failures (slow, network) | none (stdout; captured in the gitignored `physionet_prefetch.log`) |

## Known rough edges

Real, currently-unfixed issues in these scripts, recorded so a future reader
doesn't mistake them for intentional design:

- `phase4_wavelet_calibration.py`, `phase4b_wavelet_per_channel.py` and
  `phase4c_wavelet_windowed.py` hardcode `SFREQ = 250.0` instead of reading the
  sampling rate off the epoch object. Correct for IV-2a, but it will silently be
  wrong if the loader is ever pointed at another dataset.
- `phase7_riemannian_icp.py`, `phase7c_combat.py`, `phase7d_pca_concat.py` and
  most `phase4*`/`phase5`/`phase6`/`phase8`/`phase9*` scripts expose no CLI at
  all, so their conditions can only be changed by editing code. `phase2b`,
  `phase2d`, `phase3` and `diagnose_module_a_real.py` do have flags.
- `stage0_ride_validation.py` prints its conclusion instead of asserting, so a
  regression there is silent (unlike the other `stage0_*` gates).
- Several `stage0_*` and `diagnose_*` scripts import private helpers across
  modules (`src.synthetic_eeg._band_limited_carrier`, `_pink_noise`,
  `_MONTAGE_POSITIONS`; `src.adaptation.spd_shrinkage._trial_tangent_vectors`),
  so renaming a private helper can break a script outside its module.
- `whiten()` and a few small helpers are re-implemented locally in several
  scripts (`phase2b_spd_shrinkage.py`, `phase2d_openbmi_shrinkage.py`,
  `phase3_module_a.py`) rather than shared.
