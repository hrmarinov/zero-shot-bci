# `results/` — committed artifacts and their provenance

Every CSV here is a raw output of one script, committed so the numbers quoted in
the [README](../README.md) and in [`docs/progress_and_direction.md`](../docs/progress_and_direction.md)
can be inspected without re-running anything. All cross-session numbers are
9 IV-2a subjects, 4 classes, chance = 25.0%, unless noted.

Re-running a script overwrites its file. Runtime classes: *fast* = seconds to a
minute, *slow* = tens of minutes or more.

## Headline numbers

| File | Produced by | Headline result | Runtime |
|---|---|---|---|
| `phase1_baseline_iv2a.csv` | `experiments/phase1_baseline.py` | CSP+LDA, no adaptation: **62.2%** cross-session (plus within-session CV and per-subject Wilson CIs; final row is the average) | slow |
| `phase2_riemannian_alignment_iv2a.csv` | `experiments/phase2_riemannian_alignment.py` | + Riemannian alignment: **66.9%** (+4.7pp), significant for 3/9 subjects (McNemar) | slow |
| `phase6_prototypical.csv` | `experiments/phase6_prototypical_meta_learning.py` | Prototypical-network meta-learning, same-subject episodes: **49.0%** (std 16.1) | slow |
| `phase4b_wavelet_per_channel.csv` | `experiments/phase4b_wavelet_per_channel.py` | Per-session wavelet calibration, per-channel + jitter-tolerant: **31.4%** | slow |
| `phase4_wavelet_calibration.csv` | `experiments/phase4_wavelet_calibration.py` | First wavelet attempt (pooled score): exactly chance, **25.0%** — the broken scoring rule that Phase 4b replaced | slow |

## Population-data negative results

| File | Produced by | What it shows |
|---|---|---|
| `phase2b_spd_shrinkage_iv2a.csv` | `experiments/phase2b_spd_shrinkage.py` | SPD shrinkage vs `raw_ai`/`raw_le` over a 8→288-trial sweep, 8-subject leave-one-out prior: null |
| `phase2b_spd_shrinkage_iv2b.csv` | `experiments/phase2b_spd_shrinkage_iv2b.py` | Same sweep on IV-2b (3 channels, 6 tangent dims): null — tests whether dimensionality was the limit |
| `phase2c_eigenspace_k3_iv2a.csv` | `experiments/phase2b_spd_shrinkage.py --eigen-components 3` | Anisotropic eigenvoice variant, k=3: no better than isotropic. Row count (936 = 9 × 6 sizes × 5 repeats × 4 methods, minus single-repeat rows at n=288) matches that command |
| `phase2c_eigenspace_k7_iv2a.csv` | `experiments/phase2b_spd_shrinkage.py --eigen-components 7` | Same, k=7 |
| `phase2d_openbmi20_shrinkage_iv2a.csv` | `experiments/phase2d_openbmi_shrinkage.py --n-subjects 20` | Shrinkage with a 20-subject OpenBMI pool on the 21 shared channels: null on average, but large opposite-signed per-subject effects (subject 2 hurt −3.5 to −6.2pp, subject 9 helped +1.9 to +4.2pp) |
| `phase7c_combat.csv` | `experiments/phase7c_combat.py` | ComBat harmonization vs no correction at n=288: tied (66.9% vs 66.6%) |
| `phase7_riemannian_icp.csv` | `experiments/phase7_riemannian_icp.py` | 5-condition ablation (none / recenter+rescale / +ICP / +rotation / +rotation+ICP): ICP is a real-data null (60.4%, worse than its own 65.9% baseline) |
| `phase7d_pca_concat.csv` | `experiments/phase7d_pca_concat.py` | Full 253-dim vs PCA(d=20) vs concatenated: 67.2% vs 66.6%, not significant (p=0.367) |

## Other closed-out directions

| File | Produced by | What it shows |
|---|---|---|
| `phase4c_wavelet_windowed.csv` | `experiments/phase4c_wavelet_windowed.py` | Time-window features (440-dim): 31.9% vs 31.4%, a wash (p=0.99) — synthetic win, real-data sample-efficiency cost |
| `phase5_combined_features.csv` | `experiments/phase5_combined_features.py` | CSP + wavelet + MiniRocket: naive concatenation *hurts* CSP (62.2% → 54.8%, 8/9 subjects worse, p=0.008) |
| `phase8_motor_cortex_channels.csv` | `experiments/phase8_motor_cortex_channels.py` | Restricting to motor channels is worse than all 22: central-17 → 61.2%, narrow-9 → 56.1% (unadapted) |
| `phase9_flow_field_features.csv` | `experiments/phase9_flow_field_features.py` | Scalp flow-field features standalone: 28.7% (real but weak, p=0.020 vs chance) |
| `phase9b_flow_field_plus_csp.csv` | `experiments/phase9b_flow_field_plus_csp.py` | Flow fields + CSP: worse than CSP alone (59.4%, p=0.043), 8/9 subjects worse |
| `phase10_state_space_flow_frequency.csv` | `experiments/phase10_state_space_flow_frequency.py` | State-space flow frequency, nearest-template: 25.8%, a clean null (no subject significant) |
| `idea10_4_active_calibration_iv2a.csv` | `experiments/idea10_4_active_calibration.py` | Active-learning calibration protocol: real but narrow (+1.6–2.6pp, n=160–192 only); the "22% fewer trials" headline does not survive a paired test (p=0.38) |
| `cleaned_riemannian_check.csv` | `experiments/cleaned_riemannian_check.py` | Artifact-cleaned pipeline + alignment: 61.3%, worse and noisier (std 0.160) than the uncleaned 66.9%. Without alignment it collapses to near/exact chance for several subjects (3/9 at exactly 25.0%) |

## `smoketests/`

Short single-subject or reduced-configuration runs used to debug a pipeline
before committing compute to the full run. They are **not** results and are not
referenced by any document; they are kept only as a record of the debugging
sequence (their schemas drift as columns were added, which is itself the
evidence of that sequence). Names: `phase3_module_a_smoketest.csv`,
`phase4_smoketest_gpu.csv`, `phase4b_smoketest.csv`, `phase4c_smoketest.csv`,
`phase4c_smoketest2.csv`, `phase5_smoketest.csv`, `phase5_smoketest2.csv`,
`phase5_smoketest3.csv`, `phase7_smoketest.csv`, `phase7_smoketest_subj4.csv`.

## Known provenance gaps

Recorded rather than papered over:

- **`phase3_module_a_iv2a.csv` is absent.** `experiments/phase3_module_a.py`
  defaults to writing it, but only `smoketests/phase3_module_a_smoketest.csv`
  exists. The real-data Module A comparison (108-subject PhysioNet training,
  IV-2a evaluation) is therefore described in the docs but has **no committed
  artifact** — re-running it takes hours. Its conclusion (catastrophic
  cross-dataset transfer, 3–15× further from a gold covariance than doing
  nothing) is recorded in
  [`docs/progress_and_direction.md`](../docs/progress_and_direction.md).
- **Scripts that write no CSV at all.** All 15 `stage0_*validation*.py` gates,
  `idea10_5_closure_amplitude_check.py`, the four `diagnose_*.py` scripts and
  `check_wavelet_calibration_real.py` report to stdout only (pass/fail via exit
  code for the `stage0_*` gates). Their numbers live in the root `*.log` files,
  which are gitignored, and in the docs.
- **`cleaned_riemannian_check.csv`'s original script was lost.**
  `experiments/cleaned_riemannian_check.py` is a reconstructed entry point
  written when this repository was formalized. Re-running it reproduces the
  committed file **byte-identically** (verified), so this artifact is now
  reproducible; only its original script is gone.
