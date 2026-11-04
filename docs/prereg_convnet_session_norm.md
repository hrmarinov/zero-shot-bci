# Pre-registration — R3: subject-specific ShallowConvNet + per-session normalisation at every layer

**Committed before any evaluation-session (`1test`) accuracy of a convnet was computed.**
Anything that changes after the first look at E is a new pre-registration; the first
result is reported regardless.

## Hypothesis

The project's only working principle is per-session self-normalisation (Riemannian
alignment, RA). Its counterpart at *every* layer of a network — AdaBN (vision domain
adaptation) / CMVN + speaker-adaptive training (speech) — removes per-band, per-filter
multiplicative session gains that one broadband whitening matrix cannot express. A
subject-specific ShallowConvNet trained on T with RA input and AdaBN on E beats the
audited CSP+LDA+RA baseline (66.94%) on the identical T→E protocol.

## Fixed protocol

| Item | Value |
|---|---|
| Data | IV-2a, 9 subjects, train `0train` (288), test `1test` (288), 4 classes |
| Window / band (convnet) | 0–4 s post-cue (1001 samples @ 250 Hz), 4–38 Hz, `load_iv2a_subject_window` |
| Input scaling | RA arms: per-session `RiemannianAlignment` (OAS), each session fit on itself only. Non-RA arms: divide by the scalar std of T (applied to T and E) |
| Model | `ShallowConvNet` defaults (40 filters, kernel 25, pool 75/15, dropout 0.5) |
| Training | AdamW lr 1e-3, wd 1e-3, batch 48, 300 epochs, cosine; no augmentation; no early stopping; deterministic kernels |
| Seeds | 0, 1, 2, 3, 4 — identical across arms (same init, same batch order) |
| Label quarantine | E labels are read only by the scoring step; training functions have no eval-label parameter |

## Arms

| Arm | Training | Test-time normalisation |
|---|---|---|
| `cnn_none` | standard BN | T statistics |
| `cnn_adabn` | = `cnn_none` model | AdaBN on all E trials |
| `cnn_ra` | RA input, standard BN | T statistics |
| **`cnn_ra_adabn`** (primary) | = `cnn_ra` model | AdaBN on all E trials |
| `cnn_ra_sat` | RA input, run-wise BN batches (one run per batch) | run-wise AdaBN (each E run's 48 trials) |
| `csp_ra` | CSP+LDA+RA, 8–30 Hz, 0.5–2.5 s (must reproduce 0.6693672839506174 exactly) | — |
| `csp_ra_wide` | CSP+LDA+RA on the convnet's window/band (representation control) | — |
| `csp_none_wide` | CSP+LDA on the convnet's window/band | — |
| `ctrl_permuted` | `cnn_ra_adabn` trained on permuted T labels, seed 0 | negative control, must sit at chance |

All convnet arms are scored as the per-subject **mean over the 5 seeds**; a 5-seed
probability-averaged ensemble is reported as a separate, secondary line.

## Statistics

* **Primary (one test, one-sided, α = 0.05):** `cnn_ra_adabn` − `csp_ra`, mean of the 9
  per-subject differences; exact sign-flip permutation test (512 flips); paired-t 95% CI
  reported alongside.
* **Secondary family (Holm, α = 0.05, two-sided permutation):**
  S1 `cnn_ra_adabn` − `cnn_ra` (does AdaBN add anything over input alignment?);
  S2 `cnn_ra_sat` − `cnn_ra_adabn` (does run-wise SAT add anything?);
  S3 `cnn_ra_adabn` − `csp_ra_wide` (model vs representation at the same window).
* **Descriptive only:** `cnn_adabn` − `cnn_none`, `cnn_ra` − `cnn_none`, `csp_ra_wide` − `csp_ra`,
  per-subject McNemar (seed-0 predictions and ensemble), the WEAK group {2, 5, 6}, seed SD
  per subject (flag > 3 pp), the ensemble.
* **Power statement:** with n = 9 and a per-subject difference SD of ~5 pp the minimum
  detectable effect at 80% power is ~5 pp; a +2 pp true effect is not detectable here.

## Decision rule

* **ADOPT:** primary mean ≥ +3 pp, permutation p < 0.05, ≥ 7/9 subjects improve, no subject
  loses > 5 pp, negative control within 25 ± 5 pp (mean).
* **REJECT:** paired-t 95% CI upper bound < +1 pp, or mean difference ≤ 0.
* **INCONCLUSIVE:** anything else — no extra seeds, no retuning on E.
* **Tripwire:** mean > 80% → mandatory leakage audit before any claim.
* IV-2a E has been scored many times in this project; an ADOPT is provisional until it
  replicates on a dataset not yet used for model selection (IV-2b or OpenBMI).

## Synthetic gate status at registration time

`experiments/stage0_convnet_session_norm_validation.py`: **partial pass.**
G1 clean decodability, G2 no harm, G3 planted drift (−48 pp), G5 mutation (0% recovery
from a T-statistics "AdaBN") pass; **G4 (AdaBN recovers 59% < 70%) and G6 (SAT recovers
64% < 70%) fail at the pre-set thresholds.** Diagnostic: under a pure broadband gain AdaBN
recovers fully (95.7% vs 95.1% clean); the shortfall is band-*specific* gain, consistent
with the 25-tap (0.1 s, ~10 Hz resolution) temporal filter mixing mu and beta. The
implementation is verified; the mechanism's strength is limited and is what this
experiment measures.
