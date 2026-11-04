# R3 — Per-session normalisation at every layer (ShallowConvNet + AdaBN / speaker-adaptive BN)

**Status:** pre-registered ([`prereg_convnet_session_norm.md`](prereg_convnet_session_norm.md)),
synthetic gate partial pass, real-data sweep complete — results in §4.

## 1. Why this, and why now (audit → council)

The R1-C ceiling audit established that the aligned CSP+LDA pipeline (66.9%) sits within
+1.9pp of its own same-session ceiling (68.9%). That ceiling bounds the *CSP+LDA family on
an 8–30 Hz / 0.5–2.5 s representation*, not the task. The project's deep-learning verdict
was literature-only and concerned cross-subject / foundation models; a subject-specific
convnet on the identical T→E split had never been run here.

A five-member council (BCI/DL researcher, statistician, cross-disciplinary methods,
red-team, compute engineer) converged on one experiment: generalise the project's single
working principle — **per-session self-normalisation** — from the input to every layer.

## 2. The transplant

| Field | Concept | EEG counterpart here |
|---|---|---|
| Speech recognition | channel/microphone convolution → additive log-spectral offset | electrode impedance / cap shift → multiplicative per-band gain → additive offset in log-variance |
| Speech recognition | CMVN (per-utterance mean/variance normalisation) | BatchNorm statistics per session |
| Speech recognition | speaker-adaptive training (SAT) | run-wise BN during training: every batch is one recording run |
| Vision domain adaptation | AdaBN: recompute BN statistics on the unlabelled target domain | BN statistics from the unlabelled evaluation session |
| Vision feature whitening | whitening the input distribution | Riemannian alignment (the existing 66.9% method) |
| Genomics | batch effect | session drift |

**Mechanism, specific to ShallowConvNet.** Its BN sits *before* square → mean-pool → log.
Dividing a filter's output by its per-session standard deviation therefore subtracts a
per-session constant from that filter's log-variance feature, which is exactly what a
multiplicative, filter- and band-specific session gain adds. Riemannian alignment uses one
broadband whitening matrix and cannot express a gain that differs between bands; per-session
BN can. Every statistic is computed from one subject's own data — the project's design rule
holds — and no evaluation label is used.

## 3. Synthetic gate (`experiments/stage0_convnet_session_norm_validation.py`)

Planted, known drift on 22-channel synthetic EEG with power-coded classes; criteria fixed
in the script before the first run; 3 generator seeds.

| Drift on E | none | AdaBN | mutant ("AdaBN" with T stats) | SAT (run-wise) |
|---|---|---|---|---|
| none | 95.1% | 95.1% | 95.1% | 94.8% |
| band-specific gain (mu ×2.5, beta ×0.5) | 47.2% | 75.7% | 47.2% | 76.5% |
| + per-run gain drift in both sessions | 68.5% | 73.7% | 68.4% | **85.5%** |
| + 70/10/10/10 class prior in E (report only) | 21.1% | 85.6% | 21.1% | 83.7% |

* PASS: clean decodability, no harm, planted drift real (−48pp), **mutation caught**
  (0% recovery from a T-statistics "AdaBN" — a no-op looks like a no-op).
* **FAIL at the pre-set threshold:** AdaBN recovers 59% (< 70%) of the band-specific drop;
  SAT recovers 64% (< 70%) of the run-drift drop. Not loosened.
* Diagnostic: under a *broadband* gain AdaBN recovers fully (95.7% vs 95.1% clean). The
  shortfall is band-*specific* gain, consistent with the 25-tap (0.1 s ≈ 10 Hz resolution)
  temporal filter mixing mu and beta, so one per-filter gain cannot undo two band gains.
* The predicted AdaBN weakness under class-prior shift did **not** appear.
* Reproducibility: the band-gain condition reproduced bit-exactly in a separate process.

## 4. Real data — IV-2a T→E, 9 subjects, 5 seeds, 56 min on the RTX 4060

```powershell
.venv\Scripts\python experiments\r3_convnet_session_norm.py            # 56 min, 1 thermal pause (83 °C)
.venv\Scripts\python experiments\r3b_convnet_session_norm_analysis.py  # seconds + label reload
```

Per-subject accuracy (%; convnet arms are the mean of 5 seeds):

| s | csp_ra | csp_ra_wide | csp_none_wide | cnn_none | cnn_adabn | cnn_ra | **cnn_ra_adabn** | cnn_ra_sat |
|---|---|---|---|---|---|---|---|---|
| 1 | 80.9 | 74.3 | 69.1 | 72.0 | 72.4 | 82.8 | 82.6 | 80.6 |
| 2 | 44.4 | 52.8 | 48.3 | 48.8 | 52.2 | 59.4 | 59.9 | 62.2 |
| 3 | 81.6 | 83.3 | 74.0 | 81.1 | 84.1 | 88.9 | 88.9 | 89.3 |
| 4 | 60.8 | 60.8 | 56.2 | 57.8 | 58.3 | 71.4 | 71.2 | 73.0 |
| 5 | 42.4 | 45.5 | 35.4 | 46.1 | 46.7 | 57.0 | 57.2 | 59.3 |
| 6 | 49.3 | 50.3 | 49.3 | 46.1 | 46.9 | 56.9 | 57.2 | 57.9 |
| 7 | 78.5 | 67.4 | 62.8 | 75.0 | 82.9 | 88.3 | 88.7 | 89.2 |
| 8 | 80.6 | 77.8 | 75.3 | 71.2 | 74.4 | 84.7 | 84.9 | 85.1 |
| 9 | 84.0 | 76.7 | 77.1 | 70.6 | 72.2 | 82.4 | 81.8 | 81.7 |
| **mean** | **66.94** | 65.43 | 60.84 | 63.19 | 65.56 | 74.64 | **74.71** | 75.38 |

* **Harness identity:** `csp_ra` = 0.6693672839506174, bit-exact Phase 2 (asserted per subject).
* **Primary (pre-registered):** `cnn_ra_adabn − csp_ra` = **+7.78 pp**, 95% CI [+3.31, +12.25],
  8 W / 1 L, worst subject −2.22 pp (s9), exact sign-flip p = **0.0059** (Wilcoxon 0.0117).
* **Secondary (Holm):** S1 AdaBN over RA **+0.08 pp, p_holm = 0.53**; S2 run-wise SAT over
  AdaBN +0.66 pp, p_holm = 0.30; S3 convnet vs CSP+RA on the *same* window **+9.28 pp, 9/9,
  p_holm = 0.012**.
* **Descriptive:** RA on the convnet input +11.44 pp (9/9) vs +4.59 pp for CSP on the same
  window; AdaBN without RA +2.37 pp (9/9); window/band change alone for CSP −1.50 pp (4/4);
  5-seed ensemble 78.3%; WEAK {2, 5, 6} 45.4% → 58.1%.
* **Negative control** (permuted T labels): mean 27.2% (19.8–34.0%), inside 25 ± 5 pp.
* Seed SD median 1.7 pp; only subject 5's *unaligned* arms exceed 3 pp (4.4–4.5 pp).

**Pre-registered decision: ADOPT — provisional until it replicates on a dataset not used
for selection.**

### Verification

* Fresh-process rerun of subject 7 (seeds 0 and 3, all arms): 13/13 rows and every stored
  probability bit-identical.
* Independent audit (separate agent, own code): leakage trace clean (eval labels reach only
  `score()`), all statistics re-derived exactly, 36 accuracies re-scored from stored
  predictions against freshly loaded labels — **ACCEPTED WITH CAVEAT** (the attribution below).

## 5. What the result actually says

1. **The hypothesis that motivated the transplant is not supported.** Per-layer
   normalisation (AdaBN, run-wise SAT) adds nothing measurable once the input is
   Riemannian-aligned (+0.08 pp, +0.66 pp). The synthetic gate showed it *can* remove a
   band-specific gain; on real IV-2a after RA there is little such residual gain left to
   remove. Without RA, AdaBN does help (+2.4 pp, 9/9) — it recovers part of what RA does.
2. **What works is the project's own principle with a stronger model.** Per-session input
   alignment is worth +11.4 pp to a convnet (vs +4.6 pp to CSP). The correct name of the
   adopted method is **"subject-specific ShallowConvNet on per-session Riemannian-aligned
   input"** (`cnn_ra`, 74.64%); AdaBN is harmless and optional.
3. **The CSP "ceiling" was a representation ceiling.** 74.7% exceeds the CSP+LDA
   eval-session within-session ceiling (68.9%) by ~6 pp, at the same transductive,
   zero-eval-label protocol.
4. **The WEAK subjects are not (only) signal-limited.** Subjects 2, 5, 6 reach 57–60%, above
   their own CSP *within-session* accuracy (48.8 / 35.4 / 50.9%). R1-B's "signal-limited"
   verdict was a verdict about CSP's representation; it needs revisiting.
5. **Caveat on novelty:** ShallowConvNet on IV-2a T→E is published (~73.7%, Schirrmeister
   2017) and Euclidean alignment + deep nets is known (He & Wu 2020). The contribution here
   is the audited, leak-free, pre-registered measurement inside this project's protocol —
   not a new method.

## 6. Next (decided)

1. **External replication** (prereg requirement): the identical frozen recipe on IV-2b
   (T sessions → E sessions) and OpenBMI-20 (session 1 → 2), no retuning.
2. **Calibration-burden curve** for `cnn_ra` (TODO Part E criterion 7) — reuse
   `ablation_01`'s budgets.
3. **Causal / online alignment** (running RA reference over the first k eval trials) — the
   step from transductive to genuinely deployable; pairs with A3's entropy-minimisation TTA.
4. Re-run R1-B's WEAK-subject verdict against the convnet's within-session CV.
