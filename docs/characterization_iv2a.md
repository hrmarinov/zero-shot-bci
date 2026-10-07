# Per-subject signal characterization — BCI IV-2a

**Question.** Phase 2 (Riemannian alignment + CSP+LDA) averages 66.9% cross-session
accuracy on IV-2a, but the mean hides a bimodal split: WEAK {2, 5, 6} at 42–49%,
MIDDLE {4} at 60.8%, STRONG {1, 3, 7, 8, 9} at 78–84%. Are 2/5/6 bad because their
signal is genuinely weak (**signal-limited** — no method fixes them, so predict and
route them), or because the pipeline's *fixed* choices (8–30 Hz band, 0.5–2.5 s
window, 8 CSP components) are wrong for them (**configuration-limited** — they are
recoverable)?

**Verdict up front: predominantly signal-limited, at moderate confidence, with one
specific unresolved configuration caveat (subject 5) whose decisive experiment has
not been run.** The discriminating physiological fact is *whether a cue-period mu
ERD exists at all*, not whether it is lateralised. Subjects 2, 5 and 6 show no
detectable class-related band-power structure in the 8–13 Hz band, in either
session, and no measurable ERD — even when the analysis band is re-centred on each
subject's own mu peak, which is the most direct configuration probe run here.

Evidence: `experiments/char_01_iv2a_signal.py` →
`results/char_iv2a_signal_characteristics.csv` (396 rows = 9 subjects × 2 sessions ×
22 channels), `results/char_iv2a_subject_summary.csv` (9 rows).
Runtime 90.5 s, exit 0, threads pinned to 4, peak RSS 0.68 GB.

---

## 1. What was measured

| Family | Content |
|---|---|
| (a) acquisition/quality | trials per class per session; per-channel peak-to-peak amplitude, RMS, excess kurtosis; fraction of trials with max abs amplitude > 100 µV; EOG channels available |
| (b) spectra | Welch PSD per channel on the wide epoch (−1…+4 s, 1–40 Hz, which retains the pre-cue baseline); relative band power in delta/theta/alpha(8–13)/beta(13–30) and total 1–40 Hz; 6–14 Hz peak frequency after aperiodic (1/f) correction; mu/beta ratio |
| (c) ERD/ERS | per channel per class, `(P_cue − P_base)/P_base` in 8–13 and 13–30 Hz, against the pre-cue baseline; contralateral–ipsilateral laterality index for the hand classes plus pair-sign consistency; and a 1000-permutation test of whether the 22-channel band-power pattern carries any class information at all |
| (d) cross-session drift | affine-invariant Riemannian distance (pyriemann, `estimator="oas"`, matching `RiemannianAlignment`) between calibration and evaluation mean trial covariances, on the exact 8–30 Hz / 0.5–2.5 s representation the Phase 2 classifier uses; and the same distance between subjects for scale |

Definitions used below:
`rel_mu` = 8–13 Hz power ÷ 1–40 Hz power, averaged over the 16-channel sensorimotor
strip. `ERD_hands` = mean `(P_cue − P_base)/P_base` over the two hand classes and the
same channels (negative = desynchronisation). `LI_mu` = mean over the two hand
classes of (contralateral desynchronisation − ipsilateral), positive = expected
direction. `p_4class` = permutation p-value of a one-way MANOVA analogue on the
22-channel band-power pattern (`1/1001` = the smallest attainable value).
`p4c_peakband` = the same test in the band `[peak−2, peak+2]` Hz.

## 2. Estimator checks (both load-bearing choices were verified, not assumed)

1. **ERD must be a ratio of *mean powers*, not the mean of *per-trial ratios*.**
   The per-trial ratio is heavy-tailed (mu power over a 2 s window has few degrees
   of freedom; a trial with a small baseline power produces a huge ratio). On
   subject 1's calibration session the mean-of-per-trial-ratios estimator returns
   **+0.589 to +0.957 (mean +0.758) on *every* channel** — a uniform artefactual
   global ERS — while the ratio-of-mean-powers estimator returns −0.263 to −0.057
   (mean −0.156), i.e. the expected global mu ERD. Median discrepancy **+0.885**.
   Using the biased estimator would have destroyed all spatial structure while
   producing numbers that look physiologically plausible. This check is printed by
   the script on every run.
2. **The 6–14 Hz peak needs the aperiodic background removed.** Raw PSD decreases
   monotonically with frequency, so a plain argmax pins the "peak" at the 6 Hz edge
   of the search band for every subject. The script fits `log10 PSD ~ a + b·log10 f`
   outside 6–14 Hz and takes the residual argmax. The fit is confined to 1–40 Hz:
   including the loader's near-empty 40–125 Hz region drags the fitted slope to
   ≈ −12 (instead of ≈ −1.5) and pushes every peak to the *top* edge (13.9 Hz).
   After the fix, 0 of 396 channel rows have a peak at a search-band edge.
3. **Laterality implementations agree.** The class-mean and per-trial code paths
   for `LI_mu` are asserted equal to 1e−12.
4. **Trial counts.** 288 trials per session (72 per class), all 18 subject-sessions,
   asserted. Shapes asserted: 22 channels; wide epoch 1251 samples; narrow 501.
5. **Committed accuracies.** Phase 1 and Phase 2 CSVs are read, never recomputed,
   and asserted against the values quoted in the task context (Phase 2 mean aligned
   0.6693672839506174; per-subject values to 1e−12). Phase 1 cross-session and
   Phase 2 `cross_session_baseline_acc` are asserted to agree for all 9 subjects.

## 3. Per-subject feature table (calibration session unless noted)

| s | group | peak f (Hz) | rel_mu | mu/beta | ERD_hands | LI_mu | p_4class | p4c peakband | drift d | drift/btw | P1 within | P1 cross | P2 aligned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | STRONG | 11.84 | 0.187 | 1.08 | −0.197 | +0.032 | **0.001** | **0.001** | 0.77 | 0.22 | 0.772 | 0.785 | 0.809 |
| 2 | WEAK | 10.62 | 0.332 | 1.64 | **+0.130** | −0.026 | 0.119 | 0.138 | 2.23 | 0.64 | 0.488 | 0.424 | 0.444 |
| 3 | STRONG | 10.74 | 0.292 | 0.95 | −0.348 | +0.106 | **0.001** | **0.001** | 1.08 | 0.32 | 0.863 | 0.767 | 0.816 |
| 4 | MIDDLE | 11.47 | 0.182 | 0.77 | −0.047 | +0.017 | 0.003 | **0.001** | 1.12 | 0.31 | 0.501 | 0.587 | 0.608 |
| 5 | WEAK | **7.32** | **0.111** | **0.66** | −0.066 | +0.002 | 0.330 | **0.944** | 1.92 | 0.38 | 0.354 | 0.340 | 0.424 |
| 6 | WEAK | 12.09 | 0.266 | 1.74 | **+0.035** | **+0.177** | 0.362 | 0.266 | 1.25 | 0.27 | 0.509 | 0.476 | 0.493 |
| 7 | STRONG | 10.25 | 0.211 | 1.05 | −0.333 | −0.014 | 0.130 | 0.131 | 1.63 | 0.48 | 0.760 | 0.625 | 0.785 |
| 8 | STRONG | 10.25 | 0.411 | 1.59 | −0.166 | −0.027 | **0.001** | **0.001** | 1.37 | 0.29 | 0.816 | 0.788 | 0.806 |
| 9 | STRONG | 10.50 | 0.447 | 4.27 | −0.235 | +0.058 | **0.001** | **0.001** | 1.04 | 0.17 | 0.795 | 0.809 | 0.840 |

Peak frequencies are stable across sessions for every subject (subject 5:
7.32 → 7.45 Hz; the largest shift anywhere is 0.24 Hz). `drift/btw` = within-subject
calibration→evaluation Riemannian distance ÷ the median distance from that subject's
calibration mean covariance to the other 8 subjects'.

## 4. WEAK {2, 5, 6} versus STRONG {1, 3, 7, 8, 9}

Medians; AUC = P(weak > strong); Cliff's δ = 2·AUC − 1; p = exact two-sided
Mann-Whitney (n = 3 vs 5). Printed by the script for all 37 features; the
discriminating ones and the notable nulls:

| feature | weak med | strong med | AUC(W>S) | Cliff δ | p |
|---|---|---|---|---|---|
| `erd_mu_hands_mean_calib` | **+0.035** | **−0.235** | **1.000** | **+1.000** | **0.0357** |
| `perm_p_peakband_4class_calib` | **0.266** | **0.001** | **1.000** | **+1.000** | **0.0357** |
| `desync_contra_hands_calib` | 0.077 | 0.213 | 0.000 | −1.000 | 0.0357 |
| `desync_ipsi_hands_calib` | −0.060 | 0.166 | 0.000 | −1.000 | 0.0357 |
| `perm_obs_over_null_mu_4class_calib` | 1.11 | 6.25 | 0.067 | −0.867 | 0.0714 |
| `perm_p_mu_4class_calib` | 0.330 | 0.001 | 0.933 | +0.867 | 0.0714 |
| `perm_p_mu_4class_eval` | 0.690 | 0.001 | 0.933 | +0.867 | 0.0714 |
| `mean_excess_kurtosis_calib` | 1.02 | 0.54 | 0.867 | +0.733 | 0.143 |
| `riemann_drift_calib_to_eval` | 1.92 | 1.08 | 0.867 | +0.733 | 0.143 |
| `phase1_within_session_acc` | 0.488 | 0.795 | 0.000 | −1.000 | 0.0357 |
| `phase2_aligned_acc` | 0.444 | 0.809 | 0.000 | −1.000 | 0.0357 |
| `cross_session_accuracy_drop` | 0.033 | 0.028 | 0.533 | +0.067 | 1.000 |
| `peak_freq_motor_calib` | 10.62 | 10.50 | 0.533 | +0.067 | 1.000 |
| `rel_alpha_motor_calib` | 0.266 | 0.292 | 0.333 | −0.333 | 0.571 |
| `total_power_motor_uv2_per_hz_calib` | 52.8 | 77.1 | 0.333 | −0.333 | 0.571 |
| `mu_beta_ratio_motor_calib` | 1.64 | 1.08 | 0.533 | +0.067 | 1.000 |
| `li_mu_calib` | 0.002 | 0.032 | 0.533 | +0.067 | 1.000 |
| `li_pair_consistency_calib` | 0.60 | 0.60 | 0.433 | −0.133 | 1.000 |
| `perm_p_mu_lr_calib` | 0.161 | 0.142 | 0.600 | +0.200 | 0.786 |
| `mean_frac_trials_gt_100uv_calib` | 0.000 | 0.000 | 0.600 | +0.200 | 0.786 |

**Features that separate the groups**

1. **Presence of a cue-period mu ERD.** Every STRONG subject desynchronises by
   16.6–34.8% during hand imagery; the WEAK subjects range from −6.6% to **+13.0%**
   (subjects 2 and 6 *increase* mu power). This is the only separation with a real
   numerical margin (weak min −0.066 vs strong max −0.166) and the feature most
   directly tied to the mechanism: no ERD means no band-power signal to decode.
2. **Detectable class-related band-power structure, including in each subject's own
   peak band.** `p4c_peakband` = 0.001 for four of the five STRONG subjects (subject
   7 is the exception at 0.131) and 0.138 / 0.266 / 0.944 for subjects 2 / 6 / 5;
   the group separation (AUC 1.000) survives only because subject 2's 0.138 edges
   past subject 7's 0.131. This is the direct test of the configuration hypothesis,
   and for the WEAK group it comes back negative *even after the band is chosen from
   the subject's own spectrum*.
3. **Absolute contralateral and ipsilateral desynchronisation** (same construct as
   (1), summarised by hemisphere).
4. **Committed Phase 1 within-session accuracy** — computed by different code, on a
   different representation (8–30 Hz), with no cross-session transfer involved:
   0.488 / 0.354 / 0.509 for the WEAK group. The ceiling is already set within a
   single session, so a session-shift explanation is excluded.

**Features that do *not* separate the groups** — all reported as nulls, not omitted:
cross-session accuracy drop (0.033 vs 0.028, AUC 0.53), peak frequency (subject 5
aside), relative mu and beta power, total mu-band power, mu/beta ratio, aperiodic
slope, peak prominence, the laterality index and its pair consistency, the
left-vs-right permutation test, and every amplitude/kurtosis artefact proxy
(`|amp| > 100 µV` occurs in ≤ 0.69% of trials for *any* subject; only subject 5 has
an elevated kurtosis, 1.77 vs 0.54 mean).

**The lateralisation test the diagnostic was framed around does not separate the
groups.** `LI_mu` gives AUC 0.53 (medians 0.002 vs 0.032), pair consistency is
identical (0.60 vs 0.60), and the left-vs-right 22-channel permutation test gives
AUC 0.60 (medians 0.161 vs 0.142). It is neither necessary nor sufficient for
accuracy here: subjects 1, 7 and 8 reach 78–81% with left-vs-right pattern tests
that never reach significance (p = 0.075–0.614 across both sessions), while subject 6 has the largest nominal `LI_mu` of all nine (0.177,
9/10 channel pairs in the expected direction) and the permutation test on the same
data does not support it (p = 0.403; effect 0.82× the null mean, i.e. below chance). Subject 6's
large index is a single hand-picked contrast that the full 22-channel pattern does
not corroborate — an asymmetry (all of it in `left_hand`, none in `right_hand`) is
exactly what a lucky contrast looks like. **Conclusion: presence/magnitude of the mu
ERD discriminates; its lateralisation does not.**

**Confounds checked.** `ERD_hands` is a ratio with baseline power in the denominator,
so it could in principle be driven by baseline power rather than task modulation.
Across the 9 subjects, Spearman ρ with `ERD_hands` is −0.10 (total mu-band power),
−0.15 (relative mu power) and −0.03 (peak prominence) — no such confound. In
contrast, `ERD_hands` tracks accuracy: ρ = −0.733 (p = 0.025) with Phase 1
within-session accuracy and −0.767 (p = 0.016) with Phase 2 aligned accuracy, and the
permutation effect size tracks Phase 2 accuracy at ρ = +0.883 (p = 0.002). This is a
graded relationship across all nine subjects, not just a group contrast: the MIDDLE
subject 4 has intermediate values (ERD −0.047, effect 3.19) and intermediate accuracy
(60.8%).

## 5. Statistical honesty

* **n = 3 vs n = 5.** The exact two-sided Mann-Whitney minimum attainable p is
  **2/C(8,3) = 0.0357**, and that is what the complete separations return. Complete
  separation of 3 vs 5 occurs with probability ≈ 3.6% under the null, so it is
  weak-to-moderate evidence, not a demonstration. It cannot support any statement
  about the *population* of BCI users, any estimate of what fraction of users are
  signal-limited, or any claim about subjects outside these eight.
* **The apparent "seven significant features" is one finding, measured seven ways.**
  `ERD_hands`, the two desynchronisation summaries, `p4c_peakband`, the permutation
  effect sizes and the Phase 1/Phase 2 accuracies are all functions of the same
  underlying band-power modulation, and the accuracies are the outcome that
  definitionally must correlate with it. 37 features were inspected without any
  family-wise correction, and the two "independent" permutation statistics
  (ratio-of-means and log-space) disagree in places — subject 6's `p_4class_calib`
  is 0.362 under the conservative ratio statistic and 0.022 under the log statistic,
  and `p_lr` is non-significant for subjects 1 and 8 despite 80%+ accuracy. Any
  single feature in isolation here should be treated as fragile.
* **Two separations are knife-edge.** `p4c_peakband_calib` separates only because
  subject 2 (0.138) exceeds subject 7 (0.131); `desync_contra_hands_calib` separates
  only because subject 6 (0.080) falls just below subject 8 (0.094). The ERD
  separation is the one with a genuine margin.
* **The tests are not necessary conditions.** Subject 7 reaches 78.5% aligned
  accuracy while failing the same 4-class structure test that "passes" the other four
  STRONG subjects (p = 0.130, effect 1.63× null). A negative structure test therefore
  does not *prove* the absence of decodable signal in general — it bounds it for the
  specific statistic, band and window used.
* **All negative results are conditional on the measured configuration.** No sweep
  over bands, cue windows, filter orders or CSP component counts was run. "No
  detectable structure in 8–13 Hz and in `[peak ± 2]` Hz over 0.5–2.5 s" is what was
  established, not "no detectable structure whatsoever".
* What *can* be concluded: on this sample the WEAK trio differ from the STRONG group
  by a large, coherent, mechanistically interpretable amount on the presence of a
  cue-period mu ERD and on detectable class-related band-power structure; those
  differences reproduce in both sessions and in an independently computed
  within-session accuracy; and they are not explained by cross-session drift,
  band choice (for 2 and 6), or artefact load.

## 6. Verdict

**Signal-limited, predominantly, with moderate confidence — and one unresolved
configuration caveat for subject 5.**

* **Subject 2** (44.4% aligned, 48.8% within-session): a clear mu peak (10.6 Hz,
  aperiodic-adjusted residual 0.72, the most prominent in its group) and normal
  power, but **mu power *increases* 13% during hand imagery** instead of
  desynchronising, no detectable class structure in either session (p = 0.119 /
  0.690), and the largest cross-session drift of all nine subjects (2.23, 0.64× the
  between-subject median). Rhythm present, task modulation absent.
* **Subject 6** (49.3% / 50.9%): high mu power (113.8 µV²/Hz, third highest), peak
  at 12.1 Hz, and the largest nominal laterality index (0.177) — but **no ERD in
  either session** (+0.035 / +0.139) and no class structure supported by the
  permutation test in the calibration session (0.362) or the evaluation session
  (0.868). Its large laterality index is a single-contrast fluctuation. The only hint
  of recoverable structure anywhere in the group is subject 6's *evaluation* session
  in its own peak band (10.2–14.2 Hz, 4-class p = 0.058) — exploratory, uncorrected,
  band chosen from the same data, and not accompanied by a within-session advantage.
* **Subject 5** (42.4% / 35.4%, below chance within-session): the one subject whose
  fixed-configuration hypothesis has a specific, testable hook — its dominant
  rhythm peaks at **7.32 Hz, below the 8 Hz corner of the fixed 8–30 Hz band** — and
  it is also the weakest rhythm overall (lowest relative mu power 0.111, lowest
  mu/beta 0.66, least prominent peak, 0.204) and the most artefact-loaded (excess
  kurtosis 1.77 vs 0.54 group mean). But when the analysis is given its own band
  (5.3–9.3 Hz) the result gets **worse, not better** (4-class p = 0.944 vs 0.330 in
  the fixed band; effect 0.30× null). So the band mismatch is real but is unlikely to
  be the binding constraint. Subject 5's own within-session accuracy — 35.4% ± 5.9%
  (repeated-CV std), which does not clear the 25% chance level by 1.96σ — is the
  strongest single argument that its failure is not a transfer or configuration
  artefact.

**Confidence.** *Moderate* for subjects 2 and 6 (coherent, mechanistically specific,
reproduced across both sessions, independently corroborated by the committed
within-session accuracies, and showing a graded dose–response with accuracy at
ρ ≈ −0.8). *Moderate-to-low* for subject 5, where a genuine configuration mismatch
coexists with an absent ERD and the decisive experiment has not been run. *Low* for
any generalisation beyond these subjects.

**Additional measurement that would settle it** (cheap, and the right next step
before writing 2/5/6 off): re-run the Phase 2 pipeline per subject over a small
grid — high-pass corner 4/6/8 Hz; band centred on the individual mu peak; cue windows
0.5–1.5 s, 0.5–2.5 s, 1.0–3.0 s; 4/8/12 CSP components — and score each subject's
*best* within-session accuracy against a within-subject label-permutation null. If no
cell of that grid lifts subject 5's within-session accuracy clear of chance (and
neither 2's nor 6's), the signal-limited verdict is settled and the routing decision
can be made on a measured basis. If subject 5 improves sharply in a 5–9 Hz band, the
conclusion for that subject flips to configuration-limited. A second, complementary
measurement would be a per-subject test of *task engagement* (e.g. accuracy of the
feet-vs-tongue contrast alone, which does not depend on hand lateralisation), to
distinguish "no rhythm modulation" from "no compliance with the imagery task".

## 7. Outliers

* **Subject 5** — peak 7.32/7.45 Hz, the only subject whose rhythm lies outside
  8–30 Hz; lowest relative mu power, lowest mu/beta, least prominent 6–14 Hz peak,
  lowest aperiodic-adjusted residual, highest excess kurtosis (max 5.78 on one
  channel), and within-session accuracy below chance. Its profile is the only one in
  which a fixed pipeline choice is demonstrably mismatched to the signal.
* **Subject 7** — a STRONG subject with a large ERD (−0.333) that is **not
  lateralised** (`LI_mu` = −0.014, pair consistency 0.5, left-vs-right p = 0.075) and
  the only STRONG subject that fails the 4-class structure test (p = 0.130). It also
  benefits most from alignment (+0.160, the largest alignment improvement of all nine,
  p = 1.5e−7 in the committed Phase 2 results) and has the largest cross-session
  accuracy drop (0.135). It is the counterexample that keeps the structure test from
  being a necessary condition, and it suggests subject 7's accuracy rests on a
  non-lateralised, drift-sensitive representation.
* **Subject 6** — simultaneously the highest nominal laterality index, the highest
  pair consistency (0.9), high mu power, and *no* supported class structure: the
  clearest case in the dataset where a hand-picked contrast misleads and the
  permutation test is what saves the analysis.
* **Subject 9** — the opposite extreme: strongest peak prominence (1.17), highest
  relative mu power (0.447) and mu/beta (4.27), perfect 10/10 pair consistency, best
  aligned accuracy (84.0%). Useful as the positive control for every feature above.

## 8. Cross-session drift (d)

For every subject, within-subject calibration→evaluation Riemannian drift is
**1.6–5.8× smaller** than the median distance to other subjects' calibration mean
covariances (drift 0.77–2.23 vs between-subject median 3.33–6.03; ratio 0.17–0.64).
So on IV-2a, unlike the project's earlier cross-*dataset* finding, cross-session
population shift is clearly *not* comparable in size to between-subject distance —
within-subject geometry is relatively stable across days. Drift does not separate the
groups (medians 1.92 vs 1.08, AUC 0.867, p = 0.143, driven mostly by subject 2) and
does not explain the WEAK group's failure, because those subjects already fail
within-session. It does correlate with cross-session accuracy across the nine
subjects (ρ = −0.800, p = 0.010), consistent with drift degrading the transfer of an
otherwise-adequate representation rather than causing the low ceiling.

## 9. Reproduce

```powershell
.venv\Scripts\python.exe experiments\char_01_iv2a_signal.py   # ~91 s, exit 0
```

Writes `results/char_iv2a_signal_characteristics.csv` (396 rows) and
`results/char_iv2a_subject_summary.csv` (9 rows), and prints the estimator check, the
37-feature group comparison with effect sizes, the confound correlations and the
output self-checks. Deterministic apart from the seeded permutation tests
(`RANDOM_STATE = 42`). No file under `src/` or below `experiments/` other than
`experiments/char_01_iv2a_signal.py` was created or modified.
