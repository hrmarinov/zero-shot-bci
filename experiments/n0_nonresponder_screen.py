"""N0, non-responder flag screen: can a flag identify weak subjects better than training-session CV? PRE-REGISTERED.

Committed BEFORE any flag of this script is scored. Source: docs/benefit_complexity_review.md and the D2 finding that the
bottom third of Lee2019 sits at chance for every member (a new source of information is needed, not a better classifier).
CPU only, cached members only, no new network fit. Zhou2020 stays sealed, BNCI2015_001 is not used. No test labels are used
to build any flag (they are used only to score the endpoints).

TARGET. Non-responder = session-2 accuracy of R11 not above the one-sided binomial chance bound (p < 0.05) for the number
of test trials n (Lee2019: n = 100, bound 58 %, i.e. "non-responder" if accuracy <= 58 %; computed with scipy binom.ppf).
Cohorts: Lee2019 n = 54 (primary, the only one with inference); IV-2a n = 9 and IV-2b n = 9 (descriptive only).
R11 must reproduce the stored D2 / IV-2a accuracies or the script stops.

FLAGS (higher flag value = predicted BETTER responder; AUC is computed for "responder", equivalently 1 - AUC for non-responder
after sign alignment; all flags use the training session or unlabelled test trials only):
  N1 baseline  stratified 5-fold CV accuracy of TS+LR (OAS, C = 1) on the aligned training session (seed 0), plus a 200-
               permutation p against chance (reported, not a flag).
  N2 geometry  classDistinct and classStability on the aligned training covariances (OAS), pyRiemann distances (Lotte & Jeunet
               2018, Ivanov & Chau 2023): classDistinct = d(M_1, M_2) / mean within-class dispersion, classStability =
               1 / mean within-class dispersion (dispersion = mean Riemannian distance of trials to their class mean). Two
               flags: N2a distinct, N2b stability.
  N3 spectrum  mu/beta peak over C3, Cz, C4 on the training-session trials (CNN band input, 0-4 s): Welch PSD averaged over
               trials and channels, 1/f line fitted in log-log on 4-38 Hz excluding 8-30 Hz; flag N3a = peak-to-fit SNR in dB
               (max over 8-30 Hz of log10 PSD minus fitted line, x10); N3b = the 1/f slope (flatter = lower); both reported.
  N4 test-time ensemble disagreement / uncertainty (label-free, session-2 trials): N4a = minus mean pairwise disagreement of
               the five R11 members (CNN seeds 0-2, TS+LR, FTS-mean) on argmax; N4b = minus mean predictive entropy of R11;
               N4c = mean top-two margin of R11.
  N5 conformal split-conformal (LAC score 1 - p_true) per subject, alpha = 0.10, calibrated on out-of-fold training-session
               TS+LR probabilities (5-fold), applied to R11 session-2 probabilities. Flag = minus mean set size; empirical
               coverage on session 2 reported (undercoverage from session shift expected and stated).
  N6 pre-cue SMR baseline: testable only if the cached epochs contain a pre-cue period. Cached epochs start at the cue (tmin = 0),
               so N6 is reported as "not testable from caches"; raw data is NOT re-extracted.
ENDPOINTS per flag: Spearman rho with session-2 R11 accuracy, AUC for responder vs non-responder, PPV of the "flag = non-
responder" call when the lowest-flag subjects are called non-responders in the number equal to the observed prevalence; bootstrap
95 % CI over subjects (2000 resamples, seed 0). Paired AUC difference against N1 with its bootstrap CI (same resamples).
RISK-COVERAGE: for R11 and each cached member (CSP+LDA, TS+LR, CNN3), accuracy over the subjects kept at 100 / 80 / 60 % coverage
when subjects are ranked by each flag, against a random-abstention control (mean of 1000 random subsets) and the N1 flag.
Dropping subjects raises mean accuracy mechanically; coverage is always shown next to accuracy.
KILL RULE: a flag N2-N6 is "better than N1" if AUC(flag) - AUC(N1) >= +0.05 AND the bootstrap 95 % CI of the difference excludes
zero AND the Holm-adjusted one-sided bootstrap p (over N2a, N2b, N3a, N3b, N4a, N4b, N4c, N5, the testable flags) < 0.05. If no flag
qualifies the track ends with the finding "training CV is enough". Primary decision on Lee2019 only.
DESCRIPTIVE: which flags agree on the 12 weakest subjects (by R11 accuracy: overlap of each flag's 12 lowest); whether IV-2a
subjects 2 and 6 (weak for R11, 73-75 % for TCFormer) are in the bottom 2 of each flag within IV-2a.

    .venv\Scripts\python experiments\n0_nonresponder_screen.py
"""
