# Ceiling analysis: how much headroom is there above 66.9%, and is the evaluation leak-free?

Every number below comes from `experiments/char_02_ceiling.py` and its output
`results/char_ceiling_iv2a.csv` (9 subjects × 8 conditions + 8 per-condition mean rows = 80 rows).

```powershell
.venv\Scripts\python experiments\char_02_ceiling.py     # 793 s, exit code 0, peak RSS 0.61 GB
```

Protocol (identical for every condition): `src.datasets.load_iv2a_subject`,
`src.baselines.make_csp_lda()` (CSP `n_components=8`, `reg="ledoit_wolf"`,
`log=True`, `norm_trace=False`; default `LinearDiscriminantAnalysis`), and for every CV
`RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)` — the exact splitter
`src/evaluate.within_session_accuracy` uses.

**Confidence-interval policy (read before using the CSV).** `ci_kind` says which interval a
row carries:

* `wilson_binomial_pooled_trials` — the five single-pass conditions. One fixed model is
  tested on 288 independent trials, so a Wilson interval is valid.
* `t_interval_on_50_fold_means` — the three CV conditions, per subject. A binomial interval
  on the *pooled* fold counts would treat 50 folds that re-evaluate the same 288 trials as
  2880 independent draws; measured, that interval is only **0.31–0.32× the width** of a
  genuine 288-trial interval on the same accuracy (printed per subject by the script). The
  CV rows therefore carry a 95% t-interval on the mean of the 50 fold accuracies — which is
  exactly the statistic the 65.1% headline is — with `fold_std` for the fold spread.
* `t_interval_across_9_subjects` — CV mean rows. Between-subject interval; these are wide
  (e.g. 65.10% [50.79, 79.40]) because subject heterogeneity dominates, not because the
  estimate is imprecise.

---

## 0. Answers first

| Question | Answer |
|---|---|
| Do 62.2% and 66.9% reproduce? | **Yes, exactly.** All 7 reproduction checks pass at tol 1e-9 with `max abs diff = 0.000e+00` (per-subject and mean level). |
| Headroom above 66.9%? | **+1.9 pp on the mean** — the eval-session same-session CV ceiling is **68.85%** vs `cross_aligned` **66.94%**. 6/9 subjects positive, 3/9 already at or above it. Not statistically distinguishable from that ceiling (Wilcoxon p = 0.30). |
| Why is 65.1% below 66.9%? | The 65.1% was measured on the **wrong session**: it is the *calibration* session's within-session CV. The eval session is easier for these subjects under two independent measurements (within-session CV **and** cross-session direction). Resolved: no contradiction. |
| Does nested-CV CSP selection change the picture? | No — it is slightly **worse** (66.13% vs 66.94%, −0.81 pp, 2/9 wins, p = 0.30). The ceiling is not an artifact of fixed `n_components=8`. |
| Leakage audit | **6/6 clean.** No evaluation label ever reaches a fitting path; alignment is fit per session; per-fold refitting is confirmed statically *and* by a fit-count experiment; no hyperparameter is selected on held-out data; the Phase 1/Phase 2 McNemar compares identical trials; the reported mean is a macro mean over subjects (implicit, not documented — cosmetic, not a bug). |

---

## 1. Reproduction of the two committed headline numbers

`reproduction_report()` in `experiments/char_02_ceiling.py` reads the two committed CSVs and
asserts this run against them, per subject and for the mean rows, at tolerance 1e-9 (it also
asserts every per-subject comparison array is exactly length 9, so a silent reindex/NaN can
never pass as "close enough"; a failure exits non-zero).

| Check | max abs diff | Verdict |
|---|---|---|
| `within_calib_cv` vs `phase1_baseline_iv2a.csv:within_session_acc` (9 subjects) | 0.000e+00 | PASS |
| `cross_none` vs `phase1_baseline_iv2a.csv:cross_session_acc` (9 subjects) | 0.000e+00 | PASS |
| `cross_none` vs `phase2_riemannian_alignment_iv2a.csv:cross_session_baseline_acc` (9) | 0.000e+00 | PASS |
| `cross_aligned` vs `phase2_riemannian_alignment_iv2a.csv:cross_session_aligned_acc` (9) | 0.000e+00 | PASS |
| MEAN `within_calib_cv` vs Phase 1 average (65.1%) | 0.000e+00 | PASS |
| MEAN `cross_none` vs Phase 1 average (62.2%) | 0.000e+00 | PASS |
| MEAN `cross_aligned` vs Phase 2 average (66.9%) | 0.000e+00 | PASS |

Reproduced values: `within_calib_cv` **0.6509813806547019**, `cross_none`
**0.6222993827160493**, `cross_aligned` **0.6693672839506174** — bit-identical to the
committed CSVs. This is independent of the fit-count/`clone` audit in §5: the hand-rolled
fold loop in `repeated_cv` reproduces `sklearn.cross_val_score`'s per-fold mean to <1e-12 on
synthetic data (printed by every run), and here reproduces all three committed per-subject
columns to 0.0.

---

## 2. How much headroom is there above 66.9%?

Measured, per subject (all values are accuracies; `headroom = within_eval_cv − cross_aligned`):

| Subject | `within_calib_cv` | `within_eval_cv` | `cross_none` | `cross_aligned` | headroom (pp) |
|---|---|---|---|---|---|
| 1 | 0.7723 | 0.7674 | 0.7847 | 0.8090 | **−4.16** |
| 2 | 0.4884 | 0.5354 | 0.4236 | 0.4444 | **+9.09** |
| 3 | 0.8632 | 0.8317 | 0.7674 | 0.8160 | **+1.57** |
| 4 | 0.5010 | 0.6388 | 0.5868 | 0.6076 | **+3.12** |
| 5 | 0.3539 | 0.3950 | 0.3403 | 0.4236 | **−2.86** |
| 6 | 0.5088 | 0.4777 | 0.4757 | 0.4931 | **−1.53** |
| 7 | 0.7601 | 0.8593 | 0.6250 | 0.7847 | **+7.46** |
| 8 | 0.8159 | 0.8264 | 0.7882 | 0.8056 | **+2.08** |
| 9 | 0.7952 | 0.8649 | 0.8090 | 0.8403 | **+2.46** |
| **mean** | **0.6510** | **0.6885** | **0.6223** | **0.6694** | **+1.92** |

Paired across subjects (computed from `results/char_ceiling_iv2a.csv`): mean **+0.0192**,
min −0.0416, max +0.0909, 6 wins / 3 losses, Wilcoxon p = 0.30, paired-t p = 0.23.

**MEASURED.** The mean headroom above the 66.9% headline is **+1.9 pp**, and it is *not*
statistically distinguishable from zero — `cross_aligned` is already statistically
indistinguishable from the eval session's own within-session CV. Put differently, Riemannian
alignment closes **71.1%** of the session gap on the mean subject
((0.66937 − 0.62230) / (0.68852 − 0.62230)).

**MEASURED.** Three subjects (1, 5, 6) already *exceed* their own eval-session CV, so for them
there is no headroom left by this metric at all. The headroom that does exist is concentrated:
subject 7 (+7.5 pp), subject 2 (+9.1 pp) and subject 4 (+3.1 pp) account for essentially all
of it. For subject 7 the eval session is individually very decodable (85.9% within-session)
while the calibration→eval transfer only reaches 78.5% — a genuine transfer failure, not a
hard session. Subject 2 is the opposite failure: both sessions are weak (53.5% / 48.8% CV)
and the cross-session drop is large.

**INFERRED (implication).** Within this representation (band-passed 8–30 Hz, 0.5–2.5 s
epochs, CSP+LDA with 8 components, 288 calibration trials) there is very little decoder
headroom left: ~2 points on the mean, concentrated in 3 subjects, and the aligned
cross-session pipeline already matches the same-session ceiling statistically. The next real
gain has to come from changing the representation or from a *session-transfer* mechanism that
helps specifically subjects 2/4/7 — not from further tuning of CSP+LDA on this pipeline.

**Caveats (MEASURED, and they weaken the "ceiling" framing rather than the conclusion).**
`within_eval_cv` is not an absolute ceiling on achievable accuracy — it is the ceiling *for
this pipeline* at a 5-fold training size (~230 trials). The cross-session conditions train on
the full 288 calibration trials, i.e. more data than any CV fold, so the session-transfer cost
is if anything under-estimated here. And because 3/9 subjects beat their own CV number, the
"ceiling" is a soft bound, not a strict one.

---

## 3. The 65.1% vs 66.9% contradiction, explained

`docs/progress_and_direction.md:1131` labels 65.1% "Within-session (upper bound-ish)" and puts
it directly above 66.9% in a table where it is meant to bound the cross-session rows. The
inconsistency is real, and the explanation is that **65.1% is the within-session CV of the
calibration session, not of the session being tested.**

Two independent measurements, on the same 9 subjects, both say the evaluation session is
easier:

1. **Same-session CV, calib vs eval** (`within_calib_cv` vs `within_eval_cv`):
   65.10% → 68.85%, mean **+3.75 pp**, 6/9 subjects higher on eval, Wilcoxon p = 0.098,
   paired-t p = 0.089. (Subject-by-subject: eval higher for 2, 4, 5, 7, 8, 9; calib higher
   for 1, 3, 6. Subject 4 is extreme: 50.1% → 63.9%.)
2. **Cross-session in both directions** (`cross_none` = train calib → test eval, vs
   `diag_reverse_cross_none` = train eval → test calib): 62.23% vs 56.94%, mean **+5.29 pp**,
   **7 wins / 1 loss / 1 tie** (all but subjects 3 and 7), Wilcoxon p = 0.0156, paired-t
   p = 0.0183. Both directions train on 288 trials, so this isolates *test-session*
   difficulty with no training-size confound.

The reverse-direction result is the stronger of the two (it is significant, and its two arms
differ only in which session is the test set). It is therefore not an artifact of the CSP
model fitting the calibration session better.

**MEASURED / verdict.** Yes — the calibration session is genuinely harder than the evaluation
session for these subjects, and the eval session is easier under *both* framings
(within-session CV and cross-session direction). The 65.1% figure is not a pessimistic
estimate of the eval-session ceiling; it is an estimate of a different quantity on a different
session. Consequently:

* 66.9% does not "break" a 65.1% bound — the applicable bound is 68.85%, and 66.94% sits
  under it.
* A secondary, weaker observation: `cross_aligned` (66.94%) is also **+1.84 pp above**
  `within_calib_cv` (65.10%) on the mean (5/9 subjects). That is only possible because the
  cross-session pipeline is evaluated on the easier session; it is not evidence that
  cross-session transfer beats same-session calibration in general.
* The subject level is mixed (6/9 vs 7/9, not 9/9), so the correct statement is "on the mean,
  and for 6–7 of 9 subjects", not "for every subject".

**Caveat.** With n = 9 subjects, one of these two tests (p = 0.098) is suggestive rather than
significant on its own. The claim rests on the two tests agreeing, and on the
reverse-direction test being significant.

---

## 4. Does selecting `n_components` by nested CV change the picture?

`cross_aligned_nested` = the identical aligned cross-session pipeline, but `n_components` is
chosen by a *2-repeat × 5-fold* inner CV **inside the calibration session only** (aligned
pipeline refit per inner fold; grid 2, 4, 6, 8, 10, 12, 16, 22), then one frozen cross-session
run.

| Condition | Mean accuracy | vs `cross_aligned` |
|---|---|---|
| `cross_aligned` (fixed `n_components=8`) | **0.6694** | — |
| `cross_aligned_nested` (inner-CV-selected) | 0.6613 | **−0.81 pp** |

Paired: 2 wins / 5 losses / 2 ties, Wilcoxon p = 0.30, paired-t p = 0.28.
Selected values: **12, 12, 12, 12, 2, 12, 10, 16, 22** (five subjects picked 12; subject 5
picked 2; subjects 7/8/9 picked 10/16/22).

**MEASURED / verdict.** No — nested selection does not raise the ceiling; it lowers the mean by
0.8 pp and helps in only 2/9 subjects (subject 7: 78.5% → 80.9%; subject 3: 81.6% → 82.3%).
The calibration session's inner CV prefers *larger* `n_components` (≥10 for eight of the nine
subjects: 12 for five of them, plus 10, 16, 22; only subject 5 picks 2), whereas the committed
value is 8 — and that preference does not transfer: subject 2 loses 2.8 pp, subject 5 loses
4.9 pp. **INFERRED:** this is classical selection overfitting to the calibration session, and
it rules out the fixed `n_components=8` as the explanation for the apparent ceiling. The
~1.9 pp headroom in §2 is not recoverable by tuning this hyperparameter on the calibration
session.

A closely related check: alignment does **not** inflate a within-session CV.
`within_eval_cv_aligned` (alignment fit on training folds only — leak-free) = 68.93% vs
`within_eval_cv` = 68.85%, +0.08 pp, 4 wins / 5 losses, p = 0.91. Within-session, Riemannian
alignment is a no-op; cross-session it is worth +4.7 pp (9/9 subjects, p = 0.0039). That is a
strong internal consistency check on the Phase 2 gain: it is a genuine *cross-session* effect,
not an optimistic-bias artifact.

### 4b. Supplementary diagnostic: what if alignment were fit on the wrong session?

`diag_wrongsession_alignment` deliberately commits the bug the task asks about — one whitening
map fit on the calibration session and applied to *both* sessions. Measured:

* wrong-session alignment: **62.62%** vs correct per-session alignment **66.94%**
  (**+4.32 pp for doing it correctly, 9/9 subjects, Wilcoxon p = 0.0039**);
* wrong-session alignment vs no alignment at all (62.23%): **+0.39 pp, 2 wins / 5 losses,
  p = 0.78** — i.e. *nothing*.

**INFERRED (with a measurable basis).** Applying one common invertible linear map to both
sessions cannot help an unregularized CSP+LDA pipeline at all — CSP's spatial filters are
invariant under a common linear transform (`w' = W^{-T} w` yields identical features), so only
`ledoit_wolf` regularization and numerical conditioning make the wrong-session variant differ
from `cross_none` (the residual few-point per-subject deviations, e.g. subject 7 +6.3 pp, are
regularization artifacts, and they cancel to +0.4 pp on the mean). The entire Phase 2 gain
therefore comes from making the two sessions' covariance *geometries* agree, which is exactly
what the per-session fit does and what a single shared map cannot. See §5 Q2 for the
file:line evidence that the code does the right thing.

---

## 5. Leakage audit

Scope: `experiments/phase1_baseline.py`, `experiments/phase2_riemannian_alignment.py`,
`src/evaluate.py`, `src/adaptation/riemannian_align.py` (plus the library code they call).
Each verdict is followed by its evidence and, where possible, by an experiment rather than an
argument.

### Q1. Does anything in the fitting path ever see evaluation-session labels? — **CLEAN**

The strongest static evidence is structural: `src/evaluate.py:59-63`

```python
def fit_predict(pipeline, X_train, y_train, X_test):      # 59
    model = clone(pipeline)                               # 61
    model.fit(X_train, y_train)                           # 62
    return model.predict(X_test)                          # 63
```

`fit_predict` has **no `y_test` parameter**, so evaluation labels cannot even be passed into
the fitting path. `src/evaluate.py:79-84` (`cross_session_accuracy`) forwards `y_test` only to
`accuracy_summary` (`:66-76`), after prediction. Call sites:

* `experiments/phase1_baseline.py:35-37` — fits on `(X_calib, y_calib)` only; `y_eval` is first
  read at `:38` for `wilson_interval` of the finished predictions.
* `experiments/phase2_riemannian_alignment.py:42` — `fit_predict(pipeline, X_calib_aligned,
  data.y_calib, X_eval_aligned)`: `y_eval` is absent from the call. It is first used at `:43`
  (`accuracy_summary`) and `:46` (McNemar), both post hoc.
* `experiments/phase1_baseline.py:34` — the within-session CV is run on `data.X_calib,
  data.y_calib` only; the evaluation session is not touched at all by that call.

Empirical confirmation (measured, per subject, printed by the script): re-running the exact
condition gives bit-identical predictions (`predictions_identical_on_rerun = True` for all 9),
while scoring those same predictions against a permuted `y_eval` collapses to 0.198–0.292
(chance = 0.25). `y_eval`'s only role is scoring.

Also clean one level down: the band-pass filtering is applied **per session**, not to a
concatenation of both sessions — `.venv\Lib\site-packages\moabb\datasets\base.py:1139-1145`
applies the paradigm's process pipeline inside a per-`(session, run)` comprehension, and
`src/datasets.py:136-137` splits the returned epochs by MOABB session label. So no
filtering/artifact statistic is shared across the calibration/test boundary either.

*Exception, by design, not a leak:* the two new CV conditions in `char_02_ceiling.py`
(`within_eval_cv`, `within_eval_cv_aligned`) do train on eval-session labels — that is what
"within-session ceiling" means. They are ceiling estimates, not deployment numbers, and are
labelled as such in the docstring and in the CSV.

### Q2. Is `RiemannianAlignment` fit only on the session it transforms? — **CLEAN (and load-bearing)**

`src/adaptation/riemannian_align.py:32-36`:

```python
def fit(self, X, y=None):                                  # 32
    covs = Covariances(estimator=self.estimator).fit_transform(X)   # 33
    mean_cov = mean_riemann(covs)                          # 34
    self.whitening_ = invsqrtm(mean_cov)                   # 35
    return self                                            # 36
```

`fit` takes labels only as an ignored `y=None`; the map depends on nothing but the array it is
given. `:38-39` (`transform`) applies that stored map with
`np.einsum("cd,ndt->nct", self.whitening_, X)`.

`experiments/phase2_riemannian_alignment.py:39-40` constructs **two separate instances**, each
`fit_transform`-ed on exactly one session's array:

```python
X_calib_aligned = RiemannianAlignment().fit_transform(data.X_calib)   # 39
X_eval_aligned  = RiemannianAlignment().fit_transform(data.X_eval)    # 40
```

Measured confirmation (this run, all 9 subjects):

* `max |mean_riemann(OAS cov(session)) whitened by its own map − I|` = **3e-14 … 1.3e-13** →
  each session is whitened by its own mean covariance, exactly as intended;
* the *calibration* map applied to the eval session leaves the same deviation at
  **0.25 … 1.3** → the two sessions' covariance geometries genuinely differ (0.25–1.3 in
  whitened units), which is why the per-session choice matters;
* `transform` is exactly the linear map it stores (empirical-covariance equivariance dev
  ≤ 6e-14).

One measurement trap worth recording (found while building this check): the whitening
self-check must compute the mean covariance of the **raw** session and then sandwich it with
`W`, as above. Re-estimating OAS covariances on the *already whitened* data and taking their
Riemannian mean does **not** return the identity (it gives a max deviation of ~0.5 for every
subject, e.g. diagonal ≈ 0.61–0.80): Ledoit-Wolf/OAS shrinkage is not equivariant under a
linear transform, because the shrinkage intensity is re-estimated from the transformed scale.
That is a property of the estimator, not a defect in `RiemannianAlignment` — the exact
identity `W · mean_riemann(OAS cov(X)) · W = I` holds to 8e-14.

Cost of the alternative, measured (§4b): the single-map variant scores 62.62% vs 66.94%
(+4.32 pp for the correct per-session fit, 9/9 subjects).

*Caveat (not a leak):* alignment is *transductive* — `fit` uses all 288 unlabelled eval trials
of the session before any trial is decoded. No labels are involved and this is precisely the
method's selling point (zero calibration labels), but it does mean Phase 2 is an offline
result, not an online/causal one. By contrast `within_eval_cv_aligned` fits the map on
training folds only and still matches `within_eval_cv` (§4), so the aligned *within-session*
protocol carries no optimism either.

### Q3. In `within_session_accuracy`, is the pipeline refit inside each fold? — **CLEAN**

`src/evaluate.py:54-55`:

```python
cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)  # 54
scores = cross_val_score(clone(pipeline), X, y, cv=cv, scoring="accuracy")                       # 55
```

`cross_val_score` delegates to `cross_validate`
(`.venv/Lib/site-packages/sklearn/model_selection/_validation.py:677-690`), which **clones the
estimator for every split** (`:396-401`, `clone(estimator)` inside the `parallel(...)` job
list, with the comment *"We clone the estimator to make sure that all the folds are
independent"*), and `_fit_and_score` then fits that clone on the training split
(`:851-859`). A pre-fitted pipeline therefore cannot be reused: whatever state the caller's
`pipeline` object has is irrelevant, and `clone` drops fitted attributes anyway. The explicit
`clone(pipeline)` at `src/evaluate.py:55` is **redundant but harmless**.

Demonstrated, not merely asserted, by `audit_cross_val_score_refits()` in the new script
(printed on every run):

* the estimator object *handed to* `cross_val_score` is fitted **0** times, while exactly
  **50** `fit` calls occur for 5 folds × 10 repeats (both `cross_val_score` and the
  hand-rolled loop measured at 50);
* the hand-rolled loop reproduces `cross_val_score`'s per-fold mean to <1e-12 on synthetic
  data — and, on real data, reproduces all three committed per-subject columns to 0.0 (§1),
  which also proves the committed numbers came from exactly this protocol.

Second route for the same leak: `fit_predict` clones at `:61` too, so the pipeline object
created at `experiments/phase1_baseline.py:32` and passed to *both* `within_session_accuracy`
(`:34`) and `cross_session_accuracy` (`:35`) never carries CV-fitted state into the
cross-session fit. No fitted attribute is shared.

For the aligned CV in this script, alignment sits *inside* the `Pipeline`
(`aligned_pipeline()`, `char_02_ceiling.py:144-155`), so the same per-fold cloning fits the
alignment on the training folds only. If alignment were instead fit on the full session
before the CV, the training folds would see the test folds' covariance geometry; that
stronger, leak-free construction is what produces the 68.93% figure in §4.

### Q4. Are the CSP/LDA hyperparameters ever selected using held-out data? — **CLEAN (fixed a priori), one unverifiable caveat**

* `src/baselines.py:10` — `def make_csp_lda(n_components: int = 8)`; the only CSP settings are
  literals at `:18` (`reg="ledoit_wolf", log=True, norm_trace=False`) and LDA is default
  (`:19`).
* `experiments/phase1_baseline.py:32` and `experiments/phase2_riemannian_alignment.py:32` call
  `make_csp_lda()` with **no argument** → `n_components=8` in every committed result.
* A repo-wide search finds no `GridSearchCV`, `RandomizedSearchCV` or `ParameterGrid`
  anywhere; the only `n_components` knobs are CLI ablation arguments in *other* phases
  (`ablation_01_calibration_burden.py`, `phase2b`, `phase8`), none of which produce the
  62.2/66.9 headline.

So no hyperparameter is fitted on held-out data. Two honest caveats: (i) `n_components=8` is a
literature-conventional constant, and whether it was historically chosen while looking at this
dataset cannot be verified from the current tree (no git commands were run); (ii) it does not
matter numerically — §4 measures what happens when it *is* selected from the data (on the
calibration session only): the mean drops 0.8 pp. Both directions of the "tuned or not"
question are therefore bounded and small.

### Q5. Is the Phase 1↔Phase 2 difference computed on identical trial sets, as McNemar requires? — **CLEAN**

* Phase 2 does not reuse Phase 1's CSV: it **recomputes** the baseline itself at
  `experiments/phase2_riemannian_alignment.py:35`
  (`fit_predict(pipeline, data.X_calib, data.y_calib, data.X_eval)`), with the same loader
  (`:31`) and the same pipeline (`:32`) as Phase 1.
* `:46` — `mcnemar_test(data.y_eval, y_pred_baseline, y_pred_aligned)`: one `y_true`, two
  prediction vectors, all length 288, both predictions produced from the *same* `data.X_eval`
  array in the same order by the same `fit_predict`. `src/evaluate.py:96-99` compares them
  elementwise (`correct_a = y_pred_a == y_true`, etc.), so it requires — and here gets —
  trial-aligned vectors. The test is used correctly.
* Measured cross-check from the committed CSVs: Phase 1's `cross_session_acc` and Phase 2's
  `cross_session_baseline_acc` agree for **all 9 subjects with max |diff| = 0.0** (and this run
  reproduces both to 0.0). Identical trial sets, identical protocol, hence a valid paired test.
* The `n_total` column confirms it: every `cross_*` row has `n_total = 288` (72 trials × 4
  classes, both sessions), verified for all 9 subjects.

This also retro-validates the Phase 2 claim of +4.7 pp (9/9 subjects, Wilcoxon p = 0.0039,
recomputed in this run): it is a paired comparison on identical trials, not two independent
CIs.

### Q6. Does the reported mean aggregate subjects or trials, and is that stated anywhere? — **MACRO OVER SUBJECTS; documented only implicitly (cosmetic gap, not a bug)**

* `experiments/phase1_baseline.py:75-79` and `experiments/phase2_riemannian_alignment.py:71-75`
  build the "average" row from `df["..."].mean()` over the 9 per-subject rows → an unweighted
  **mean of subject accuracies**.
* For the cross-session conditions macro and micro are numerically identical, because every
  subject contributes exactly 288 trials (measured: 1613/2592 = 0.6222993827… = the mean of the
  9 subject accuracies; likewise 1735/2592 = 0.66936728…). The ambiguity has no effect on 62.2%
  or 66.9%.
* For the within-session CV it is a *two-level* unweighted mean (subjects, then folds):
  macro = 0.6509813806547019 vs pooled-fold micro = 16873/25920 = 0.6509645062 — a difference
  of 1.7e-5 that is rounding-invisible but conceptually distinct.
* Is it stated? Effectively no. `docs/progress_and_direction.md:1129` says only "Cross-session
  accuracy (9 IV-2a subjects, 4-class)", and the CSVs' only hint is the literal string
  `average` in the `subject` column. Nothing states "unweighted mean over subjects" or "the
  CV number is a mean of per-subject fold means". This is a documentation gap worth a
  one-line fix in the doc/CSV header; it is not a methodological error, and with the equal-n
  design it cannot change any headline number except in the 5th decimal.

---

## 6. Measured vs inferred, and what is unresolved

**MEASURED** (all reproducible from `results/char_ceiling_iv2a.csv`):

* exact reproduction of 65.10% / 62.23% / 66.94%; the eval-session ceiling 68.85%;
  the +1.92 pp headroom and its per-subject spread; +3.75 pp calib→eval CV and +5.29 pp
  cross-direction session difficulty; the nested-selection result (66.13%); the
  leak-free aligned CV result (68.93% vs 68.85%); the wrong-session-alignment result
  (62.62%, i.e. +4.32 pp for per-session fitting, 9/9, p = 0.0039); the alignment
  self-consistency checks (≤1.3e-13 self, 0.25–1.3 cross); the `clone`/refit fit counts
  (0 on the caller's object, 50 actual); the CV-CI width ratio (~0.32×); and the
  Phase 1 ≡ Phase 2 baseline equality (0.0).

**INFERRED** (arguments, not direct measurements):

* that the *cause* of the eval session being easier is intrinsic session quality rather than
  the CSP model fitting one session better — the reverse-direction cross-session test supports
  this but does not prove it;
* the CSP-invariance explanation for why the wrong-session variant adds ~nothing (its
  theoretical basis is solid; the code uses `ledoit_wolf`, whose shrinkage breaks the exact
  invariance, so the residual per-subject deviations are expected and observed);
* that the remaining ~1.9 pp is "representation-limited" rather than a fixable bug.

**Unresolved / not attempted:**

* Whether a *strict* ceiling exists that bounds `within_eval_cv` itself (it is a ceiling for
  CSP+LDA at 5-fold training size, not for the task). A stronger representation could exceed
  it, in which case "headroom" here means "headroom for this pipeline family".
* The 3 subjects (1, 5, 6) whose cross-session aligned accuracy exceeds their eval-session CV
  are not explained — the likely causes (288 vs ~230 training trials; alignment changing the
  CSP's operating point) were not separated experimentally.
* The calibration-session inner CV prefers larger `n_components` for 8/9 subjects while the
  eval session prefers ~8; that systematic disagreement is measured but not explained.
* No multi-seed study of the CV estimate was run: `random_state=42` fixes the folds, matching
  the committed protocol exactly (which is what reproduction required). The repeated-CV std
  (`fold_std` ≈ 0.037–0.072) is the only fold-to-fold variability reported.
* Scope note: this audit covers the four named files. Other call sites of
  `RiemannianAlignment` exist elsewhere in `experiments/`
  (`ablation_01_calibration_burden.py:116` fits a *calibration-session* reference map and
  reuses it — a different design from Phase 2's, in an ablation script); those files were not
  audited, and `ablation_01`/`ablation_02` were being written by a concurrent agent while this
  analysis ran.
