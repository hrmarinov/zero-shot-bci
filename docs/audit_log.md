# Audit log

Record of what was claimed, what was independently verified, and what was accepted. The
rule for this project: **no result is treated as real until it has been re-derived or
re-run by someone other than its author.** Subagent output is a claim; this file records
the verdict.

Numbering: `R<round>` for the work round, `A<n>` for the agent task within it.

---

## Verification protocol (applied to every agent deliverable)

Before a result is accepted:

1. **Re-run it.** The committed script is run again from a clean shell and must exit 0 and
   reproduce its own CSV.
2. **Cross-check against known numbers.** Any script that touches a protocol already
   committed (`results/phase1_baseline_iv2a.csv`, `results/phase2_riemannian_alignment_iv2a.csv`)
   must reproduce those values *exactly* at the matching configuration. A near-miss is a
   finding, not a rounding error.
3. **Read the code for leakage.** Evaluation-session labels must never enter any fitting
   path. Unsupervised use of evaluation-session trials (whitening) is allowed but must be
   stated.
4. **Check the failure modes this project has already been bitten by:** scale/normalization
   confounds, peak-picking/multiple-comparison bias, in-sample scoring-rule bugs,
   samples-per-feature collapse, and claims of significance that the n cannot support
   (n=9 → Wilcoxon floor p=0.0039).
5. **Check the reported statistics against the design.** Per-subject rows, not just means;
   paired tests on identical trial sets; no significance claim beyond the power available.
6. **Look for the unstated assumption.** What did the agent have to assume to make its
   numbers work, and is that assumption true?

Verdicts: **ACCEPTED**, **ACCEPTED WITH CAVEAT** (true but limited in a way that must be
stated), **REJECTED** (wrong or unverifiable), **REWORK** (fixable, sent back).

---

## R1 — Dataset understanding and characterization (the "deep understanding" round)

Three parallel agent tasks, each owning distinct files. Machine state during this round:
i9-13900HX (8 P-cores + 16 E-cores), ~8 GB RAM available, RTX 4060 8 GB, so agents were
instructed to pin 4 threads each and stay under ~3 GB RSS
(`docs/compute_environment.md`).

| Task | Agent | Deliverables | Status |
|---|---|---|---|
| R1-A | dataset methodology audit | `experiments/char_00_dataset_audit.py`, `results/char_dataset_audit.csv`, `docs/datasets.md` | running |
| R1-B | IV-2a per-subject signal characterization | `experiments/char_01_iv2a_signal.py`, `results/char_iv2a_signal_characteristics.csv`, `results/char_iv2a_subject_summary.csv`, `docs/characterization_iv2a.md` | running |
| R1-C | ceiling estimate + leakage audit | `experiments/char_02_ceiling.py`, `results/char_ceiling_iv2a.csv`, `docs/ceiling_analysis.md` | running |

## R2 — Attacking the score, and auditing my own code

### R2-I1. `experiments/ablation_02_alignment_variants.py` — two no-op bugs in my own script, caught by a guard I added for that purpose

The first Phase-2 experiment (TODO A12 + the cheap end of A5) asks a question the project
never asked: **is the Riemannian mean the right whitening reference?** Every condition is a
drop-in replacement for the single line that estimates the session's whitening matrix, with
the loader, CSP+LDA configuration, trial sets and scoring held fixed.

Writing it, I introduced a silent no-op **twice**, and both times the same smoke test caught
it — all three `subspace_k{10,30,80}` conditions returned **bit-identical** accuracy to
`riemann`:

1. **First version** averaged the *tangent vectors* after a rank-k projection. `tangent_space`
   centres at the reference mean, so their mean is ~0 and `untangent_space(0, cref)` returns
   `cref`. No-op.
2. **Second version** reconstructed each trial's tangent vector and then took their Fréchet
   mean — *still* in the tangent space at the session's own mean. Also a no-op, and
   **necessarily** so: the affine-invariant Riemannian mean satisfies
   `sum_i log_cref(covs_i) = 0` exactly (that is its first-order condition), so any linear
   projection P of the centred vectors still sums to zero, which means `cref` still satisfies
   the mean's own first-order condition. **Rank-limiting cannot change a mean when it is
   applied in the tangent space anchored at that mean.**
3. **Working version** anchors the projection at a *fixed external* reference (the identity),
   so the projection bites and the result depends on k.

**Action taken beyond the fix:** the report now prints an `intervention check` — for every
condition, how many subjects produce predictions bit-identical to `riemann`, with
`<-- NO-OP SUSPECTED` when it is all of them. Both broken versions would have been flagged
automatically by that line. This is the generalisable lesson: **any experiment whose
"variant" is a transformation should assert that the transformation actually changed
something**, because a no-op is indistinguishable from a true null result in the final table.

**Verdict on the script: ACCEPTED for the full run** — harness identity verified (see below),
intervention check passing (0/1 subjects identical for every condition), exit 0.

Harness-identity check on subject 1 (the property that makes any later comparison
interpretable), tolerance 1e-9:

| Condition | Must reproduce | Got | Match |
|---|---|---|---|
| `none` | Phase 1 cross-session, subject 1 = 0.7847222222222222 | identical | exact |
| `riemann` | Phase 2 aligned, subject 1 = 0.8090277777777778 | identical | exact |

Preliminary single-subject observations — **not results**, 1 subject, and at 288 trials one
trial = 0.35pp so ±1 trial is noise: `logeuclid`, `wasserstein` and `power_p0.5` each scored
+1 trial over `riemann`; `euclid` scored −1 trial; and **`perclass_calib` collapsed to 0.455
(−35pp)**. The last one is mechanistically explicable and worth testing properly across
subjects: whitening each calibration class by its *own* mean covariance drives both class
means to the identity, which erases exactly the between-class covariance structure CSP is
built to exploit. If it replicates, the conclusion is a real design constraint — *an
alignment reference must remain class-agnostic* — rather than a bug.

### R2-I2. Scale framing added to the variants report

At 288 evaluation trials per subject, **one trial = 0.35pp**. A per-subject delta of one or
two trials is noise by construction. The report now states this explicitly and points
inference at the paired tests and per-subject McNemar (which counts only *discordant*
trials) rather than at raw mean differences. Relevant because several of the candidate
variants differ by exactly one trial on a single subject.

### Checks performed by me (not delegated)

**1. Leakage review of the headline numbers — read line by line.**

| Question | Evidence | Verdict |
|---|---|---|
| Does the fitting path ever see evaluation-session labels? | `experiments/phase1_baseline.py:35-37` and `phase2_riemannian_alignment.py:35,42` pass only `data.y_calib` to `fit`; evaluation labels appear only in `accuracy_summary`/`mcnemar_test`, i.e. scoring | clean |
| Is `RiemannianAlignment` fit only on the session it transforms? | `phase2_riemannian_alignment.py:39-40` constructs two separate instances, one per session. `riemannian_align.py:32-36` fits on the array it is given | clean |
| Is the pipeline refit inside every CV fold, or reused (leak)? | `src/evaluate.py:55` `cross_val_score(clone(pipeline), ...)`; `src/evaluate.py:60-63` `fit_predict` clones. Also asserted by `tests/test_evaluate.py::test_fit_predict_does_not_mutate_the_supplied_pipeline` | clean |
| Are hyperparameters selected on held-out data? | `src/baselines.py:10-20` — `n_components=8`, `reg="ledoit_wolf"` are constants, never tuned | clean |
| Is the Phase 1 vs Phase 2 comparison computed on identical trials? | `phase2_riemannian_alignment.py:35-46` recomputes the baseline and the aligned predictions on the same eval set, then `mcnemar_test` on the two prediction vectors | clean |
| Does the reported mean aggregate subjects or trials? | `phase1_baseline.py:73-84` — mean over per-subject accuracies; every subject has exactly 288 eval trials, so subject-mean equals pooled-trial accuracy here | clean (coincidence of equal n; worth stating) |

**Conclusion:** the 62.2% / 66.9% headline numbers contain no label leakage. The
unsupervised use of evaluation-session trials to estimate that session's own whitening
matrix is the intended transductive design of Riemannian alignment, and it is documented
in `src/adaptation/riemannian_align.py`'s docstring. The open question is therefore *not*
"is the number inflated" but "how much headroom is above it" — assigned to R1-C.

**2. New experiment written and verified: `experiments/ablation_01_calibration_burden.py`**
(TODO.md A29 — accuracy as a function of the *labelled* calibration budget.)

- Found and fixed one bug of my own via smoke test: the pandas aggregation renamed the
  accuracy column to `mean`, so the trials-to-threshold reporter raised `KeyError`.
  Fixed by explicit renaming to `accuracy_mean`/`accuracy_std`/`n_samples`.
- Verified exit code 0 on a real run (an earlier `[exit code: 1]` was an artifact of piping
  through `Select-Object`, confirmed by re-running with output redirected to a file).
- **Bit-for-bit consistency check at n=288** (the configuration that must agree with the
  committed baselines), subject 1:

  | Quantity | Committed | Ablation script | Match |
  |---|---|---|---|
  | Phase 1 cross-session, no adaptation | `0.7847222222222222` | `0.7847222222222222` | exact |
  | Phase 2 cross-session, aligned | `0.8090277777777778` | `0.8090277777777778` | exact |

  This is the strongest available evidence that the new script's protocol is identical to
  the established one. **Verdict: ACCEPTED (protocol identity verified).** The full
  9-subject sweep is pending a free machine.
- Early observation (subject 1 only, 2 repeats — **not** a result, do not cite): at n=32,
  `none` 0.672 vs `aligned` 0.656, i.e. alignment may *hurt* at small budgets for this
  subject, which is the opposite of the hypothesis motivating A29. Flagged for the full
  run; if it holds across subjects, the interesting conclusion is that the *alignment
  reference* needs more data than the *classifier*, which is what the
  `aligned_ref_full` diagnostic condition was added to separate.

### R1-I1. Independent finding from committed data — the "weak three" are not drift-limited

Not delegated; computed by me from the two committed CSVs only (no EEG loading, no new
compute). Reproduces the committed means exactly (62.23% / 66.94%).

| Group | n | Within-session CV (**calibration** session) | Cross-session, no adaptation | Cross-session + alignment | Alignment gain |
|---|---|---|---|---|---|
| "weak" — subjects 2, 5, 6 | 3 | **45.0%** | 41.3% | 45.4% | +4.1pp |
| "middle" — subject 4 | 1 | 50.1% | 58.7% | 60.8% | +2.1pp |
| "strong" — subjects 1, 3, 7, 8, 9 | 5 | **80.1%** | 75.5% | 81.1% | +5.6pp |

**Finding 1 — the weak subjects' deficit exists *without any cross-session drift*.**
Subjects 2, 5 and 6 score 45.0% in *within-session* cross-validation (train and test inside
one session, CSP refit per fold) versus 80.1% for the strong group: a 35pp gap that needs no
session change to appear, against a 34pp gap cross-session. Their failure therefore **cannot
be attributed to miscalibration or session drift**, which is where this project has aimed
almost all of its effort. This is consistent with the earlier per-subject observation that
subject 5 sits at chance under every method tried, but it has never been stated as a
group-level fact.

**Finding 2 — what this does *not* establish.** It does **not** yet separate "irreducible
(electrode placement, anatomy, genuinely weak sensorimotor rhythm)" from "recoverable but
outside the fixed pipeline configuration (8–30 Hz band, 0.5–2.5 s window, 8 CSP
components)". That distinction is exactly what R1-B's lateralised-ERD and peak-frequency
measurements are designed to settle, and it is the difference between "predict and route
these users elsewhere" (TODO A19) and "the pipeline is misconfigured for a third of
subjects" (TODO A6). Treat Finding 1 as settled and this as open.

**Finding 3 — alignment's benefit is uncorrelated with baseline ability.**
Pearson r = −0.16 (p = 0.68), Spearman ρ = −0.08 (p = 0.85), n = 9. Alignment is close to a
*uniform* +2 to +16pp correction, not a rescue mechanism for struggling subjects. Subject 5
gained the second-largest amount (+8.3pp) and still finished at 42.4% — the cleanest single
illustration that alignment is not the tool that fixes the weak group.

**Finding 4 — the "65.1% within-session upper bound" is wrong in kind, not merely
pessimistic.** Only 3/9 subjects (1, 4, 9) have cross-session accuracy above their
calibration-session CV; for 6/9 the calibration CV *over*-states what transfers (subject 7:
76.0% within vs 62.5% cross). The two means merely look contradictory (65.1% < 66.9%)
because they average different quantities. **A within-session estimate on the *calibration*
session neither bounds nor predicts cross-session accuracy.** The only meaningful ceiling is
within-session CV on the *evaluation* session — R1-C's job. The "within-session (upper
bound-ish)" row in `docs/progress_and_direction.md` (~line 1081) needs correcting once
R1-C reports.

**Verdict: ACCEPTED** — arithmetic on already-audited committed data, means reproduce
exactly, and Findings 1 vs 2 are kept distinct rather than elided.

**Consequence for the plan, stated now so it cannot be retconned later.** If R1-B finds the
weak group has *preserved* contralateral ERD, per-subject configuration (A6) becomes the
priority and the project's central thesis survives. If R1-B finds the weak group has *no*
lateralised ERD, the honest headline for a third of this dataset is "signal-limited, route
elsewhere", and any "beat the baseline" claim should also be reported against the
drift-limited subset. For subjects {1, 3, 4, 7, 8, 9} the aligned mean is
(0.8090 + 0.8160 + 0.6076 + 0.7847 + 0.8056 + 0.8403) / 6 = **77.7%** — nobody has computed
that number before, and it is the more honest target for the six subjects where drift is
actually the binding constraint.

### Process correction: do not audit a file an agent is still writing

Recorded because it cost real time and nearly produced a false bug report.

I read `experiments/char_01_iv2a_signal.py` at 383 lines and again minutes later at 729
lines, and `experiments/char_02_ceiling.py` at 660 → 595 → 680 lines across three reads.
In the first `char_01` read, `_band_erd_by_class` appeared to reference an undefined
`p_cue` (a NameError) — a plausible-looking bug that I was about to report. Verifying it
empirically before reporting (calling the function with dummy data, and re-reading the
function via `inspect.getsource`) showed the *current* file assigns `p_cue` correctly and
uses `cue_sl`: I had read a **torn, mid-write revision**. The bug did not exist.

**Rules adopted from this:**
1. Treat a file's mtime/size as part of its state. Before auditing, confirm the file is
   stable (two reads several minutes apart agreeing, or an explicit "final" from the agent).
2. Never report a defect based on a single read; reproduce it first (import the module and
   call the function, or run the script) — a torn read cannot survive execution.
3. Line numbers quoted at an agent must be accompanied by the identifying content, because
   an active editor invalidates them.

The one real defect I did confirm in `char_02_ceiling.py` was verified against the actual
CSV on disk (`astype(int)` on a series containing the string `'average'` → `ValueError`),
not merely inferred from reading. That is the standard.

### R1-B (`experiments/char_01_iv2a_signal.py`) — pre-delivery observations

Audited only far enough to check its assumptions; the file was still being edited, so this
is not a verdict. Verified independently:

| Assumption in the script | My independent check | Result |
|---|---|---|
| Data reaching the pipeline is in **microvolts**, so a 100 µV artifact threshold is meaningful | Loaded subject 1 narrow-window calibration directly: `min=-42.3, max=40.47, std=4.892, ptp=82.77` | **Confirmed µV** (would be ~1e-4 if volts) |
| The wide epoch is 1251 samples and the narrow session is 288 trials × 22 channels | Printed shapes: narrow `(288, 22, 501)`; script asserts 1251 for the wide window | consistent |
| `_band_erd_by_class` computes `(P_cue − P_base)/P_base` | Called it with dummy `(4, 22, 1251)` data — returns `(per_class, (n_trials, n_ch))` correctly | works |

**Caveat to carry into the write-up (not a defect):** the artifact features
(`frac_trials_abs_gt_100uv`, `..._150uv`) are computed on data MOABB has **already
band-passed to 1–40 Hz**, and on subject 1 the fraction of samples above 100 µV is
**exactly 0.0**. The script's own docstring flags these as a lower bound; expect them to be
near-degenerate across subjects, in which case they carry no information and should be
reported as such rather than as a null result. Confirm variance across all 9 subjects on
delivery.

### R1-A / R1-B / R1-C audit verdicts

*(pending — filled in when each agent reports; each entry will list the re-run command,
the reproduction result, and the verdict)*

---

## Rejected / corrected so far

| Item | What was wrong | Resolution |
|---|---|---|
| My own `ablation_01_calibration_burden.py` | pandas `agg` renamed the accuracy column; trials-to-threshold crashed | fixed, re-verified |
| `[exit code: 1]` reported for a passing script | PowerShell pipeline artifact, not a script failure | confirmed exit 0 by redirecting to a file; noting it because a false failure signal is itself a hazard |
