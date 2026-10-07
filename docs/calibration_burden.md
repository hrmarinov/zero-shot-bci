# Calibration burden: how much labelled data does the working method actually need?

**Status: measured, audited, committed.** Source data
[`results/ablation_calibration_burden.csv`](../results/ablation_calibration_burden.csv)
(2,727 rows: 9 subjects × 3 conditions × 6 budgets × 20 repeats), summary
[`results/ablation_calibration_burden_summary.csv`](../results/ablation_calibration_burden_summary.csv).

This is TODO A29. It answers a question the project had never asked in six phases: every
accuracy it has ever reported was measured at **one** calibration budget — all 288 labelled
calibration trials. Here the budget is swept, stratified (n/4 trials per class) and repeated,
with accuracy always measured on the same full 288-trial evaluation session.

## Why this was worth measuring

Three reasons, in order of how much they ended up mattering:

1. **It is the metric the field and industry actually optimise.** Neuralink's headline result
   is a calibration-burden reduction (10 min/day → 10 min/week), not a peak-accuracy number —
   see [`docs/progress_and_direction.md`](progress_and_direction.md) §"External literature
   check".
2. **The project's best method uses zero labels from the evaluation session.** That suggested
   a specific, testable hypothesis: alignment's advantage over unadapted CSP+LDA should
   *grow* as the labelled budget shrinks.
3. **No such curve existed.** `phase2b` sweeps the *eval-side reference* size, never the
   *training* size.

## The result

Mean accuracy on the full evaluation session, 9 subjects, chance 25.0%:

| Labelled calibration trials | 32 | 64 | 96 | 144 | 192 | 288 |
|---|---|---|---|---|---|---|
| no adaptation | 51.04% | 56.37% | 59.08% | 60.79% | 61.68% | **62.23%** |
| **+ Riemannian alignment** | 53.82% | 60.10% | **62.18%** | 64.74% | 65.49% | **66.94%** |
| alignment gain | +2.78pp | +3.73pp | +3.11pp | +3.95pp | +3.81pp | +4.71pp |
| Wilcoxon p (n=9) | 0.027 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 |
| subjects improved | 7/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |

Harness identity: at n=288 the two conditions reproduce the committed Phase 1 and Phase 2
means **bit-exactly** (`0.6222993827160493`, `0.6693672839506174`), which is what makes every
other row in this table comparable to the published baseline.

### 1. The motivating hypothesis is refuted

Alignment helps at **every** budget, and significantly from n=64 upward (9/9 subjects,
p=0.0039 — the exact floor for n=9). But its gain does **not** grow as labels shrink: it is
**smallest at the smallest budget** (+2.78pp at n=32) and **largest at the full budget**
(+4.71pp at n=288).

That is the opposite of the prediction, and the reason is informative rather than
disappointing: the gain is not mainly a *label-scarcity* effect. Alignment corrects a
session-level covariance shift, and the better-trained the classifier is, the more it can
exploit the corrected covariance. Consistently, a diagnostic condition below shows the
alignment *reference* is not the thing that degrades at small n — the classifier is.

### 2. What survives is a data-efficiency result

**96 labelled calibration trials with alignment match what unadapted CSP+LDA achieves with
all 288** — 62.18% vs 62.23%, a difference of −0.05pp, 3 subjects up / 6 down, p=0.43
(indistinguishable). That is a **3× reduction in calibration burden for equal accuracy**, and
it is a cleaner, more product-relevant claim than "66.9% vs 62.2%".

Read it precisely, because the two halves point in opposite directions:

| Claim | Verdict |
|---|---|
| aligned@96 = none@288 (parity) | **supported** (−0.05pp, p=0.43) |
| aligned@96 = aligned@288 | **refuted** (−4.75pp, p=0.0039) — you do lose accuracy relative to the full-budget aligned method |
| aligned@144 beats none@288 | not established (+2.51pp, 5 wins / 3 losses, p=0.38) |
| aligned@192 beats none@288 | marginal (+3.26pp, 7/2, p=0.074) |
| aligned@288 beats none@288 | **established** (+4.71pp, 9/0, p=0.0039) |

So the honest summary is: **alignment buys back the 192 trials you did not collect — not
more.** The "22% fewer trials" style of headline is available here in a much stronger form
(3× fewer, for parity), but only as *parity*, not as a free accuracy gain.

### 3. The alignment reference is not a data bottleneck

A diagnostic condition (`aligned_ref_full`) estimates the alignment reference from the
**full 288 unlabeled** calibration trials while training the classifier on only n labelled
ones. It differs from `aligned` by **+0.05 to +0.22pp across every budget** — i.e. by roughly
half an evaluation trial:

| n | 32 | 64 | 96 | 144 | 192 | 288 |
|---|---|---|---|---|---|---|
| aligned vs aligned_ref_full | +0.11pp | +0.22pp | +0.12pp | +0.05pp | +0.16pp | 0.00pp |
| Wilcoxon p | 0.63 | 0.039 | 0.14 | 0.66 | 0.031 | 1.0 |

Two of those p-values are nominally "significant" while the effect is 0.2pp ≈ half a trial.
This is a textbook case of statistical significance without practical significance, and it is
reported rather than hidden: **accumulating extra unlabeled calibration data buys essentially
nothing for the alignment reference.** At deployment, the reference can be estimated from the
same handful of trials as the classifier.

### 4. Per-subject caveat: the parity result is not uniform

`aligned@96` minus `none@288`, per subject: **+14.51pp** (subject 7), +2.55 (3), +0.42 (5),
−0.97 (1), −1.28 (8), −1.55 (2), −3.52 (9), −5.07 (4), −5.50 (6). The aggregate parity is
held up by subject 7's very large gain; six of nine subjects are slightly *worse* at a third
of the budget. So the 3× claim is an average effect and should always be quoted with this
breakdown.

## Caveats

- **Sweep granularity.** Only {32, 64, 96, 144, 192, 288} are visited, so the true
  parity crossing point lies in the interval (64, 96]. No interpolation is claimed.
- **Repeats.** n<288 uses 20 independent stratified draws per subject (subsampling noise is
  averaged out); n=288 is the single deterministic full-session draw, matching Phase 1/2.
- **Inferential unit.** n=9 subjects, so the Wilcoxon floor is p=0.0039 and differences below
  ~2pp average are not detectable. Effect sizes in pp are reported next to every p-value for
  this reason.
- **Scale.** At 288 evaluation trials, one trial = **0.35pp**. Per-subject differences of one
  or two trials are noise by construction.
- **Evaluation-side budget is a separate axis.** This sweep varies the *labelled training*
  budget. The *unlabeled evaluation-side reference* size was already swept by
  `experiments/phase2b_spd_shrinkage.py`, and these two axes are not interchangeable.
- **The evaluation session's unlabeled trials are still used** for its own whitening — the
  standard transductive design of Riemannian alignment, unchanged here.

## Reproduction

```powershell
.venv\Scripts\python experiments\ablation_01_calibration_burden.py        # ~25 min, all 9 subjects
.venv\Scripts\python experiments\ablation_01b_calibration_burden_analysis.py   # seconds
```

The analysis script asserts the harness identity against the committed baselines and exits
non-zero if it fails, so a silent protocol drift cannot produce a plausible-looking curve.
