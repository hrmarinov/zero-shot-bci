# autoresearch run, 27 November 2026

Rules: [program.md](program.md). Every experiment: [results.tsv](results.tsv). Holdout scores: `final_holdout.csv`.

**Outcome: both pre-registered holdout criteria are met.** On the 27 holdout subjects (never read by the loop):

| Arm | holdout accuracy | against R11 |
|---|---|---|
| R11 | 76.04 % | - |
| reference (R11 + pooled member, known from N8) | 78.04 % | +2.00 pp, p = 0.0034 |
| loop candidate | 79.44 % | +3.41 pp (19 up / 6 down), p = 0.0005 |

Candidate against reference: +1.41 pp (16 up / 6 down), p = 0.0084. P values are one-sided sign-flip.

**Candidate:** CNN3, FTS2 and a zero-shot pooled TS+LR member at weight 2; R11's own TS+LR member removed; class probabilities balanced
over the test session. 25 experiments, 5 kept. Tune score 73.04 -> 77.59 % (optimistic: it was used for selection).

Limits:
- The holdout is clean for this loop only. N7-N12 had already used all 54 subjects, and the first kept idea (the pooled member) came from them.
- Class balancing needs the whole test session at once and equal class counts. It does not apply to online, trial-by-trial decoding.
  The holdout run does not separate its share of the gain. S1 had not adopted it on its own (+0.70 pp, Holm p 0.060).
- Lee2019 only. The pooled member failed to replicate on IV-2a (N9, N12).
- The CNN members are still trained on the subject's own session 1, so the pipeline is not calibration-free.
- The holdout is spent. It must not be used to choose among further variants.
- Zhou2020 was not opened.
