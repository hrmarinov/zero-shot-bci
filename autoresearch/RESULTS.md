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

## Round 2 and the Zhou2020 gate

**Plateau.** Round 2 (no class balancing) ran experiments 26-48 and kept none. The final candidate is round 1's winner without
balancing: (CNN3 + FTS2 + 2 x zero-shot pooled TS+LR) / 4. TUNE 77.26 % against R11 73.04 % (used for selection, so optimistic).
It was not scored on the Lee2019 holdout: the holdout number above (79.44 %) belongs to the version with balancing.
Balancing, reported separately: +0.33 pp on TUNE in experiment 11.

**Gate on IV-2a (`gate_iv2a.py`, committed before scoring): FAIL.** R11 82.45 %, candidate 77.70 % (-4.75 pp, 0 subjects up, 8 down);
with balancing 78.90 %. The candidate drops the subject's own TS+LR member and doubles a pooled member that is weak with 8 library
subjects and 4 classes (N9, N12).

**Zhou2020 was not opened and not downloaded.** It is 3-class with a same-montage library of 11 (subjects 1-12) or 7 (subjects 13-20),
closer to IV-2a than to Lee2019 (2-class, 53 subjects). Protocol rule 8 requires a stated expected effect that n = 20 can detect;
the transfer evidence points to a loss. No Zhou2020 claim is registered for this candidate.

## Cross-field round (`alien.py`, rules committed in f4eb630; every arm: `alien_results.csv`)

Keep rule: Lee TUNE >= +0.3 pp (more up than down) AND IV-2a not below the incumbent by more than 0.25 pp. All arms use no test labels;
the three test-time ideas are causal.

**Kept: dawid_skene, then gate_slot.** R11 -> stack: Lee TUNE 73.04 -> 74.59 % (+1.56 pp), IV-2a 82.45 -> 82.56 % (+0.12 pp, too small
to read as a gain). Stopped after round 3 kept nothing.

- The per-subject evidence gate gives the pooled decoder a mean weight of 0.43 on Lee TUNE and 0.12 on IV-2a.
- Alone on R11, gate_4th gave Lee +1.96 pp (17 up / 5 down) and IV-2a -0.46 pp; experts_4th Lee +2.00 (17 / 1) and IV-2a -0.69. Both fail
  only the IV-2a condition. The fixed-weight candidate of rounds 1-2 lost 4.75 pp on IV-2a.
- Discarded: spectral_meta and graph_diffusion lost on both cohorts.
- dawid_skene ignores member weights, so after it was kept, weighted variants (gate_double, gate_4th) count members unweighted.
- Limits: TUNE and IV-2a were both used for selection, so these numbers are optimistic; the two kept steps are each small; no untouched
  confirmation set exists apart from the sealed Zhou2020, which was not opened.
