# Sealed lockbox: Zhou2020 (PRE-REGISTERED, data not downloaded)

Written and committed BEFORE any Zhou2020 file is downloaded or inspected. BNCI2015_001 is exhausted (opened
5 times), so the next claim family needs a fresh dataset. Only MOABB source metadata was read to write this.

**Dataset** (MOABB `Zhou2020`, Zenodo 10.5281/zenodo record in `moabb/datasets/zhou2020.py`): 20 subjects,
7 sessions each (days apart), 500 Hz, MI of left hand / right hand / feet plus an idle "rest" class.
S-subjects 1-12: 41 EEG channels (unnamed `EEG1..EEG41` in MOABB, so no channel can be selected by name);
A-subjects 13-20: 26 named EEG channels (F3..CP6 layout).

**Fixed rules**
1. One opening per claim family. A claim family is a pre-registered set of contrasts; reruns are forbidden. An
   implementation bug found afterwards is fixed in a separate commit that leaves the protocol unchanged.
2. Classes: left_hand, right_hand, feet (3-class). `rest` is excluded (a different paradigm: no cue-locked
   imagery). Chance 33.3 %.
3. Channels: all EEG channels of the subject (41 or 26), no selection; models are per subject so the count may
   differ between subjects.
4. Window as the other datasets: resample 250 Hz; CNN 4-38 Hz, 0-4 s post-cue; TS+LR 8-30 Hz, 0.5-2.5 s;
   pipeline `src.pipeline` unchanged, with whatever configuration the claim names, frozen before opening.
5. Split: train on session 1, test on each of sessions 2-7, per-session Riemannian alignment (never mixed across
   sessions). Per-subject score = mean accuracy over test sessions 2-7 (primary); session 2 alone (secondary,
   the IV-2a-like next-day setting) and the session-7 score (drift, secondary).
6. Primary test: one-sided Monte-Carlo sign-flip over the 20 subjects (exact, n = 20), contrasts and Holm set
   fixed with the claim; a gain is confirmed only if pooled >= +0.5 pp, Holm p < 0.05 and the effect is > 0 in
   both subject groups (S and A). Non-inferiority: one-sided 95 % bootstrap lower bound > -1 pp, strict.
7. Co-primary endpoints reported for every claim: mean, share of subjects >= 70 %, bottom-third mean (the 7
   subjects with the lowest csp_lda accuracy).
8. MDE at n = 20 is ~1.9x the n = 54 value; the claim must state its expected effect against it BEFORE opening
   (power analysis from `experiments/d2_lee2019_dev.py`). Claims that cannot be detected at n = 20 are not opened.
9. No tuning, selection or early stopping on any Zhou2020 label before the opening.
10. Reserve lockboxes if Zhou2020 proves unusable (not verified, not opened): Yang2025, Zuo2025.

The loader and the claim script are written when a claim exists, with a `--dry-run` on MOABB `FakeDataset`.
