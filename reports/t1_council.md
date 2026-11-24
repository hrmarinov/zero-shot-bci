# T1 council: weak points, gaps and ways to pass TCFormer

**Date:** 2026-11-24. **Inputs:** T1 results (`docs/diag_gates_d1_d4.md`, last section), cached T1 predictions, and four
reviewer notes in [`t1_council/`](t1_council/): skeptic, state of the art, alien space, domain expert.

**Status of the numbers.** Everything computed here from cached predictions is **post hoc**. None of it is a selection or a
claim. Citations in the reviewer notes are checked only where the note says so.

## 1. What T1 settles, and what it does not

- The published TCFormer recipe is reproduced (raw 84.36 %, seed 0 85.15 % against 84.80 % published).
- The ocular share on IV-2a is small (0.5 pp, p = 0.46). The IV-2b result does not carry over.
- The TCFormer minus R11 comparison is **not settled**. With n = 9 the minimum detectable effect is about 3.9 pp
  (80 % power). The 95 % interval for eog minus R11 is about [−2.5, +5.3] pp. Seeds differ by 1.5–2.4 pp per subject.
- TCFormer's lead sits on subjects 2 and 6. Without them, eog minus R11 is −0.4 pp.
- The comparison is not like for like. R11 is a 5-model ensemble with transductive alignment. TCFormer was scored as single
  fits with no alignment. Averaging TCFormer's two seeds gives 85.38 % (raw) and 84.49 % (eog).
- T1 cannot say whether the recipe or the architecture wins. A1 (TCFormer inside our trainer, 74.5 %) against T1 (recipe,
  84.4 %) points to the recipe.

## 2. The main finding: the two decoders make different errors

Equal-weight average of R11 and the 2-seed TCFormer probabilities (post hoc, verified by us from the cache):

| IV-2a | accuracy | vs R11 | subjects up |
|---|---|---|---|
| R11 | 82.45 | | |
| TCFormer raw, 2 seeds | 85.38 | +2.9 | |
| R11 + TCFormer raw | **90.32** | +7.9 | 9 / 9 |
| R11 + TCFormer eog | 88.85 | +6.4 | 9 / 9 |
| family-equal 4-way (3 CNN, TS+LR, 2 FTS, TCFormer) | 86.88 | +4.4 | 9 / 0 |

- Weight sweep (0, .25, .5, .75, 1) gives 82.4, 86.9, 90.3, 87.7, 85.4. The top is flat, not a knife edge.
- An oracle that picks whichever model is right reaches about 95 %.
- Why: on subject 2 R11 fails right hand (recall 33 %, predicts class 0 for 99 of 288 trials). TCFormer recalls it at 82 %.
  On subject 4 R11 mixes up feet and tongue. On subjects 5 and 7 TCFormer loses a hand class instead.
- **Caution.** One dataset, test-session predictions only, two blends looked at together, and both models have a history on
  these sessions. The size is also unusually large for two models of this accuracy. It needs a size-matched control and a held-out
  confirmation before it becomes a claim. The class order of the T1 cache is [feet, left, right, tongue]; R11 uses
  [left, right, feet, tongue]. Remap before any blend.

## 3. Weak points, ideas and tests

Each row is one gap. State of the art (SOTA) and alien-space (ALN) ideas come from the reviewer notes.

| # | Gap | Idea | Test and rule | What we learn |
|---|---|---|---|---|
| G1 | Position against TCFormer is unmeasured at power | TCFormer recipe on Lee2019 (n = 54, MDE 1.9 pp) | R11 minus TCFormer, one seed; rule fixed before the run | Whether TCFormer is really ahead, or only on 2 of 9 subjects |
| G2 | The complementarity may not generalise | Blend R11 + TCFormer (SOTA, ALN: Dawid-Skene weights) | Lee2019: blend minus R11 ≥ +1.5 pp, Holm p < 0.05, and blend beats TCFormer alone; size-matched control | Whether the final model is a 4th family or a new ensemble |
| G3 | We do not know which part of the recipe helps | Transplant pieces onto ShallowConvNet: z-score against alignment, interaug, schedule, broadband | IV-2a, one piece per cell, about 0.25 h per cell | Which cheap piece to add to R11 without TCFormer's cost |
| G4 | R11 is transductive, TCFormer is not | Causal R11 (C1) against TCFormer; adaptive BatchNorm and entropy minimisation (SOTA) | CPU: causal R11 + TCFormer blend on IV-2a | Whether the blend survives a live setting |
| G5 | Class bias on weak subjects (subject 2) | Check per-session alignment under drift; per-class confusion before and after | CPU on cached predictions | Why R11 fails these subjects; whether D3's "signal-limited" label is wrong more widely |
| G6 | Ensemble weights are equal | Dawid-Skene or out-of-fold weights, shrunk to equal (SOTA, ALN) | IV-2a and Lee2019: ≥ +1 pp over equal weights | Whether weights are worth the complexity |
| G7 | Robustness of the covariance member | Tyler or Huber covariance in TS+LR and in the alignment mean (ALN, radar; present in pyRiemann 0.12) | CPU on Lee2019: ≥ +1 pp, ≥ 70 % of subjects not negative | Cheap and unrelated to TCFormer |
| G8 | Sample-limited ceiling (D4) | Mixup, SAM, FixMatch-style consistency on the unlabelled test session, self-supervised pretraining on the subject's own inter-trial EEG | Lee2019 one-day screens; effects of +0.5 to +2 pp, so stack G6–G8 into **one** hypothesis | Whether any variance-reduction lever beats the 1.9 pp MDE |
| G9 | IV-2b is the weak spot (3 channels) | Run TCFormer raw and eog on IV-2b | IV-2b, 2–4 h | Whether the ocular finding holds for TCFormer, which the preprint needs |
| G10 | Evidence for the preprint | Specificity control (shuffled-EOG regression), training against feedback sessions, equivalence bound for the IV-2a null, ICA or SSP as other cleaners | See §5 | Whether the IV-2b claim survives a hostile reviewer |
| G11 | Winner's curse and spent lockbox | Lead with Lee2019 and Zhou2020 numbers, not IV-2a | Zhou2020 opened **once**, only if the Lee2019 effect is ≥ 3 pp (MDE 3.1 at n = 20) | An unbiased final number |

Foundation-model fine-tuning (CBraMod and others) scored far below 80 % in subject-specific settings in the sources the
SOTA reviewer found. Keep it closed. Also closed: active learning, subject-pooling meta-learning, domain-adversarial
training, anything that compares covariances across subjects.

## 4. Order of work

1. **Freeze the rules on paper (0 h).** Write the blend rule, the control and the decision rules in a pre-registration
   commit before any new fit. The 0.5/0.5 blend was found post hoc and cannot also be the confirmation.
2. **CPU, free, now:** G4, G5, and the Dawid-Skene part of G6 on IV-2a; G7 on Lee2019. Each gives an actionable answer
   without GPU.
3. **GPU, about 9 h:** G3 on IV-2a, then recipe leave-one-out on TCFormer's front end (about 4.2 h each).
4. **GPU, about 6–12 h:** G1 and G2 on Lee2019, one seed first.
5. **GPU, 2–4 h:** G9 on IV-2b.
6. **Only then:** a second seed on IV-2a (4.2 h), and one Zhou2020 opening.

Total about 35–55 GPU hours. Steps 1–3 (about 9 h) decide whether step 4 is worth running.

## 5. What the preprint needs

The ocular-confound preprint needs more than "large on IV-2b, null on IV-2a":
1. TCFormer itself showing the IV-2b effect, not only ShallowConvNet (G9).
2. A specificity control. Regressing EOG removes about a third of IV-2a EEG variance, so a shuffled-EOG regression must
   remove less accuracy.
3. A comparison of training and feedback sessions on IV-2b.
4. An equivalence bound, or a second EOG-bearing dataset, to support the IV-2a null.
5. A second cleaner (ICA or SSP).
6. A quantified literature audit of which published IV-2b results use broadband input.

The blend result is a second, separate paper-sized claim. Do not mix it into the ocular preprint until it is confirmed.

## 6. Known gaps in this review

- The SOTA reviewer read four search results and no full texts. The alien-space reviewer did no web search. Treat all
  citations as unverified until you open them.
- Costs on Lee2019 and IV-2b are estimates. Only the IV-2a fit time (14 min) is measured.
- The class-order mapping used by the reviewers was inferred and then checked by reproducing every T1 accuracy to 1e-9.
