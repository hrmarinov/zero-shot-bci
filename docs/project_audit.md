# Project audit (2026-11-18): what have we actually achieved?

Written after a read of the README, `docs/progress_and_direction.md`, `docs/diag_gates_d1_d4.md`,
`docs/audit_log.md`, `docs/state_of_the_art.md`, `TODO.md`, the reports, and the TCFormer repository
(code, configs and per-subject result files; read as data, nothing run). It exists because the work was
reactive: each round answered the last round's problem, and no target was fixed in advance.

## The answer in plain words

We built something real, but narrower than the repository's name. **With a short calibration session, our
ensemble (R11) is clearly better than the classical decoder and than a single neural network, and that
holds on four datasets.** It needs half the labelled trials of a single network, and a quarter with
pretraining, and it can run live. **We have not built a calibration-free decoder** (the "zero-shot" in the
name), and **we have not shown we beat the best published decoder**: on the one dataset where we can compare,
a published single network scores higher (84.8 % vs 82.5 % on IV-2a). That comparison has never been run on
our own split, so we do not actually know our position.

## Scorecard: what is established, and how well

| Claim | Evidence | Strength |
|---|---|---|
| R11 beats CSP+LDA+RA | IV-2a +15.5 pp, IV-2b +3.2, lockbox +10.9 (12/12 up), Lee2019 +5.8 (41/10) | strong, 4 datasets |
| R11 beats a single ShallowConvNet | lockbox +3.7 (p = 0.002), Lee2019 +6.5 (49/5, p = 0.0002) | strong |
| The mixed ensemble beats a same-size CNN ensemble | dev +2.3, lockbox +1.3 (not significant), Lee2019 +2.9 (p = 0.0002) | moderate to strong |
| Half the labelled trials match a single CNN | lockbox confirmed (L2); with pretraining a quarter (P1, lockbox confirmed) | strong; pretraining hurts at the full session |
| Works live (causal alignment) | −0.3 pp dev (non-inferior), lockbox bound exactly −1.0 (not shown); no closed-loop test | moderate |
| Zero-shot (no calibration) | 55 % IV-2a, 65 % lockbox, 77 % IV-2b, a by-product of P1 | weak, never optimised |
| Competitive with the published state of the art | R11 82.5 vs TCFormer 84.8 on IV-2a; IV-2b gap is ocular (D6) | not shown; below on IV-2a |
| Population-data correction, drift trackers, self-training, spatial priors fail | many nulls, most at n = 9 (MDE above 3 pp) | real, but each only excludes large effects |

## What the protocol did well, and where it let us down

Well: evaluation labels never used for fitting; pre-registration committed before every real run; one
lockbox opening per claim; an EOG control; honest reporting of failed confirmations (A1, C1, E1).

Weaknesses:
1. **No target was fixed at the start.** The goal drifted from "zero-shot" to "a better ensemble with fewer
   labels". The first nine phases (wavelet calibration 31 %, prototypical networks 49 %) were dead ends before
   the ShallowConvNet on aligned input (R3) changed the picture.
2. **The 18 development sessions were reused for about 14 rounds** (R3-R13): winner's curse. The lockbox
   claims are the unbiased ones; BNCI2015_001 has been opened five times and is spent.
3. **Early rounds were underpowered.** With 9 subjects, effects below about +2 pp are undetectable (TODO §0).
   "Closed" for the small-effect ideas means "no large effect", not "no effect".
4. **We never measured the bar we compare against.** TCFormer's 84.8 % is from its paper. A1 ran TCFormer
   inside our trainer, which is not its recipe.
5. **Our "irreducible weak subjects" claim is not safe.** The council and D3 said subjects 2 and 6 are
   signal-limited, from within-session ceilings measured with *our own* linear feature family (8-30 Hz band
   power). TCFormer, trained on the same training session, scores 75.4 % on subject 2 and 72.9 % on subject 6
   (seed 0); R11 scores 66.0 % on both. TCFormer wins on subjects 2, 4 and 6 (+9.4, +6.6, +6.9 pp) and loses
   on 5, 7 and 8. Either those subjects carry usable signal outside our band (possibly ocular, see below) or
   our single-network recipe is weak for them.
6. **Our alignment is transductive** (it sees the test session's unlabelled trials); published models predict
   trial by trial. C1 shows the causal version costs only 0.1-0.3 pp, so this is small but worth stating.

## Leads missed, and obvious tests not run

1. **Reproduce the published recipe on our split.** TCFormer's official code is public. Its recipe differs
   from ours in the following ways: no band-pass (the whole 0.5-100 Hz recording), a per-channel z-score
   fitted on the training session, segment-and-recombine augmentation across trials, 1000 epochs, Adam
   (lr 9e-4, beta1 0.5) with warm-up and a schedule, 5 seeds, last epoch scored. Only two of these were ever
   tried (1000 epochs and z-score, both on IV-2b only, no gain). **Segment-and-recombine and the schedule
   were never tested, and nothing was tested on IV-2a.**
2. **Is the published IV-2a number partly eye movement?** On IV-2b, D6 showed that the 4-5 pp advantage of
   the broadband recipe is ocular. The same check was never done on IV-2a, where it decides whether the 84.8 %
   bar is neural. One run of the published recipe with the EOG channels regressed out answers it.
3. **Zero-shot was never pursued after the pivot.** TCFormer's own leave-one-subject-out result on IV-2a is
   63.1 % (seed 0, last epoch, from its result files); ours is 54.6 %. Our P1 pretraining was a 100-epoch
   by-product built for calibration, not tuned for zero-shot. The repository is named for this goal.
4. **Never tuned the CNN itself.** Learning rate, weight decay, epochs, window and band are the R3 defaults;
   per-subject choices by inner cross-validation (TODO A6) were not tried. Lee2019 (54 subjects) is now large
   enough to do this without fitting noise.
5. **Per-subject member weights and out-of-fold stacking** (A16): the equal weighting costs 0.5-1 pp on
   subjects where TS+LR is weak.
6. **Longevity** (A20, A31): accuracy over more than two sessions is unmeasured, and that is what a user
   living with the decoder would feel. The only multi-session set (Zhou2020, 7 sessions) is sealed as a
   lockbox; another multi-session set is needed, or the lockbox must be spent on this.
7. **No closed-loop session** ever; every "live" number is an offline replay.
8. **Items with no recorded status:** the TODO's 32 candidate items carry no done/closed marks (3 are
   annotated); FBCSP (A9), within-subject shrinkage (A8), alignment-estimator variants (A12), the illiteracy
   predictor (A19), cross-paradigm fingerprints (A21), self-supervised pretraining on the target's own
   unlabelled data (A30) appear untouched. Also never done: abstention on low-confidence trials.

## Housekeeping found

- `docs/progress_and_direction.md` summary is stale (still says CSP+LDA is the best classifier); a banner now
  points to the README.
- The D3 section was appended twice in `docs/diag_gates_d1_d4.md` and `TODO.md`; deduplicated.

## Proposed fixed targets (so rounds stop being reactive)

| # | Target | Done when |
|---|---|---|
| T1 | Know our position | TCFormer (official recipe, 5 seeds) and the same with EOG regressed, run on IV-2a, IV-2b and Lee2019 under our split; paired per-subject comparison with R11 |
| T2 | Accuracy | R11-family >= the reproduced TCFormer on Lee2019 (n = 54, MDE 1.9 pp) and on IV-2a/2b, no worse on either |
| T3 | Weak subjects | share of subjects >= 70 % on Lee2019 from 59 % to >= 70 %; bottom third up |
| T4 | Calibration | 48 labelled trials match the full-session single CNN (done on the lockbox; replicate on Lee2019) |
| T5 | Zero-shot | leave-one-subject-out with per-session alignment >= 63 % on IV-2a (TCFormer's own number), and a Lee2019 figure |
| T6 | Live | one closed-loop run, or an explicit statement that none is planned |
