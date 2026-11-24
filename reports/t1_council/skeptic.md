> Raw reviewer note. Written by a subagent. Citations were checked only where the text says so; treat the rest as unverified.

# Council review: statistical skeptic

Sources: D = docs/diag_gates_d1_d4.md, A = docs/project_audit.md, T1csv = results/t1_tcformer_head_to_head.csv.
"CPU" = computed by me from cached predictions (scratch scripts only, no repo edits).

## 1. What T1 does and does not establish

Reproduction guard (CPU). I rebuilt R11 per-trial probabilities for IV-2a as (mean of r3 cnn_ra seeds 0-2 + TS-LR + mean of r8
ftsurrogate seeds 0-1)/3 (the "fts" arm, experiments/r11_round3_diverse_members.py:118-124). All 9 subject accuracies match
results/r11_round3_diverse_members.csv exactly (mean 82.45). T1 cached preds use braindecode class order [feet, left, right,
tongue], R11 uses [left, right, feet, tongue]. I remapped columns and checked that all 36 cached fits reproduce T1csv accuracy
to 1e-9. Anyone reusing t1_preds against R11 labels without this remap gets ~23% (chance).

Numbers (CPU, n=9 subjects, paired):

| contrast | mean pp | SD of diff | up/down | t p | exact sign-flip p | d_z |
|---|---|---|---|---|---|---|
| eog (2-seed mean) - R11 | +1.41 | 5.10 | 5/3 | 0.43 | 0.43 | 0.28 |
| raw (2-seed mean) - R11 | +1.91 | 4.01 | 6/3 | 0.19 | 0.20 | 0.48 |
| eog 2-seed prob-ensemble - R11 | +2.04 | 4.61 | 6/3 | 0.22 | - | 0.44 |
| raw 2-seed prob-ensemble - R11 | +2.93 | 4.63 | 5/3 | 0.09 | - | 0.63 |

- Matches the doc (D lines 1355-1357). 95% t-CI for eog-R11 is [-2.5, +5.3] pp; subject-cluster bootstrap [-0.9, +4.8].
  The data are compatible with TCFormer being 2.5 pp worse and 5 pp better. T1 is "inconclusive" and nothing more.
- Power: with SD of diff ~5 pp, n=9 gives an 80%-power MDE of ~3.9 pp. For d_z 0.3/0.5/0.8, power is 13%/26%/56%. The T1 rule
  (>= +1 pp AND >= 6/9 up) is a pass on neither side at a realistic effect; "NEURAL not shown" is NOT evidence of "ocular".
- Seed variance (CPU, cached preds): per-subject seed-0 vs seed-1 accuracy differs by 2.35 pp on average (raw, SD 2.8) and 1.5 pp
  (eog, SD 2.0). Single-seed raw subject 5 is 78.1 vs 71.5. Prediction agreement between seeds is only ~90% of trials (raw 90.0,
  eog 89.9). Seed means over 2 seeds still carry ~1-1.5 pp SE per subject; the doc's per-subject table (D 1349-1351) should not be
  read at 1 pp resolution. Seed 0 alone: raw 85.15, eog 83.80; seed 1: raw 83.56, eog 83.91.
- Ensembling the two TCFormer seeds helps: raw 84.36 -> 85.38 (+1.0), eog 83.85 -> 84.49 (+0.6). R11 is already a 5-model ensemble
  plus TS-LR, TCFormer was scored as 2 single models. The like-for-like comparison is ensemble vs ensemble: +2.0 (eog) to
  +2.9 (raw) in TCFormer's favour, still p>0.09.
- Per-subject pattern (CPU, D 1349-1351): TCFormer wins big on 2 (+8.3 eog), 4, 6 (+7.1); R11 wins on 5 (-7.1), 7 (-3.7), 8.
  Excluding subjects 2 and 6 the eog-R11 mean is -0.40 pp (raw +0.64): the whole T1 lead sits in two subjects.
- NEW FINDING, not in the docs: the two decoders are highly complementary. Averaging TCFormer 2-seed probabilities with R11
  probabilities (equal weight, no fitting) gives raw+R11 = 90.32%, eog+R11 = 88.85% on IV-2a, vs R11 82.45 and TCFormer
  ensemble 85.38/84.49. All 9 subjects improve over R11 (paired p = 0.003/0.007), all 9 over the TCFormer ensemble (raw, p<0.001).
  Oracle (either right) is 95.1%; both wrong only 4.9-5.4% of trials. Caveats: test-session preds only, no tuning, but equal-weight
  combination of two arbitrary fixed models is not selected on the test set, so selection bias is small. R11 is the 5th-6th
  decoder-family experiment on this set (D 1371-1373 style winner's curse), TCFormer is not; both favour caution on exact sizes,
  not on direction. Existing project note: A1 found CTNet "passes screen, NOT CONFIRMED on lockbox" (D 979), i.e. an
  architecture-diverse member has failed before; TCFormer-as-member has not been lockbox-tested.
- Recipe vs architecture: T1 cannot separate them. The raw arm changes six things at once vs R11 (broadband input, z-score,
  interaug, 1000 epochs/Adam schedule, non-aligned input, a different network; A lines 60-65). No arm trains the TCFormer
  network in R11's recipe, nor ShallowConvNet in TCFormer's recipe. Also asymmetry: TCFormer sees no test-session statistics
  (no transductive alignment), R11 does (A line 55, D 1055-1070).
- Provenance: eog subjects 1-8 ran on Colab, the rest on an RTX 4060 (D 1371-1373), non-bit-identical. Same-subject seed noise
  (1.5-2.4 pp) is larger than any plausible GPU effect, so this is a minor issue but the eog-vs-raw contrast (+0.5 pp) is
  confounded with hardware for 8 of 9 subjects.
- "Not ocular on IV-2a" (raw-eog +0.50 pp, p 0.46) is also underpowered: CI roughly +/-3 pp; do not quote it as a finding of
  absence. Also the EOG-regression check removes linear leakage only (training-session coefficients), not artefact-correlated
  neural activity.

## 2. Ranked weak points

1. **R11 is not the best known decoder and the gap to a published recipe is unmeasured at power (HIGH).**
   Evidence: T1 +1.4 to +2.9 pp, CI spans 0 (CPU above); A line 28 "below on IV-2a, not significant at n=9"; A target T1 still has
   IV-2b and Lee2019 open (A line 96). Cheap test: run the same T1 raw recipe on Lee2019 dev (54 subj, MDE ~1.9 pp, A line 98),
   1 seed, EOG not needed (no EOG there). Metric: paired per-subject acc vs R11. Rule: TCFormer - R11 >= +1.9 pp and p<0.05 ->
   TCFormer is better; |diff|<1 pp with CI inside +/-1.9 -> parity. Decision: whether to claim competitiveness or switch base model.
2. **Complementarity is untested outside IV-2a test preds (HIGH, opportunity not weakness).** Evidence: CPU +7.9 pp, 9/9 subjects.
   Too large to be taken at face value from one dataset and 2 seeds. Cheap test: same equal-weight R11 + TCFormer average on
   Lee2019 (needs 1 TCFormer seed per subject; preds already needed for test 1) and on the lockbox, ONE opening, pre-registered.
   Rule: ensemble - R11 >= +2 pp with >= 75% of subjects up on Lee2019 -> adopt TCFormer as a 4th R11 member; also compare to
   a size-matched TCFormer-only seed ensemble (3 seeds vs 3+1) to rule out "just more models". Decision: promote TCFormer to member,
   or not. Note also R11 + TCFormer weights are not tuned: no tuning budget needed.
3. **Winner's curse on the dev set (HIGH).** Evidence: 18 dev sessions reused for ~14 rounds R3-R13; lockbox spent after five
   openings (A lines 40-42). R11 82.45 is an upper-biased estimate of its true IV-2a accuracy, so the true TCFormer gap is
   likely larger than +1.4. Cheap test: none needed on data; report instead R11 on Lee2019 (74.5, never used for selection) and
   lockbox-free numbers only. Rule: when quoting a headline, quote the Lee2019/lockbox figure, flag the IV-2a one as selected.
   Decision: how to word the README claim (currently 82.45 first).
4. **Transductive alignment vs inductive comparator (MEDIUM).** Evidence: A line 55; C1 causal cost "-0.3 pp dev; lockbox bound
   exactly -1.0, not shown" (A line 26). The comparator (TCFormer) uses no test statistics. Cheap test (CPU, cached): recompute
   R11 causal on IV-2a and compare to TCFormer in same table; if causal R11 = 82.1, TCFormer lead grows to +1.7/+2.2 pp.
   Rule: report both; decision: which R11 is the baseline in all head-to-heads (causal is the deployable one).
5. **Lockbox is single-dataset and spent (MEDIUM).** Evidence: BNCI2015_001 opened five times (A line 41), 12 subjects, strong
   numbers (+3.7 p 0.002) but any new claim has no clean confirmation set. Zhou2020 sealed. Cheap test: pick one new held-out set
   (e.g. a second MOABB MI set not used) and freeze R11 + the planned ensemble; compute. Rule: pre-register, open once.
   Decision: whether any further claim (e.g. TCFormer member) may be called confirmed.
6. **Weak-subject narrative was wrong once; rest of D3/D4 rests on a linear-feature ceiling (MEDIUM).** Evidence: D3 said s2 and s6
   are signal-limited, T1 refutes (A 47-54). D4/E1 conclusions on Lee2019 inherit the same band-power-ceiling logic. Cheap test:
   run TCFormer single seed on the Lee2019 bottom third (about 18 subj) and compare to R11/R10 there. Rule: if TCFormer >= R11
   +3 pp on >= 70% of bottom-third subjects, D3's "signal-limited" verdict is retired. Decision: whether more weak-subject work
   is worthwhile.
7. **Seed count and n=9 invalidate small-effect nulls (MEDIUM).** Evidence: A line 29-30 (MDE above 3 pp for most nulls);
   T1 itself uses 2 seeds when the audit target specifies 5 (A line 96); CPU seed SD 2-3 pp. Cheap test: for any decision that
   rests on a < 3 pp contrast, require 3 seeds + the 54-subject set. Rule: no "closed" verdicts below 3 pp at n=9.
   Decision: stops closing ideas on noise.
8. **Missing baselines (LOW-MEDIUM).** FBCSP (A8 in TODO), within-subject shrinkage, per-subject member weights are untouched
   (A lines 75-84). Cheap test: FBCSP+LDA with RA on IV-2a/Lee2019, CPU. Rule: R11 - FBCSP >= 5 pp on Lee2019 else the
   "classical baseline" in the README is too weak. Decision: baseline wording only.
9. **T1 provenance and recipe fidelity (LOW).** 5 seeds used by the paper vs 2 here, mixed GPU, "last epoch" scored. Cheap test:
   rerun eog arm seeds 2-4 for 9 subjects. Rule: if 5-seed eog-R11 mean CI excludes 0, flip the claim. Decision: solidifies
   (or removes) the "TCFormer is ahead" statement before it enters the README.

## 3. What each test changes (one line each)

- Test 1 (Lee2019 TCFormer): whether the project claims parity, or re-bases on TCFormer.
- Test 2 (R11 + TCFormer, equal weight, Lee2019 then lockbox once): adopt TCFormer as an R11 member, or stop. Highest value per
  GPU hour given the CPU +7.9 pp already visible; the size-matched control is mandatory.
- Test 3 (wording): which number goes first in the README.
- Test 4 (causal R11 in the head-to-head, CPU only): which R11 is the baseline; free to run now.
- Test 5 (new held-out set): whether anything beyond "R11 beats CSP+LDA+RA" may be called confirmed.
- Test 6: whether weak-subject research continues.
- Tests 7-9: guard against closing/opening ideas on noise.

## Reproducing my CPU numbers

Scratch scripts (not in repo): $TEMP/sk/a.py rebuilds R11 probabilities per subject (imports r11 script functions, matches
CSV exactly); $TEMP/sk/b.py computes the table above (class remap [2,0,1,3] columns, labels [1,2,0,3][ye]).
