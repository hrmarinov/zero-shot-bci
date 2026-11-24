> Raw reviewer note. Written by a subagent. Citations were checked only where the text says so; treat the rest as unverified.

# Council note: BCI domain expert + research strategist
Sources: D = docs/diag_gates_d1_d4.md, A = docs/project_audit.md, T1csv = results/t1_tcformer_head_to_head.csv,
T1py = experiments/t1_tcformer_head_to_head.py. "CPU-new" = computed by me now from cached probs
(scripts: scratchpad an.py / an2.py), POST HOC, descriptive, on E sessions already searched ~60 times (r14 report s1).

## 0. Headline finding (CPU-new, needs a pre-registered replication before anyone believes it)
The T1 TCFormer probabilities and R11 are strongly complementary. Probability-average of R11 (5 NN fits + TS+LR) and the
2-seed T1 TCFormer (seed-mean softmax), IV-2a, 9 subjects, all test-session-free:

| arm | mean | vs R11 82.45 | up/down | sign-flip p |
|---|---|---|---|---|
| TCFormer raw, 2-seed prob mean | 85.38 | +2.93 | 5/3 | 0.117 |
| TCFormer raw, single fit (= T1 table, D:1349) | 84.36 | +1.91 | 6/3 | 0.199 |
| 0.5 R11 + 0.5 TCF raw | 90.32 | +7.87 | 9/0 | 0.004 |
| (CNN3+TS+FTS2+TCF)/4, family-equal (like A1 aug_tcf) | 86.88 | +4.44 | 9/0 | 0.004 |
| 0.5 R11 + 0.5 TCF **eog** arm | 88.85 | +6.40 | 9/0 | 0.004 |
| 0.5 R11 + 0.5 TCF seed 0 only | 89.62 | +7.18 | 9/0 | 0.004 |
| TS+LR + TCF raw (2 fits) | 87.31 | +4.86 | 8/1 | 0.016 |

Weight sweep 0/.25/.5/.75/1 gives 82.4/86.9/90.3/87.7/85.4: flat-topped, so not a knife-edge weight. Per subject (0.5 blend):
95.1 80.6 95.1 92.7 83.7 82.6 99.0 92.0 92.0. Pitfalls handled: T1 stores braindecode class order, ours differs. Our
class k = T1 column [1,2,0,3] (one global map; it restores TCF accuracy to 85.4 and trial-order agreement with R11 is
57-90 %, so trial order matches; map = left,right,feet,tongue vs braindecode alphabetical; mapping inferred, not documented).
Caveats: (a) weight 0.5 and the two blends were looked at together (2 forks); (b) A1 found our-trainer TCFormer adds only
+0.57 (D:1013) because that TCFormer scored 74.54 alone (D:1007), the T1 recipe member scores 85, a different member;
(c) 1000-epoch last-epoch nets are probably overconfident, so a 50/50 prob average may be dominated by TCF (untested; a
temperature/rank check is free on CPU); (d) 8/9 eog-arm fits ran on Colab, not bit-identical (D:1368-1370 region).

## 1. Why TCFormer wins on s2, s4, s6 and loses on s5, s7 (CPU-new, per-class recall %, our class order L,R,feet,tongue)
| subj | R11 acc / TCF raw-2seed | R11 recall | TCF recall | seed agreement (raw s0 vs s1) |
|---|---|---|---|---|
| 2 | 66.0 / 76.7 | 69 33 89 72 | 78 82 92 56 | 81 % |
| 4 | 78.8 / 85.8 | 88 82 64 82 | 90 90 90 72 | 90 % |
| 6 | 66.0 / 74.0 | 68 58 74 64 | 82 88 62 64 | 83 % |
| 5 | 78.5 / 76.7 | 92 94 64 64 | 89 82 65 71 | 85 % |
| 7 | 95.5 / 94.1 | 100 100 92 90 | 83 99 97 97 | 96 % |
Others: seed agreement 92-96 % (s1,s3,s8,s9). Facts: (i) the gains are NOT in tongue (TCF is worse on tongue for s2, s4, s6);
(ii) s2 = R11 fails right hand (33 %, 31 of 72 right-hand trials called left, and ~99 of 288 trials called class 0: a
class-bias signature), s4 = R11 feet/tongue confusion (21 feet trials called tongue), s6 = diffuse. (iii) The seed
agreement is lowest exactly on the "TCF wins" subjects (81-83 % vs 95 %), so a 2-seed T1 estimate there has +-2-3 pp noise
(s2 raw seeds 76.0/72.2, eog 74.7/74.0, T1csv rows 7-10; s6 raw 70.1/70.8, eog 74.3/71.9); the "+6.9 on s6" in A:52 is
seed 0 published, the 2-seed raw number is +4.5. (iv) Where R11 wins (s5 -7.1 in D table, s7): TCF loses left-hand on s7
(83 vs 100) and right-hand on s5 (82 vs 94). s7 is a ceiling subject (95.5), s5 is the one subject where EOG removal costs
TCF 3.4 pp (74.8 -> 71.4, D:1349-1350), so part of TCF's s5 deficit is ocular-dependence lost, not neural.
(v) both-right overlap on s2 only 53 %, TCF-only-right 24 %, R11-only-right 13 %: complementary errors, which is why 0 is 90 %.

Mechanistic hypotheses, with the cheapest separating test (all on IV-2a, E-session untouched, trained on 0train only):
| H | Prediction if true | Cheapest ablation (GPU) |
|---|---|---|
| H1 per-session RA hurts weak subjects (RA on a drifting/non-stationary E session pulls whole class-bias, s2 shows class-0 bias) | TCF with z-score loses its edge when RA input is fed; R11 gains when RA is replaced by train-fitted z-score | Shallow with train-z-score instead of RA: D6 did it on IV-2b only (D:707, -1.94 pp), never on IV-2a. 9 subj x 3 seeds x 0.5 min = 14 min |
| H2 broadband 0.5-100 Hz (mu/beta harmonics, 38-100 Hz gamma) | feeding 4-38 Hz to TCF recipe drops s2/s4/s6 | TCF recipe with 4-38 Hz band: 9 subj x 2 seeds x 14 min = 4.2 h |
| H3 interaug (segment-recombine) as regulariser for 288 trials | TCF without interaug drops, FTSurrogate-like gain | TCF recipe minus interaug 4.2 h; Shallow + interaug 14 min |
| H4 long training / warm-up / cosine / Adam b1 .5 / last-epoch | TCF at 300 epochs, our R3 trainer: A1 gave 74.54 (D:1007) vs 84.4 recipe, so recipe-not-architecture is already established (+10 pp) | Shallow at 1000 ep did nothing on IV-2b (D:710); do on IV-2a, 14 min |
| H5 attention/TCN (architecture) | little: contradicted by the 10 pp A1-vs-T1 gap at identical architecture | skip |
| H6 ocular | contradicted on IV-2a: raw - eog = +0.50 pp, p 0.457 (D:1356) | already done |
| H7 R11's own ceiling is the 8-30 Hz-flavoured feature family (D3), s2/s6 labelled signal-limited wrongly (A:47) | TS+LR is 47.6 / 49.7 on s2/s6 (CPU-new, 69.25 mean) | none |
Most informative single GPU test is the TCF recipe with R11's front end (4-38 Hz + per-session RA) because it separates H1+H2
at once and its output is the member we would ship.

## 2. Route past 84.79 leak-free
Plausible stack: R11 + TCFormer-recipe members (2 seeds), possibly with RA input if E3 shows it does not hurt. Honest expected effects:
- T1 single recipe fit is an unselected estimate (published recipe, run once on our split), R11 is selected on this E over ~14
  rounds (A:15-16), so the observed +1.9 / +2.9 is if anything biased AGAINST TCF. The blend delta is not selection-inflated
  except for weight (flat) and for my looking at 2 blends.
- Observed blend delta +4.4 (family-equal) to +7.9 (50/50) on 9 subjects. Prior projects' replication: R11 effect
  shrank 2.1 -> ~1.2 pp in empirical-Bayes (r14 report s1); lockbox replication of A1 fell to non-significant (D:1030-1040).
  My shrunk judgment (not measured): true IV-2a gain +3 pp (plausible +1.5 to +5); Lee2019 gain smaller, +1 to +3, because TCF
  with ~100 labelled trials (D:1219) may be weak, its main risk. Beating 84.79 on IV-2a is then likely (blend 86-90 vs bar 84.8)
  but IV-2a E is not clean, so the claim needs a fresh set: Lee2019 (n=54, MDE 1.9, D:1247) and then Zhou2020 once (MDE 3.1).
- Interpretation of value: the project's claim becomes "diverse ensemble incl. a published-recipe net", not "we out-engineered TCFormer".

## 3. Pre-registered sequence (RTX 4060 8 GB; TCF fit 14 min on IV-2a-shape data; Lee/IV-2b costs are ESTIMATES, unmeasured)
| # | test | dataset | GPU h | metric / decision rule | decision changed |
|---|---|---|---|---|---|
| E0 | freeze the blend rule on paper: w=0.5 prob average, TCF = recipe, final epoch; also logit/temperature variant and one rank-average, report all three; no new fits | IV-2a cache | 0 | descriptive, writes the confirmation protocol before any new data | prevents forking paths on Lee2019 |
| E1 | cheap transplants onto ShallowConvNet in our pipeline: (a) train-z-score vs RA (H1), (b) +interaug (H3), (c) 1000 ep + warm-up cosine + Adam b1 .5 (H4), 3 seeds | IV-2a, then Lee2019 | IV-2a 0.25 each (0.5 min x 27); Lee2019 ~1.4 each | mean over subjects vs base RA 4-38 Hz; promote a piece if >= +1 pp IV-2a and >0 on Lee2019 n=54 Holm | tells whether interaug/schedule can improve R11's OWN members for free; kills or keeps H1 |
| E2 | TCF recipe leave-one-out: (i) 4-38 Hz + per-session RA input, (ii) no interaug; 2 seeds each | IV-2a | 4.2 each = 8.4 | per-subject vs recipe 85.4 and vs R11; rule: if (i) within 1 pp of recipe, RA-input TCF becomes the shipped member | fixes the front end of the member; separates H1-H3 for TCF |
| E3 | the blend itself: 1 seed of the recipe TCF on Lee2019 (session 1 -> 2, 20 sensorimotor ch as D2), blended 0.5 with cached R11 (data/cache/d2_lee2019) | Lee2019 n=54 | ~6-12 for 54 fits (est. 7-14 min each, 100 labelled trials is shorter), run 1 seed first, 2nd seed only if gate passes | primary: blend - R11 >= +1.5 pp with Holm p<0.05 one-sided AND blend - TCF alone > 0; MDE 1.9 pp (D:1247) so power for +1.5 is ~62 % (D:1248 table), so call it screen not confirmation | go / no-go for any further TCF spend; if TCF alone < R11 - 5 pp, abandon member idea |
| E4 | IV-2b TCF stage 2 (500 ep, no aug, 3 ch as T1py:44-46 states) raw and EOG-regressed arms, then blend | IV-2b | est 2-4 (3 ch, small input) | blend gain on raw vs eog arm; D6 predicts raw-eog ~ +4.6 pp (D:713) | ocular paper evidence + checks that member survives an EOG-clean world |
| E5 | EOG-clean blend on IV-2a/2b/Lee (regress where EOG exists) | as E3/E4 | 0 extra if E4 runs eog arm | blend_eog vs R11_clean (81.98 IV-2a, 79.81 IV-2b, r14 D:597-609) | leak-free headline must be reported EOG-clean beside raw (r14 D:648) |
| E6 | 2 extra seeds of E2 winner on IV-2a (to 4 seeds, near paper 5) | IV-2a | 4.2 | seed-mean vs 84.79 on 9 subj; only descriptive | the 'surpasses 84.79' sentence |
| E7 | single opening of Zhou2020 (20 subj, MDE 3.1) | sealed | ~4-5 (est.) | only if E3 effect >= 3 pp; pre-registered one test | the only confirmatory claim; do NOT touch before E3 passes |
Optional free check: BNCI2015_001 (spent, 12 subj, 2.8 h per seed) as non-confirmatory sanity of E3 direction.
Total GPU if all run: about 35-55 h; E0-E2 (~9 h) decide whether E3 is worth it.

## 4. Ocular-confound preprint: extra evidence needed beyond "IV-2b large, IV-2a null"
Have: IV-2b ocular share +4.57 pp, 8/1, p 0.004 (D:713); EOG-only CNN 52 % 4-class / 70 % 2-class (D:604); IV-2a R11 share
+0.46 pp (D:603 region), TCF share +0.50 p 0.457 (D:1356). Missing:
1. A published architecture, not just ShallowConvNet, shows it on IV-2b (E4; TCFormer raw vs eog). Otherwise reviewers say "ShallowConvNet-specific".
2. A specificity control: regress out 3 non-ocular nuisance channels / time-shuffled EOG; regression removes 33 % of IV-2a EEG
   variance but 7.4 % on IV-2b (r14, D:631), so variance-removal alone must be ruled out. IV-2b loses, IV-2a does not, which already helps.
3. Training vs feedback sessions: gaze is class-locked in IV-2b feedback sessions (D:726 reading); test T->T (screening) vs T->E drop of the broadband ocular share.
4. Equivalence not just non-significance for IV-2a: n=9, MDE >3 pp, so "null" is "no large effect"; need a TOST bound or a second EOG-bearing dataset (do not claim one exists; check Lee2019/other for EOG).
5. Alternate cleaners (ICA, SSP) beside least squares, plus a leave-train-only fit sanity (already train-only, D:1346 area).
6. A literature audit: list of papers that use the broadband 2b pipeline and which numbers would change (D:728-729 statement is not yet quantified).
This stays a short paper-supporting item; memory says paper is on hold (no institution), so prioritise E0-E3 first.

## Not done / not verified
No GPU run, no repo edits. Class-name mapping and all CPU-new numbers are post hoc. Lee2019/IV-2b TCF costs are guesses; only
the 14 min IV-2a figure is given.
