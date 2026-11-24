> Raw reviewer note. Written by a subagent. Citations were checked only where the text says so; treat the rest as unverified.

# Council SOTA lens: what could beat TCFormer leak-free (2023-2026)

Verification key: [V] = I read a search summary or the repo doc this session; [U] = unverified, from memory or a snippet only. All GPU hours are for an RTX 4060 8GB. Search access was shallow (4 searches, no full-text reads), so full-text checks are listed per idea.

## Framing from the literature
- The leak-free published bar on IV-2a is 83-85 and nobody reports an EOG-controlled IV-2b number (project docs). Our TCFormer reproduction (84.36 raw, 83.85 EOG-regressed) already matches it. The gap to beat is statistical, not architectural: R11 and TCFormer are within noise on the pooled mean but differ by subject. The cheapest lever is therefore COMBINING them, not replacing either.
- Per-subject pattern: TCFormer wins 2, 4, 6 (weak, low-SNR subjects); R11 wins 5, 7. That is a diversity signal, the same mechanism the council already proved (diverse members beat size-matched members, 16 up / 0 down).
- Every number we make on IV-2a/2b is dev-set. Any claim needs the lockbox (BNCI2015_001 / Lee2019 / Zhou2020) opened once.

## 1. TCFormer as an ensemble member of R11 (highest expected value)
- Source: TCFormer, Sci Rep 2025, doi 10.1038/s41598-025-16219-7 [V, in repo docs].
- What: add the reproduced TCFormer (multi-kernel CNN + grouped-query transformer + TCN) as a 4th probability-averaged member next to ShallowConvNet, FTS-ShallowConvNet and TS+LR.
- Why: different inductive bias (temporal multi-scale + attention) from ShallowConvNet's log-variance prior, and it is stronger on 2, 4, 6. Errors on those subjects should be decorrelated from R11's.
- Cost: weights already trained from T1 reproduction; ~1-2 GPU-h for 3 seeds on 2b and Lee2019 if not cached; 0 h for IV-2a if predictions were saved.
- Pre-registered test: equal-weight average of (R11 members + TCFormer) vs R11 and vs a size-matched R11 control (same member count), IV-2a and IV-2b, s5 excluded and included. Metric: paired per-subject delta pooled over the 18 subject-datasets. Pass: >= +1.0 pp vs size-matched control, >= 12/18 subjects up, and mean delta on {2,4,6} > 0. Then confirm once on the lockbox.
- Learn: pass = TCFormer is a complementary view, adopt R11+TC as the new frozen pipeline. Fail = TCFormer's advantage on 2/4/6 is the same information the ensemble already has (shared ocular or low-frequency signal; check with the EOG-only member).

## 2. Per-subject member weighting by inner cross-validation on session 1 (stacking)
- Source: classic stacking; no BCI-specific citation claimed [U].
- What: pick member weights per subject from out-of-fold T predictions (5-fold on session 1), never touching E. Equal weights were best globally, but the subject-level sign flip (2/4/6 vs 5/7) says the optimum is subject-specific.
- Why: directly targets the subject heterogeneity; costs no E labels. Risk: with 288 trials per subject the weights are noisy, so shrink toward equal.
- Cost: requires OOF predictions from each member, ~3-5 GPU-h unless cached (TS+LR is CPU).
- Test: shrunk-weight ensemble (w = 0.5*equal + 0.5*OOF-softmax) vs equal weight on IV-2a/2b dev. Pass: >= +0.7 pp pooled and OOF weights correlate with E-session member ranking (Spearman > 0.3 across subjects). Learn: pass = subject-specific weighting is estimable from T; fail = the 2/4/6 vs 5/7 split is cross-session shift that T cannot reveal, which is itself a publishable negative.

## 3. Entropy-minimisation / adaptive-BN test-time adaptation on the ensemble members
- Source: Wimpff, Doebler, Yang, "Calibration-free online test-time adaptation for EEG motor imagery decoding", arXiv 2311.18520 [V]: RA(EMA) + entropy minimisation gives 79.74 +- 0.57 on IV-2a and 86.59 on IV-2b cross-session (their baseline, not ours). Related: T-TIME, arXiv 2412.07228 [V, abstract only].
- What: after per-session RA, update BN statistics on the unlabelled E trials and take a few entropy-minimisation steps on the CNN members (online, trial by trial, to remove our transductive advantage as a side effect).
- Why: we already do transductive RA, so the gain comes from the classifier side (BN stats and decision boundary shift). Most promising where session shift is large (subjects 2, 6) but dangerous where the model is near chance (entropy minimisation confirms wrong predictions).
- Cost: ~2-3 GPU-h (inference-time only, reuse trained members).
- Test: TTA on/off for FTS-ShallowConvNet, IV-2a and IV-2b. Pass: pooled >= +0.8 pp and no subject below -2 pp. Learn: pass = remaining shift is in the classifier; fail on weak subjects = labels too noisy, supports pseudo-label-free methods only. Gives also a per-trial (non-transductive) number, which is more comparable to the published numbers.

## 4. Richer augmentation inside the FTSurrogate family (FrequencyShift, channel dropout, interaug)
- Source: Rommel et al., "Data augmentation for learning predictive models on EEG: a systematic comparison", arXiv 2206.14483 [V]: 13 augmentations, FTSurrogate best, FrequencyShift also strong on BCI. Interaug: EEG-Conformer (Song et al. 2022; code read in repo docs) [V]. Channel dropout thesis (ZHAW 2022) [V, snippet only].
- What: replace the FTS member's single augmentation by a mixed-policy (FTSurrogate + FrequencyShift small shift + channel dropout p=0.1) and compare to interaug segment recombination.
- Why: IV-2b has 3 channels, so channel dropout is weak there; the benefit should be on IV-2a, subjects with low SNR. Augmentation was the only lever with a clean positive effect on the single model (FTS1 79.3 vs CNN1 75.3 on 2a).
- Cost: ~3 GPU-h for 2 arms, 3 seeds (budget cap from R14: <= 2 arms).
- Test: arm A = FTS+FrequencyShift, arm B = FTS+channel dropout vs FTS alone; paired over 18 subject-datasets with seed-averaged results. Pass: >= +0.8 pp for one arm AND > seed noise (0.5 pp). Learn: pass = augmentation space is not exhausted; fail = FTS already captures the useful invariance (amplitude/phase perturbation), so further augmentation is saturated.

## 5. RatioWaveNet-style learnable undecimated wavelet front end (TCFormer successor)
- Source: Siino et al., RatioWaveNet, arXiv 2510.21841 [V, abstract and search summary]; journal version in Array 30 (2026) [V, not read].
- What: TCFormer backbone with a learnable RDWT front end (shift-invariant multi-resolution subbands), then multi-kernel CNN + GQA transformer + TCN. Claims mainly worst-subject gains.
- Why: the paper reports gains of only +0.17 to +0.42 pp in worst-subject accuracy on 2a over its transformer backbone (arXiv), and the 3-7 pp claim in the Array abstract is not mapped to dataset or protocol [U]. Treat as low expected effect; only attractive as an architecture-diverse member.
- Cost: ~2 h per full run per the paper's 120 min on their GPU; ~4-6 GPU-h for 3 seeds (our 4060 may be slower).
- Test: add as member alongside TCFormer in idea 1. Pass: worst-3-subject mean (2, 4, 6) +1 pp vs TCFormer alone, same seeds. Learn: tells us if the wavelet front end matters at all versus just ensembling two transformers. Check: read Array paper for protocol and whether epochs were selected on test.

## 6. Checkpoint averaging / SWA and last-epoch hygiene
- Source: "Overcoming the BCI Calibration Bottleneck" (2026, arXiv 2607.16225, in repo docs) [U, 2-class, not checked]; TCFormer README states last-epoch, 5-run mean [per repo doc].
- What: SWA over the last 20% of epochs for the CNN members; no validation selection.
- Why: variance reduction on weak subjects (seed spread is the main noise on 2, 6) at zero data cost.
- Cost: ~1.5 GPU-h.
- Test: FTS-ShallowConvNet SWA vs last epoch, paired seeds. Pass: >= +0.5 pp and lower across-seed SD on {2,6}. Learn: whether the ensemble already averages away the variance (likely fail, which would confirm saturation).

## 7. Leak audit of the comparison set (protocol, not a method)
- Source: repo notes: EEG-Conformer reports max test accuracy over epochs (code read) [V]; CTNet uses a 30% validation split from T [V]; ATCNet code not retrieved [U]; CIACNet, SATrans-Net not checked [U].
- What: re-run TCFormer with early stopping on a T-only validation split, versus fixed last epoch, to show how much "validation-selected" costs on IV-2a.
- Why: tells us whether our R11 vs TCFormer comparison is truly apples to apples; the ranking could flip by +-1 pp depending on the protocol, which matters for claims.
- Cost: ~1-2 GPU-h.
- Test: TCFormer (last epoch) vs TCFormer (T-val selected) vs R11, paired over subjects. Decision: if the T-val selected TCFormer drops >= 1 pp, the reported 84.79 is not reachable under a stricter protocol and R11's relative position improves. Learn: gives defensible headline numbers.

## 8. EOG-controlled IV-2b for TCFormer and R11 (dataset-specific, 3-channel)
- Source: repo diag D6 [V]. No literature number exists for EOG-controlled IV-2b [V in docs].
- What: regress the 3 EOG channels out (fit on T only), then train TCFormer and R11 members; report 3-channel EOG-clean accuracy.
- Why: the 6.9 pp gap to TCFormer on IV-2b is partly ocular (we measured 4-5 pp). A clean TCFormer number may fall to or below R11's, making the honest IV-2b ranking different.
- Cost: ~2-3 GPU-h (TCFormer 3 seeds).
- Test: TCFormer raw vs TCFormer EOG-regressed on IV-2b; pass for "TCFormer's IV-2b lead is ocular": drop >= 3 pp with >= 7/9 subjects down. Learn: the real bar on IV-2b, which is the project's strongest paper-quality contribution.

## 9. Pretrained EEG foundation models for MI (LaBraM / CBraMod) as a negative-control member
- Source: CBraMod arXiv 2412.07236 [V]: balanced accuracy 0.514 on IV-2a (frozen/LOSO setting); independent controlled evaluation arXiv 2610.32067 reports frozen probes near 0.34-0.35 within-subject [V, snippet only, not read]; MI audit arXiv 2609.23924 [V, snippet]. JEPA-style variants already closed in repo.
- What: fine-tune CBraMod fully on session 1 per subject.
- Why: literature numbers are far below 80% in the subject-specific setting; I would expect no gain over ShallowConvNet. A cheap 1-arm check would close this lane definitively rather than rely on literature.
- Cost: ~3-5 GPU-h; probably memory-tight on 8 GB, skip unless idea 1 fails.
- Test: CBraMod full fine-tune on IV-2a vs FTS single. Pass: >= FTS-1 (79.3) minus 2 pp as a member improves ensemble > +0.5 pp. Learn: expected null; gives a closed-lane citation. Recommended: lowest priority.

## 10. SPD-network / Riemannian deep geometry (SPDNet-style)
- Source: SPDNet (Huang and Van Gool, AAAI 2017) [U, from memory, not re-verified]; no 2023-2026 MI-specific SPD network was verified in my searches.
- What: learn bilinear maps and ReEig on the 22x22 or 3x3 covariance, trained end-to-end, as an alternative to TS+LR.
- Why: could replace the TS+LR member with a stronger Riemannian one; on 3 channels the covariance has only 6 degrees of freedom, so I expect little on IV-2b.
- Cost: ~2 GPU-h (small); risky to implement on Windows without geoopt.
- Test: SPDNet vs TS+LR single on IV-2a only. Pass: >= +1 pp over TS+LR as single member and ensemble gain >= +0.5 pp. Learn: whether the Riemannian member's weak side (linear classifier) is the bottleneck.

## Recommended order (cheapest, highest information first)
1. Idea 1 (TCFormer member), CPU or cached. 2. Idea 7 + 8 (protocol and EOG controls; they decide what headline we can claim). 3. Idea 3 (TTA). 4. Idea 2 (stacking, only if idea 1 shows a subject-dependent effect). 5. Ideas 4-6, then 9-10 only if budget remains. Respect the R14 rule: <= 2 arms per round, lockbox opened once, size-matched control as baseline.

## Unverified items to flag
- Array 2026 RatioWaveNet 3-7 pp claim; ATCNet 85.38 protocol; CIACNet / SATrans-Net checkpoint rules; SPDNet citation; 2608.16134 (Multi-Feature Riemannian Hypergraph) not examined; 2607.16225 numbers; all foundation-model numbers are from search snippets.
