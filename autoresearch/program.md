# autoresearch for calibIllit (adapted from github.com/karpathy/autoresearch, MIT; upstream 228791f)

Loop: change `candidate.py` -> run `prepare.py` -> keep (commit) or discard -> log in `results.tsv`. Rules below were committed BEFORE
the first experiment. Only `candidate.py` is edited by the loop; `prepare.py` is the fixed harness.

- DATA: Lee2019_MI, session 1 -> 2, cached. CPU only (CNN3 / FTS2 probabilities cached). Zhou2020 sealed, BNCI2015_001 unused.
- SPLIT: numpy default_rng(2026).permutation of subjects 1..54; first 27 = TUNE, last 27 = HOLDOUT. The loop reads TUNE only. A TUNE
  subject's library is the other 26 TUNE subjects; a HOLDOUT subject's library is the 27 TUNE subjects. No holdout label enters the loop.
- LEAKAGE: `predict(own, test, library)` gets no test labels. Test trials may be used unlabelled (per-session alignment).
- METRIC: mean session-2 accuracy over TUNE. KEEP iff mean >= incumbent + 0.3 pp AND more subjects up than down; else discard.
  Simpler wins ties: a simplification within -0.1 pp may be kept.
- BUDGET: at most 30 experiments, then stop. No "never stop".
- BASELINE: R11 (run 0 must equal D2's R11 on every TUNE subject).
- FINAL (once, `prepare.py --final`): HOLDOUT, incumbent vs R11 and vs the KNOWN-BEFORE candidate `reference.py` (R11 plus a pooled
  fourth member, own weight 20, as found in N8). SUCCESS iff incumbent - R11 >= +1.0 pp with one-sided sign-flip p < 0.05; the loop
  ADDED something iff incumbent - reference >= +0.5 pp with p < 0.05. Whatever the outcome, no second holdout run.
- CAVEAT fixed in advance: HOLDOUT is clean for this loop only; N7-N12 already used all 54 subjects, so the starting ideas are not.

## Round 2+ (committed before its first experiment)
- Class balancing is removed from the candidate (not usable online). It is reported separately at the end and never kept by the loop.
- Same TUNE subjects, metric, keep rule and harness. The spent HOLDOUT is not read again. Start = round 1's candidate without balancing.
- PLATEAU: ideas run in rounds (loop2.py: round 2 new ideas, later rounds neighbours of the numeric settings and the round-2 ideas
  again on the new incumbent). Stop after the first round that keeps nothing, or at 90 experiments in total across all rounds.
