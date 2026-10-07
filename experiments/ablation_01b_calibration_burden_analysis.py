"""Analysis of the calibration-burden sweep (TODO A29).

Reads `results/ablation_calibration_burden.csv` (produced by
`experiments/ablation_01_calibration_burden.py`) and tests the claims that matter,
rather than reading them off the means. Pandas-only: no EEG, no recomputation, a few
seconds.

The four questions asked, in order of how much they change the story:

  1. Does alignment help at every budget? (gain at n, paired across subjects)
  2. Does its advantage *grow* as the labelled budget shrinks? This is the hypothesis that
     motivated A29 - that Riemannian alignment's zero-label property should matter more
     when labels are scarce. Answering it honestly, including refuting it, is the point.
  3. Data efficiency: how many labelled trials does aligned need to match the unadapted
     baseline's *full-budget* accuracy? (the product-relevant framing)
  4. Is the alignment *reference* a data bottleneck? `aligned` estimates it from the n
     labelled trials; `aligned_ref_full` estimates it from all 288 unlabeled calibration
     trials. If those two agree, unlabeled-data accumulation is not a deployment
     requirement.

Statistics: the 9 subjects are the independent units (the repeats at a fixed n estimate the
same quantity and are averaged). Wilcoxon signed-rank across subjects, with the n=9 floor of
p=0.0039 stated rather than glossed. Effect sizes in percentage points are reported next to
every p-value, because at 288 trials one trial is 0.35pp and significance without a
practical effect is easy to manufacture here.

Output: results/ablation_calibration_burden_summary.csv
    condition, n_trials, accuracy_mean, accuracy_std_across_subjects,
    gain_vs_none_same_n_pp, p_vs_none_same_n,
    diff_vs_none_full_pp, p_vs_none_full, diff_vs_aligned_full_pp, p_vs_aligned_full
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import IV2A_SUBJECTS, RESULTS_DIR

RAW = RESULTS_DIR / "ablation_calibration_burden.csv"
OUT = RESULTS_DIR / "ablation_calibration_burden_summary.csv"

# Committed means this sweep must reproduce at n=288, or the harness is not the
# established protocol and nothing else here is comparable.
EXPECTED_NONE_FULL = 0.6222993827160493
EXPECTED_ALIGNED_FULL = 0.6693672839506174
TOL = 1e-9

WILCOXON_FLOOR_N9 = 0.0039


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int, int]:
    """(mean difference in pp, wilcoxon p, wins, losses) for a vs b."""
    d = a - b
    if np.allclose(d, 0.0):
        return 0.0, 1.0, 0, 0
    return (
        100.0 * float(d.mean()),
        float(wilcoxon(a, b).pvalue),
        int((d > 0).sum()),
        int((d < 0).sum()),
    )


def main() -> int:
    if not RAW.exists():
        raise SystemExit(f"missing {RAW} - run experiments/ablation_01_calibration_burden.py first")

    df = pd.read_csv(RAW)
    assert set(df.subject.unique()) == set(IV2A_SUBJECTS), "unexpected subject set in the sweep"
    ns = sorted(int(n) for n in df.n_trials.unique())

    # Per-subject accuracy at each (condition, n), averaged over repeats: the repeats are
    # different random subsamples of the SAME session, so their mean estimates that
    # subject's expected accuracy at that budget.
    per = df.groupby(["subject", "condition", "n_trials"])["accuracy"].mean().unstack(["condition", "n_trials"])

    print("=== per-subject accuracy (mean over repeats) ===")
    print(per.round(4).to_string())

    # ---- harness identity ----------------------------------------------------
    print(f"\n=== harness identity (must match committed means at n=288, tol {TOL:.0e}) ===")
    ok = True
    for condition, expected, label in (
        ("none", EXPECTED_NONE_FULL, "Phase 1 cross-session (no adaptation)"),
        ("aligned", EXPECTED_ALIGNED_FULL, "Phase 2 (+ Riemannian alignment)"),
    ):
        got = float(per[(condition, 288)].mean())
        passed = abs(got - expected) <= TOL
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {condition} @288 = {got!r} vs {label} = {expected!r}")
    if not ok:
        print("!! harness does not match the established protocol - the rest is not comparable !!")

    none_full = per[("none", 288)].to_numpy()
    aligned_full = per[("aligned", 288)].to_numpy()

    rows = []
    print(f"\n=== Q1/Q2: alignment gain AT each budget (aligned - none, paired, n=9) ===")
    print(f"{'n':>5} {'none':>7} {'aligned':>8} {'gain_pp':>8} {'wins':>5} {'losses':>7} {'wilcoxon_p':>11}")
    q1 = {}
    for n in ns:
        a, b = per[("aligned", n)].to_numpy(), per[("none", n)].to_numpy()
        g, p, w, l = _paired(a, b)
        q1[n] = (g, p, w, l)
        print(f"{n:>5} {b.mean():7.4f} {a.mean():8.4f} {g:+8.2f} {w:5d} {l:7d} {p:11.4f}")
    gains = [q1[n][0] for n in ns]
    print(
        f"  -> gain at the smallest budget (n={ns[0]}) = {gains[0]:+.2f}pp vs at the full budget "
        f"(n={ns[-1]}) = {gains[-1]:+.2f}pp"
    )
    print(
        "  -> the A29 hypothesis was that this gain GROWS as labels shrink. "
        + ("It does." if gains[0] > gains[-1] + 1.0 else "It does not - see the write-up.")
    )

    print(f"\n=== Q3: data efficiency - aligned@n vs none@288 (the baseline's full-budget accuracy) ===")
    print(f"{'n':>5} {'aligned@n':>10} {'none@288':>9} {'diff_pp':>8} {'wins':>5} {'losses':>7} {'wilcoxon_p':>11}")
    q3 = {}
    for n in ns:
        a = per[("aligned", n)].to_numpy()
        d, p, w, l = _paired(a, none_full)
        q3[n] = (d, p, w, l)
        print(f"{n:>5} {a.mean():10.4f} {none_full.mean():9.4f} {d:+8.2f} {w:5d} {l:7d} {p:11.4f}")

    print(f"\n=== Q4: is the alignment reference a data bottleneck? aligned vs aligned_ref_full ===")
    print(f"{'n':>5} {'aligned':>8} {'ref_full':>9} {'diff_pp':>8} {'wins':>5} {'losses':>7} {'wilcoxon_p':>11}")
    q4 = {}
    for n in ns:
        a, b = per[("aligned", n)].to_numpy(), per[("aligned_ref_full", n)].to_numpy()
        d, p, w, l = _paired(a, b)
        q4[n] = (d, p, w, l)
        print(f"{n:>5} {a.mean():8.4f} {b.mean():9.4f} {d:+8.2f} {w:5d} {l:7d} {p:11.4f}")

    print(f"\n=== cost of shrinking the budget: aligned@n vs aligned@288 ===")
    print(f"{'n':>5} {'diff_pp':>8} {'wilcoxon_p':>11}")
    for n in ns:
        a = per[("aligned", n)].to_numpy()
        d, p, _, _ = _paired(a, aligned_full)
        print(f"{n:>5} {d:+8.2f} {p:11.4f}")

    # ---- summary artifact ---------------------------------------------------
    for condition in sorted(df.condition.unique()):
        for n in ns:
            acc = per[(condition, n)].to_numpy()
            if condition == "none":
                gain, p_same = 0.0, 1.0  # baseline versus itself
            else:
                gain, p_same, _, _ = _paired(acc, per[("none", n)].to_numpy())
            d_full, p_full, _, _ = _paired(acc, none_full)
            d_afull, p_afull, _, _ = _paired(acc, aligned_full)
            rows.append(
                {
                    "condition": condition,
                    "n_trials": n,
                    "accuracy_mean": float(acc.mean()),
                    "accuracy_std_across_subjects": float(acc.std(ddof=1)),
                    "gain_vs_none_same_n_pp": float(gain),
                    "p_vs_none_same_n": float(p_same),
                    "diff_vs_none_full_pp": float(d_full),
                    "p_vs_none_full": float(p_full),
                    "diff_vs_aligned_full_pp": float(d_afull),
                    "p_vs_aligned_full": float(p_afull),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["condition", "n_trials"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)
    print(f"\nSaved summary -> {OUT}")
    print(f"(wilcoxon floor at n=9 subjects is p={WILCOXON_FLOOR_N9})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
