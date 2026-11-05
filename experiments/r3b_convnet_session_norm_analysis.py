"""R3 analysis — implements docs/prereg_convnet_session_norm.md's statistics exactly.

Reads results/r3_convnet_session_norm.csv (+ per-run probabilities in data/cache/r3_preds/
for the seed ensemble and McNemar), writes results/r3_convnet_session_norm_summary.csv and
prints the pre-registered decision. Exits non-zero if the harness identity check fails
(csp_ra must reproduce Phase 2's 0.6693672839506174) or the sweep is incomplete.

    .venv\\Scripts\\python experiments\\r3b_convnet_session_norm_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(4)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.adaptation.convnet import encode_labels  # noqa: E402
from src.config import IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR  # noqa: E402
from src.evaluate import holm_adjust, mcnemar_test, sign_flip_permutation_test  # noqa: E402

CSV = RESULTS_DIR / "r3_convnet_session_norm.csv"
SUMMARY = RESULTS_DIR / "r3_convnet_session_norm_summary.csv"
PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "r3_preds"
PHASE2_MEAN = 0.6693672839506174
SEEDS = (0, 1, 2, 3, 4)
CNN_ARMS = ("cnn_none", "cnn_adabn", "cnn_ra", "cnn_ra_adabn", "cnn_ra_sat")
CSP_ARMS = ("csp_ra", "csp_ra_wide", "csp_none_wide")
WEAK = (2, 5, 6)
SEED_SD_FLAG = 0.03


def per_subject_table(df: pd.DataFrame) -> pd.DataFrame:
    """subject x arm accuracy: seed mean for convnet arms, the single run for CSP arms."""
    cnn = df[df.arm.isin(CNN_ARMS)]
    counts = cnn.groupby(["subject", "arm"]).seed.nunique()
    if (counts != len(SEEDS)).any() or set(cnn.subject) != set(IV2A_SUBJECTS):
        raise SystemExit(f"incomplete sweep:\n{counts[counts != len(SEEDS)]}")
    table = df[df.arm.isin(CNN_ARMS + CSP_ARMS)].pivot_table(index="subject", columns="arm", values="acc", aggfunc="mean")
    return table[list(CSP_ARMS + CNN_ARMS)]


def paired(a: pd.Series, b: pd.Series, alternative: str) -> dict:
    d = (a - b).to_numpy()
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return {
        "mean_pp": 100 * d.mean(), "ci_lo_pp": 100 * lo, "ci_hi_pp": 100 * hi,
        "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "min_pp": 100 * d.min(),
        "p_perm": sign_flip_permutation_test(d, alternative),
        "p_wilcoxon": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
    }


def fmt(name: str, r: dict) -> str:
    return (f"  {name:42s} {r['mean_pp']:+6.2f}pp  95%CI [{r['ci_lo_pp']:+.2f}, {r['ci_hi_pp']:+.2f}]  "
            f"{r['wins']}W/{r['losses']}L  worst {r['min_pp']:+.2f}  p_perm={r['p_perm']:.4f}  p_wilcoxon={r['p_wilcoxon']:.4f}")


def load_probs(subject: int, arm: str, seed: int) -> np.ndarray:
    return np.load(PRED_DIR / f"s{subject}_{arm}_seed{seed}.npy")


def ensemble_and_mcnemar(table: pd.DataFrame) -> pd.DataFrame:
    """5-seed probability-averaged ensemble (secondary) + per-subject McNemar vs csp_ra."""
    from src.datasets import load_iv2a_subject_window

    rows = []
    for s in IV2A_SUBJECTS:
        y = encode_labels(load_iv2a_subject_window(s, 4.0, 38.0, 0.0, 4.0).y_eval, IV2A_CLASSES)
        csp = load_probs(s, "csp_ra", -1)
        if not np.isclose((csp == y).mean(), table.loc[s, "csp_ra"]):
            raise SystemExit(f"stored csp_ra predictions disagree with CSV for subject {s}")
        row = {"subject": s}
        ensembles = {}
        for arm in ("cnn_ra_adabn", "cnn_ra_sat", "cnn_ra"):
            probs = [load_probs(s, arm, k) for k in SEEDS]
            seed_accs = [(p.argmax(1) == y).mean() for p in probs]
            if not np.isclose(np.mean(seed_accs), table.loc[s, arm]):
                raise SystemExit(f"stored {arm} probabilities disagree with CSV for subject {s}")
            ensembles[arm] = np.mean(probs, axis=0).argmax(1)
            row[f"{arm}_ensemble"] = (ensembles[arm] == y).mean()
            row[f"{arm}_seed_sd"] = float(np.std(seed_accs, ddof=1))
        row["mcnemar_p_seed0_vs_csp_ra"] = mcnemar_test(y, load_probs(s, "cnn_ra_adabn", 0).argmax(1), csp)
        row["mcnemar_p_ens_vs_csp_ra"] = mcnemar_test(y, ensembles["cnn_ra_adabn"], csp)
        rows.append(row)
    return pd.DataFrame(rows).set_index("subject")


def decide(primary: dict, ctrl_mean: float) -> str:
    if primary["ci_hi_pp"] < 1.0 or primary["mean_pp"] <= 0:
        return "REJECT"
    adopt = (primary["mean_pp"] >= 3.0 and primary["p_perm"] < 0.05 and primary["wins"] >= 7
             and primary["min_pp"] >= -5.0 and abs(ctrl_mean - 0.25) <= 0.05)
    return "ADOPT (provisional until external replication)" if adopt else "INCONCLUSIVE"


def main() -> int:
    df = pd.read_csv(CSV)
    table = per_subject_table(df)
    if not np.isclose(table["csp_ra"].mean(), PHASE2_MEAN, rtol=0, atol=1e-12):
        print(f"HARNESS IDENTITY FAILED: csp_ra mean {table['csp_ra'].mean()!r} != {PHASE2_MEAN!r}")
        return 1
    print(f"[PASS] harness identity: csp_ra mean = {table['csp_ra'].mean()!r} (Phase 2, bit-exact)")
    ctrl = df[df.arm == "ctrl_permuted"].acc
    print(f"negative control (permuted T labels): mean {ctrl.mean():.4f} over {len(ctrl)} subjects "
          f"(range {ctrl.min():.3f}-{ctrl.max():.3f}); must be within 0.25 +/- 0.05")

    print("\n=== per-subject accuracy (convnet arms: mean of 5 seeds) ===")
    print((table * 100).round(1).to_string())
    print("MEAN  " + "  ".join(f"{a}={100*table[a].mean():.2f}" for a in table.columns))

    extra = ensemble_and_mcnemar(table)
    sd = df[df.arm.isin(CNN_ARMS)].groupby(["subject", "arm"]).acc.std(ddof=1).unstack()
    flagged = sd[sd > SEED_SD_FLAG].stack()
    print(f"\nseed SD (pp): median {100*np.nanmedian(sd.values):.2f}, max {100*np.nanmax(sd.values):.2f}; "
          f"flagged > {100*SEED_SD_FLAG:.0f}pp: {[(int(s), a, round(100*v, 1)) for (s, a), v in flagged.items()]}")

    print("\n=== PRIMARY (one-sided sign-flip permutation, 512 flips) ===")
    primary = paired(table["cnn_ra_adabn"], table["csp_ra"], "greater")
    print(fmt("cnn_ra_adabn - csp_ra", primary))

    print("\n=== SECONDARY family (two-sided, Holm) ===")
    sec = {
        "S1 cnn_ra_adabn - cnn_ra (AdaBN over RA)": paired(table["cnn_ra_adabn"], table["cnn_ra"], "two-sided"),
        "S2 cnn_ra_sat - cnn_ra_adabn (run-wise SAT)": paired(table["cnn_ra_sat"], table["cnn_ra_adabn"], "two-sided"),
        "S3 cnn_ra_adabn - csp_ra_wide (model, same window)": paired(table["cnn_ra_adabn"], table["csp_ra_wide"], "two-sided"),
    }
    for (name, r), p_holm in zip(sec.items(), holm_adjust([r["p_perm"] for r in sec.values()])):
        r["p_holm"] = p_holm
        print(fmt(name, r) + f"  p_holm={p_holm:.4f}")

    print("\n=== DESCRIPTIVE (no inference) ===")
    for name, a, b in [("cnn_adabn - cnn_none (AdaBN, no RA)", "cnn_adabn", "cnn_none"),
                       ("cnn_ra - cnn_none (RA input)", "cnn_ra", "cnn_none"),
                       ("csp_ra_wide - csp_ra (window/band only)", "csp_ra_wide", "csp_ra"),
                       ("csp_ra_wide - csp_none_wide (RA, wide)", "csp_ra_wide", "csp_none_wide")]:
        print(fmt(name, paired(table[a], table[b], "two-sided")))
    print(f"  ensemble (5-seed prob. average): cnn_ra_adabn {100*extra['cnn_ra_adabn_ensemble'].mean():.2f}%  "
          f"cnn_ra_sat {100*extra['cnn_ra_sat_ensemble'].mean():.2f}%  cnn_ra {100*extra['cnn_ra_ensemble'].mean():.2f}%")
    weak = table.loc[list(WEAK)]
    print(f"  WEAK {{2,5,6}}: csp_ra {100*weak['csp_ra'].mean():.1f}%  cnn_ra_adabn {100*weak['cnn_ra_adabn'].mean():.1f}%  "
          f"cnn_ra_sat {100*weak['cnn_ra_sat'].mean():.1f}%")
    print("  per-subject McNemar p (cnn_ra_adabn vs csp_ra): seed0 "
          + str(extra["mcnemar_p_seed0_vs_csp_ra"].round(4).tolist()) + "; ensemble "
          + str(extra["mcnemar_p_ens_vs_csp_ra"].round(4).tolist()))

    verdict = decide(primary, float(ctrl.mean()))
    tripwire = table["cnn_ra_adabn"].mean() > 0.80
    print(f"\n=== PRE-REGISTERED DECISION: {verdict} ===" + ("  [TRIPWIRE >80%: leakage audit required]" if tripwire else ""))

    out = table.join(extra).reset_index()
    out.to_csv(SUMMARY, index=False)
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
