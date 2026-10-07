"""Phase 8a - does restricting to motor-cortex electrodes help?

User's idea, part 1: IV-2a's 22-channel montage is not a full-scalp array -
it already leans heavily central (only Fz, P1, Pz, P2, POz sit outside the
frontocentral/central/centroparietal strip). This tests whether trimming to
that strip - or to just the classic C3/Cz/C4 sensorimotor row - improves
motor-imagery classification, on top of both the non-adapted CSP+LDA baseline
(Phase 1) and the best-performing method so far (Phase 2's Riemannian
alignment), or whether CSP's own data-driven spatial filters already capture
whatever benefit channel restriction would give.

Three nested channel sets (full 22 -> central 17 -> narrow 9), each run
through both pipelines, compared back to the full-22 condition via paired
McNemar tests on the same eval trials.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.adaptation.riemannian_align import RiemannianAlignment
from src.baselines import make_csp_lda
from src.config import IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import IV2A_CHANNELS, load_iv2a_subject, restrict_channels
from src.evaluate import accuracy_summary, fit_predict, mcnemar_test, wilson_interval

# Drops Fz (frontal) and P1/Pz/P2/POz (parietal/occipital) - keeps every
# frontocentral/central/centroparietal channel.
CENTRAL_17 = [c for c in IV2A_CHANNELS if c not in {"Fz", "P1", "Pz", "P2", "POz"}]

# The classic sensorimotor row plus its immediate FC/CP neighbors - as tight
# a motor-cortex focus as this montage supports.
NARROW_9 = ["FCz", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CPz"]

CHANNEL_SETS = {"full_22": IV2A_CHANNELS, "central_17": CENTRAL_17, "narrow_9": NARROW_9}


def run() -> pd.DataFrame:
    rows = []
    for subject in IV2A_SUBJECTS:
        print(f"[subject {subject}] loading...")
        data = load_iv2a_subject(subject)

        baseline_pred = {}
        for set_name, channels in CHANNEL_SETS.items():
            n_comp = min(8, len(channels))
            X_calib = restrict_channels(data.X_calib, IV2A_CHANNELS, channels)
            X_eval = restrict_channels(data.X_eval, IV2A_CHANNELS, channels)

            pipeline = make_csp_lda(n_components=n_comp)
            y_pred_none = fit_predict(pipeline, X_calib, data.y_calib, X_eval)
            acc_none, correct_none, n_total = accuracy_summary(data.y_eval, y_pred_none)

            X_calib_al = RiemannianAlignment().fit_transform(X_calib)
            X_eval_al = RiemannianAlignment().fit_transform(X_eval)
            y_pred_align = fit_predict(pipeline, X_calib_al, data.y_calib, X_eval_al)
            acc_align, correct_align, _ = accuracy_summary(data.y_eval, y_pred_align)

            if set_name == "full_22":
                baseline_pred["none"] = y_pred_none
                baseline_pred["align"] = y_pred_align

            ci_low_none, ci_high_none = wilson_interval(correct_none, n_total)
            ci_low_align, ci_high_align = wilson_interval(correct_align, n_total)
            p_none = mcnemar_test(data.y_eval, baseline_pred["none"], y_pred_none)
            p_align = mcnemar_test(data.y_eval, baseline_pred["align"], y_pred_align)

            print(
                f"[subject {subject}] {set_name} ({len(channels)}ch): "
                f"none={acc_none:.3f} (p={p_none:.3f})  align={acc_align:.3f} (p={p_align:.3f})"
            )
            rows.append(
                {
                    "subject": subject, "channel_set": set_name, "n_channels": len(channels),
                    "acc_none": acc_none, "acc_none_ci_low": ci_low_none, "acc_none_ci_high": ci_high_none,
                    "acc_none_vs_full22_p": p_none,
                    "acc_align": acc_align, "acc_align_ci_low": ci_low_align, "acc_align_ci_high": ci_high_align,
                    "acc_align_vs_full22_p": p_align,
                }
            )

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase8_motor_cortex_channels.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    summary = df.groupby("channel_set")[["acc_none", "acc_align"]].mean().reindex(CHANNEL_SETS.keys())
    print(summary.to_string())
    return df


if __name__ == "__main__":
    run()
