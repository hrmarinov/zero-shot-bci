"""R11b, POST-HOC descriptive check (not pre-registered, no gate): is the R11 low-frequency member eye movement?

R11's 0.5-4 Hz time-course member decodes far above chance on its own (IV-2a 46.6%, IV-2b 67.4%)
and adds +2.06 pp as a third ensemble member. Slow (< 4 Hz) activity phase-locked to a lateralised
visual cue is the classic signature of eye movements (towards the cue arrow, or following the IV-2b
feedback smiley), which leak into EEG channels by volume conduction.

Check: rebuild the R11 low-frequency member exactly (same band, window, 100 ms bins, per-session
z-scoring, shrinkage LDA) on (a) the EEG channels, which must reproduce R11's m_lowfreq, and (b) the
three EOG channels ONLY. If EOG alone decodes the class and adds a similar gain as a third member, the
low-frequency gain is ocular, not neural motor imagery.

Epochs are cut from the raw MOABB recordings (MOABB's own MotorImagery paradigm drops EOG) with
MOABB's filter (IIR band-pass on the continuous run) and window (dataset.interval[0] + 0.5 .. + 2.5 s).
Trial order is asserted equal to the harness loader's labels.

    .venv\\Scripts\\python experiments\\r11b_lowfreq_eog_check.py     # ~10 min, CPU
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from moabb.datasets import BNCI2014_001, BNCI2014_004  # noqa: E402

from experiments.r3_convnet_session_norm import align_per_session  # noqa: E402
from experiments.r7_self_training_iv2b import CLASSES as IV2B_CLASSES  # noqa: E402
from experiments.r7_self_training_iv2b import EVAL_RECORDINGS, TRAIN_RECORDINGS  # noqa: E402
from experiments.r10_round2_stack_position import align_by_recording, tangent, ts_lr  # noqa: E402
from experiments.r11_round3_diverse_members import CACHE, LF_BAND, lowfreq_member, rec_acc  # noqa: E402
from src.adaptation.convnet import encode_labels  # noqa: E402
from src.config import FMAX, FMIN, IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.datasets import load_iv2a_subject_window, load_iv2b_subject_window  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402

mne.set_log_level("ERROR")
OUT_CSV = RESULTS_DIR / "r11b_lowfreq_eog_check.csv"


def epochs_by_type(dataset, subject: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """{'eeg': X, 'eog': X}, labels (str), session tag, in MOABB's session/run order."""
    inv = {v: k for k, v in dataset.event_id.items()}
    out, ys, sess = {"eeg": [], "eog": []}, [], []
    for sname, runs in dataset.get_data(subjects=[subject])[subject].items():
        for raw in runs.values():
            ev = mne.find_events(raw, shortest_event=0, verbose=False)
            ev = ev[np.isin(ev[:, 2], list(inv))]
            filt = raw.copy().filter(*LF_BAND, method="iir", picks=["eeg", "eog"])
            t0 = dataset.interval[0]
            for kind in out:
                ep = mne.Epochs(filt, ev, tmin=t0 + TMIN, tmax=t0 + TMAX, picks=kind, baseline=None, preload=True)
                out[kind].append(ep.get_data(copy=False))
            ys += [inv[c] for c in ev[:, 2]]
            sess += [sname] * len(ev)
    return {k: np.concatenate(v) for k, v in out.items()}, np.array(ys), np.array(sess)


def stacks(rows, ds, s, y, rec, cnn3, ts, members: dict) -> None:
    rows.append({"dataset": ds, "subject": s, "arm": "r10", "acc": rec_acc(0.5 * cnn3 + 0.5 * ts, y, rec)})
    for name, p in members.items():
        rows.append({"dataset": ds, "subject": s, "arm": f"m_{name}", "acc": rec_acc(p, y, rec)})
        rows.append({"dataset": ds, "subject": s, "arm": f"stack_{name}", "acc": rec_acc((cnn3 + ts + p) / 3, y, rec)})


def run_iv2a(s: int, rows: list) -> None:
    X, ys, sess = epochs_by_type(BNCI2014_001(), s)
    d = load_iv2a_subject_window(s, FMIN, FMAX, TMIN, TMAX)
    t, e = sess == "0train", sess == "1test"
    if not (np.array_equal(ys[t], d.y_calib) and np.array_equal(ys[e], d.y_eval)):
        raise AssertionError(f"IV-2a s{s}: raw-epoch label order differs from the harness loader")
    yt, ye = encode_labels(d.y_calib, IV2A_CLASSES), encode_labels(d.y_eval, IV2A_CLASSES)
    Xt, Xe = align_per_session(d.X_calib, d.X_eval)
    ts = ts_lr(*_tan_pair(Xt, yt, Xe))
    cnn3 = np.mean([np.load(CACHE / "r3_preds" / f"s{s}_cnn_ra_seed{k}.npy") for k in range(3)], 0)
    zt, ze = np.zeros(len(yt)), np.zeros(len(ye))
    members = {k: lowfreq_member(X[k][t], yt, zt, X[k][e], ze) for k in X}
    stacks(rows, "IV-2a", s, ye, ze, cnn3, ts, members)


def _tan_pair(Xt, yt, Xe):
    Ft, Fe = tangent(Xt, Xe)
    return Ft, yt, Fe


def run_iv2b(s: int, rows: list) -> None:
    X, ys, sess = epochs_by_type(BNCI2014_004(), s)
    d = load_iv2b_subject_window(s, FMIN, FMAX, TMIN, TMAX)
    if not (np.array_equal(ys, d.y) and np.array_equal(sess, d.session)):
        raise AssertionError(f"IV-2b s{s}: raw-epoch label/recording order differs from the harness loader")
    y = encode_labels(d.y, IV2B_CLASSES)
    tr = np.isin(d.session, TRAIN_RECORDINGS)
    ev = np.concatenate([np.flatnonzero(d.session == r) for r in EVAL_RECORDINGS])
    A = align_by_recording(d.X, d.session)
    ts = ts_lr(*_tan_pair(A[tr], y[tr], A[ev]))
    cnn3 = np.mean([np.load(CACHE / "r7_preds" / f"s{s}_teacher_seed{k}.npy") for k in range(3)], 0)
    members = {k: lowfreq_member(X[k][tr], y[tr], d.session[tr], X[k][ev], d.session[ev]) for k in X}
    stacks(rows, "IV-2b", s, y[ev], d.session[ev], cnn3, ts, members)


def main() -> int:
    rows: list[dict] = []
    for s in IV2A_SUBJECTS:
        run_iv2a(s, rows)
        run_iv2b(s, rows)
        print(f"  s{s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    piv = 100 * df.pivot_table(index=["dataset", "subject"], columns="arm", values="acc")
    r11 = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv")
    ref = 100 * r11[r11.arm == "m_lowfreq"].set_index(["dataset", "subject"]).acc
    print("\n=== R11b post-hoc: low-frequency member, EEG vs EOG-only ===")
    print(piv.groupby("dataset").mean().round(2).T.to_string())
    print(f"reproduction of R11 m_lowfreq (EEG): max |diff| = {(piv['m_eeg'] - ref).abs().max():.3f} pp")
    for k in ("eeg", "eog"):
        dd = piv[f"stack_{k}"] - piv["r10"]
        p = sign_flip_permutation_test(dd.to_numpy(), "greater")
        print(f"  stack_{k} - r10: pooled {dd.mean():+.2f} pp, {(dd > 0).sum()}/18 up, p = {p:.4f}; "
              + ", ".join(f"{a} {v:+.2f}" for a, v in dd.groupby('dataset').mean().items()))
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
