"""L1, LOCKBOX confirmation of the R11 pipeline on an untouched dataset: BNCI2015_001. PRE-REGISTERED.

This file is committed BEFORE the dataset is downloaded. Nothing in it was tuned on BNCI2015_001. The
dataset is opened once; no reruns, no changes of protocol after the data are seen. An implementation bug
found after the run may be fixed in a separate commit that names the bug and leaves the protocol unchanged.

DATASET (MOABB BNCI2015_001, Faller et al. 2012): 12 subjects, 13 EEG channels (FC3 ... CP4, CAR), 512 Hz,
right hand vs feet, 2 sessions (A, B; subjects 8-11 also have C). Train on session A, test on session B
(first -> second session, as in IV-2a). Session C is not used.
Limitation fixed in advance: the dataset has no EOG channels, so R14's EOG control is impossible; our
4-38 Hz / 8-30 Hz bands kept the ocular share to ~1 pp on IV-2a/2b (R14, D6).

FROZEN PIPELINE (src/pipeline.py, unchanged; every setting as R3/R10/R11):
  resample to 250 Hz (the networks' kernels are in samples; the input length is taken from the data,
  ~1000 samples); CNN input 4-38 Hz, 0-4 s post-cue;
  TS+LR input 8-30 Hz, 0.5-2.5 s; Riemannian alignment per session; ShallowConvNet, 300 epochs, R3 CFG,
  integer targets; FTSurrogate p = 0.5, phase noise 0.9. Fits per subject: base seeds 0-4, FTS seeds 0-1.

ARMS (per subject; score = accuracy on session B):
  cnn1     mean single-model accuracy of base seeds 0-4 (the R3 single-CNN analogue)
  csp_lda  CSP (8 comp., Ledoit-Wolf) + LDA on aligned 8-30 Hz, 0.5-2.5 s (Phase 2 baseline)
  tslr     TS+LR alone
  r10      (CNN3 + TSLR) / 2, CNN3 = mean of base seeds 0-2
  r11      (CNN3 + TSLR + FTS2) / 3, FTS2 = mean of FTS seeds 0-1        <- the pipeline under test
  ctrl     (CNN3 + TSLR + mean of base seeds 3-4) / 3                   (size-matched, no FTSurrogate)
  lite     (FTS seed 0 + TSLR) / 2                                        (1 GPU fit)

PRIMARY HYPOTHESES (one-sided exact sign-flip over 12 subjects, Holm over H1-H3):
  H1 r11 - cnn1    > 0   the pipeline beats a single CNN.            R11 REPLICATES iff H1 Holm p < 0.05.
  H2 r11 - ctrl    > 0   the FTSurrogate diversity gain replicates.
  H3 r11 - csp_lda > 0   the pipeline beats the classical baseline.
  Expected (council shrinkage, reports/r14_council_and_ideation.md): H1 ~ +3.5 pp, H2 ~ +1 to +2 pp.
SECONDARY (reported, no gate):
  r10 - cnn1, r11 - r10; LITE non-inferiority: LITE replaces R11 only if the one-sided 95% bootstrap lower
  bound (10 000 resamples over subjects, seed 0) of lite - r11 is above -1.0 pp.
Reported for every contrast: mean, median, up/down count, 95% bootstrap CI.

CHECKS: trial order equal between the two bands; per session the class counts are printed; RA self-check
(in align_by_group); alignment groups never mix sessions; --dry-run exercises the whole path on MOABB's
FakeDataset (2 epochs, short windows) without touching BNCI2015_001.

    .venv\\Scripts\\python experiments\\l1_lockbox_bnci2015_001.py --dry-run   # plumbing only, no real data
    .venv\\Scripts\\python experiments\\l1_lockbox_bnci2015_001.py             # THE run (~40 min GPU)
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from moabb.paradigms import MotorImagery  # noqa: E402

from src.adaptation.convnet import encode_labels  # noqa: E402
from src.baselines import make_csp_lda  # noqa: E402
from src.config import DATA_DIR, FMAX, FMIN, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.evaluate import holm_adjust, sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.pipeline import CNN_BAND, CNN_TRAIN, PipelineConfig, SubjectData, align_by_group, fit_predict, tslr_member  # noqa: E402

mne.set_log_level("ERROR")
CLASSES = ["right_hand", "feet"]
SFREQ = 250
N_BASE, N_FTS = 5, 2
HYPOTHESES = {"H1": ("r11", "cnn1"), "H2": ("r11", "ctrl"), "H3": ("r11", "csp_lda")}
SECONDARY = {"r10-cnn1": ("r10", "cnn1"), "r11-r10": ("r11", "r10"), "lite-r11": ("lite", "r11")}
NONINF_MARGIN_PP = -1.0
N_BOOT = 10_000


def load(dataset, subject: int, cnn_band: dict, ts_band: dict) -> SubjectData:
    """Session A (first) -> session B (second), both bands, resampled to 250 Hz; other sessions dropped."""
    def get(b):
        p = MotorImagery(events=CLASSES, n_classes=2, resample=SFREQ, **b)
        return p.get_data(dataset=dataset, subjects=[subject])

    Xw, yw, mw = get(cnn_band)
    Xn, yn, mn = get(ts_band)
    if not (np.array_equal(yw, yn) and np.array_equal(mw.session.to_numpy(), mn.session.to_numpy())):
        raise AssertionError(f"s{subject}: trial order differs between bands")
    sessions = sorted(mw.session.unique())
    keep = mw.session.isin(sessions[:2]).to_numpy()
    sess = mw.session.to_numpy()[keep]
    data = SubjectData(X_cnn=Xw[keep], X_tslr=Xn[keep], y=encode_labels(yw[keep], CLASSES),
                       groups=sess, train=sess == sessions[0])
    counts = {s: np.bincount(data.y[sess == s], minlength=2).tolist() for s in sessions[:2]}
    print(f"  s{subject}: sessions {sessions[:2]} (of {sessions}), class counts {counts}, "
          f"X_cnn {Xw.shape[1:]}, X_tslr {Xn.shape[1:]}", flush=True)
    return data


def acc(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(1) == y).mean())


def score_subject(data: SubjectData, cfg: PipelineConfig, cache: Path, device) -> dict:
    fit_predict(data, 2, cfg, device, cache, before_fit=wait_until_healthy if device.type == "cuda" else None)
    base = [np.load(cache / f"cnn_seed{k}.npy") for k in range(N_BASE)]
    fts = [np.load(cache / f"fts_seed{k}.npy") for k in range(N_FTS)]
    ts = tslr_member(data)
    y = data.y[~data.train]
    A = align_by_group(data.X_tslr, data.groups)
    csp = make_csp_lda().fit(A[data.train], data.y[data.train]).predict(A[~data.train])
    cnn3 = np.mean(base[:3], 0)
    return {"cnn1": float(np.mean([acc(p, y) for p in base])), "csp_lda": float((csp == y).mean()),
            "tslr": acc(ts, y), "r10": acc((cnn3 + ts) / 2, y), "r11": acc((cnn3 + ts + np.mean(fts, 0)) / 3, y),
            "ctrl": acc((cnn3 + ts + np.mean(base[3:], 0)) / 3, y), "lite": acc((fts[0] + ts) / 2, y),
            "n_test": int(len(y))}


def boot_ci(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    means = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(np.percentile(means, 5))


def summarise(df: pd.DataFrame) -> None:
    pp = 100 * df.set_index("subject")[["cnn1", "csp_lda", "tslr", "r10", "r11", "ctrl", "lite"]]
    print("\n=== L1 lockbox: BNCI2015_001, session A -> B, 12 subjects ===")
    print(pp.mean().round(2).to_string())
    rng = np.random.default_rng(0)
    ps = {h: sign_flip_permutation_test((pp[a] - pp[b]).to_numpy(), "greater") for h, (a, b) in HYPOTHESES.items()}
    holm = dict(zip(ps, holm_adjust(list(ps.values()))))
    print("\nPRIMARY (one-sided sign-flip, Holm over H1-H3):")
    for h, (a, b) in HYPOTHESES.items():
        d = (pp[a] - pp[b]).to_numpy()
        lo, hi, _ = boot_ci(d, rng)
        ok = holm[h] < 0.05 and d.mean() > 0
        print(f"  {h} {a} - {b}: mean {d.mean():+.2f} pp [95% CI {lo:+.2f}, {hi:+.2f}], median {np.median(d):+.2f}, "
              f"{(d > 0).sum()}/12 up, {(d < 0).sum()} down, p = {ps[h]:.4f} (Holm {holm[h]:.4f}) "
              f"-> {'SUPPORTED' if ok else 'not supported'}")
    print(f"\nR11 {'REPLICATES' if holm['H1'] < 0.05 and (pp.r11 - pp.cnn1).mean() > 0 else 'DOES NOT REPLICATE'} "
          "on the lockbox (H1).")
    print("\nSECONDARY:")
    for name, (a, b) in SECONDARY.items():
        d = (pp[a] - pp[b]).to_numpy()
        lo, hi, lo90 = boot_ci(d, rng)
        extra = (f"; non-inferiority lower bound {lo90:+.2f} vs margin {NONINF_MARGIN_PP:+.1f} -> "
                 f"{'NON-INFERIOR' if lo90 > NONINF_MARGIN_PP else 'not shown'}") if name == "lite-r11" else ""
        print(f"  {name}: mean {d.mean():+.2f} pp [95% CI {lo:+.2f}, {hi:+.2f}], {(d > 0).sum()}/12 up, "
              f"{(d < 0).sum()} down, p = {sign_flip_permutation_test(d, 'greater'):.4f}{extra}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="exercise the code on MOABB's FakeDataset")
    args = ap.parse_args()
    cnn_band, ts_band = dict(CNN_BAND), dict(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
    cfg = PipelineConfig(n_cnn=N_BASE, n_fts=N_FTS)
    if args.dry_run:
        from moabb.datasets.fake import FakeDataset
        chans = ("FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CPz", "CP4")
        dataset = FakeDataset(event_list=CLASSES, n_sessions=2, n_runs=1, n_subjects=2, channels=chans,
                              sfreq=512, duration=400, n_events=80, seed=0)
        cnn_band["tmax"], ts_band["tmax"] = 2.0, 2.0
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(CNN_TRAIN, epochs=2))
        subjects, out_csv, cache_root = dataset.subject_list, None, DATA_DIR / "cache" / "l1_dryrun"
        device = torch.device("cpu")
    else:
        from moabb.datasets import BNCI2015_001
        dataset = BNCI2015_001()
        subjects, out_csv = dataset.subject_list, RESULTS_DIR / "l1_lockbox_bnci2015_001.csv"
        cache_root, device = DATA_DIR / "cache" / "l1_preds", torch.device("cuda")
    rows = []
    for s in subjects:
        data = load(dataset, s, cnn_band, ts_band)
        rows.append({"subject": s, **score_subject(data, cfg, cache_root / f"s{s}", device)})
    df = pd.DataFrame(rows)
    if args.dry_run:
        print(df.round(3).to_string())
        print("DRY RUN OK (no real data touched)")
        return 0
    df.to_csv(out_csv, index=False)
    summarise(df)
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
