"""D2, Lee2019_MI (OpenBMI) as the new, larger DEVELOPMENT cohort + power analysis. PRE-REGISTERED.

Committed BEFORE Lee2019 is downloaded (protocol: docs/lockbox_zhou2020_protocol.md and this docstring).
Why: the 18 IV-2a/2b sessions give an MDE of ~1.2-1.8 pp at 80 % power, so a "+0.5 pp" gate selects noise
(council, docs/diag_gates_d1_d4.md). Lee2019_MI has 54 subjects x 2 sessions (different days).

DATA (MOABB Lee2019_MI, default sessions (1, 2); left vs right hand MI; 1000 Hz; train phase only, 100 labelled
trials per session; the online "test" phase has no labels and is NOT used). Session 1 -> session 2 (as IV-2a:
train on the first day, test on the second), per-session Riemannian alignment.
Channels fixed a priori (the 20 sensorimotor channels of Lee et al. 2019, no selection on data):
FC5 FC3 FC1 FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP3 CP1 CPz CP2 CP4 CP6 (CPz and Cz in the 62-ch layout).
Resample 250 Hz; CNN input 4-38 Hz, 0-4 s; TS+LR 8-30 Hz, 0.5-2.5 s (src.pipeline, unchanged).

PIPELINE: src.pipeline.PipelineConfig(n_cnn=5, n_fts=2, fast=True) (F1: the stacked fast path equals the exact
one to 0.005 in probability; used for speed); base seeds 0-4, FTS seeds 0-1.

ARMS (accuracy on session 2): cnn1 (mean single seed 0-4), csp_lda, tslr, r10 ((CNN3+TSLR)/2), r11 ((CNN3+TSLR
+FTS2)/3), ctrl ((CNN3+TSLR+base seeds 3-4)/3).

ENDPOINTS (co-primary for every contrast A - B over the 54 subjects):
  E1 mean accuracy difference (one-sided exact sign-flip, Holm over the contrasts H1-H4);
  E2 share of subjects with accuracy >= 70 % (BCI-literacy threshold); difference in share, exact one-sided
     sign test on discordant subjects, Holm over H1-H4;
  E3 bottom-third mean: the 18 subjects with the lowest csp_lda accuracy (a reference independent of the CNN
     noise); difference of means on them, sign-flip over those 18.
CONTRASTS: H1 r11-cnn1, H2 r11-csp_lda, H3 r11-ctrl (FTSurrogate diversity), H4 r11-r10.
This run is a BASELINE for the new cohort (no new method is judged by it); future candidate screens use it as
the primary gate cohort (pooled-over-subjects gain >= +0.5 pp AND Holm p < 0.05 on its 54 subjects, then a
sealed-lockbox opening).

POWER (reported from the observed paired differences): SD of the per-subject paired difference, MDE at 80 %
power one-sided alpha 0.05 for n = 54 (this cohort), 20 (Zhou2020 lockbox), 18 (IV-2a/2b); Monte-Carlo power of
the sign-flip test for true effects 0.5-3 pp; the binomial floor sqrt(p(1-p)/100)*sqrt(2) of a paired
100-trial comparison (6.1 pp at p = 0.75) shows how much of the SD is trial noise.

    .venv\\Scripts\\python experiments\\d2_lee2019_dev.py --dry-run     # plumbing on FakeDataset, no real data
    .venv\\Scripts\\python experiments\\d2_lee2019_dev.py --prefetch    # download only (~66 GB, resumable)
    .venv\\Scripts\\python experiments\\d2_lee2019_dev.py               # THE run (cached per subject)
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
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
from scipy.stats import binom  # noqa: E402

from src.adaptation.convnet import encode_labels  # noqa: E402
from src.baselines import make_csp_lda  # noqa: E402
from src.config import DATA_DIR, FMAX, FMIN, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.evaluate import holm_adjust, sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.pipeline import CNN_BAND, CNN_TRAIN, PipelineConfig, SubjectData, align_by_group, fit_predict, tslr_member  # noqa: E402

mne.set_log_level("ERROR")
CLASSES = ["left_hand", "right_hand"]
CHANNELS = ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]
SFREQ = 250
N_BASE, N_FTS = 5, 2
ARMS = ["cnn1", "csp_lda", "tslr", "r10", "r11", "ctrl"]
CONTRASTS = {"H1": ("r11", "cnn1"), "H2": ("r11", "csp_lda"), "H3": ("r11", "ctrl"), "H4": ("r11", "r10")}
LITERACY = 0.70
N_POWER_SIM = 500
EFFECTS_PP = (0.5, 1.0, 1.5, 2.0, 3.0)
Z_ALPHA, Z_POWER = 1.6449, 0.8416
CACHE = DATA_DIR / "cache" / "d2_lee2019"


def load(dataset, subject: int, cnn_band: dict, ts_band: dict, channels: list[str] | None) -> SubjectData:
    """Session 1 (train) -> session 2 (test), both bands at 250 Hz; cached as a compact npz per subject."""
    f = CACHE / f"s{subject}.npz"
    if channels is not None and f.exists():
        z = np.load(f, allow_pickle=True)
        return SubjectData(X_cnn=z["X_cnn"], X_tslr=z["X_tslr"], y=z["y"], groups=z["groups"], train=z["train"])

    def get(b):
        kw = {"channels": channels} if channels else {}
        return MotorImagery(events=CLASSES, n_classes=2, resample=SFREQ, **kw, **b).get_data(
            dataset=dataset, subjects=[subject])

    Xw, yw, mw = get(cnn_band)
    Xn, yn, mn = get(ts_band)
    if not (np.array_equal(yw, yn) and np.array_equal(mw.session.to_numpy(), mn.session.to_numpy())):
        raise AssertionError(f"s{subject}: trial order differs between bands")
    sessions = sorted(mw.session.unique())
    if len(sessions) != 2:
        raise AssertionError(f"s{subject}: expected 2 sessions, got {sessions}")
    sess = mw.session.to_numpy()
    data = SubjectData(X_cnn=Xw.astype(np.float32), X_tslr=Xn.astype(np.float32), y=encode_labels(yw, CLASSES),
                       groups=sess, train=sess == sessions[0])
    counts = {s: np.bincount(data.y[sess == s], minlength=2).tolist() for s in sessions}
    print(f"  s{subject}: class counts {counts}, X_cnn {Xw.shape[1:]}, X_tslr {Xn.shape[1:]}", flush=True)
    if channels is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, X_cnn=data.X_cnn, X_tslr=data.X_tslr, y=data.y, groups=data.groups, train=data.train)
    return data


def acc(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(1) == y).mean())


def score_subject(data: SubjectData, cfg: PipelineConfig, cache: Path, device) -> dict:
    fit_predict(data, 2, cfg, device, cache, before_fit=wait_until_healthy if device.type == "cuda" else None)
    suffix = "_fast" if cfg.fast else ""
    base = [np.load(cache / f"cnn_seed{k}{suffix}.npy") for k in range(N_BASE)]
    fts = [np.load(cache / f"fts_seed{k}{suffix}.npy") for k in range(N_FTS)]
    ts = tslr_member(data)
    y = data.y[~data.train]
    A = align_by_group(data.X_tslr, data.groups)
    csp = make_csp_lda().fit(A[data.train], data.y[data.train]).predict(A[~data.train])
    cnn3 = np.mean(base[:3], 0)
    return {"cnn1": float(np.mean([acc(p, y) for p in base])), "csp_lda": float((csp == y).mean()),
            "tslr": acc(ts, y), "r10": acc((cnn3 + ts) / 2, y), "r11": acc((cnn3 + ts + np.mean(fts, 0)) / 3, y),
            "ctrl": acc((cnn3 + ts + np.mean(base[3:], 0)) / 3, y), "n_test": int(len(y))}


def sign_flip_p(d: np.ndarray, n_mc: int = 20_000, seed: int = 0, exact: bool = True) -> float:
    """One-sided sign-flip p: exact for n <= 20 (src.evaluate), Monte-Carlo (n_mc random sign vectors, seed 0,
    observed included) beyond, since 2**n sign vectors are not enumerable for the 54-subject cohort."""
    d = np.asarray(d, dtype=float)
    if exact and len(d) <= 20:
        return sign_flip_permutation_test(d, "greater")
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_mc, len(d)))
    return float((1 + ((signs * d).mean(1) >= d.mean() - 1e-12).sum()) / (1 + n_mc))


def sign_test_share(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact one-sided sign test on discordant subjects for 'a clears 70 % more often than b'."""
    up, down = int(((a >= LITERACY) & (b < LITERACY)).sum()), int(((a < LITERACY) & (b >= LITERACY)).sum())
    n = up + down
    return up, down, float(binom.sf(up - 1, n, 0.5)) if n else 1.0


def power_table(sd_pp: float, ns: tuple[int, ...], rng: np.random.Generator) -> pd.DataFrame:
    """MDE (80 % power, one-sided 0.05, normal approximation) and Monte-Carlo sign-flip power per effect."""
    rows = []
    for n in ns:
        row = {"n": n, "mde_pp": (Z_ALPHA + Z_POWER) * sd_pp / np.sqrt(n)}
        for eff in EFFECTS_PP:
            hits = 0
            for _ in range(N_POWER_SIM):
                hits += sign_flip_p(rng.normal(eff, sd_pp, n), n_mc=2000, exact=False) < 0.05
            row[f"power@{eff}pp"] = hits / N_POWER_SIM
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    pp = 100 * df.set_index("subject")[ARMS]
    n = len(pp)
    print(f"\n=== D2 baseline: Lee2019_MI, session 1 -> 2, {n} subjects ===")
    print(pp.mean().round(2).to_string())
    third = pp.csp_lda.sort_values().index[: n // 3]
    print(f"\nshare of subjects >= {LITERACY:.0%}: " +
          ", ".join(f"{a} {(pp[a] >= 100 * LITERACY).mean():.1%}" for a in ARMS))
    print(f"bottom third by csp_lda ({len(third)} subjects): " +
          ", ".join(f"{a} {pp.loc[third, a].mean():.2f}" for a in ARMS))
    res = []
    for h, (a, b) in CONTRASTS.items():
        d = (pp[a] - pp[b]).to_numpy()
        up, down, p_share = sign_test_share(pp[a].to_numpy(), pp[b].to_numpy())
        d3 = (pp.loc[third, a] - pp.loc[third, b]).to_numpy()
        res.append({"contrast": h, "a": a, "b": b, "mean_pp": d.mean(), "up": int((d > 0).sum()),
                    "down": int((d < 0).sum()), "p_mean": sign_flip_p(d),
                    "share_up": up, "share_down": down, "p_share": p_share, "bottom3_mean_pp": d3.mean(),
                    "p_bottom3": sign_flip_p(d3)})
    out = pd.DataFrame(res)
    for col in ("p_mean", "p_share"):
        out[f"holm_{col}"] = holm_adjust(out[col].tolist())
    print("\nCONTRASTS (one-sided; Holm over H1-H4):")
    print(out.round(4).to_string(index=False))
    sd = float((pp.r11 - pp.cnn1).std(ddof=1))
    floor = 100 * np.sqrt(2 * pp.r11.mean() / 100 * (1 - pp.r11.mean() / 100) / df.n_test.mean())
    print(f"\nPOWER: SD of the per-subject paired difference r11 - cnn1 = {sd:.2f} pp "
          f"(binomial floor of two independent 100-trial scores ~ {floor:.2f} pp)")
    sds = {h: float((pp[a] - pp[b]).std(ddof=1)) for h, (a, b) in CONTRASTS.items()}
    print("  paired SD by contrast: " + ", ".join(f"{h} {v:.2f}" for h, v in sds.items()))
    print(power_table(sd, (54, 20, 18), rng).round(3).to_string(index=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="exercise the code on MOABB's FakeDataset")
    ap.add_argument("--prefetch", action="store_true", help="download the raw files only")
    args = ap.parse_args()
    cnn_band, ts_band = dict(CNN_BAND), dict(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
    cfg = PipelineConfig(n_cnn=N_BASE, n_fts=N_FTS, fast=True)
    if args.dry_run:
        from moabb.datasets.fake import FakeDataset
        dataset = FakeDataset(event_list=CLASSES, n_sessions=2, n_runs=1, n_subjects=3, channels=tuple(CHANNELS),
                              sfreq=1000, duration=400, n_events=80, seed=0)
        cnn_band["tmax"], ts_band["tmax"] = 2.0, 2.0
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(CNN_TRAIN, epochs=2))
        subjects, cache_root, device = dataset.subject_list, DATA_DIR / "cache" / "d2_dryrun", torch.device("cpu")
        channels = None
    else:
        from moabb.datasets import Lee2019_MI
        dataset = Lee2019_MI()
        dataset._selected_sessions = None  # MOABB reads the default (1, 2) as 0-based positions and keeps only session 2
        subjects, cache_root, device, channels = dataset.subject_list, CACHE / "preds", torch.device("cuda"), CHANNELS
        if args.prefetch:
            for s in subjects:
                dataset.data_path(s)
                print(f"  downloaded s{s}", flush=True)
            return 0
    rows = []
    for s in subjects:
        data = load(dataset, s, cnn_band, ts_band, channels)
        rows.append({"subject": s, **score_subject(data, cfg, cache_root / f"s{s}", device)})
        del data
        gc.collect()
    df = pd.DataFrame(rows)
    if args.dry_run:
        print(df.round(3).to_string())
        summarise(df.assign(), np.random.default_rng(0)) if len(df) >= 3 else None
        print("DRY RUN OK (no real data touched)")
        return 0
    out = RESULTS_DIR / "d2_lee2019_dev.csv"
    df.to_csv(out, index=False)
    summarise(df, np.random.default_rng(0)).to_csv(RESULTS_DIR / "d2_lee2019_dev_contrasts.csv", index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
