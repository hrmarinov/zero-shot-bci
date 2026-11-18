"""E1, spatial-information screen on Lee2019_MI: surface Laplacian (CSD) and all 62 channels. PRE-REGISTERED.

Why (docs/diag_gates_d1_d4.md, D3): weak subjects are mostly signal-limited; R11 uses 20 of 62 channels and no
spatial sharpening beyond Riemannian alignment. Two ways to add spatial information, from array processing and
geophysics: (1) a surface Laplacian / current-source-density transform (spherical spline, MNE, standard_1005
montage already attached by MOABB) that undoes volume conduction; (2) the extra 42 channels (non-motor channels
carry noise structure, and the covariance-based members see the whole field).

ARMS (everything else IDENTICAL to D2: session 1 -> 2, 100 trials each, per-session RA, R11 with
PipelineConfig(n_cnn=5, n_fts=2, fast=True), bands 4-38 Hz / 0-4 s and 8-30 Hz / 0.5-2.5 s, resample 250 Hz):
  l20   CSD on the 62-channel raw (before any filtering), then the 20 sensorimotor channels of D2
  a62   all 62 EEG channels, no CSD
  l62   CSD, all 62 channels
BASELINE: the D2 R11 (no transform, 20 channels) from results/d2_lee2019_dev.csv, same subjects and seeds.
CSD output is relabelled as EEG; scale does not matter (RA whitens every trial).

CONTRASTS (paired over the 54 subjects, one-sided sign-flip Monte-Carlo (src of D2), Holm over H1-H3):
  H1 r11[l20] - r11[D2]     H2 r11[a62] - r11[D2]     H3 r11[l62] - r11[D2]
CO-PRIMARY endpoints per contrast: mean difference, share of subjects >= 70 % (exact sign test on discordant
subjects), bottom-third mean (18 subjects with the lowest D2 csp_lda accuracy).
SCREEN GATE (as the Lee2019 convention): mean gain >= +0.5 pp AND Holm p < 0.05. Passing is NOT a confirmation:
a claim goes to the sealed Zhou2020 lockbox only if its mean gain is >= 3 pp (the lockbox MDE at n = 20, D2) and
a channel-agnostic CSD must also be checked on IV-2a/2b (22/3 channels: may not apply, then not claimed there).
SECONDARY (reported): each arm's own r11 - cnn1, csp_lda, tslr (does the arm change the member balance?).

    .venv\\Scripts\\python experiments\\e1_spatial_prior_screen.py --dry-run   # plumbing on FakeDataset
    .venv\\Scripts\\python experiments\\e1_spatial_prior_screen.py             # THE run (~5-6 h, resumable)
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
from moabb.datasets import Lee2019_MI  # noqa: E402
from moabb.paradigms import MotorImagery  # noqa: E402
from mne.preprocessing import compute_current_source_density as csd  # noqa: E402

from experiments.d2_lee2019_dev import (CHANNELS, CLASSES, N_BASE, N_FTS, SFREQ, score_subject,  # noqa: E402
                                        sign_flip_p, sign_test_share)
from src.adaptation.convnet import encode_labels  # noqa: E402
from src.config import DATA_DIR, FMAX, FMIN, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.evaluate import holm_adjust  # noqa: E402
from src.pipeline import CNN_BAND, CNN_TRAIN, PipelineConfig, SubjectData  # noqa: E402

mne.set_log_level("ERROR")
ARMS = {"l20": ("csd", CHANNELS), "a62": ("none", None), "l62": ("csd", None)}
CONTRASTS = {"H1": "l20", "H2": "a62", "H3": "l62"}
CACHE = DATA_DIR / "cache" / "e1_spatial"


def csd_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Spherical-spline CSD on the EEG channels, relabelled as EEG, stim channel kept (EMG dropped)."""
    c = csd(raw.copy().pick(mne.pick_types(raw.info, eeg=True)))
    out = mne.io.RawArray(c.get_data(), mne.create_info(c.ch_names, c.info["sfreq"], "eeg"), verbose=False)
    return out.add_channels([raw.copy().pick(mne.pick_types(raw.info, eeg=False, stim=True))])


class LeeArms(Lee2019_MI):
    """Lee2019_MI with an optional CSD transform on every run; the raw files of one subject are loaded once."""

    def __init__(self, ensure_montage: bool = False) -> None:
        super().__init__()
        self._selected_sessions = None  # MOABB reads (1, 2) as 0-based positions otherwise (D2 fix)
        self.transform, self._memo, self._ensure_montage = "none", {}, ensure_montage

    def _get_single_subject_data(self, subject):
        if subject not in self._memo:
            self._memo = {subject: super()._get_single_subject_data(subject)}
        sessions = self._memo[subject]
        if self.transform == "none":
            return sessions
        return {s: {r: csd_raw(raw) for r, raw in runs.items()} for s, runs in sessions.items()}


def load_arm(dataset: LeeArms, subject: int, arm: str, cnn_band: dict, ts_band: dict) -> SubjectData:
    transform, channels = ARMS[arm]
    f = CACHE / f"{arm}_s{subject}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return SubjectData(X_cnn=z["X_cnn"], X_tslr=z["X_tslr"], y=z["y"], groups=z["groups"], train=z["train"])
    dataset.transform = transform

    def get(b):
        kw = {"channels": channels} if channels else {}
        return MotorImagery(events=CLASSES, n_classes=2, resample=SFREQ, **kw, **b).get_data(
            dataset=dataset, subjects=[subject])

    Xw, yw, mw = get(cnn_band)
    Xn, yn, mn = get(ts_band)
    if not (np.array_equal(yw, yn) and np.array_equal(mw.session.to_numpy(), mn.session.to_numpy())):
        raise AssertionError(f"s{subject} {arm}: trial order differs between bands")
    sessions = sorted(mw.session.unique())
    if len(sessions) != 2:
        raise AssertionError(f"s{subject} {arm}: expected 2 sessions, got {sessions}")
    sess = mw.session.to_numpy()
    data = SubjectData(X_cnn=Xw.astype(np.float32), X_tslr=Xn.astype(np.float32), y=encode_labels(yw, CLASSES),
                       groups=sess, train=sess == sessions[0])
    print(f"  s{subject} {arm}: X_cnn {Xw.shape[1:]}, X_tslr {Xn.shape[1:]}, "
          f"counts {np.bincount(data.y[~data.train]).tolist()}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, X_cnn=data.X_cnn, X_tslr=data.X_tslr, y=data.y, groups=data.groups, train=data.train)
    return data


def summarise(res: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows, third = [], base.csp_lda.sort_values().index[: len(base) // 3]
    print(f"\n=== E1 spatial-information screen, Lee2019_MI, {len(base)} subjects (baseline = D2 R11) ===")
    print(f"baseline r11 {100 * base.r11.mean():.2f} | share >= 70 % {(base.r11 >= .7).mean():.1%} | "
          f"bottom third {100 * base.loc[third, 'r11'].mean():.2f}")
    for h, arm in CONTRASTS.items():
        a = res[res.arm == arm].set_index("subject")
        d = 100 * (a.r11 - base.r11)
        up, down, p_share = sign_test_share(a.r11.to_numpy(), base.r11.to_numpy())
        d3 = d.loc[third].to_numpy()
        rows.append({"contrast": h, "arm": arm, "r11": 100 * a.r11.mean(), "mean_pp": d.mean(),
                     "up": int((d > 0).sum()), "down": int((d < 0).sum()), "p_mean": sign_flip_p(d.to_numpy()),
                     "share": 100 * (a.r11 >= .7).mean(), "share_up": up, "share_down": down, "p_share": p_share,
                     "bottom3_pp": d3.mean(), "p_bottom3": sign_flip_p(d3),
                     "own_r11_minus_cnn1": 100 * (a.r11 - a.cnn1).mean(),
                     "own_r11_minus_csp": 100 * (a.r11 - a.csp_lda).mean(),
                     "own_r11_minus_tslr": 100 * (a.r11 - a.tslr).mean()})
    out = pd.DataFrame(rows)
    for col in ("p_mean", "p_share"):
        out[f"holm_{col}"] = holm_adjust(out[col].tolist())
    out["screen_pass"] = (out.mean_pp >= 0.5) & (out.holm_p_mean < 0.05)
    print(out.round(4).to_string(index=False))
    return out


def main() -> int:
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cnn_band, ts_band = dict(CNN_BAND), dict(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
    cfg = PipelineConfig(n_cnn=N_BASE, n_fts=N_FTS, fast=True)
    if args.dry_run:
        from moabb.datasets.fake import FakeDataset

        class FakeArms(FakeDataset):
            transform = "none"

            def _get_single_subject_data(self, subject):
                sessions = super()._get_single_subject_data(subject)
                if self.transform == "none":
                    return sessions
                out = {}
                for s, runs in sessions.items():
                    out[s] = {}
                    for r, raw in runs.items():
                        raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
                        out[s][r] = csd_raw(raw)
                return out

        extra = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8")
        dataset = FakeArms(event_list=CLASSES, n_sessions=2, n_runs=1, n_subjects=2, channels=tuple(CHANNELS + list(extra)),
                           sfreq=1000, duration=400, n_events=80, seed=0)
        cnn_band["tmax"], ts_band["tmax"] = 2.0, 2.0
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(CNN_TRAIN, epochs=2))
        subjects, device, cache_root = dataset.subject_list, torch.device("cpu"), DATA_DIR / "cache" / "e1_dryrun"
        CACHE = cache_root / "data"
    else:
        dataset = LeeArms()
        subjects, device, cache_root = dataset.subject_list, torch.device("cuda"), CACHE / "preds"
    rows = []
    for s in subjects:
        for arm in ARMS:
            data = load_arm(dataset, s, arm, cnn_band, ts_band)
            rows.append({"subject": s, "arm": arm, **score_subject(data, cfg, cache_root / f"{arm}_s{s}", device)})
            del data
            gc.collect()
    res = pd.DataFrame(rows)
    if args.dry_run:
        print(res.round(3).to_string())
        print("DRY RUN OK (no real data touched)")
        return 0
    res.to_csv(RESULTS_DIR / "e1_spatial_prior_screen.csv", index=False)
    base = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    summarise(res, base).to_csv(RESULTS_DIR / "e1_spatial_prior_screen_contrasts.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
