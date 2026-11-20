"""T1, TCFormer's published recipe on OUR data split, head to head with R11. PRE-REGISTERED (commit before the run).

Why (docs/project_audit.md): we have never measured the published bar. TCFormer (Altaheri et al., Sci. Rep. 2025,
MIT, https://github.com/Altaheri/TCFormer) reports 84.79 % on IV-2a (mean of 5 runs, last-epoch checkpoint, no
early stopping or validation selection). Our one earlier run of it (A1) used OUR trainer, not its recipe. Its
per-subject seed-0 result file is public and beats R11 on subjects 2, 4 and 6, which our D3 diagnosis called
signal-limited. Its input is broadband (the whole recorded band; D6 showed that on IV-2b the broadband advantage is
ocular), so the question this run answers is whether the 84.8 % bar is neural.

RECIPE (from the repository's configs/tcformer.yaml and the config.yaml saved beside the 84.8 % result files;
the yaml's default trans_depth 5 is NOT what produced them, depth 2 / 77 820 parameters did, and our vendored
model reproduces the 77 820 count):
  data      braindecode MOABBDataset BNCI2014_001 (our MOABB's name), EEG only, x 1e6, 250 Hz, NO filter, windows
            0-4 s after the cue (1000 samples, checked: 288 train / 288 test trials per subject), session 0train ->
            1test (same split as ours); per-channel z-score fitted on the training session and applied to both
  model     TCFormerModule(F1 32, kernels [20, 32, 64], D 2, pools 8 / 7, d_group 16, dropout_conv 0.4, TCN depth 2,
            kernel 4, dropout 0.3, group attention, q_heads 4, kv_heads 2, transformer depth 2, dropout 0.4)
  training  1000 epochs, batch 48, segment-and-recombine augmentation on every batch (utils/interaug.py: the batch
            doubles to 96), Adam lr 9e-4, betas (0.5, 0.999), weight decay 1e-3, 20 warm-up epochs then cosine decay
            stepped per epoch, cross-entropy, no validation set, last-epoch model scored once on the test session
  seeds     0, 1 for the raw arm; 0, 1 for the EOG arm (the paper averages 5; two is the budget, ~16 min per fit)
Deviations (stated, none expected to matter): braindecode 1.7 / MOABB renamed things, trial order inside the
session may differ from the original, fp32 without the repository's `set_float32_matmul_precision('medium')`.

ARMS
  raw   as above
  eog   as above but the three EOG channels are regressed out of the EEG first (the R14 regression: coefficients
        by least squares on the TRAINING session's trials only, applied to both sessions; the same code as
        experiments/r14_eog_gate.eog_regress), before the z-score
COMPARATORS: R11 per subject (results/r11_round3_diverse_members.csv, arm 'fts', = 82.45 %), the published seed-0
per-subject accuracies (listed below, from the repository's result file, used only as a reproduction check).

QUESTIONS (all descriptive: 9 subjects detect only large effects, MDE > 3 pp; no gate)
  V   reproduction: raw-arm mean over seeds vs 84.8 % (expected within ~1.5 pp) and per-subject seed-0 differences
      from the published seed-0 file
  Q1  raw TCFormer - R11, per subject and mean (paired sign-flip, two-sided)
  Q2  ocular share = raw - eog, per subject and mean
  Q3  eog TCFormer - R11 (the honest bar)
INTERPRETATION RULES fixed here: the bar is called NEURAL if eog - R11 >= +1 pp on the mean with at least 6 of 9
subjects up; called OCULAR/INFLATED if eog - R11 < +1 pp; otherwise inconclusive. Stage 2 (IV-2b, 500 epochs, no
augmentation, 3 channels, window stop -0.5 s as the repository) and Lee2019 only after seeing stage 1.

    .venv\\Scripts\\python experiments\\t1_tcformer_head_to_head.py --dry-run   # 2 epochs, one subject, no accuracies
    .venv\\Scripts\\python experiments\\t1_tcformer_head_to_head.py             # THE run (~9 h GPU, resumable)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from braindecode.datasets import MOABBDataset  # noqa: E402
from braindecode.preprocessing import Preprocessor, create_windows_from_events, preprocess  # noqa: E402

from src.config import DATA_DIR, IV2A_SUBJECTS, RESULTS_DIR  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.third_party.tcformer.tcformer_model import TCFormerModule  # noqa: E402

mne.set_log_level("ERROR")
CACHE = DATA_DIR / "cache" / "t1_preds"
PUBLISHED_SEED0 = [89.58, 75.35, 94.79, 85.42, 74.65, 72.92, 93.40, 88.54, 88.54]  # repository result file, IV-2a
PUBLISHED_MEAN = 84.79
MODEL_KW = dict(F1=32, temp_kernel_lengths=[20, 32, 64], D=2, pool_length_1=8, pool_length_2=7, dropout_conv=0.4,
                d_group=16, tcn_depth=2, kernel_length_tcn=4, dropout_tcn=0.3, use_group_attn=True, q_heads=4,
                kv_heads=2, trans_depth=2, trans_dropout=0.4)
EPOCHS, BATCH, LR, BETAS, WD, WARMUP = 1000, 48, 9e-4, (0.5, 0.999), 1e-3, 20
N_EEG, N_CLASSES = 22, 4
NEURAL_PP, NEURAL_MIN_UP = 1.0, 6


def load(subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X_train, y_train, X_test, y_test) with 22 EEG + 3 EOG channels (EEG first), broadband, 0-4 s, 250 Hz."""
    ds = MOABBDataset("BNCI2014_001", subject_ids=[subject])
    preprocess(ds, [Preprocessor("pick_types", eeg=True, eog=True, meg=False, stim=False),
                    Preprocessor(lambda x: x * 1e6, apply_on_array=True), Preprocessor("resample", sfreq=250)])
    wins = create_windows_from_events(ds, trial_start_offset_samples=0, trial_stop_offset_samples=0, preload=False)
    parts = wins.split("session")
    names = sorted(parts)
    if names != ["0train", "1test"]:
        raise AssertionError(f"unexpected sessions {names}")
    out = []
    for name in names:
        d = parts[name]
        out += [np.stack([d[i][0] for i in range(len(d))]).astype(np.float32),
                np.array([d[i][1] for i in range(len(d))], dtype=np.int64)]
    Xt, yt, Xe, ye = out
    if Xt.shape != (288, N_EEG + 3, 1000) or Xe.shape != (288, N_EEG + 3, 1000):
        raise AssertionError(f"s{subject}: unexpected shapes {Xt.shape} {Xe.shape}")
    return Xt, yt, Xe, ye


def eog_regress(eeg: np.ndarray, eog: np.ndarray) -> np.ndarray:
    """Same regression as experiments/r14_eog_gate.eog_regress; coefficients from the training trials passed in."""
    a = eog.transpose(0, 2, 1).reshape(-1, eog.shape[1])
    b = eeg.transpose(0, 2, 1).reshape(-1, eeg.shape[1])
    return np.linalg.lstsq(a, b, rcond=None)[0]


def prepare(arm: str, Xt: np.ndarray, Xe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    et, ee = Xt[:, :N_EEG].astype(np.float64), Xe[:, :N_EEG].astype(np.float64)
    if arm == "eog":
        coef = eog_regress(et, Xt[:, N_EEG:].astype(np.float64))
        et = et - np.einsum("nct,ce->net", Xt[:, N_EEG:].astype(np.float64), coef)
        ee = ee - np.einsum("nct,ce->net", Xe[:, N_EEG:].astype(np.float64), coef)
    mu, sd = et.transpose(1, 0, 2).reshape(N_EEG, -1).mean(1), et.transpose(1, 0, 2).reshape(N_EEG, -1).std(1)
    f = lambda a: ((a - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)  # noqa: E731
    return f(et), f(ee)


def interaug(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """utils/interaug.py of the repository: per class, new trials from random time chunks of that class's trials."""
    new_x, new_y, n_chunks = torch.zeros_like(x), torch.zeros_like(y), 8 if x.shape[-1] % 8 == 0 else 7
    cur = 0
    for cls in torch.unique(y):
        xc = x[y == cls]
        chunks = torch.cat(torch.chunk(xc, chunks=n_chunks, dim=-1))
        idx = torch.randint(0, len(xc), size=(len(xc), n_chunks), device=xc.device)
        for row in idx:
            row = row + torch.arange(0, chunks.shape[0], len(xc), device=xc.device)
            new_x[cur] = chunks[row].permute(1, 0, 2).reshape(1, xc.shape[1], xc.shape[2])
            new_y[cur] = cls
            cur += 1
    cx, cy = torch.cat((x, new_x)), torch.cat((y, new_y))
    perm = torch.randperm(len(cx), device=x.device)
    return cx[perm], cy[perm]


def lr_factor(epoch: int) -> float:
    if epoch < WARMUP:
        return epoch / max(1, WARMUP)
    return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP) / max(1, EPOCHS - WARMUP)))


def fit_predict(Xt: np.ndarray, yt: np.ndarray, Xe: np.ndarray, seed: int, epochs: int, device) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = TCFormerModule(N_EEG, N_CLASSES, **MODEL_KW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    X, y = torch.as_tensor(Xt, device=device), torch.as_tensor(yt, device=device)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X), device=device)
        for i in range(0, len(X), BATCH):
            xb, yb = interaug(X[order[i:i + BATCH]], y[order[i:i + BATCH]])
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        xe = torch.as_tensor(Xe, device=device)
        probs = np.concatenate([torch.softmax(model(xe[i:i + BATCH]), dim=1).cpu().numpy()
                                for i in range(0, len(xe), BATCH)])  # eval mode: chunking is exact
    del model, opt, X, y
    torch.cuda.empty_cache()  # the guard reads this process's own cached VRAM (7.4 GB after the 288-trial eval pass)
    return probs


def summarise(df: pd.DataFrame) -> None:
    r11 = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv")
    r11 = 100 * r11[(r11.dataset == "IV-2a") & (r11.arm == "fts")].set_index("subject").acc
    t = df.groupby(["subject", "arm"]).acc.mean().mul(100).unstack("arm")
    t["r11"] = r11
    print("\n=== T1 TCFormer head to head, IV-2a, seed-mean per subject (%) ===")
    print(t.round(1).T.to_string())
    for arm in ("raw", "eog"):
        print(f"{arm}: mean {t[arm].mean():.2f}")
    s0 = df[(df.seed == 0) & (df.arm == "raw")].set_index("subject").acc.mul(100)
    pub = pd.Series(PUBLISHED_SEED0, index=s0.index)
    print(f"\nV  reproduction: raw seed-mean {t.raw.mean():.2f} vs published {PUBLISHED_MEAN}; raw seed 0 vs published "
          f"seed 0: mean {s0.mean():.2f} vs {pub.mean():.2f}, per-subject diff mean {(s0 - pub).mean():+.2f}, "
          f"max |diff| {(s0 - pub).abs().max():.1f}")
    for name, a, b in (("Q1 raw - R11", "raw", "r11"), ("Q2 raw - eog (ocular share)", "raw", "eog"),
                       ("Q3 eog - R11", "eog", "r11")):
        d = (t[a] - t[b]).to_numpy()
        p = sign_flip_permutation_test(d, "two-sided")
        print(f"{name}: mean {d.mean():+.2f} pp, {(d > 0).sum()} up / {(d < 0).sum()} down, two-sided p = {p:.3f}; "
              f"per subject {np.round(d, 1).tolist()}")
    d = (t.eog - t.r11).to_numpy()
    verdict = ("NEURAL" if d.mean() >= NEURAL_PP and (d > 0).sum() >= NEURAL_MIN_UP else
               "OCULAR/INFLATED" if d.mean() < NEURAL_PP else "inconclusive")
    print(f"\nTHE BAR (eog TCFormer vs R11): {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", default="raw,eog")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--subjects", default=",".join(map(str, IV2A_SUBJECTS)))
    args = ap.parse_args()
    device = torch.device("cuda")
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in [int(x) for x in args.subjects.split(",")][: 1 if args.dry_run else None]:
        Xt, yt, Xe, ye = load(s)
        for arm in args.arms.split(","):
            At, Ae = prepare(arm, Xt, Xe)
            for seed in [int(x) for x in args.seeds.split(",")][: 1 if args.dry_run else None]:
                f = CACHE / f"IV-2a_{arm}_s{s}_seed{seed}.npy"
                t0 = time.time()
                if args.dry_run:
                    wait_until_healthy()
                    p = fit_predict(At, yt, Ae, seed, 2, device)
                    wait_until_healthy()  # must pass after a fit (the abort happened here)
                    print(f"dry run s{s} {arm} seed {seed}: 2 epochs in {time.time() - t0:.1f} s, probs {p.shape}, "
                          f"sum {p.sum(1).mean():.3f}")
                    continue
                if not f.exists():
                    wait_until_healthy()
                    np.save(f, fit_predict(At, yt, Ae, seed, EPOCHS, device))
                    print(f"  s{s} {arm} seed {seed}: {(time.time() - t0) / 60:.1f} min", flush=True)
                p = np.load(f)
                rows.append({"subject": s, "arm": arm, "seed": seed, "acc": float((p.argmax(1) == ye).mean())})
                torch.cuda.empty_cache()
    if args.dry_run:
        print("DRY RUN OK")
        return 0
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "t1_tcformer_head_to_head.csv", index=False)
    summarise(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
