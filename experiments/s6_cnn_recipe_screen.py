"""S6, CNN training-recipe screen: interaug and Mixup on R11's ShallowConvNet members. PRE-REGISTERED.

Committed BEFORE any arm is scored. Source: docs/benefit_complexity_review.md (S6) and T1: the published TCFormer recipe
beats our trainer by about 10 pp on the same architecture (A1 74.5 % against T1 84.4 %), so the recipe may matter more than
the architecture. Our trainer already uses cosine-annealed AdamW (src/adaptation/convnet.py), 300 epochs, batch 48,
lr 1e-3, weight decay 1e-3. FTSurrogate was the biggest single gain of the project (+4.1 pp, R8), so augmentation is the
lever with a track record. Two augmentations were never screened here: interaug (segment-and-recombine, TCFormer's) and
Mixup. The rest of the TCFormer recipe is not tested separately: epochs (1000) and per-channel z-score gave no gain on IV-2b.

ARMS (the CNN3 member of R11 = ShallowConvNet seeds 0-2 on per-session aligned 4-38 Hz input; all other members unchanged):
  base      the current trainer, retrained in this script (same loop as the arms, so the contrast is paired and exact)
  interaug  every batch is doubled with new trials made of random time chunks (8 chunks of 0.5 s) of same-class trials of
            that batch, as in TCFormer's utils/interaug.py; hard labels
  mixup     every batch is mixed with a shuffled copy of itself, lambda ~ Beta(0.4, 0.4) per batch, soft labels
Settings are fixed here and not tuned. Same seeds 0-2, one training run per seed and arm.

PRIMARY ENDPOINT. R11 with its CNN3 member replaced by the arm's CNN3 (TS+LR and the two FTSurrogate members are the cached
D2 / IV-2a ones), per-subject accuracy on session 2, Lee2019_MI, n = 54. Contrast: R11[arm] - R11[base], one-sided
sign-flip Monte Carlo (as D2), Holm over the 2 arms.
SECONDARY (reported): the single-model mean over seeds; CNN3 alone; IV-2a (9 subjects) with the same arms.
COST. Training wall-clock per fit of the arm divided by base (median over fits). R11 trains in about 0.5 min per user with
the stacked fast path; the rule allows up to +1 min per user, so the ratio must be <= 2.5.
ADOPT an arm if ALL hold: Lee2019 primary gain >= +1.0 pp with Holm p < 0.05; IV-2a primary gain >= 0; cost ratio <= 2.5.
An adopted arm is a candidate only: it would need a stacked-path implementation, an IV-2b replication, and (only if the
claimed effect is >= 3 pp) one Zhou2020 opening. If neither arm is adopted the recipe lane is closed.

    .venv\\Scripts\\python experiments\\s6_cnn_recipe_screen.py --dry-run   # 2 subjects, 5 epochs, plumbing only
    .venv\\Scripts\\python experiments\\s6_cnn_recipe_screen.py             # THE run (GPU, about 1-2 h, resumable)
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.d2_lee2019_dev import CACHE, load, sign_flip_p  # noqa: E402
from experiments.t1_tcformer_head_to_head import interaug  # noqa: E402
from src.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.evaluate import holm_adjust  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.adaptation.convnet import make_batches, predict_proba, set_deterministic  # noqa: E402
from src.pipeline import CNN_TRAIN, align_by_group, build_shallow, load_subject_data, tslr_member  # noqa: E402

ARMS = ("base", "interaug", "mixup")
TESTED = ("interaug", "mixup")
SEEDS = (0, 1, 2)
MIXUP_ALPHA = 0.4
MIN_GAIN_PP, ALPHA, MAX_COST_RATIO = 1.0, 0.05, 2.5
PRED_DIR = DATA_DIR / "cache" / "s6_preds"


def train_arm(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, cfg, seed: int, arm: str) -> torch.nn.Module:
    """The R11 trainer (AdamW, cosine, no early stopping) with a batch augmentation; ``y`` is one-hot."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    beta = torch.distributions.Beta(MIXUP_ALPHA, MIXUP_ALPHA)
    n_classes = y.shape[1]
    model.train()
    for _ in range(cfg.epochs):
        for b in make_batches(X.shape[0], None, cfg.batch_size, gen):
            idx = torch.as_tensor(b, device=X.device)
            xb, yb = X[idx], y[idx]
            if arm == "interaug":
                xb, cls = interaug(xb, yb.argmax(1))
                yb = F.one_hot(cls, n_classes).float()
            elif arm == "mixup":
                lam, perm = beta.sample().item(), torch.randperm(xb.shape[0], device=xb.device)
                xb, yb = lam * xb + (1 - lam) * xb[perm], lam * yb + (1 - lam) * yb[perm]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def fit_arm(tag: str, A: np.ndarray, train: np.ndarray, y: np.ndarray, n_classes: int, arm: str, seed: int,
            cfg, device) -> tuple[np.ndarray, float]:
    """Test-trial probabilities of one ShallowConvNet; cached per (tag, arm, seed). Returns (probs, seconds)."""
    f = PRED_DIR / f"{tag}_{arm}_seed{seed}.npy"
    if f.exists():
        z = np.load(f, allow_pickle=True).item()
        return z["p"], z["t"]
    wait_until_healthy()
    Xt = torch.as_tensor(A[train].astype(np.float32), device=device)
    Xe = torch.as_tensor(A[~train].astype(np.float32), device=device)
    yt = torch.as_tensor(np.eye(n_classes)[y[train]].astype(np.float32), device=device)
    set_deterministic(seed)
    model = build_shallow(A.shape[1], n_classes, A.shape[2], None).to(device)
    t0 = time.perf_counter()
    model = train_arm(model, Xt, yt, cfg, seed, arm)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    p = predict_proba(model, Xe)
    del model
    torch.cuda.empty_cache()
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, {"p": p, "t": sec}, allow_pickle=True)
    return p, sec


def subject_row(tag: str, data, n_classes: int, ts: np.ndarray, fts2: np.ndarray, cfg, seeds, device) -> dict[str, float]:
    A = align_by_group(data.X_cnn, data.groups)
    y_test = data.y[~data.train]
    acc = lambda p: float((p.argmax(1) == y_test).mean())  # noqa: E731
    row: dict[str, float] = {}
    for arm in ARMS:
        runs = [fit_arm(tag, A, data.train, data.y, n_classes, arm, s, cfg, device) for s in seeds]
        probs = [r[0] for r in runs]
        row[f"{arm}_single"] = float(np.mean([acc(p) for p in probs]))
        row[f"{arm}_cnn3"] = acc(np.mean(probs, 0))
        row[f"{arm}_r11"] = acc((np.mean(probs, 0) + ts + fts2) / 3)
        row[f"{arm}_sec"] = float(np.median([r[1] for r in runs]))
    return row


def lee_members(s: int) -> tuple[object, np.ndarray, np.ndarray]:
    data = load(None, s, {}, {}, ["cached"])
    c = CACHE / "preds" / f"s{s}"
    fts2 = np.mean([np.load(c / f"fts_seed{k}_fast.npy") for k in range(2)], 0)
    return data, tslr_member(data), fts2


def iv2a_members(s: int) -> tuple[object, np.ndarray, np.ndarray]:
    data, _ = load_subject_data("IV-2a", s)
    fts2 = np.mean([np.load(DATA_DIR / "cache" / "r8_preds" / f"s{s}_ftsurrogate_seed{k}.npy") for k in range(2)], 0)
    return data, tslr_member(data), fts2


def contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"arm": a, "mean_pp": 100 * (df[f"{a}_r11"] - df.base_r11).mean(),
             "single_pp": 100 * (df[f"{a}_single"] - df.base_single).mean(),
             "up": int((df[f"{a}_r11"] > df.base_r11).sum()), "down": int((df[f"{a}_r11"] < df.base_r11).sum()),
             "p": sign_flip_p((100 * (df[f"{a}_r11"] - df.base_r11)).to_numpy()),
             "cost_ratio": float((df[f"{a}_sec"] / df.base_sec).median())} for a in TESTED]
    out = pd.DataFrame(rows)
    out["holm_p"] = holm_adjust(out.p.tolist())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    dry = ap.parse_args().dry_run
    device = torch.device("cuda")
    cfg = dataclasses.replace(CNN_TRAIN, epochs=5) if dry else CNN_TRAIN
    seeds = SEEDS[:1] if dry else SEEDS
    global PRED_DIR
    if dry:
        PRED_DIR = DATA_DIR / "cache" / "s6_dryrun"
    lee_ids, iv_ids = (range(1, 3), range(1, 2)) if dry else (range(1, 55), range(1, 10))
    def rows(prefix: str, ids, members, n_classes: int) -> pd.DataFrame:
        out = {}
        for s in ids:
            data, ts, fts2 = members(s)
            out[s] = subject_row(f"{prefix}_s{s}", data, n_classes, ts, fts2, cfg, seeds, device)
        return pd.DataFrame(out).T

    lee, iv = rows("lee", lee_ids, lee_members, 2), rows("iv2a", iv_ids, iv2a_members, 4)
    if dry:
        print(f"DRY RUN OK: arms {list(ARMS)} on {len(lee)} Lee2019 and {len(iv)} IV-2a subjects (no accuracy printed)")
        return 0
    cl, ci = contrasts(lee), contrasts(iv)
    cl["iv2a_mean_pp"] = ci.mean_pp.to_numpy()
    cl["adopt"] = (cl.mean_pp >= MIN_GAIN_PP) & (cl.holm_p < ALPHA) & (cl.iv2a_mean_pp >= 0) & (cl.cost_ratio <= MAX_COST_RATIO)
    print("\n=== S6 CNN recipe screen: R11 with the arm's CNN3 member, gain over base ===")
    print(f"base R11: Lee2019 {100 * lee.base_r11.mean():.2f}, IV-2a {100 * iv.base_r11.mean():.2f}; "
          f"base single: Lee2019 {100 * lee.base_single.mean():.2f}")
    print(cl.round(4).to_string(index=False))
    for a in ARMS:
        print(f"  {a}: single {100 * lee[f'{a}_single'].mean():.2f}, cnn3 {100 * lee[f'{a}_cnn3'].mean():.2f}, "
              f"r11 {100 * lee[f'{a}_r11'].mean():.2f} (Lee2019)")
    print(f"\nADOPTED: {list(cl.loc[cl.adopt, 'arm']) or 'none: the recipe lane is closed'}")
    lee.to_csv(RESULTS_DIR / "s6_cnn_recipe_lee2019.csv")
    iv.to_csv(RESULTS_DIR / "s6_cnn_recipe_iv2a.csv")
    cl.to_csv(RESULTS_DIR / "s6_cnn_recipe_contrasts.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
