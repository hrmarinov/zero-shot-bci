"""A1: an architecture-diverse ensemble member (CTNet, TCFormer) in the R11 ensemble. PRE-REGISTERED.

Committed BEFORE the first real fit of either architecture on IV-2a/IV-2b. Implementation bugs found later
are fixed in a separate commit that leaves this protocol unchanged.

WHY THESE TWO (of braindecode 1.7.0's EEGNet, CTNet, TCFormer, ATCNet, EEGConformer; timing on random
IV-2a-shaped data, 300 epochs, our train_model: EEGNet 0.7, CTNet 0.9, EEGConformer 2.5, TCFormer 4.8,
ATCNet 6.7 min per fit; ShallowConvNet 0.5):
  TCFormer  the strongest leak-free published result on our exact split (84.8% IV-2a,
            docs/state_of_the_art.md section 0): the most likely to ADD accuracy. Temporal-conv + transformer
            + TCN head; no square/log variance pooling, so its errors should differ from ShallowConvNet's.
  CTNet     CNN + transformer encoder, strong published subject-specific results, and cheap (0.9 min):
            the same diversity argument at a fifth of TCFormer's cost.
  Not chosen: EEGNet (weaker standalone on cross-session IV-2a than ShallowConvNet in the literature, so a
  weak member), ATCNet / EEGConformer (cost; ATCNet alone would exceed the GPU budget).
  GPU budget: 2 seeds x 18 subjects x (0.9 + 4.8) min ~ 3.5 h, plus 2 determinism reruns.

FROZEN SETTINGS (exactly R11's CNN members): per-group Riemannian-aligned input (IV-2a per session, IV-2b
per recording), 4-38 Hz, 0-4 s, 250 Hz (src.pipeline.load_subject_data / align_by_group); our train_model
with R3's CFG (300 epochs, batch 48, AdamW lr 1e-3, wd 1e-3, cosine, no early stopping); integer targets
on IV-2a, one-hot on IV-2b (as the cached members); braindecode defaults for every architecture
hyper-parameter (n_chans, n_outputs, n_times, sfreq=250 given); each model built AFTER
set_deterministic(seed); seeds 0 and 1 for each architecture; src.gpu_health.wait_until_healthy before
each fit. Training sees the training session only.

MEMBERS (test-session probabilities; families averaged over their seeds first):
  CNN3 = base ShallowConvNet seeds 0-2, CNN1 = seed 0, CNNx = seeds 3-4 (cached r3_preds / r7_preds);
  FTS2 = FTSurrogate ShallowConvNet seeds 0-1 (cached r8_preds / r9_preds); TSLR recomputed (src.pipeline);
  CTN2 = CTNet seeds 0-1, TCF2 = TCFormer seeds 0-1 (trained here, cached in data/cache/a1_preds).
COMPARATOR  r11  = (CNN3 + TSLR + FTS2) / 3
ARMS (4, equal weights per family, fixed a priori):
  aug_ctn = (CNN3 + TSLR + FTS2 + CTN2) / 4      a new family ADDED (7 network fits)
  aug_tcf = (CNN3 + TSLR + FTS2 + TCF2) / 4
  rep_ctn = (CNN1 + TSLR + FTS2 + CTN2) / 4      two base seeds REPLACED: same 5 network fits as R11
  rep_tcf = (CNN1 + TSLR + FTS2 + TCF2) / 4
SIZE-MATCHED CONTROL  ctrl = (CNN3 + TSLR + FTS2 + CNNx) / 4   (fourth family = 2 more base seeds)

GATE (as R10-R13; score = accuracy, IV-2b averaged over its two test recordings as R11):
  arm - r11: pooled 18-subject mean >= +0.5 pp, one-sided exact sign-flip p < 0.05 after Holm over the 4
  arms, and mean > 0 on each dataset separately. Reported, not gated: aug_* - ctrl (diversity credit, the
  fourth family vs two more ShallowConvNet seeds), each architecture alone (seed-mean single-model and
  2-seed ensemble accuracy) vs a single ShallowConvNet.
WINNER: among passing arms the largest pooled gain (a tie within 0.1 pp goes to the cheaper arm: rep before
  aug, CTNet before TCFormer). No winner -> documented stop; R11 stays.
LOCKBOX (only if there is a winner; separate script committed before its run): the winner vs R11 once on
  BNCI2015_001 (L1 data path and caches), one-sided sign-flip over 12 subjects, p < 0.05.

VALIDITY: V1 trial order (load_subject_data) and every cached member has one row per test trial;
V2 r11 from cached members reproduces results/r11_round3_diverse_members.csv (arm 'fts') exactly (< 1e-12);
V3 on IV-2a s1, seed 0 of each architecture is retrained and must be bit-identical (determinism).

    .venv\\Scripts\\python experiments\\a1_arch_member_screen.py --smoke   # 2 epochs on IV-2a s1, nothing cached
    .venv\\Scripts\\python experiments\\a1_arch_member_screen.py           # THE run (~3.5 h GPU, resumable)
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from braindecode.models import CTNet, TCFormer  # noqa: E402

from experiments.check_pipeline_reproduces_r11 import REFERENCE  # noqa: E402
from experiments.r11_round3_diverse_members import CACHE, rec_acc  # noqa: E402
from src.adaptation.convnet import TrainConfig, predict_proba, set_deterministic, train_model  # noqa: E402
from src.config import IV2A_SUBJECTS, RESULTS_DIR  # noqa: E402
from src.evaluate import holm_adjust, sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.pipeline import CNN_TRAIN, SubjectData, align_by_group, load_subject_data, tslr_member  # noqa: E402

ARCHS = {"ctn": CTNet, "tcf": TCFormer}
ARCH_SEEDS = (0, 1)
SFREQ = 250
A1_CACHE = CACHE / "a1_preds"
OUT_CSV = RESULTS_DIR / "a1_arch_member_screen.csv"
ARMS = ("aug_ctn", "aug_tcf", "rep_ctn", "rep_tcf")
GATE_PP = 0.5
CONTROL_CACHE = {"IV-2a": lambda s, k: CACHE / "r3_preds" / f"s{s}_cnn_ra_seed{k}.npy",
                 "IV-2b": lambda s, k: CACHE / "r7_preds" / f"s{s}_teacher_seed{k}.npy"}


def arch_member(arch: str, data: SubjectData, A: np.ndarray, n_classes: int, seed: int, one_hot: bool,
                train: TrainConfig, device: torch.device) -> np.ndarray:
    """One braindecode network on the aligned CNN input, trained exactly as R11's ShallowConvNets."""
    Xt = torch.as_tensor(A[data.train].astype(np.float32), device=device)
    Xe = torch.as_tensor(A[~data.train].astype(np.float32), device=device)
    yt = data.y[data.train]
    target = (torch.as_tensor(np.eye(n_classes)[yt].astype(np.float32), device=device) if one_hot
              else torch.as_tensor(yt, dtype=torch.long, device=device))
    set_deterministic(seed)
    model = ARCHS[arch](n_chans=A.shape[1], n_outputs=n_classes, n_times=A.shape[2], sfreq=SFREQ).to(device)
    model = train_model(model, Xt, target, train, seed=seed)
    probs = predict_proba(model, Xe)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


def members_for(ds: str, s: int, data: SubjectData, n_classes: int, device, train: TrainConfig,
                cache: Path | None) -> dict[str, np.ndarray]:
    A = None
    out = {}
    for arch in ARCHS:
        for k in ARCH_SEEDS:
            path = None if cache is None else cache / f"{arch}_seed{k}.npy"
            if path is not None and path.exists():
                out[f"{arch}{k}"] = np.load(path)
                continue
            if A is None:
                A = align_by_group(data.X_cnn, data.groups)
            if device.type == "cuda":
                wait_until_healthy()
            p = arch_member(arch, data, A, n_classes, k, ds == "IV-2b", train, device)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, p)
            out[f"{arch}{k}"] = p
            if path is not None:  # the smoke run prints nothing scored on the evaluation session
                print(f"    {ds} s{s} {arch} seed{k}: acc {100 * (p.argmax(1) == data.y[~data.train]).mean():.2f}",
                      flush=True)
    return out


def score(ds: str, s: int, data: SubjectData, new: dict[str, np.ndarray], ref: pd.Series) -> dict:
    y = data.y[~data.train]
    rec = data.groups[~data.train] if ds == "IV-2b" else np.zeros(len(y))
    base = [np.load(REFERENCE[ds]["cnn"](s, k)) for k in range(3)] + \
           [np.load(CONTROL_CACHE[ds](s, k)) for k in (3, 4)]
    fts = [np.load(REFERENCE[ds]["fts"](s, k)) for k in range(2)]
    for p in base + fts + list(new.values()):
        if p.shape[0] != len(y):
            raise AssertionError(f"V1: {ds} s{s}: a member has {p.shape[0]} rows, expected {len(y)}")
    ts = tslr_member(data)
    cnn3, cnn1, cnnx, fts2 = np.mean(base[:3], 0), base[0], np.mean(base[3:], 0), np.mean(fts, 0)
    fam = {a: np.mean([new[f"{a}{k}"] for k in ARCH_SEEDS], 0) for a in ARCHS}
    probs = {"r11": (cnn3 + ts + fts2) / 3, "ctrl": (cnn3 + ts + fts2 + cnnx) / 4,
             "aug_ctn": (cnn3 + ts + fts2 + fam["ctn"]) / 4, "aug_tcf": (cnn3 + ts + fts2 + fam["tcf"]) / 4,
             "rep_ctn": (cnn1 + ts + fts2 + fam["ctn"]) / 4, "rep_tcf": (cnn1 + ts + fts2 + fam["tcf"]) / 4,
             "ctn2": fam["ctn"], "tcf2": fam["tcf"]}
    row = {"dataset": ds, "subject": s, **{k: rec_acc(p, y, rec) for k, p in probs.items()}}
    row["cnn1_mean"] = float(np.mean([rec_acc(p, y, rec) for p in base[:3]]))
    for a in ARCHS:
        row[f"{a}1_mean"] = float(np.mean([rec_acc(new[f"{a}{k}"], y, rec) for k in ARCH_SEEDS]))
    if not abs(row["r11"] - ref[(ds, s, "fts")]) < 1e-12:
        raise AssertionError(f"V2: {ds} s{s} r11 {row['r11']:.6f} != recorded {ref[(ds, s, 'fts')]:.6f}")
    return row


def check_determinism(device) -> None:
    """V3: IV-2a s1, seed 0 of each architecture retrained -> bit-identical to the cached prediction."""
    data, n_classes = load_subject_data("IV-2a", 1)
    A = align_by_group(data.X_cnn, data.groups)
    for arch in ARCHS:
        wait_until_healthy()
        again = arch_member(arch, data, A, n_classes, 0, False, CNN_TRAIN, device)
        cached = np.load(A1_CACHE / "IV-2a_s1" / f"{arch}_seed0.npy")
        if not np.array_equal(again, cached):
            raise AssertionError(f"V3: {arch} seed 0 is not bit-identical on rerun "
                                 f"(max diff {np.abs(again - cached).max():.2e})")
    print("  V3 passed: both architectures rerun bit-identically", flush=True)


def summarise(df: pd.DataFrame) -> None:
    pp = df.set_index(["dataset", "subject"]) * 100
    print("\n=== A1: architecture-diverse member, IV-2a + IV-2b, 18 subjects ===")
    cols = ["r11", "ctrl", *ARMS, "cnn1_mean", "ctn1_mean", "tcf1_mean", "ctn2", "tcf2"]
    print(pp[cols].groupby(level="dataset").mean().round(2).to_string())
    ps = {a: sign_flip_permutation_test((pp[a] - pp.r11).to_numpy(), "greater") for a in ARMS}
    holm = dict(zip(ps, holm_adjust(list(ps.values()))))
    print("\nGATE: arm - r11 (pooled >= +0.5 pp, Holm one-sided sign-flip p < 0.05 over 4 arms, > 0 on each dataset)")
    passing = {}
    for a in ARMS:
        d = pp[a] - pp.r11
        per = d.groupby(level="dataset").mean()
        ok = d.mean() >= GATE_PP and holm[a] < 0.05 and (per > 0).all()
        if ok:
            passing[a] = d.mean()
        print(f"  {a} - r11: pooled {d.mean():+.2f} pp, {(d > 0).sum()} up / {(d < 0).sum()} down, "
              f"p = {ps[a]:.4f} (Holm {holm[a]:.4f}); " + ", ".join(f"{k} {v:+.2f}" for k, v in per.items())
              + f" -> {'PASS' if ok else 'fail'}")
    print("\nREPORTED (no gate):")
    for a in ("aug_ctn", "aug_tcf", "ctrl"):
        other = "r11" if a == "ctrl" else "ctrl"
        d = pp[a] - pp[other]
        print(f"  {a} - {other}: pooled {d.mean():+.2f} pp, {(d > 0).sum()} up / {(d < 0).sum()} down, "
              f"p = {sign_flip_permutation_test(d.to_numpy(), 'greater'):.4f}; "
              + ", ".join(f"{k} {v:+.2f}" for k, v in d.groupby(level='dataset').mean().items()))
    for a in ARCHS:
        d = pp[f"{a}1_mean"] - pp.cnn1_mean
        print(f"  {a} single model - ShallowConvNet single model: pooled {d.mean():+.2f} pp, "
              f"{(d > 0).sum()} up / {(d < 0).sum()} down")
    if not passing:
        print("\nNO WINNER: no arm passes the gate; R11 stays the pipeline.")
        return
    order = {"rep_ctn": 0, "rep_tcf": 1, "aug_ctn": 2, "aug_tcf": 3}
    best = max(passing.values())
    winner = min((a for a, g in passing.items() if g >= best - 0.1), key=order.get)
    print(f"\nWINNER: {winner} (pooled {passing[winner]:+.2f} pp) -> confirm once on the BNCI2015_001 lockbox")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 epochs on IV-2a s1, nothing cached or written")
    args = ap.parse_args()
    device = torch.device("cuda")
    if args.smoke:
        data, n_classes = load_subject_data("IV-2a", 1)
        train = dataclasses.replace(CNN_TRAIN, epochs=2)
        new = members_for("IV-2a", 1, data, n_classes, device, train, None)
        new2 = members_for("IV-2a", 1, data, n_classes, device, train, None)
        for k in new:
            if not np.array_equal(new[k], new2[k]):
                raise AssertionError(f"smoke: {k} not deterministic")
        print("SMOKE OK (deterministic 2-epoch fits; nothing cached)")
        return 0
    ref = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv").set_index(["dataset", "subject", "arm"]).acc
    rows = []
    for s in IV2A_SUBJECTS:
        for ds in ("IV-2a", "IV-2b"):
            data, n_classes = load_subject_data(ds, s)
            new = members_for(ds, s, data, n_classes, device, CNN_TRAIN, A1_CACHE / f"{ds}_s{s}")
            rows.append(score(ds, s, data, new, ref))
            r = rows[-1]
            print(f"  {ds} s{s}: r11 {100 * r['r11']:.2f} | " + " ".join(f"{a} {100 * r[a]:.2f}" for a in ARMS),
                  flush=True)
        if s == IV2A_SUBJECTS[0]:
            check_determinism(device)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summarise(df)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
