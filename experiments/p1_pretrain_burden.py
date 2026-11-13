"""P1: population-pretrained ShallowConvNets in R11, to cut the labelled calibration trials. PRE-REGISTERED.

Committed BEFORE the first real run (development or lockbox). Implementation bugs found later are fixed in a
separate commit that leaves this protocol unchanged.

QUESTION (TODO A23). R11 needs ~144 labelled trials to match a single CNN on the full session (B1, L2). Its
CNN members start from random weights. If they start from weights pretrained on OTHER subjects, do fewer
labelled trials of the new user suffice?

POPULATION DATA RULE. Pretraining for target subject s uses only the TRAINING sessions of the other subjects of
the same dataset (IV-2a: session T of the 8 others; IV-2b: recordings 1-3 of the 8 others; lockbox: session A
of the 11 others), each aligned on its OWN trials (src.pipeline.align_by_group), with their labels. Nobody's
evaluation session is used for pretraining. The target's own data enter only at fine-tuning (its drawn
training trials) and test (its unlabelled evaluation session, aligned on itself as in R11). No computation
aligns or compares two subjects' covariances; subjects only share a supervised loss (TODO's "pop-data, safe").

PRETRAINING (per target subject, fixed a priori): R11's five ShallowConvNets (base seeds 0-2, FTSurrogate
seeds 0-1, FTSurrogate p = 0.5 during pretraining too), each built after set_deterministic(seed), trained
together with the stacked fast path (src/adaptation/stacked_convnet.py; F1: accuracy identical to the exact
path) on the pooled aligned trials for 100 epochs (R3 CFG otherwise: batch 48, AdamW lr 1e-3, wd 1e-3,
cosine; the pool is ~8x one session, so 100 epochs is ~2.7x R11's gradient steps).
FINE-TUNING: each member re-seeded (set_deterministic(seed)), initialised with its pretrained weights and BN
statistics, then trained on the target's drawn trials with R3 CFG for 100 epochs (stacked fast path). TS+LR
is R11's (trained on the drawn trials only; unchanged).
BUDGETS AND DRAWS: exactly B1's (IV-2a/2b: 48, 96, 144 trials x draws 0-2, seed 1000 n + 100 d + s, stratified,
plus FULL) and L2's on the lockbox (48, 100, 144 x 3, plus FULL). Training-side alignment fitted on the drawn
trials only, as B1. Comparators are read from B1/L2/R11 caches (exact path; F1 showed fast = exact accuracy).

ARMS (per budget; score = accuracy, IV-2b averaged over its two test recordings; draws averaged):
  r11     R11 at that budget (B1 / L2 caches; FULL: R11 / L1 caches)
  r11pt   (CNN3pt + TSLR + FTS2pt) / 3, the CNN members pretrained then fine-tuned    <- under test
  cnn1pt  mean single-model accuracy of the 3 fine-tuned base seeds;  cnn1 the same from scratch (B1)
  zs      ZERO-SHOT, no calibration at all: (CNN3pre + FTS2pre) / 2, the pretrained members applied to the
          target's aligned evaluation session without fine-tuning (reported once per subject; budget 0)
HYPOTHESES (development, 18 subjects pooled; gate as R10-R13):
  H1  r11pt - r11 at 48 trials;  H2  the same at 96 trials. Each PASSES iff pooled mean >= +0.5 pp, one-sided
      exact sign-flip p < 0.05 after Holm over {H1, H2}, and mean > 0 on each dataset.
REPORTED (no gate): every budget incl. 144 and FULL; parity r11pt@n - cnn1@FULL with the one-sided 95%
  bootstrap lower bound (B1's parity test; margin -1 pp); cnn1pt - cnn1; zero-shot accuracy vs chance and
  vs r11@48.
LOCKBOX (`--lockbox`, only if H1 or H2 passes; run once): the same contrasts at 48 and 100 trials on
  BNCI2015_001 (12 subjects), Holm over the two, one-sided sign-flip p < 0.05 and mean > 0 -> CONFIRMED.

VALIDITY: V1 trial order (loaders); V2 every comparator is read from a cache whose recomputed accuracy equals
its CSV (B1/L2: r11 per draw < 1e-12); V3 the pretraining pool never contains the target subject (asserted)
and never an evaluation session; V4 fine-tuned members start from the pretrained weights (asserted equal
before training).

    .venv\\Scripts\\python experiments\\p1_pretrain_burden.py             # development (~1.5 h GPU, resumable)
    .venv\\Scripts\\python experiments\\p1_pretrain_burden.py --lockbox   # BNCI2015_001, once, if H1 or H2 passes
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

from experiments.b1_r11_calibration_burden import stratified_draw, subset  # noqa: E402
from experiments.r11_round3_diverse_members import CACHE, rec_acc  # noqa: E402
from src.adaptation.augmentation import make_augmentation  # noqa: E402
from src.adaptation.convnet import ShallowConvNet, set_deterministic  # noqa: E402
from src.adaptation.stacked_convnet import (StackedShallowConvNet, dropout_rng_state,  # noqa: E402
                                            predict_proba_stacked, train_stacked)
from src.config import DATA_DIR, FMAX, FMIN, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.evaluate import holm_adjust, sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.pipeline import CNN_BAND, CNN_TRAIN, SubjectData, align_by_group, load_subject_data, tslr_member  # noqa: E402

mne.set_log_level("ERROR")
SPECS = [(0, None), (1, None), (2, None), (0, "ftsurrogate"), (1, "ftsurrogate")]  # (seed, aug): CNN3 + FTS2
PRE_CFG = dataclasses.replace(CNN_TRAIN, epochs=100)
FT_CFG = dataclasses.replace(CNN_TRAIN, epochs=100)
DEV_BUDGETS, LOCK_BUDGETS, DRAWS = (48, 96, 144), (48, 100, 144), (0, 1, 2)
GATE_PP, PARITY_MARGIN_PP, N_BOOT = 0.5, -1.0, 10_000
P1_CACHE = CACHE / "p1_preds"


def _target(y: np.ndarray, n_classes: int, one_hot: bool, device) -> torch.Tensor:
    return (torch.as_tensor(np.eye(n_classes)[y].astype(np.float32), device=device) if one_hot
            else torch.as_tensor(y, dtype=torch.long, device=device))


def _stack(n_chans: int, n_classes: int, n_times: int, init: list[dict] | None, device):
    members, states, augs = [], [], []
    for i, (seed, aug) in enumerate(SPECS):
        set_deterministic(seed)
        augs.append(None if aug is None else make_augmentation(aug, seed))
        m = ShallowConvNet(n_chans, n_classes, n_times)
        if init is not None:
            m.load_state_dict(init[i])
        members.append(m.to(device))
        states.append(dropout_rng_state(device))
    model = StackedShallowConvNet(members).to(device)
    if init is not None:  # V4
        for i, m in enumerate(members):
            if not torch.equal(model.temporal_w[i].detach().cpu(), init[i]["temporal_w"]):
                raise AssertionError("V4: a fine-tuned member does not start from its pretrained weights")
    return model, states, augs


def pretrain(pool_X: np.ndarray, pool_y: np.ndarray, n_classes: int, one_hot: bool, path: Path, device) -> list[dict]:
    if path.exists():
        return torch.load(path)
    wait_until_healthy()
    model, states, augs = _stack(pool_X.shape[1], n_classes, pool_X.shape[2], None, device)
    X = torch.as_tensor(pool_X, device=device)
    train_stacked(model, X, _target(pool_y, n_classes, one_hot, device), PRE_CFG, [s for s, _ in SPECS], augs, states)
    out = [{k: v.cpu() for k, v in model.unstack(i, ShallowConvNet(pool_X.shape[1], n_classes, pool_X.shape[2]))
            .state_dict().items()} for i in range(len(SPECS))]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    del model, X
    torch.cuda.empty_cache()
    return out


def predict_members(init: list[dict], data: SubjectData, n_classes: int, one_hot: bool, finetune: bool,
                    cache: Path, device) -> list[np.ndarray]:
    """Test probabilities of the 5 members: pretrained (finetune=False) or fine-tuned on data's training trials."""
    paths = [cache / f"{'cnn' if a is None else 'fts'}_seed{s}.npy" for s, a in SPECS]
    if all(p.exists() for p in paths):
        return [np.load(p) for p in paths]
    A = align_by_group(data.X_cnn, data.groups)
    model, states, augs = _stack(A.shape[1], n_classes, A.shape[2], init, device)
    if finetune:
        wait_until_healthy()
        Xt = torch.as_tensor(A[data.train].astype(np.float32), device=device)
        train_stacked(model, Xt, _target(data.y[data.train], n_classes, one_hot, device), FT_CFG,
                      [s for s, _ in SPECS], augs, states)
    probs = list(predict_proba_stacked(model, torch.as_tensor(A[~data.train].astype(np.float32), device=device)))
    cache.mkdir(parents=True, exist_ok=True)
    for p, path in zip(probs, paths):
        np.save(path, p)
    del model
    torch.cuda.empty_cache()
    return probs


def pool_for(target: int, pools: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    others = [s for s in pools if s != target]
    if target in others or len(others) != len(pools) - 1:
        raise AssertionError("V3: the pretraining pool contains the target subject")
    return np.concatenate([pools[s][0] for s in others]), np.concatenate([pools[s][1] for s in others])


def own_pool(data: SubjectData) -> tuple[np.ndarray, np.ndarray]:
    """A subject's TRAINING trials only, aligned on themselves (per group), float32."""
    tr = data.train
    return align_by_group(data.X_cnn[tr], data.groups[tr]).astype(np.float32), data.y[tr]


def subject_rows(name: str, s: int, data: SubjectData, n_classes: int, one_hot: bool, init: list[dict],
                 budgets: tuple[int, ...], comparator, score, device) -> list[dict]:
    """Every budget's arms for one target subject. ``comparator(budget, draw)`` -> (cnn3 list, fts2 list)."""
    tag = f"{name}_s{s}"
    rows = []
    zs = predict_members(init, data, n_classes, one_hot, False, P1_CACHE / tag / "zeroshot", device)
    y = data.y[~data.train]
    zs_ens = (np.mean(zs[:3], 0) + np.mean(zs[3:], 0)) / 2
    rows.append({"dataset": name, "subject": s, "budget": "0", "draw": 0, "zs": score(data, zs_ens),
                 "cnn1zs": float(np.mean([score(data, p) for p in zs[:3]]))})
    plan = [("FULL", 0, data)] + [(str(n), d, subset(data, stratified_draw(data.y[data.train], n,
                                                                          seed=1000 * n + 100 * d + s)))
                                  for n in budgets for d in DRAWS]
    for budget, d, sub in plan:
        ft = predict_members(init, sub, n_classes, one_hot, True, P1_CACHE / tag / f"n{budget}_d{d}", device)
        base, fts = comparator(budget, d)
        ts = tslr_member(sub)
        for p in [*ft, *base, *fts]:
            if p.shape[0] != len(y):
                raise AssertionError(f"V1: {tag} n{budget} d{d}: a member has {p.shape[0]} rows, expected {len(y)}")
        rows.append({"dataset": name, "subject": s, "budget": budget, "draw": d,
                     "r11": score(sub, (np.mean(base, 0) + ts + np.mean(fts, 0)) / 3),
                     "r11pt": score(sub, (np.mean(ft[:3], 0) + ts + np.mean(ft[3:], 0)) / 3),
                     "cnn1": float(np.mean([score(sub, p) for p in base])),
                     "cnn1pt": float(np.mean([score(sub, p) for p in ft[:3]]))})
    return rows


def run_dev(device) -> pd.DataFrame:
    b1 = pd.read_csv(RESULTS_DIR / "b1_r11_calibration_burden.csv")
    b1 = b1.set_index(["dataset", "subject", "budget", "draw"]).r11
    rows = []
    for ds in ("IV-2a", "IV-2b"):
        pools = {s: own_pool(load_subject_data(ds, s)[0]) for s in IV2A_SUBJECTS}  # RAM: pools only
        for s in IV2A_SUBJECTS:
            data, n_classes = load_subject_data(ds, s)
            one_hot = ds == "IV-2b"
            init = pretrain(*pool_for(s, pools), n_classes, one_hot, P1_CACHE / f"{ds}_s{s}" / "pretrained.pt", device)

            def comparator(budget, d, ds=ds, s=s):
                c = CACHE / ("pipeline_preds" / Path(f"{ds}_s{s}") if budget == "FULL"
                             else Path("b1_preds") / f"{ds}_s{s}_n{budget}_d{d}")
                return ([np.load(c / f"cnn_seed{k}.npy") for k in range(3)],
                        [np.load(c / f"fts_seed{k}.npy") for k in range(2)])

            def score(sub, p):
                rec = sub.groups[~sub.train] if ds == "IV-2b" else np.zeros((~sub.train).sum())
                return rec_acc(p, sub.y[~sub.train], rec)

            new = subject_rows(ds, s, data, n_classes, one_hot, init, DEV_BUDGETS, comparator, score, device)
            for r in new[1:]:  # V2
                if not abs(r["r11"] - b1[(ds, s, r["budget"], r["draw"])]) < 1e-12:
                    raise AssertionError(f"V2: {ds} s{s} n{r['budget']} d{r['draw']} r11 differs from B1")
            rows += new
            print(f"  {ds} s{s}: zero-shot {100 * new[0]['zs']:.2f} | " + " ".join(
                f"n{r['budget']}d{r['draw']} {100 * r['r11']:.1f}->{100 * r['r11pt']:.1f}" for r in new[1:5]), flush=True)
        del pools
    return pd.DataFrame(rows)


def run_lockbox(device) -> pd.DataFrame:
    from moabb.datasets import BNCI2015_001

    from experiments.l1_lockbox_bnci2015_001 import load
    dataset = BNCI2015_001()
    l2 = pd.read_csv(RESULTS_DIR / "l2_lockbox_burden.csv").set_index(["subject", "budget", "draw"]).r11
    ts_band = dict(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
    loaded = {s: load(dataset, s, dict(CNN_BAND), ts_band) for s in dataset.subject_list}
    pools = {s: own_pool(d) for s, d in loaded.items()}
    rows = []
    for s in dataset.subject_list:
        init = pretrain(*pool_for(s, pools), 2, False, P1_CACHE / f"lockbox_s{s}" / "pretrained.pt", device)

        def comparator(budget, d, s=s):
            c = (DATA_DIR / "cache" / "l1_preds" / f"s{s}" if budget == "FULL"
                 else DATA_DIR / "cache" / "l2_preds" / f"s{s}_n{budget}_d{d}")
            return ([np.load(c / f"cnn_seed{k}.npy") for k in range(3)],
                    [np.load(c / f"fts_seed{k}.npy") for k in range(2)])

        def score(sub, p):
            return float((p.argmax(1) == sub.y[~sub.train]).mean())

        new = subject_rows("BNCI2015_001", s, loaded[s], 2, False, init, LOCK_BUDGETS, comparator, score, device)
        for r in new[1:]:  # V2
            if not abs(r["r11"] - l2[(s, r["budget"], r["draw"])]) < 1e-12:
                raise AssertionError(f"V2: lockbox s{s} n{r['budget']} d{r['draw']} r11 differs from L2")
        rows += new
        print(f"  lockbox s{s}: zero-shot {100 * new[0]['zs']:.2f}", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, budgets: tuple[int, ...], primary: tuple[int, int], label: str) -> None:
    zs = df[df.budget == "0"].set_index(["dataset", "subject"])[["zs", "cnn1zs"]] * 100
    per = (df[df.budget != "0"].groupby(["dataset", "subject", "budget"])[["r11", "r11pt", "cnn1", "cnn1pt"]]
           .mean() * 100)
    order = [str(n) for n in budgets] + ["FULL"]
    print(f"\n=== P1 {label}: pretrained CNN members in R11, {len(zs)} subjects ===")
    print("zero-shot (no calibration):", zs.groupby(level="dataset").mean().round(2).to_dict("index"))
    tab = per.groupby(level=["budget", "dataset"]).mean().round(2)
    print(tab.reindex(order, level="budget").to_string())
    full_cnn1 = per.xs("FULL", level="budget").cnn1
    ps, ds_means = {}, {}
    for n in primary:
        d = per.xs(str(n), level="budget")
        diff = d.r11pt - d.r11
        ps[n] = sign_flip_permutation_test(diff.to_numpy(), "greater")
        ds_means[n] = diff.groupby(level="dataset").mean()
    holm = dict(zip(ps, holm_adjust(list(ps.values()))))
    print("\nGATE r11pt - r11 (pooled >= +0.5 pp, Holm one-sided sign-flip p < 0.05, > 0 on each dataset):")
    passed = []
    for n in primary:
        d = per.xs(str(n), level="budget")
        diff = d.r11pt - d.r11
        ok = diff.mean() >= GATE_PP and holm[n] < 0.05 and (ds_means[n] > 0).all()
        passed += [n] if ok else []
        print(f"  n={n}: pooled {diff.mean():+.2f} pp, {(diff > 0).sum()} up / {(diff < 0).sum()} down, "
              f"p = {ps[n]:.4f} (Holm {holm[n]:.4f}); " + ", ".join(f"{k} {v:+.2f}" for k, v in ds_means[n].items())
              + f" -> {'PASS' if ok else 'fail'}")
    print("\nREPORTED:")
    rng = np.random.default_rng(0)
    for b in order:
        d = per.xs(b, level="budget")
        par = (d.r11pt - full_cnn1).to_numpy()
        lo = np.percentile(par[rng.integers(0, len(par), (N_BOOT, len(par)))].mean(1), 5)
        print(f"  n={b}: r11pt - r11 {(d.r11pt - d.r11).mean():+.2f}, cnn1pt - cnn1 {(d.cnn1pt - d.cnn1).mean():+.2f}; "
              f"parity r11pt@n - cnn1@FULL {par.mean():+.2f} (lower bound {lo:+.2f} vs {PARITY_MARGIN_PP:+.1f} -> "
              f"{'non-inferior' if lo > PARITY_MARGIN_PP else 'not shown'})")
    print(f"\n{'PASSED: ' + ', '.join(f'n={n}' for n in passed) if passed else 'NO BUDGET PASSES'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lockbox", action="store_true", help="the single BNCI2015_001 run (only if dev passes)")
    args = ap.parse_args()
    device = torch.device("cuda")
    if args.lockbox:
        df = run_lockbox(device)
        out = RESULTS_DIR / "p1_pretrain_burden_lockbox.csv"
        df.to_csv(out, index=False)
        summarise(df, LOCK_BUDGETS, (48, 100), "LOCKBOX (BNCI2015_001)")
    else:
        df = run_dev(device)
        out = RESULTS_DIR / "p1_pretrain_burden.csv"
        df.to_csv(out, index=False)
        summarise(df, DEV_BUDGETS, (48, 96), "development (IV-2a + IV-2b)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
