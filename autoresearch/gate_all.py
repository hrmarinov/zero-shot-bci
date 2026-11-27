"""Evidence gate over ALL members, with GPU out-of-fold CNN predictions on session 1. PRE-REGISTERED.

Committed BEFORE any out-of-fold fit is made or any arm is scored. alien.py gated only the covariance members. Here CNN3 and FTS2 also
get an honest session-1 score: session 1 is aligned on its own trials (label-free), split into 5 stratified folds (seed 0), and for each
fold R11's five networks (CNN seeds 0-2, FTSurrogate seeds 0-1; stacked fast path, R11 training settings) are trained on the other four
folds and predict the held-out one. The test-session probabilities are the cached R11 ones. Out-of-fold nets see 80 % of the trials, so
their session-1 score slightly understates the full nets (a mild bias toward the pooled member, stated here).

MEMBERS: CNN3, own TS+LR, FTS2, zero-shot pool. Scores on session 1: out-of-fold for the first three, direct for the pool (it never
sees the subject's labels). ARMS = weighted mean of test probabilities with per-subject weights fitted on session 1 (constants fixed):
  full_ml      weights on the 4-simplex maximising the session-1 log-likelihood of the linear opinion pool (EM, 500 steps)
  full_map50   the same with a Dirichlet prior of 50 pseudo-trials toward equal weights
  full_map200  the same with 200 pseudo-trials
  full_bma     weights proportional to exp(10 * mean session-1 log-likelihood) of each member
  r11_map50    R11's three members only (no pool), prior of 50 pseudo-trials toward equal weights
COHORTS as alien.py: Lee2019 TUNE (27, library 26) and IV-2a (9, library 8). HOLDOUT not read; Zhou2020 sealed.
INCUMBENT = alien.py's stack (dawid_skene, gate_slot; Lee TUNE 74.593, IV-2a 82.562), which must be reproduced. Dawid-Skene ignores
member weights, so the arms replace that stack rather than add to it.
KEEP (as alien.py): Lee TUNE >= incumbent + 0.3 pp with more subjects up than down AND IV-2a >= incumbent - 0.25 pp. If several arms
pass, the one with the highest Lee TUNE mean is the new incumbent. Against R11 each arm is also reported. No further rounds.
IV-2a and TUNE are selection sets: any pass is a development result.

    .venv\\Scripts\\python autoresearch\\gate_all.py --fit     # GPU, resumable
    .venv\\Scripts\\python autoresearch\\gate_all.py           # score
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import prepare  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.pipeline import SubjectData, align_by_group, load_subject_data  # noqa: E402

OOF = DATA_DIR / "cache" / "ar_oof"
SPECS = [(0, None), (1, None), (2, None), (0, "ftsurrogate"), (1, "ftsurrogate")]
EM_STEPS, BMA_TEMP = 500, 10.0
ARMS = {"full_ml": ("ml", 0, 4), "full_map50": ("ml", 50, 4), "full_map200": ("ml", 200, 4), "full_bma": ("bma", 0, 4),
        "r11_map50": ("ml", 50, 3)}
MIN_GAIN, IV_TOL = 0.3, 0.25


def subjects():
    """(cohort, subject, loader of SubjectData, n_classes)."""
    for s in prepare.TUNE:
        yield "lee", s, (lambda s=s: prepare.read(s)["d"]), 2
    for s in range(1, 10):
        yield "iv2a", s, (lambda s=s: load_subject_data("IV-2a", s)[0]), 4


def fit() -> None:
    import torch

    from src.gpu_health import wait_until_healthy
    from src.pipeline import PipelineConfig, cnn_members_stacked

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    device, cfg = torch.device("cuda"), PipelineConfig(fast=True)
    OOF.mkdir(parents=True, exist_ok=True)
    for cohort, s, get, k in subjects():
        f = OOF / f"{cohort}_s{s}.npz"
        if f.exists():
            continue
        d = get()
        X, y = d.X_cnn[d.train], d.y[d.train]
        A = align_by_group(X, np.zeros(len(y)))
        oof = np.full((len(SPECS), len(y), k), np.nan)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(A, y):
            mask = np.zeros(len(y), bool)
            mask[tr] = True
            wait_until_healthy()
            fold = SubjectData(X_cnn=A, X_tslr=A, y=y, groups=np.where(mask, 0, 1), train=mask)
            probs = cnn_members_stacked(fold, A, k, SPECS, cfg, device)
            for i, p in enumerate(probs):
                oof[i, ~mask] = p
        if not np.isfinite(oof).all():
            raise AssertionError(f"{cohort} s{s}: incomplete out-of-fold predictions")
        np.savez(f, cnn=oof[:3].mean(0), fts=oof[3:].mean(0), y=y)
        acc = {m: float((v.argmax(1) == y).mean()) for m, v in (("cnn", oof[:3].mean(0)), ("fts", oof[3:].mean(0)))}
        print(f"fitted {cohort} s{s}: oof cnn {acc['cnn']:.2f}, fts {acc['fts']:.2f}", flush=True)
    print("FIT DONE", flush=True)


def weights(s1: np.ndarray, y: np.ndarray, kind: str, pseudo: float) -> np.ndarray:
    """s1: (members, trials, classes) session-1 probabilities."""
    like = np.clip(s1[:, np.arange(len(y)), y], 1e-6, 1)
    m = len(s1)
    if kind == "bma":
        score = np.log(like).mean(1)
        w = np.exp(BMA_TEMP * (score - score.max()))
        return w / w.sum()
    w = np.full(m, 1 / m)
    for _ in range(EM_STEPS):
        r = w[:, None] * like
        r /= r.sum(0, keepdims=True)
        w = (r.sum(1) + pseudo / m) / (len(y) + pseudo)
    return w


def main() -> int:
    if "--fit" in sys.argv:
        fit()
        return 0
    import alien

    cohorts = dict(zip(("lee", "iv2a"), alien.build()))
    acc = lambda p, v: 100 * float((p.argmax(1) == v["y_te"]).mean())  # noqa: E731
    res, used = {}, {}
    for name, S in cohorts.items():
        table = {}
        for s, v in S.items():
            z = np.load(OOF / f"{name}_s{s}.npz")
            if not np.array_equal(z["y"], v["y_tr"]):
                raise AssertionError(f"{name} s{s}: out-of-fold labels do not match")
            test = np.stack([v["cnn"], v["own"], v["fts"], v["pool"]])
            s1 = np.stack([z["cnn"], v["own_oof"], z["fts"], v["pool_s1"]])
            row = {"r11": acc(test[:3].mean(0), v), "incumbent": acc(alien.predict(v, ["dawid_skene", "gate_slot"]), v)}
            for arm, (kind, pseudo, m) in ARMS.items():
                w = weights(s1[:m], v["y_tr"], kind, pseudo)
                row[arm] = acc(np.einsum("m,mnc->nc", w, test[:m]), v)
                used.setdefault((name, arm), []).append(np.r_[w, np.zeros(4 - m)])
            table[s] = row
        res[name] = pd.DataFrame(table).T
        res[name].to_csv(HERE / f"gate_all_{name}.csv")
    lee, iv = res["lee"], res["iv2a"]
    for name, want in (("lee", 74.593), ("iv2a", 82.562)):
        if abs(res[name].incumbent.mean() - want) > 5e-3:
            raise AssertionError(f"{name}: incumbent not reproduced ({res[name].incumbent.mean():.3f})")
    passed = []
    for arm in ["r11", "incumbent", *ARMS]:
        print(f"{arm:12s} Lee {lee[arm].mean():.3f}  IV-2a {iv[arm].mean():.3f}")
        if arm in ARMS:
            for ref in ("incumbent", "r11"):
                dl, di = lee[arm] - lee[ref], iv[arm] - iv[ref]
                print(f"    vs {ref:9s} Lee {dl.mean():+.2f} ({(dl > 0).sum()} up/{(dl < 0).sum()} down)  "
                      f"IV-2a {di.mean():+.2f} ({(di > 0).sum()}/{(di < 0).sum()})")
            for name in res:
                print(f"    mean weights {name} [cnn, own, fts, pool]: {np.mean(used[(name, arm)], 0).round(2).tolist()}")
            dl, di = lee[arm] - lee.incumbent, iv[arm] - iv.incumbent
            if dl.mean() >= MIN_GAIN and (dl > 0).sum() > (dl < 0).sum() and di.mean() >= -IV_TOL:
                passed.append(arm)
    winner = max(passed, key=lambda a: lee[a].mean()) if passed else None
    print(f"PASSED: {passed or 'none'}; new incumbent: {winner or 'unchanged (dawid_skene + gate_slot)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
