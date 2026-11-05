"""R6 screen: does the unlabelled evaluation session help through the model's own predictions?

D4a (docs/diag_gates_d1_d4.md) left this family open: the 5-seed ensemble's most confident
half of E is ~90% correct (~78% in WEAK subjects). Teacher = R3 `cnn_ra` (ShallowConvNet on
per-session RA input), 5 seeds, retrained here and checked against the cached R3 predictions.
E labels are used ONLY for scoring; every arm receives (T data, T labels, E data, teacher probs).

ARMS (pre-declared, one setting each, no tuning; 5 seeds each, seed k student starts from seed k):
  teacher   cnn_ra as in R3
  st        pseudo-label self-training: retrain on T + the confident 50% of E (per predicted
            class, by 5-seed ensemble max-probability) with hard ensemble pseudo-labels
  distill   ensemble distillation: retrain on T + all E with the ensemble's soft probabilities
  im        IM-TTA (SHOT-IM / Tent): teacher seed k, BN affine params only, full-batch E,
            100 Adam steps, lr 1e-3, then BN statistics of E
Same TrainConfig as R3 (300 epochs; more trials -> more steps per epoch).

RULES — fixed before the first run:
  Primary contrast per arm: 5-seed ENSEMBLE accuracy of the arm minus the teacher's 5-seed
  ensemble accuracy (ensembling alone must not count as a gain).
  G (screen): arm OPEN iff that contrast >= +1.0 pp subject mean, two-sided sign-flip p < 0.10.
  ADOPT (for a confirmatory step only): Holm-adjusted one-sided p < 0.05 over the 3 arms, mean
     > 0, and the WEAK group {2, 5, 6} mean contrast >= 0.
  Secondary (reported): single-model mean of each arm vs the teacher ensemble (can one model
     match the ensemble?); accuracy on the teacher's UNconfident half of E (where any real gain
     must come from - the confident half is already labelled as the teacher predicts).
  VALIDITY (screen INVALID if any fails):
     V1 retrained teacher reproduces the cached R3 cnn_ra probabilities (argmax identical) for
        every subject and seed.
     V2 shuffled-pseudo-label control (subjects 1, 3, 7, seed 0): `st` with the selected
        pseudo-labels randomly permuted must NOT beat the teacher seed-0 model by >= 1 pp
        (catches any path by which true E labels could reach training).

    .venv\\Scripts\\python experiments\\r6_self_training.py     # ~90 min on the RTX 4060
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiments.r3_convnet_session_norm import CFG, N_TIMES_WIDE, WIDE, align_per_session  # noqa: E402
from src.adaptation.convnet import ShallowConvNet, encode_labels, predict_proba, set_deterministic, train_model  # noqa: E402
from src.adaptation.self_training import im_tta, one_hot, select_confident  # noqa: E402
from src.config import DATA_DIR, IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR  # noqa: E402
from src.datasets import load_iv2a_subject_window  # noqa: E402
from src.evaluate import holm_adjust, sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402

OUT_CSV = RESULTS_DIR / "r6_self_training.csv"
R3_PRED_DIR = DATA_DIR / "cache" / "r3_preds"
PRED_DIR = DATA_DIR / "cache" / "r6_preds"
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("st", "distill", "im")
CONF_FRAC = 0.5
IM_STEPS, IM_LR = 100, 1e-3
CONTROL_SUBJECTS = (1, 3, 7)
WEAK = (2, 5, 6)
N_CLASSES = len(IV2A_CLASSES)
GATE_PP = 1.0


class Session:
    """Tensors of one subject on the GPU. Holds NO evaluation labels."""

    def __init__(self, Xc: np.ndarray, yc: np.ndarray, Xe: np.ndarray, device) -> None:
        self.device = device
        self.Xc = torch.as_tensor(Xc.astype(np.float32), device=device)
        self.Xe = torch.as_tensor(Xe.astype(np.float32), device=device)
        self.yc = yc
        self.yc_soft = one_hot(yc, N_CLASSES)

    def train(self, X: torch.Tensor, y: torch.Tensor, seed: int) -> ShallowConvNet:
        wait_until_healthy()
        set_deterministic(seed)
        return train_model(ShallowConvNet(22, N_CLASSES, N_TIMES_WIDE).to(self.device), X, y, CFG, seed=seed)

    def train_with_e(self, e_idx: np.ndarray, e_targets: np.ndarray, seed: int) -> np.ndarray:
        X = torch.cat([self.Xc, self.Xe[torch.as_tensor(e_idx, device=self.device)]])
        y = torch.as_tensor(np.concatenate([self.yc_soft, e_targets.astype(np.float32)]), device=self.device)
        model = self.train(X, y, seed)
        probs = predict_proba(model, self.Xe)
        del model
        torch.cuda.empty_cache()
        return probs


def cached(path: Path, compute) -> np.ndarray:
    if path.exists():
        return np.load(path)
    probs = compute()
    np.save(path, probs.astype(np.float32))
    return probs


def run_subject(s: int, device) -> tuple[dict[str, list[np.ndarray]], np.ndarray, dict[int, float]]:
    """All arms for one subject. Returns probs per arm per seed, the confident index and V2 accs.

    Evaluation labels are not loaded here except to verify V1 against the cached R3 preds
    (compared on predictions, not labels) - scoring happens in the caller.
    """
    wide = load_iv2a_subject_window(s, **WIDE)
    Xc, Xe = align_per_session(wide.X_calib, wide.X_eval)
    sess = Session(Xc, encode_labels(wide.y_calib, IV2A_CLASSES), Xe, device)
    probs: dict[str, list[np.ndarray]] = {"teacher": [], **{a: [] for a in ARMS}}
    models = {}
    for seed in SEEDS:
        model = sess.train(sess.Xc, torch.as_tensor(sess.yc, device=device), seed)
        p = predict_proba(model, sess.Xe)
        ref = np.load(R3_PRED_DIR / f"s{s}_cnn_ra_seed{seed}.npy")
        if not np.array_equal(p.argmax(1), ref.argmax(1)):
            raise AssertionError(f"V1: s{s} seed{seed} teacher does not reproduce R3 cnn_ra")
        probs["teacher"].append(p)
        models[seed] = model
    ens = np.mean(probs["teacher"], axis=0)
    conf_idx = select_confident(ens, CONF_FRAC)
    pseudo = one_hot(ens.argmax(1)[conf_idx], N_CLASSES)
    for seed in SEEDS:
        tag = f"s{s}_seed{seed}"
        probs["st"].append(cached(PRED_DIR / f"{tag}_st.npy", lambda: sess.train_with_e(conf_idx, pseudo, seed)))
        probs["distill"].append(cached(PRED_DIR / f"{tag}_distill.npy",
                                       lambda: sess.train_with_e(np.arange(len(ens)), ens, seed)))
        probs["im"].append(cached(PRED_DIR / f"{tag}_im.npy",
                                  lambda: predict_proba(im_tta(models[seed], sess.Xe, IM_STEPS, IM_LR), sess.Xe)))
        print(f"  s{s} seed{seed} done", flush=True)
    v2 = {}
    if s in CONTROL_SUBJECTS:
        rng = np.random.default_rng(0)
        shuffled = pseudo[rng.permutation(len(pseudo))]
        v2[0] = sess.train_with_e(conf_idx, shuffled, 0)
    del models
    torch.cuda.empty_cache()
    return probs, conf_idx, v2


def score(s: int, ye: np.ndarray, probs, conf_idx, v2, rows: list[dict]) -> None:
    unconf = np.setdiff1d(np.arange(len(ye)), conf_idx)

    def add(arm, metric, value):
        rows.append({"subject": s, "arm": arm, "metric": metric, "value": float(value)})

    add("teacher", "confident_half_acc", (np.mean(probs["teacher"], 0).argmax(1)[conf_idx] == ye[conf_idx]).mean())
    for arm, ps in probs.items():
        ens = np.mean(ps, axis=0).argmax(1)
        add(arm, "ens_acc", (ens == ye).mean())
        add(arm, "single_mean_acc", np.mean([(p.argmax(1) == ye).mean() for p in ps]))
        add(arm, "ens_unconf_acc", (ens[unconf] == ye[unconf]).mean())
    if v2:
        add("st_shuffled", "seed0_acc", (v2[0].argmax(1) == ye).mean())
        add("teacher", "seed0_acc", (probs["teacher"][0].argmax(1) == ye).mean())


def summarise(df: pd.DataFrame) -> None:
    def ps(arm: str, metric: str) -> pd.Series:
        q = df[(df.arm == arm) & (df.metric == metric)]
        return 100 * q.set_index("subject").value.sort_index()

    t_ens = ps("teacher", "ens_acc")
    print("\n=== R6 self-training screen (ShallowConvNet, 9 subjects) ===")
    print(f"teacher: single-model mean {ps('teacher', 'single_mean_acc').mean():.2f}%, "
          f"ensemble {t_ens.mean():.2f}%, unconfident half {ps('teacher', 'ens_unconf_acc').mean():.2f}%, "
          f"confident half {ps('teacher', 'confident_half_acc').mean():.2f}%")
    one_sided = {}
    for arm in ARMS:
        d = ps(arm, "ens_acc") - t_ens
        p2 = sign_flip_permutation_test(d.to_numpy(), "two-sided")
        one_sided[arm] = sign_flip_permutation_test(d.to_numpy(), "greater")
        gate = d.mean() >= GATE_PP and p2 < 0.10
        weak = d.loc[list(WEAK)].mean()
        single = ps(arm, "single_mean_acc") - t_ens
        unconf = ps(arm, "ens_unconf_acc") - ps("teacher", "ens_unconf_acc")
        print(f"  {arm:8s} ens {ps(arm, 'ens_acc').mean():6.2f}%  {d.mean():+5.2f}pp  {(d > 0).sum()}/9 up  "
              f"p={p2:.3f}  WEAK {weak:+.2f}pp  -> {'OPEN' if gate else 'closed'}")
        print(f"           single-model mean vs teacher ensemble {single.mean():+.2f}pp; "
              f"unconfident half {unconf.mean():+.2f}pp")
    adj = dict(zip(one_sided, holm_adjust(list(one_sided.values()))))
    for arm in ARMS:
        d = ps(arm, "ens_acc") - t_ens
        adopt = adj[arm] < 0.05 and d.mean() > 0 and d.loc[list(WEAK)].mean() >= 0
        print(f"  Holm one-sided {arm}: p_adj={adj[arm]:.3f}  ADOPT {adopt}")
    sh, t0 = ps("st_shuffled", "seed0_acc"), ps("teacher", "seed0_acc")
    if len(sh):
        diff = sh - t0
        ok = bool((diff < 1.0).all())
        print(f"V2 shuffled pseudo-labels vs teacher seed 0: {', '.join(f's{k} {v:+.1f}pp' for k, v in diff.items())}"
              f" -> {'PASS' if ok else 'FAIL - screen INVALID'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(IV2A_SUBJECTS))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in args.subjects:
        print(f"=== subject {s} ===", flush=True)
        probs, conf_idx, v2 = run_subject(s, device)
        ye = encode_labels(load_iv2a_subject_window(s, **WIDE).y_eval, IV2A_CLASSES)
        score(s, ye, probs, conf_idx, v2, rows)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    if len(args.subjects) == len(IV2A_SUBJECTS):
        summarise(pd.DataFrame(rows))
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
