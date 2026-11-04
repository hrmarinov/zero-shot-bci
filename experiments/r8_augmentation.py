"""R8 screen: training-time augmentation priors for ShallowConvNet on per-session RA input.

Catalogue item 1 (reports/Cross field BCI calibration methods.md): the only cross-field import
with published gains on this exact protocol (Rommel et al. 2022, IV-2a session 1 -> 2). Settings
are fixed a priori from that paper (src/adaptation/augmentation.py), never tuned on E.

ARMS: baseline = R3 cnn_ra (cached predictions; seed 0 retrained here as a reproduction check);
  ftsurrogate, freqshift, timemask, timereverse, chandrop - each alone, p = 0.5 per trial,
  same TrainConfig (300 epochs), seeds 0-2 (model seed k, augmentation random_state k).
  E is never seen in training; inputs are RA'd per session exactly as in R3.

RULES — fixed before the first run:
  Primary per arm: 3-seed single-model mean accuracy minus R3 cnn_ra seeds 0-2 mean (seed-matched).
  OPEN (screen pass) iff mean >= +1.0 pp, two-sided sign-flip p < 0.10, and fewer than 3 subjects
     lose > 2 pp. KILLED otherwise.
  PROMOTE (candidate for one stacked, pre-registered confirmatory run on a second dataset)
     iff OPEN and mean >= +2.0 pp.
  Reported: Holm-adjusted one-sided p over the 5 arms; 3-seed ensemble contrast; WEAK {2, 5, 6}.
  VALIDITY: V1 seed-0 baseline retrained here reproduces R3 cnn_ra seed 0 per subject exactly;
     V2 each augmentation is seed-reproducible, touches ~50% of trials and does what it claims
     (tests/test_augmentation.py); train_model with augment=None is bit-identical (test_convnet.py).

    .venv\\Scripts\\python experiments\\r8_augmentation.py     # ~75 min on the RTX 4060
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiments.r3_convnet_session_norm import CFG, N_TIMES_WIDE, WIDE, align_per_session  # noqa: E402
from src.adaptation.augmentation import AUGMENTATIONS, make_augmentation  # noqa: E402
from src.adaptation.convnet import ShallowConvNet, encode_labels, predict_proba, set_deterministic, train_model  # noqa: E402
from src.config import DATA_DIR, IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR  # noqa: E402
from src.datasets import load_iv2a_subject_window  # noqa: E402
from src.evaluate import holm_adjust, sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402

OUT_CSV = RESULTS_DIR / "r8_augmentation.csv"
R3_PRED_DIR = DATA_DIR / "cache" / "r3_preds"
PRED_DIR = DATA_DIR / "cache" / "r8_preds"
SEEDS = (0, 1, 2)
WEAK = (2, 5, 6)
GATE_PP, PROMOTE_PP, LOSS_PP, MAX_LOSERS = 1.0, 2.0, 2.0, 3


def fit_predict(Xc: torch.Tensor, yc: torch.Tensor, Xe: torch.Tensor, seed: int, aug: str | None) -> np.ndarray:
    wait_until_healthy()
    set_deterministic(seed)
    augment = None if aug is None else make_augmentation(aug, seed)
    model = train_model(ShallowConvNet(22, 4, N_TIMES_WIDE).to(Xc.device), Xc, yc, CFG, seed=seed, augment=augment)
    probs = predict_proba(model, Xe)
    del model
    torch.cuda.empty_cache()
    return probs


def run_subject(s: int, device, rows: list[dict]) -> None:
    wide = load_iv2a_subject_window(s, **WIDE)
    Xc, Xe = align_per_session(wide.X_calib, wide.X_eval)
    yc, ye = encode_labels(wide.y_calib, IV2A_CLASSES), encode_labels(wide.y_eval, IV2A_CLASSES)
    Xc_t = torch.as_tensor(Xc.astype(np.float32), device=device)
    Xe_t = torch.as_tensor(Xe.astype(np.float32), device=device)
    yc_t = torch.as_tensor(yc, device=device)

    base = [np.load(R3_PRED_DIR / f"s{s}_cnn_ra_seed{k}.npy") for k in SEEDS]
    if not np.array_equal(fit_predict(Xc_t, yc_t, Xe_t, 0, None).argmax(1), base[0].argmax(1)):
        raise AssertionError(f"V1: s{s} seed-0 baseline does not reproduce R3 cnn_ra")
    probs = {"baseline": base}
    for aug in AUGMENTATIONS:
        probs[aug] = []
        for k in SEEDS:
            path = PRED_DIR / f"s{s}_{aug}_seed{k}.npy"
            if path.exists():
                p = np.load(path)
            else:
                p = fit_predict(Xc_t, yc_t, Xe_t, k, aug)
                np.save(path, p.astype(np.float32))
            probs[aug].append(p)
    for arm, ps in probs.items():
        single = np.mean([(p.argmax(1) == ye).mean() for p in ps])
        ens = (np.mean(ps, 0).argmax(1) == ye).mean()
        rows += [{"subject": s, "arm": arm, "metric": "single_mean_acc", "value": float(single)},
                 {"subject": s, "arm": arm, "metric": "ens_acc", "value": float(ens)}]
        print(f"  s{s} {arm:12s} single {100*single:.1f}%  ens {100*ens:.1f}%", flush=True)


def summarise(df: pd.DataFrame) -> None:
    def ps(arm: str, metric: str) -> pd.Series:
        q = df[(df.arm == arm) & (df.metric == metric)]
        return 100 * q.set_index("subject").value.sort_index()

    base, base_ens = ps("baseline", "single_mean_acc"), ps("baseline", "ens_acc")
    print(f"\n=== R8 augmentation screen (ShallowConvNet + RA, 3 seeds) ===\n"
          f"baseline: single {base.mean():.2f}%, 3-seed ensemble {base_ens.mean():.2f}%")
    one_sided = {}
    for arm in AUGMENTATIONS:
        d = ps(arm, "single_mean_acc") - base
        p2 = sign_flip_permutation_test(d.to_numpy(), "two-sided")
        one_sided[arm] = sign_flip_permutation_test(d.to_numpy(), "greater")
        losers = int((d < -LOSS_PP).sum())
        is_open = d.mean() >= GATE_PP and p2 < 0.10 and losers < MAX_LOSERS
        verdict = ("PROMOTE" if d.mean() >= PROMOTE_PP else "OPEN") if is_open else "killed"
        de = ps(arm, "ens_acc") - base_ens
        print(f"  {arm:12s} {ps(arm, 'single_mean_acc').mean():6.2f}%  {d.mean():+5.2f}pp  {(d > 0).sum()}/9 up  "
              f"p={p2:.3f}  losers>2pp {losers}  WEAK {d.loc[list(WEAK)].mean():+.2f}pp  "
              f"ens {de.mean():+.2f}pp  -> {verdict}")
        print("      per subject: " + ", ".join(f"s{k} {v:+.1f}" for k, v in d.items()))
    adj = holm_adjust(list(one_sided.values()))
    print("  Holm one-sided: " + ", ".join(f"{a} {p:.3f}" for a, p in zip(one_sided, adj)))


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in IV2A_SUBJECTS:
        print(f"=== subject {s} ===", flush=True)
        run_subject(s, device, rows)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    summarise(pd.DataFrame(rows))
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
