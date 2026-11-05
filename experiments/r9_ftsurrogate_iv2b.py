"""R9 CONFIRMATION: R8 FTSurrogate augmentation, setting frozen, on BCI IV-2b.

R8 (docs/diag_gates_d1_d4.md) promoted FTSurrogate on IV-2a: +4.12 pp single-model, 8/9 subjects,
Holm 0.029. This is the confirmatory test on a dataset the method has never seen. Nothing is
tuned. The augmentation is FTSurrogate with p = 0.5 per trial and phase noise 0.9, built by
make_augmentation("ftsurrogate", seed). Everything else is also fixed: ShallowConvNet, TrainConfig
(CFG), band/window (WIDE), and seeds 0-2 (model seed k, augmentation random_state k).

PROTOCOL (fixed before the first run). It uses the same harness as R7:
  Data: BNCI2014_004, all 9 subjects. Train on recordings 0train+1train+2train. Evaluate on
    3test and 4test. Riemannian alignment is fitted per RECORDING, on that recording only.
  Baseline: ShallowConvNet(3, 2, 1001) without augmentation, seeds 0-2. These are R7's cached
    teacher predictions (data/cache/r7_preds/s{s}_teacher_seed{k}.npy).
  Arm: the same model and seeds, trained with FTSurrogate. E is never seen in training.
  Subject score = mean accuracy over the two evaluation recordings.

HYPOTHESIS (one, so no multiplicity correction). The primary metric matches R8's:
  H1: FTSurrogate 3-seed single-model mean accuracy > baseline 3-seed single-model mean accuracy.
  CONFIRMED iff the subject-mean difference > 0 AND the one-sided exact sign-flip p < 0.05 (n = 9).
  Reported regardless: per-subject differences, 95% t-interval, 3-seed ensemble contrast, and the
  late-minus-early contrast (4test minus 3test).

VALIDITY. The run is UNINFORMATIVE if any check fails. A failure is reported, never re-run with changes.
  V1 For every subject, seed-0 baseline retrained here reproduces R7 teacher seed 0 exactly (argmax).
  V2 Baseline 3-seed single-model mean is >= 60%.
  V3 Every alignment self-check passes (asserted in align_recordings).
  V4 The augmentation unit tests pass (tests/test_augmentation.py).

    .venv\\Scripts\\python experiments\\r9_ftsurrogate_iv2b.py     # ~25 min on the RTX 4060
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
from scipy import stats  # noqa: E402

from experiments.r3_convnet_session_norm import CFG, WIDE  # noqa: E402
from experiments.r7_self_training_iv2b import (  # noqa: E402
    CLASSES, EVAL_RECORDINGS, N_TIMES, SUBJECTS, TRAIN_RECORDINGS, align_recordings, cached,
)
from src.adaptation.augmentation import make_augmentation  # noqa: E402
from src.adaptation.self_training import one_hot  # noqa: E402
from src.adaptation.convnet import ShallowConvNet, encode_labels, predict_proba, set_deterministic, train_model  # noqa: E402
from src.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.datasets import load_iv2b_subject_window  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402

OUT_CSV = RESULTS_DIR / "r9_ftsurrogate_iv2b.csv"
R7_PRED_DIR = DATA_DIR / "cache" / "r7_preds"
PRED_DIR = DATA_DIR / "cache" / "r9_preds"
SEEDS = (0, 1, 2)
AUG = "ftsurrogate"


def fit_predict(Xt: np.ndarray, Yt: np.ndarray, Xe: np.ndarray, seed: int, augmented: bool, device) -> np.ndarray:
    """Train on one-hot targets Yt, exactly as R7's teacher (so V1 can demand bit-exact reproduction)."""
    wait_until_healthy()
    set_deterministic(seed)
    augment = make_augmentation(AUG, seed) if augmented else None
    model = train_model(ShallowConvNet(3, 2, N_TIMES).to(device),
                        torch.as_tensor(Xt.astype(np.float32), device=device),
                        torch.as_tensor(Yt.astype(np.float32), device=device), CFG, seed=seed, augment=augment)
    probs = predict_proba(model, torch.as_tensor(Xe.astype(np.float32), device=device))
    del model
    torch.cuda.empty_cache()
    return probs


def run_subject(s: int, device, rows: list[dict]) -> None:
    d = load_iv2b_subject_window(s, **WIDE)
    if d.X.shape[1:] != (3, N_TIMES):
        raise AssertionError(f"unexpected shape {d.X.shape}")
    X = align_recordings(d.X, d.session)
    y = encode_labels(d.y, CLASSES)
    tr = np.isin(d.session, TRAIN_RECORDINGS)
    Xt, Yt = X[tr], one_hot(y[tr], 2)
    ev_idx = {rec: np.flatnonzero(d.session == rec) for rec in EVAL_RECORDINGS}
    order = np.concatenate(list(ev_idx.values()))
    X_eval_all, y_eval_all = X[order], y[order]

    base = [np.load(R7_PRED_DIR / f"s{s}_teacher_seed{k}.npy") for k in SEEDS]
    again = fit_predict(Xt, Yt, X_eval_all, 0, False, device)
    if not np.array_equal(again.argmax(1), base[0].argmax(1)):
        raise AssertionError(f"V1: s{s} seed-0 baseline does not reproduce R7 teacher seed 0")
    aug = [cached(PRED_DIR / f"s{s}_{AUG}_seed{k}.npy", lambda k=k: fit_predict(Xt, Yt, X_eval_all, k, True, device))
           for k in SEEDS]

    offset = 0
    for rec in EVAL_RECORDINGS:
        sl = slice(offset, offset + len(ev_idx[rec]))
        offset += len(ev_idx[rec])
        ye = y_eval_all[sl]
        for arm, ps in (("baseline", base), (AUG, aug)):
            ps = [p[sl] for p in ps]
            for metric, value in (("single_mean_acc", np.mean([(p.argmax(1) == ye).mean() for p in ps])),
                                  ("ens_acc", (np.mean(ps, 0).argmax(1) == ye).mean())):
                rows.append({"subject": s, "recording": rec, "arm": arm, "metric": metric, "value": float(value)})
    q = pd.DataFrame([r for r in rows if r["subject"] == s and r["metric"] == "single_mean_acc"])
    m = q.groupby("arm").value.mean()
    print(f"  s{s}: baseline {100*m['baseline']:.1f}%  {AUG} {100*m[AUG]:.1f}%", flush=True)


def summarise(df: pd.DataFrame) -> None:
    def per_subject(arm: str, metric: str, rec: str | None = None) -> pd.Series:
        q = df[(df.arm == arm) & (df.metric == metric)]
        if rec is not None:
            q = q[q.recording == rec]
        return 100 * q.groupby("subject").value.mean()

    b, a = per_subject("baseline", "single_mean_acc"), per_subject(AUG, "single_mean_acc")
    d = a - b
    p = sign_flip_permutation_test(d.to_numpy(), "greater")
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    de = per_subject(AUG, "ens_acc") - per_subject("baseline", "ens_acc")
    print("\n=== R9 confirmation: FTSurrogate on IV-2b (9 subjects, mean of 3test/4test, 3 seeds) ===")
    print("per subject (ftsurrogate - baseline, single-model pp): " + ", ".join(f"s{k} {v:+.1f}" for k, v in d.items()))
    print(f"baseline single {b.mean():.2f}%   ftsurrogate single {a.mean():.2f}%")
    print(f"difference {d.mean():+.2f} pp, 95% CI [{d.mean()-half:+.2f}, {d.mean()+half:+.2f}], "
          f"{(d > 0).sum()}/9 up, one-sided sign-flip p = {p:.4f}")
    print(f"ensemble: baseline {per_subject('baseline', 'ens_acc').mean():.2f}%, "
          f"ftsurrogate {per_subject(AUG, 'ens_acc').mean():.2f}% ({de.mean():+.2f} pp)")
    for arm in ("baseline", AUG):
        gap = per_subject(arm, "single_mean_acc", EVAL_RECORDINGS[1]) - per_subject(arm, "single_mean_acc", EVAL_RECORDINGS[0])
        print(f"late-minus-early ({EVAL_RECORDINGS[1]} - {EVAL_RECORDINGS[0]}) {arm}: {gap.mean():+.2f} pp")
    v2 = b.mean() >= 60.0
    print(f"V1, V3: PASS (asserted);  V2 baseline >= 60%: {'PASS' if v2 else 'FAIL'}")
    if not v2:
        print("VERDICT: UNINFORMATIVE (validity failed)")
    else:
        print(f"VERDICT: H1 {'CONFIRMED' if d.mean() > 0 and p < 0.05 else 'NOT CONFIRMED'}")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in SUBJECTS:
        print(f"=== subject {s} ===", flush=True)
        run_subject(s, device, rows)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    summarise(pd.DataFrame(rows))
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
