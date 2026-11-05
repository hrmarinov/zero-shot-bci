"""D6, diagnostic: why does our single ShallowConvNet trail the published one on IV-2b? PRE-REGISTERED.

Fact (docs/state_of_the_art.md §0): under the same split (recordings 0-2train -> 3test + 4test), CTNet's
leak-free reimplementation of ShallowConvNet scores 85.13% on IV-2b; ours (R7 teacher, seeds 0-2) scores
78.58%. That ~6.5 pp gap sits below any ensembling, so it is a pipeline difference, not an architecture one.

Published recipe (EEG-Conformer / CTNet data path): the 0-4 s post-cue window of the competition's
0.5-100 Hz recordings with no further band-pass, a global z-score from training-session statistics, no
alignment, and 1000-2000 epochs (+ segment-and-recombine augmentation, NOT tested here).

CELLS (IV-2b only, ShallowConvNet(3, 2, 1001), seeds 0-2, R7 one-hot training path, 0-4 s window):
  base        per-recording RA, 4-38 Hz, 300 epochs          = R7 teacher (cached; reproduced as V2)
  no_ra       global train z-score, 4-38 Hz, 300 epochs      (alignment only)
  broadband   per-recording RA, 0.5-100 Hz, 300 epochs       (band only)
  ep1000      per-recording RA, 4-38 Hz, 1000 epochs         (training budget only)
  recipe      global train z-score, 0.5-100 Hz, 1000 epochs  (the published data path)
  recipe_eog  recipe on EOG-regressed EEG (R14's regression, coefficients fitted on training recordings)

QUESTIONS (descriptive; no gate, no new best is selected here):
  Q1 Which single change closes how much of the gap? (cell - base, per subject, sign-flip p, 9 subjects)
  Q2 Does the published data path reproduce ~85%? (recipe vs 85.13)
  Q3 Is whatever broadband adds ocular? R11b/R14: < 4 Hz on IV-2a/2b is class-locked eye movement and
     IV-2b's E recordings are feedback sessions. recipe - recipe_eog > 1 pp would mean the published
     IV-2b numbers partly decode eye movement.
Any change adopted from this diagnostic must be re-screened jointly on IV-2a + IV-2b and confirmed on
the lockbox; nothing here is a selection.

VALIDITY:
  V1 the 4-38 Hz EEG cut here equals the harness loader's array exactly (as R14's V1).
  V2 the base cell retrained here equals the cached R7 teacher predictions (argmax equal, |dp| < 1e-5)
     for subject 1, seed 0 (the other base predictions are read from the R7 cache).
  V3 determinism: a warm rerun gives a byte-identical CSV.

    .venv\\Scripts\\python experiments\\d6_iv2b_gap.py     # ~1 h GPU (136 fits), resumable
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from moabb.datasets import BNCI2014_004  # noqa: E402

from experiments.r7_self_training_iv2b import TRAIN_RECORDINGS  # noqa: E402
from experiments.r11_round3_diverse_members import CACHE, rec_acc  # noqa: E402
from experiments.r14_eog_gate import check_v1, eog_regress, raw_epochs  # noqa: E402
from src.adaptation.convnet import TrainConfig, encode_labels  # noqa: E402
from src.config import IV2A_SUBJECTS, RESULTS_DIR  # noqa: E402
from src.datasets import load_iv2b_subject_window  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402
from src.pipeline import CNN_BAND, CNN_TRAIN, IV2B_CLASSES, PipelineConfig, SubjectData, align_by_group, cnn_member  # noqa: E402

OUT_CSV = RESULTS_DIR / "d6_iv2b_gap.csv"
PRED_DIR = CACHE / "d6_preds"
SEEDS = (0, 1, 2)
BANDS = {"cnn": CNN_BAND, "broad": dict(fmin=0.5, fmax=100.0, tmin=0.0, tmax=4.0)}
LONG = dataclasses.replace(CNN_TRAIN, epochs=1000)
# cell: (band, normalisation, training config, EOG-regressed)
CELLS = {
    "no_ra": ("cnn", "zscore", CNN_TRAIN, False),
    "broadband": ("broad", "ra", CNN_TRAIN, False),
    "ep1000": ("cnn", "ra", LONG, False),
    "recipe": ("broad", "zscore", LONG, False),
    "recipe_eog": ("broad", "zscore", LONG, True),
}
PUBLISHED_SHALLOW = 85.13
TOL_V2 = 1e-5


def zscore(X: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Global z-score with training-session statistics (the EEG-Conformer / CTNet data path)."""
    return (X - X[train].mean()) / X[train].std()


def predict(tag: str, data: SubjectData, A: np.ndarray, cfg: TrainConfig, seed: int, device) -> np.ndarray:
    path = PRED_DIR / f"{tag}_seed{seed}.npy"
    if path.exists():
        return np.load(path)
    wait_until_healthy()
    p = cnn_member(data, A, 2, seed, None, PipelineConfig(n_cnn=1, n_fts=0, one_hot_targets=True, train=cfg), device)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, p)
    return p


def run(s: int, rows: list, device) -> None:
    arr, ys, sess = raw_epochs(BNCI2014_004(), s, BANDS)
    h = load_iv2b_subject_window(s, **CNN_BAND)
    if not (np.array_equal(ys, h.y) and np.array_equal(sess, h.session)):
        raise AssertionError(f"V1: IV-2b s{s} label/recording order differs")
    check_v1(f"IV-2b s{s} 4-38 Hz", h.X, arr["cnn"]["eeg"])
    y = encode_labels(ys, IV2B_CLASSES)
    train = np.isin(sess, TRAIN_RECORDINGS)
    data = SubjectData(X_cnn=arr["cnn"]["eeg"], X_tslr=arr["cnn"]["eeg"], y=y, groups=sess, train=train)
    ye, rec = y[~train], sess[~train]

    base = [np.load(CACHE / "r7_preds" / f"s{s}_teacher_seed{k}.npy") for k in SEEDS]
    if s == 1:
        p = predict(f"s{s}_base_check", data, align_by_group(arr["cnn"]["eeg"], sess), CNN_TRAIN, 0, device)
        if not (np.array_equal(p.argmax(1), base[0].argmax(1)) and np.abs(p - base[0]).max() < TOL_V2):
            raise AssertionError(f"V2: base retrain differs from the R7 cache ({np.abs(p - base[0]).max():.2e})")
    preds = {"base": base}
    for cell, (band, norm, cfg, eog) in CELLS.items():
        X = arr[band]["eeg"]
        if eog:
            X, _ = eog_regress(X, arr[band]["eog"], train)
        A = align_by_group(X, sess) if norm == "ra" else zscore(X, train)
        preds[cell] = [predict(f"s{s}_{cell}", data, A, cfg, k, device) for k in SEEDS]
    for cell, ps in preds.items():
        rows.append({"subject": s, "cell": cell, "single": float(np.mean([rec_acc(p, ye, rec) for p in ps])),
                     "ens3": rec_acc(np.mean(ps, 0), ye, rec)})


def summarise(df: pd.DataFrame) -> None:
    for metric in ("single", "ens3"):
        piv = 100 * df.pivot_table(index="subject", columns="cell", values=metric)
        print(f"\n=== D6 IV-2b gap, {metric} (mean over 3 seeds)" if metric == "single" else "\n=== 3-seed ensemble")
        print(piv.mean().round(2).to_string())
        for cell in [c for c in piv.columns if c != "base"]:
            d = piv[cell] - piv["base"]
            print(f"  {cell:11s} - base: {d.mean():+.2f} pp, {(d > 0).sum()}/9 up, {(d < 0).sum()} down, "
                  f"p = {sign_flip_permutation_test(d.to_numpy(), 'greater'):.4f}")
        d = piv["recipe"] - piv["recipe_eog"]
        print(f"  Q3 recipe - recipe_eog (ocular share): {d.mean():+.2f} pp, {(d > 0).sum()}/9 up, "
              f"p = {sign_flip_permutation_test(d.to_numpy(), 'greater'):.4f}")
    print(f"\nQ2: published ShallowConvNet {PUBLISHED_SHALLOW:.2f}; recipe single "
          f"{100 * df[df.cell == 'recipe'].single.mean():.2f}")


def main() -> int:
    device = torch.device("cuda")
    rows: list[dict] = []
    for s in IV2A_SUBJECTS:
        run(s, rows, device)
        print(f"  s{s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("V1, V2 passed")
    summarise(df)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
