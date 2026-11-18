"""D4, is the within-session ceiling sample-limited? Learning curves on Lee2019_MI. PRE-REGISTERED, CPU only.

Diagnostic only (no model is deployed or selected). Evaluation labels are used for the curves, as in D3.
Data: the D2 caches (20 channels, 8-30 Hz, 0.5-2.5 s, per-session Riemannian alignment), 54 subjects, 100
trials per session.

CURVE A (within session): for each session, 5-fold stratified CV (shuffle, seed 0); the training fold (80
trials) is subsampled stratified to n in {20, 40, 80} (n = 80 is the whole fold; 4 subsample repeats for
n < 80, rng seeded by subject/session/fold/repeat). Accuracy = mean over sessions, folds, repeats.
CURVE B (pooled): both sessions together (each aligned on itself), 5-fold stratified CV over the 200 trials,
training n in {80, 160} (80 = stratified subsample of the 160-trial fold).
Decoders (reported separately, no selection): TS+LR (src.pipeline.fit_tslr) and CSP+LDA.

STATISTICS
  gain per doubling = least-squares slope of accuracy on log2(n) over n = 20, 40, 80 (curve A), per subject.
  Group means with 95 % bootstrap CI over subjects (10 000 resamples, seed 0) for: all 54 subjects, bottom third,
  middle third, top third (thirds by the D2 csp_lda accuracy, session 1 -> 2).
  Extrapolation (group-mean, optimistic): acc(160 same-session) = acc(80) + slope * 1 (one more doubling);
  reported next to the measured pooled n = 160 accuracy (curve B), which also mixes sessions.
VERDICT RULES fixed in advance, per group and decoder:
  SAMPLE-LIMITED if mean gain per doubling >= 1.5 pp and its CI lower bound > 0;
  SATURATED if the CI upper bound < 1.0 pp; otherwise INCONCLUSIVE.
  A sample-limited bottom third supports data-adding ideas (physics priors, pretraining pool, noise handling);
  a saturated one says weak subjects are signal-limited and only new information (new physiology, different
  data generation) can help.

    .venv\\Scripts\\python experiments\\d4_learning_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from experiments.d3_weak_subject_diagnosis import DECODERS  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.pipeline import align_by_group  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "d2_lee2019"
SIZES_A = (20, 40, 80)
SIZES_B = (80, 160)
REPEATS = 4
N_BOOT = 10_000
SAMPLE_GAIN_PP, SATURATED_PP = 1.5, 1.0


def subsample(y: np.ndarray, idx: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified subsample of n indices from idx (equal per class)."""
    if n >= len(idx):
        return idx
    return np.concatenate([rng.choice(idx[y[idx] == c], n // 2, replace=False) for c in np.unique(y[idx])])


def curve(A: np.ndarray, y: np.ndarray, sizes: tuple[int, ...], seed: int) -> dict[tuple[str, int], float]:
    hits = {(d, n): [] for d in DECODERS for n in sizes}
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(A, y)):
        for n in sizes:
            for rep in range(1 if n >= len(tr) else REPEATS):
                sub = subsample(y, tr, n, np.random.default_rng([seed, fold, n, rep]))
                for d, fp in DECODERS.items():
                    hits[(d, n)].append(float((fp(A[sub], y[sub], A[te]) == y[te]).mean()))
    return {k: float(np.mean(v)) for k, v in hits.items()}


def subject_curves(subject: int) -> dict:
    z = np.load(CACHE / f"s{subject}.npz", allow_pickle=True)
    A = align_by_group(z["X_tslr"].astype(np.float64), z["groups"])
    y, tr = z["y"], z["train"]
    a = [curve(A[m], y[m], SIZES_A, 1000 * subject + s) for s, m in enumerate((tr, ~tr))]
    b = curve(A, y, SIZES_B, 1000 * subject + 7)
    row = {"subject": subject}
    for d in DECODERS:
        for n in SIZES_A:
            row[f"{d}_A{n}"] = float(np.mean([c[(d, n)] for c in a]))
        for n in SIZES_B:
            row[f"{d}_B{n}"] = b[(d, n)]
        acc = np.array([row[f"{d}_A{n}"] for n in SIZES_A])
        row[f"{d}_gain"] = float(np.polyfit(np.log2(SIZES_A), acc, 1)[0])
    return row


def verdict(lo: float, hi: float, mean: float) -> str:
    if mean >= SAMPLE_GAIN_PP / 100 and lo > 0:
        return "SAMPLE-LIMITED"
    return "SATURATED" if hi < SATURATED_PP / 100 else "INCONCLUSIVE"


def main() -> int:
    d2 = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    df = pd.DataFrame([subject_curves(s) for s in d2.index]).set_index("subject")
    df.to_csv(RESULTS_DIR / "d4_learning_curve.csv")
    order = d2.csp_lda.sort_values().index
    k = len(order) // 3
    groups = {"all": order, "bottom third": order[:k], "middle third": order[k:-k], "top third": order[-k:]}
    rng = np.random.default_rng(0)
    print(f"=== D4 learning curves, Lee2019_MI, {len(df)} subjects (thirds by csp_lda D2) ===")
    for d in DECODERS:
        print(f"\n-- decoder {d} --")
        for g, idx in groups.items():
            sub = df.loc[idx]
            gain = sub[f"{d}_gain"].to_numpy()
            boots = gain[rng.integers(0, len(gain), (N_BOOT, len(gain)))].mean(1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            a = [100 * sub[f"{d}_A{n}"].mean() for n in SIZES_A]
            b = [100 * sub[f"{d}_B{n}"].mean() for n in SIZES_B]
            extra = a[-1] + 100 * gain.mean()
            print(f"  {g:13s} A(20,40,80) {a[0]:.1f}/{a[1]:.1f}/{a[2]:.1f} | gain/doubling {100 * gain.mean():+.2f} pp "
                  f"[{100 * lo:+.2f}, {100 * hi:+.2f}] | extrapolated n=160 {extra:.1f} | pooled B(80,160) "
                  f"{b[0]:.1f}/{b[1]:.1f} -> {verdict(lo, hi, gain.mean())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
