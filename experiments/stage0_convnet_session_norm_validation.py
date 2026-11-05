"""Stage 0 gate for src/adaptation/convnet.py: does per-session BN normalisation recover a
*planted, known* session drift — and does a deliberately broken version fail to?

Synthetic EEG (22 ch, 250 Hz, 2 s, 4 classes, 288 trials/session in 6 runs of 48):
four narrowband task sources (10, 12, 20, 24 Hz); class k desynchronises source k
(amplitude x ERD_FACTOR) — an ERD-like, power-coded class signal; 18 broadband background
sources; one random mixing matrix A shared by both sessions; white sensor noise.

Drift conditions applied to the evaluation session E (T is never drifted):
  none        no drift                                   -> adaptation must do no harm
  band_gain   mu-band sources x G_MU, beta-band x G_BETA  -> the per-band gain the method
              (a multiplicative, band-specific gain, i.e.    targets; one broadband whitening
              an additive per-band log-power offset)         cannot express it
  run_drift   band_gain + an extra random per-run band gain in BOTH sessions
              (slow within-session drift)                 -> what run-wise (SAT) BN targets
  prior_shift band_gain + E class proportions 70/10/10/10 -> known AdaBN weakness; REPORTED only

Arms (identical trained weights unless noted):
  none          standard training, BN running stats from T  (the unadapted model)
  adabn         + AdaBN on all E trials (label-free)
  mutant_tstats "AdaBN" that recomputes stats on T instead of E — must NOT help
  sat           run-wise BN training + run-wise AdaBN at test (separately trained)

PASS criteria (fixed before the first run; the gate exits non-zero on any failure):
  G1  clean decodability: `none` on drift=none in [0.70, 0.98] for every seed
      (SNR is the only knob allowed to be adjusted to satisfy G1, never the drift).
  G2  no harm: on drift=none, adabn and sat within 3pp of none (seed mean).
  G3  planted drift is real: none drops >= 15pp from drift=none to band_gain (seed mean).
  G4  recovery: adabn recovers >= 70% of that drop on band_gain (seed mean).
  G5  mutation: mutant_tstats recovers <= 10% of the drop (a no-op must look like a no-op).
  G6  sat recovers >= 70% of the run_drift drop (seed mean).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import copy  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from src.adaptation.convnet import (  # noqa: E402
    ShallowConvNet,
    TrainConfig,
    adapt_bn_statistics,
    predict_proba,
    predict_proba_groupwise,
    set_deterministic,
    train_model,
)
from src.config import RESULTS_DIR  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402

SFREQ, N_TIMES, N_CHANS, N_CLASSES = 250, 500, 22, 4
N_RUNS, PER_RUN = 6, 48
TASK_FREQS = (10.0, 12.0, 20.0, 24.0)  # first two "mu", last two "beta"
ERD_FACTOR = 0.4
N_BACKGROUND = N_CHANS - len(TASK_FREQS)
BACKGROUND_AMP = 1.0
SENSOR_NOISE = 0.5
G_MU, G_BETA = 2.5, 0.5
RUN_GAIN_RANGE = (0.5, 2.0)
SEEDS = (0, 1, 2)
CFG = TrainConfig(epochs=150, batch_size=48, lr=1e-3, weight_decay=1e-3)
OUT_CSV = RESULTS_DIR / "stage0_convnet_session_norm.csv"


def _bandlimited(rng: np.random.Generator, n: int, f: float) -> np.ndarray:
    sos = butter(4, [f - 1.0, f + 1.0], btype="band", fs=SFREQ, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal((n, N_TIMES + 200)), axis=-1)[:, 100:-100]
    return x / x.std(axis=-1, keepdims=True).mean()


def make_session(
    rng: np.random.Generator,
    A: np.ndarray,
    labels: np.ndarray,
    band_gain: tuple[float, float],
    run_gains: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, runs). run_gains: (N_RUNS, 2) extra (mu, beta) gains per run, or None."""
    n = len(labels)
    runs = np.repeat(np.arange(N_RUNS), n // N_RUNS)
    src = np.zeros((n, N_CHANS, N_TIMES))
    for k, f in enumerate(TASK_FREQS):
        amp = np.where(labels == k, ERD_FACTOR, 1.0)
        band = 0 if k < 2 else 1
        gain = band_gain[band] * (run_gains[runs, band] if run_gains is not None else 1.0)
        src[:, k] = (amp * gain)[:, None] * _bandlimited(rng, n, f)
    src[:, len(TASK_FREQS):] = BACKGROUND_AMP * rng.standard_normal((n, N_BACKGROUND, N_TIMES))
    X = np.einsum("cs,nst->nct", A, src) + SENSOR_NOISE * rng.standard_normal((n, N_CHANS, N_TIMES))
    return X.astype(np.float32), runs


def balanced_labels(rng: np.random.Generator, n: int, props=(0.25, 0.25, 0.25, 0.25)) -> np.ndarray:
    counts = np.round(np.array(props) * n).astype(int)
    counts[-1] = n - counts[:-1].sum()
    return rng.permutation(np.repeat(np.arange(N_CLASSES), counts))


def build(seed: int, drift: str):
    rng = np.random.default_rng(1000 + seed)
    A = rng.standard_normal((N_CHANS, N_CHANS)) / np.sqrt(N_CHANS) + np.eye(N_CHANS)
    n = N_RUNS * PER_RUN
    y_t = balanced_labels(rng, n)
    props = (0.7, 0.1, 0.1, 0.1) if drift == "prior_shift" else (0.25,) * 4
    y_e = balanced_labels(rng, n, props)
    run_t = run_e = None
    if drift == "run_drift":
        run_t = rng.uniform(*RUN_GAIN_RANGE, size=(N_RUNS, 2))
        run_e = rng.uniform(*RUN_GAIN_RANGE, size=(N_RUNS, 2))
    gain_e = (1.0, 1.0) if drift == "none" else (G_MU, G_BETA)
    X_t, r_t = make_session(rng, A, y_t, (1.0, 1.0), run_t)
    X_e, r_e = make_session(rng, A, y_e, gain_e, run_e)
    return X_t, y_t, r_t, X_e, y_e, r_e


def evaluate(seed: int, drift: str, device: torch.device) -> list[dict]:
    X_t, y_t, r_t, X_e, y_e, r_e = build(seed, drift)
    Xt = torch.as_tensor(X_t, device=device)
    Xe = torch.as_tensor(X_e, device=device)
    yt = torch.as_tensor(y_t, device=device)

    def model() -> ShallowConvNet:
        set_deterministic(seed)
        return ShallowConvNet(N_CHANS, N_CLASSES, N_TIMES).to(device)

    acc = lambda p: float((p.argmax(1) == y_e).mean())  # noqa: E731
    std = train_model(model(), Xt, yt, CFG, seed=seed)
    rows = {"none": acc(predict_proba(std, Xe))}
    rows["adabn"] = acc(predict_proba(adapt_bn_statistics(copy.deepcopy(std), Xe), Xe))
    rows["mutant_tstats"] = acc(predict_proba(adapt_bn_statistics(copy.deepcopy(std), Xt), Xe))
    sat = train_model(model(), Xt, yt, CFG, seed=seed, groups=r_t)
    rows["sat"] = acc(predict_proba_groupwise(sat, Xe, r_e))
    return [{"seed": seed, "drift": drift, "arm": a, "acc": v} for a, v in rows.items()]


def check_gate(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    m = df.groupby(["drift", "arm"])["acc"].mean()
    clean = df[(df.drift == "none") & (df.arm == "none")]["acc"]
    drop = m["none", "none"] - m["band_gain", "none"]
    rec = lambda d, a: (m[d, a] - m[d, "none"]) / (m["none", "none"] - m[d, "none"])  # noqa: E731
    run_drop = m["none", "none"] - m["run_drift", "none"]
    return [
        ("G1 clean decodability in [0.70,0.98]", bool(clean.between(0.70, 0.98).all()), f"{clean.round(3).tolist()}"),
        ("G2 no harm (adabn, sat within 3pp)",
         bool(abs(m["none", "adabn"] - m["none", "none"]) <= 0.03 and abs(m["none", "sat"] - m["none", "none"]) <= 0.03),
         f"none={m['none','none']:.3f} adabn={m['none','adabn']:.3f} sat={m['none','sat']:.3f}"),
        ("G3 planted drift drops none >= 15pp", bool(drop >= 0.15), f"drop={drop*100:.1f}pp"),
        ("G4 adabn recovers >= 70% (band_gain)", bool(rec("band_gain", "adabn") >= 0.70), f"{rec('band_gain','adabn')*100:.0f}%"),
        ("G5 mutant recovers <= 10% (band_gain)", bool(rec("band_gain", "mutant_tstats") <= 0.10), f"{rec('band_gain','mutant_tstats')*100:.0f}%"),
        ("G6 sat recovers >= 70% (run_drift)", bool(run_drop > 0 and rec("run_drift", "sat") >= 0.70),
         f"drop={run_drop*100:.1f}pp recovery={rec('run_drift','sat')*100:.0f}%"),
    ]


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    rows = []
    for drift in ("none", "band_gain", "run_drift", "prior_shift"):
        for seed in SEEDS:
            wait_until_healthy()
            rows += evaluate(seed, drift, device)
            print(f"  {drift:12s} seed {seed}: " + " ".join(f"{r['arm']}={r['acc']:.3f}" for r in rows[-4:]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\n=== seed-mean accuracy ===")
    print(df.pivot_table(index="drift", columns="arm", values="acc").round(3).to_string())
    print("\n=== gate ===")
    results = check_gate(df)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("  [REPORT] prior_shift (known AdaBN weakness, not gated) - see table")
    print(f"\nwrote {OUT_CSV}  runtime {time.time()-t0:.0f}s")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
