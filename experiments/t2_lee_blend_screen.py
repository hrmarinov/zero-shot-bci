"""T2, Lee2019_MI blend screen: does adding TCFormer to R11 help on 54 new subjects? PRE-REGISTERED.

Committed BEFORE any TCFormer fit on Lee2019. Motivation (reports/t1_council.md): on IV-2a an equal-weight average of the
R11 probabilities and the 2-seed TCFormer probabilities scored 90.32 % against 82.45 % for R11 (9 of 9 subjects up).
That was found after looking at IV-2a, so it is a lead, not a result. This screen tests it on data it was not found on.

DATA (same as D2): MOABB Lee2019_MI, left against right hand, session 1 (100 labelled trials) -> session 2 (100 test
trials), the 20 sensorimotor channels of D2, 0-4 s, resampled to 250 Hz. TCFormer input is BROADBAND (0.5-100 Hz, as the
published recipe uses no band-pass), exported once by ``--export`` and checked trial by trial against the D2 cache
(labels, session labels and order must be identical).

MODELS
  r11      the D2 R11 predictions, unchanged (data/cache/d2_lee2019/preds; (CNN seeds 0-2 + TS+LR + FTS seeds 0-1)/3).
  tcf      TCFormer, published recipe of T1 (per-channel z-score fitted on the training session, interaug, 1000 epochs,
           Adam lr 9e-4 betas (0.5, 0.999) wd 1e-3, 20 warm-up epochs plus cosine, last epoch scored), seed 0 only.
           Two deliberate changes for 100 training trials and 2 classes: 20 input channels, batch 20 (5 steps per epoch,
           5000 steps in total, close to the 6000 of the recipe on IV-2a), no EOG arm (Lee2019 has no EOG channels).
  blend    0.5 * r11 + 0.5 * tcf, probabilities. The weight is fixed here and is not tuned.
  control  0.5 * r11 + 0.5 * mean(CNN seeds 3 and 4), D2's cached ShallowConvNets. This replaces TCFormer by the same
           kind of second opinion that R11 already contains, so it asks whether TCFormer in particular is needed.

CONTRASTS over the 54 subjects (accuracy on session 2), one-sided sign-flip Monte-Carlo test (as D2), Holm over B1-B3:
  B1  blend - r11        B2  blend - tcf        B3  blend - control
DECISION. PASS only if ALL hold: B1 mean >= +1.5 pp; B1, B2 and B3 each have Holm p < 0.05. (+1.5 pp is below the
1.9 pp MDE of D2, so a pass is a screen, not a confirmation: a Zhou2020 opening needs an effect >= 3 pp.)
  PASS        the blend is a candidate; next, a second TCFormer seed and the sealed lockbox rules in
              docs/lockbox_zhou2020_protocol.md.
  PARTIAL     B1 passes and B3 does not: any second model helps, TCFormer is not special; use the cheaper one.
  FAIL        B1 mean < +1.5 pp or Holm p >= 0.05: the IV-2a blend result does not generalise to this cohort.
REPORTED, NOT USED FOR THE DECISION: tcf - r11 (is TCFormer ahead at n = 54), share of subjects >= 70 %, bottom third
by csp_lda (D2), per-subject up/down, a descriptive weight sweep 0, 0.25, 0.5, 0.75, 1.

    .venv\\Scripts\\python experiments\\t2_lee_blend_screen.py --export                # laptop, once (needs Lee2019 raw data)
    .venv\\Scripts\\python experiments\\t2_lee_blend_screen.py --fit --data-dir DIR --out-dir DIR2 [--subjects 1-27]
    .venv\\Scripts\\python experiments\\t2_lee_blend_screen.py --summarise --out-dir DIR2        # laptop
    .venv\\Scripts\\python experiments\\t2_lee_blend_screen.py --dry-run                        # 2 epochs, no accuracy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

N_SUBJECTS, EPOCHS_LEE, BATCH_LEE, N_CHANS, N_CLASSES = 54, 1000, 20, 20, 2
BLEND_W = 0.5
MIN_GAIN_PP = 1.5
HOLM_ALPHA = 0.05
LITERACY = 0.70
SEED = 0


def parse_subjects(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    bad = [s for s in out if not 1 <= s <= N_SUBJECTS]
    if bad:
        raise ValueError(f"subjects out of range: {bad}")
    return out


def zscore_by_train(X: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Per-channel z-score fitted on the training session (the recipe's normalisation)."""
    ref = X[train].transpose(1, 0, 2).reshape(X.shape[1], -1)
    mu, sd = ref.mean(1), ref.std(1)
    return ((X - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)


def export(out_dir: Path) -> int:
    """Laptop only: broadband trials per subject, checked against the D2 cache."""
    from moabb.datasets import Lee2019_MI
    from moabb.paradigms import MotorImagery

    from experiments.d2_lee2019_dev import CACHE, CHANNELS, CLASSES, SFREQ
    from src.adaptation.convnet import encode_labels

    ds = Lee2019_MI()
    ds._selected_sessions = None  # same MOABB workaround as D2
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in range(1, N_SUBJECTS + 1):
        f = out_dir / f"s{s}.npz"
        if f.exists():
            continue
        X, y, meta = MotorImagery(events=CLASSES, n_classes=2, resample=SFREQ, channels=CHANNELS, fmin=0.5, fmax=100.0,
                                  tmin=0.0, tmax=4.0).get_data(dataset=ds, subjects=[s])
        sess = meta.session.to_numpy()
        ref = np.load(CACHE / f"s{s}.npz", allow_pickle=True)
        yy = encode_labels(y, CLASSES)
        if not (np.array_equal(yy, ref["y"]) and np.array_equal(sess, ref["groups"])) or X.shape != ref["X_cnn"].shape:
            raise AssertionError(f"s{s}: broadband trials do not line up with the D2 cache")
        np.savez_compressed(f, X=X.astype(np.float32), y=yy, train=ref["train"])
        print(f"  exported s{s}: {X.shape}", flush=True)
    return 0


def fit(data_dir: Path, out_dir: Path, subjects: list[int], epochs: int, dry: bool) -> int:
    import torch

    from experiments.t1_tcformer_head_to_head import fit_predict
    from src.gpu_health import wait_until_healthy

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    for s in subjects[: 1 if dry else None]:
        f = out_dir / f"s{s}_seed{SEED}.npy"
        if f.exists() and not dry:
            continue
        z = np.load(data_dir / f"s{s}.npz")
        X, y, tr = zscore_by_train(z["X"], z["train"]), z["y"], z["train"]
        t0 = time.time()
        wait_until_healthy()
        p = fit_predict(X[tr], y[tr], X[~tr], SEED, epochs, device, N_CHANS, N_CLASSES, BATCH_LEE)
        if dry:
            print(f"dry run s{s}: {epochs} epochs in {time.time() - t0:.1f} s, probs {p.shape}, sum {p.sum(1).mean():.3f}")
            print("DRY RUN OK")
            return 0
        np.save(f, p)
        print(f"  s{s}: {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


def summarise(out_dir: Path) -> int:
    import pandas as pd

    from experiments.d2_lee2019_dev import CACHE, load, sign_flip_p, sign_test_share
    from src.config import RESULTS_DIR
    from src.evaluate import holm_adjust
    from src.pipeline import tslr_member

    d2 = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    rows = []
    for s in range(1, N_SUBJECTS + 1):
        data = load(None, s, {}, {}, ["cached"])  # served from the D2 cache; no download
        y = data.y[~data.train]
        c = CACHE / "preds" / f"s{s}"
        base = [np.load(c / f"cnn_seed{k}_fast.npy") for k in range(5)]
        fts = np.mean([np.load(c / f"fts_seed{k}_fast.npy") for k in range(2)], 0)
        r11 = (np.mean(base[:3], 0) + tslr_member(data) + fts) / 3
        if abs((r11.argmax(1) == y).mean() - d2.loc[s, "r11"]) > 1e-9:
            raise AssertionError(f"s{s}: rebuilt R11 differs from the D2 result")
        tcf = np.load(out_dir / f"s{s}_seed{SEED}.npy")
        ctrl = np.mean(base[3:], 0)
        a = lambda p: float((p.argmax(1) == y).mean())  # noqa: E731
        row = {"subject": s, "r11": a(r11), "tcf": a(tcf), "blend": a(BLEND_W * r11 + (1 - BLEND_W) * tcf),
               "control": a(BLEND_W * r11 + (1 - BLEND_W) * ctrl), "csp_lda": d2.loc[s, "csp_lda"]}
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            row[f"w{w}"] = a(w * r11 + (1 - w) * tcf)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("subject")
    pp = 100 * df
    df.to_csv(RESULTS_DIR / "t2_lee_blend_screen.csv")
    third = pp.csp_lda.sort_values().index[: N_SUBJECTS // 3]
    print(f"\n=== T2 Lee2019 blend screen, {len(pp)} subjects, mean accuracy (%) ===")
    print(pp[["r11", "tcf", "blend", "control"]].mean().round(2).to_string())
    print("share >= 70 %: " + ", ".join(f"{k} {(pp[k] >= 100 * LITERACY).mean():.1%}" for k in ("r11", "tcf", "blend", "control")))
    print(f"bottom third by csp_lda ({len(third)}): " + ", ".join(f"{k} {pp.loc[third, k].mean():.2f}" for k in ("r11", "tcf", "blend", "control")))
    print("weight sweep (weight on r11): " + ", ".join(f"{w} {pp[f'w{w}'].mean():.2f}" for w in (0.0, 0.25, 0.5, 0.75, 1.0)))
    cons = {"B1 blend-r11": ("blend", "r11"), "B2 blend-tcf": ("blend", "tcf"), "B3 blend-control": ("blend", "control")}
    res = []
    for name, (x, y_) in cons.items():
        d = (pp[x] - pp[y_]).to_numpy()
        up, down, p_share = sign_test_share(pp[x].to_numpy() / 100, pp[y_].to_numpy() / 100)
        res.append({"contrast": name, "mean_pp": d.mean(), "up": int((d > 0).sum()), "down": int((d < 0).sum()),
                    "p_mean": sign_flip_p(d), "share_up": up, "share_down": down, "p_share": p_share,
                    "bottom3_pp": float((pp.loc[third, x] - pp.loc[third, y_]).mean())})
    out = pd.DataFrame(res)
    out["holm_p_mean"] = holm_adjust(out.p_mean.tolist())
    print("\nCONTRASTS (one-sided, Holm over B1-B3):")
    print(out.round(4).to_string(index=False))
    d = (pp.tcf - pp.r11).to_numpy()
    print(f"\nreported only, tcf - r11: mean {d.mean():+.2f} pp, {(d > 0).sum()} up / {(d < 0).sum()} down, "
          f"two-sided-ish p (one-sided greater) {sign_flip_p(d):.4f}")
    out.to_csv(RESULTS_DIR / "t2_lee_blend_screen_contrasts.csv", index=False)
    b1, b2, b3 = out.iloc[0], out.iloc[1], out.iloc[2]
    passed_b1 = b1.mean_pp >= MIN_GAIN_PP and b1.holm_p_mean < HOLM_ALPHA
    if passed_b1 and b2.holm_p_mean < HOLM_ALPHA and b3.holm_p_mean < HOLM_ALPHA:
        verdict = "PASS"
    elif passed_b1:
        verdict = "PARTIAL (a second model helps, TCFormer is not shown to be special)"
    else:
        verdict = "FAIL"
    print(f"\nTHE SCREEN: {verdict}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("data/cache/t2_preds"))
    ap.add_argument("--subjects", default=f"1-{N_SUBJECTS}")
    a = ap.parse_args()
    if a.export:
        return export(a.data_dir or Path("data/cache/t2_lee_broadband"))
    if a.fit or a.dry_run:
        return fit(a.data_dir or Path("data/cache/t2_lee_broadband"), a.out_dir, parse_subjects(a.subjects),
                   2 if a.dry_run else EPOCHS_LEE, a.dry_run)
    if a.summarise:
        return summarise(a.out_dir)
    ap.error("choose --export, --fit, --summarise or --dry-run")
    return 2


if __name__ == "__main__":
    sys.exit(main())
