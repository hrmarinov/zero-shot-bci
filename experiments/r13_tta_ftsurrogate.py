"""R13, round-5 screen: test-time FTSurrogate augmentation (IV-2a + IV-2b, 18 subjects).

Ideation and scoring: reports/r13_ideation_test_time_ftsurrogate.md. R12 showed member addition has
saturated; this round changes how each CNN member predicts, not which members there are.

COMPARATOR (R11 best, fixed): B = (CNN3 + TSLR + FTS2) / 3, exactly as in R11/R12.

RETRAINING. The CNN weights were never saved, so every CNN member is retrained here with the exact
code path that produced its cache:
  IV-2a: R8's fit_predict (integer targets, ShallowConvNet(22, 4, 1001), CFG, WIDE, per-session RA);
         base = no augmentation, seeds 0-2 (= R3 cnn_ra); FTS = make_augmentation("ftsurrogate", k), seeds 0-1.
  IV-2b: R9's fit_predict (one-hot targets, ShallowConvNet(3, 2, 1001), CFG, WIDE, per-recording RA,
         train 0-2train, predict 3test+4test); base = R7 teacher seeds 0-2; FTS = R9 seeds 0-1.

TTA (fixed a priori): K = 8 copies per test trial from
  FTSurrogate(probability=1.0, phase_noise_magnitude=0.9, random_state=1000 + model seed), drawn in
  sequence on the full evaluation tensor; member TTA prediction = (clean softmax + sum of the K copy
  softmaxes) / (K + 1).

ARMS:
  tta_fts  (CNN3 + TSLR + TTA(FTS2)) / 3      TTA on the FTSurrogate members only.
  tta_all  (TTA(CNN3) + TSLR + TTA(FTS2)) / 3 TTA on every CNN member.
PRIMARY: arm - B, pooled over 18 subjects (subject score = mean over its evaluation recordings).
  PASS iff pooled >= +0.5 pp AND one-sided sign-flip p < 0.05 (Holm over the 2 arms) AND > 0 on each dataset.
  No member-count control applies (the member set is unchanged); R12's ctrl - B (-0.18 pp) is the
  reference for "more of the same".
REPORTED: single-model and 2/3-seed ensemble accuracy with vs without TTA, per family and dataset.
NEW BEST: the PASS arm with the larger pooled gain; else B stays and the loop stops again.

VALIDITY:
  V1 Trial order: the 8-30 Hz loader's labels/runs/recordings equal the wide loader's.
  V2 Every retrained model's CLEAN predictions equal its cached predictions (argmax identical, and
     max |prob diff| < 1e-5), for all 90 models. This proves the TTA models are the cached members.
  V3 B recomputed from the cache equals the R11 "fts" arm (< 1e-12).
  V4 Determinism: (a) a second run with warm caches gives a byte-identical CSV; (b) the IV-2b
     subject-1 TTA cache is deleted and recomputed from scratch (6 fits are retrained), and the
     recomputed files must equal the first ones byte for byte.

    .venv\\Scripts\\python experiments\\r13_tta_ftsurrogate.py     # ~90 min GPU
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
from braindecode.augmentation import FTSurrogate  # noqa: E402

from experiments.r3_convnet_session_norm import CFG, N_TIMES_WIDE, WIDE, align_per_session  # noqa: E402
from experiments.r7_self_training_iv2b import CLASSES as IV2B_CLASSES  # noqa: E402
from experiments.r7_self_training_iv2b import EVAL_RECORDINGS, N_TIMES, TRAIN_RECORDINGS, align_recordings  # noqa: E402
from experiments.r10_round2_stack_position import align_by_recording, tangent, ts_lr  # noqa: E402
from experiments.r11_round3_diverse_members import CACHE, contrast_line, rec_acc  # noqa: E402
from src.adaptation.augmentation import make_augmentation  # noqa: E402
from src.adaptation.convnet import ShallowConvNet, encode_labels, predict_proba, set_deterministic, train_model  # noqa: E402
from src.adaptation.self_training import one_hot  # noqa: E402
from src.config import FMAX, FMIN, IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.datasets import load_iv2a_subject_window, load_iv2b_subject_window  # noqa: E402
from src.evaluate import holm_adjust, sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402

OUT_CSV = RESULTS_DIR / "r13_tta_ftsurrogate.csv"
PRED_DIR = CACHE / "r13_preds"
K = 8
TTA_PHASE_NOISE = 0.9
TTA_SEED_OFFSET = 1000
GATE_PP = 0.5
ARMS = ("tta_fts", "tta_all")
FAMILIES = {"base": (0, 1, 2), "fts": (0, 1)}
TOL_V2, TOL_V3 = 1e-5, 1e-12


def tta_predict(model, Xe: torch.Tensor, seed: int) -> np.ndarray:
    """Mean softmax over K FTSurrogate copies of every test trial (the clean pass is added by the caller)."""
    tf = FTSurrogate(1.0, phase_noise_magnitude=TTA_PHASE_NOISE, random_state=TTA_SEED_OFFSET + seed)
    with torch.no_grad():
        return np.mean([predict_proba(model, tf(Xe)) for _ in range(K)], 0)


def fit_clean_and_tta(make_model, Xt, yt, Xe, seed: int, aug: str | None) -> tuple[np.ndarray, np.ndarray]:
    wait_until_healthy()
    set_deterministic(seed)
    augment = None if aug is None else make_augmentation(aug, seed)
    model = train_model(make_model(), Xt, yt, CFG, seed=seed, augment=augment)  # built AFTER seeding, as in R8/R9
    clean = predict_proba(model, Xe)
    tta = tta_predict(model, Xe, seed)
    del model
    torch.cuda.empty_cache()
    return clean, tta


def members(tag: str, s: int, cached_paths: dict, train) -> dict:
    """{family: [(clean, tta_mean), ...]} with V2 asserted against the cached predictions."""
    out = {}
    for fam, seeds in FAMILIES.items():
        out[fam] = []
        for k in seeds:
            path = PRED_DIR / f"{tag}_s{s}_{fam}_seed{k}.npz"
            if path.exists():
                z = np.load(path)
                clean, tta = z["clean"], z["tta"]
            else:
                clean, tta = train(k, None if fam == "base" else "ftsurrogate")
                np.savez(path, clean=clean, tta=tta)
            ref = np.load(cached_paths[fam](k))
            if not (np.array_equal(clean.argmax(1), ref.argmax(1)) and np.abs(clean - ref).max() < TOL_V2):
                raise AssertionError(f"V2: {tag} s{s} {fam} seed{k} retrain does not reproduce the cache "
                                     f"(max diff {np.abs(clean - ref).max():.2e})")
            out[fam].append((ref, tta))
    return out


def score(rows, dataset, s, y, rec, mem, ts) -> None:
    def ens(fam, use_tta):
        return np.mean([(c + K * t) / (K + 1) if use_tta else c for c, t in mem[fam]], 0)

    probs = {"B": (ens("base", False) + ts + ens("fts", False)) / 3,
             "tta_fts": (ens("base", False) + ts + ens("fts", True)) / 3,
             "tta_all": (ens("base", True) + ts + ens("fts", True)) / 3}
    for fam in FAMILIES:
        for use in (False, True):
            sfx = "_tta" if use else ""
            probs[f"ens_{fam}{sfx}"] = ens(fam, use)
            rows.append({"dataset": dataset, "subject": s, "arm": f"single_{fam}{sfx}",
                         "acc": float(np.mean([rec_acc((c + K * t) / (K + 1) if use else c, y, rec)
                                               for c, t in mem[fam]]))})
    for arm, p in probs.items():
        rows.append({"dataset": dataset, "subject": s, "arm": arm, "acc": rec_acc(p, y, rec)})


def run_iv2a(s: int, rows: list, device) -> None:
    w = load_iv2a_subject_window(s, **WIDE)
    d = load_iv2a_subject_window(s, FMIN, FMAX, TMIN, TMAX)
    if not (np.array_equal(d.y_eval, w.y_eval) and np.array_equal(d.y_calib, w.y_calib)
            and np.array_equal(d.runs_eval, w.runs_eval)):
        raise AssertionError(f"V1: IV-2a s{s} trial order differs between loaders")
    yt, ye = encode_labels(d.y_calib, IV2A_CLASSES), encode_labels(d.y_eval, IV2A_CLASSES)
    At, Ae = align_per_session(d.X_calib, d.X_eval)
    Ft, Fe = tangent(At, Ae)
    ts = ts_lr(Ft, yt, Fe)
    Xc, Xe = align_per_session(w.X_calib, w.X_eval)
    Xc_t = torch.as_tensor(Xc.astype(np.float32), device=device)
    Xe_t = torch.as_tensor(Xe.astype(np.float32), device=device)
    yc_t = torch.as_tensor(encode_labels(w.y_calib, IV2A_CLASSES), device=device)

    def train(k, aug):  # R8's fit_predict
        return fit_clean_and_tta(lambda: ShallowConvNet(22, 4, N_TIMES_WIDE).to(device), Xc_t, yc_t, Xe_t, k, aug)

    paths = {"base": lambda k: CACHE / "r3_preds" / f"s{s}_cnn_ra_seed{k}.npy",
             "fts": lambda k: CACHE / "r8_preds" / f"s{s}_ftsurrogate_seed{k}.npy"}
    score(rows, "IV-2a", s, ye, np.zeros(len(ye)), members("iv2a", s, paths, train), ts)


def run_iv2b(s: int, rows: list, device) -> None:
    w = load_iv2b_subject_window(s, **WIDE)
    d = load_iv2b_subject_window(s, FMIN, FMAX, TMIN, TMAX)
    if not (np.array_equal(d.y, w.y) and np.array_equal(d.session, w.session)):
        raise AssertionError(f"V1: IV-2b s{s} label/recording order differs between loaders")
    y = encode_labels(d.y, IV2B_CLASSES)
    tr = np.isin(d.session, TRAIN_RECORDINGS)
    ev = np.concatenate([np.flatnonzero(d.session == r) for r in EVAL_RECORDINGS])
    A = align_by_recording(d.X, d.session)
    Ft, Fe = tangent(A[tr], A[ev])
    ts = ts_lr(Ft, y[tr], Fe)
    X = align_recordings(w.X, w.session)
    Xt = torch.as_tensor(X[tr].astype(np.float32), device=device)
    Yt = torch.as_tensor(one_hot(y[tr], 2).astype(np.float32), device=device)
    Xe = torch.as_tensor(X[ev].astype(np.float32), device=device)

    def train(k, aug):  # R9's fit_predict
        return fit_clean_and_tta(lambda: ShallowConvNet(3, 2, N_TIMES).to(device), Xt, Yt, Xe, k, aug)

    paths = {"base": lambda k: CACHE / "r7_preds" / f"s{s}_teacher_seed{k}.npy",
             "fts": lambda k: CACHE / "r9_preds" / f"s{s}_ftsurrogate_seed{k}.npy"}
    score(rows, "IV-2b", s, y[ev], d.session[ev], members("iv2b", s, paths, train), ts)


def check_v3(df: pd.DataFrame) -> None:
    ref = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv")
    ref = ref[ref.arm == "fts"].set_index(["dataset", "subject"]).acc
    got = df[df.arm == "B"].set_index(["dataset", "subject"]).acc
    diff = (got - ref.reindex(got.index)).abs().max()
    if len(got) != 18 or not diff < TOL_V3:
        raise AssertionError(f"V3: R11 best not reproduced (max diff {diff})")
    print("V3: R11 best (B) reproduced for all 18 subjects")


def summarise(df: pd.DataFrame) -> None:
    piv = 100 * df.pivot_table(index=["dataset", "subject"], columns="arm", values="acc")
    print("\n=== R13 test-time FTSurrogate (IV-2a + IV-2b, 18 subjects) ===")
    print(piv.groupby("dataset").mean().round(2).T.to_string())
    ps = {a: sign_flip_permutation_test((piv[a] - piv["B"]).to_numpy(), "greater") for a in ARMS}
    holm = dict(zip(ps, holm_adjust(list(ps.values()))))
    passed = {}
    print("\nPrimary (arm - B):")
    for a in ARMS:
        d = piv[a] - piv["B"]
        ok = d.mean() >= GATE_PP and holm[a] < 0.05 and (d.groupby("dataset").mean() > 0).all()
        print(contrast_line(a, d, ps[a], holm[a]) + f"  -> {'PASS' if ok else 'killed'}")
        if ok:
            passed[a] = d.mean()
    print("Reported (TTA minus clean):")
    for fam in FAMILIES:
        for kind in ("single", "ens"):
            d = piv[f"{kind}_{fam}_tta"] - piv[f"{kind}_{fam}"]
            print(contrast_line(f"{kind}_{fam}", d, sign_flip_permutation_test(d.to_numpy(), "greater")))
    best = max(passed, key=passed.get) if passed else "B"
    print(f"\nNEW BEST: {best}" + ("" if passed else " (no arm passed; R11 best stays: no improvement)"))


def main() -> int:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    rows: list[dict] = []
    for s in IV2A_SUBJECTS:
        run_iv2a(s, rows, device)
        run_iv2b(s, rows, device)
        print(f"  s{s} done (V1, V2 passed)", flush=True)
    df = pd.DataFrame(rows)
    check_v3(df)
    df.to_csv(OUT_CSV, index=False)
    summarise(df)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
