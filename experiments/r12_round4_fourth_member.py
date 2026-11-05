"""R12, round-4 screen: a fourth ensemble member (IV-2a + IV-2b, 18 subjects).

Ideation and scoring: reports/r12_ideation_fourth_member.md.

COMPARATOR (R11 best, fixed): B = (CNN3 + TSLR + FTS2) / 3, exactly as in R11
  (CNN3 = cached base ShallowConvNet seeds 0-2; FTS2 = cached FTSurrogate seeds 0-1;
   TSLR = 8-30 Hz, 0.5-2.5 s, per-session/recording RA, OAS -> TangentSpace -> LR(C=1)).

ARMS (4th member, equal 1/4 weights, every setting a priori, nothing tuned on E):
  zoo   (CNN3 + TSLR + FTS2 + ZOO) / 4. ZOO = mean softmax of ShallowConvNet seed 0 trained with
        each of freqshift, timemask, timereverse (make_augmentation(name, 0), p = 0.5, the R8 settings).
        IV-2a: cached R8 predictions (data/cache/r8_preds). IV-2b: trained here exactly like R9
        (one-hot targets, CFG, WIDE, per-recording RA, train 0-2train, predict 3test+4test), cached
        in data/cache/r12_preds. ChannelsDropout is excluded a priori (-2.5 pp alone in R8).
  late  (CNN3 + TSLR + FTS2 + LATE) / 4. LATE = TSLR with the window 2.0-4.0 s instead of 0.5-2.5 s.

SIZE-MATCHED CONTROL: ctrl = (CNN3 + TSLR + FTS2 + MORE) / 4, MORE = mean softmax of base CNN seeds
  3, 4 and FTSurrogate seed 2 (three nets of the existing families, as ZOO has three nets).

PRIMARY: arm - B, pooled over 18 subjects (subject score = mean over its evaluation recordings).
  PASS iff pooled >= +0.5 pp AND one-sided sign-flip p < 0.05 (Holm over the 2 arms) AND > 0 on each dataset.
SECONDARY: arm - ctrl (one-sided p, unadjusted): diversity credited iff pooled > 0 and p < 0.05.
REPORTED: ctrl - B; each member alone.
NEW BEST: the PASS arm with the largest pooled gain; if none passes, B stays and the loop's
  stopping rule counts this round as "no improvement".

VALIDITY:
  V1 Every loader's labels/runs/recordings equal the wide CNN loader's (cache order).
  V2 Every cache's recomputed accuracy equals its recorded value (< 1e-9): r3, r7, r8 (incl. the
     zoo augmentations' 3-seed single means), r9.
  V3 R11 reproduction: this run's B per subject equals the R11 CSV arm "fts" (< 1e-12).
  V4 GPU path: for every subject, the seed-0 unaugmented IV-2b model retrained here reproduces R7
     teacher seed 0 exactly (argmax), and for subject 1 FTSurrogate seed 0 reproduces R9 seed 0
     (argmax), so the augmentation path is the same code path as R9.
  V5 Determinism: a second full run (all caches warm) gives a byte-identical CSV.

    .venv\\Scripts\\python experiments\\r12_round4_fourth_member.py     # ~30 min GPU (first run) + ~10 min CPU
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

from experiments.r3_convnet_session_norm import CFG, WIDE, align_per_session  # noqa: E402
from experiments.r7_self_training_iv2b import CLASSES as IV2B_CLASSES  # noqa: E402
from experiments.r7_self_training_iv2b import EVAL_RECORDINGS, N_TIMES, TRAIN_RECORDINGS, align_recordings, cached  # noqa: E402
from experiments.r10_round2_stack_position import align_by_recording, tangent, ts_lr  # noqa: E402
from experiments.r11_round3_diverse_members import CACHE, check_v2, contrast_line, rec_acc  # noqa: E402
from src.adaptation.augmentation import make_augmentation  # noqa: E402
from src.adaptation.convnet import ShallowConvNet, encode_labels, predict_proba, set_deterministic, train_model  # noqa: E402
from src.adaptation.self_training import one_hot  # noqa: E402
from src.config import FMAX, FMIN, IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.datasets import load_iv2a_subject_window, load_iv2b_subject_window  # noqa: E402
from src.evaluate import holm_adjust, sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402

OUT_CSV = RESULTS_DIR / "r12_round4_fourth_member.csv"
PRED_DIR = CACHE / "r12_preds"
ZOO = ("freqshift", "timemask", "timereverse")
LATE = (2.0, 4.0)
GATE_PP = 0.5
ARMS = ("zoo", "late")
TOL_V3 = 1e-12


# ---------------------------------------------------------------- GPU (IV-2b zoo)
def fit_predict_iv2b(Xt: np.ndarray, Yt: np.ndarray, Xe: np.ndarray, seed: int, aug: str | None, device) -> np.ndarray:
    """R9's fit_predict with the augmentation name as a parameter (aug=None is the R7 teacher)."""
    wait_until_healthy()
    set_deterministic(seed)
    augment = None if aug is None else make_augmentation(aug, seed)
    model = train_model(ShallowConvNet(3, 2, N_TIMES).to(device),
                        torch.as_tensor(Xt.astype(np.float32), device=device),
                        torch.as_tensor(Yt.astype(np.float32), device=device), CFG, seed=seed, augment=augment)
    probs = predict_proba(model, torch.as_tensor(Xe.astype(np.float32), device=device))
    del model
    torch.cuda.empty_cache()
    return probs


def iv2b_zoo(s: int, w, device) -> list[np.ndarray]:
    paths = [PRED_DIR / f"s{s}_{a}_seed0.npy" for a in ZOO]
    if all(p.exists() for p in paths):
        return [np.load(p) for p in paths]
    X = align_recordings(w.X, w.session)
    y = encode_labels(w.y, IV2B_CLASSES)
    tr = np.isin(w.session, TRAIN_RECORDINGS)
    ev = np.concatenate([np.flatnonzero(w.session == r) for r in EVAL_RECORDINGS])
    Xt, Yt, Xe = X[tr], one_hot(y[tr], 2), X[ev]
    again = fit_predict_iv2b(Xt, Yt, Xe, 0, None, device)
    if not np.array_equal(again.argmax(1), np.load(CACHE / "r7_preds" / f"s{s}_teacher_seed0.npy").argmax(1)):
        raise AssertionError(f"V4: IV-2b s{s} seed-0 baseline does not reproduce R7 teacher seed 0")
    if s == 1:
        fts = fit_predict_iv2b(Xt, Yt, Xe, 0, "ftsurrogate", device)
        if not np.array_equal(fts.argmax(1), np.load(CACHE / "r9_preds" / "s1_ftsurrogate_seed0.npy").argmax(1)):
            raise AssertionError("V4: IV-2b s1 FTSurrogate seed 0 does not reproduce R9")
    return [cached(p, lambda a=a: fit_predict_iv2b(Xt, Yt, Xe, 0, a, device)) for p, a in zip(paths, ZOO)]


# ---------------------------------------------------------------- scoring
def score(rows, dataset, s, y, rec, base, fts, zoo, ts, late) -> None:
    cnn3, fts2 = np.mean(base[:3], 0), np.mean(fts[:2], 0)
    more = np.mean([base[3], base[4], fts[2]], 0)
    zoo_m = np.mean(zoo, 0)
    core = cnn3 + ts + fts2
    probs = {"B": core / 3, "ctrl": (core + more) / 4, "zoo": (core + zoo_m) / 4, "late": (core + late) / 4,
             "m_zoo": zoo_m, "m_late": late, "m_more": more}
    for a, z in zip(ZOO, zoo):
        probs[f"m_{a}"] = z
    for arm, p in probs.items():
        rows.append({"dataset": dataset, "subject": s, "arm": arm, "acc": rec_acc(p, y, rec)})


def tslr_pair(Xt, yt, Xe) -> np.ndarray:
    Ft, Fe = tangent(Xt, Xe)
    return ts_lr(Ft, yt, Fe)


def run_iv2a(s: int, rows: list, tables: dict) -> None:
    w = load_iv2a_subject_window(s, **WIDE)
    d = load_iv2a_subject_window(s, FMIN, FMAX, TMIN, TMAX)
    dl = load_iv2a_subject_window(s, FMIN, FMAX, *LATE)
    for name, x in (("8-30", d), ("late", dl)):
        if not (np.array_equal(x.y_eval, w.y_eval) and np.array_equal(x.y_calib, w.y_calib)
                and np.array_equal(x.runs_eval, w.runs_eval)):
            raise AssertionError(f"V1: IV-2a s{s} {name} trial order differs from the wide loader")
    yt, ye = encode_labels(d.y_calib, IV2A_CLASSES), encode_labels(d.y_eval, IV2A_CLASSES)
    ts = tslr_pair(*_al(align_per_session(d.X_calib, d.X_eval), yt))
    late = tslr_pair(*_al(align_per_session(dl.X_calib, dl.X_eval), yt))
    base = [np.load(CACHE / "r3_preds" / f"s{s}_cnn_ra_seed{k}.npy") for k in range(5)]
    fts = [np.load(CACHE / "r8_preds" / f"s{s}_ftsurrogate_seed{k}.npy") for k in range(3)]
    r3, r8 = tables["r3"], tables["r8"]
    for k in range(5):
        want = r3[(r3.subject == s) & (r3.arm == "cnn_ra") & (r3.seed == k)].acc.item()
        check_v2(f"r3 s{s} seed{k}", (base[k].argmax(1) == ye).mean(), want)
    for arm in ("ftsurrogate",) + ZOO:
        ps = [np.load(CACHE / "r8_preds" / f"s{s}_{arm}_seed{k}.npy") for k in range(3)]
        want = r8[(r8.subject == s) & (r8.arm == arm) & (r8.metric == "single_mean_acc")].value.item()
        check_v2(f"r8 s{s} {arm}", np.mean([(p.argmax(1) == ye).mean() for p in ps]), want)
    zoo = [np.load(CACHE / "r8_preds" / f"s{s}_{a}_seed0.npy") for a in ZOO]
    score(rows, "IV-2a", s, ye, np.zeros(len(ye)), base, fts, zoo, ts, late)


def _al(pair, yt):
    return pair[0], yt, pair[1]


def run_iv2b(s: int, rows: list, tables: dict, device) -> None:
    w = load_iv2b_subject_window(s, **WIDE)
    d = load_iv2b_subject_window(s, FMIN, FMAX, TMIN, TMAX)
    dl = load_iv2b_subject_window(s, FMIN, FMAX, *LATE)
    for name, x in (("8-30", d), ("late", dl)):
        if not (np.array_equal(x.y, w.y) and np.array_equal(x.session, w.session)):
            raise AssertionError(f"V1: IV-2b s{s} {name} label/recording order differs from the wide loader")
    y = encode_labels(d.y, IV2B_CLASSES)
    tr = np.isin(d.session, TRAIN_RECORDINGS)
    ev = np.concatenate([np.flatnonzero(d.session == r) for r in EVAL_RECORDINGS])
    ye, ge = y[ev], d.session[ev]
    X, XL = align_by_recording(d.X, d.session), align_by_recording(dl.X, dl.session)
    ts, late = tslr_pair(X[tr], y[tr], X[ev]), tslr_pair(XL[tr], y[tr], XL[ev])
    base = [np.load(CACHE / "r7_preds" / f"s{s}_teacher_seed{k}.npy") for k in range(5)]
    fts = [np.load(CACHE / "r9_preds" / f"s{s}_ftsurrogate_seed{k}.npy") for k in range(3)]
    for tab, arm, preds in (("r7", "teacher", base), ("r9", "ftsurrogate", fts)):
        t = tables[tab]
        for r in EVAL_RECORDINGS:
            m = ge == r
            want = t[(t.subject == s) & (t.recording == r) & (t.arm == arm) & (t.metric == "single_mean_acc")].value.item()
            check_v2(f"{tab} s{s} {r}", np.mean([(p.argmax(1)[m] == ye[m]).mean() for p in preds]), want)
    zoo = iv2b_zoo(s, w, device)
    score(rows, "IV-2b", s, ye, ge, base, fts, zoo, ts, late)


# ---------------------------------------------------------------- analysis
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
    print("\n=== R12 round-4 screen (IV-2a + IV-2b, 18 subjects) ===")
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
    print("Secondary (arm - ctrl):")
    for a in ARMS:
        d = piv[a] - piv["ctrl"]
        p = sign_flip_permutation_test(d.to_numpy(), "greater")
        print(contrast_line(a, d, p) + f"  -> {'diversity' if d.mean() > 0 and p < 0.05 else 'not credited'}")
    d = piv["ctrl"] - piv["B"]
    print("Reported:\n" + contrast_line("ctrl - B", d, sign_flip_permutation_test(d.to_numpy(), "greater")))
    best = max(passed, key=passed.get) if passed else "B"
    print(f"\nNEW BEST: {best}" + ("" if passed else " (no arm passed; R11 best stays: no improvement)"))


def main() -> int:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tables = {k: pd.read_csv(RESULTS_DIR / f) for k, f in (
        ("r3", "r3_convnet_session_norm.csv"), ("r7", "r7_self_training_iv2b.csv"),
        ("r8", "r8_augmentation.csv"), ("r9", "r9_ftsurrogate_iv2b.csv"))}
    rows: list[dict] = []
    for s in IV2A_SUBJECTS:
        run_iv2a(s, rows, tables)
        run_iv2b(s, rows, tables, device)
        print(f"  s{s} done (V1, V2, V4 passed)", flush=True)
    df = pd.DataFrame(rows)
    check_v3(df)
    df.to_csv(OUT_CSV, index=False)
    summarise(df)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
