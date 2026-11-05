"""R14, EOG validity gate on the R11 best pipeline (IV-2a + IV-2b, 18 subjects). PRE-REGISTERED.

Why (reports/r14_council_and_ideation.md, item V): R11b showed that the cue-locked < 4 Hz signal is
class-dependent eye movement. No member of the R11 best pipeline B = (CNN3 + TSLR + FTS2) / 3 has an
EOG control, the CNN input (4-38 Hz, 0-4 s from cue) spans the saccade window, and IV-2b's E
recordings are feedback sessions where gaze follows the class-dependent smiley.

ARMS (all settings fixed a priori; no E labels used for anything but scoring):
  B         R11 best from the caches (CNN3 = r3/r7 seeds 0-2, FTS2 = r8/r9 seeds 0-1, TSLR as R10).
  B_clean   the same pipeline with EVERY member retrained on EOG-regressed EEG:
            per subject and per band (WIDE 4-38 Hz / 0-4 s for the CNNs, 8-30 Hz / 0.5-2.5 s for TS+LR),
            B = least squares of EEG on the 3 EOG channels over all TRAINING-session samples
            (IV-2b: recordings 0-2train), EEG_clean = EEG - EOG @ B applied to train and test alike,
            then the unchanged R8/R9 (CNN) and R10 (TS+LR) code paths, same seeds, same alignment.
  r10_clean 0.5 CNN3_clean + 0.5 TSLR_clean (from the same fits).
  eog_tslr      TS+LR (R10 code, RA) on the 3 EOG channels only, 8-30 Hz, 0.5-2.5 s.
  eog_tslr_wide TS+LR on the 3 EOG channels only, 4-38 Hz, 0-4 s (the CNN's band and window).
  eog_cnn       ShallowConvNet on the 3 EOG channels only, WIDE, RA, seed 0, R8/R9 training path.

PRIMARY GATE (ocular share of the headline):
  D = B - B_clean per subject (subject score = mean over its evaluation recordings).
  OCULAR-MATERIAL (gate FAILS) iff pooled D >= +1.0 pp AND one-sided sign-flip p < 0.05.
  Otherwise the gate PASSES: the R11 headline survives EOG removal, and D is reported as its ocular share.
  Pre-declared consequence: on FAIL, B_clean replaces B as the reference pipeline for the lockbox
  confirmation and the headline numbers are restated; on PASS, B stays.
  Caveat fixed in advance: regression also removes any neural signal that reaches the EOG electrodes,
  so D is an UPPER bound on the ocular share.

SECONDARY (reported, no gate):
  S1 B_clean - r10_clean: does the FTSurrogate diversity gain survive cleaning? (R11 rule, as description)
  S2 EOG-only members vs chance (25% IV-2a, 50% IV-2b), one-sided sign-flip on acc - chance.
  S3 Excess agreement: on trials the raw CNN3 gets wrong, the rate at which its prediction equals
     eog_cnn's, minus the rate expected if the two were independent within the true class.
  S4 Member-level raw - clean deltas; fraction of EEG variance removed by the regression (on E).

VALIDITY:
  V1 The raw-epoch EEG arrays cut here (MOABB IIR filter, MOABB window) equal the harness loader's
     arrays for both bands (max |diff| <= 1e-6 x max |X|), with identical labels and recordings.
  V2 TS+LR on the raw EEG reproduces R11's m_tslr per subject (< 1e-12); the regression residual is
     orthogonal to the EOG channels on the training session (max |uncentred cosine| < 1e-4; float64 round-off on
     near-collinear EOG channels reaches ~1e-6, i.e. ~1e-12 of shared variance).
  V3 B recomputed from the caches equals the R11 "fts" arm (< 1e-12).
  V4 Determinism: a second run with warm caches gives a byte-identical CSV; the IV-2b subject-1 caches
     are deleted and retrained, and must be byte-identical.

    .venv\\Scripts\\python experiments\\r14_eog_gate.py     # ~40 min GPU (108 fits)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import setup_threads  # noqa: E402

setup_threads(8)

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from moabb.datasets import BNCI2014_001, BNCI2014_004  # noqa: E402

from experiments.r3_convnet_session_norm import CFG, WIDE, align_per_session  # noqa: E402
from experiments.r7_self_training_iv2b import CLASSES as IV2B_CLASSES  # noqa: E402
from experiments.r7_self_training_iv2b import EVAL_RECORDINGS, TRAIN_RECORDINGS, align_recordings  # noqa: E402
from experiments.r10_round2_stack_position import align_by_recording, tangent, ts_lr  # noqa: E402
from experiments.r11_round3_diverse_members import CACHE, contrast_line, rec_acc  # noqa: E402
from src.adaptation.augmentation import make_augmentation  # noqa: E402
from src.adaptation.convnet import ShallowConvNet, encode_labels, predict_proba, set_deterministic, train_model  # noqa: E402
from src.adaptation.self_training import one_hot  # noqa: E402
from src.config import FMAX, FMIN, IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.datasets import load_iv2a_subject_window, load_iv2b_subject_window  # noqa: E402
from src.evaluate import sign_flip_permutation_test  # noqa: E402
from src.gpu_health import wait_until_healthy  # noqa: E402

mne.set_log_level("ERROR")
OUT_CSV = RESULTS_DIR / "r14_eog_gate.csv"
PRED_DIR = CACHE / "r14_preds"
NARROW = dict(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
GATE_PP = 1.0
MOABB_SCALE = 1e6  # the MOABB paradigm returns microvolts; raw MNE epochs are in volts
FAMILIES = {"base": (0, 1, 2), "fts": (0, 1)}
TOL_V1, TOL_V2, TOL_CORR = 1e-6, 1e-12, 1e-4
CHANCE = {"IV-2a": 0.25, "IV-2b": 0.5}


def raw_epochs(dataset, subject: int, bands: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """{band: {'eeg': X, 'eog': X}}, labels, session, in MOABB order, cut like the MOABB paradigm."""
    inv = {v: k for k, v in dataset.event_id.items()}
    out = {b: {"eeg": [], "eog": []} for b in bands}
    ys, sess = [], []
    for sname, runs in dataset.get_data(subjects=[subject])[subject].items():
        for raw in runs.values():
            ev = mne.find_events(raw, shortest_event=0, verbose=False)
            ev = ev[np.isin(ev[:, 2], list(inv))]
            t0 = dataset.interval[0]
            for b, w in bands.items():
                filt = raw.copy().filter(w["fmin"], w["fmax"], method="iir", picks=["eeg", "eog"])
                for kind in ("eeg", "eog"):
                    ep = mne.Epochs(filt, ev, tmin=t0 + w["tmin"], tmax=t0 + w["tmax"], picks=kind,
                                    baseline=None, preload=True)
                    out[b][kind].append(ep.get_data(copy=False))
            ys += [inv[c] for c in ev[:, 2]]
            sess += [sname] * len(ev)
    arrays = {b: {k: MOABB_SCALE * np.concatenate(v) for k, v in d.items()} for b, d in out.items()}
    return arrays, np.array(ys), np.array(sess)


def check_v1(tag: str, mine: np.ndarray, harness: np.ndarray) -> None:
    diff = np.abs(mine - harness).max() if mine.shape == harness.shape else np.inf
    if not diff <= TOL_V1 * np.abs(harness).max():
        raise AssertionError(f"V1: {tag} raw-epoch EEG differs from the harness loader (max diff {diff:.3e})")


def eog_regress(eeg: np.ndarray, eog: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, float]:
    """EEG minus its least-squares projection on EOG, coefficients from the training trials only."""
    a = eog[train].transpose(0, 2, 1).reshape(-1, eog.shape[1])
    b = eeg[train].transpose(0, 2, 1).reshape(-1, eeg.shape[1])
    coef = np.linalg.lstsq(a, b, rcond=None)[0]
    clean = eeg - np.einsum("nct,ce->net", eog, coef)
    resid = clean[train].transpose(0, 2, 1).reshape(-1, eeg.shape[1])
    # The fit has no intercept, so least squares guarantees UNCENTRED orthogonality (resid . eog = 0);
    # a centred correlation is not zero when epochs keep a DC offset (e.g. a 0.5 Hz high-pass).
    corr = (resid.T @ a) / np.outer(np.linalg.norm(resid, axis=0), np.linalg.norm(a, axis=0))
    if not np.abs(corr).max() < TOL_CORR:
        raise AssertionError(f"V2: regression residual still correlates with EOG ({np.abs(corr).max():.2e})")
    removed = 1 - clean[~train].var() / eeg[~train].var()
    return clean, float(removed)


def fit_predict(make_model, Xt, yt, Xe, seed: int, aug: str | None) -> np.ndarray:
    wait_until_healthy()
    set_deterministic(seed)
    augment = None if aug is None else make_augmentation(aug, seed)
    model = train_model(make_model(), Xt, yt, CFG, seed=seed, augment=augment)  # built after seeding (R8/R9)
    p = predict_proba(model, Xe)
    del model
    torch.cuda.empty_cache()
    return p


def cached(path: Path, compute) -> np.ndarray:
    if path.exists():
        return np.load(path)
    p = compute()
    np.save(path, p)
    return p


def excess_agreement(p_cnn: np.ndarray, p_eog: np.ndarray, y: np.ndarray) -> float:
    a, b = p_cnn.argmax(1), p_eog.argmax(1)
    wrong = a != y
    if not wrong.any():
        return 0.0
    expected = [np.mean(b[y == y[i]] == a[i]) for i in np.flatnonzero(wrong)]
    return float(np.mean(a[wrong] == b[wrong]) - np.mean(expected))


def score(rows, ds, s, y, rec, raw: dict, clean: dict, eog: dict, removed: float) -> None:
    def add(arm, value):
        rows.append({"dataset": ds, "subject": s, "arm": arm, "acc": value})

    for tag, m in (("", raw), ("_clean", clean)):
        add(f"B{tag}", rec_acc((m["cnn3"] + m["tslr"] + m["fts2"]) / 3, y, rec))
        add(f"r10{tag}", rec_acc(0.5 * m["cnn3"] + 0.5 * m["tslr"], y, rec))
        for k, p in m.items():
            add(f"m_{k}{tag}", rec_acc(p, y, rec))
    for k, p in eog.items():
        add(k, rec_acc(p, y, rec))
    add("excess_agree", excess_agreement(raw["cnn3"], eog["eog_cnn"], y))
    add("var_removed", removed)


def tslr_from(Xt, yt, Xe) -> np.ndarray:
    return ts_lr(*_tangent_pair(Xt, yt, Xe))


def _tangent_pair(Xt, yt, Xe):
    Ft, Fe = tangent(Xt, Xe)
    return Ft, yt, Fe


def run_iv2a(s: int, rows: list, r11_tslr: pd.Series, device) -> None:
    arr, ys, sess = raw_epochs(BNCI2014_001(), s, {"wide": WIDE, "narrow": NARROW})
    w = load_iv2a_subject_window(s, **WIDE)
    d = load_iv2a_subject_window(s, **NARROW)
    t = sess == "0train"
    if not (np.array_equal(ys[t], d.y_calib) and np.array_equal(ys[~t], d.y_eval)
            and np.array_equal(d.y_eval, w.y_eval) and np.array_equal(d.y_calib, w.y_calib)):
        raise AssertionError(f"V1: IV-2a s{s} trial order differs")
    check_v1(f"IV-2a s{s} wide", np.concatenate([w.X_calib, w.X_eval]), arr["wide"]["eeg"])
    check_v1(f"IV-2a s{s} narrow", np.concatenate([d.X_calib, d.X_eval]), arr["narrow"]["eeg"])
    yt, ye = encode_labels(d.y_calib, IV2A_CLASSES), encode_labels(d.y_eval, IV2A_CLASSES)
    clean_w, removed = eog_regress(arr["wide"]["eeg"], arr["wide"]["eog"], t)
    clean_n, _ = eog_regress(arr["narrow"]["eeg"], arr["narrow"]["eog"], t)

    raw = {"tslr": ra_tslr(np.concatenate([d.X_calib, d.X_eval]), t, yt)}
    if not abs(rec_acc(raw["tslr"], ye, np.zeros(len(ye))) - r11_tslr[("IV-2a", s)]) < TOL_V2:
        raise AssertionError(f"V2: IV-2a s{s} TS+LR does not reproduce R11")
    raw["cnn3"] = np.mean([np.load(CACHE / "r3_preds" / f"s{s}_cnn_ra_seed{k}.npy") for k in range(3)], 0)
    raw["fts2"] = np.mean([np.load(CACHE / "r8_preds" / f"s{s}_ftsurrogate_seed{k}.npy") for k in range(2)], 0)

    def cnn(X, n_chans, tag, k, aug):
        Xc, Xe = align_per_session(X[t], X[~t])
        tens = [torch.as_tensor(a.astype(np.float32), device=device) for a in (Xc, Xe)]
        yc = torch.as_tensor(yt, device=device)
        return cached(PRED_DIR / f"iv2a_s{s}_{tag}_seed{k}.npy",
                      lambda: fit_predict(lambda: ShallowConvNet(n_chans, 4, X.shape[2]).to(device),
                                          tens[0], yc, tens[1], k, aug))

    clean = {"tslr": ra_tslr(clean_n, t, yt),
             "cnn3": np.mean([cnn(clean_w, 22, "clean_base", k, None) for k in FAMILIES["base"]], 0),
             "fts2": np.mean([cnn(clean_w, 22, "clean_fts", k, "ftsurrogate") for k in FAMILIES["fts"]], 0)}
    eog = {"eog_tslr": ra_tslr(arr["narrow"]["eog"], t, yt),
           "eog_tslr_wide": ra_tslr(arr["wide"]["eog"], t, yt),
           "eog_cnn": cnn(arr["wide"]["eog"], 3, "eog_cnn", 0, None)}
    score(rows, "IV-2a", s, ye, np.zeros(len(ye)), raw, clean, eog, removed)


def ra_tslr(X: np.ndarray, t: np.ndarray, yt: np.ndarray) -> np.ndarray:
    """IV-2a TS+LR as R10/R11: per-session RA (train = t, test = ~t), tangent space, LR."""
    Xt, Xe = align_per_session(X[t], X[~t])
    return tslr_from(Xt, yt, Xe)


def run_iv2b(s: int, rows: list, r11_tslr: pd.Series, device) -> None:
    arr, ys, sess = raw_epochs(BNCI2014_004(), s, {"wide": WIDE, "narrow": NARROW})
    w = load_iv2b_subject_window(s, **WIDE)
    d = load_iv2b_subject_window(s, **NARROW)
    if not (np.array_equal(ys, d.y) and np.array_equal(sess, d.session)
            and np.array_equal(d.y, w.y) and np.array_equal(d.session, w.session)):
        raise AssertionError(f"V1: IV-2b s{s} label/recording order differs")
    check_v1(f"IV-2b s{s} wide", w.X, arr["wide"]["eeg"])
    check_v1(f"IV-2b s{s} narrow", d.X, arr["narrow"]["eeg"])
    y = encode_labels(d.y, IV2B_CLASSES)
    tr = np.isin(d.session, TRAIN_RECORDINGS)
    ev = np.concatenate([np.flatnonzero(d.session == r) for r in EVAL_RECORDINGS])
    rec = d.session[ev]
    clean_w, removed = eog_regress(arr["wide"]["eeg"], arr["wide"]["eog"], tr)
    clean_n, _ = eog_regress(arr["narrow"]["eeg"], arr["narrow"]["eog"], tr)

    def tslr(X):
        A = align_by_recording(X, d.session)
        return tslr_from(A[tr], y[tr], A[ev])

    raw = {"tslr": tslr(d.X)}
    if not abs(rec_acc(raw["tslr"], y[ev], rec) - r11_tslr[("IV-2b", s)]) < TOL_V2:
        raise AssertionError(f"V2: IV-2b s{s} TS+LR does not reproduce R11")
    raw["cnn3"] = np.mean([np.load(CACHE / "r7_preds" / f"s{s}_teacher_seed{k}.npy") for k in range(3)], 0)
    raw["fts2"] = np.mean([np.load(CACHE / "r9_preds" / f"s{s}_ftsurrogate_seed{k}.npy") for k in range(2)], 0)

    def cnn(X, n_chans, tag, k, aug):
        A = align_recordings(X, d.session)
        Xt = torch.as_tensor(A[tr].astype(np.float32), device=device)
        Yt = torch.as_tensor(one_hot(y[tr], 2).astype(np.float32), device=device)
        Xe = torch.as_tensor(A[ev].astype(np.float32), device=device)
        return cached(PRED_DIR / f"iv2b_s{s}_{tag}_seed{k}.npy",
                      lambda: fit_predict(lambda: ShallowConvNet(n_chans, 2, X.shape[2]).to(device),
                                          Xt, Yt, Xe, k, aug))

    clean = {"tslr": tslr(clean_n),
             "cnn3": np.mean([cnn(clean_w, 3, "clean_base", k, None) for k in FAMILIES["base"]], 0),
             "fts2": np.mean([cnn(clean_w, 3, "clean_fts", k, "ftsurrogate") for k in FAMILIES["fts"]], 0)}
    eog = {"eog_tslr": tslr(arr["narrow"]["eog"]),
           "eog_tslr_wide": tslr(arr["wide"]["eog"]),
           "eog_cnn": cnn(arr["wide"]["eog"], 3, "eog_cnn", 0, None)}
    score(rows, "IV-2b", s, y[ev], rec, raw, clean, eog, removed)


def check_v3(df: pd.DataFrame) -> None:
    ref = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv")
    ref = ref[ref.arm == "fts"].set_index(["dataset", "subject"]).acc
    got = df[df.arm == "B"].set_index(["dataset", "subject"]).acc
    diff = (got - ref.reindex(got.index)).abs().max()
    if len(got) != 18 or not diff < TOL_V2:
        raise AssertionError(f"V3: R11 best not reproduced (max diff {diff})")
    print("V1, V2, V3 passed: arrays, TS+LR and the R11 best (B) reproduced for all 18 subjects")


def summarise(df: pd.DataFrame) -> None:
    piv = df.pivot_table(index=["dataset", "subject"], columns="arm", values="acc")
    acc = 100 * piv.drop(columns=["excess_agree", "var_removed"])
    print("\n=== R14 EOG gate (IV-2a + IV-2b, 18 subjects) ===")
    print(acc.groupby("dataset").mean().round(2).T.to_string())
    d = acc["B"] - acc["B_clean"]
    p = sign_flip_permutation_test(d.to_numpy(), "greater")
    fail = d.mean() >= GATE_PP and p < 0.05
    print("\nPRIMARY (B - B_clean, ocular share upper bound):")
    print(contrast_line("B - B_clean", d, p) + f"  -> {'OCULAR-MATERIAL (FAIL)' if fail else 'PASS'}")
    print("SECONDARY:")
    s1 = acc["B_clean"] - acc["r10_clean"]
    print(contrast_line("S1 B_clean-r10_clean", s1, sign_flip_permutation_test(s1.to_numpy(), "greater")))
    for k in ("eog_tslr", "eog_tslr_wide", "eog_cnn"):
        chance = 100 * piv.index.get_level_values("dataset").map(CHANCE).to_numpy()
        a = acc[k] - chance
        print(contrast_line(f"S2 {k}-chance", a, sign_flip_permutation_test(a.to_numpy(), "greater")))
    ex = 100 * piv["excess_agree"]
    print(contrast_line("S3 excess agree", ex, sign_flip_permutation_test(ex.to_numpy(), "greater")))
    for m in ("cnn3", "tslr", "fts2"):
        dm = acc[f"m_{m}"] - acc[f"m_{m}_clean"]
        print(contrast_line(f"S4 {m} raw-clean", dm, sign_flip_permutation_test(dm.to_numpy(), "greater")))
    print("S4 EEG variance removed on E (WIDE): "
          + ", ".join(f"{k} {100 * v:.1f}%" for k, v in piv["var_removed"].groupby("dataset").mean().items()))
    print(f"\nDECISION: {'B_clean replaces B as the reference' if fail else 'B stays the reference'}")


def main() -> int:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    r11 = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv")
    r11_tslr = r11[r11.arm == "m_tslr"].set_index(["dataset", "subject"]).acc
    rows: list[dict] = []
    for s in IV2A_SUBJECTS:
        run_iv2a(s, rows, r11_tslr, device)
        run_iv2b(s, rows, r11_tslr, device)
        print(f"  s{s} done", flush=True)
    df = pd.DataFrame(rows)
    check_v3(df)
    df.to_csv(OUT_CSV, index=False)
    summarise(df)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
