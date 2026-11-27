"""Round "alien": cross-field ideas, kept only if they help on Lee2019 AND do not lose on IV-2a. PRE-REGISTERED.

Committed BEFORE any arm is scored. The plateaued candidate failed IV-2a (-4.75 pp) because it gave the pooled decoder the same fixed
weight for every subject. Each idea below, taken from outside motor-imagery decoding, lets the data of ONE subject decide how much to
trust a source, with no test label. CPU only; CNN3 / FTS2 cached; Lee2019 HOLDOUT is not read; Zhou2020 stays sealed.

KEY FACT USED. The zero-shot pooled decoder never sees the subject's labels, so its score on the subject's own labelled session 1 is an
honest estimate of its worth for that subject; the own TS+LR member is scored on session 1 by 5-fold out-of-fold predictions.

IDEAS (all constants fixed here, none tuned). "slot" replaces R11's TS+LR member; "4th" adds a member with weight up to 2.
  gate_slot / gate_double / gate_4th   (econometrics, linear opinion pool of forecasts) a = argmax over {0, .05 .. 1} of the session-1
      log-likelihood of a*pool + (1-a)*own_oof; T = a*pool_test + (1-a)*own_test. slot: (CNN + T + FTS)/3; double: (CNN + 2T + FTS)/4;
      4th: (CNN + own + FTS + 2a*pool)/(3 + 2a).
  experts_slot / experts_4th   (Bayesian model averaging; mixture of experts) one LR per library subject; weight proportional to
      exp(10 * mean session-1 log-likelihood); the mixture E takes the place of pool in gate_slot / gate_4th (E also never saw the
      subject's labels, so its gate is honest apart from the weighting on the same trials, stated as a mild optimism of the gate).
  dawid_skene   (medical statistics: raters without a gold standard) online EM over the incumbent's members, CAUSAL: trial t uses trials
      <= t only; each member's confusion matrix has a Dirichlet prior of 20 pseudo-trials per class with 0.7 on the diagonal; 10 EM
      iterations; output = posterior of trial t.
  spectral_meta   (statistics: ranking classifiers without labels, Parisi et al. 2014) CAUSAL: weights proportional to the leading
      eigenvector (clipped at 0) of the covariance of the members' centred probability vectors over trials <= t, the diagonal replaced by
      each column's largest off-diagonal value; equal incumbent weights until 20 trials.
  graph_diffusion   (network science: label spreading) CAUSAL: p_t = 0.7 p_t + 0.3 * similarity-weighted mean of the incumbent
      probabilities of the 7 nearest EARLIER test trials (tangent-space distance, Gaussian weights, median distance as width).
COHORTS. Lee2019 TUNE (27 subjects, library = the other 26) and IV-2a (9, library = the other 8). Per-session alignment as R11.
KEEP iff Lee TUNE mean >= incumbent + 0.3 pp with more subjects up than down AND IV-2a mean >= incumbent - 0.25 pp. Start = R11 (must
reproduce stored accuracies). Rounds over the idea list on the current incumbent; stop after a round that keeps nothing (max 3 rounds).
HONESTY. IV-2a now takes part in selection and is no longer an independent check; the Lee holdout is spent; whatever survives is a
development-set result with no untouched confirmation short of Zhou2020, which stays sealed.

    .venv\\Scripts\\python autoresearch\\alien.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

import prepare  # noqa: E402
from experiments.n7_drift_forecast_screen import tangent_I  # noqa: E402
from experiments.n8_pooled_decoder_screen import pool_proba  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.pipeline import align_by_group, load_subject_data, tslr_member  # noqa: E402

GRID = np.linspace(0, 1, 21)
BMA_TEMP, DS_PSEUDO, DS_DIAG, DS_ITER, SML_WARM, KNN, BETA = 10.0, 20.0, 0.7, 10, 20, 7, 0.3
MIN_GAIN, IV_TOL, MAX_ROUNDS = 0.3, 0.25, 3
IDEAS = ("gate_slot", "gate_double", "gate_4th", "experts_slot", "experts_4th", "dawid_skene", "spectral_meta", "graph_diffusion")


def loglik(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1)).mean())


def base_subject(data, cnn3: np.ndarray, fts2: np.ndarray) -> dict:
    A = align_by_group(data.X_tslr, data.groups)
    F, tr = tangent_I(A, np.eye(A.shape[1])), data.train
    return {"F_tr": F[tr], "y_tr": data.y[tr], "F_te": F[~tr], "y_te": data.y[~tr], "cnn": cnn3, "fts": fts2, "own": tslr_member(data)}


def add_sources(S: dict) -> None:
    for s, v in S.items():
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        lr = LogisticRegression(C=1.0, max_iter=5000)
        v["own_oof"] = cross_val_predict(lr, v["F_tr"], v["y_tr"], cv=cv, method="predict_proba")
        n = len(v["y_tr"])
        both = np.vstack([v["F_tr"], v["F_te"]])
        p = pool_proba(S, {s}, None, 0, {"F_te": both})
        v["pool_s1"], v["pool"] = p[:n], p[n:]
        experts = []
        for o, u in S.items():
            if o != s:
                m = LogisticRegression(C=1.0, max_iter=2000)
                m.fit(np.vstack([u["F_tr"], u["F_te"]]), np.r_[u["y_tr"], u["y_te"]])
                experts.append(m.predict_proba(both))
        score = np.array([loglik(e[:n], v["y_tr"]) for e in experts])
        pi = np.exp(BMA_TEMP * (score - score.max()))
        mix = np.einsum("k,knc->nc", pi / pi.sum(), np.stack(experts))
        v["exp_s1"], v["exp"] = mix[:n], mix[n:]
        print(f"  sources ready: subject {s}", flush=True)


def gate(v: dict, src: str) -> float:
    s1 = v[f"{src}_s1"]
    return float(GRID[int(np.argmax([loglik(a * s1 + (1 - a) * v["own_oof"], v["y_tr"]) for a in GRID]))])


def mix_members(members: list) -> np.ndarray:
    return sum(w * p for w, p in members) / sum(w for w, _ in members)


def dawid_skene(probs: np.ndarray) -> np.ndarray:
    """probs: (members, trials, classes). Causal soft Dawid-Skene with a diagonal Dirichlet prior."""
    m, n, k = probs.shape
    prior = DS_PSEUDO * (np.full((k, k), (1 - DS_DIAG) / (k - 1)) + np.eye(k) * (DS_DIAG - (1 - DS_DIAG) / (k - 1)))
    out, start = np.empty((n, k)), probs.mean(0)
    for t in range(n):
        seen, q = probs[:, : t + 1], start[: t + 1]
        for _ in range(DS_ITER):
            conf = prior[None] + np.einsum("nk,mnl->mkl", q, seen)  # member, true class, predicted class
            conf /= conf.sum(2, keepdims=True)
            log_post = np.einsum("mnl,mkl->nk", seen, np.log(conf))
            q = np.exp(log_post - log_post.max(1, keepdims=True))
            q /= q.sum(1, keepdims=True)
        out[t] = q[t]
    return out


def spectral_meta(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    m, n, k = probs.shape
    out = np.empty((n, k))
    for t in range(n):
        w = weights
        if t + 1 >= SML_WARM:
            Z = probs[:, : t + 1] - probs[:, : t + 1].mean(1, keepdims=True)
            C = np.cov(Z.reshape(m, -1))
            off = C - np.diag(np.diag(C))
            C = off + np.diag(off.max(1))
            lead = np.linalg.eigh(C)[1][:, -1]
            lead = np.clip(lead * np.sign(lead.sum() or 1.0), 0, None)
            if lead.sum() > 0:
                w = lead
        out[t] = np.einsum("m,mc->c", w / w.sum(), probs[:, t])
    return out


def graph_diffusion(p: np.ndarray, F: np.ndarray) -> np.ndarray:
    out = p.copy()
    for t in range(KNN, len(p)):
        d = np.linalg.norm(F[:t] - F[t], axis=1)
        nn = np.argsort(d)[:KNN]
        w = np.exp(-((d[nn] / (np.median(d) + 1e-12)) ** 2))
        out[t] = (1 - BETA) * p[t] + BETA * (w @ p[nn]) / w.sum()
    return out


def predict(v: dict, stack: list) -> np.ndarray:
    """Apply the kept ideas in order. State: weighted member list, then post-processing of the combined probabilities."""
    members = [(1.0, v["cnn"]), (1.0, v["own"]), (1.0, v["fts"])]
    combine, post = None, []
    for idea in stack:
        if idea.startswith(("gate", "experts")):
            src = "pool" if idea.startswith("gate") else "exp"
            a = gate(v, src)
            T = a * v[src] + (1 - a) * v["own"]
            if idea.endswith("slot"):
                members = [(1.0, v["cnn"]), (1.0, T), (1.0, v["fts"])]
            elif idea.endswith("double"):
                members = [(1.0, v["cnn"]), (2.0, T), (1.0, v["fts"])]
            elif a > 0:  # 4th: added to whatever is already there
                members = [*members, (2.0 * a, v[src])]
        elif idea in ("dawid_skene", "spectral_meta"):
            combine = idea
        else:
            post.append(idea)
    probs = np.stack([p for _, p in members])
    if combine == "dawid_skene":
        out = dawid_skene(probs)
    elif combine == "spectral_meta":
        out = spectral_meta(probs, np.array([w for w, _ in members]))
    else:
        out = mix_members(members)
    for _ in post:
        out = graph_diffusion(out, v["F_te"])
    return out


def score(S: dict, stack: list) -> pd.Series:
    return pd.Series({s: 100 * float((predict(v, stack).argmax(1) == v["y_te"]).mean()) for s, v in S.items()})


def build() -> tuple[dict, dict]:
    lee = {}
    for s in prepare.TUNE:
        r = prepare.read(s)
        lee[s] = base_subject(r["d"], r["cnn3"], r["fts2"])
    iv = {}
    for s in range(1, 10):
        data, _ = load_subject_data("IV-2a", s)
        c = HERE.parent / "data" / "cache"
        cnn = np.mean([np.load(c / "r3_preds" / f"s{s}_cnn_ra_seed{k}.npy") for k in range(3)], 0)
        fts = np.mean([np.load(c / "r8_preds" / f"s{s}_ftsurrogate_seed{k}.npy") for k in range(2)], 0)
        iv[s] = base_subject(data, cnn, fts)
    add_sources(lee)
    add_sources(iv)
    return lee, iv


def main() -> int:
    lee, iv = build()
    stack: list = []
    best_l, best_i = score(lee, stack), score(iv, stack)
    d2 = pd.read_csv(RESULTS_DIR / "d2_lee2019_dev.csv").set_index("subject")
    r11 = pd.read_csv(RESULTS_DIR / "r11_round3_diverse_members.csv")
    ref = r11[(r11.dataset == "IV-2a") & (r11.arm == "fts")].set_index("subject").acc
    if (best_l - 100 * d2.loc[prepare.TUNE, "r11"]).abs().max() > 1e-9 or (best_i - 100 * ref).abs().max() > 1e-9:
        raise AssertionError("start is not R11")
    rows = [{"round": 0, "idea": "R11", "lee": best_l.mean(), "iv2a": best_i.mean(), "keep": True}]
    print(f"R11: Lee TUNE {best_l.mean():.3f}, IV-2a {best_i.mean():.3f}", flush=True)
    for rnd in range(1, MAX_ROUNDS + 1):
        kept = 0
        for idea in IDEAS:
            if idea in stack:
                continue
            L, I = score(lee, [*stack, idea]), score(iv, [*stack, idea])
            dl, di = L - best_l, I - best_i
            keep = bool(dl.mean() >= MIN_GAIN and (dl > 0).sum() > (dl < 0).sum() and di.mean() >= -IV_TOL)
            rows.append({"round": rnd, "idea": idea, "lee": L.mean(), "iv2a": I.mean(), "keep": keep,
                         "lee_delta": dl.mean(), "lee_up": int((dl > 0).sum()), "lee_down": int((dl < 0).sum()),
                         "iv_delta": di.mean(), "iv_up": int((di > 0).sum()), "iv_down": int((di < 0).sum())})
            print(f"round {rnd} {idea:16s} Lee {dl.mean():+.2f} ({(dl > 0).sum()}/{(dl < 0).sum()}) IV-2a {di.mean():+.2f} "
                  f"({(di > 0).sum()}/{(di < 0).sum()}) {'KEEP' if keep else 'discard'}", flush=True)
            pd.DataFrame(rows).to_csv(HERE / "alien_results.csv", index=False)
            if keep:
                stack, best_l, best_i, kept = [*stack, idea], L, I, kept + 1
        if kept == 0:
            break
    gates = pd.DataFrame({"cohort": ["lee"] * len(lee) + ["iv2a"] * len(iv),
                          "a_pool": [gate(v, "pool") for v in [*lee.values(), *iv.values()]]})
    print("gate weight on the pooled decoder, mean by cohort:", gates.groupby("cohort").a_pool.mean().round(2).to_dict())
    print(f"DONE stack {stack}: Lee TUNE {best_l.mean():.3f}, IV-2a {best_i.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
