"""The ONLY file the loop edits (it rewrites the CFG line). With the initial CFG this is exactly R11."""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from src.pipeline import SubjectData, align_by_group, tslr_member

CFG = {'w_pool': 2.0, 'pool_w': 0.0, 'w_own': 0.0}

DEFAULT = {"w_cnn": 1.0, "w_own": 1.0, "w_fts": 1.0, "w_pool": 0.0, "w_pool2": 0.0, "own_mode": "r11", "own_C": 1.0, "lam": 0.0,
           "own_feats": [None], "pool_feats": [None], "pool2_feats": [((8, 13), (0.5, 2.5)), ((13, 30), (0.5, 2.5))],
           "pool_w": 20.0, "pool_C": 1.0, "cov": "oas", "scale": False, "align": "riemann", "lib_session": None,
           "pool2_C": None, "pool2_clf": None, "pool_clf": "lr", "knn": 0, "geo": False, "balance": False}
_CACHE: dict = {}


def feats(key: tuple, s: dict, spec: list) -> np.ndarray:
    """Per-session aligned OAS covariance -> tangent space at I, concatenated over ``spec``. None = cached 8-30 Hz 0.5-2.5 s;
    otherwise ((low, high), (t0, t1)) cut from the 4-38 Hz 0-4 s signal (250 Hz)."""
    out = []
    for item in spec:
        cfg = {**DEFAULT, **CFG}
        k = (*key, item, cfg["cov"], cfg["scale"], cfg["align"])
        if k not in _CACHE:
            if item is None:
                X = s["X_ts"]
            else:
                band, (t0, t1) = item
                sos = butter(4, band, "bandpass", fs=250, output="sos")
                X = sosfiltfilt(sos, s["X_cnn"], axis=-1)[:, :, int(t0 * 250):int(t1 * 250)].astype(np.float32)
            if cfg["align"] == "euclid":
                C = Covariances("oas").fit_transform(X).mean(0)
                val, vec = np.linalg.eigh(C)
                A = np.einsum("cd,ndt->nct", (vec / np.sqrt(val)) @ vec.T, X)
            else:
                A = align_by_group(X, np.zeros(len(X)))
            ts = TangentSpace(metric="riemann").fit(np.eye(A.shape[1])[None])
            F = ts.transform(Covariances(cfg["cov"]).fit_transform(A))
            if cfg["scale"]:  # unit mean squared distance to I per session (label-free dispersion matching)
                F = F / np.sqrt((F**2).sum(1).mean())
            _CACHE[k] = F
        out.append(_CACHE[k])
    return np.hstack(out)


def means(F: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([F[y == k].mean(0) for k in (0, 1)])


def pooled(cfg: dict, own: dict, test: dict, library: list, spec: list) -> np.ndarray:
    Fo, Fe = feats((own["id"], 0), own, spec), feats((own["id"], 1), test, spec)
    lib = library
    if cfg["knn"]:
        m = means(Fo, own["y"])

        def distance(o: dict) -> float:
            first = o["sessions"][0]
            return float(((means(feats((o["id"], 0), first, spec), first["y"]) - m) ** 2).sum())

        lib = sorted(library, key=distance)[: cfg["knn"]]
    pairs = [(o, j, ses) for o in lib for j, ses in enumerate(o["sessions"]) if cfg["lib_session"] in (None, j)]
    X = [feats((o["id"], j), ses, spec) for o, j, ses in pairs]
    y = [ses["y"] for _, _, ses in pairs]
    sw = [np.ones(len(v)) for v in y]
    if cfg["pool_w"] > 0:
        X, y, sw = X + [Fo], y + [own["y"]], sw + [np.full(len(own["y"]), cfg["pool_w"])]
    X, y, sw = np.vstack(X), np.concatenate(y), np.concatenate(sw)
    if cfg["pool_clf"] == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y).predict_proba(Fe)
    return LogisticRegression(C=cfg["pool_C"], max_iter=2000).fit(X, y, sample_weight=sw).predict_proba(Fe)


def own_member(cfg: dict, own: dict, test: dict, library: list) -> np.ndarray:
    if cfg["own_mode"] == "r11":
        n, m = len(own["y"]), len(test["X_ts"])
        data = SubjectData(X_cnn=np.concatenate([own["X_cnn"], test["X_cnn"]]),
                           X_tslr=np.concatenate([own["X_ts"], test["X_ts"]]), y=np.r_[own["y"], np.zeros(m, int)],
                           groups=np.r_[np.zeros(n), np.ones(m)], train=np.r_[np.ones(n, bool), np.zeros(m, bool)])
        return tslr_member(data)
    spec = cfg["own_feats"]
    Fo, Fe = feats((own["id"], 0), own, spec), feats((own["id"], 1), test, spec)
    if cfg["lam"]:
        mu = np.mean([means(feats((o["id"], 0), o["sessions"][0], spec), o["sessions"][0]["y"]) for o in library], 0)
        Fo = Fo + (cfg["lam"] * (mu - means(Fo, own["y"])))[own["y"]]
    return LogisticRegression(C=cfg["own_C"], max_iter=5000).fit(Fo, own["y"]).predict_proba(Fe)


def predict(own: dict, test: dict, library: list) -> np.ndarray:
    cfg = {**DEFAULT, **CFG}
    parts = [(cfg["w_cnn"], test["cnn3"]), (cfg["w_fts"], test["fts2"])]
    if cfg["w_own"]:
        parts.append((cfg["w_own"], own_member(cfg, own, test, library)))
    if cfg["w_pool"]:
        parts.append((cfg["w_pool"], pooled(cfg, own, test, library, cfg["pool_feats"])))
    if cfg["w_pool2"]:
        second = {**cfg, "pool_C": cfg["pool2_C"] or cfg["pool_C"], "pool_clf": cfg["pool2_clf"] or cfg["pool_clf"]}
        parts.append((cfg["w_pool2"], pooled(second, own, test, library, cfg["pool2_feats"])))
    total = sum(w for w, _ in parts)
    if cfg["geo"]:
        p = np.exp(sum(w * np.log(np.clip(q, 1e-12, None)) for w, q in parts) / total)
    else:
        p = sum(w * q for w, q in parts) / total
    p = p / p.sum(1, keepdims=True)
    if cfg["balance"]:
        for _ in range(200):
            p = p / p.sum(0, keepdims=True)
            p = p / p.sum(1, keepdims=True)
    return p
