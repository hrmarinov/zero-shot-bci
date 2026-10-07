"""Characterization 01 - per-subject signal diagnostics on BCI IV-2a.

Motivating question. The project's best method (Riemannian alignment + CSP+LDA,
Phase 2) reaches 66.9% mean cross-session accuracy over the 9 IV-2a subjects,
but the per-subject spread is bimodal: {1, 3, 7, 8, 9} sit at 78-84%, subject 4
at 60.8%, and {2, 5, 6} at 42-49% (near the 25% chance level). No per-subject
diagnostic has ever been run. The open question this script feeds:

    Are subjects 2, 5 and 6 bad because their signal is genuinely weak
    (signal-limited: no method will fix them, the honest answer is to predict
    and route them), or because the pipeline's *fixed* choices - 8-30 Hz band,
    0.5-2.5 s window, 8 CSP components - are wrong for them
    (configuration-limited: they are recoverable)?

This script computes, for all 9 subjects x both sessions, four families of
measurements that discriminate between those hypotheses:

  (a) acquisition/quality basics   - trial counts, per-channel peak-to-peak
      amplitude, kurtosis, fraction of trials with a large max amplitude.
  (b) spectra                      - Welch PSD per channel on the wide
      (-1..+4 s, 1-40 Hz) epochs, relative band power, the 6-14 Hz peak
      frequency after removing the aperiodic (1/f) background, mu/beta ratio.
  (c) ERD/ERS                      - cue-period band power relative to the
      pre-cue baseline, per channel per class, in the mu (8-13) and beta
      (13-30) bands; and the *spatial* pattern: left hand must produce
      stronger ERD over the right hemisphere (C4/CP4) than over the left, and
      vice versa. A subject with no lateralised ERD has no decodable
      sensorimotor-rhythm signal; a subject with strong lateralised ERD but
      low accuracy has a pipeline/configuration problem. Because a laterality
      index can be non-zero by chance, the same computation is also run under
      1000 label permutations to give it a calibrated null.
  (d) cross-session drift          - affine-invariant Riemannian distance
      between the calibration and evaluation sessions' mean trial covariances
      (pyriemann, estimator="oas", matching RiemannianAlignment), against the
      between-subject distance for context.

Two estimator choices are load-bearing and were verified directly rather than
assumed (see docs/characterization_iv2a.md, "Estimator checks"):

  * ERD is the ratio of class-mean powers, (mean_t P_cue - mean_t P_base) /
    mean_t P_base - NOT the mean over trials of the per-trial ratio. The
    per-trial ratio is heavy-tailed (mu power over a 2 s window has few
    degrees of freedom) and its trial mean is biased upward by ~+0.5 in this
    data on *every* channel of subject 1, which would have washed out all
    spatial structure while looking like a plausible "global ERS".
  * The 6-14 Hz peak is found after subtracting a fitted aperiodic background
    (log10 PSD ~ a + b log10 f, fitted outside 6-14 Hz). Without that
    correction the argmax is pinned at the 6 Hz edge of the search band for
    every subject, because the raw PSD decreases monotonically with frequency.

Determinism: the only random numbers are the seeded permutation tests
(np.random.default_rng(RANDOM_STATE)), so a rerun reproduces the CSVs. Threads
are pinned to 4 (a concurrent moderate job is running on this machine - see
docs/compute_environment.md sections 1 and 4). One subject is held in memory at
a time; only the 22x22 mean covariances (9 x 2 x ~4 kB) persist across
subjects.

Outputs:
  results/char_iv2a_signal_characteristics.csv  9 x 2 x 22 = 396 rows
  results/char_iv2a_subject_summary.csv         9 rows

It also prints descriptive group comparisons (WEAK {2,5,6} vs STRONG
{1,3,7,8,9}) with effect sizes. n=3 vs n=5 gives a Mann-Whitney test almost no
power - the exact two-sided minimum attainable p is 2 / C(8,3) = 0.0357, so the
test can only ever reach significance under complete separation. The printed
numbers are therefore reported as descriptive in docs/characterization_iv2a.md,
never as "significant".
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import mem_report, setup_threads

# After src.perf, before numpy: 4 threads because another agent is running a
# concurrent moderate job on this machine (docs/compute_environment.md rule 1).
setup_threads(4)

import gc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.geometry.distance import distance_riemann  # noqa: E402
from pyriemann.geometry.mean import mean_riemann  # noqa: E402
from scipy.signal import butter, sosfiltfilt, welch  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from src.config import IV2A_CLASSES, IV2A_SUBJECTS, RANDOM_STATE, RESULTS_DIR  # noqa: E402
from src.datasets import (  # noqa: E402
    IV2A_CHANNELS,
    WAVELET_CALIB_TMIN,
    load_iv2a_subject,
    load_iv2a_subject_wide_window,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SFREQ = 250.0
WIDE_TMIN = WAVELET_CALIB_TMIN  # -1.0 s: the wide loader's epoch start
EXPECTED_TRIALS_PER_SESSION = 288
EXPECTED_CHANNELS = 22
# IV-2a ships 25 recorded channels: 22 EEG + 3 EOG (verified directly in
# data/MNE-bnci-data/~bci/database/001-2014/A01T.mat: the 'X' field is
# (29683, 25); MOABB's bnci_2014.py declares ch_types = 22 eeg + 3 eog).
# MOABB's MotorImagery paradigm returns EEG only, so these are available in
# the raw recording but are NOT present in any array this script analyses.
EOG_CHANNELS_IN_RAW = ("EOG1", "EOG2", "EOG3")

# Artifact proxy. 100 uV is the conventional EEG artifact criterion in the
# motor-imagery literature (and the value BCI IV-2a's own documentation
# discusses for rejecting contaminated trials). Note the data reaching this
# script has already been 1-40 Hz band-passed by MOABB, which removes much of
# the EMG/glitch content a raw-data criterion would catch, so these fractions
# are a *lower* bound on true artifact prevalence.
ARTIFACT_THRESHOLD_UV = 100.0
ARTIFACT_THRESHOLD_HIGH_UV = 150.0

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
TOTAL_BAND = (1.0, 40.0)
MU_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
PEAK_SEARCH_BAND = (6.0, 14.0)  # mu/alpha peak search window
APERIODIC_FIT_EXCLUDE = (6.0, 14.0)  # excluded from the 1/f fit so the peak
# itself cannot bias its own background estimate
PERMUTATIONS = 1000

# Sample-exact windows (the wide epoch's times are -1.0 + i/250 s, so all of
# these land on sample boundaries).
BASELINE_INTERIOR_S = (-0.8, -0.3)  # primary baseline: keeps 200 ms away from
# the epoch start, where a zero-phase 8-13 Hz filter still has an edge transient
BASELINE_WIDE_S = (-1.0, -0.2)  # the project's existing convention (phase4),
# used as a sensitivity check rather than as the primary baseline
CUE_WINDOW_S = (0.5, 2.5)  # matches src/config.py's TMIN/TMAX exactly

# Zero-phase filtering is padded by a full second of (odd-reflected) signal so
# that the transient is settled before the window of interest, instead of being
# extrapolated from scipy's ~15-sample default pad.
FILTER_PADLEN = 250
ERD_FILTER_ORDER = 4

LEFT_HEMI = ["C5", "C3", "C1", "CP3", "CP1"]
RIGHT_HEMI = ["C6", "C4", "C2", "CP4", "CP2"]
HEMI_PAIRS = [("C3", "C4"), ("C5", "C6"), ("C1", "C2"), ("CP3", "CP4"), ("CP1", "CP2")]
LEFT_IDX = [IV2A_CHANNELS.index(c) for c in LEFT_HEMI]
RIGHT_IDX = [IV2A_CHANNELS.index(c) for c in RIGHT_HEMI]
# Sensorimotor strip used for the subject-level spectra (peak frequency, band
# power) - the same channels CSP is free to weight most heavily.
MOTOR_CHANNELS = ["FC3", "FC1", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                  "CP3", "CP1", "CPz", "CP2", "CP4"]
MOTOR_IDX = [IV2A_CHANNELS.index(c) for c in MOTOR_CHANNELS]
# Feet and tongue imagery have no comparable left/right lateralisation
# prediction (feet is mesial foot-area, tongue is inferior-central/centro-
# temporal), so the laterality index is only defined for the hand classes.
HAND_CLASSES = ("left_hand", "right_hand")
CLASS_CODES = {c: i for i, c in enumerate(IV2A_CLASSES)}

GROUPS = {"STRONG": [1, 3, 7, 8, 9], "MIDDLE": [4], "WEAK": [2, 5, 6]}
GROUP_OF = {s: name for name, subs in GROUPS.items() for s in subs}

# Values quoted in the task context, asserted against the committed CSVs so a
# silent change in either file cannot go unnoticed (see `_load_committed_results`).
EXPECTED_PHASE2_ALIGNED = {
    1: 0.8090277777777778, 2: 0.4444444444444444, 3: 0.8159722222222222,
    4: 0.6076388888888888, 5: 0.4236111111111111, 6: 0.4930555555555556,
    7: 0.7847222222222222, 8: 0.8055555555555556, 9: 0.8402777777777778,
}
EXPECTED_PHASE2_MEAN_ALIGNED = 0.6693672839506174
EXPECTED_PHASE1_CROSS = {
    1: 0.7847222222222222, 2: 0.4236111111111111, 3: 0.7673611111111112,
    4: 0.5868055555555556, 5: 0.3402777777777778, 6: 0.4756944444444444,
    7: 0.625, 8: 0.7881944444444444, 9: 0.8090277777777778,
}


# --------------------------------------------------------------------------
# Array helpers
# --------------------------------------------------------------------------


def _window_slice(window_s: tuple[float, float], tmin: float = WIDE_TMIN) -> slice:
    """Inclusive sample slice for a window in seconds, for an epoch starting at
    `tmin` sampled at SFREQ."""
    lo = int(round((window_s[0] - tmin) * SFREQ))
    hi = int(round((window_s[1] - tmin) * SFREQ))
    return slice(lo, hi + 1)


def _chunked_welch_psd(
    X: np.ndarray, nperseg: int = 500, nfft: int = 2048, chunk_trials: int = 96
) -> tuple[np.ndarray, np.ndarray]:
    """Mean Welch PSD over trials, per channel: (n_trials, ch, time) -> (ch, freq).

    Chunked over trials so the intermediate per-trial PSD never exceeds a few
    tens of MB (the wide epoch is 288 x 22 x 1251 float64 = 63 MB by itself).
    """
    acc = None
    n_chunks = 0
    freqs = None
    for start in range(0, X.shape[0], chunk_trials):
        block = X[start : start + chunk_trials]
        freqs, pxx = welch(
            block, fs=SFREQ, nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft,
            detrend="constant", axis=-1,
        )
        acc = pxx.mean(axis=0) if acc is None else acc + pxx.mean(axis=0)
        n_chunks += 1
        del block, pxx
    return freqs, acc / n_chunks


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Integrate a PSD over a band: (ch, freq) -> (ch,) band power."""
    mask = (freqs >= band[0]) & (freqs < band[1])
    df = float(freqs[1] - freqs[0])
    return psd[:, mask].sum(axis=1) * df


def _aperiodic_adjusted_peak(
    freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float] = PEAK_SEARCH_BAND,
    exclude: tuple[float, float] = APERIODIC_FIT_EXCLUDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Peak frequency inside `band` after removing the aperiodic background.

    Returns (peak frequency, raw PSD at that frequency, residual log10 power at
    that frequency, fitted aperiodic slope). A straight argmax of the raw PSD
    is useless here: EEG power falls monotonically with frequency, so it pins
    the "peak" to the lower edge of the search band for every subject. The
    aperiodic fit is confined to TOTAL_BAND: above 40 Hz the loader's low-pass
    leaves only numerical noise, and including it drags the fitted slope to
    absurd values (~-12 instead of the expected ~-1.5) and pushes the peak to
    the top edge of the search window for every subject.
    """
    fit_mask = (
        (freqs >= TOTAL_BAND[0]) & (freqs < exclude[0])
        | (freqs > exclude[1]) & (freqs <= TOTAL_BAND[1])
    )
    slopes = np.empty(psd.shape[0])
    peaks = np.empty(psd.shape[0])
    raw_at_peak = np.empty(psd.shape[0])
    residual_at_peak = np.empty(psd.shape[0])
    search = (freqs >= band[0]) & (freqs <= band[1])
    log_f = np.log10(freqs[fit_mask])
    for ci in range(psd.shape[0]):
        slope, intercept = np.polyfit(log_f, np.log10(psd[ci, fit_mask]), 1)
        residuals = np.log10(psd[ci]) - (intercept + slope * np.log10(np.maximum(freqs, 1e-12)))
        idx = np.argmax(residuals[search])
        peaks[ci] = freqs[search][idx]
        raw_at_peak[ci] = psd[ci, search][idx]
        residual_at_peak[ci] = residuals[search][idx]
        slopes[ci] = slope
    return peaks, raw_at_peak, residual_at_peak, slopes


def _excess_kurtosis(X: np.ndarray, chunk_trials: int = 96) -> np.ndarray:
    """Fisher (excess) kurtosis per channel, pooled over all trials/samples.

    Two chunked passes (mean, then central moments) rather than a
    (ch, n_trials*time) transpose copy, which would materialise another ~63 MB
    per session for no benefit.
    """
    mean = X.mean(axis=(0, 2))
    n_total = float(X.shape[0] * X.shape[-1])
    m2 = np.zeros(X.shape[1])
    m4 = np.zeros(X.shape[1])
    for start in range(0, X.shape[0], chunk_trials):
        d = X[start : start + chunk_trials] - mean[None, :, None]
        m2 += (d ** 2).sum(axis=(0, 2))
        m4 += (d ** 4).sum(axis=(0, 2))
        del d
    m2 /= n_total
    m4 /= n_total
    return m4 / m2**2 - 3.0


def _band_power_trials(X: np.ndarray, band: tuple[float, float], window_s: slice) -> np.ndarray:
    """Per-trial, per-channel band power in one time window: (n_trials, ch).

    Band-limited with a zero-phase Butterworth, padded by FILTER_PADLEN samples
    so the epoch-edge transient is settled before the window of interest.
    """
    sos = butter(ERD_FILTER_ORDER, list(band), btype="band", fs=SFREQ, output="sos")
    Xf = sosfiltfilt(sos, X, axis=-1, padlen=FILTER_PADLEN)
    p = (Xf[:, :, window_s] ** 2).mean(axis=-1)
    del Xf
    return p


def _erd_from_trial_powers(p_cue: np.ndarray, p_base: np.ndarray, y: np.ndarray) -> dict:
    """ERD = (mean P_cue - mean P_base) / mean P_base, per class and channel.

    Ratio of mean powers, deliberately: the mean of the per-trial ratio is
    biased and destroys spatial structure (verified - see module docstring).
    """
    out = {}
    for cls in IV2A_CLASSES:
        mask = y == cls
        cue_mean = p_cue[mask].mean(axis=0)
        base_mean = p_base[mask].mean(axis=0)
        out[cls] = (cue_mean - base_mean) / base_mean
    return out


def _perm_p_two_group(
    p_cue: np.ndarray, p_base: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray,
    rng: np.random.Generator, logspace: bool = False, n_perm: int = PERMUTATIONS,
) -> tuple[float, float]:
    """Permutation p-value for "the two groups' band-power patterns differ".

    Statistic: sum over channels of the squared difference between the two
    groups' mean ERD vectors. Permuting trial labels gives an exact null for
    "labels carry no information about the cue-period band-power pattern", and
    the statistic uses no learned spatial filter - so unlike CSP+LDA accuracy
    it cannot be inflated by overfitting a projection. Two-sided by
    construction (a squared difference has no direction).
    """
    sel = mask_a | mask_b
    cue = p_cue[sel]
    base = p_base[sel]
    lab = mask_a[sel]

    def stat(a: np.ndarray) -> float:
        ea = _pattern(cue, base, a, logspace)
        eb = _pattern(cue, base, ~a, logspace)
        return float(((ea - eb) ** 2).sum())

    return _perm_pvalue(stat, lab, rng, n_perm)


def _pattern(cue: np.ndarray, base: np.ndarray, sel: np.ndarray, logspace: bool) -> np.ndarray:
    """Mean ERD pattern (22,) over the selected trials, in ratio or log space."""
    if logspace:
        return np.log(cue[sel]).mean(axis=0) - np.log(base[sel]).mean(axis=0)
    return cue[sel].mean(axis=0) / base[sel].mean(axis=0)


def _perm_pvalue(stat, labels: np.ndarray, rng: np.random.Generator, n_perm: int) -> tuple[float, float]:
    """(permutation p, observed/null-mean effect size) for a label-permutation test.

    The second value is the statistic divided by the mean of its permutation
    null - the permutation analogue of an effect size. With a heavy-tailed
    per-trial power distribution the null mean is large, so this ratio is a
    conservative but interpretable "how far above chance" number; it is
    comparable across subjects but not across statistic definitions.
    """
    observed = stat(labels)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = stat(rng.permutation(labels))
    p = (1 + int((null >= observed).sum())) / (1 + n_perm)
    return p, float(observed / null.mean())


def _perm_p_k_group(
    p_cue: np.ndarray, p_base: np.ndarray, codes: np.ndarray,
    rng: np.random.Generator, logspace: bool = False, n_perm: int = PERMUTATIONS,
) -> tuple[float, float]:
    """Permutation test for class structure among all 4 classes.

    Statistic: sum over classes of the squared distance from that class's mean
    ERD pattern to the grand-mean pattern (a permutation analogue of a one-way
    MANOVA on the 22-channel band-power pattern). This asks the question the
    classifier's accuracy cannot ask on its own: is there *any* reliable
    class-related band-power structure in this subject's own calibration
    session, using no learned spatial filter at all?
    """

    def stat(codes_now: np.ndarray) -> float:
        means = np.stack([
            _pattern(p_cue, p_base, codes_now == c, logspace)
            for c in range(len(IV2A_CLASSES))
        ])
        return float(((means - means.mean(axis=0)) ** 2).sum())

    return _perm_pvalue(stat, codes, rng, n_perm)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 * P(pos == neg) - the Mann-Whitney / common-language
    effect size, computed from average ranks so ties are handled."""
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _laterality(erd_mu: dict, erd_beta: dict) -> dict:
    """Contralateral-vs-ipsilateral desynchronisation for the hand classes.

    Convention: D = -ERD = (P_base - P_cue)/P_base, so D > 0 is a drop in band
    power (desynchronisation). The expected motor-imagery pattern is a
    *contralateral* drop: left hand -> larger D over the right hemisphere
    (C4/CP4 side), right hand -> larger D over the left (C3/CP3 side).

    li_<class> = mean(D, contralateral) - mean(D, ipsilateral); positive is the
    expected direction. li_mu = mean of the two hand-class values.
    """
    out: dict[str, float] = {}
    d_mu = {cls: -erd_mu[cls] for cls in IV2A_CLASSES}
    d_beta = {cls: -erd_beta[cls] for cls in IV2A_CLASSES}
    sides = {"left_hand": (RIGHT_IDX, LEFT_IDX), "right_hand": (LEFT_IDX, RIGHT_IDX)}
    for cls, (contra, ipsi) in sides.items():
        out[f"li_mu_{cls}"] = float(d_mu[cls][contra].mean() - d_mu[cls][ipsi].mean())
        out[f"li_beta_{cls}"] = float(d_beta[cls][contra].mean() - d_beta[cls][ipsi].mean())
    out["li_mu"] = 0.5 * (out["li_mu_left_hand"] + out["li_mu_right_hand"])
    out["li_beta"] = 0.5 * (out["li_beta_left_hand"] + out["li_beta_right_hand"])

    # Sign consistency over the 5 homologous pairs x 2 hand classes. Under a
    # purely noise lateralisation this sits near the 0.5 chance level.
    n_ok = 0
    for ch_l, ch_r in HEMI_PAIRS:
        i_l, i_r = IV2A_CHANNELS.index(ch_l), IV2A_CHANNELS.index(ch_r)
        if d_mu["left_hand"][i_r] > d_mu["left_hand"][i_l]:
            n_ok += 1
        if d_mu["right_hand"][i_l] > d_mu["right_hand"][i_r]:
            n_ok += 1
    out["li_pair_consistency"] = n_ok / (2 * len(HEMI_PAIRS))

    # Physiological quantities behind the index, for the write-up: left-hand
    # right-hemisphere and right-hand left-hemisphere pooled (contralateral),
    # and the mirrored ipsilateral set, over both bands.
    contra = np.concatenate([
        d_mu["left_hand"][RIGHT_IDX], d_mu["right_hand"][LEFT_IDX],
        d_beta["left_hand"][RIGHT_IDX], d_beta["right_hand"][LEFT_IDX],
    ])
    ipsi = np.concatenate([
        d_mu["left_hand"][LEFT_IDX], d_mu["right_hand"][RIGHT_IDX],
        d_beta["left_hand"][LEFT_IDX], d_beta["right_hand"][RIGHT_IDX],
    ])
    out["desync_contra_hands"] = float(contra.mean())
    out["desync_ipsi_hands"] = float(ipsi.mean())
    return out


# --------------------------------------------------------------------------
# Per-session and per-subject computation
# --------------------------------------------------------------------------


def _channel_rows(subject: int, session: str, X: np.ndarray, y: np.ndarray, rng) -> tuple:
    """All channel-level features for one subject's one session.

    Returns (channel rows, laterality dict, wide-baseline laterality dict,
    subject-level extras dict). The extras are computed here because they reuse
    the same PSD/ERD arrays; recomputing them later would double the cost.
    """
    n_trials, n_ch = X.shape[0], X.shape[1]
    assert n_ch == EXPECTED_CHANNELS, f"expected {EXPECTED_CHANNELS} EEG channels, got {n_ch}"
    if n_trials != EXPECTED_TRIALS_PER_SESSION:
        print(f"  !! DEVIATION: subject {subject} session {session} has {n_trials} trials, "
              f"expected {EXPECTED_TRIALS_PER_SESSION}")

    counts = {c: int((y == c).sum()) for c in IV2A_CLASSES}
    ptp = np.ptp(X, axis=(0, 2))
    rms = np.sqrt((X ** 2).mean(axis=(0, 2)))
    kurt = _excess_kurtosis(X)
    max_abs = np.maximum(X.max(axis=-1), -X.min(axis=-1))  # (n_trials, ch)
    frac_100 = (max_abs > ARTIFACT_THRESHOLD_UV).mean(axis=0)
    frac_150 = (max_abs > ARTIFACT_THRESHOLD_HIGH_UV).mean(axis=0)

    freqs, psd = _chunked_welch_psd(X)
    band_pow = {name: _band_power(freqs, psd, band) for name, band in BANDS.items()}
    total_pow = _band_power(freqs, psd, TOTAL_BAND)
    ch_peak_f, ch_peak_psd, ch_peak_res, _ = _aperiodic_adjusted_peak(freqs, psd)

    cue_sl = _window_slice(CUE_WINDOW_S)
    base_int_sl = _window_slice(BASELINE_INTERIOR_S)
    base_wide_sl = _window_slice(BASELINE_WIDE_S)

    p_cue_mu = _band_power_trials(X, MU_BAND, cue_sl)
    p_base_mu = _band_power_trials(X, MU_BAND, base_int_sl)
    p_cue_beta = _band_power_trials(X, BETA_BAND, cue_sl)
    p_base_beta = _band_power_trials(X, BETA_BAND, base_int_sl)
    p_base_mu_wide = _band_power_trials(X, MU_BAND, base_wide_sl)
    p_base_beta_wide = _band_power_trials(X, BETA_BAND, base_wide_sl)

    erd_mu = _erd_from_trial_powers(p_cue_mu, p_base_mu, y)
    erd_beta = _erd_from_trial_powers(p_cue_beta, p_base_beta, y)
    erd_mu_wide = _erd_from_trial_powers(p_cue_mu, p_base_mu_wide, y)
    erd_beta_wide = _erd_from_trial_powers(p_cue_beta, p_base_beta_wide, y)

    rows = []
    for ci, ch in enumerate(IV2A_CHANNELS):
        row = {
            "subject": subject, "session": session, "channel": ch, "channel_index": ci,
            "n_trials": n_trials,
            **{f"n_{c}": counts[c] for c in IV2A_CLASSES},
            "ptp_uv": float(ptp[ci]), "rms_uv": float(rms[ci]), "excess_kurtosis": float(kurt[ci]),
            "frac_trials_abs_gt_100uv": float(frac_100[ci]),
            "frac_trials_abs_gt_150uv": float(frac_150[ci]),
            "total_power_1_40_uv2_per_hz": float(total_pow[ci]),
            **{f"abs_{name}_uv2": float(band_pow[name][ci]) for name in BANDS},
            **{f"rel_{name}": float(band_pow[name][ci] / total_pow[ci]) for name in BANDS},
            "mu_beta_ratio": float(band_pow["alpha"][ci] / band_pow["beta"][ci]),
            "peak_freq_6_14_hz": float(ch_peak_f[ci]),
            "peak_psd_6_14_uv2_per_hz": float(ch_peak_psd[ci]),
            "peak_residual_log10": float(ch_peak_res[ci]),
        }
        for cls in IV2A_CLASSES:
            row[f"erd_mu_{cls}"] = float(erd_mu[cls][ci])
            row[f"erd_beta_{cls}"] = float(erd_beta[cls][ci])
        for cls in IV2A_CLASSES:
            row[f"erd_mu_widebase_{cls}"] = float(erd_mu_wide[cls][ci])
            row[f"erd_beta_widebase_{cls}"] = float(erd_beta_wide[cls][ci])
        rows.append(row)

    # ---- subject-level extras that come from the same arrays --------------
    motor_psd = psd[MOTOR_IDX].mean(axis=0, keepdims=True)
    motor_band = {name: float(_band_power(freqs, motor_psd, band)[0]) for name, band in BANDS.items()}
    motor_total = float(_band_power(freqs, motor_psd, TOTAL_BAND)[0])
    pk_f, pk_psd, pk_res, pk_slope = _aperiodic_adjusted_peak(freqs, motor_psd)
    all_peak_f, _, _, _ = _aperiodic_adjusted_peak(freqs, psd.mean(axis=0, keepdims=True))

    lat = _laterality(erd_mu, erd_beta)
    lat_wide = _laterality(erd_mu_wide, erd_beta_wide)

    # Welch-based third estimate of the mu laterality, sharing a frequency
    # resolution between the baseline and cue windows.
    p_cue_w = _band_power_trials_welch(X, MU_BAND, cue_sl)
    p_base_w = _band_power_trials_welch(X, MU_BAND, base_int_sl)
    erd_mu_welch = _erd_from_trial_powers(p_cue_w, p_base_w, y)
    li_welch = _laterality(erd_mu_welch, erd_mu_welch)["li_mu"]

    # Permutation tests: is there class-related band-power structure at all?
    # Reported for two statistic definitions (ratio of mean powers, and
    # difference of mean log powers) because the per-trial power distribution is
    # heavy-tailed and the ratio-of-means null is the more conservative of the
    # two; where they disagree, the disagreement is itself reported.
    hand = np.isin(y, HAND_CLASSES)
    is_left = y == "left_hand"
    codes = np.array([CLASS_CODES[c] for c in y])
    perm_p_lr, perm_eff_lr = _perm_p_two_group(p_cue_mu, p_base_mu, is_left, ~is_left & hand, rng)
    perm_p_4c, perm_eff_4c = _perm_p_k_group(p_cue_mu, p_base_mu, codes, rng)
    perm_p_4c_log, _ = _perm_p_k_group(p_cue_mu, p_base_mu, codes, rng, logspace=True)

    # Secondary, exploratory configuration probe: the same tests in a band
    # centred on this subject's own mu peak - the fixed 8-13 Hz band is a
    # pipeline choice, this one is not. Subject-adaptive and chosen on the same
    # data, so it is optimistic by construction.
    peak_lo = float(max(5.0, pk_f[0] - 2.0))
    peak_hi = float(min(15.0, pk_f[0] + 2.0))
    p_cue_pk = _band_power_trials(X, (peak_lo, peak_hi), cue_sl)
    p_base_pk = _band_power_trials(X, (peak_lo, peak_hi), base_int_sl)
    perm_p_pk_lr, perm_eff_pk_lr = _perm_p_two_group(p_cue_pk, p_base_pk, is_left, ~is_left & hand, rng)
    perm_p_pk_4c, perm_eff_pk_4c = _perm_p_k_group(p_cue_pk, p_base_pk, codes, rng)
    erd_peak = _erd_from_trial_powers(p_cue_pk, p_base_pk, y)

    extras = {
        "rel_delta_motor": motor_band["delta"] / motor_total,
        "rel_theta_motor": motor_band["theta"] / motor_total,
        "rel_alpha_motor": motor_band["alpha"] / motor_total,
        "rel_beta_motor": motor_band["beta"] / motor_total,
        "total_power_motor_uv2_per_hz": motor_total,
        "mu_beta_ratio_motor": motor_band["alpha"] / motor_band["beta"],
        "peak_freq_motor": float(pk_f[0]),
        "peak_psd_motor": float(pk_psd[0]),
        "peak_residual_log10_motor": float(pk_res[0]),
        "aperiodic_slope_motor": float(pk_slope[0]),
        "peak_freq_allchan": float(all_peak_f[0]),
        "erd_mu_hands_mean": float(
            0.5 * (erd_mu["left_hand"][MOTOR_IDX].mean() + erd_mu["right_hand"][MOTOR_IDX].mean())
        ),
        "erd_beta_hands_mean": float(
            0.5 * (erd_beta["left_hand"][MOTOR_IDX].mean() + erd_beta["right_hand"][MOTOR_IDX].mean())
        ),
        "erd_peakband_hands_mean": float(
            0.5 * (erd_peak["left_hand"][MOTOR_IDX].mean() + erd_peak["right_hand"][MOTOR_IDX].mean())
        ),
        "li_mu_welch": li_welch,
        "perm_p_mu_lr": perm_p_lr,
        "perm_obs_over_null_mu_lr": perm_eff_lr,
        "perm_p_mu_4class": perm_p_4c,
        "perm_obs_over_null_mu_4class": perm_eff_4c,
        "perm_p_mu_4class_log": perm_p_4c_log,
        "perm_p_peakband_lr": perm_p_pk_lr,
        "perm_obs_over_null_peakband_lr": perm_eff_pk_lr,
        "perm_p_peakband_4class": perm_p_pk_4c,
        "perm_obs_over_null_peakband_4class": perm_eff_pk_4c,
        "peak_band_lo_hz": peak_lo,
        "peak_band_hi_hz": peak_hi,
    }
    return rows, lat, lat_wide, extras


def _band_power_trials_welch(
    X: np.ndarray, band: tuple[float, float], window_s: slice,
    nperseg: int = 126, chunk_trials: int = 96,
) -> np.ndarray:
    """Per-trial band power in one window via Welch. Robustness variant of the
    filter-based estimate used for the laterality index; the same nperseg is
    used for the baseline and cue windows, so both share a frequency resolution
    and therefore so does their ratio.
    """
    seg = X[:, :, window_s]
    parts = []
    for start in range(0, seg.shape[0], chunk_trials):
        freqs, pxx = welch(
            seg[start : start + chunk_trials], fs=SFREQ, nperseg=nperseg,
            noverlap=nperseg // 2, detrend="constant", axis=-1,
        )
        mask = (freqs >= band[0]) & (freqs < band[1])
        df = float(freqs[1] - freqs[0])
        parts.append(pxx[..., mask].sum(axis=-1) * df)
    return np.concatenate(parts, axis=0)


def _session_summary(prefix: str, lat: dict, lat_wide: dict, extras: dict) -> dict:
    """Flatten one session's laterality/extras into columns suffixed with the
    session name."""
    out = {
        f"li_mu_{prefix}": lat["li_mu"],
        f"li_mu_left_hand_{prefix}": lat["li_mu_left_hand"],
        f"li_mu_right_hand_{prefix}": lat["li_mu_right_hand"],
        f"li_beta_{prefix}": lat["li_beta"],
        f"li_mu_widebase_{prefix}": lat_wide["li_mu"],
        f"li_pair_consistency_{prefix}": lat["li_pair_consistency"],
        f"desync_contra_hands_{prefix}": lat["desync_contra_hands"],
        f"desync_ipsi_hands_{prefix}": lat["desync_ipsi_hands"],
    }
    out.update({f"{key}_{prefix}": val for key, val in extras.items()})
    return out


def _load_committed_results() -> pd.DataFrame:
    """Join the committed Phase 1 / Phase 2 accuracies (read, never recomputed),
    asserting they still match the values quoted in the task context."""
    p1 = pd.read_csv(RESULTS_DIR / "phase1_baseline_iv2a.csv")
    p2 = pd.read_csv(RESULTS_DIR / "phase2_riemannian_alignment_iv2a.csv")

    p1["subject_num"] = pd.to_numeric(p1["subject"], errors="coerce")
    p2["subject_num"] = pd.to_numeric(p2["subject"], errors="coerce")
    p1_sub = p1.dropna(subset=["subject_num"]).set_index("subject_num")
    p2_sub = p2.dropna(subset=["subject_num"]).set_index("subject_num")

    assert set(p1_sub.index.astype(int)) == set(IV2A_SUBJECTS), "Phase 1 CSV subject set changed"
    assert set(p2_sub.index.astype(int)) == set(IV2A_SUBJECTS), "Phase 2 CSV subject set changed"

    for s, expected in EXPECTED_PHASE2_ALIGNED.items():
        got = float(p2_sub.loc[s, "cross_session_aligned_acc"])
        assert np.isclose(got, expected, rtol=0, atol=1e-12), (
            f"Phase 2 aligned accuracy for subject {s} is {got!r}, but the task context says {expected!r}"
        )
    for s, expected in EXPECTED_PHASE1_CROSS.items():
        got = float(p1_sub.loc[s, "cross_session_acc"])
        assert np.isclose(got, expected, rtol=0, atol=1e-12), (
            f"Phase 1 cross-session accuracy for subject {s} is {got!r}, but the task context says {expected!r}"
        )
        got2 = float(p2_sub.loc[s, "cross_session_baseline_acc"])
        assert np.isclose(got2, expected, rtol=0, atol=1e-12), (
            f"Phase 1 and Phase 2 baseline accuracies disagree for subject {s}: {got!r} vs {got2!r}"
        )
    mean_aligned = float(p2_sub["cross_session_aligned_acc"].mean())
    assert np.isclose(mean_aligned, EXPECTED_PHASE2_MEAN_ALIGNED, rtol=0, atol=1e-12), (
        f"Phase 2 mean aligned accuracy is {mean_aligned!r}, expected {EXPECTED_PHASE2_MEAN_ALIGNED!r}"
    )
    print(f"[check] committed accuracies verified: Phase 1 cross-session mean "
          f"{p1_sub['cross_session_acc'].mean():.4f}, Phase 2 aligned mean {mean_aligned:.4f}")

    summary = pd.DataFrame(index=pd.Index(IV2A_SUBJECTS, name="subject"))
    summary["phase1_within_session_acc"] = [float(p1_sub.loc[s, "within_session_acc"]) for s in IV2A_SUBJECTS]
    summary["phase1_within_session_std"] = [float(p1_sub.loc[s, "within_session_std"]) for s in IV2A_SUBJECTS]
    summary["phase1_cross_session_acc"] = [float(p1_sub.loc[s, "cross_session_acc"]) for s in IV2A_SUBJECTS]
    summary["phase1_cross_session_ci_low"] = [float(p1_sub.loc[s, "cross_session_ci_low"]) for s in IV2A_SUBJECTS]
    summary["phase1_cross_session_ci_high"] = [float(p1_sub.loc[s, "cross_session_ci_high"]) for s in IV2A_SUBJECTS]
    summary["phase2_baseline_acc"] = [float(p2_sub.loc[s, "cross_session_baseline_acc"]) for s in IV2A_SUBJECTS]
    summary["phase2_aligned_acc"] = [float(p2_sub.loc[s, "cross_session_aligned_acc"]) for s in IV2A_SUBJECTS]
    return summary.reset_index()


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    committed = _load_committed_results()
    estimator_check()

    channel_rows: list[dict] = []
    subject_rows: list[dict] = []
    mean_covs: dict[int, dict[str, np.ndarray]] = {}

    for subject in IV2A_SUBJECTS:
        t_sub = time.time()
        print(f"\n[subject {subject}] loading wide window (-1..4 s, 1-40 Hz)...")
        wide = load_iv2a_subject_wide_window(subject)
        assert wide.X_calib.shape[1] == EXPECTED_CHANNELS
        assert wide.X_calib.shape[-1] == 1251, f"wide epoch length changed: {wide.X_calib.shape}"
        assert wide.X_eval.shape[1] == EXPECTED_CHANNELS

        summary_row: dict = {
            "subject": subject, "group": GROUP_OF[subject],
            "n_channels_eeg": EXPECTED_CHANNELS,
            "n_channels_eog_available": len(EOG_CHANNELS_IN_RAW),
        }
        for session, (X, y) in (("calib", (wide.X_calib, wide.y_calib)),
                                ("eval", (wide.X_eval, wide.y_eval))):
            n_trials = X.shape[0]
            counts = {c: int((y == c).sum()) for c in IV2A_CLASSES}
            print(f"  [{session}] {X.shape}  trials={n_trials}  per-class={counts}")
            rows, lat, lat_wide, extras = _channel_rows(subject, session, X, y, rng)
            assert {r["n_trials"] for r in rows} == {n_trials}
            channel_rows.extend(rows)
            summary_row.update(_session_summary(session, lat, lat_wide, extras))
            summary_row[f"n_trials_{session}"] = n_trials
            for cls in IV2A_CLASSES:
                summary_row[f"n_{cls}_{session}"] = counts[cls]

            ptp = np.array([r["ptp_uv"] for r in rows])
            kurt = np.array([r["excess_kurtosis"] for r in rows])
            frac = np.array([r["frac_trials_abs_gt_100uv"] for r in rows])
            summary_row.update({
                f"mean_ptp_uv_{session}": float(ptp.mean()),
                f"mean_excess_kurtosis_{session}": float(kurt.mean()),
                f"max_excess_kurtosis_{session}": float(kurt.max()),
                f"mean_frac_trials_gt_100uv_{session}": float(frac.mean()),
                f"max_frac_trials_gt_100uv_{session}": float(frac.max()),
            })
            print(f"    peak={extras['peak_freq_motor']:.2f}Hz li_mu={lat['li_mu']:+.3f} "
                  f"p_lr={extras['perm_p_mu_lr']:.3f} p_4class={extras['perm_p_mu_4class']:.3f} "
                  f"(eff={extras['perm_obs_over_null_mu_4class']:.2f} vs null) "
                  f"p_peakband_4class={extras['perm_p_peakband_4class']:.3f} "
                  f"({extras['peak_band_lo_hz']:.1f}-{extras['peak_band_hi_hz']:.1f}Hz)")
            del rows
        del wide
        gc.collect()

        # ---- cross-session drift, on the exact representation the Phase 2
        # pipeline uses (8-30 Hz, 0.5-2.5 s) and the exact covariance estimator
        # RiemannianAlignment uses ("oas").
        print("  [drift] loading narrow window (8-30 Hz, 0.5-2.5 s)...")
        narrow = load_iv2a_subject(subject)
        mean_covs[subject] = {}
        for session, Xn in (("calib", narrow.X_calib), ("eval", narrow.X_eval)):
            covs = Covariances(estimator="oas").fit_transform(Xn)
            assert covs.shape == (Xn.shape[0], EXPECTED_CHANNELS, EXPECTED_CHANNELS)
            mean_covs[subject][session] = mean_riemann(covs)
            del covs
        del narrow
        gc.collect()

        subject_rows.append(summary_row)
        print(f"[subject {subject}] done in {time.time() - t_sub:.1f}s")
        mem_report(f"subject {subject}")

    # ---- between-subject / within-subject Riemannian distances -------------
    print("\n[distances] affine-invariant Riemannian distance matrix over calibration mean covariances")
    dist_matrix = np.full((len(IV2A_SUBJECTS), len(IV2A_SUBJECTS)), np.nan)
    for i, si in enumerate(IV2A_SUBJECTS):
        for j, sj in enumerate(IV2A_SUBJECTS):
            if i != j:
                dist_matrix[i, j] = distance_riemann(mean_covs[si]["calib"], mean_covs[sj]["calib"])
    print("        " + "".join(f"{s:>8d}" for s in IV2A_SUBJECTS))
    for i, si in enumerate(IV2A_SUBJECTS):
        print(f"  subj {si}: " + "".join(
            "     ---" if i == j else f"{dist_matrix[i, j]:8.3f}" for j in range(len(IV2A_SUBJECTS))
        ))

    for row in subject_rows:
        s = row["subject"]
        i = IV2A_SUBJECTS.index(s)
        others = np.delete(dist_matrix[i], i)
        within = distance_riemann(mean_covs[s]["calib"], mean_covs[s]["eval"])
        row["riemann_drift_calib_to_eval"] = float(within)
        row["between_subject_dist_median"] = float(np.median(others))
        row["between_subject_dist_min"] = float(others.min())
        row["between_subject_dist_max"] = float(others.max())
        row["drift_ratio_to_between_median"] = float(within / np.median(others))

    # ---- assemble, join committed accuracies, write ------------------------
    summary = pd.DataFrame(subject_rows).merge(committed, on="subject", how="left")
    assert len(summary) == len(IV2A_SUBJECTS)
    assert not summary.isna().any().any(), (
        f"NaNs in summary: {summary.columns[summary.isna().any()].tolist()}"
    )
    summary["cross_session_accuracy_drop"] = (
        summary["phase1_within_session_acc"] - summary["phase1_cross_session_acc"]
    )
    summary["alignment_improvement"] = summary["phase2_aligned_acc"] - summary["phase2_baseline_acc"]
    summary["phase1_cross_acc_above_chance"] = summary["phase1_cross_session_ci_low"] > 0.25

    chars = pd.DataFrame(channel_rows)
    assert len(chars) == len(IV2A_SUBJECTS) * 2 * EXPECTED_CHANNELS, (
        f"expected {len(IV2A_SUBJECTS) * 2 * EXPECTED_CHANNELS} channel rows, got {len(chars)}"
    )
    assert not chars.isna().any().any(), (
        f"NaNs in channel table: {chars.columns[chars.isna().any()].tolist()}"
    )
    trial_counts = chars.groupby(["subject", "session"])["n_trials"].first()
    assert (trial_counts == EXPECTED_TRIALS_PER_SESSION).all(), (
        f"unexpected trial counts: {trial_counts[trial_counts != EXPECTED_TRIALS_PER_SESSION].to_dict()}"
    )
    print(f"[check] {EXPECTED_TRIALS_PER_SESSION} trials per subject/session confirmed for all "
          f"{len(trial_counts)} subject-sessions")

    chars_path = RESULTS_DIR / "char_iv2a_signal_characteristics.csv"
    summary_path = RESULTS_DIR / "char_iv2a_subject_summary.csv"
    chars.to_csv(chars_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {len(chars)} channel rows -> {chars_path}")
    print(f"Saved {len(summary)} subject rows -> {summary_path}")

    report_group_comparison(summary)
    report_confound_checks(summary)
    output_self_checks(chars)
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    return chars, summary


# --------------------------------------------------------------------------
# Descriptive group comparison (no significance claims: n=3 vs n=5)
# --------------------------------------------------------------------------

GROUP_FEATURES = [
    "phase1_within_session_acc", "phase1_cross_session_acc", "cross_session_accuracy_drop",
    "phase2_aligned_acc", "alignment_improvement",
    "peak_freq_motor_calib", "peak_freq_motor_eval",
    "rel_alpha_motor_calib", "rel_beta_motor_calib", "mu_beta_ratio_motor_calib",
    "total_power_motor_uv2_per_hz_calib", "peak_psd_motor_calib",
    "peak_residual_log10_motor_calib", "aperiodic_slope_motor_calib",
    "li_mu_calib", "li_mu_left_hand_calib", "li_mu_right_hand_calib", "li_beta_calib",
    "li_mu_widebase_calib", "li_mu_welch_calib", "li_pair_consistency_calib",
    "desync_contra_hands_calib", "desync_ipsi_hands_calib", "erd_mu_hands_mean_calib",
    "perm_p_mu_lr_calib", "perm_p_mu_4class_calib", "perm_p_mu_4class_log_calib",
    "perm_obs_over_null_mu_4class_calib", "perm_p_peakband_4class_calib",
    "perm_obs_over_null_peakband_4class_calib",
    "perm_p_mu_4class_eval", "perm_obs_over_null_mu_4class_eval",
    "mean_ptp_uv_calib", "mean_excess_kurtosis_calib", "mean_frac_trials_gt_100uv_calib",
    "riemann_drift_calib_to_eval", "drift_ratio_to_between_median",
]


def report_group_comparison(summary: pd.DataFrame) -> None:
    """Print WEAK vs STRONG medians with rank-based effect sizes.

    Deliberately descriptive: with n=3 vs n=5 the exact two-sided Mann-Whitney
    minimum attainable p is 2/C(8,3) = 0.0357, so the test has power only under
    complete separation, and even then the p-value is a statement about these 8
    subjects, not about a population. Effect sizes (AUC / Cliff's delta) are
    reported so the magnitude of any separation is visible regardless of p.
    """
    from scipy.stats import mannwhitneyu

    missing = [f for f in GROUP_FEATURES if f not in summary.columns]
    assert not missing, f"GROUP_FEATURES names not present in the summary table: {missing}"

    weak = summary[summary["group"] == "WEAK"]
    strong = summary[summary["group"] == "STRONG"]
    n_w, n_s = len(weak), len(strong)
    min_p = 2.0 / math.comb(n_w + n_s, n_w)
    print(f"\n[groups] WEAK {sorted(weak['subject'].tolist())} (n={n_w}) vs "
          f"STRONG {sorted(strong['subject'].tolist())} (n={n_s}); exact two-sided "
          f"Mann-Whitney minimum attainable p = {min_p:.4f}")
    print(f"{'feature':38s} {'weak_med':>9s} {'strong_med':>11s} {'AUC(W>S)':>9s} "
          f"{'cliff_d':>8s} {'p_exact':>8s}")
    for feat in GROUP_FEATURES:
        a = weak[feat].to_numpy(dtype=float)
        b = strong[feat].to_numpy(dtype=float)
        auc = _auc(a, b)
        try:
            p = float(mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue)
        except ValueError:  # ties present: scipy's exact method refuses
            p = float("nan")
        print(f"{feat:38s} {np.median(a):9.4f} {np.median(b):11.4f} {auc:9.3f} "
              f"{2 * auc - 1:8.3f} {p:8.4f}")


def estimator_check() -> None:
    """Reproduce the ERD-estimator comparison that justifies the estimator used.

    Runs on subject 1's calibration session only (cheap). The point is that the
    two candidate estimators of the same quantity disagree by ~+0.5 on *every*
    channel, in the direction that erases spatial structure - so this is not a
    cosmetic difference and the choice has to be recorded, not assumed.
    """
    from src.datasets import load_iv2a_subject_wide_window as _load

    print("\n[estimator check] subject 1 calibration, mu band, per-channel ERD")
    data = _load(1)
    p_cue = _band_power_trials(data.X_calib, MU_BAND, _window_slice(CUE_WINDOW_S))
    p_base = _band_power_trials(data.X_calib, MU_BAND, _window_slice(BASELINE_INTERIOR_S))
    mean_of_ratios = ((p_cue - p_base) / p_base).mean(axis=0)  # the biased estimator
    ratio_of_means = _erd_from_trial_powers(p_cue, p_base, data.y_calib)["left_hand"]
    print("  per-channel mean-of-per-trial-ratios (biased): "
          f"min={mean_of_ratios.min():+.3f} max={mean_of_ratios.max():+.3f} "
          f"mean={mean_of_ratios.mean():+.3f}")
    print("  per-channel ratio-of-mean-powers (used here):  "
          f"min={ratio_of_means.min():+.3f} max={ratio_of_means.max():+.3f} "
          f"mean={ratio_of_means.mean():+.3f}")
    print(f"  median discrepancy = {np.median(mean_of_ratios - ratio_of_means):+.3f} "
          "(the biased estimator is positive on every channel, i.e. it reports a "
          "global ERS where the data show a global ERD)")
    assert (mean_of_ratios > 0).all(), "expected the biased estimator to be positive on every channel"
    del data, p_cue, p_base
    gc.collect()


def report_confound_checks(summary: pd.DataFrame) -> None:
    """Spearman correlations that test the obvious confounds of the headline finding.

    The headline is "STRONG subjects show a cue-period mu ERD, WEAK subjects do
    not". The natural objection is that ERD is a ratio with baseline power in the
    denominator, so it could be driven by baseline mu power rather than by any
    task-related modulation. These correlations check that directly.
    """
    from scipy.stats import spearmanr

    pairs = [
        ("erd_mu_hands_mean_calib", "total_power_motor_uv2_per_hz_calib"),
        ("erd_mu_hands_mean_calib", "rel_alpha_motor_calib"),
        ("erd_mu_hands_mean_calib", "peak_residual_log10_motor_calib"),
        ("erd_mu_hands_mean_calib", "phase1_within_session_acc"),
        ("erd_mu_hands_mean_calib", "phase2_aligned_acc"),
        ("perm_obs_over_null_mu_4class_calib", "phase2_aligned_acc"),
        ("riemann_drift_calib_to_eval", "phase2_aligned_acc"),
    ]
    print("\n[confounds] Spearman rho across the 9 subjects "
          "(|rho|=0.68 is the two-sided 5% critical value at n=9; "
          "no multiplicity correction, so treat as descriptive)")
    for a, b in pairs:
        rho, p = spearmanr(summary[a], summary[b])
        print(f"  {a:42s} vs {b:36s} rho={rho:+.3f} p={p:.3f}")


def output_self_checks(chars: pd.DataFrame) -> None:
    """Checks on the written channel table: the assumptions behind the two
    headline scalars are checked here rather than trusted."""
    rel_cols = [f"rel_{name}" for name in BANDS]
    rel_sum = chars[rel_cols].sum(axis=1)
    assert (rel_sum <= 1.0 + 1e-9).all(), "relative band powers exceed 1"
    print(f"[check] relative band powers (1-30 Hz) sum to {rel_sum.mean():.3f} "
          f"[{rel_sum.min():.3f}, {rel_sum.max():.3f}] of the total 1-40 Hz power "
          "(the remainder is the 30-40 Hz band)")

    # The aperiodic-adjusted peak must be interior to the search band; a peak
    # pinned at an edge would mean the background fit failed for that channel.
    edge = (chars["peak_freq_6_14_hz"] <= 6.0 + 0.15) | (chars["peak_freq_6_14_hz"] >= 14.0 - 0.15)
    print(f"[check] peak frequency at a search-band edge in {int(edge.sum())} / {len(chars)} "
          f"channel rows (min={chars['peak_freq_6_14_hz'].min():.2f} Hz, "
          f"max={chars['peak_freq_6_14_hz'].max():.2f} Hz)")
    print(f"[check] max trial fraction with |amplitude| > 100 uV over any channel/session: "
          f"{chars['frac_trials_abs_gt_100uv'].max():.4f}")
    print(f"[check] channel rows per session: "
          f"{chars.groupby(['subject', 'session']).size().unique().tolist()}")


if __name__ == "__main__":
    run()
