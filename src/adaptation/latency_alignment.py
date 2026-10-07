"""Woody's adaptive filter (Woody 1967): iteratively estimate each trial's
temporal latency against an evolving template via cross-correlation, shift
trials into alignment, re-average to refine the template, repeat. The
classical precursor to RIDE and other latency-variability-correction
methods (see docs/progress_and_direction.md's literature-check section) -
the same alternating-optimization structure as riemannian_icp.py's ICP
(alternate correspondence-finding and rigid-transform refitting), just in
1D time instead of covariance/tangent space.

Operates on a trial's amplitude *envelope* (Hilbert transform of the
band-limited signal), not the raw oscillation - the oscillatory carrier's
phase is not trial-locked (random per trial in this project's synthetic
generator, and there's no reason to expect it locked in real EEG either),
so only the envelope's slower onset/trough shape is a genuinely alignable
template.

A structurally similar mechanism (searching many candidate lags for a
"best match") was tried once already in this project and made things
*worse*, not better - Phase 4's jitter-tolerant max-over-lags classifier
scoring, a multiple-comparisons/peak-picking bias found to hurt accuracy
even at zero true jitter (docs/progress_and_direction.md). This module's
use of lag search is for a different purpose (re-averaging into a
template, not scoring a classification match), but that precedent is
exactly why stage0_latency_alignment_validation.py tests a zero-true-
jitter control condition directly, rather than assuming this is safe by
analogy to Woody's filter working elsewhere.
"""

import numpy as np


def smooth_signal(x: np.ndarray, window_samples: int) -> np.ndarray:
    """Moving-average smoothing. Found necessary, not cosmetic: a raw
    Hilbert envelope of an oscillatory, band-limited-noise-driven carrier
    has *intrinsic* amplitude fluctuations set by the carrier's own
    bandwidth (a narrowband random process's envelope has a natural
    correlation time of roughly 1/bandwidth, independent of any additive
    background noise) - verified directly in
    stage0_latency_alignment_validation.py by raising additive SNR from
    -3dB to 60dB with *no* reduction in cross-correlation's spurious-lag
    variance, ruling out additive noise as the cause. Woody's filter,
    applied to that raw envelope, confuses this intrinsic carrier
    randomness for genuine trial-to-trial latency structure and produces
    spurious "best" lags - degrading the template versus simply averaging
    unaligned trials, the same failure shape as this project's earlier
    max-over-lags classifier scoring (see this module's own docstring).
    Smoothing over a window comparable to the genuine ERD/ERS onset-to-
    trough timescale (several hundred ms, not the oscillation's own
    period) removes the carrier's fast envelope wiggle while preserving
    the slower shape Woody's filter is actually meant to align.
    """
    if window_samples <= 1:
        return x
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(x, kernel, mode="same")


def cross_correlate_lag(trial: np.ndarray, template: np.ndarray, max_lag_samples: int) -> tuple[int, float]:
    """Finds the integer sample lag in [-max_lag_samples, max_lag_samples]
    maximizing normalized cross-correlation between trial and template,
    each candidate lag evaluated over their common overlap only. Returns
    (best_lag, best_correlation). best_lag > 0 means trial's content at
    time t matches template's content at time t - best_lag (i.e. trial is
    delayed relative to template by best_lag samples).
    """
    n = len(trial)
    best_lag, best_corr = 0, -np.inf
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag >= 0:
            a, b = trial[lag:], template[: n - lag]
        else:
            a, b = trial[: n + lag], template[-lag:]
        if len(a) < 10:
            continue
        a_c, b_c = a - a.mean(), b - b.mean()
        denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
        if denom < 1e-12:
            continue
        corr = float(np.dot(a_c, b_c) / denom)
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag, best_corr


def shift_signal(x: np.ndarray, lag: int) -> np.ndarray:
    """Shifts x by -lag samples (undoing a delay of `lag`), holding the
    edge value constant past the shifted boundary rather than wrapping
    (np.roll's default), so a shifted trial doesn't get contaminated by
    its own opposite-end content wrapping around."""
    shifted = np.roll(x, -lag)
    if lag > 0:
        shifted[-lag:] = x[-1]
    elif lag < 0:
        shifted[:-lag] = x[0]
    return shifted


def ride_two_component(
    trials: np.ndarray, max_lag_samples: int, n_iters: int = 5, c_smooth_window_samples: int = 1
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """A minimal, two-component version of RIDE (Ouyang, Herzmann, Zhou &
    Sommer 2011/2015): each trial is modeled as S(t) + C(t - lag_i) + noise,
    where S is a *fixed-latency* component (same timing on every trial,
    like a stimulus/cue-evoked response) and C is a *jitter-latency*
    component (Woody's filter's target, like this project's ERD/ERS onset).
    RIDE's actual documented advantage over applying Woody's filter to the
    raw, unseparated signal: estimating C's per-trial latency from the
    residual *after removing the current S estimate* is a cleaner problem
    than estimating it from the raw S+C mixture, since S doesn't move and
    so only ever adds interference to the cross-correlation, never signal.

    Iterates: (1) estimate each trial's C-latency by cross-correlating
    (trial - S_est) against the current C template, (2) refine the C
    template from the realigned residuals, (3) reconstruct each trial's own
    C contribution at its own (unaligned) latency and subtract it, (4)
    refine S_est from what's left (should have no jitter to fight, so a
    trial-average of the corrected residual). Returns (S_estimate,
    C_estimate, per_trial_lags in samples).

    trials: (n_trials, n_times), the raw S+C+noise mixture.

    Initialization matters and was a real bug the first time this was
    written: starting S_est at the naive trial mean and C_est at zero
    creates a self-consistent but degenerate fixed point - the first
    cross-correlation is against an all-zero template (cross_correlate_lag
    can't score any lag against it, so every trial defaults to lag=0),
    which makes C_est's first update exactly
    mean(trials - mean(trials)) = 0 by construction, and every later
    iteration reproduces the same all-zero C_est forever. Starting C_est at
    the naive trial mean instead (and S_est at zero) gives the first
    cross-correlation a genuine, if still S-contaminated, template to work
    with, breaking the degeneracy.

    c_smooth_window_samples > 1 smooths (smooth_signal) only the copies of
    the residual and template used for *lag-finding*, not the copies used
    to update C_est itself - the same "estimate on smoothed, apply to raw"
    separation validated in stage0_latency_alignment_validation.py's check
    3. Needed whenever C's own component carries intrinsic envelope
    randomness (an oscillatory-carrier-driven signal, not a clean
    transient) - without it, cross_correlate_lag mistakes that randomness
    for genuine latency structure, the same pitfall documented in
    smooth_signal's own docstring. Leave at the default (1, no smoothing)
    when C is already a clean, non-oscillatory shape.
    """
    n_trials = len(trials)
    S_est = np.zeros(trials.shape[1])
    C_est = trials.mean(axis=0)
    lags = [0] * n_trials

    for _ in range(n_iters):
        residual_for_C = trials - S_est[None, :]
        lags = []
        aligned_residuals = []
        c_est_for_lag_finding = smooth_signal(C_est, c_smooth_window_samples)
        for trial_residual in residual_for_C:
            residual_for_lag_finding = smooth_signal(trial_residual, c_smooth_window_samples)
            lag, _ = cross_correlate_lag(residual_for_lag_finding, c_est_for_lag_finding, max_lag_samples)
            lags.append(lag)
            aligned_residuals.append(shift_signal(trial_residual, lag))
        C_est = np.array(aligned_residuals).mean(axis=0)

        C_reconstructed = np.array([shift_signal(C_est, -lag) for lag in lags])  # each trial's C, at its own timing
        residual_for_S = trials - C_reconstructed
        S_est = residual_for_S.mean(axis=0)

    return S_est, C_est, lags


def woody_align(trials: np.ndarray, max_lag_samples: int, n_iters: int = 3) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Woody's adaptive filter. trials: (n_trials, n_times). Initializes
    the template as the naive (unaligned) mean, then alternates: estimate
    each trial's lag against the current template, shift trials by their
    estimated lag, re-average into a new template. Returns (aligned_trials,
    final_template, per_trial_lags in samples).

    Found in the literature (see this module's docstring) to have most of
    its value in the first iteration - later iterations can drift once the
    template starts reflecting the fit's own errors rather than the true
    shape. n_iters=3 is a starting point to check that behavior directly,
    not a value assumed safe.
    """
    template = trials.mean(axis=0)
    lags = [0] * len(trials)
    aligned = trials.copy()
    for _ in range(n_iters):
        lags = []
        aligned = []
        for trial in trials:
            lag, _ = cross_correlate_lag(trial, template, max_lag_samples)
            lags.append(lag)
            aligned.append(shift_signal(trial, lag))
        aligned = np.array(aligned)
        template = aligned.mean(axis=0)
    return aligned, template, lags
