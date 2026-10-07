"""Per-session wavelet calibration via backprop against a known-shape target.

Replaces Module A's population-shrinkage mechanism (see geometric_correction.py's
"Gated, bounded correction" note - shrinking toward a cross-dataset population
mean was found to actively hurt, not help). This mechanism never borrows from
other subjects or sessions at all: it fits a small set of parameters (per-
channel wavelet center frequency + bandwidth, plus a shared ERD/ERS envelope
shape) *directly* against one session's own cue-locked trials, using gradient
descent to match a known, textbook-plausible target shape - the same
"eye exam with a known letter chart" idea discussed in this session's
conversation, rather than "assume you're near the population average."

Two sessions of the *same* subject, fit independently this way, land in the
same canonical target space without ever being compared to each other or to
any other subject - the fitted parameters are what "corrects the room," per
session, the way you'd re-calibrate a voice assistant to a new room's
acoustics using a known test phrase rather than assuming every room sounds
like the average room.

Design choices:

- **Envelope shape reused, not reinvented**: the target shape is the exact
  same 5-phase piecewise envelope from synthetic_eeg.py's
  erd_power_envelope (baseline -> onset -> trough -> hold -> rebound ->
  recovery), ported to torch and made differentiable w.r.t. its own
  parameters. The *only* thing assumed a priori is the qualitative shape
  (smooth, unimodal dip-then-partial-rebound, physiologically bounded
  timing) - never a specific borrowed number. onset/trough/magnitude are
  fit fresh per session, not fixed to a population average.
- **Per-channel gain, shared timing**: matches synthetic_eeg.py's own
  generative structure (one shared temporal envelope x per-channel spatial
  weight), which is a defensible simplification of real ERD/ERS physiology
  (the same movement-related desynchronization event is time-locked across
  the scalp; only its magnitude varies by electrode distance from the
  motor cortex). channel_gain can be negative (ERS-favoring channel) or
  near zero (task-irrelevant channel) - nothing constrains its sign.
- **Bounded, safely-initialized parameterization**: every fitted quantity
  is reparameterized (sigmoid/softplus) to stay in a physiologically
  plausible range and to start at a "predict nothing unusual" default
  (channel_gain=0, erd_magnitude=0) - the same safe-default philosophy as
  geometric_correction.py's gate, for the same reason: an optimizer that
  starts confidently wrong is harder to trust than one that must earn its
  correction via gradient signal from the data.
- **Normalization handled by construction**: dividing each channel's power
  by its own pre-cue baseline power makes the fit scale-free automatically
  (no separate trace-normalization step needed, unlike spd_shrinkage.py -
  there's no cross-session or cross-hardware comparison happening here at
  all, so there's no scale-mismatch confound to correct for in the first
  place).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-8

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FREQ_MIN_HZ, FREQ_MAX_HZ = 3.0, 35.0
SIGMA_MIN_S, SIGMA_MAX_S = 0.02, 0.5
DEFAULT_INIT_FREQ_HZ = 10.0
DEFAULT_INIT_SIGMA_S = 0.15
DEFAULT_KERNEL_HALF_WIDTH_S = 0.4

ONSET_MIN_S, ONSET_MAX_S = 0.1, 1.0  # kept physiologically grounded (0.1s = already
                                       # faster than typical voluntary-response reaction time)
TROUGH_GAP_MIN_S = 0.15  # loosened from 0.3 - real data (see diagnose_wavelet_onset.py
                          # findings) wants a faster onset-to-trough transition than 0.3s
                          # allowed; a fast *transition* is more physiologically defensible
                          # to relax than pushing onset itself earlier than 0.1s
DEFAULT_INIT_ONSET_S = 0.5
DEFAULT_INIT_TROUGH_GAP_S = 1.0
DEFAULT_INIT_REBOUND_MAGNITUDE = 0.2
# Multi-start onsets for fit_wavelet_calibration_multistart - a real subject's
# fit was found to land on a completely degenerate solution (trough pinned
# near the end of the window) from a single late (0.9s) initialization while
# converging to a sensible answer from other starts. See wavelet_calibration
# module history / diagnose_wavelet_onset.py for the finding that motivated
# this: single-init fitting isn't reliable enough to trust on its own.
DEFAULT_MULTISTART_ONSETS_S = (0.15, 0.4, 0.7)

DEFAULT_N_EPOCHS = 400
DEFAULT_LR = 0.02


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _inverse_sigmoid_scaled(value: float, lo: float, hi: float) -> float:
    return _logit((value - lo) / (hi - lo))


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def _smooth_transition_torch(t: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    frac = torch.clamp((t - t0) / (t1 - t0), 0.0, 1.0)
    smooth = 0.5 - 0.5 * torch.cos(torch.pi * frac)
    return v0 + (v1 - v0) * smooth


def erd_envelope_torch(t: torch.Tensor, onset: torch.Tensor, trough: torch.Tensor, erd_magnitude: torch.Tensor, rebound_magnitude: torch.Tensor) -> torch.Tensor:
    """Differentiable port of synthetic_eeg.py's erd_power_envelope. t is a
    fixed (n_times,) tensor; onset/trough/erd_magnitude/rebound_magnitude
    are scalars (possibly requiring grad). torch.where evaluates every
    branch for every element regardless of which is selected, so each
    segment's formula must stay well-defined (no NaN/inf) outside its own
    window - _smooth_transition_torch's internal clamp already guarantees
    this.
    """
    v_baseline = torch.ones((), dtype=t.dtype)
    v_trough = 1.0 - erd_magnitude
    rebound_start = trough + 0.5
    rebound_peak = rebound_start + 0.5
    recovery_end = rebound_peak + 1.0
    v_rebound = 1.0 + rebound_magnitude

    seg1 = _smooth_transition_torch(t, onset, trough, v_baseline, v_trough)
    seg3 = _smooth_transition_torch(t, rebound_start, rebound_peak, v_trough, v_rebound)
    seg4 = _smooth_transition_torch(t, rebound_peak, recovery_end, v_rebound, v_baseline)

    return torch.where(
        t < onset, v_baseline,
        torch.where(
            t < trough, seg1,
            torch.where(
                t < rebound_start, v_trough,
                torch.where(t < rebound_peak, seg3, torch.where(t < recovery_end, seg4, v_baseline)),
            ),
        ),
    )


class MorletBank(nn.Module):
    """Per-channel learnable complex Morlet wavelet bank - a single (center
    frequency, bandwidth) pair per channel, not a multi-scale decomposition,
    matching the pipeline doc's "2 numbers x n_channels" personalized-wavelet
    design. forward(x) returns the analytic power envelope via depthwise
    convolution (each channel gets its own kernel, via conv1d groups=n_channels)."""

    def __init__(
        self,
        n_channels: int,
        sfreq: float,
        kernel_half_width_s: float = DEFAULT_KERNEL_HALF_WIDTH_S,
        init_freq_hz: float = DEFAULT_INIT_FREQ_HZ,
        init_sigma_s: float = DEFAULT_INIT_SIGMA_S,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.sfreq = sfreq

        kernel_half_samples = int(round(kernel_half_width_s * sfreq))
        kernel_t = torch.arange(-kernel_half_samples, kernel_half_samples + 1, dtype=torch.float32) / sfreq
        self.register_buffer("kernel_t", kernel_t)
        self.padding = kernel_half_samples

        init_freq_logit = _inverse_sigmoid_scaled(init_freq_hz, FREQ_MIN_HZ, FREQ_MAX_HZ)
        init_sigma_logit = _inverse_sigmoid_scaled(init_sigma_s, SIGMA_MIN_S, SIGMA_MAX_S)
        self.raw_freq = nn.Parameter(torch.full((n_channels,), init_freq_logit, dtype=torch.float32))
        self.raw_sigma = nn.Parameter(torch.full((n_channels,), init_sigma_logit, dtype=torch.float32))

    @property
    def center_freq(self) -> torch.Tensor:
        return FREQ_MIN_HZ + (FREQ_MAX_HZ - FREQ_MIN_HZ) * torch.sigmoid(self.raw_freq)

    @property
    def sigma(self) -> torch.Tensor:
        return SIGMA_MIN_S + (SIGMA_MAX_S - SIGMA_MIN_S) * torch.sigmoid(self.raw_sigma)

    def _kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.kernel_t.unsqueeze(0)  # (1, K)
        f_c = self.center_freq.unsqueeze(1)  # (n_channels, 1)
        sigma = self.sigma.unsqueeze(1)  # (n_channels, 1)

        gauss = torch.exp(-t**2 / (2 * sigma**2))
        phase = 2 * torch.pi * f_c * t
        norm = 1.0 / (sigma * (2 * torch.pi) ** 0.5)

        real_kernel = (norm * gauss * torch.cos(phase)).unsqueeze(1)  # (n_channels, 1, K)
        imag_kernel = (norm * gauss * torch.sin(phase)).unsqueeze(1)
        return real_kernel, imag_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_trials, n_channels, n_times) -> power envelope, same shape."""
        real_kernel, imag_kernel = self._kernels()
        real_part = F.conv1d(x, real_kernel, padding=self.padding, groups=self.n_channels)
        imag_part = F.conv1d(x, imag_kernel, padding=self.padding, groups=self.n_channels)
        # conv1d with an odd kernel and padding=K//2 can emit one extra
        # trailing sample versus the input length; trim to match exactly.
        real_part = real_part[..., : x.shape[-1]]
        imag_part = imag_part[..., : x.shape[-1]]
        return real_part**2 + imag_part**2


@dataclass
class WaveletCalibrationResult:
    wavelet: MorletBank
    onset_s: float
    trough_s: float
    erd_magnitude: float
    rebound_magnitude: float
    channel_gain: np.ndarray
    channel_names: list[str]
    train_losses: list[float]


def score_trials(result: WaveletCalibrationResult, X: np.ndarray, sfreq: float, t0_s: float, baseline_window_s: tuple[float, float]) -> np.ndarray:
    """Per-trial match loss against a *fitted* template - same MSE formula
    fit_wavelet_calibration trains against, but evaluated per-trial (no
    trial averaging, no further optimization) for new data.

    Lower = this trial's own power pattern matches the fitted class
    template more closely. The natural classification rule this enables:
    fit one WaveletCalibrationResult per class on calibration-session
    trials, then predict whichever class's template gives an eval trial
    the lowest score - a template-matching classifier that falls directly
    out of the calibration mechanism, no separate classifier needed.
    """
    device = next(result.wavelet.parameters()).device
    n_trials, n_channels, n_times = X.shape
    t = t0_s + torch.arange(n_times, dtype=torch.float32, device=device) / sfreq
    baseline_mask = (t >= baseline_window_s[0]) & (t < baseline_window_s[1])

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        power = result.wavelet(X_t)  # (n_trials, n_channels, n_times)
        baseline_power = power[:, :, baseline_mask].mean(dim=2, keepdim=True)
        normalized_power = power / (baseline_power + EPS)

        onset_t = torch.tensor(result.onset_s, device=device)
        trough_t = torch.tensor(result.trough_s, device=device)
        erd_t = torch.tensor(result.erd_magnitude, device=device)
        rebound_t = torch.tensor(result.rebound_magnitude, device=device)
        shared_envelope = erd_envelope_torch(t, onset_t, trough_t, erd_t, rebound_t)  # (n_times,)
        channel_gain_t = torch.tensor(result.channel_gain, dtype=torch.float32, device=device)
        predicted = 1.0 + channel_gain_t.unsqueeze(1) * (shared_envelope.unsqueeze(0) - 1.0)  # (n_channels, n_times)

        per_trial_mse = ((normalized_power - predicted.unsqueeze(0)) ** 2).mean(dim=(1, 2))  # (n_trials,)

    return per_trial_mse.cpu().numpy()


def match_filter_score(result: WaveletCalibrationResult, X: np.ndarray, sfreq: float, t0_s: float, baseline_window_s: tuple[float, float]) -> np.ndarray:
    """Normalized matched-filter correlation (cosine similarity) against a
    fitted template - higher is a better match. Use this for
    classification; use score_trials (MSE) only for the fitting objective
    itself.

    Found necessary after Phase 4's first pass at nearest-template
    classification landed at exactly chance level (25.0% +/- 1.7% across
    all 9 IV-2a subjects, 4 classes) - including *in-sample*, classifying
    the same calibration trials the templates were fit on (see
    diagnose_template_matching.py). That ruled out cross-session
    generalization as the cause: the scoring rule itself was broken.

    MSE = mean(observed^2) - 2*mean(observed*predicted) + mean(predicted^2).
    Only the middle (cross) term reflects genuine co-modulation between the
    trial and the template; the mean(observed^2) term reflects that single
    trial's own noise/power scale, which has nothing to do with which class
    it belongs to - and dominated the metric badly enough (per-class score
    std running 2-3x the mean) to erase whatever separable signal the
    trial-averaged templates actually captured.

    Dropping that term alone (an *unnormalized* correlation) fixed part of
    the problem but introduced another scale confound: templates fit with a
    larger overall channel_gain/erd_magnitude produce systematically larger
    raw correlation values against *any* input, including unrelated trials,
    so a "loud" template (e.g. one class happening to fit a stronger
    effect) wins more often regardless of true match quality - confirmed
    directly (true=feet trials scored *higher* against a large-magnitude
    tongue template than against their own feet template). Normalizing by
    the template's own RMS magnitude (making this a cosine similarity, the
    standard matched-filter normalization) makes templates of different
    magnitudes comparable on equal footing.
    """
    device = next(result.wavelet.parameters()).device
    n_trials, n_channels, n_times = X.shape
    t = t0_s + torch.arange(n_times, dtype=torch.float32, device=device) / sfreq
    baseline_mask = (t >= baseline_window_s[0]) & (t < baseline_window_s[1])

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        power = result.wavelet(X_t)
        baseline_power = power[:, :, baseline_mask].mean(dim=2, keepdim=True)
        normalized_power = power / (baseline_power + EPS)
        observed_dev = normalized_power - 1.0

        onset_t = torch.tensor(result.onset_s, device=device)
        trough_t = torch.tensor(result.trough_s, device=device)
        erd_t = torch.tensor(result.erd_magnitude, device=device)
        rebound_t = torch.tensor(result.rebound_magnitude, device=device)
        shared_envelope = erd_envelope_torch(t, onset_t, trough_t, erd_t, rebound_t)
        channel_gain_t = torch.tensor(result.channel_gain, dtype=torch.float32, device=device)
        predicted_dev = channel_gain_t.unsqueeze(1) * (shared_envelope.unsqueeze(0) - 1.0)  # (n_channels, n_times)
        template_rms = torch.sqrt((predicted_dev**2).mean() + EPS)

        raw_correlation = (observed_dev * predicted_dev.unsqueeze(0)).mean(dim=(1, 2))  # (n_trials,)
        per_trial_score = raw_correlation / template_rms

    return per_trial_score.cpu().numpy()


DEFAULT_JITTER_TOLERANCE_S = 0.15


def _gaussian_smooth_time(x: torch.Tensor, sfreq: float, sigma_s: float) -> torch.Tensor:
    """Blur a (n_channels, n_times) signal along time with a Gaussian kernel
    of std sigma_s (depthwise conv1d, same kernel every channel)."""
    if sigma_s <= 0:
        return x
    half_width = max(1, int(round(3 * sigma_s * sfreq)))
    kernel_t = torch.arange(-half_width, half_width + 1, dtype=torch.float32, device=x.device) / sfreq
    kernel = torch.exp(-(kernel_t**2) / (2 * sigma_s**2))
    kernel = (kernel / kernel.sum()).view(1, 1, -1).expand(x.shape[0], 1, -1)
    smoothed = F.conv1d(x.unsqueeze(0), kernel, padding=half_width, groups=x.shape[0])
    return smoothed[..., : x.shape[-1]].squeeze(0)


def match_filter_score_per_channel(
    result: WaveletCalibrationResult, X: np.ndarray, sfreq: float, t0_s: float, baseline_window_s: tuple[float, float],
    jitter_tolerance_s: float = DEFAULT_JITTER_TOLERANCE_S,
) -> np.ndarray:
    """Per-channel, jitter-tolerant matched-filter score. Returns
    (n_trials, n_channels) instead of match_filter_score's pooled
    (n_trials,) - a richer feature representation for a downstream
    classifier instead of one number per class.

    Two changes from match_filter_score, both empirically motivated:

    1. **Per-channel, not pooled across channels.** Pooling into one scalar
       throws away exactly the information a classifier could use to learn
       *which* channels are actually informative for a given class, instead
       of committing upfront to a single fixed spatial weighting baked into
       the template's own channel_gain.
    2. **Jitter-tolerant via a smoothed template, not a per-trial lag
       search.** Confirmed on synthetic data with *known* injected trial-
       to-trial onset jitter (diagnose_jitter_hypothesis.py): a jitter std
       of just 0.1s - a physiologically modest amount of reaction-time-like
       variability - dropped fixed-lag classification accuracy from 75.0%
       to 67.5%, and reproduced the exact onset-underestimation bias seen
       fitting real IV-2a data. The first fix tried was a per-trial max
       over a window of candidate lags - the "obvious" jitter-tolerance
       trick - but it made accuracy *worse* even at zero jitter (75.0% ->
       58.1%): taking the max over ~251 candidate alignments (times 22
       channels) is a multiple-comparisons problem, and a hard max
       systematically inflates *every* trial's score, including pure noise,
       enough to erase the true separation between classes (confirmed
       directly by testing it). Blurring the *template* by the expected
       jitter amount instead - a single, deterministic transform computed
       once, not a per-trial search - avoids that bias entirely: it widens
       the template's tolerance window without ever picking a
       suspiciously-good alignment out of many candidates.
    """
    device = next(result.wavelet.parameters()).device
    n_trials, n_channels, n_times = X.shape
    t = t0_s + torch.arange(n_times, dtype=torch.float32, device=device) / sfreq
    baseline_mask = (t >= baseline_window_s[0]) & (t < baseline_window_s[1])

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        power = result.wavelet(X_t)
        baseline_power = power[:, :, baseline_mask].mean(dim=2, keepdim=True)
        normalized_power = power / (baseline_power + EPS)
        observed_dev = normalized_power - 1.0  # (n_trials, n_channels, n_times)

        onset_t = torch.tensor(result.onset_s, device=device)
        trough_t = torch.tensor(result.trough_s, device=device)
        erd_t = torch.tensor(result.erd_magnitude, device=device)
        rebound_t = torch.tensor(result.rebound_magnitude, device=device)
        shared_envelope = erd_envelope_torch(t, onset_t, trough_t, erd_t, rebound_t)
        channel_gain_t = torch.tensor(result.channel_gain, dtype=torch.float32, device=device)
        predicted_dev = channel_gain_t.unsqueeze(1) * (shared_envelope.unsqueeze(0) - 1.0)  # (n_channels, n_times)
        predicted_dev = _gaussian_smooth_time(predicted_dev, sfreq, jitter_tolerance_s)
        template_rms_per_channel = torch.sqrt((predicted_dev**2).mean(dim=1) + EPS)  # (n_channels,)

        raw_correlation = (observed_dev * predicted_dev.unsqueeze(0)).mean(dim=2)  # (n_trials, n_channels)
        per_channel_score = raw_correlation / template_rms_per_channel.unsqueeze(0)

    return per_channel_score.cpu().numpy()


N_PHASE_WINDOWS = 5


def match_filter_score_windowed(
    result: WaveletCalibrationResult, X: np.ndarray, sfreq: float, t0_s: float, baseline_window_s: tuple[float, float],
    jitter_tolerance_s: float = DEFAULT_JITTER_TOLERANCE_S,
) -> np.ndarray:
    """Per-channel, per-phase-window, jitter-tolerant matched-filter score.
    Returns (n_trials, n_channels, 5) instead of match_filter_score_per_channel's
    (n_trials, n_channels) - adds temporal resolution the same way per-
    channel scoring added spatial resolution over the original single
    pooled scalar.

    Windows follow the fitted template's own 5-phase envelope structure
    (see erd_envelope_torch): baseline (before onset), onset-transition,
    trough/hold, rebound-transition, recovery. Splitting by each template's
    *own* fitted boundaries (not a fixed clock-time grid) keeps the windows
    physiologically meaningful per class, since different classes fit
    different onset/trough timing.

    A window with fewer than 2 samples for this X's time axis (e.g. the
    recovery window falling outside a shorter real epoch) scores 0 for
    every trial in that window rather than raising - a graceful "no
    information available here" rather than a crash, since callers may
    reuse one result across differently-windowed X.
    """
    device = next(result.wavelet.parameters()).device
    n_trials, n_channels, n_times = X.shape
    t = t0_s + torch.arange(n_times, dtype=torch.float32, device=device) / sfreq
    baseline_mask = (t >= baseline_window_s[0]) & (t < baseline_window_s[1])

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        power = result.wavelet(X_t)
        baseline_power = power[:, :, baseline_mask].mean(dim=2, keepdim=True)
        normalized_power = power / (baseline_power + EPS)
        observed_dev = normalized_power - 1.0  # (n_trials, n_channels, n_times)

        onset_t = torch.tensor(result.onset_s, device=device)
        trough_t = torch.tensor(result.trough_s, device=device)
        erd_t = torch.tensor(result.erd_magnitude, device=device)
        rebound_t = torch.tensor(result.rebound_magnitude, device=device)
        shared_envelope = erd_envelope_torch(t, onset_t, trough_t, erd_t, rebound_t)
        channel_gain_t = torch.tensor(result.channel_gain, dtype=torch.float32, device=device)
        predicted_dev = channel_gain_t.unsqueeze(1) * (shared_envelope.unsqueeze(0) - 1.0)  # (n_channels, n_times)
        predicted_dev = _gaussian_smooth_time(predicted_dev, sfreq, jitter_tolerance_s)

        onset_s, trough_s = result.onset_s, result.trough_s
        rebound_start_s = trough_s + 0.5
        rebound_peak_s = rebound_start_s + 0.5
        recovery_end_s = rebound_peak_s + 1.0
        windows = [
            (-float("inf"), onset_s),
            (onset_s, trough_s),
            (trough_s, rebound_start_s),
            (rebound_start_s, rebound_peak_s),
            (rebound_peak_s, float("inf")),
        ]

        window_scores = []
        for w_start, w_end in windows:
            mask = (t >= w_start) & (t < w_end)
            if int(mask.sum().item()) < 2:
                window_scores.append(torch.zeros(n_trials, n_channels, device=device))
                continue
            obs_w = observed_dev[:, :, mask]
            pred_w = predicted_dev[:, mask]
            template_rms_w = torch.sqrt((pred_w**2).mean(dim=1) + EPS)  # (n_channels,)
            raw_corr_w = (obs_w * pred_w.unsqueeze(0)).mean(dim=2)  # (n_trials, n_channels)
            window_scores.append(raw_corr_w / template_rms_w.unsqueeze(0))

        per_window_score = torch.stack(window_scores, dim=2)  # (n_trials, n_channels, 5)

    return per_window_score.cpu().numpy()


def fit_wavelet_calibration(
    X: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    t0_s: float,
    baseline_window_s: tuple[float, float],
    n_epochs: int = DEFAULT_N_EPOCHS,
    lr: float = DEFAULT_LR,
    random_state: int = 42,
    init_onset_s: float = DEFAULT_INIT_ONSET_S,
    init_trough_gap_s: float = DEFAULT_INIT_TROUGH_GAP_S,
    device: torch.device | str = DEFAULT_DEVICE,
    verbose: bool = False,
) -> WaveletCalibrationResult:
    """Fit wavelet + envelope-shape parameters to one session's trials of a
    single condition (e.g. all left_hand trials), via gradient descent
    against the shared target shape - see module docstring.

    X: (n_trials, n_channels, n_times) raw (unfiltered) EEG for one
    condition, epoched relative to the cue at t=0 with t0_s the time (in
    seconds, negative for pre-cue) of the first sample.
    baseline_window_s: (start, end) relative to the cue, used to normalize
    each channel's power to a dimensionless ~1.0-at-baseline scale before
    comparing to the target envelope.

    Every trainable quantity here is initialized deterministically (not
    from random_state - there is nothing stochastic in this fit at all,
    same input always converges to the same optimum), so random_state only
    matters if a caller wants reproducible *hashing*/bookkeeping elsewhere.
    init_onset_s/init_trough_gap_s exist specifically so callers can test
    initialization sensitivity - if refits from different starting points
    converge to very different answers, that's a sign of a poorly-
    conditioned optimization landscape, not a confirmed physiological
    finding.

    device defaults to CUDA when available - the dominant cost here is a
    depthwise conv1d over long (multi-second) real trials, repeated every
    epoch, which is exactly the kind of workload a GPU accelerates well
    versus the small-tensor-heavy geometric_correction.py MLP (where CPU
    was already fast enough that GPU transfer overhead wasn't worth it).
    """
    torch.manual_seed(random_state)
    n_trials, n_channels, n_times = X.shape
    t = t0_s + torch.arange(n_times, dtype=torch.float32, device=device) / sfreq

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    wavelet = MorletBank(n_channels, sfreq).to(device)

    raw_onset = nn.Parameter(torch.tensor(_inverse_sigmoid_scaled(init_onset_s, ONSET_MIN_S, ONSET_MAX_S), device=device))
    raw_trough_gap = nn.Parameter(torch.tensor(_inverse_softplus(init_trough_gap_s - TROUGH_GAP_MIN_S), device=device))
    raw_erd = nn.Parameter(torch.tensor(0.0, device=device))  # erd_magnitude starts at 0 - "no effect" is the safe default
    raw_rebound = nn.Parameter(torch.tensor(_inverse_softplus(DEFAULT_INIT_REBOUND_MAGNITUDE), device=device))
    channel_gain = nn.Parameter(torch.zeros(n_channels, device=device))  # starts at 0 - no channel presumed special

    params = list(wavelet.parameters()) + [raw_onset, raw_trough_gap, raw_erd, raw_rebound, channel_gain]
    optimizer = torch.optim.Adam(params, lr=lr)

    baseline_mask = (t >= baseline_window_s[0]) & (t < baseline_window_s[1])
    if baseline_mask.sum() < 2:
        raise ValueError(f"baseline_window_s={baseline_window_s} selects fewer than 2 samples given t0_s={t0_s}")

    train_losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        power = wavelet(X_t)
        trial_avg_power = power.mean(dim=0)  # (n_channels, n_times)
        baseline_power = trial_avg_power[:, baseline_mask].mean(dim=1, keepdim=True)
        normalized_power = trial_avg_power / (baseline_power + EPS)

        onset = ONSET_MIN_S + (ONSET_MAX_S - ONSET_MIN_S) * torch.sigmoid(raw_onset)
        trough = onset + TROUGH_GAP_MIN_S + F.softplus(raw_trough_gap)
        erd_magnitude = 2.0 * torch.tanh(raw_erd)
        rebound_magnitude = F.softplus(raw_rebound)

        shared_envelope = erd_envelope_torch(t, onset, trough, erd_magnitude, rebound_magnitude)
        predicted = 1.0 + channel_gain.unsqueeze(1) * (shared_envelope.unsqueeze(0) - 1.0)

        loss = F.mse_loss(normalized_power, predicted)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"wavelet calibration loss became non-finite at epoch {epoch} (loss={loss.item()}) - "
                "not something to silently train through"
            )
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"    epoch {epoch:4d}  loss={loss.item():.5f}  onset={onset.item():.3f}  "
                  f"trough={trough.item():.3f}  erd_mag={erd_magnitude.item():.3f}  "
                  f"rebound_mag={rebound_magnitude.item():.3f}")

    with torch.no_grad():
        onset = ONSET_MIN_S + (ONSET_MAX_S - ONSET_MIN_S) * torch.sigmoid(raw_onset)
        trough = onset + TROUGH_GAP_MIN_S + F.softplus(raw_trough_gap)
        erd_magnitude = 2.0 * torch.tanh(raw_erd)
        rebound_magnitude = F.softplus(raw_rebound)

    return WaveletCalibrationResult(
        wavelet=wavelet,
        onset_s=onset.item(),
        trough_s=trough.item(),
        erd_magnitude=erd_magnitude.item(),
        rebound_magnitude=rebound_magnitude.item(),
        channel_gain=channel_gain.detach().cpu().numpy(),
        channel_names=channel_names,
        train_losses=train_losses,
    )


def fit_wavelet_calibration_multistart(
    X: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    t0_s: float,
    baseline_window_s: tuple[float, float],
    init_onsets_s: tuple[float, ...] = DEFAULT_MULTISTART_ONSETS_S,
    n_epochs: int = DEFAULT_N_EPOCHS,
    lr: float = DEFAULT_LR,
    random_state: int = 42,
    device: torch.device | str = DEFAULT_DEVICE,
    verbose: bool = False,
) -> WaveletCalibrationResult:
    """fit_wavelet_calibration from several onset initializations, keeping
    the lowest-final-training-loss result.

    Found necessary, not just cautious: a single real-subject fit from a
    late (0.9s) onset init converged to a degenerate solution (trough
    pinned near the end of the fitting window, i.e. "no real transition
    found") while other inits on the *same* data converged to a sensible
    answer - see diagnose_wavelet_onset.py. The optimization landscape here
    has real local optima; one fixed starting point isn't reliable enough
    to trust on its own.
    """
    best_result = None
    best_loss = float("inf")
    for init_onset_s in init_onsets_s:
        result = fit_wavelet_calibration(
            X, channel_names, sfreq, t0_s, baseline_window_s,
            n_epochs=n_epochs, lr=lr, random_state=random_state,
            init_onset_s=init_onset_s, device=device, verbose=False,
        )
        final_loss = result.train_losses[-1]
        if verbose:
            print(f"    [multistart init_onset={init_onset_s:.2f}] final_loss={final_loss:.5f}  "
                  f"onset={result.onset_s:.3f}  trough={result.trough_s:.3f}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_result = result

    assert best_result is not None
    return best_result
