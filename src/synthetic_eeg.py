"""Synthetic EEG generator with known ERD/ERS ground truth.

Stage 0 of docs/wavelet_personalization_pipeline.md: real EEG never gives
ground truth, so this exists purely to let later stages be checked against a
known-correct answer before touching real data - the same validation pattern
that caught real bugs earlier this project (the SPD shrinkage estimator's
synthetic test, the closure-amplitude invariance check).

Deliberately varies the ERD/ERS envelope's timing, frequency, spatial
topology, and *direction* across synthetic subjects, mirroring a confirmed
real finding (some people show ERD where others show ERS for the identical
task) - a benchmark where every synthetic subject shares one template would
validate denoising but not personalization, which is the actual claim any
later stage needs to be checked against.
"""

from dataclasses import dataclass, replace

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

# Real 3D electrode positions (not hand-guessed coordinates) for physiologically
# accurate contralateral spatial falloff.
_MONTAGE_POSITIONS = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]

CONTRALATERAL_ELECTRODE = {"left_hand": "C4", "right_hand": "C3"}


@dataclass(frozen=True)
class SyntheticSubjectParams:
    """Per-subject ERD/ERS characteristics, randomized within realistic
    ranges. erd_magnitude > 0 is a power *decrease* during imagery (ERD,
    the textbook expectation); erd_magnitude < 0 is a power *increase*
    (ERS instead of ERD for this synthetic subject, matching the real
    literature finding that this happens for a minority of real people).
    """

    peak_freq_hz: float
    onset_s: float
    trough_s: float
    erd_magnitude: float
    rebound_magnitude: float
    spatial_sigma_m: float


def sample_subject_params(rng: np.random.Generator) -> SyntheticSubjectParams:
    is_ers_subject = rng.random() < 0.15  # matches literature: a minority show ERS not ERD
    erd_magnitude = rng.uniform(-0.5, -0.2) if is_ers_subject else rng.uniform(0.25, 0.75)
    return SyntheticSubjectParams(
        peak_freq_hz=rng.uniform(9.0, 12.0),
        onset_s=rng.uniform(0.3, 0.8),
        trough_s=rng.uniform(1.2, 2.0),
        erd_magnitude=erd_magnitude,
        rebound_magnitude=rng.uniform(0.0, 0.6),
        spatial_sigma_m=rng.uniform(0.04, 0.09),
    )


def _smooth_transition(t: np.ndarray, t0: float, t1: float, v0: float, v1: float) -> np.ndarray:
    """Raised-cosine interpolation from v0 at t0 to v1 at t1 (smoother, less
    trivially-detectable than a hard step, without needing real physiological
    modeling)."""
    frac = np.clip((t - t0) / (t1 - t0), 0, 1)
    smooth = 0.5 - 0.5 * np.cos(np.pi * frac)
    return v0 + (v1 - v0) * smooth


def erd_power_envelope(t: np.ndarray, params: SyntheticSubjectParams) -> np.ndarray:
    """Time-varying *power* gain (1.0 = baseline): baseline -> trough (ERD or
    ERS) -> post-movement rebound -> back to baseline. Piecewise-smooth, five
    phases."""
    onset, trough = params.onset_s, params.trough_s
    rebound_start = trough + 0.5
    rebound_peak = rebound_start + 0.5
    recovery_end = rebound_peak + 1.0

    v_baseline = 1.0
    v_trough = 1.0 - params.erd_magnitude
    v_rebound = 1.0 + params.rebound_magnitude

    env = np.full_like(t, v_baseline)
    m1 = (t >= onset) & (t < trough)
    env[m1] = _smooth_transition(t[m1], onset, trough, v_baseline, v_trough)
    m2 = (t >= trough) & (t < rebound_start)
    env[m2] = v_trough
    m3 = (t >= rebound_start) & (t < rebound_peak)
    env[m3] = _smooth_transition(t[m3], rebound_start, rebound_peak, v_trough, v_rebound)
    m4 = (t >= rebound_peak) & (t < recovery_end)
    env[m4] = _smooth_transition(t[m4], rebound_peak, recovery_end, v_rebound, v_baseline)
    return env


def spatial_weights(channel_names: list[str], contralateral_channel: str, sigma_m: float) -> np.ndarray:
    """Gaussian falloff (real 3D electrode distance) around the contralateral
    electrode - full effect at that electrode, decaying with true anatomical
    distance, not an arbitrary channel-index-based falloff."""
    center = _MONTAGE_POSITIONS[contralateral_channel]
    weights = np.zeros(len(channel_names))
    for i, ch in enumerate(channel_names):
        dist = np.linalg.norm(_MONTAGE_POSITIONS[ch] - center)
        weights[i] = np.exp(-0.5 * (dist / sigma_m) ** 2)
    return weights


def _band_limited_carrier(n_times: int, sfreq: float, center_hz: float, rng: np.random.Generator) -> np.ndarray:
    """White noise bandpass-filtered to a ~4Hz window around center_hz -
    a band-limited oscillation without the unrealistic perfect periodicity
    a pure sinusoid would have (and without accidentally building an easy
    case closer to SSVEP than motor imagery)."""
    white = rng.normal(size=n_times)
    sos = butter(4, [center_hz - 2.0, center_hz + 2.0], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, white)


def _pink_noise(n_times: int, rng: np.random.Generator) -> np.ndarray:
    """1/f-shaped background noise via FFT-domain amplitude scaling -
    standard technique, closer to real EEG's spectral shape than white noise."""
    white = rng.normal(size=n_times)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_times)
    freqs[0] = freqs[1]  # avoid divide-by-zero at DC
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n_times)
    return pink / (pink.std() + 1e-12)


def generate_trial(
    params: SyntheticSubjectParams,
    label: str,
    channel_names: list[str],
    sfreq: float,
    duration_s: float,
    snr_db: float,
    rng: np.random.Generator,
    onset_jitter_std_s: float = 0.0,
) -> np.ndarray:
    """One synthetic trial: spatially-weighted, ERD/ERS-modulated carrier
    plus background noise at a controlled SNR. Returns (n_channels, n_times).

    onset_jitter_std_s (default 0, preserving every existing Stage 0 gate's
    exact-timing behavior): standard deviation of a per-*trial* Gaussian
    shift applied to this trial's onset/trough (the whole envelope shifts
    together, keeping the onset-to-trough gap fixed - the simplest jitter
    model, matching real trial-to-trial reaction-time-like variability
    around a stable subject-level mean). Added specifically to test whether
    trial-averaged template fitting and fixed-lag per-trial matched-filter
    scoring degrade under jitter as hypothesized (see
    wavelet_calibration.py's per-subject real-data diagnostics, where
    fitted onset kept hugging its parameter floor in a way consistent with
    real trial-to-trial jitter smearing a trial-averaged fit) - a known-
    ground-truth way to confirm that diagnosis before trusting a fix for it.
    """
    n_times = int(duration_s * sfreq)
    t = np.arange(n_times) / sfreq

    if onset_jitter_std_s > 0:
        jitter = rng.normal(0.0, onset_jitter_std_s)
        params = replace(params, onset_s=params.onset_s + jitter, trough_s=params.trough_s + jitter)

    carrier = _band_limited_carrier(n_times, sfreq, params.peak_freq_hz, rng)
    power_gain = erd_power_envelope(t, params)
    amplitude_gain = np.sqrt(np.clip(power_gain, 1e-6, None))
    signal_1d = carrier * amplitude_gain

    weights = spatial_weights(channel_names, CONTRALATERAL_ELECTRODE[label], params.spatial_sigma_m)
    signal = weights[:, np.newaxis] * signal_1d[np.newaxis, :]

    # SNR is defined at the peak (weight=1) channel, i.e. "if you were sitting
    # right at the contralateral electrode, this is the SNR you'd have" - and
    # the SAME absolute noise power is then added to every channel (uniform
    # sensor/background noise, as in real recordings), not noise scaled by
    # each channel's own signal power. This is deliberate: it's what makes
    # the spatial falloff observable at all. A channel far from the
    # contralateral electrode has a much weaker *signal* but the *same*
    # noise floor, so its effective SNR - and therefore how much ERD/ERS
    # modulation is actually detectable there - drops with distance, exactly
    # matching real spatial specificity. Scaling noise by the *trial-average*
    # signal power (across all channels, dominated by the few high-weight
    # ones) was tried first and produced a bug where a channel with near-zero
    # spatial weight still showed clearly measurable "ERD" purely because its
    # noise floor was set too low relative to its own (tiny) residual signal
    # - caught by the physiological-plausibility check in the verification
    # script, not by this code merely running without errors.
    peak_signal_power = np.mean(signal_1d**2) + 1e-12
    noise_power_target = peak_signal_power / (10 ** (snr_db / 10))
    noise = np.array([_pink_noise(n_times, rng) for _ in channel_names])
    per_channel_noise_power = np.mean(noise**2, axis=1, keepdims=True) + 1e-12
    noise = noise * np.sqrt(noise_power_target / per_channel_noise_power)

    return signal + noise


@dataclass
class SyntheticDataset:
    X: np.ndarray  # (n_trials, n_channels, n_times)
    y: np.ndarray  # (n_trials,) of "left_hand"/"right_hand"
    subject_params: SyntheticSubjectParams  # ground truth for this "subject"
    channel_names: list[str]
    sfreq: float


def generate_synthetic_subject(
    channel_names: list[str],
    sfreq: float,
    n_trials_per_class: int,
    duration_s: float,
    snr_db: float,
    rng: np.random.Generator,
    onset_jitter_std_s: float = 0.0,
) -> SyntheticDataset:
    """One synthetic "subject": own randomized ERD/ERS parameters, n_trials_per_class
    left_hand and right_hand trials."""
    params = sample_subject_params(rng)
    X, y = [], []
    for label in ["left_hand", "right_hand"]:
        for _ in range(n_trials_per_class):
            X.append(generate_trial(params, label, channel_names, sfreq, duration_s, snr_db, rng, onset_jitter_std_s))
            y.append(label)
    order = rng.permutation(len(y))
    X = np.array(X)[order]
    y = np.array(y)[order]
    return SyntheticDataset(X=X, y=y, subject_params=params, channel_names=channel_names, sfreq=sfreq)
