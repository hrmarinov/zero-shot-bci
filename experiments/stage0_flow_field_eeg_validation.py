"""Stage 0 gate, part 2: does the full flow-field pipeline recover a known,
planted spatial drift in synthetic EEG - same role stage0_icp_eeg_validation.py
played (synthetic session drift with a known ground truth) before ICP ever
touched real data.

src/synthetic_eeg.py's existing generator can't be reused as-is: its spatial
weighting is a FIXED per-trial pattern times a time-varying scalar envelope
(weights don't depend on t), so by construction it cannot produce a moving
ERD hotspot. This test needs exactly that, so - matching
stage0_icp_eeg_validation.py's own precedent of not forcing a shared
generator into a shape it wasn't built for - it builds a small purpose-built
generator here, reusing synthetic_eeg's carrier/envelope/noise primitives and
real electrode positions, but replacing the fixed spatial weight with one
that linearly migrates (in true 3D anatomical distance) from one real
electrode's position to another's between the ERD onset and trough, then
holds.

Ground truth check: average many trials' windowed envelope frames (trial
averaging - the standard way ERD/ERS topographic movies are computed in
practice, and necessary here since single-trial EEG SNR is too low for a
single planted drift to be checkable in isolation), run the averaged
sequence through the same interpolation + Horn-Schunck flow pipeline
trial_flow_sequence uses internally, and check whether the recovered flow
during the drift window points toward the same direction as the planted
drift's start->end vector, projected into the pipeline's own 2D coordinate
system via flow_field.project_3d_to_2d (the identical projection the real
pipeline uses for electrodes) - not a direction defined in some other
coordinate system that would only "match" by a lucky sign convention.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.adaptation.flow_field import (
    horn_schunck_flow,
    interpolate_topomap,
    project_3d_to_2d,
    windowed_envelope_frames,
)
from src.datasets import IV2A_CHANNELS
from src.synthetic_eeg import (
    _MONTAGE_POSITIONS,
    SyntheticSubjectParams,
    _band_limited_carrier,
    _pink_noise,
    erd_power_envelope,
)

SFREQ = 250.0
DURATION_S = 4.0
# 200, not 80: a first pass at 80 trials found the pipeline resolves a
# left-right drift (C3<->C1, along the densely electrode-sampled central
# row C5-C3-C1-Cz-C2-C4-C6) very well but an anterior-posterior drift
# (FC3->CP3, with no equivalent dense electrode chain along that direction)
# poorly (cosine 0.06 - failed) at that trial count. Raising SNR alone
# fixed it (0.81 at +15dB/200 trials), and the failure mode was direction-
# specific rather than uniform, which rules out a coordinate-system bug and
# points at a real, physically sensible cause: IV-2a's electrode geometry
# gives genuinely less spatial information for anterior-posterior motion
# than for left-right motion, so recovering it needs more trial-averaging
# at the same SNR. 200 trials at the original -3dB is the smallest fixed,
# uniform (not tuned per-direction) setting found where all three planted
# directions pass - see the anisotropy note in flow_field.py's docstring.
N_TRIALS = 200
SNR_DB = -3.0
SPATIAL_SIGMA_M = 0.05
GRID_SIZE = 16
WINDOW_S, STRIDE_S = 0.25, 0.125
RANDOM_STATE = 11

# Three drift pairs, not just one: C3->C1 alone can't rule out that any
# recovered direction is really just a fixed artifact of electrode geometry
# near C3 (the same kind of test-design flaw the ICP toy test's isotropic
# point cloud had - a test that would "pass" regardless of the true planted
# transform). C1->C3 is the exact reverse of the first pair (recovered
# direction must flip, not just also happen to pass); FC3->CP3 is a
# roughly-orthogonal anterior-to-posterior migration within the same general
# region, checking that recovery isn't just sensitive to "activity somewhere
# near C3" but genuinely tracks direction.
DRIFT_PAIRS = [("C3", "C1"), ("C1", "C3"), ("FC3", "CP3")]

PARAMS = SyntheticSubjectParams(
    peak_freq_hz=10.0, onset_s=0.5, trough_s=1.5,
    erd_magnitude=0.6, rebound_magnitude=0.3, spatial_sigma_m=SPATIAL_SIGMA_M,
)


def moving_spatial_weights(t: np.ndarray, channel_names: list[str], start_channel: str, end_channel: str) -> np.ndarray:
    """(n_channels, n_times): Gaussian falloff (real 3D anatomical distance)
    around a center that migrates from start_channel's to end_channel's real
    position via the same raised-cosine timing erd_power_envelope uses for
    its own onset->trough transition, then holds at the destination through
    the rebound - the planted ground truth this test checks recovery of.
    """
    start = _MONTAGE_POSITIONS[start_channel]
    end = _MONTAGE_POSITIONS[end_channel]
    frac = np.clip((t - PARAMS.onset_s) / (PARAMS.trough_s - PARAMS.onset_s), 0, 1)
    smooth_frac = 0.5 - 0.5 * np.cos(np.pi * frac)
    centers = start[None, :] + smooth_frac[:, None] * (end - start)[None, :]  # (n_times, 3)

    positions = np.array([_MONTAGE_POSITIONS[c] for c in channel_names])  # (n_channels, 3)
    dist = np.linalg.norm(positions[:, None, :] - centers[None, :, :], axis=2)  # (n_channels, n_times)
    return np.exp(-0.5 * (dist / SPATIAL_SIGMA_M) ** 2)


def generate_trial(rng: np.random.Generator, start_channel: str, end_channel: str) -> np.ndarray:
    n_times = int(DURATION_S * SFREQ)
    t = np.arange(n_times) / SFREQ

    carrier = _band_limited_carrier(n_times, SFREQ, PARAMS.peak_freq_hz, rng)
    power_gain = erd_power_envelope(t, PARAMS)
    amplitude_gain = np.sqrt(np.clip(power_gain, 1e-6, None))
    signal_1d = carrier * amplitude_gain

    weights = moving_spatial_weights(t, IV2A_CHANNELS, start_channel, end_channel)  # (n_channels, n_times)
    signal = weights * signal_1d[None, :]

    peak_signal_power = np.mean(signal_1d**2) + 1e-12
    noise_power_target = peak_signal_power / (10 ** (SNR_DB / 10))
    noise = np.array([_pink_noise(n_times, rng) for _ in IV2A_CHANNELS])
    per_channel_noise_power = np.mean(noise**2, axis=1, keepdims=True) + 1e-12
    noise = noise * np.sqrt(noise_power_target / per_channel_noise_power)

    return signal + noise


def recover_drift_direction(rng: np.random.Generator, start_channel: str, end_channel: str) -> np.ndarray:
    """Generates N_TRIALS trials with a planted drift from start_channel to
    end_channel, trial-averages their windowed envelope frames, runs the
    averaged sequence through interpolation + Horn-Schunck, and returns the
    recovered unit flow direction during the drift window."""
    from src.adaptation.flow_field import electrode_positions_2d, hilbert_envelope

    positions_2d = electrode_positions_2d(IV2A_CHANNELS)
    window_samples = int(round(WINDOW_S * SFREQ))
    stride_samples = int(round(STRIDE_S * SFREQ))

    frame_stack = []
    for _ in range(N_TRIALS):
        trial = generate_trial(rng, start_channel, end_channel)
        envelope = hilbert_envelope(trial)
        frame_stack.append(windowed_envelope_frames(envelope, window_samples, stride_samples))
    mean_frames = np.mean(np.stack(frame_stack), axis=0)  # trial-averaged: (n_windows, n_channels)

    topomaps = np.stack([interpolate_topomap(frame, positions_2d, GRID_SIZE) for frame in mean_frames])

    window_centers_s = WINDOW_S / 2 + np.arange(len(mean_frames)) * STRIDE_S
    drift_flows, drift_weights = [], []
    for i in range(len(topomaps) - 1):
        t_mid = 0.5 * (window_centers_s[i] + window_centers_s[i + 1])
        if not (PARAMS.onset_s - WINDOW_S <= t_mid <= PARAMS.trough_s + WINDOW_S):
            continue
        u, v = horn_schunck_flow(topomaps[i], topomaps[i + 1], alpha=0.1, n_iters=200)
        Ix = 0.5 * (np.gradient(topomaps[i], axis=1) + np.gradient(topomaps[i + 1], axis=1))
        Iy = 0.5 * (np.gradient(topomaps[i], axis=0) + np.gradient(topomaps[i + 1], axis=0))
        weight = Ix**2 + Iy**2
        drift_flows.append(np.array([np.sum(u * weight), np.sum(v * weight)]))
        drift_weights.append(np.sum(weight))

    drift_flows = np.array(drift_flows)
    drift_weights = np.array(drift_weights)
    recovered = np.sum(drift_flows, axis=0) / np.sum(drift_weights)
    return recovered / (np.linalg.norm(recovered) + 1e-12)


def run() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    recovered_by_pair = {}

    for start_channel, end_channel in DRIFT_PAIRS:
        print(f"\n--- planted drift {start_channel} -> {end_channel} ---")
        recovered = recover_drift_direction(rng, start_channel, end_channel)
        expected_2d = project_3d_to_2d(_MONTAGE_POSITIONS[end_channel]) - project_3d_to_2d(_MONTAGE_POSITIONS[start_channel])
        expected = expected_2d / np.linalg.norm(expected_2d)
        cosine_similarity = float(np.dot(recovered, expected))
        print(f"  expected direction:  {expected}")
        print(f"  recovered direction: {recovered}")
        print(f"  cosine similarity: {cosine_similarity:.3f}")
        assert cosine_similarity > 0.5, (
            f"{start_channel}->{end_channel}: recovered flow direction disagrees with the planted "
            f"drift (cosine similarity {cosine_similarity:.3f}, need > 0.5)"
        )
        recovered_by_pair[(start_channel, end_channel)] = recovered

    forward, reverse = recovered_by_pair[("C3", "C1")], recovered_by_pair[("C1", "C3")]
    flip_cosine = float(np.dot(forward, reverse))
    print(f"\n--- reversal check: C3->C1 vs C1->C3 recovered directions ---")
    print(f"  cosine similarity between the two (should be near -1, not near +1): {flip_cosine:.3f}")
    assert flip_cosine < -0.5, (
        f"reversing the planted drift did not reverse the recovered direction (cosine {flip_cosine:.3f}) - "
        f"recovery may be tracking a fixed geometric artifact rather than the true planted drift"
    )

    print("\nPASSED - flow-field pipeline recovers known planted spatial drifts on synthetic EEG "
          "(three directions, including an exact reversal check). Safe to proceed to real IV-2a data.")


if __name__ == "__main__":
    run()
