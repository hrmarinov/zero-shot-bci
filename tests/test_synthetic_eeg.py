"""Tests for the synthetic ground-truth generator (`src/synthetic_eeg.py`).

This generator is the project's validation gate for every new method: a result
only gets trusted on real data after the method recovers known ground truth
from this synthetic data. So the generator's own contract - deterministic under
a fixed seed, correct shapes, and the documented ERD/ERS envelope shape - needs
to hold, otherwise every downstream "validated on synthetic" claim is suspect.
"""

import numpy as np
import pytest

from src.synthetic_eeg import (
    SyntheticSubjectParams,
    erd_power_envelope,
    generate_synthetic_subject,
    sample_subject_params,
    spatial_weights,
)

CHANNELS = ["C3", "Cz", "C4", "CP3", "CP4"]


def test_generate_synthetic_subject_shapes_and_labels() -> None:
    dataset = generate_synthetic_subject(
        channel_names=CHANNELS,
        sfreq=64.0,
        n_trials_per_class=5,
        duration_s=4.0,
        snr_db=0.0,
        rng=np.random.default_rng(0),
    )
    assert dataset.X.shape == (10, len(CHANNELS), 256)
    assert dataset.y.shape == (10,)
    assert set(dataset.y) == {"left_hand", "right_hand"}
    assert dataset.channel_names == CHANNELS
    assert dataset.sfreq == 64.0


def test_generate_synthetic_subject_is_deterministic_given_a_seed() -> None:
    def build():
        return generate_synthetic_subject(
            channel_names=CHANNELS,
            sfreq=64.0,
            n_trials_per_class=4,
            duration_s=2.0,
            snr_db=5.0,
            rng=np.random.default_rng(42),
        )

    first, second = build(), build()
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.y, second.y)
    assert first.subject_params == second.subject_params


def test_erd_envelope_follows_the_documented_five_phase_shape() -> None:
    params = SyntheticSubjectParams(
        peak_freq_hz=10.0,
        onset_s=0.5,
        trough_s=1.5,
        erd_magnitude=0.5,  # power *decrease* (ERD)
        rebound_magnitude=0.3,
        spatial_sigma_m=0.06,
    )
    t = np.linspace(0.0, 4.0, 401)
    envelope = erd_power_envelope(t, params)

    def at(second: float) -> float:
        return float(envelope[np.argmin(np.abs(t - second))])

    assert at(0.0) == pytest.approx(1.0)  # pre-cue baseline
    assert at(1.5) == pytest.approx(1.0 - params.erd_magnitude)  # ERD trough
    assert at(2.5) == pytest.approx(1.0 + params.rebound_magnitude)  # rebound peak
    assert at(4.0) == pytest.approx(1.0)  # recovered to baseline
    assert envelope.min() == pytest.approx(1.0 - params.erd_magnitude, abs=1e-6)
    assert envelope.max() == pytest.approx(1.0 + params.rebound_magnitude, abs=1e-6)


def test_ers_subject_gets_a_power_increase_at_the_trough() -> None:
    # A minority of real people show ERS instead of ERD; the generator encodes
    # that as a negative erd_magnitude, and the envelope must respect it.
    params = SyntheticSubjectParams(
        peak_freq_hz=10.0,
        onset_s=0.5,
        trough_s=1.5,
        erd_magnitude=-0.4,
        rebound_magnitude=0.0,
        spatial_sigma_m=0.06,
    )
    t = np.linspace(0.0, 4.0, 401)
    envelope = erd_power_envelope(t, params)
    assert float(envelope[np.argmin(np.abs(t - 1.75))]) == pytest.approx(1.4)


def test_sampled_params_stay_in_their_documented_ranges() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        params = sample_subject_params(rng)
        assert 9.0 <= params.peak_freq_hz <= 12.0
        assert 0.3 <= params.onset_s <= 0.8
        assert params.trough_s > params.onset_s
        # Either a genuine ERD (positive) or the minority ERS case (negative),
        # never a meaningless near-zero effect.
        assert params.erd_magnitude <= -0.2 or params.erd_magnitude >= 0.25


def test_spatial_weights_peak_at_the_contralateral_channel() -> None:
    weights = spatial_weights(CHANNELS, "C3", sigma_m=0.06)
    assert len(weights) == len(CHANNELS)
    assert CHANNELS[int(np.argmax(weights))] == "C3"
    assert weights[CHANNELS.index("C3")] == pytest.approx(1.0)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    # Shrinking sigma must sharpen the focus on the target electrode.
    sharp = spatial_weights(CHANNELS, "C3", sigma_m=0.01)
    assert sharp.sum() < weights.sum()
