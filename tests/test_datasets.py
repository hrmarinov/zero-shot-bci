"""Tests for the dataset layer (`src/datasets.py`).

Only the parts that are checkable without downloading anything are tested: the
channel-restriction maths, the montage invariants the cross-dataset work depends
on, and the guards that must fire *before* a download is attempted.
"""

import numpy as np
import pytest

from src.config import IV2A_CHANCE_LEVEL, IV2A_CLASSES, IV2A_SUBJECTS
from src.datasets import (
    IV2A_CHANNELS,
    OPENBMI_CHANNELS,
    PHYSIONET_CHANNELS,
    PHYSIONET_EXCLUDED_SUBJECTS,
    PHYSIONET_MI_EVENTS,
    SHARED_CHANNELS,
    load_physionet_reference_subject,
    load_physionet_reference_subject_labeled,
    restrict_channels,
)


def test_iv2a_setup_constants() -> None:
    assert IV2A_SUBJECTS == list(range(1, 10))
    assert len(IV2A_CLASSES) == 4
    assert IV2A_CHANCE_LEVEL == pytest.approx(0.25)


def test_channel_montages_have_the_expected_relationship() -> None:
    assert len(IV2A_CHANNELS) == 22
    # OpenBMI is missing FCz; every other IV-2a channel is present.
    assert set(IV2A_CHANNELS) - set(OPENBMI_CHANNELS) == {"FCz"}
    assert len(SHARED_CHANNELS) == 21
    assert "FCz" not in SHARED_CHANNELS
    # PhysioNet is a full superset - this is why it is the preferred pool.
    assert set(IV2A_CHANNELS) <= set(PHYSIONET_CHANNELS)


def test_shared_channels_preserve_iv2a_order() -> None:
    assert SHARED_CHANNELS == [c for c in IV2A_CHANNELS if c != "FCz"]


def test_restrict_channels_selects_and_reorders() -> None:
    X = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    out = restrict_channels(X, ["A", "B", "C"], ["C", "A"])
    assert out.shape == (2, 2, 4)
    assert np.array_equal(out[:, 0, :], X[:, 2, :])
    assert np.array_equal(out[:, 1, :], X[:, 0, :])


def test_restrict_channels_rejects_an_unknown_channel() -> None:
    X = np.zeros((1, 2, 3))
    with pytest.raises(ValueError):
        restrict_channels(X, ["A", "B"], ["A", "Z"])


def test_physionet_events_exclude_rest() -> None:
    # `n_classes` alone silently includes ~48% "rest" trials for this dataset,
    # so the explicit event list is load-bearing, not cosmetic.
    assert "rest" not in PHYSIONET_MI_EVENTS
    assert len(PHYSIONET_MI_EVENTS) == 4


def test_physionet_excluded_subject_is_refused_before_downloading() -> None:
    # Subject 88 was recorded at 128 Hz instead of 160 Hz. The guard must fire
    # on the argument check, not after fetching data.
    for subject in sorted(PHYSIONET_EXCLUDED_SUBJECTS):
        with pytest.raises(ValueError, match="excluded"):
            load_physionet_reference_subject(subject)
        with pytest.raises(ValueError, match="excluded"):
            load_physionet_reference_subject_labeled(subject)
