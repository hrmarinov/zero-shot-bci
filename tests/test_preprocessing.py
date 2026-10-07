"""Tests for `src/preprocessing.py`.

The important property here is a *non-change*: the artifact-cleaning pipeline is cached
under `data/cleaned/` for IV-2a, and the notch frequencies used to be a hardcoded
constant. Deriving them from the recording's own `line_freq` must return exactly the same
value for IV-2a, or every cached cleaned recording (and the committed
`results/cleaned_riemannian_check.csv`) would have been produced by a different filter than
the code now describes.

The second property is why the derivation exists at all: the dataset audit
(`results/char_dataset_audit.csv`) shows the mains frequency genuinely differs by dataset
(50 Hz for Graz IV-2a/2b, 60 Hz for OpenBMI in Korea and PhysioNet in the US), so a
hardcoded 50 Hz notch would silently leave 60 Hz line noise in place.
"""

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from src.preprocessing import NOTCH_FREQS, _notch_freqs_for


def _raw_with_line_freq(line_freq):
    """Minimal single-channel Raw carrying the requested line_freq."""
    info = mne.create_info(["Cz"], 100.0, ["eeg"])
    if line_freq is not None:
        info["line_freq"] = line_freq
    return mne.io.RawArray(np.zeros((1, 200)), info, verbose=False)


def test_iv2a_line_frequency_reproduces_the_old_hardcoded_constant():
    # IV-2a is recorded in Graz, Austria: 50 Hz mains. The old code used the constant
    # [50.0, 100.0] unconditionally, so this must match exactly for the cached cleaned
    # recordings to remain valid.
    assert _notch_freqs_for(_raw_with_line_freq(50.0)) == NOTCH_FREQS == [50.0, 100.0]


def test_sixty_hertz_dataset_does_not_get_a_fifty_hertz_notch():
    # OpenBMI (Korea) and PhysioNet (US) are 60 Hz. Notching 50 Hz there would notch a
    # frequency with no interference while leaving the real line noise untouched.
    assert _notch_freqs_for(_raw_with_line_freq(60.0)) == [60.0, 120.0]


def test_missing_line_frequency_falls_back_to_the_constant():
    raw = _raw_with_line_freq(None)
    assert raw.info.get("line_freq") is None
    assert _notch_freqs_for(raw) == NOTCH_FREQS
