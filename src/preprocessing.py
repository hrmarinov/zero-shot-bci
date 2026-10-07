"""Artifact-cleaned, cached raw EEG recordings.

Every experiment so far in this project has loaded IV-2a through MOABB's
MotorImagery paradigm directly (src/datasets.py's load_iv2a_subject),
which band-passes and epochs but never removes eye-blink, saccade, or
muscle artifacts - the 3 EOG channels IV-2a ships are present in the raw
recording but were never used for anything. This module does that
cleaning once per subject/session and caches the result, so downstream
loaders don't repeat an expensive ICA fit on every run.

Pipeline, applied to each session's *continuous* raw recording (all runs
concatenated first, then cleaned as one unit - ICA needs enough data to
estimate stable components, and ICA/filtering must happen before epoching
since epoch boundaries would otherwise cut into filter edge effects):
notch filter (line frequency taken from *the recording itself* - 50Hz and
its 100Hz harmonic for these Graz, Austria recordings, but 60Hz for e.g.
OpenBMI (Korea) or PhysioNet (US); see `_notch_freqs_for`) -> a
general-purpose wide bandpass
(1-40Hz, matching wavelet_calibration.py's existing "lightly band-limited"
convention, not any one analysis's own narrower band - CSP's 8-30Hz or any
other downstream filtering is applied separately, on top of this cache,
by each loader that needs it) -> ICA fit on the filtered EEG channels only
-> automatic exclusion of components correlated with the EOG channels
(ica.find_bads_eog) or showing a muscle-artifact spectral signature
(ica.find_bads_muscle) -> reconstruction with those components removed.

Cached as MNE's native .fif format (preserves channel info, annotations/
events, everything downstream code needs) under data/cleaned/, keyed by
subject and session - never re-cleaned once cached. Delete the relevant
file(s) under data/cleaned/ to force a re-clean (e.g. after changing
NOTCH_FREQS, BANDPASS, or the ICA parameters below).
"""

import math
from pathlib import Path

import mne

from src.config import DATA_DIR

CLEANED_DIR = DATA_DIR / "cleaned"
NOTCH_FREQS = [50.0, 100.0]  # fallback: European mains (Graz, Austria) + first harmonic


def _notch_freqs_for(raw: mne.io.BaseRaw) -> list[float]:
    """Line-noise frequencies to notch for *this particular recording*.

    Prefers the recording's own `line_freq` (MOABB populates it per dataset from that
    dataset's metadata: 50 Hz for the Graz IV-2a/2b sets, 60 Hz for OpenBMI, recorded in
    Korea, and PhysioNet, recorded in the US) and falls back to `NOTCH_FREQS` when the
    file carries none.

    Why this exists: this module used to hardcode 50 Hz, which is correct for IV-2a - the
    only dataset it currently cleans - but would silently leave 60 Hz line noise
    untouched on any other dataset while notching a frequency that has no interference
    there. The dataset audit (`results/char_dataset_audit.csv`, the `line_freq` column)
    surfaced that the per-dataset mains frequency genuinely differs, so the value is read
    from the data rather than assumed.

    For IV-2a this returns exactly the old constant `[50.0, 100.0]`, so the existing
    cache under `data/cleaned/` remains valid and no committed result changes.
    """
    line = raw.info.get("line_freq")
    try:
        line = float(line) if line is not None else None
    except (TypeError, ValueError):
        line = None
    if line is None or not math.isfinite(line) or line <= 0:
        return list(NOTCH_FREQS)
    return [line, 2.0 * line]
BANDPASS = (1.0, 40.0)  # general-purpose; narrower analysis-specific bands applied downstream, not baked in here
# A variance-fraction n_components (e.g. 0.99) was tried first and found
# broken: one dominant artifact captured 85.7% of this data's variance
# alone, so 0.99 kept only 7 components total. find_bads_eog/find_bads_muscle
# score each component as an outlier *relative to the others* - with only 7
# components there isn't enough of a distribution to reliably call anything
# an outlier, and both returned zero detections even for a component
# correlating at r=0.81 with an EOG channel. A fixed component count close
# to full rank (22 EEG channels) gives the detectors what they need -
# verified directly: the same data with n_components=20 found 2 genuine
# EOG components and 1 muscle component.
N_ICA_COMPONENTS = 20
ICA_RANDOM_STATE = 42


def clean_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Notch -> bandpass -> ICA-based EOG/muscle artifact removal. Returns
    a new Raw; does not mutate the input. Requires the input to still have
    its EOG channels (needed for find_bads_eog) - callers should not drop
    them before calling this.
    """
    raw = raw.copy().load_data()
    raw.notch_filter(_notch_freqs_for(raw), picks="eeg", verbose=False)
    raw.filter(l_freq=BANDPASS[0], h_freq=BANDPASS[1], picks="eeg", verbose=False)

    ica = mne.preprocessing.ICA(n_components=N_ICA_COMPONENTS, random_state=ICA_RANDOM_STATE, max_iter="auto")
    ica.fit(raw, picks="eeg", verbose=False)

    eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
    muscle_indices, _ = ica.find_bads_muscle(raw, verbose=False)
    ica.exclude = sorted(set(eog_indices) | set(muscle_indices))

    cleaned = raw.copy()
    ica.apply(cleaned, verbose=False)
    return cleaned


def get_cleaned_session_raw(dataset, subject: int, session_name: str) -> mne.io.BaseRaw:
    """Returns the artifact-cleaned, concatenated-across-runs continuous
    raw recording for one subject's one session, from cache if already
    cleaned, otherwise cleans it now and caches the result.
    """
    cache_path = CLEANED_DIR / f"subject{subject}_{session_name}_cleaned_raw.fif"
    if cache_path.exists():
        return mne.io.read_raw_fif(cache_path, preload=True, verbose=False)

    data = dataset.get_data(subjects=[subject])
    runs = data[subject][session_name]
    raw = mne.concatenate_raws([r.copy() for r in runs.values()], verbose=False)
    cleaned = clean_raw(raw)

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.save(cache_path, overwrite=True, verbose=False)
    return cleaned
