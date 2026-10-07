"""MOABB dataset loaders/wrappers.

BCI Competition IV-2a (MOABB: BNCI2014_001) ships two sessions per subject,
named by MOABB "0train" (calibration) and "1test" (evaluation). These are
recorded on different days and are the standard calibration-vs-live-session
split used to measure cross-session accuracy drop.
"""

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
import requests
from moabb.datasets import BNCI2014_001, BNCI2014_004, Lee2019_MI, PhysionetMI
from moabb.paradigms import MotorImagery

from src.config import FMAX, FMIN, TMAX, TMIN

DOWNLOAD_RETRIES = 5
DOWNLOAD_RETRY_DELAY_S = 15

T = TypeVar("T")


def _fetch_with_retry(fetch: Callable[[], T], description: str) -> T:
    """Retry a MOABB fetch call on transient network failures.

    Both OpenBMI's and PhysioNet's hosts have been observed to drop the
    connection mid-download intermittently - retrying is safe since pooch
    verifies file integrity and only re-downloads what's actually incomplete,
    never anything already cached.
    """
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            return fetch()
        except requests.exceptions.RequestException as exc:
            if attempt == DOWNLOAD_RETRIES:
                raise
            print(
                f"[{description}] download failed (attempt {attempt}/{DOWNLOAD_RETRIES}): "
                f"{exc!r} - retrying in {DOWNLOAD_RETRY_DELAY_S}s"
            )
            time.sleep(DOWNLOAD_RETRY_DELAY_S)
    raise AssertionError("unreachable")  # loop always returns or raises


CALIB_SESSION = "0train"
EVAL_SESSION = "1test"

IV2B_CALIB_SESSIONS = {"0train", "1train", "2train"}
IV2B_EVAL_SESSIONS = {"3test", "4test"}

# Fixed channel order for each dataset (same montage for every subject within
# a single study) - verified once against the raw MNE info, not re-derived
# per subject/call.
IV2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
OPENBMI_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "FC3",
    "FC4", "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4", "P1", "P2", "POz",
    "FT9", "FTT9h", "TTP7h", "TP7", "TPP9h", "FT10", "FTT10h", "TPP8h",
    "TP8", "TPP10h", "F9", "F10", "AF7", "AF3", "AF4", "AF8", "PO3", "PO4",
]
PHYSIONET_CHANNELS = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "Fp1",
    "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1",
    "Fz", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7", "T8", "T9", "T10",
    "TP7", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
]
# IV-2a's 22 channels minus FCz (the one not present in OpenBMI's montage),
# in IV-2a's original order - the shared reference space both datasets get
# restricted to for cross-dataset population statistics.
SHARED_CHANNELS = [c for c in IV2A_CHANNELS if c in OPENBMI_CHANNELS]

# PhysioNet's montage is a full superset of IV-2a's 22 channels (verified,
# unlike OpenBMI which is missing FCz) - use the full 22-channel set when
# combining with PhysioNet specifically.
assert all(c in PHYSIONET_CHANNELS for c in IV2A_CHANNELS), (
    "PhysioNet channel list no longer covers all of IV-2a's channels - "
    "verify PHYSIONET_CHANNELS against the actual montage before proceeding"
)

# Documented in moabb's PhysionetMI docstring: subject 88 was recorded at
# 128Hz instead of 160Hz like every other subject, and mixing it in without
# excluding it corrupts any downstream sampling-rate-dependent processing.
# Verified directly (not just trusted from the docstring) - see the
# investigation in this project's session history before this constant was
# added.
PHYSIONET_EXCLUDED_SUBJECTS = {88}
# MOABB's MotorImagery paradigm silently ignores n_classes when `events` is
# not also given explicitly, for datasets whose event_id has more entries
# than n_classes - verified directly: n_classes=2 alone against PhysionetMI
# returned all 5 raw event types including "rest" (48% of returned trials),
# not the 2 classes requested. Explicit events= is required to get a clean,
# correctly-filtered set - excluding "rest" specifically, since a reference
# pool meant to characterize engaged motor-imagery covariance structure
# should not be contaminated with non-task baseline trials.
PHYSIONET_MI_EVENTS = ["left_hand", "right_hand", "feet", "hands"]


def restrict_channels(X: np.ndarray, source_channels: list[str], target_channels: list[str]) -> np.ndarray:
    """Select and reorder X's channel axis (axis=1) from source_channels'
    order to target_channels' order. target_channels must be a subset of
    source_channels."""
    idx = [source_channels.index(name) for name in target_channels]
    return X[:, idx, :]


@dataclass
class SubjectSessions:
    subject: int
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray


def load_iv2a_subject(subject: int) -> SubjectSessions:
    """Load one subject's calibration and evaluation session trials for IV-2a.

    Returns band-passed (8-30 Hz), epoched (0.5-2.5s post-cue) motor-imagery
    trials split by MOABB's native session labels, ready for CSP-based
    classifiers.
    """
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])

    calib_mask = metadata["session"] == CALIB_SESSION
    eval_mask = metadata["session"] == EVAL_SESSION

    return SubjectSessions(
        subject=subject,
        X_calib=X[calib_mask.to_numpy()],
        y_calib=y[calib_mask.to_numpy()],
        X_eval=X[eval_mask.to_numpy()],
        y_eval=y[eval_mask.to_numpy()],
    )


WAVELET_CALIB_FMIN, WAVELET_CALIB_FMAX = 1.0, 40.0
WAVELET_CALIB_TMIN, WAVELET_CALIB_TMAX = -1.0, 4.0


def load_iv2a_subject_wide_window(subject: int) -> SubjectSessions:
    """load_iv2a_subject, but epoched -1 to 4s relative to the cue (verified
    safely available - no truncation/error) instead of 0.5-2.5s, and only
    lightly band-limited (1-40Hz) instead of the CSP-tuned 8-30Hz.

    For src/adaptation/wavelet_calibration.py's per-session fitting: it
    needs the pre-cue baseline (t<0, absent from load_iv2a_subject's window
    entirely) to normalize power against, and enough post-cue duration for
    the full ERD/rebound envelope (up to ~3.5s post-cue in the worst case
    within the fitted parameter ranges). It also intentionally does *not*
    pre-constrain the signal to the 8-30Hz mu/beta band the way the CSP
    pipeline does - the wavelet's own center frequency is what's supposed
    to discover which band actually carries this subject's ERD/ERS, and
    pre-filtering to 8-30Hz would silently rule out that search finding
    anything outside it.
    """
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4, fmin=WAVELET_CALIB_FMIN, fmax=WAVELET_CALIB_FMAX,
        tmin=WAVELET_CALIB_TMIN, tmax=WAVELET_CALIB_TMAX,
    )
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])

    calib_mask = metadata["session"] == CALIB_SESSION
    eval_mask = metadata["session"] == EVAL_SESSION

    return SubjectSessions(
        subject=subject,
        X_calib=X[calib_mask.to_numpy()],
        y_calib=y[calib_mask.to_numpy()],
        X_eval=X[eval_mask.to_numpy()],
        y_eval=y[eval_mask.to_numpy()],
    )


def load_iv2a_subject_cleaned(subject: int) -> SubjectSessions:
    """load_iv2a_subject, but from artifact-cleaned (notch + wide bandpass
    + ICA-based EOG/muscle removal, see src/preprocessing.py) cached
    continuous recordings - cleaned once per subject/session and cached,
    not repeated on every call. Applies the same CSP-tuned 8-30Hz band and
    0.5-2.5s epoching as load_iv2a_subject on top of the shared cleaned
    cache, so this is a drop-in replacement with the same output shape and
    scale (microvolts, same channel order) - not a differently-shaped
    dataset requiring downstream code changes.
    """
    import mne

    from src.preprocessing import get_cleaned_session_raw

    dataset = BNCI2014_001()

    def epoch(session_name: str) -> tuple[np.ndarray, np.ndarray]:
        raw = get_cleaned_session_raw(dataset, subject, session_name)
        raw = raw.copy().filter(l_freq=FMIN, h_freq=FMAX, picks="eeg", verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(
            raw, events, event_id, tmin=TMIN, tmax=TMAX, picks="eeg",
            baseline=None, preload=True, verbose=False,
        )
        X = epochs.get_data(copy=False) * 1e6  # volts -> microvolts, matching load_iv2a_subject's scale
        id_to_label = {v: k for k, v in event_id.items()}
        y = np.array([id_to_label[code] for code in epochs.events[:, 2]])
        return X, y

    X_calib, y_calib = epoch(CALIB_SESSION)
    X_eval, y_eval = epoch(EVAL_SESSION)

    return SubjectSessions(subject=subject, X_calib=X_calib, y_calib=y_calib, X_eval=X_eval, y_eval=y_eval)


def load_iv2a_subject_shared_channels(subject: int) -> SubjectSessions:
    """load_iv2a_subject, restricted to the 21 channels IV-2a shares with
    OpenBMI (drops FCz), for combining with an OpenBMI-derived population
    prior."""
    data = load_iv2a_subject(subject)
    return SubjectSessions(
        subject=data.subject,
        X_calib=restrict_channels(data.X_calib, IV2A_CHANNELS, SHARED_CHANNELS),
        y_calib=data.y_calib,
        X_eval=restrict_channels(data.X_eval, IV2A_CHANNELS, SHARED_CHANNELS),
        y_eval=data.y_eval,
    )


def load_openbmi_reference_subject(subject: int) -> np.ndarray:
    """Load one OpenBMI (Lee2019_MI) subject's trials (MOABB's MotorImagery
    paradigm returns one recording session per subject for this dataset,
    ~100 trials - plenty for a well-estimated reference mean, well above the
    low-data regime this experiment probes), restricted to the 21 channels
    shared with IV-2a in matching order. Unsupervised use only (population-
    prior reference) - labels are not returned since the reference pool
    never needs them.

    The host serving these files (a large ~600MB .mat per subject/session)
    drops the connection mid-download intermittently - retries with a short
    delay, since pooch verifies and re-downloads only the incomplete file,
    not anything already cached.
    """
    dataset = Lee2019_MI()
    paradigm = MotorImagery(n_classes=2, fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)

    def fetch():
        X, _, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
        return restrict_channels(X, OPENBMI_CHANNELS, SHARED_CHANNELS)

    return _fetch_with_retry(fetch, f"load_openbmi_reference_subject subject={subject}")


def load_physionet_reference_subject(subject: int) -> np.ndarray:
    """Load one PhysioNet (EEG Motor Movement/Imagery) subject's trials,
    restricted to IV-2a's full 22-channel set (verified: PhysioNet's montage
    is a complete superset, unlike OpenBMI which is missing FCz). Unsupervised
    use only - labels discarded, same as the OpenBMI loader.

    Uses only the 4 genuine motor-imagery event types (left_hand, right_hand,
    feet, hands) via explicit `events=`, deliberately not `n_classes` alone -
    see PHYSIONET_MI_EVENTS' docstring for why that matters (n_classes alone
    silently includes ~48% "rest" trials for this dataset). Raises ValueError
    for subject 88 (different sampling rate, see PHYSIONET_EXCLUDED_SUBJECTS)
    rather than silently returning inconsistent data.
    """
    if subject in PHYSIONET_EXCLUDED_SUBJECTS:
        raise ValueError(
            f"PhysioNet subject {subject} is excluded (recorded at a different "
            f"sampling rate than the rest of the dataset - see PHYSIONET_EXCLUDED_SUBJECTS)"
        )

    dataset = PhysionetMI()
    paradigm = MotorImagery(
        events=PHYSIONET_MI_EVENTS, n_classes=len(PHYSIONET_MI_EVENTS),
        fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX,
    )

    def fetch():
        X, _, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
        return restrict_channels(X, PHYSIONET_CHANNELS, IV2A_CHANNELS)

    return _fetch_with_retry(fetch, f"load_physionet_reference_subject subject={subject}")


def load_physionet_reference_subject_labeled(subject: int) -> tuple[np.ndarray, np.ndarray]:
    """load_physionet_reference_subject, but also returns labels.

    Needed for src/adaptation/prototypical_network.py's episodic meta-
    training, which - unlike the shrinkage/Module A population-prior use
    of this reference pool - needs each PhysioNet subject's trials split
    by class, not pooled as an unsupervised reference mean.
    """
    if subject in PHYSIONET_EXCLUDED_SUBJECTS:
        raise ValueError(
            f"PhysioNet subject {subject} is excluded (recorded at a different "
            f"sampling rate than the rest of the dataset - see PHYSIONET_EXCLUDED_SUBJECTS)"
        )

    dataset = PhysionetMI()
    paradigm = MotorImagery(
        events=PHYSIONET_MI_EVENTS, n_classes=len(PHYSIONET_MI_EVENTS),
        fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX,
    )

    def fetch():
        X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
        return restrict_channels(X, PHYSIONET_CHANNELS, IV2A_CHANNELS), y

    return _fetch_with_retry(fetch, f"load_physionet_reference_subject_labeled subject={subject}")


def load_iv2b_subject(subject: int) -> SubjectSessions:
    """Load one subject's calibration and evaluation trials for IV-2b.

    IV-2b ships 5 sessions per subject (3 channels, 2-class L/R hand); the
    dataset's own canonical split pools the first 3 sessions as calibration
    and the last 2 as evaluation (MOABB session labels 0train/1train/2train
    vs. 3test/4test) - used as-is here for a same-protocol generalization
    check against the IV-2a cross-session results.
    """
    dataset = BNCI2014_004()
    paradigm = MotorImagery(n_classes=2, fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])

    sessions = metadata["session"]
    calib_mask = sessions.isin(IV2B_CALIB_SESSIONS)
    eval_mask = sessions.isin(IV2B_EVAL_SESSIONS)

    return SubjectSessions(
        subject=subject,
        X_calib=X[calib_mask.to_numpy()],
        y_calib=y[calib_mask.to_numpy()],
        X_eval=X[eval_mask.to_numpy()],
        y_eval=y[eval_mask.to_numpy()],
    )
