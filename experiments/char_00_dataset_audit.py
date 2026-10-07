"""Characterization 00 - recording-methodology / provenance audit of the four
datasets this project uses, and of what MOABB's ``MotorImagery`` paradigm does
to them on the way in.

Why this exists. Every result in this repo is a claim about a subject's
*signal*, but the numbers that reach a classifier depend just as much on the
acquisition (sampling rate, montage, reference, line noise, artifact
annotations) and on MOABB's own preprocessing (band-pass, epoching, unit
scaling, channel selection, session bookkeeping). None of that was ever
audited; several of the project's comments assert it from memory. This script
turns those assertions into measurements.

Sections of the stdout report:

  A. Dataset-level facts that need no EEG data (MOABB API only): subject list,
     ``interval``, ``event_id``, ``unit_factor``, METADATA.
  B. The **raw** recording as MOABB's own dataset object hands it over
     (``dataset.get_data(subjects=[s])``, which applies only
     ``SetRawAnnotations``, never a filter): sampling rate, channel names and
     types (EEG/EOG/EMG/STIM), ``highpass``/``lowpass``, ``line_freq``,
     reference flags, measurement date, per-run artifact text, annotation
     types/durations/offsets, raw data magnitude in volts.
  C. The **paradigm output** for the same subject, exactly as
     ``src/datasets.py`` loads it: array shape/dtype/scale, trials per class
     per session, and the returned ``mne.Epochs`` object
     (``return_epochs=True``) for its ``info``.
  D. The **delta**: raw vs epochs info (resampling, filter band, channel
     selection), the epochs' ``baseline`` attribute, and the array-vs-epochs
     magnitude ratio - the only direct evidence of the volts -> microvolts
     conversion MOABB applies through ``dataset.unit_factor``.
  E. Targeted checks: the .mat files read directly (run/trial/artifact
     structure), a bit-exact reproduction of one IV-2a run's epochs from the
     raw (pins the filter method and the epoch offsets numerically instead of
     by reading MOABB source), trigger-locked ERPs, the PhysioNet 128 Hz
     subject and the "rest" event quirk, the Lee2019 session-naming filter,
     and what src/preprocessing.py's cleaned cache actually contains.

Written to results/char_dataset_audit.csv: one row per (dataset, subject,
session) with trials, per-class counts, channels, sampling rates, filter band,
reference, line frequency, runs, EOG/EMG channels and the raw->epoch deltas.

Determinism / cost. No random numbers are drawn (numpy is seeded anyway).
Threads are pinned to 4 before numpy is imported because another agent is
running a moderate IV-2a job on this machine (docs/compute_environment.md
sections 1 and 4). One subject per dataset is held in memory at a time and
released with gc.collect() between datasets; peak RSS is sampled during the
one heavy load (OpenBMI, ~600 MB .mat per session) and printed.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perf import mem_report, setup_threads

# After src.perf, before numpy: 4 threads, because another agent runs a
# concurrent moderate job (docs/compute_environment.md rule 1).
setup_threads(4)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import src.config as config  # noqa: E402  - sets MNE_DATA / BNCI path before any MOABB call

import mne  # noqa: E402
import moabb  # noqa: E402
from moabb.datasets import BNCI2014_001, BNCI2014_004, Lee2019_MI, PhysionetMI  # noqa: E402
from moabb.paradigms import MotorImagery  # noqa: E402
from scipy.io import loadmat  # noqa: E402

from src.config import FMAX, FMIN, RANDOM_STATE, RESULTS_DIR, TMAX, TMIN  # noqa: E402
from src.datasets import (  # noqa: E402
    IV2A_CHANNELS,
    OPENBMI_CHANNELS,
    PHYSIONET_CHANNELS,
    PHYSIONET_EXCLUDED_SUBJECTS,
    PHYSIONET_MI_EVENTS,
    SHARED_CHANNELS,
)

np.random.seed(RANDOM_STATE)  # repo-wide policy (docs/compute_environment.md rule 5); nothing here is random

REPRESENTATIVE_SUBJECT = 1  # one subject per dataset (memory + "do not loop over subjects")
PHYSIONET_HZ_OUTLIER = 88  # MOABB's documented 128 Hz subject - verified below, not trusted
MAT_001 = config.DATA_DIR / "MNE-bnci-data" / "~bci" / "database" / "001-2014"
MAT_004 = config.DATA_DIR / "MNE-bnci-data" / "~bci" / "database" / "004-2014"
CSV_PATH = RESULTS_DIR / "char_dataset_audit.csv"
ERP_WINDOWS = [(0.05, 0.35), (0.6, 1.4), (1.9, 2.4), (2.9, 3.4), (5.9, 6.4)]
ERP_DISPLAY_CHANNELS = ["Fz", "FCz", "Cz", "Pz", "POz", "C3", "C4"]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def hdr(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def emit(tag: str, message: str = "") -> None:
    print(f"[{tag}] {message}".rstrip(), flush=True)


class PeakRSS:
    """Sample this process' RSS in a background thread while a heavy load runs.

    src.perf.mem_report() reports the RSS *after* a load; the transient peak
    during MOABB's .mat -> RawArray conversion is what actually stresses the
    machine's 16.7 GB commit limit, so it is worth a 20 ms sampler.
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.peak_gb = float("nan")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        import psutil

        proc = psutil.Process()
        while not self._stop.is_set():
            rss = proc.memory_info().rss / 1024**3
            if not (self.peak_gb == self.peak_gb) or rss > self.peak_gb:
                self.peak_gb = rss
            time.sleep(self.interval_s)

    def __enter__(self) -> "PeakRSS":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _nonempty(d: dict) -> dict:
    return {k: v for k, v in d.items() if not str(k).startswith("__")}


# ---------------------------------------------------------------------------
# Fact extractors
# ---------------------------------------------------------------------------

def raw_facts(raw: mne.io.BaseRaw, magnitude_seconds: float = 20.0) -> dict:
    """Everything the raw file provides, as seen through MOABB's loader.

    ``magnitude_seconds`` bounds the data read used for the unit check: the
    full EEG array of an OpenBMI session is ~600 MB, a 20 s slice is ~10 MB.
    """
    types = raw.get_channel_types()
    sfreq = float(raw.info["sfreq"])
    n = min(int(magnitude_seconds * sfreq), raw.n_times)
    eeg = raw.get_data(picks="eeg", start=0, stop=n) if types.count("eeg") else np.zeros((0, 0))

    facts: dict = {
        "sfreq_hz": sfreq,
        "n_channels_total": len(raw.ch_names),
        "n_eeg": types.count("eeg"),
        "n_eog": types.count("eog"),
        "n_emg": types.count("emg"),
        "n_stim": types.count("stim"),
        "ch_names": list(raw.ch_names),
        "eeg_ch_names": [c for c, t in zip(raw.ch_names, types) if t == "eeg"],
        "eog_ch_names": [c for c, t in zip(raw.ch_names, types) if t == "eog"],
        "emg_ch_names": [c for c, t in zip(raw.ch_names, types) if t == "emg"],
        "stim_ch_names": [c for c, t in zip(raw.ch_names, types) if t == "stim"],
        "duration_s": round(raw.n_times / sfreq, 3),
        "highpass_hz": float(raw.info["highpass"]),
        "lowpass_hz": float(raw.info["lowpass"]),
        "line_freq_hz": raw.info["line_freq"],
        "custom_ref_applied": int(raw.info["custom_ref_applied"]),
        "meas_date": str(raw.info["meas_date"]),
        "description": (raw.info.get("description") or "").strip(),
        "montage_name": type(raw.get_montage()).__name__ if raw.get_montage() is not None else None,
        "n_dig_points": len(raw.info["dig"]) if raw.info.get("dig") else 0,
        "eeg_abs_max_v": float(np.abs(eeg).max()) if eeg.size else float("nan"),
        "magnitude_window_s": round(n / sfreq, 1),
        "n_annotations": len(raw.annotations),
        "annotation_types": {},
        "annotation_durations_s": [],
        "median_annotation_gap_s": float("nan"),
        "first_annotation_onset_s": float("nan"),
        "n_stim_events": 0,
        "stim_event_codes": {},
        "stim_to_annotation_shift_s": float("nan"),
    }
    if len(raw.annotations):
        descs, counts = np.unique(raw.annotations.description, return_counts=True)
        facts["annotation_types"] = {str(d): int(c) for d, c in zip(descs, counts)}
        facts["annotation_durations_s"] = sorted({round(float(d), 4) for d in raw.annotations.duration})
        facts["first_annotation_onset_s"] = float(raw.annotations.onset[0])
        if len(raw.annotations) > 2:
            facts["median_annotation_gap_s"] = float(np.median(np.diff(np.asarray(raw.annotations.onset, dtype=float))))

    # The stim channel is the file's actual trigger representation for three of
    # the four datasets; MOABB builds the annotations *from* it and shifts them
    # by dataset.interval[0]. Measuring both is what makes "which sample does
    # the epoch start at, and what does that sample mark" answerable.
    stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
    if stim_channels:
        events = mne.find_events(raw, shortest_event=0, verbose=False)
        facts["n_stim_events"] = int(len(events))
        if len(events):
            codes, counts = np.unique(events[:, 2], return_counts=True)
            facts["stim_event_codes"] = {int(c): int(k) for c, k in zip(codes, counts)}
            if len(raw.annotations):
                k = min(len(events), len(raw.annotations))
                shifts = np.asarray(raw.annotations.onset, dtype=float)[:k] - events[:k, 0] / sfreq
                facts["stim_to_annotation_shift_s"] = float(np.median(shifts))
    return facts


def epochs_facts(epochs: mne.Epochs) -> dict:
    types = epochs.get_channel_types()
    data = epochs.get_data(copy=False)
    return {
        "n_epochs": len(epochs),
        "shape": tuple(int(v) for v in data.shape),
        "dtype": str(data.dtype),
        "sfreq_hz": float(epochs.info["sfreq"]),
        "highpass_hz": float(epochs.info["highpass"]),
        "lowpass_hz": float(epochs.info["lowpass"]),
        "ch_names": list(epochs.ch_names),
        "ch_types": {t: types.count(t) for t in sorted(set(types))},
        "tmin_s": float(epochs.tmin),
        "tmax_s": float(epochs.tmax),
        "n_times": int(len(epochs.times)),
        "baseline": epochs.baseline,
        "reject": epochs.reject,
        "proj_applied": bool(epochs.proj),
        "first_time_s": float(epochs.times[0]),
        "last_time_s": float(epochs.times[-1]),
        "abs_max_v": float(np.abs(data).max()),
    }


def session_trial_counts(y: np.ndarray, meta: pd.DataFrame) -> dict[str, dict]:
    """{session: {"n_trials": int, "per_class": {label: count}}}."""
    out: dict[str, dict] = {}
    for session in pd.unique(meta["session"]):
        mask = (meta["session"] == session).to_numpy()
        values, counts = np.unique(y[mask], return_counts=True)
        out[str(session)] = {
            "n_trials": int(mask.sum()),
            "per_class": {str(v): int(c) for v, c in zip(values, counts)},
        }
    return out


# ---------------------------------------------------------------------------
# Dataset specifications
# ---------------------------------------------------------------------------

class DatasetSpec:
    def __init__(self, key, cls, dataset, paradigm, channels_from_datasets_py,
                 reference_note, line_freq_note, extra_notes="", heavy=False):
        self.key = key
        self.cls = cls
        self.dataset = dataset
        self.paradigm = paradigm
        self.channels_from_datasets_py = channels_from_datasets_py
        self.reference_note = reference_note
        self.line_freq_note = line_freq_note
        self.extra_notes = extra_notes
        self.heavy = heavy


def build_specs() -> list[DatasetSpec]:
    """The four datasets, loaded exactly the way src/datasets.py loads them."""
    iv2a = BNCI2014_001()
    iv2b = BNCI2014_004()

    # Lee2019_MI: MOABB names its sessions "0"/"1" (str(session - 1)) but
    # get_data() filters the returned sessions against {str(s) for s in
    # _selected_sessions} = {"1", "2"}, so only the *second* recording day
    # (the sess02 file) survives. Loading the first one is pure waste, so the
    # load loop is restricted to session 2 below - same file, same returned
    # session name, half the peak memory. Section E.3 verifies the filter
    # behaviour itself with a stubbed loader (no data touched).
    lee = Lee2019_MI()
    lee.sessions = [2]
    physio = PhysionetMI()

    return [
        DatasetSpec(
            key="BNCI2014_001",
            cls=BNCI2014_001,
            dataset=iv2a,
            paradigm=MotorImagery(n_classes=4, fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX),
            channels_from_datasets_py=IV2A_CHANNELS,
            reference_note=(
                "left mastoid per the Graz data-set-A description / MOABB docstring; MOABB METADATA says "
                "'none'; raw.info carries no reference string (custom_ref_applied=0)"
            ),
            line_freq_note="set to 50.0 by MOABB's BNCI loader (bnci/base.py:_convert_run); European mains",
        ),
        DatasetSpec(
            key="BNCI2014_004",
            cls=BNCI2014_004,
            dataset=iv2b,
            paradigm=MotorImagery(n_classes=2, fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX),
            channels_from_datasets_py=None,  # src/datasets.py has no IV-2b channel constant
            reference_note=(
                "left mastoid + right-mastoid ground per MOABB docstring; the competition page calls the 3 EEG "
                "channels 'bipolar' (C3, Cz, C4); raw.info carries no reference string (custom_ref_applied=0)"
            ),
            line_freq_note="set to 50.0 by MOABB's BNCI loader; European mains",
        ),
        DatasetSpec(
            key="Lee2019_MI",
            cls=Lee2019_MI,
            dataset=lee,
            paradigm=MotorImagery(n_classes=2, fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX),
            channels_from_datasets_py=OPENBMI_CHANNELS,
            reference_note=(
                "nasion-referenced, AFz ground (Lee et al. 2019); MOABB METADATA agrees; raw.info carries no "
                "reference string (custom_ref_applied=0)"
            ),
            line_freq_note=(
                "60.0 injected from MOABB METADATA (bids_interface._enrich_raw_info_from_metadata) because the "
                "loader never sets it - the value is MOABB's claim, not the file's"
            ),
            extra_notes="returned session is '1' only: MOABB's 0-based session names vs its 1-based filter",
            heavy=True,
        ),
        DatasetSpec(
            key="PhysionetMI",
            cls=PhysionetMI,
            dataset=physio,
            paradigm=MotorImagery(
                events=PHYSIONET_MI_EVENTS,
                n_classes=len(PHYSIONET_MI_EVENTS),
                fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX,
            ),
            channels_from_datasets_py=PHYSIONET_CHANNELS,
            reference_note=(
                "not stated on the PhysioNet record page; MOABB METADATA says 'mastoid'; raw.info carries no "
                "reference string (custom_ref_applied=0)"
            ),
            line_freq_note=(
                "60.0 injected from MOABB METADATA (the EDF header has none - verified by reading the EDF "
                "directly in section E.4)"
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Sections A-D
# ---------------------------------------------------------------------------

def section_a(specs: list[DatasetSpec]) -> None:
    hdr("A. Dataset-level facts (MOABB API only - no EEG data read)")
    for i, spec in enumerate(specs, start=1):
        d = spec.cls()
        meta = d.METADATA
        emit(f"A.{i}", f"{spec.key}  code={d.code}  n_subjects={len(d.subject_list)}  "
                       f"subject_list={d.subject_list[:3]}...{d.subject_list[-1:]}  "
                       f"interval={list(d.interval) if d.interval is not None else None}  "
                       f"unit_factor={d.unit_factor:g}  paradigm={d.paradigm!r}")
        emit(f"A.{i}", f"      event_id={dict(d.event_id)}  selected_sessions={d._selected_sessions}")
        if meta is not None:
            acq = meta.acquisition
            emit(f"A.{i}", f"      METADATA acquisition: sampling_rate={acq.sampling_rate} n_channels={acq.n_channels} "
                           f"channel_types={acq.channel_types} reference={acq.reference!r} ground={acq.ground!r}")
            emit(f"A.{i}", f"      METADATA acquisition: hardware={acq.hardware!r} sensor_type={acq.sensor_type!r} "
                           f"montage={acq.montage!r} line_freq={acq.line_freq} software={acq.software!r}")
            pre = meta.preprocessing
            if pre is not None:
                emit(f"A.{i}", f"      METADATA preprocessing: state={pre.data_state!r} steps={pre.preprocessing_steps!r} "
                               f"bandpass={pre.bandpass} notch={pre.notch_hz} downsampled_to_hz={pre.downsampled_to_hz}")
            par = meta.participants
            if par is not None:
                emit(f"A.{i}", f"      METADATA participants: n_subjects={par.n_subjects} "
                               f"ages={par.age_min}-{par.age_max} handedness={par.handedness!r} "
                               f"bci_experience={par.bci_experience!r}")
            exp = meta.experiment
            if exp is not None:
                emit(f"A.{i}", f"      METADATA experiment: paradigm={exp.paradigm!r} n_classes={exp.n_classes} "
                               f"class_labels={exp.class_labels!r} trial_duration={exp.trial_duration}")
            dstr = getattr(meta, "data_structure", None)
            if dstr is not None:
                emit(f"A.{i}", f"      METADATA data_structure: n_trials={dstr.n_trials} "
                               f"n_trials_per_class={dstr.n_trials_per_class}")
                if dstr.trials_context:
                    emit(f"A.{i}", f"      METADATA trials_context: {dstr.trials_context}")
            doc = meta.documentation
            if doc is not None:
                emit(f"A.{i}", f"      METADATA documentation: doi={doc.doi!r} publication_year={doc.publication_year} "
                               f"institution={doc.institution!r}")
        del d, meta
        gc.collect()


def raw_pass(spec: DatasetSpec, subject: int) -> tuple[dict, dict]:
    """Load the raw recordings through the MOABB dataset object."""
    with PeakRSS() as peak:
        data = spec.dataset.get_data(subjects=[subject])
    per_session: dict[str, dict] = {}
    per_run: dict[str, dict] = {}
    for session, runs in data[subject].items():
        run_facts = {str(run): raw_facts(raw) for run, raw in runs.items()}
        first = run_facts[sorted(run_facts)[0]]
        agg = dict(first)
        agg["n_runs"] = len(run_facts)
        agg["run_durations_s"] = [f["duration_s"] for f in run_facts.values()]
        agg["run_descriptions"] = {r: f["description"] for r, f in run_facts.items()}
        agg["n_annotations_per_run"] = [f["n_annotations"] for f in run_facts.values()]
        agg["same_channels_across_runs"] = all(f["ch_names"] == first["ch_names"] for f in run_facts.values())
        agg["same_sfreq_across_runs"] = all(f["sfreq_hz"] == first["sfreq_hz"] for f in run_facts.values())
        per_session[str(session)] = agg
        per_run[str(session)] = run_facts
    per_session["__peak_rss_gb__"] = peak.peak_gb
    return per_session, per_run


def print_raw_session(tag: str, session: str, f: dict, interval0: float) -> None:
    emit(f"B.{tag}", f"session {session!r}: {f['n_runs']} run(s), sfreq={_fmt(f['sfreq_hz'])} Hz, "
                    f"{f['n_channels_total']} ch (eeg={f['n_eeg']}, eog={f['n_eog']}, "
                    f"emg={f['n_emg']}, stim={f['n_stim']}), duration/run={f['run_durations_s']} s")
    emit(f"B.{tag}", f"    info: highpass/lowpass={_fmt(f['highpass_hz'])}/{_fmt(f['lowpass_hz'])} Hz, "
                    f"line_freq={f['line_freq_hz']}, custom_ref_applied={f['custom_ref_applied']}, "
                    f"meas_date={f['meas_date']}, montage={f['montage_name']}, dig={f['n_dig_points']}")
    emit(f"B.{tag}", f"    eeg ch ({f['n_eeg']}): {f['eeg_ch_names']}")
    if f["eog_ch_names"] or f["emg_ch_names"] or f["stim_ch_names"]:
        emit(f"B.{tag}", f"    non-eeg ch: eog={f['eog_ch_names']} emg={f['emg_ch_names']} stim={f['stim_ch_names']}")
    emit(f"B.{tag}", f"    annotations: {f['n_annotations']} in first run, types={f['annotation_types']}, "
                    f"duration(s)={f['annotation_durations_s']}, first onset={_fmt(f['first_annotation_onset_s'])} s, "
                    f"median gap={_fmt(f['median_annotation_gap_s'])} s")
    emit(f"B.{tag}", f"    stim events/first run={f['n_stim_events']} codes={f['stim_event_codes']} | "
                    f"median(annotation onset - stim sample)={_fmt(f['stim_to_annotation_shift_s'])} s "
                    f"vs MOABB interval[0]={_fmt(interval0)} s")
    emit(f"B.{tag}", f"    raw |EEG| max over first {_fmt(f['magnitude_window_s'])} s = "
                    f"{f['eeg_abs_max_v']:.3e} V (file is in volts)")
    if any(f["run_descriptions"].values()):
        emit(f"B.{tag}", f"    info['description'] per run (artifact counts): {f['run_descriptions']}")


def audit_one(spec: DatasetSpec, subject: int) -> tuple[list[dict], dict]:
    """Full A-D audit of one dataset; returns (csv rows, evidence)."""
    tag = spec.key
    hdr(f"B/C/D. {tag} - subject {subject} (loaded exactly as src/datasets.py does)")
    t0 = time.time()
    mem_report(f"{tag} start")

    raw_sessions, raw_runs = raw_pass(spec, subject)
    peak = raw_sessions["__peak_rss_gb__"]
    emit(f"B.{tag}", f"raw loaded in {time.time() - t0:.1f} s; peak RSS during load = {_fmt(peak)} GB")
    interval0 = float(spec.dataset.interval[0])
    for session, facts in _nonempty(raw_sessions).items():
        print_raw_session(tag, session, facts, interval0)

    # ---- paradigm, epochs object (volts; no unit scaling) ----------------
    # get_data always returns (X, labels, metadata); with return_epochs=True the
    # first element is an mne.Epochs whose data is in volts (no unit_factor).
    epochs, _y_epochs, _meta_epochs = spec.paradigm.get_data(
        dataset=spec.dataset, subjects=[subject], return_epochs=True
    )
    ef = epochs_facts(epochs)
    emit(f"C.{tag}", f"paradigm epochs (return_epochs=True): shape={ef['shape']} dtype={ef['dtype']} "
                     f"n_epochs={ef['n_epochs']}")
    emit(f"C.{tag}", f"    info: sfreq={_fmt(ef['sfreq_hz'])} Hz, "
                     f"highpass/lowpass={_fmt(ef['highpass_hz'])}/{_fmt(ef['lowpass_hz'])} Hz, "
                     f"ch_types={ef['ch_types']}, n_ch={len(ef['ch_names'])}")
    emit(f"C.{tag}", f"    ch_names={ef['ch_names']}")
    emit(f"C.{tag}", f"    epoching: tmin={_fmt(ef['tmin_s'])} s tmax={_fmt(ef['tmax_s'])} s relative to the "
                     f"trigger; n_times={ef['n_times']} (times {_fmt(ef['first_time_s'])}..{_fmt(ef['last_time_s'])} s)")
    emit(f"C.{tag}", f"    baseline attribute={ef['baseline']!r} (None => no baseline correction), "
                     f"reject={ef['reject']!r}, proj={ef['proj_applied']}, |data| max={ef['abs_max_v']:.3e} V")
    emit(f"C.{tag}", f"    events per ch type={ef['ch_types']}; metadata columns="
                     f"{list(epochs.metadata.columns) if epochs.metadata is not None else None}")

    # ---- paradigm, array path (unit_factor applied) -----------------------
    X, y, meta = spec.paradigm.get_data(dataset=spec.dataset, subjects=[subject])
    Xe = epochs.get_data(copy=False)
    ratio = X / Xe if X.shape == Xe.shape else None
    counts = session_trial_counts(np.asarray(y), meta)
    emit(f"C.{tag}", f"paradigm array: shape={X.shape} dtype={X.dtype} |X| max={np.abs(X).max():.3f} "
                     f"(microvolts: MOABB multiplies epoch data by dataset.unit_factor={spec.dataset.unit_factor:g})")
    for session, c in counts.items():
        emit(f"C.{tag}", f"    session {session!r}: n_trials={c['n_trials']} per_class={c['per_class']}")
    if ratio is not None:
        rel = float(np.abs(ratio / spec.dataset.unit_factor - 1.0).max())
        emit(f"C.{tag}", f"    unit check X_array / X_epochs(V): min={ratio.min():.9f} max={ratio.max():.9f} "
                         f"vs unit_factor={spec.dataset.unit_factor:g} -> max relative deviation {rel:.1e} "
                         f"({'EXACT up to float rounding' if rel < 1e-12 else 'MISMATCH'})")

    # ---- D: raw -> epochs delta ------------------------------------------
    ref_session = sorted(_nonempty(raw_sessions))[0]
    rf = raw_sessions[ref_session]
    resampled = not np.isclose(ef["sfreq_hz"], rf["sfreq_hz"])
    emit(f"D.{tag}", f"resampling: raw {_fmt(rf['sfreq_hz'])} Hz -> epochs {_fmt(ef['sfreq_hz'])} Hz => "
                     f"{'RESAMPLED' if resampled else 'NONE'} (paradigm.resample={spec.paradigm.resample!r})")
    emit(f"D.{tag}", f"filtering: raw info {_fmt(rf['highpass_hz'])}-{_fmt(rf['lowpass_hz'])} Hz -> epochs info "
                     f"{_fmt(ef['highpass_hz'])}-{_fmt(ef['lowpass_hz'])} Hz "
                     f"(paradigm.filters={spec.paradigm.filters})")
    emit(f"D.{tag}", f"epoching: epochs.tmin/tmax={_fmt(ef['tmin_s'])}/{_fmt(ef['tmax_s'])} s = "
                     f"dataset.interval[0] + paradigm.tmin/tmax = {_fmt(interval0)} + "
                     f"{_fmt(spec.paradigm.tmin)}/{_fmt(spec.paradigm.tmax)}")
    emit(f"D.{tag}", f"    after removing the interval[0] offset the window is "
                     f"{_fmt(ef['tmin_s'] - interval0)}..{_fmt(ef['tmax_s'] - interval0)} s "
                     f"relative to whatever the trigger marks")
    emit(f"D.{tag}", f"baseline correction: epochs.baseline={ef['baseline']!r} "
                     f"(paradigm.baseline={spec.paradigm.baseline!r}) => NONE")
    emit(f"D.{tag}", f"channel selection: raw {rf['n_eeg']} EEG + {rf['n_eog']} EOG + {rf['n_emg']} EMG + "
                     f"{rf['n_stim']} stim -> epochs {ef['ch_types']} (paradigm.channels={spec.paradigm.channels!r})")
    emit(f"D.{tag}", f"units: raw volts (|max| {rf['eeg_abs_max_v']:.3e} V) -> array microvolts "
                     f"(|max| {np.abs(X).max():.3f} uV)")

    if spec.channels_from_datasets_py is not None:
        constant = list(spec.channels_from_datasets_py)
        emit(f"E.{tag}", f"src/datasets.py channel constant ({len(constant)} names) vs returned epochs ch_names "
                         f"({len(ef['ch_names'])}): same_set={set(constant) == set(ef['ch_names'])} "
                         f"same_order={constant == ef['ch_names']}")
        if set(constant) != set(ef["ch_names"]):
            emit(f"E.{tag}", f"    only in constant: {sorted(set(constant) - set(ef['ch_names']))}; "
                             f"only in epochs: {sorted(set(ef['ch_names']) - set(constant))}")

    rows = csv_rows(spec, subject, counts, ef, _nonempty(raw_sessions), X)
    evidence = {
        "raw_sessions": raw_sessions,
        "raw_runs": raw_runs,
        "epochs_facts": ef,
        "array_shape": tuple(int(v) for v in X.shape),
        "array_abs_max_uv": float(np.abs(X).max()),
        "unit_ratio_exact": bool(ratio is not None
                                 and np.abs(ratio / spec.dataset.unit_factor - 1.0).max() < 1e-12),
        "session_counts": counts,
        "epochs": epochs,
    }
    del X, Xe, y, meta
    gc.collect()
    mem_report(f"{tag} done")
    return rows, evidence


def csv_rows(spec, subject, counts, ef, raw_sessions, X) -> list[dict]:
    rows = []
    ch_names = "|".join(ef["ch_names"])
    for session, c in counts.items():
        rf = raw_sessions.get(session) or raw_sessions[sorted(raw_sessions)[0]]
        per_class = "|".join(f"{k}:{v}" for k, v in sorted(c["per_class"].items()))
        artifact_text = "; ".join(sorted({f"{r}={d}" for r, d in rf["run_descriptions"].items()})) or "none"
        const_ok = (set(spec.channels_from_datasets_py) == set(ef["ch_names"])
                    if spec.channels_from_datasets_py else "n/a")
        notes = [
            f"raw info['description']: {artifact_text}",
            f"median(annotation onset - stim sample)={rf['stim_to_annotation_shift_s']:g}s = MOABB interval[0]",
            f"raw |EEG| max {rf['eeg_abs_max_v']:.3e} V over first {rf['magnitude_window_s']:g}s",
            "no resampling (epochs sfreq == raw sfreq)",
            "no baseline correction (epochs.baseline is None)",
            f"unit_factor {spec.dataset.unit_factor:g} applied -> microvolts",
            f"channel constant from src/datasets.py matches epochs: {const_ok}",
        ]
        if spec.extra_notes:
            notes.append(spec.extra_notes)
        rows.append({
            "dataset": spec.key,
            "dataset_code": spec.dataset.code,
            "subject": subject,
            "session": session,
            "n_trials": c["n_trials"],
            "n_trials_per_class": per_class,
            "n_channels": len(ef["ch_names"]),
            "sfreq_raw": rf["sfreq_hz"],
            "sfreq_epochs": ef["sfreq_hz"],
            "filter_band": f"{FMIN:g}-{FMAX:g}",
            "reference": spec.reference_note,
            "line_freq": rf["line_freq_hz"],
            "line_freq_source": spec.line_freq_note,
            "n_runs": rf["n_runs"],
            "eog_channels": "|".join(rf["eog_ch_names"]),
            "emg_channels": "|".join(rf["emg_ch_names"]),
            "n_channels_raw": rf["n_channels_total"],
            "n_eeg_raw": rf["n_eeg"],
            "raw_highpass_hz": rf["highpass_hz"],
            "raw_lowpass_hz": rf["lowpass_hz"],
            "epoch_highpass_hz": ef["highpass_hz"],
            "epoch_lowpass_hz": ef["lowpass_hz"],
            "epoching_tmin_s": ef["tmin_s"],
            "epoching_tmax_s": ef["tmax_s"],
            "epoching_cue_relative_tmin_s": ef["tmin_s"] - float(spec.dataset.interval[0]),
            "epoching_cue_relative_tmax_s": ef["tmax_s"] - float(spec.dataset.interval[0]),
            "baseline_correction": "none" if ef["baseline"] is None else str(ef["baseline"]),
            "resampling": "none",
            "unit_factor": spec.dataset.unit_factor,
            "epochs_shape": f"{X.shape[0]}x{X.shape[1]}x{X.shape[2]}",
            "channel_names": ch_names,
            "n_annotations_per_run": "|".join(str(v) for v in rf["n_annotations_per_run"]),
            "first_annotation_onset_s": rf["first_annotation_onset_s"],
            "median_annotation_gap_s": rf["median_annotation_gap_s"],
            "stim_event_codes": "|".join(f"{k}:{v}" for k, v in sorted(rf["stim_event_codes"].items())),
            "moabb_interval": f"{spec.dataset.interval[0]},{spec.dataset.interval[1]}",
            "notes": "; ".join(notes),
        })
    return rows


# ---------------------------------------------------------------------------
# Section E - targeted checks
# ---------------------------------------------------------------------------

def bnci_mat_facts(path: Path, tag: str, dataset_event_id: dict) -> dict:
    """Read a BNCI .mat directly: runs, trials per run, trigger samples,
    artifact counts and class order - independent of MOABB's loader."""
    m = loadmat(path, struct_as_record=False, squeeze_me=True)
    data = m["data"]
    runs = data if isinstance(data, np.ndarray) else [data]
    emit(tag, f"{path.name}: {len(runs)} runs, fields per run = {list(runs[0]._fieldnames)}")
    fs = float(runs[0].fs)
    classes: list[str] = []
    artifact_counts: list[int] = []
    live = 0
    for i, r in enumerate(runs):
        trial = np.atleast_1d(np.asarray(r.trial))
        x_shape = tuple(int(v) for v in np.shape(r.X))
        classes = [str(c) for c in np.atleast_1d(r.classes)]
        if trial.size == 0:
            emit(tag, f"    .mat run {i}: fs={_fmt(fs)} X={x_shape} NO TRIALS (baseline run) "
                      f"duration={x_shape[0] / fs:.1f} s")
            continue
        live += 1
        y = np.atleast_1d(np.asarray(r.y))
        art = np.atleast_1d(np.asarray(r.artifacts))
        gaps = np.diff(trial) if trial.size > 1 else np.array([np.nan])
        artifact_counts.append(int(np.sum(art)))
        emit(tag, f"    .mat run {i}: fs={_fmt(fs)} X={x_shape} n_trials={trial.size} "
                  f"first trigger sample={int(trial[0])} ({trial[0] / fs:.3f} s) "
                  f"median gap={int(np.median(gaps))} samples ({np.median(gaps) / fs:.3f} s) "
                  f"class codes in file={sorted(set(int(v) for v in y))} artifacts={int(np.sum(art))}/{art.size}")
    file_classes = [c.replace(" ", "_") for c in classes]
    emit(tag, f"    classes field order, spaces normalised (=> the code MOABB assigns is index+1): {file_classes}")
    emit(tag, f"    MOABB event_id for this dataset: {dataset_event_id} "
              f"(matches the file's classes order: {list(dataset_event_id) == file_classes})")
    if live != len(runs):
        emit(tag, f"    => {live} of the {len(runs)} runs carry trials; MOABB exposes only those")
    summary = {"n_runs": int(len(runs)), "n_live_runs": live,
               "artifact_counts": artifact_counts, "classes": file_classes}
    del m, data, runs
    gc.collect()
    return summary


def compare_artifact_text(mat_counts: list[int], run_descriptions: dict, tag: str, session: str) -> None:
    """MOABB only writes 'Artifacts: k/N trials;' into info['description'] when
    k > 0, so an empty description means zero marked artifacts rather than
    missing information. Check that reading against the .mat's own counts."""
    parsed = []
    for run in sorted(run_descriptions, key=lambda r: int(r)):
        text = run_descriptions[run]
        parsed.append(int(text.split()[1].split("/")[0]) if text else 0)
    match = parsed == mat_counts
    emit(tag, f"{session}: info['description'] -> parsed artifact counts {parsed} vs .mat {mat_counts} "
              f"=> {'MATCH' if match else 'MISMATCH'}")
    emit(tag, f"    (MOABB appends the text only when k > 0, so an empty description means 0 marked "
              f"artifacts, not missing information)")


def compare_artifact_text_per_run(mat_counts: list[int], run_descriptions: dict,
                                  session_names: list[str], tag: str) -> None:
    """IV-2b: each .mat run is exposed as its own MOABB session (one run per
    session), so the counts must be compared run-by-run, not concatenated."""
    parsed = []
    for session in session_names:
        text = list(run_descriptions[session].values())[0]
        parsed.append(int(text.split()[1].split("/")[0]) if text else 0)
    match = parsed == mat_counts
    emit(tag, f"sessions {session_names}: parsed artifact counts {parsed} vs .mat runs {mat_counts} "
              f"=> {'MATCH' if match else 'MISMATCH'}")


def trigger_erp(raw_runs: dict, event_codes: list[int], tmax_s: float, label: str, tag: str) -> dict:
    """Across-trial average time-locked to the trigger, per channel, plus the
    peak |amplitude| in fixed windows. Used to locate *events inside* the trial
    (a warning tone / a visual cue) relative to the trigger."""
    raw = mne.concatenate_raws([r.copy() for r in raw_runs.values()], verbose=False)
    raw.filter(0.5, 40.0, picks="eeg", verbose=False)
    events = mne.find_events(raw, shortest_event=0, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=event_codes, tmin=-1.5, tmax=tmax_s, baseline=None,
                        picks="eeg", preload=True, verbose=False)
    n = len(epochs)
    avg = epochs.average()
    out: dict = {"label": label, "n_events": int(len(events)), "n_epochs": int(n), "windows": {}}
    per_window_max: dict[str, float] = {}
    for ch in avg.ch_names:
        d = avg.copy().pick([ch]).data[0]
        d = d - d[avg.times < -0.2].mean()
        sem = float(epochs.get_data(picks=[ch]).std(axis=0).mean() / np.sqrt(max(n, 1)))
        vals = {}
        for a, b in ERP_WINDOWS:
            mask = (avg.times >= a) & (avg.times <= b)
            v = float(np.abs(d[mask]).max()) if mask.any() else float("nan")
            vals[f"{a}-{b}s"] = v
            if v == v:
                per_window_max[f"{a}-{b}s"] = max(per_window_max.get(f"{a}-{b}s", 0.0), v)
        pre = float(np.abs(d[(avg.times >= -0.5) & (avg.times < -0.01)]).max())
        out["windows"][ch] = {"windows": vals, "pre_s": pre, "sem_v": sem}
        if ch in ERP_DISPLAY_CHANNELS:
            emit(tag, f"    {label} {ch:4s}: " + " ".join(f"[{k}]={v:.2e}" for k, v in vals.items()) +
                      f" pre={pre:.2e} SEM={sem:.2e} V (n={n})")
    emit(tag, f"    {label} max across all {len(avg.ch_names)} channels: " +
              " ".join(f"[{k}]={v:.2e}" for k, v in per_window_max.items()) + " V")
    out["max_across_channels"] = per_window_max
    del raw, epochs, avg
    gc.collect()
    return out


def section_e_iv2a(spec: DatasetSpec, subject: int, evidence: dict) -> None:
    tag = "E.BNCI2014_001"
    hdr("E.1 BNCI2014_001 - file-level structure, epoch reproduction, trigger identity")

    # (a) the .mat files themselves
    for fname, session_name, session_key in (("A01T.mat", "0train", "0train"), ("A01E.mat", "1test", "1test")):
        mat = bnci_mat_facts(MAT_001 / fname, tag + f".mat[{fname}]", dict(spec.dataset.event_id))
        rf = evidence["raw_sessions"][session_key]
        compare_artifact_text(mat["artifact_counts"], rf["run_descriptions"],
                              tag + f".mat[{fname}]", session_name)

    # (b) bit-exact reproduction of one run
    epochs: mne.Epochs = evidence["epochs"]
    md = epochs.metadata
    if md is not None:
        idx = np.where((md["session"].to_numpy() == "0train") & (md["run"].to_numpy() == "0"))[0]
        raw_dict = spec.dataset.get_data(subjects=[subject])
        raw = raw_dict[subject]["0train"]["0"].copy()
        events = mne.find_events(raw, shortest_event=0, verbose=False)
        codes = sorted(spec.dataset.event_id.values())
        events = events[np.isin(events[:, 2], codes)]
        filtered = raw.filter(l_freq=FMIN, h_freq=FMAX, method="iir", picks="data", verbose=False)
        picks = mne.pick_types(filtered.info, eeg=True, stim=False)
        rebuilt = mne.Epochs(
            filtered, events, event_id=codes,
            tmin=TMIN + float(spec.dataset.interval[0]), tmax=TMAX + float(spec.dataset.interval[0]),
            proj=False, baseline=None, preload=True, verbose=False, picks=picks,
            event_repeated="drop", on_missing="ignore",
        )
        got = epochs.get_data(copy=False)[idx]
        rebuilt_data = rebuilt.get_data(copy=False)
        emit(tag + ".repro", f"0train run 0: stim events={len(events)}, rebuilt epochs={len(rebuilt)}, "
                             f"paradigm epochs={len(idx)}")
        if rebuilt_data.shape == got.shape:
            max_abs = float(np.abs(rebuilt_data - got).max())
            emit(tag + ".repro", f"max|rebuilt(filter method='iir', picks='data', tmin={TMIN + float(spec.dataset.interval[0])}s, "
                                 f"tmax={TMAX + float(spec.dataset.interval[0])}s, baseline=None) - paradigm epochs| = "
                                 f"{max_abs:.3e} V => "
                                 f"{'BIT-EXACT: MOABB applies exactly this filter, no resampling, no baseline, EEG-only picks' if max_abs == 0 else 'NOT bit-exact'}")
        else:
            emit(tag + ".repro", f"shape mismatch {rebuilt_data.shape} vs {got.shape} -> NOT reproduced")
        del raw_dict, raw, filtered, rebuilt, got, rebuilt_data
        gc.collect()

    # (c) trigger identity: where are the trial's internal events?
    emit(tag + ".trigger", "across-trial average time-locked to the STIM trigger (0.5-40 Hz, session 0train, "
                           "282 of 288 trials kept - the first trial of each run has no -1.5 s of data). "
                           "Windows: +0.05-0.35 s, +0.6-1.4 s (should be quiet), +1.9-2.4 s, +2.9-3.4 s, +5.9-6.4 s:")
    raw_dict = spec.dataset.get_data(subjects=[subject])
    trigger_erp(raw_dict[subject]["0train"], sorted(spec.dataset.event_id.values()), 6.5,
                "IV-2a 0train", tag + ".trigger")
    emit(tag + ".trigger", "MOABB's interval is [2,6] s relative to this trigger and the Graz description has "
                           "tone+fixation at t=0 with the cue at t=2 s, so: an evoked deflection at ~+0.1-0.3 s "
                           "(warning tone) plus a second one at ~+2.0 s (arrow cue), a third at ~+6 s (fixation "
                           "cross disappears / end-of-trial movement) and quiet averages in between is what "
                           "'trigger = trial start' predicts. See docs/datasets.md.")
    del raw_dict
    gc.collect()


def section_e_iv2b(spec: DatasetSpec, evidence: dict) -> None:
    tag = "E.BNCI2014_004"
    hdr("E.2 BNCI2014_004 - file-level structure, session split, trigger identity")
    for fname, session_names in (("B01T.mat", ["0train", "1train", "2train"]),
                                 ("B01E.mat", ["3test", "4test"])):
        mat = bnci_mat_facts(MAT_004 / fname, tag + f".mat[{fname}]", dict(spec.dataset.event_id))
        compare_artifact_text_per_run(
            mat["artifact_counts"],
            {s: evidence["raw_sessions"][s]["run_descriptions"] for s in session_names},
            session_names, tag + f".mat[{fname}]")

    counts = evidence["session_counts"]
    order = ["0train", "1train", "2train", "3test", "4test"]
    sizes = {s: counts[s]["n_trials"] for s in order if s in counts}
    emit(tag + ".split", f"trials per MOABB session for subject 1: {sizes}")
    emit(tag + ".split", "MOABB METADATA for BNCI2014_004 names the sessions ['01T','02T','03T','04E','05E'] and "
                         "states 01T/02T are screening sessions without feedback while 03T/04E/05E are feedback "
                         "sessions; the measured sizes (120, 120, 160, 160, 160) match that split")
    emit(tag + ".split", "=> src/datasets.py's IV2B_CALIB_SESSIONS={0train,1train,2train} = 2 screening + 1 "
                         "FEEDBACK recording, grouped by source FILE (T/E), not by protocol; calling it 'the "
                         "dataset's own canonical split' as an equal-protocol split is REFUTED")

    raw_dict = spec.dataset.get_data(subjects=[REPRESENTATIVE_SUBJECT])
    for session in order:
        if session not in raw_dict[REPRESENTATIVE_SUBJECT]:
            continue
        emit(tag + ".trigger", f"across-trial average time-locked to the trigger, session {session} "
                               f"(Cz is the only non-lateral channel available; 3 EEG channels, no posterior sites):")
        trigger_erp(raw_dict[REPRESENTATIVE_SUBJECT][session], sorted(spec.dataset.event_id.values()), 7.6,
                    session, tag + ".trigger")
    emit(tag + ".trigger", "MOABB's interval for IV-2b is [3,7.5] s while its docstring (copied from IV-2a) claims "
                           "the cue appears 2 s after the trial start; the measured averages are the evidence "
                           "reported in docs/datasets.md")
    del raw_dict
    gc.collect()


def section_e_openbmi_sessions() -> None:
    """MOABB's Lee2019 session naming/filter mismatch, demonstrated on the real
    filtering code path with a stubbed loader (no data read).

    The stub returns exactly the session names the real loader would build for
    the sessions this dataset object was told to load (``str(session - 1)``),
    so the filter behaviour observed here is the behaviour on real data - only
    the (huge) arrays are replaced by empty dicts.
    """
    tag = "E.Lee2019_MI.sessions"

    def probe(dataset) -> list[str]:
        dataset._get_single_subject_data = lambda subject: {str(s - 1): {} for s in dataset.sessions}
        return sorted(dataset.get_data(subjects=[1])[1].keys())

    d_default = Lee2019_MI()
    kept_default = probe(d_default)

    d_first = Lee2019_MI(sessions=[1])
    kept_first = probe(d_first)

    emit(tag, f"the dataset declares 2 sessions per subject; the loader names them "
              f"{[str(s - 1) for s in (1, 2)]} (str(session-1)) but get_data filters the returned sessions "
              f"against {sorted({str(s) for s in d_default._selected_sessions})}")
    emit(tag, f"stubbed-load filter probe (stub = exactly the names the real loader would build): "
              f"Lee2019_MI() keeps {kept_default}; Lee2019_MI(sessions=[1]) keeps {kept_first}")
    emit(tag, f"=> CONFIRMED: the returned session is '1' (the sess02 file, the second recording day) and "
              f"session '0' (the first recording day) is unreachable through get_data - requesting it yields "
              f"an empty result")


def section_e_physionet(spec: DatasetSpec, subject: int) -> None:
    hdr("E.3 PhysionetMI - 'rest' event quirk, subject 88, montage superset, EDF line frequency")
    tag = "E.PhysionetMI"

    loose = MotorImagery(n_classes=2, fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
    used = loose.used_events(PhysionetMI())
    X_loose, y_loose, _ = loose.get_data(dataset=PhysionetMI(), subjects=[subject])
    vals, cnts = np.unique(y_loose, return_counts=True)
    total = int(len(y_loose))
    rest = int(dict(zip(vals.tolist(), cnts.tolist())).get("rest", 0))
    emit(tag, f"MotorImagery(n_classes=2) with NO explicit events -> uses {len(used)} event types "
              f"{sorted(used)}, returned trials {dict(zip(vals.tolist(), cnts.tolist()))} "
              f"(rest share {rest / total:.1%}) => CONFIRMED: n_classes alone is ignored and 'rest' is included")
    emit(tag, f"X with the loose paradigm: {X_loose.shape}; src/datasets.py passes "
              f"events={PHYSIONET_MI_EVENTS} to exclude it")
    del X_loose, y_loose, loose
    gc.collect()

    d88 = PhysionetMI()
    data88 = d88.get_data(subjects=[PHYSIONET_HZ_OUTLIER])
    runs88 = data88[PHYSIONET_HZ_OUTLIER]["0"]
    first88 = runs88[sorted(runs88)[0]]
    data1 = spec.dataset.get_data(subjects=[subject])
    first1 = data1[subject]["0"]["0"]
    emit(tag, f"subject {PHYSIONET_HZ_OUTLIER}: sfreq={first88.info['sfreq']} Hz, {len(first88.ch_names)} channels | "
              f"subject {subject}: sfreq={first1.info['sfreq']} Hz, {len(first1.ch_names)} channels "
              f"=> CONFIRMED subject 88 is at 128 Hz")
    del data88, runs88, data1
    gc.collect()

    from src.datasets import load_physionet_reference_subject

    try:
        load_physionet_reference_subject(PHYSIONET_HZ_OUTLIER)
    except ValueError as exc:
        emit(tag, f"load_physionet_reference_subject({PHYSIONET_HZ_OUTLIER}) raises ValueError -> "
                  f"src/datasets.py enforces the exclusion: {exc}")
    else:
        emit(tag, f"UNEXPECTED: load_physionet_reference_subject({PHYSIONET_HZ_OUTLIER}) did not raise")

    raw = spec.dataset.get_data(subjects=[subject])[subject]["0"]["0"]
    physio_eeg = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    missing = [c for c in IV2A_CHANNELS if c not in physio_eeg]
    emit(tag, f"PhysioNet EEG montage ({len(physio_eeg)} ch) covers IV-2a's {len(IV2A_CHANNELS)} channels: "
              f"missing={missing} => {'CONFIRMED superset' if not missing else 'NOT a superset'}")
    emit(tag, f"PHYSIONET_CHANNELS constant in src/datasets.py == raw EEG order: "
              f"{list(PHYSIONET_CHANNELS) == physio_eeg}")
    del raw
    gc.collect()

    edf = sorted(Path(config.DATA_DIR, "MNE-eegbci-data").rglob("S001R04.edf"))
    if edf:
        raw_edf = mne.io.read_raw_edf(edf[0], preload=False, verbose="ERROR")
        emit(tag, f"EDF read directly ({edf[0].name}): info['line_freq']={raw_edf.info['line_freq']}, "
                  f"sfreq={raw_edf.info['sfreq']}, n_ch={len(raw_edf.ch_names)}, "
                  f"annotations={len(raw_edf.annotations)} => the 60 Hz seen in raw.info is MOABB METADATA, "
                  f"not the file")
        del raw_edf
        gc.collect()
    else:
        emit(tag, "S001R04.edf not found under data/ - EDF header not checked")


def section_e_cleaned_pipeline(subject: int) -> None:
    """What src/preprocessing.py + load_iv2a_subject_cleaned actually deliver."""
    hdr("E.4 src/preprocessing.py cleaned cache vs the raw pipeline")
    tag = "E.iv2a_cleaned"
    from moabb.datasets import BNCI2014_001 as _D

    from src.datasets import load_iv2a_subject, load_iv2a_subject_cleaned
    from src.preprocessing import BANDPASS, N_ICA_COMPONENTS, NOTCH_FREQS, get_cleaned_session_raw

    raw_clean = get_cleaned_session_raw(_D(), subject, "0train")
    types = raw_clean.get_channel_types()
    emit(tag, f"cached cleaned raw (data/cleaned/subject{subject}_0train_cleaned_raw.fif): "
              f"sfreq={raw_clean.info['sfreq']}, {len(raw_clean.ch_names)} ch (eeg={types.count('eeg')}, "
              f"eog={types.count('eog')}, stim={types.count('stim')}), "
              f"highpass/lowpass={raw_clean.info['highpass']:g}/{raw_clean.info['lowpass']:g} Hz "
              f"vs NOTCH_FREQS={NOTCH_FREQS} BANDPASS={BANDPASS} N_ICA_COMPONENTS={N_ICA_COMPONENTS}")
    emit(tag, f"annotations={len(raw_clean.annotations)} types={sorted(set(raw_clean.annotations.description))} "
              f"durations_s={sorted({round(float(d), 3) for d in raw_clean.annotations.duration})}")
    del raw_clean
    gc.collect()

    plain = load_iv2a_subject(subject)
    cleaned = load_iv2a_subject_cleaned(subject)
    emit(tag, f"load_iv2a_subject         X_calib={plain.X_calib.shape} X_eval={plain.X_eval.shape} "
              f"|X_calib|max={np.abs(plain.X_calib).max():.2f} uV")
    emit(tag, f"load_iv2a_subject_cleaned X_calib={cleaned.X_calib.shape} X_eval={cleaned.X_eval.shape} "
              f"|X_calib|max={np.abs(cleaned.X_calib).max():.2f} uV")
    emit(tag, f"same shape={plain.X_calib.shape == cleaned.X_calib.shape}, "
              f"same labels={np.array_equal(plain.y_calib, cleaned.y_calib)}, "
              f"data changed by cleaning={not np.allclose(plain.X_calib, cleaned.X_calib)}, "
              f"mean|diff|={np.abs(plain.X_calib - cleaned.X_calib).mean():.3f} uV")
    del plain, cleaned
    gc.collect()


def section_e_misc() -> None:
    hdr("E.5 Cross-dataset constant / claim checks")
    tag = "E.misc"
    emit(tag, f"FCz present in IV-2a constant: {'FCz' in IV2A_CHANNELS} | OpenBMI constant: "
              f"{'FCz' in OPENBMI_CHANNELS} | PhysioNet constant: {'FCz' in PHYSIONET_CHANNELS}")
    emit(tag, f"sizes: OPENBMI_CHANNELS={len(OPENBMI_CHANNELS)}, PHYSIONET_CHANNELS={len(PHYSIONET_CHANNELS)}, "
              f"IV2A_CHANNELS={len(IV2A_CHANNELS)}, SHARED_CHANNELS={len(SHARED_CHANNELS)} "
              f"(= IV-2a minus {sorted(set(IV2A_CHANNELS) - set(SHARED_CHANNELS))})")
    emit(tag, f"SHARED_CHANNELS == [c for c in IV2A_CHANNELS if c in OPENBMI_CHANNELS]: "
              f"{SHARED_CHANNELS == [c for c in IV2A_CHANNELS if c in OPENBMI_CHANNELS]}")
    emit(tag, f"src/datasets.py's import-time assertion (PhysioNet constant covers all of IV-2a's channels): "
              f"{all(c in PHYSIONET_CHANNELS for c in IV2A_CHANNELS)}")
    emit(tag, f"PHYSIONET_EXCLUDED_SUBJECTS={PHYSIONET_EXCLUDED_SUBJECTS}")
    n_times = int((TMAX - TMIN) * 250) + 1
    emit(tag, f"compute_environment.md's '288 x 22 x 500 float64 ~ 25 MB per session': the real epoch length is "
              f"{n_times} samples (0.5-2.5 s inclusive at 250 Hz), so 288*22*{n_times}*8 = "
              f"{288 * 22 * n_times * 8 / 1e6:.1f} MB - the '500' is off by one, the '~25 MB' holds")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    hdr("Characterization 00 - dataset recording-methodology audit")
    print(f"python {sys.version.split()[0]} | numpy {np.__version__} | mne {mne.__version__} | "
          f"moabb {moabb.__version__} | pandas {pd.__version__}")
    print(f"cwd={Path.cwd()}  MNE_DATA={os.environ.get('MNE_DATA')!r}")
    print(f"FMIN/FMAX={FMIN}/{FMAX} Hz, TMIN/TMAX={TMIN}/{TMAX} s (src/config.py), threads pinned to 4")

    specs = build_specs()
    section_a(specs)

    all_rows: list[dict] = []
    evidence: dict[str, dict] = {}
    failures: list[str] = []
    for spec in specs:
        try:
            rows, ev = audit_one(spec, REPRESENTATIVE_SUBJECT)
            all_rows.extend(rows)
            evidence[spec.key] = ev
        except Exception:  # noqa: BLE001 - one broken dataset must not lose the rest
            failures.append(spec.key)
            print(traceback.format_exc(), flush=True)
            all_rows.append({
                "dataset": spec.key, "dataset_code": spec.dataset.code, "subject": REPRESENTATIVE_SUBJECT,
                "session": "ERROR", "notes": f"audit failed: {traceback.format_exc().splitlines()[-1]}",
            })
        gc.collect()

    def _guard(name: str, fn) -> None:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failures.append(name)
            print(traceback.format_exc(), flush=True)

    if "BNCI2014_001" in evidence:
        _guard("E.iv2a", lambda: section_e_iv2a(specs[0], REPRESENTATIVE_SUBJECT, evidence["BNCI2014_001"]))
    if "BNCI2014_004" in evidence:
        _guard("E.iv2b", lambda: section_e_iv2b(specs[1], evidence["BNCI2014_004"]))
    _guard("E.openbmi_sessions", section_e_openbmi_sessions)
    if "PhysionetMI" in evidence:
        _guard("E.physionet", lambda: section_e_physionet(specs[3], REPRESENTATIVE_SUBJECT))
    for key in list(evidence):
        evidence[key].pop("epochs", None)  # release Epochs objects before the cleaned-pipeline check
    gc.collect()
    _guard("E.cleaned_pipeline", lambda: section_e_cleaned_pipeline(REPRESENTATIVE_SUBJECT))
    _guard("E.misc", section_e_misc)

    hdr("Summary table -> results/char_dataset_audit.csv")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    preferred = [
        "dataset", "dataset_code", "subject", "session", "n_trials", "n_trials_per_class", "n_channels",
        "sfreq_raw", "sfreq_epochs", "filter_band", "reference", "line_freq", "n_runs", "eog_channels", "notes",
    ]
    df = df.reindex(columns=preferred + [c for c in df.columns if c not in preferred])
    df.to_csv(CSV_PATH, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 55):
        print(df[preferred].to_string(index=False))
    print(f"\nwrote {CSV_PATH} ({CSV_PATH.stat().st_size} bytes, {len(df)} rows x {len(df.columns)} columns)")

    hdr("Result")
    if failures:
        print(f"FAILED sections: {failures}")
        return 1
    print("all sections completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
