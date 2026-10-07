# The four datasets: recording methodology, what MOABB hands us, and what our pipeline does to it

Every number in this repo is a statement about a subject's signal, but the pipeline that
produces it is a chain of *other people's* decisions: the amplifier and montage, the
reference, the online filters, how a "trial" was cued, which runs MOABB exposes, which
channels MOABB keeps, what units the array arrives in and where the epoch window actually
lands in the recording. This document records those facts for the four datasets the project
uses, with the evidence for each one.

**How to read the evidence tags.** Everything marked **[V]** was measured by
`experiments/char_00_dataset_audit.py`; the tag in brackets is the section of that script's
stdout the number comes from (e.g. `[B.BNCI2014_001]`). Everything marked **[S]** is
attributed to a named source with a link and was *not* independently measured. Anything
neither measured nor attributable is marked **UNVERIFIED** — a few of those are important
and are called out in [Known gaps](#8-known-gaps--things-i-could-not-verify).

Reproduce all of it with:

```powershell
.venv\Scripts\python experiments\char_00_dataset_audit.py
```

which also writes `results/char_dataset_audit.csv` (9 rows × 37 columns: one row per
dataset × subject × session, with trials, per-class counts, channels, both sampling rates,
filter band, reference, line frequency, runs, EOG/EMG channels and the raw→epoch deltas).
Measured with python 3.12.3, mne 1.12.1, **moabb 1.5.0**, numpy 2.4.6, pandas 2.3.3
`[header of the run]`. MOABB version matters: session names, the `interval` offsets and the
`MotorImagery` pipeline are all version-specific and were read from the installed 1.5.0.

---

## 0. Summary table (all values **[V]** unless marked)

| | BCI IV-2a (`BNCI2014_001`) | BCI IV-2b (`BNCI2014_004`) | OpenBMI (`Lee2019_MI`) | PhysioNet MI (`PhysionetMI`) |
|---|---|---|---|---|
| subjects on disk | 9 (all) | 9 (all) | 20 of 54 | 109 (108 usable) |
| MOABB sessions per subject | 2 (`0train`, `1test`) | 5 (`0train`…`4test`) | 2 declared, **1 reachable** (`1`) | 1 (`0`) |
| runs per MOABB session | 6 | 1 | 1 (train phase only) | 6 |
| file-level runs | 9 per session (3 have no trials) | 3 (T file) + 2 (E file) | 1 train + 1 test phase | 14 in the dataset, **6 cached** |
| EEG channels | 22 + 3 EOG + 1 stim | 3 + 3 EOG + 1 stim | 62 + 4 EMG + 1 stim | 64 + 1 stim |
| sampling rate (file) | 250 Hz | 250 Hz | 1000 Hz | 160 Hz (subject 88: **128 Hz**) |
| returned epochs rate | 250 Hz | 250 Hz | 1000 Hz | 160 Hz |
| line frequency | 50 Hz | 50 Hz | 60 Hz (from MOABB metadata only) | 60 Hz (from MOABB metadata only) |
| trial window MOABB uses | trigger + 2.5…4.5 s | trigger + 3.5…5.5 s | trigger + 0.5…2.5 s | trigger + 0.5…2.5 s |
| trials/session | 288 (72/class) | 120 or 160 (60 or 80/class) | 100 (50/class) | 90 across 6 runs (subject 1) |
| artifact annotations | none (counts in `info['description']`) | none (same mechanism) | none | `T0` rest between trials |
| EOG available in raw? | yes, 3 ch, unused by us | yes, 3 ch, unused by us | **no EOG at all** (4 EMG instead) | **no EOG** |

---

## 1. What MOABB's `MotorImagery` paradigm does to every dataset

This is identical for all four datasets and was verified numerically, not read off the
source. `[D.*]`, `[C.*]`, `[E.BNCI2014_001.repro]`

| step | what MOABB does | evidence |
|---|---|---|
| annotations | builds annotations **from the STIM channel** at `trigger_sample + interval[0]`, duration `interval[1]-interval[0]` | annotation onset − stim sample = exactly `interval[0]`: 2 s (2a), 3 s (2b), 0 s (OpenBMI, PhysioNet) `[B.*]` |
| filtering | `raw.filter(l_freq=fmin, h_freq=fmax, method="iir", picks="data")` with our `fmin/fmax = 8/30` | `epochs.info` highpass/lowpass = **8/30 Hz** for all four; and a from-scratch rebuild with `method="iir"`, `picks="data"` reproduces one IV-2a run's epochs **bit-exactly** (max abs difference `0.000e+00 V`) `[E.BNCI2014_001.repro]` |
| epoching | `mne.Epochs(..., tmin=paradigm.tmin+interval[0], tmax=paradigm.tmax+interval[0], baseline=None, proj=False, event_repeated="drop", on_missing="ignore")` on the stim events | `epochs.tmin/tmax` = 2.5/4.5 s (2a), 3.5/5.5 s (2b), 0.5/2.5 s (OpenBMI, PhysioNet); `n_times` = 501 / 501 / 2001 / 321 `[C.*]` |
| baseline correction | **none** — `paradigm.baseline` is `None`, so `epochs.baseline is None` | `[C.*]`, `[D.*]` |
| resampling | **none** — `paradigm.resample` is `None`; epochs sfreq == raw sfreq in all four | `[D.*]` |
| channel selection | EEG only (`return_all_modalities=False` ⇒ `pick_types(eeg=True, stim=False)`): EOG/EMG/STIM are dropped | raw 25 → epochs 22 ch (2a), 7 → 3 (2b), 67 → 62 (OpenBMI), 65 → 64 (PhysioNet) `[D.*]` |
| unit conversion | array path multiplies epoch data (volts) by `dataset.unit_factor` = 1e6 | `X_array / X_epochs(V)` ≈ 1e6 (max relative deviation ~1e-16) `[C.*]` |
| artifact rejection | **none**: artifact-marked trials are still epoched | IV-2a returns 288 trials/session although 4-10 trials per session are marked as artifacts `[C.*]`, `[E.BNCI2014_001.mat]` |

**Consequence for reading this repo's comments.** `TMIN/TMAX = 0.5/2.5` is *not* "0.5-2.5 s
post-cue" by construction; it is "0.5-2.5 s after the trigger, offset by the dataset's
`interval[0]`". For OpenBMI and PhysioNet the trigger is the cue onset (0 s offset), so the
epoch is literally 0.5-2.5 s post-cue. For IV-2a the trigger precedes the cue by 2 s, so the
epoch is cue + 0.5…2.5 s. For IV-2b the offset is 3 s and the cue timing itself is
ambiguous (see [IV-2b quirks](#bci-iv-2b--bnci2014_004)).

---

## 2. BCI Competition IV-2a — `BNCI2014_001` (Graz data set A)

### Subjects, sessions, runs
9 subjects, two recording sessions each, MOABB names them `0train` / `1test` `[A.1]`, 6 runs
per session, 386.94 s each `[B.BNCI2014_001]`. **The `.mat` file actually contains 9 runs per
session**: runs 0-2 carry no trials at all and are baseline recordings (118.7 / 80.7 / 165.9 s
in `A01T.mat`; 137.2 / 137.8 / 148.2 s in `A01E.mat`). MOABB's loader drops them, which is
why a "session" is 6 runs, not 9 `[E.BNCI2014_001.mat[A01T.mat]]`, `[E.BNCI2014_001.mat[A01E.mat]]`.

**Inter-session interval: UNVERIFIED.** The two sessions are documented as "on different
days" **[S]** ([MOABB BNCI2014_001 docstring](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html),
quoting the Graz description), but both sessions come back from MOABB with an identical
*synthetic* `meas_date = 2008-01-01 00:00 UTC` (MOABB sets the dataset year, not the real
recording date) `[B.BNCI2014_001]`, so the actual gap cannot be recovered from the files on
disk.

### Hardware, montage, reference
22 Ag/AgCl EEG electrodes at 3.5 cm inter-electrode distance, BrainAmp MR plus amplifiers,
Ag/AgCl cap, 250 Hz, online 0.5-100 Hz band-pass plus a 50 Hz notch. 3 EOG channels were
recorded in addition **[S]** ([BCI Competition IV data sets 2a/2b](https://www.bbci.de/competition/iv/):
"22 EEG channels (0.5-100Hz; notch filtered), 3 EOG channels, 250Hz sampling rate, 4 classes,
9 subjects"). Reference: left mastoid, ground right mastoid **[S]** (Graz description as
quoted in MOABB's docstring). Evidence status in the data itself: the 22 EEG channel names
and their order are exactly `IV2A_CHANNELS` `[E.BNCI2014_001]`, the 3 EOG channels are
`EOG1/EOG2/EOG3` `[B.BNCI2014_001]`, but **`raw.info` encodes no reference at all**
(`custom_ref_applied = 0`) and its `highpass/lowpass` are `0/125 Hz` — the online
0.5-100 Hz filter leaves no trace in the file, because the `.mat` holds only `X`, `trial`,
`y`, `fs`, `classes`, `artifacts`, `gender`, `age` `[E.BNCI2014_001.mat]`.

### Trial / cue timing and classes
4 classes: **left hand, right hand, both feet, tongue**, codes 1-4 assigned by MOABB in the
`.mat`'s own `classes` order, which is `['left hand', 'right hand', 'feet', 'tongue']`
`[E.BNCI2014_001.mat]`. Cued paradigm: fixation cross + warning tone at trial start (t = 0),
arrow cue at t = 2 s staying 1.25 s, imagery instructed until the fixation cross disappears
at t = 6 s, no feedback **[S]** ([competition page](https://www.bbci.de/competition/iv/) and
the Graz description as quoted in MOABB's docstring).

Measured structure `[B.BNCI2014_001]`, `[E.BNCI2014_001.mat]`, `[E.BNCI2014_001.trigger]`:

* 48 stim events per run, codes {1: 12, 2: 12, 3: 12, 4: 12}; 288 per session, exactly
  **72 per class** — the "expected 72 → 288" assumption is **CONFIRMED**.
* median inter-trigger interval 8.032 s (2008 samples at 250 Hz), i.e. the trial is 8.03 s
  long trigger-to-trigger (documented content 0-6 s plus a ~2 s break); first trigger 1.004 s
  into each run; the last trial ends before the run's 386.94 s.
* MOABB's annotations sit exactly 2.0 s after each trigger with a 4.0 s duration, i.e. at
  the documented cue, with a duration equal to the documented imagery period (2→6 s).
* Trigger identity, measured: the across-trial average of the unfiltered EEG (0.5-40 Hz,
  session `0train`, 282 of 288 trials — the first trial of each run has no 1.5 s of history)
  peaks at 1.5-2.5 µV in the quiet windows 0.6-1.4 s (2.5-3.1e-6 V) and 2.9-3.4 s
  (1.8-2.7e-6 V), i.e. at pre-trigger noise level (1.7-3.1e-6 V), while three windows carry a
  real deflection: **+0.05…0.35 s = 1.3-1.9e-5 V**, **+1.9…2.4 s = 0.8-1.2e-5 V**, and
  **+5.9…6.4 s = 4.2-8.4e-6 V** (per-channel SEM 0.58-0.76e-6 V; max across the 22 channels
  1.89e-5 / 1.22e-5 / 8.4e-6 V) `[E.BNCI2014_001.trigger]`. Three discrete stimulus-locked
  events at +0.2 s, +2.1 s and +6.1 s is exactly the documented timeline (warning
  tone + fixation cross at t = 0, arrow cue at t = 2 s, fixation cross disappears at t = 6 s)
  and is hard to explain if the trigger were already the cue. This is *indirect but
  quantitative*: I did not find a literal cue marker in the file, so "trigger = trial start,
  cue = trigger + 2 s" rests on the documented timeline plus this ERP structure. (The
  +5.9…6.4 s deflection could equally be end-of-trial movement/eye activity; either way it
  lands on a documented trial landmark.)

### Artifacts / EOG
There are **no artifact annotations**. The 3 EOG channels are present in the raw
`[B.BNCI2014_001]` but are dropped by the paradigm (`[D.BNCI2014_001]`), so nothing in this
repo's IV-2a arrays ever contains EOG. Trial-level artifact marks appear only as free text
in `raw.info['description']`, one string per run
`[B.BNCI2014_001]`, `[E.BNCI2014_001.mat]`:

| session | per-run count in `info['description']` | `.mat` `artifacts` sum |
|---|---|---|
| `0train` (A01T) runs 0-5 | 2, 4, 3, 1, 2, 3 (of 48) | 2, 4, 3, 1, 2, 3 — **MATCH** |
| `1test` (A01E) runs 0-5 | 3, 2, 1, 1, *(empty)*, *(empty)* | 3, 2, 1, 1, **0, 0** — **MATCH** |

Reading note that matters: MOABB appends the text **only when k > 0**, so an *empty*
description means **zero marked artifacts**, not missing information `[E.BNCI2014_001.mat]`.
Those marked trials are not excluded anywhere in our pipeline — the paradigm still returns
all 288 `[C.BNCI2014_001]`.

### Published baselines
* **Official BCI Competition IV results, data set 2a** (metric: Cohen's kappa, mean over 9
  subjects): 1. Kai Keng Ang (FBCSP) **0.57** (per-subject 0.68/0.42/0.75/0.48/0.40/0.27/0.77/0.75/0.61),
  2. Liu Guangquan **0.52**, 3. Wei Song **0.31**; chance kappa = 0.
  **[S]** [bbci.de results page](https://www.bbci.de/competition/iv/results/). kappa 0.57 on
  4 classes corresponds to ≈ 0.25 + 0.57·0.75 = **67.8 %** accuracy.
* **MOABB within-session benchmark** (5-fold CV, 2-class left vs right hand, ROC-AUC;
  reused from `docs/state_of_the_art.md` §1): CSP+LDA **82.34 ± 17.26**, MDM **81.69 ± 14.94**,
  ShallowConvNet **86.17 ± 13.74**, EEGNet_8_2 **77.15 ± 19.33**; all-classes accuracy:
  CSP+LDA **65.99 ± 15.47**, FgMDM **70.14 ± 15.13**, ShallowConvNet **72.47 ± 16.50**.
  **[S]** [MOABB benchmark results](https://moabb.neurotechx.com/docs/paper_results.html) /
  [MOABB paper, arXiv:2404.15319](https://arxiv.org/abs/2404.15319). The `state_of_the_art.md`
  numbers (82.3 / 81.7 / 86.2 / 77.2) match that page.
  *(The same page also shows `BNCI2014_001` "right hand vs feet" at 91-94 % and TS+LR 87.41 —
  useful context for how class choice, not method, drives most of the reported spread.)*
* Dataset paper: Tangermann et al. 2012, *Review of the BCI competition IV*, Front. Neurosci.
  **[S]** [doi:10.3389/fnins.2012.00055](https://doi.org/10.3389/fnins.2012.00055).
* `docs/state_of_the_art.md` §1 also lists **Won et al. 2022** (68.8 / 53.7 / 78.9 %). That is
  a *different* dataset (25 subjects, 5 sessions 2-3 days apart) and is **not one of the four
  datasets this project uses** — reusing its numbers as an IV-2a reference would be an
  apples-to-oranges comparison.

### 2a quirks
1. **MOABB's METADATA for `BNCI2014_001` is wrong and belongs to a different data set**
   `[A.1]`: it claims `n_subjects = 4`, `n_classes = 2`, `class_labels = ['left_hand','right_hand','foot']`,
   `trial_duration = 4.0`, `n_trials = {training: 200, test: 240}`,
   `bandpass 0.05-200 Hz`, `downsampled_to_hz = 100`, hardware `BrainAmp MR plus`, software
   `BCI2000`, `reference = 'none'` — that profile is BCI Competition IV **data set 1**
   (Berlin, 59 EEG channels, 1000 Hz), not Graz data set A. Everything in `METADATA` for this
   dataset should be treated as unreliable; the channel counts (`n_channels = 25`,
   `{eeg: 22, eog: 3}`) happen to be right. Only the docstring and the competition page agree
   with the data.
2. 50 Hz European mains (Graz) `[B.BNCI2014_001]`, plus 100 Hz as the first harmonic — which
   is what `src/preprocessing.py`'s `NOTCH_FREQS = [50.0, 100.0]` assumes.
3. The amplifier's online 0.5-100 Hz filter is invisible in the file (`highpass/lowpass = 0/125`),
   so any claim about the file's own bandwidth is attribution, not measurement.
4. The 9-runs-in-file / 6-runs-exposed discrepancy above, and the identical synthetic
   `meas_date` for both sessions.

---

## 3. BCI Competition IV-2b — `BNCI2014_004` (Graz data set B)

### Subjects, sessions, runs
9 subjects, **5 recordings each**: MOABB exposes them as `0train`, `1train`, `2train`,
`3test`, `4test`, one run per session, 2419.2 / 2209.8 / 1876.0 / 1865.3 / 1897.5 s
`[B.BNCI2014_004]`. The recordings come from two `.mat` files: `B01T.mat` holds 3 runs
(120, 120, 160 trials) and `B01E.mat` holds 2 (160, 160) `[E.BNCI2014_004.mat]`.

**Inter-session interval: UNVERIFIED** (both files again report a synthetic
`meas_date = 2008-01-01` `[B.BNCI2014_004]`).

### Hardware, montage, reference
"3 bipolar EEG channels (0.5-100Hz; notch filtered), 3 EOG channels, 250Hz sampling rate,
2 classes, 9 subjects" **[S]** ([competition page](https://www.bbci.de/competition/iv/));
MOABB's METADATA adds `hardware='g.tec'`, `software='rtsBCI (MATLAB/Simulink)'`,
`montage='standard_1020'`, `line_freq=50.0`, `reference='left mastoid'`, `ground='Fz'`,
`cap_manufacturer='Easycap'`, `electrode_material='Ag/AgCl'` `[A.2]`. The channel names are
`C3, Cz, C4` + `EOG1-3` `[B.BNCI2014_004]`, but note the **reference conflict**: the
competition page calls the three EEG channels *bipolar*, while MOABB's docstring says
"reference was the left mastoid and the ground was the right mastoid" (and MOABB's METADATA
says ground `Fz`). `raw.info` encodes nothing (`custom_ref_applied = 0`) `[B.BNCI2014_004]`.
For a 3-channel bipolar montage this matters: "C3/Cz/C4" are almost certainly not
commonly-referenced derivations, but I could not verify which pairs.

### Trial / cue timing and classes
2 classes: left hand (code 1) / right hand (code 2) `[A.2]`, `[E.BNCI2014_004.mat]`.
MOABB's `interval` is `[3, 7.5]`, so our epoch lands at trigger + 3.5…5.5 s with a 4.5 s
annotation duration `[B.BNCI2014_004]`, `[C.BNCI2014_004]`.

Measured structure:
* 120 trials (60/class) in the first two recordings, 160 (80/class) in the last three
  `[C.BNCI2014_004]`, matching MOABB's own METADATA `n_trials = {'screening': 120, 'feedback': 160}`
  `[A.2]`. MOABB's METADATA gives `trial_duration = 7.5 s` (consistent with its `interval [3, 7.5]`);
  the measured trigger-to-trigger spacing is 9.8-10.3 s in the four dense recordings and
  17.3 s in `0train` (the recording includes the inter-trial break).
* median inter-trigger interval: **17.308 s** in `0train` but 9.82 / 10.07 / 10.06 / 10.33 s
  in the other four `[B.BNCI2014_004]`; the first trigger sits 220.56 s into the recording in
  four of five sessions (20.6 s in `1train`) `[E.BNCI2014_004.mat]`. The sparse first
  recording is a real property of subject 1's data, not an artifact of our loading.
* Trigger identity: 2b has only 3 central EEG channels and no posterior sites, so the ERP
  test is weak. Max-across-channels amplitudes per session for subject 1
  `[E.BNCI2014_004.trigger]`:

  | session | +0.05-0.35 s | +0.6-1.4 s | +1.9-2.4 s | +2.9-3.4 s | +5.9-6.4 s |
  |---|---|---|---|---|---|
  | `0train` | 1.03e-6 | 1.09e-6 | 1.63e-6 | **2.86e-6** | 6.7e-7 |
  | `1train` | 8.6e-7 | 1.93e-6 | 2.68e-6 | **3.13e-6** | 8.5e-7 |
  | `2train` | 2.46e-6 | 2.56e-6 | 1.73e-6 | **3.45e-6** | 6.9e-7 |
  | `3test` | 2.39e-6 | 7.1e-7 | 2.08e-6 | **3.33e-6** | 5.0e-7 |
  | `4test` | 2.42e-6 | 1.93e-6 | 1.45e-6 | 2.39e-6 | 8.3e-7 |

  (per-channel SEM 2.3-4.5e-7 V; pre-trigger window 0.5-1.6e-6 V). The largest deflection is
  at **+2.9…3.4 s in 4 of the 5 sessions** and effectively tied in the fifth — *suggestive* of a
  cue ~3 s after the trigger, and hence of MOABB's `interval[0] = 3`, but all the windows are
  within a factor of ~3-4 of each other, so this is **not conclusive**. The exact cue latency
  is **UNVERIFIED**.

### Artifacts / EOG
No artifact annotations; per-recording text in `info['description']` matches the `.mat`
counts exactly: 18/120, 22/120, 36/160, 48/160, 44/160 `[B.BNCI2014_004]`,
`[E.BNCI2014_004.mat]`. 3 EOG channels are present and dropped by the paradigm.

### Published baselines
* **Official BCI Competition IV results, data set 2b** (kappa): 1. Zheng Yang Chin (FBCSP)
  **0.60** (0.40/0.21/0.22/0.95/0.86/0.61/0.56/0.85/0.74), 2. Huang Gan **0.58**,
  3. Damien Coyle **0.46** **[S]** [results page](https://www.bbci.de/competition/iv/results/).
  kappa 0.60 on 2 classes ≈ 80 % accuracy.
* **MOABB within-session benchmark** (2-class, ROC-AUC) **[S]**
  [paper results](https://moabb.neurotechx.com/docs/paper_results.html): CSP+LDA
  **80.10 ± 14.93**, MDM **77.66 ± 15.78**, FgMDM **79.28 ± 15.25**, ShallowConvNet
  **72.36 ± 18.05**, EEGNet_8_2 **69.50 ± 19.50**, TS+LR **80.09 ± 15.01**.
* Dataset paper: Leeb et al. 2007 (IEEE TNSRE) **[S]**
  [doi:10.1109/TNSRE.2007.906956](https://doi.org/10.1109/TNSRE.2007.906956) (as listed in
  MOABB's METADATA `[A.2]`). The competition's own PDF description
  (`https://www.bbci.de/competition/iv/desc_2b.pdf`) could not be fetched by this session's
  tooling (PDF content type), so any detail taken from it is **UNVERIFIED**.

### 2b quirks (the biggest one is a real problem for this repo)
1. **The 5 recordings are not 5 protocol-equivalent sessions, and MOABB's `train`/`test`
   naming is file-based, not protocol-based.** MOABB's METADATA states the sessions are
   `['01T','02T','03T','04E','05E']`, with **01T and 02T the screening sessions without
   feedback** and **03T, 04E, 05E the feedback sessions** `[A.2]`; the measured trial counts
   (120, 120, 160, 160, 160) match that split `[E.BNCI2014_004.split]`. So
   `src/datasets.py`'s `IV2B_CALIB_SESSIONS = {0train, 1train, 2train}` is **2 screening
   recordings + 1 feedback recording**, grouped by which file they came from (T vs E), while
   `IV2B_EVAL_SESSIONS = {3test, 4test}` are 2 feedback recordings. The docstring's claim that
   this is "the dataset's own canonical split" is only true in the sense that MOABB's session
   names are the T/E files; as an equal-protocol split it is **REFUTED**.
2. MOABB's 2b docstring says the cue appears 2 s after the trial start (text copied from 2a)
   while MOABB's own `interval` for 2b implies a 3 s pre-cue period. The two cannot both be
   right; the ERP evidence above leans towards 3 s but is not decisive.
3. 50 Hz European mains `[B.BNCI2014_004]`.
4. `1train` for subject 1 has an unusual 20.6 s pre-first-trial onset and a 9.82 s median
   trial spacing versus 17.3 s in `0train`.

---

## 4. OpenBMI / Lee2019 MI — `Lee2019_MI`

### Subjects, sessions, runs
54 healthy subjects (24-35 y; 25 female; 38 BCI-naive), **2 sessions on different days**,
54 subjects × 2 sessions in the full release; 20 subjects are cached locally `[A.3]`
**[S]** ([Lee et al. 2019, GigaScience, PMC6501944](https://pmc.ncbi.nlm.nih.gov/articles/PMC6501944/)).
The exact number of days between sessions is **UNVERIFIED** (the paper says only "two sessions
on different days"). Each session contains an offline training phase and an online test phase,
each 100 balanced left/right trials **[S]** (paper). MOABB's `Lee2019_MI` loads **only the
offline training run** by default (`train_run=True`, `test_run = paradigm == "p300"` is False
for MI), so the "session" we see is 1 run of 1519.2 s `[B.Lee2019_MI]`.

### Hardware, montage, reference
BrainAmp (Brain Products), **62 Ag/AgCl electrodes** covering the whole head, sampled at
**1000 Hz**, **nasion-referenced, grounded to AFz**, impedances < 10 kΩ, plus **4 EMG
electrodes** (flexor digitorum profundus, olecranon reference) **[S]** (paper, as quoted above
and in MOABB's docstring). Measured in the raw `[B.Lee2019_MI]`: 62 EEG + 4 EMG + 1 stim
channel, `sfreq = 1000 Hz`, `highpass/lowpass = 0/500`, montage `standard_1005`,
`custom_ref_applied = 0` (again no reference recorded in the file), and **no EOG channels at
all** — this dataset provides EMG, not EOG.

### Trial / cue timing and classes
2 classes, and note the code order: MOABB's `event_id` is `{'left_hand': 2, 'right_hand': 1}`
`[A.3]` — the **opposite** convention to the two Graz datasets (1 = left hand there). The
paradigm used by `src/datasets.py` returns labels, so this is harmless today, but any code
that compares raw trigger codes across datasets is comparing different things.

Paradigm: 3 s of black fixation cross, then the imagery task for **4 s** cued by a left/right
arrow, then a blank screen for **6 s ± 1.5 s**; training = 100 balanced trials (50/class), the
online test phase feeds back a moving cross **[S]** (paper). The paper's own MI analysis
segments "1,000 to 3,500 ms with respect to stimulus onset", which only makes sense if the
trigger *is* the cue onset **[S]** (paper).

Measured `[B.Lee2019_MI]`, `[C.Lee2019_MI]`: 100 stim events, codes {1: 50, 2: 50};
annotations at trigger + 0 s with a 4.0 s duration; median inter-trigger interval **11.067 s**,
consistent with a cue-onset trigger if the trial is 4 s imagery + a 6±1.5 s blank; first
trigger 155.9 s into the recording. (Documented trial ≈ 3 s fixation + 4 s imagery +
6±1.5 s blank ≈ 13 s, so 11.07 s cue-to-cue is inside the documented tolerance and is *not*
compatible with a fixation-cross-onset trigger, which would space cues ≈ 13 s apart.)

### Artifacts / EOG
No artifact annotations, no EOG. The dataset instead ships *separate* 10 s recordings of eye
blinks, horizontal/vertical eye movements, teeth clenching and arm flexing, plus pre/post-task
resting-state recordings, and the 4 arm-EMG channels in every file **[S]** (paper; and
`[B.Lee2019_MI]` confirms the 4 EMG channels). MOABB ignores all of that except the EMG
channels, and `info['description']` is empty `[B.Lee2019_MI]`. Our pipeline never removes eye
blinks from this dataset — there is nothing to remove them with.

### Published baselines
* **Lee et al. 2019** (same paper, 54 subjects × 2 sessions): mean MI decoding accuracy
  **71.1 % (± 0.15)**, MI **BCI-illiteracy rate 53.7 %** (vs 11.1 % ERP and 10.2 % SSVEP);
  27.8 % (15/54) of users were universally literate **[S]**
  ([PMC6501944](https://pmc.ncbi.nlm.nih.gov/articles/PMC6501944/),
  [doi:10.1093/gigascience/giz002](https://doi.org/10.1093/gigascience/giz002)). These are the
  numbers `docs/state_of_the_art.md` §1 quotes, and they check out against the paper.
* **MOABB within-session benchmark** (2-class, ROC-AUC) **[S]**
  [paper results](https://moabb.neurotechx.com/docs/paper_results.html): CSP+LDA
  **76.88 ± 17.41**, MDM **70.23 ± 13.87**, FgMDM **81.34 ± 13.93**, ShallowConvNet
  **75.83 ± 15.04**, EEGNet_8_2 **65.67 ± 16.43**, TS+LR **83.09 ± 13.46**.

### OpenBMI quirks
1. **MOABB ≤ 1.5.0 exposes only one of the two recording days, and it is the *second* one.**
   The loader builds session names as `str(session - 1)` (so `"0"`, `"1"`) but `get_data`
   filters the returned sessions against `{str(s) for s in _selected_sessions}` = `{"1","2"}`
   `[E.Lee2019_MI.sessions]`. A stubbed-load probe on the real filtering path shows
   `Lee2019_MI()` (default `sessions=(1,2)`) keeps only `['1']`; with real data,
   `Lee2019_MI(sessions=[1])` returns an **empty dict** — the first recording day (session
   name `"0"`) is unreachable through `get_data`. `src/datasets.py`'s comment "MOABB's
   MotorImagery paradigm returns one recording session per subject for this dataset, ~100
   trials" is **CONFIRMED as an outcome** but the stated reason is wrong: it is a
   session-naming/filter mismatch in the *dataset loader*, not a paradigm behaviour.
2. **Loading cost.** Measuring one subject's raw peaked at **4.14 GB RSS** `[B.Lee2019_MI]`
   (the loader reads the whole ~580 MB `.mat` into memory and MOABB then builds several
   1.5 Msample × 66-channel copies). `docs/compute_environment.md` budgets ≤ 5 GB per process
   and ≤ 2 GB for this audit. The audit halves this by restricting the loader loop to the one
   surviving session; the *repo's own* `Lee2019_MI()` call iterates over both session files
   (`Lee2019.py`, `_get_single_subject_data`), so it holds both sessions' arrays at once
   before discarding one — I did not run that configuration, and the exact figure is
   **UNVERIFIED** (it is the reason this audit does not).
3. FCz is **not** in the 62-channel montage `[E.misc]` — see §6.
4. `line_freq = 60.0` comes from MOABB's METADATA injection, not from the file
   `[B.Lee2019_MI]`, `[D.Lee2019_MI]`; the 1000 Hz sampling rate is **not** downsampled by
   MOABB, so an epoch is 62 × 2001 = 124 k samples (≈1 MB float64) versus 22 × 501 = 11 k for
   IV-2a — about **11× more data per trial**, and 4× the samples per channel
   `[C.Lee2019_MI]`. The paper's own analysis downsampled to 100 Hz; we do not.

---

## 5. PhysioNet EEG Motor Movement/Imagery — `PhysionetMI`

### Subjects, sessions, runs
109 subjects, 1 MOABB session, **6 runs** per subject for the default
`PhysionetMI(imagined=True, executed=False)`: the imagined left/right fist runs (4, 8, 12) and
the imagined both-fists/both-feet runs (6, 10, 14), 125 s each `[B.PhysionetMI]`,
`[A.4]`. The dataset itself has 14 runs (two 1-minute baselines with eyes open/closed, and
three repetitions of each of the four execution/imagery conditions, 2 minutes each) **[S]**
([PhysioNet EEGMMIDB 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/)); MOABB silently
drops the 8 runs it does not use, including the executed-movement runs. **On disk** the cache
holds exactly **109 subject directories and 654 EDF files = 6 files per subject**, i.e. only
the six imagined-movement runs (`R04, R06, R08, R10, R12, R14` — checked for `S001` and
`S088`); the two baseline runs and the six executed runs are not cached, so anything that
switches `imagined`/`executed` would need a download `[filesystem check]`.
"Sessions": the dataset has one recording per subject, so there is no inter-session interval —
but also no cross-session data at all. **[V]** (1 session) / **[S]** (structure).

### Hardware, montage, reference
64-channel EEG recorded with the **BCI2000** system, electrodes placed on the international
10-10 system, **160 Hz**, EDF+ with annotation channels; the montage excludes Nz, F9, F10,
FT9, FT10, A1, A2, TP9, TP10, P9 and P10 **[S]** (PhysioNet record page). Measured: 64 EEG +
1 STIM channel, `sfreq = 160 Hz` (except subject 88), `highpass/lowpass = 0/80`,
`standard_1005` montage, `custom_ref_applied = 0` `[B.PhysionetMI]`. **Reference: UNVERIFIED** —
the PhysioNet page does not state it; MOABB's METADATA says `'mastoid'` `[A.4]`, and no
reference is encoded in the file. The `meas_date` here *is* real (taken from the EDF header,
e.g. 2009-08-12 16:15 UTC) `[B.PhysionetMI]`.

### Trial / cue timing and classes
Runs come in two flavours, and the meaning of the EDF's `T0/T1/T2` codes depends on which
flavour a run is **[S]** (PhysioNet page): in the unilateral runs `T1` = left-fist onset,
`T2` = right-fist onset; in the bilateral runs `T1` = both-fists, `T2` = both-feet; `T0` = rest.
MOABB maps them to `rest=1, left_hand=2, right_hand=3, hands=4, feet=5` `[A.4]`. Each run is
125 s with **29 annotations** for subject 1 (14 `rest`, 8/7 for the two task classes),
annotation duration 3.0 s (= MOABB's `interval [0,3]`), median inter-annotation gap 4.15 s,
first onset 4.2 s `[B.PhysionetMI]`. The trigger is the annotation itself (offset 0 s), i.e.
the target/cue onset `[B.PhysionetMI]`. **Per-trial target duration is not documented** on the
PhysioNet page (it documents 2-minute task runs); MOABB's dataset comment says the trial
spacing is "4 second" and its `interval` is `[0,3]`, and the measured annotation spacing is
4.15 s — so a trial is ≈4.1 s with MOABB epoching 0.5-2.5 s post-onset `[C.PhysionetMI]`.

With the 4 imagery event types, subject 1 yields **90 trials** (feet 24, hands 21, left 23,
right 22) `[C.PhysionetMI]`, `[E.PhysionetMI]`.

### Artifacts / EOG
There are **no EOG channels** and no artifact annotations `[B.PhysionetMI]`. The `rest`
annotations are not artifacts, they are inter-trial baseline periods (see the quirk below).

### Published baselines
* The dataset paper (Schalk et al. 2004, *BCI2000*, IEEE TBME **[S]**
  [doi:10.1109/TBME.2004.827072](https://doi.org/10.1109/TBME.2004.827072)) describes the
  system, **not** a motor-imagery decoding baseline; nor does the PhysioNet record page.
  There is therefore **no dataset-paper baseline number to quote for PhysioNet MI**.
* **MOABB within-session benchmark** (the only numbers I could attribute) **[S]**
  [paper results](https://moabb.neurotechx.com/docs/paper_results.html): 2-class left vs right
  ROC-AUC — CSP+LDA **65.75 ± 17.37**, MDM **54.76 ± 16.79**, FgMDM **68.46 ± 19.06**,
  ShallowConvNet **65.19 ± 15.80**, EEGNet_8_2 **59.55 ± 15.95**, TS+LR **67.28 ± 19.19**;
  all-classes accuracy — CSP+LDA **47.73 ± 14.35**, FgMDM **55.04 ± 14.17**, ShallowConvNet
  **41.87 ± 12.50**, TS+EL **59.93 ± 14.07**. PhysioNet is by far the weakest of the four
  datasets in that benchmark.

### PhysioNet quirks
1. **Subject 88 is recorded at 128 Hz instead of 160 Hz — CONFIRMED** `[E.PhysionetMI]`
   (subject 88: 128.0 Hz; subject 1: 160.0 Hz; both 65 channels). `src/datasets.py` excludes
   it from the reference pool and raises `ValueError` if asked for it, which the audit
   confirms by calling `load_physionet_reference_subject(88)` and catching the error
   `[E.PhysionetMI]`. Mixing it in would silently produce 257-sample epochs next to
   321-sample epochs.
2. **`MotorImagery(n_classes=2)` without an explicit `events=` includes `rest` — CONFIRMED**
   `[E.PhysionetMI]`: it returns **all 5** event types and 174 trials for subject 1, of which
   **84 are `rest` (48.3 %)** — exactly the ~48 % `src/datasets.py` warns about. The cause is
   `MotorImagery.used_events`: with `events=None` it copies *every* entry of
   `dataset.event_id` and then overwrites `n_classes` with that count, so the requested
   `n_classes` never filters anything. `PHYSIONET_MI_EVENTS = ['left_hand','right_hand','feet','hands']`
   fixes it (90 trials, no `rest`) `[C.PhysionetMI]`.
3. `line_freq = 60.0` is MOABB METADATA, not the file: reading `S001R04.edf` directly gives
   `info['line_freq'] = None`, 160 Hz, 64 channels, 30 annotations `[E.PhysionetMI]`.
   (US recording site ⇒ 60 Hz is plausible, but it is MOABB's claim.)
4. The raw MOABB hands over carries **both** the EDF annotations and a synthesized `STIM`
   channel built from them (`stim_channels_with_selected_ids`), so the trigger offset is 0 s
   `[B.PhysionetMI]`.

---

## 6. Cross-dataset checks (what the repo's constants assume)

All **[V]** `[E.misc]`:

| claim in the repo | verdict |
|---|---|
| `IV2A_CHANNELS` (22) is the channel order MOABB returns for IV-2a | **CONFIRMED**, same set *and* same order `[E.BNCI2014_001]` |
| `OPENBMI_CHANNELS` (62) is the order MOABB returns for OpenBMI | **CONFIRMED**, same set and order `[E.Lee2019_MI]` |
| `PHYSIONET_CHANNELS` (64) is the order MOABB returns for PhysioNet | **CONFIRMED**, same set and order `[E.PhysioNetMI]` |
| OpenBMI's montage is missing FCz vs IV-2a's 22-channel montage | **CONFIRMED**: `'FCz' in IV2A_CHANNELS` is True, `'FCz' in OPENBMI_CHANNELS` is False `[E.misc]` |
| `SHARED_CHANNELS` = IV-2a minus FCz = 21 channels | **CONFIRMED** (`len = 21`, difference exactly `['FCz']`) |
| PhysioNet's montage is a complete superset of IV-2a's 22 channels | **CONFIRMED**: all 22 present in the 64-channel raw EEG list (FCz included) `[E.PhysionetMI]` |
| OpenBMI samples at a much higher rate than the Graz sets | **CONFIRMED**: 1000 Hz vs 250 Hz (and no MOABB resampling) `[D.*]` |
| IV-2a mains is 50 Hz and it is the only dataset with "European" mains | **CONFIRMED for the value (50 Hz)**, and 2b is also 50 Hz; OpenBMI/PhysioNet report 60 Hz **but only via MOABB METADATA**, not from their files `[B.*]` |
| IV-2a yields exactly 288 trials/session, 72 per class | **CONFIRMED** in both sessions `[C.BNCI2014_001]` |
| `docs/compute_environment.md`: "`288 × 22 × 500 float64` ≈ 25 MB per session" | **mostly CONFIRMED**: the real epoch length is **501** samples (0.5-2.5 s inclusive at 250 Hz) ⇒ 25.4 MB; the "500" is off by one `[E.misc]` |

Also worth stating plainly because it is easy to get wrong: **the same integer event code
means different classes in different datasets** (2a: left=1; 2b: left=1; OpenBMI: left=2;
PhysioNet: left=2, rest=1) `[A.*]`, `[E.*.mat]`.

---

## 7. What our pipeline does vs what the dataset provides

"Where it comes from" is the file and line in this repo. Everything MOABB-side was verified in
§1; the repo-side choices are read from the source (they are ours, not third-party).

| pipeline choice | value used | what the dataset provides | where it comes from |
|---|---|---|---|
| band-pass | **8-30 Hz**, `method="iir"`, `picks="data"` | 2a/2b were recorded with an online 0.5-100 Hz band-pass + 50 Hz notch; OpenBMI/PhysioNet files carry no filter record at all | `FMIN/FMAX = 8.0/30.0` in `src/config.py:27`, passed to `MotorImagery(fmin=, fmax=)` in every loader in `src/datasets.py`; MOABB applies it (§1) |
| epoch window | **0.5-2.5 s** after the trigger, `baseline=None` | 2a: cue is 2 s after the trigger ⇒ we use cue + 0.5…2.5 s; OpenBMI/PhysioNet: trigger *is* the cue ⇒ literally 0.5-2.5 s post-cue; 2b: offset 3 s, cue latency unverified | `TMIN/TMAX = 0.5/2.5` in `src/config.py:30`; MOABB adds `interval[0]` (§1) |
| wide window (wavelet calibration) | **-1.0…4.0 s**, **1-40 Hz** — deliberately *not* 8-30 to keep the ERD/ERS band search open and to have a pre-cue baseline | same raw recordings; the pre-cue part exists because the 2a trigger precedes the cue | `WAVELET_CALIB_*` in `src/datasets.py:148-149`, `load_iv2a_subject_wide_window` |
| artifact cleaning | notch 50+100 Hz → 1-40 Hz → ICA (20 components, `random_state=42`) → drop components from `find_bads_eog` ∪ `find_bads_muscle`, on the **continuous** session, cached in `data/cleaned/` | 2a ships 3 EOG channels and per-trial artifact *marks* (nothing else); OpenBMI ships EMG + separate artifact recordings; PhysioNet/2b ship EOG only | `src/preprocessing.py:39-74`; consumed only by `load_iv2a_subject_cleaned` (`src/datasets.py:187-219`), used by `experiments/cleaned_riemannian_check.py` |
| ICA cleaning is default? | **No.** The mainline IV-2a loaders (`load_iv2a_subject`, `load_iv2a_subject_wide_window`) never clean; artifact-marked trials are still trained and tested on | marked trials are present in the paradigm output (288/session) | `src/datasets.py:125-184`; `[C.BNCI2014_001]` |
| channel restriction | default: none (22 ch). For the OpenBMI-derived population prior: the **21** channels shared with IV-2a (`SHARED_CHANNELS`, drops FCz). For the PhysioNet-derived prior: all **22** | 2a: 22 EEG; OpenBMI: 62 EEG with **no FCz**; PhysioNet: 64 EEG (superset of the 22) | `src/datasets.py:57-88`, `restrict_channels` at `:108`, used by `load_iv2a_subject_shared_channels`, `load_openbmi_reference_subject`, `load_physionet_reference_subject` |
| trace normalisation | **off** in the main classifier: `CSP(n_components=8, reg="ledoit_wolf", log=True, norm_trace=False)`; the tangent-space modules expose `normalize_trace` and default it **False** (`spd_shrinkage`, `geometric_correction`), except `prototypical_network.trial_tangent_vectors` which defaults it **True** | raw amplitudes differ a lot between subjects and hardware (per-epoch `|X|` max in µV: 42 (2a), 16 (2b), 157 (OpenBMI), e.g. `[C.*]`), so this is a real cross-dataset scale difference that is *not* normalised in the default path | `src/baselines.py:18`; `src/adaptation/spd_shrinkage.py:68-87`, `src/adaptation/geometric_correction.py:165-170`, `src/adaptation/prototypical_network.py:95-110` |
| units | arrays are **microvolts** (raw files are volts; MOABB multiplies by `unit_factor = 1e6`) | all four files are in volts | §1; `src/datasets.py:211` converts the cleaned path by hand (`* 1e6`) to match |
| resampling | **none anywhere** | 250 / 250 / 1000 / 160 Hz; subject 88 128 Hz (excluded) | §1; `PARADIGM` has `resample=None` |
| artifact-rejection / bad-trial exclusion | **none** | 2a/2b mark artifacts per trial but provide no annotations; PhysioNet's `rest` is not an artifact | §1, §2, §3 |

The one place where "what we do" silently differs from "what the dataset provides" in a way
that could change results: **we never use the EOG/EMG channels that exist in 2a, 2b and
OpenBMI**, and we never exclude the trials the dataset itself flagged as artifactual
(2a: 1-4 of 48 per run; 2b: 18-48 of 120/160). Both are choices, not oversights — but they
mean our accuracy numbers are "as the file came", strictly above the protocol the recording
lab intended.

---

## 8. Known gaps / things I could not verify

1. **Real recording dates / inter-session gaps.** 2a and 2b report a synthetic
   `meas_date = 2008-01-01` for both sessions; only "different days" is documented. OpenBMI's
   gap is not stated in the paper. (PhysioNet's `meas_date` is real but it has one session.)
2. **References.** No dataset encodes a reference in `raw.info`; the values in section 2-5 are
   attribution (competition page / MOABB docstrings / paper), and for 2b the sources
   contradict each other ("bipolar" vs "left mastoid"). PhysioNet's reference is not stated on
   the PhysioNet page at all.
3. **The exact cue latency in IV-2b** (2 s per a copy-pasted docstring, 3 s per MOABB's
   interval) — the ERP evidence leans 3 s but only 3 central channels exist.
4. **Whether an online mains notch was applied** to OpenBMI and PhysioNet: neither file
   records a filter, and `line_freq` is MOABB metadata.
5. **The competition's 2a/2b description PDFs** (`desc_2a.pdf`, `desc_2b.pdf`) could not be
   fetched with this session's tooling (PDF content type unsupported), so the trial-structure
   details attributed to them come via MOABB's docstrings.
6. **Peak memory of the repo's own OpenBMI call path** (both session files loaded): measured
   only for the single-session variant (4.14 GB).
7. **Everything beyond subject 1** (IV-2a, IV-2b, OpenBMI) and subject 88 for PhysioNet: the
   audit inspects one representative subject per dataset by design
   (`docs/compute_environment.md` rule 2). Per-subject variation in trial counts or
   descriptions is therefore not characterised.

## 9. Sources

* BCI Competition IV home page (data set 2a/2b channel counts, filters, sampling rates) —
  <https://www.bbci.de/competition/iv/>
* BCI Competition IV official results (2a kappa 0.57, 2b kappa 0.60, per-subject columns) —
  <https://www.bbci.de/competition/iv/results/>
* Tangermann et al. 2012, *Review of the BCI Competition IV* —
  <https://doi.org/10.3389/fnins.2012.00055>
* Leeb et al. 2007, Graz data set B (2b) — <https://doi.org/10.1109/TNSRE.2007.906956>
* Lee et al. 2019, *EEG dataset and OpenBMI toolbox…*, GigaScience (54 subjects, 1000 Hz,
  62 Ag/AgCl, nasion/AFz, trial timing, 71.1 %, 53.7 % illiteracy) —
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC6501944/> ·
  <https://doi.org/10.1093/gigascience/giz002>
* Schalk et al. 2009, *EEG Motor Movement/Imagery Dataset* v1.0.0, PhysioNet (64 ch, 10-10,
  160 Hz, 14 runs, T0/T1/T2, montage exclusions) — <https://physionet.org/content/eegmmidb/1.0.0/>
* Schalk et al. 2004, *BCI2000* — <https://doi.org/10.1109/TBME.2004.827072>
* MOABB benchmark results (within-session 5-fold CV numbers for all four datasets) —
  <https://moabb.neurotechx.com/docs/paper_results.html> ·
  MOABB paper <https://arxiv.org/abs/2404.15319>
* MOABB 1.5.0 dataset/paradigm source read locally:
  `.venv/Lib/site-packages/moabb/{datasets/{bnci/bnci_2014.py,bnci/base.py,Lee2019.py,physionet_mi.py,base.py,preprocessing.py,bids_interface.py,_channel_pick.py},paradigms/{base.py,motor_imagery.py}}`
