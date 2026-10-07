"""Scalp flow-field features: optical flow (Horn & Schunck 1981, borrowed from
computer vision — the same cross-disciplinary-transplant pattern as
riemannian_icp.py's robotics loan) applied to the spatial topography of
motor-imagery ERD/ERS power as it evolves across a trial.

Motivation, not yet validated on real data: ERD/ERS during motor imagery is
known to spread spatially across the sensorimotor cortex over the course of
a trial (traveling-wave / spatial-spread literature), not just change in
overall magnitude at fixed electrodes. CSP and Riemannian alignment both
discard that temporal-spatial trajectory — they summarize a trial as a single
covariance matrix. This module keeps it: each trial becomes a short sequence
of scalp topography "frames" (mu/beta power envelope, windowed), and optical
flow between consecutive frames gives a per-timestep 2D velocity field
describing how that power pattern is moving across the scalp — a feature no
prior phase in this project has used.

Pipeline per trial: band-limited signal -> Hilbert envelope -> windowed frames
-> per-frame interpolation onto a 2D grid (electrodes are sparse, non-uniform
points; optical flow needs a dense image) -> Horn-Schunck flow between
consecutive frames.

Everything here is a signal-processing primitive with a known ground truth to
check it against (electrode projection topology, synthetic translating-blob
flow recovery) — see experiments/stage0_flow_field_validation.py. A second
gate, experiments/stage0_flow_field_eeg_validation.py, checks the full
pipeline (envelope -> windows -> interpolation -> flow) against a known,
planted spatial drift in synthetic EEG and found a real, direction-dependent
limitation worth knowing before using this on real trials: IV-2a's electrode
layout has a dense chain of electrodes along the left-right central row
(C5-C3-C1-Cz-C2-C4-C6) but no equivalent dense chain in the anterior-
posterior direction, so left-right drift is recovered accurately from far
fewer trials than anterior-posterior drift needs at the same SNR (a
planted FC3->CP3 drift needed ~200 trial-averaged repeats to resolve
cleanly, vs. a clean result for C3<->C1 from far fewer) — confirmed to be
this geometric cause and not a coordinate-system bug by checking that a
reversed planted direction (C1->C3) reverses the recovered flow almost
exactly (cosine similarity -0.998), rather than a fixed setting improving
one direction and not the reverse of it. Practical implication: any
downstream use of this module on real, low-trial-count data should expect
directionally uneven reliability, not treat the flow field as spatially
isotropic.

On real IV-2a data (experiments/phase9_flow_field_features.py,
phase9b_flow_field_plus_csp.py): the 6-scalar summary features
(flow_summary_features) carry real, statistically significant signal above
chance standalone (28.7-29.3% vs. 25% chance, Wilcoxon p<0.05 across 9
subjects) but are far weaker than CSP (62.2%) or Riemannian alignment
(66.9%), and naively concatenating them onto CSP's features hurts rather
than helps (59.4% vs. 62.2%, p=0.043) — the mechanism works and transfers
to real data, but isn't currently a useful classifier on its own or in
simple combination. See docs/progress_and_direction.md's Phase 8/9 section.
"""

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.signal import hilbert


def project_3d_to_2d(xyz: np.ndarray) -> np.ndarray:
    """Projects 3D scalp position(s) to a 2D disc: azimuth preserved, radius
    = (pi/2 - elevation) / (pi/2), so the vertex (Cz) lands near the origin
    and positions further from it land further from the origin — the same
    polar projection used for EEG topomaps. xyz: (..., 3) -> (..., 2), so a
    single (3,) position or a stack of them both work.
    """
    xyz = np.asarray(xyz, dtype=float)
    r = np.linalg.norm(xyz, axis=-1)
    azimuth = np.arctan2(xyz[..., 1], xyz[..., 0])
    elevation = np.arcsin(xyz[..., 2] / r)
    r2d = (np.pi / 2 - elevation) / (np.pi / 2)
    return np.stack([r2d * np.cos(azimuth), r2d * np.sin(azimuth)], axis=-1)


def electrode_positions_2d(channel_names: list[str]) -> np.ndarray:
    """Looks up each channel's standard_1020 3D scalp position and projects
    it to 2D via project_3d_to_2d. Returns an (n_channels, 2) array in the
    same order as channel_names.
    """
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    positions_3d = montage.get_positions()["ch_pos"]
    missing = [c for c in channel_names if c not in positions_3d]
    if missing:
        raise ValueError(f"channels not found in standard_1020 montage: {missing}")

    xyz = np.array([positions_3d[c] for c in channel_names])
    return project_3d_to_2d(xyz)


def interpolate_topomap(values: np.ndarray, positions_2d: np.ndarray, grid_size: int = 20) -> np.ndarray:
    """Interpolates per-electrode scalar values onto a dense grid_size x
    grid_size grid over [-1.2, 1.2]^2, via thin-plate-spline RBF interpolation
    (smooth, exact at the electrode points, well-defined off them — unlike
    nearest-neighbor or linear, which would give a flow field with sharp
    facets from the sparse, non-uniform electrode layout).
    """
    axis = np.linspace(-1.2, 1.2, grid_size)
    grid_x, grid_y = np.meshgrid(axis, axis)
    query = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    rbf = RBFInterpolator(positions_2d, values, kernel="thin_plate_spline")
    return rbf(query).reshape(grid_size, grid_size)


_HORN_SCHUNCK_KERNEL = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], dtype=float) / 12.0


def _neighborhood_average(field: np.ndarray) -> np.ndarray:
    from scipy.signal import convolve2d

    return convolve2d(field, _HORN_SCHUNCK_KERNEL, mode="same", boundary="symm")


def horn_schunck_flow(frame1: np.ndarray, frame2: np.ndarray, alpha: float = 0.1, n_iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Horn & Schunck (1981) dense optical flow between two equal-shaped 2D
    frames. Returns (u, v): the x- and y-velocity fields, one value per grid
    cell, describing the apparent motion of frame1's pattern into frame2's.

    alpha controls the smoothness penalty (higher = flow varies more slowly
    across the grid), scaled against the frames' own gradient magnitude in
    the update rule (denom = alpha**2 + Ix**2 + Iy**2) — so its correct
    value is data-scale-dependent, not a universal constant. Horn & Schunck's
    classic alpha=1.0 default assumes ~8-bit-scale image intensities with
    correspondingly large gradients; this module's topomaps are normalized
    envelope values in ~[0, 1] with much smaller gradients, so alpha=1.0
    made alpha**2 dominate the denominator and shrank recovered flow to
    ~43% of a known planted velocity regardless of that velocity's
    magnitude — a flat-ratio signature of over-regularization, not the
    small-motion linearization breaking down (which would degrade more at
    larger displacements, not stay constant). alpha=0.1 was verified
    empirically (experiments/stage0_flow_field_validation.py) to recover
    >=95% of a planted translation at this data's gradient scale.
    """
    Ix = 0.5 * (np.gradient(frame1, axis=1) + np.gradient(frame2, axis=1))
    Iy = 0.5 * (np.gradient(frame1, axis=0) + np.gradient(frame2, axis=0))
    It = frame2 - frame1

    u = np.zeros_like(frame1)
    v = np.zeros_like(frame1)
    denom = alpha**2 + Ix**2 + Iy**2
    for _ in range(n_iters):
        u_avg = _neighborhood_average(u)
        v_avg = _neighborhood_average(v)
        numerator = Ix * u_avg + Iy * v_avg + It
        u = u_avg - Ix * numerator / denom
        v = v_avg - Iy * numerator / denom
    return u, v


def hilbert_envelope(X: np.ndarray) -> np.ndarray:
    """Per-channel instantaneous amplitude envelope. X: (n_channels, n_times),
    already band-limited to the mu/beta motor-imagery band (as
    load_iv2a_subject's output already is) — the envelope of a band-limited
    signal is what ERD/ERS topography is conventionally computed from,
    unlike raw voltage which is dominated by frame-to-frame noise at EEG's
    SNR and would not give a physically meaningful flow field.
    """
    return np.abs(hilbert(X, axis=-1))


def windowed_envelope_frames(envelope: np.ndarray, window_samples: int, stride_samples: int) -> np.ndarray:
    """Averages the envelope over sliding windows. envelope: (n_channels,
    n_times) -> frames: (n_windows, n_channels). Windowing (rather than the
    per-sample envelope) is what keeps consecutive topomap frames smooth
    enough for optical flow's brightness-constancy assumption to be
    meaningful, and matches how ERD/ERS topographies are conventionally
    computed (short sliding windows, not single time points).
    """
    n_times = envelope.shape[1]
    starts = list(range(0, n_times - window_samples + 1, stride_samples))
    return np.stack([envelope[:, s : s + window_samples].mean(axis=1) for s in starts])


def trial_flow_sequence(
    X_trial: np.ndarray, positions_2d: np.ndarray, sfreq: float,
    window_s: float = 0.25, stride_s: float = 0.125, grid_size: int = 16, flow_alpha: float = 0.1,
) -> np.ndarray:
    """Full per-trial pipeline: X_trial (n_channels, n_times) -> envelope ->
    windowed frames -> interpolated topomap sequence -> Horn-Schunck flow
    between every consecutive frame pair.

    Returns an (n_windows - 1, 2, grid_size, grid_size) array: for each
    consecutive frame pair, the (u, v) flow field describing how the ERD/ERS
    topography moved across the scalp during that window-to-window step.
    """
    envelope = hilbert_envelope(X_trial)
    window_samples = max(1, int(round(window_s * sfreq)))
    stride_samples = max(1, int(round(stride_s * sfreq)))
    frames = windowed_envelope_frames(envelope, window_samples, stride_samples)

    topomaps = np.stack([interpolate_topomap(frame, positions_2d, grid_size) for frame in frames])

    flows = []
    for i in range(len(topomaps) - 1):
        u, v = horn_schunck_flow(topomaps[i], topomaps[i + 1], alpha=flow_alpha)
        flows.append(np.stack([u, v]))
    return np.stack(flows)


def grid_roi_mask(grid_size: int, radius: float) -> np.ndarray:
    """Boolean (grid_size, grid_size) mask, True within `radius` of the
    center of interpolate_topomap's [-1.2, 1.2]^2 grid — matches
    electrode_positions_2d's radius convention (Cz near 0, rim channels
    near 1), so e.g. radius=0.6 roughly covers phase8's central_17 subset.
    """
    axis = np.linspace(-1.2, 1.2, grid_size)
    grid_x, grid_y = np.meshgrid(axis, axis)
    return (grid_x**2 + grid_y**2) <= radius**2


def flow_summary_features(flow_sequence: np.ndarray, roi_mask: np.ndarray | None = None) -> np.ndarray:
    """Reduces a trial_flow_sequence's (n_steps, 2, grid, grid) output to a
    small, fixed-size feature vector — full per-grid-cell flow is far too
    high-dimensional for the ~200-trial calibration sets this project works
    with (the exact sample-efficiency trap Phase 4c's time-window features
    hit). Returns 6 features: [mean_u, mean_v, mean_speed, std_u, std_v,
    std_speed] pooled over every grid cell (inside roi_mask if given) and
    every consecutive-frame step in the trial.

    Mean captures net directional drift (what stage0_flow_field_eeg_validation
    checked); std captures how much the flow varies across the trial/grid,
    a separate signal even when the net direction averages toward zero.
    """
    u, v = flow_sequence[:, 0], flow_sequence[:, 1]
    if roi_mask is not None:
        u = u[:, roi_mask]
        v = v[:, roi_mask]
    speed = np.sqrt(u**2 + v**2)
    return np.array([u.mean(), v.mean(), speed.mean(), u.std(), v.std(), speed.std()])


def trial_flow_features(
    X_trial: np.ndarray, positions_2d: np.ndarray, sfreq: float,
    window_s: float = 0.25, stride_s: float = 0.125, grid_size: int = 16, flow_alpha: float = 0.1,
    roi_radius: float | None = None,
) -> np.ndarray:
    """trial_flow_sequence followed by flow_summary_features — the
    end-to-end per-trial feature vector for classification."""
    flow_sequence = trial_flow_sequence(X_trial, positions_2d, sfreq, window_s, stride_s, grid_size, flow_alpha)
    roi_mask = grid_roi_mask(grid_size, roi_radius) if roi_radius is not None else None
    return flow_summary_features(flow_sequence, roi_mask)
