"""Stage 0 gate, part 1: geometric/signal-processing primitives behind
src/adaptation/flow_field.py, checked against known ground truth before any
EEG data touches them — same role stage0_icp_validation.py played for
riemannian_icp.py's Procrustes/ICP primitives.

Two checks:
  1. electrode_positions_2d projects IV-2a's montage into a topologically
     sane 2D disc (Cz near the center, frontal/occipital channels near the
     rim, correct 22-channel count) — not a numerical ground truth, but the
     projection would otherwise silently mislabel motor-cortex geometry.
  2. horn_schunck_flow recovers a known translation velocity applied to a
     synthetic 2D Gaussian blob between two frames — the same "does this
     primitive recover a planted, known transform" pattern the ICP toy test
     used for rotation recovery.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.adaptation.flow_field import electrode_positions_2d, horn_schunck_flow
from src.datasets import IV2A_CHANNELS


def check_electrode_projection() -> None:
    print("--- electrode_positions_2d: topology sanity ---")
    positions = electrode_positions_2d(IV2A_CHANNELS)
    assert positions.shape == (len(IV2A_CHANNELS), 2), f"unexpected shape {positions.shape}"

    radii = {ch: float(np.linalg.norm(positions[i])) for i, ch in enumerate(IV2A_CHANNELS)}
    for ch, r in radii.items():
        print(f"  {ch:>4s}: r={r:.3f}")

    cz_radius = radii["Cz"]
    rim_channels = ["Fz", "P1", "Pz", "P2", "POz"]  # the ones phase8 found unhelpful to drop
    for ch in rim_channels:
        assert radii[ch] > cz_radius, f"{ch} (r={radii[ch]:.3f}) should be farther from center than Cz (r={cz_radius:.3f})"
    assert cz_radius < 0.15, f"Cz should project near the origin, got r={cz_radius:.3f}"

    print(f"  Cz radius={cz_radius:.3f}, all of {rim_channels} farther out - PASSED\n")


def check_horn_schunck_translation_recovery() -> None:
    print("--- horn_schunck_flow: known-translation recovery on a synthetic blob ---")
    grid_size = 32
    axis = np.arange(grid_size, dtype=float)
    gx, gy = np.meshgrid(axis, axis)

    center1 = np.array([14.0, 16.0])
    true_velocity = np.array([2.0, -1.5])  # (dx, dy) in grid cells between frames
    center2 = center1 + true_velocity
    sigma = 4.0

    def gaussian_blob(center: np.ndarray) -> np.ndarray:
        return np.exp(-(((gx - center[0]) ** 2 + (gy - center[1]) ** 2) / (2 * sigma**2)))

    frame1 = gaussian_blob(center1)
    frame2 = gaussian_blob(center2)

    u, v = horn_schunck_flow(frame1, frame2, alpha=0.1, n_iters=200)

    # Horn-Schunck flow is only well-determined where the image has gradient
    # (the aperture problem leaves flat regions filled in by the smoothness
    # term alone, not by evidence) - weight by gradient magnitude so the
    # comparison uses the part of the field optical flow can actually see.
    Ix = 0.5 * (np.gradient(frame1, axis=1) + np.gradient(frame2, axis=1))
    Iy = 0.5 * (np.gradient(frame1, axis=0) + np.gradient(frame2, axis=0))
    weight = Ix**2 + Iy**2
    recovered = np.array([np.sum(u * weight), np.sum(v * weight)]) / np.sum(weight)

    print(f"  true velocity (dx, dy)      = {true_velocity}")
    print(f"  recovered velocity (dx, dy) = {recovered}")

    same_sign = np.all(np.sign(recovered) == np.sign(true_velocity))
    rel_error = np.linalg.norm(recovered - true_velocity) / np.linalg.norm(true_velocity)
    print(f"  same sign on both axes: {same_sign}, relative magnitude error: {rel_error:.3f}")

    assert same_sign, "recovered flow direction disagrees with the planted translation"
    assert rel_error < 0.15, f"recovered flow magnitude off by {rel_error:.1%} - too far from the planted velocity"

    print("  PASSED\n")


def run() -> None:
    check_electrode_projection()
    check_horn_schunck_translation_recovery()
    print("Flow-field primitives validated on synthetic ground truth - "
          "safe to proceed to a synthetic-EEG spatial-drift test.")


if __name__ == "__main__":
    run()
