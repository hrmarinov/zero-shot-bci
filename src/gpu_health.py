"""GPU / system health guard for long GPU sweeps on the laptop (docs/compute_environment.md).

Call :func:`wait_until_healthy` between training runs. It polls ``nvidia-smi`` (≈50 ms) and
``psutil``, pauses while the GPU is hot or thermally throttled, and raises
:class:`HealthAbort` when waiting would not help (VRAM / RAM exhausted, or a pause longer
than ``max_pause_s``) so the caller can stop cleanly — every sweep in this repo writes its
CSV row by row and resumes, so an abort loses at most one run.
"""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass

PAUSE_AT_C = 80  # lowered from 83 (2026-11-13): fits run ~2 min unchecked and peaked at 83 C
RESUME_BELOW_C = 72
MAX_VRAM_MB = 7200
MIN_FREE_RAM_GB = 1.5

_FIELDS = (
    "temperature.gpu,clocks.sm,power.draw,memory.used,"
    "clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown"
)


class HealthAbort(RuntimeError):
    """Raised when the machine is in a state where continuing would be unsafe."""


@dataclass(frozen=True)
class GpuStatus:
    temp_c: float
    sm_clock_mhz: float
    power_w: float
    mem_used_mb: float
    thermal_throttle: bool


def _num(s: str) -> float:
    try:
        return float(s.strip().split()[0])
    except (ValueError, IndexError):
        return float("nan")


def parse_status(line: str) -> GpuStatus:
    """Parse one ``--format=csv,noheader`` line of the query in :data:`_FIELDS`."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        raise ValueError(f"unexpected nvidia-smi line: {line!r}")
    throttle = any(p.lower() == "active" for p in parts[4:6])
    return GpuStatus(_num(parts[0]), _num(parts[1]), _num(parts[2]), _num(parts[3]), throttle)


def query_gpu() -> GpuStatus | None:
    """Current GPU status, or None if nvidia-smi is unavailable (guard then becomes a no-op)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_FIELDS}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_status(out[0]) if out else None


def free_ram_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / 1024**3


def wait_until_healthy(max_pause_s: float = 1200.0, poll_s: float = 30.0, log=print) -> GpuStatus | None:
    """Block while the GPU is hot/throttled; raise HealthAbort on resource exhaustion."""
    if free_ram_gb() < MIN_FREE_RAM_GB:
        raise HealthAbort(f"free RAM {free_ram_gb():.2f} GB < {MIN_FREE_RAM_GB} GB")
    status = query_gpu()
    if status is None:
        return None
    if status.mem_used_mb > MAX_VRAM_MB:
        raise HealthAbort(f"VRAM used {status.mem_used_mb:.0f} MB > {MAX_VRAM_MB} MB")
    if math.isnan(status.temp_c):  # '[N/A]' reading: no evidence of heat, do not stall
        log("[health] GPU temperature unavailable - continuing without thermal check")
        return status
    if status.temp_c < PAUSE_AT_C and not status.thermal_throttle:
        return status
    waited = 0.0
    while True:
        log(f"[health] GPU {status.temp_c:.0f}C throttle={status.thermal_throttle} - pausing ({waited:.0f}s so far)")
        time.sleep(poll_s)
        waited += poll_s
        status = query_gpu()
        if status is None or (status.temp_c < RESUME_BELOW_C and not status.thermal_throttle):
            return status
        if waited >= max_pause_s:
            raise HealthAbort(f"GPU still {status.temp_c:.0f}C after {waited:.0f}s pause")
