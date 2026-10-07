"""Local machine performance setup: thread pinning, device selection, memory reporting.

Target machine (see docs/compute_environment.md):
  - Intel i9-13900HX: 8 P-cores + 16 E-cores = 24 physical, 32 logical
  - 15.7 GB RAM, ~8 GB typically available, 1 GB page file -> keep each process under ~5 GB
  - RTX 4060 Laptop, 8 GB VRAM, CUDA 12.4, torch 2.6.0+cu124

Thread oversubscription is the default failure mode on this machine: numpy, sklearn and
torch all default to 32 threads, which oversubscribes the 8 fast cores and measures
*slower* than 8 threads, while starving any concurrent process.

Import this module and call `setup_threads()` **before importing numpy**:

    from src.perf import setup_threads
    setup_threads(8)
    import numpy as np

This module therefore imports nothing beyond the standard library at module level.
"""

import gc
import os

#: BLAS/OMP thread count that performs best for this CPU's P-core count.
DEFAULT_THREADS = 8

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def setup_threads(n: int = DEFAULT_THREADS) -> int:
    """Pin BLAS/OMP/torch thread counts to `n`. Returns `n`.

    Sets the environment variables only if the caller has not already set them, so an
    explicit outer override (e.g. `$env:OMP_NUM_THREADS = 16`) still wins. torch reads its
    thread count lazily, so this is effective even when called after torch is imported.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, str(n))
    try:  # torch is optional for the CPU-only parts of this repo
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass
    return n


def get_device(prefer_cuda: bool = True):
    """Return a torch.device: cuda when available (and preferred), else cpu."""
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed numpy and torch (CPU and all CUDA devices) for reproducibility."""
    import numpy as np

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def available_ram_gb() -> float:
    """Best-effort available physical memory in GB (Windows and POSIX)."""
    try:  # Windows
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / 1024**3
    except Exception:  # noqa: BLE001 - reporting must never break a run
        pass
    try:  # POSIX
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (AttributeError, ValueError, OSError):
        return float("nan")


def process_rss_gb() -> float:
    """Resident set size of this process in GB (nan if unavailable)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**3
    except Exception:  # noqa: BLE001
        return float("nan")


def mem_report(label: str = "", collect: bool = False) -> str:
    """Return and print a one-line RSS / available-RAM report."""
    if collect:
        gc.collect()
    line = f"[mem{' ' + label if label else ''}] rss={process_rss_gb():.2f}GB avail={available_ram_gb():.2f}GB"
    print(line, flush=True)
    return line


def guard_memory(min_available_gb: float = 1.0) -> None:
    """Raise before a heavy step if the machine is nearly out of memory.

    Cheap insurance: on this machine the commit limit (16.7 GB with a 1 GB page file) is
    close enough that a large allocation can thrash the whole system rather than fail
    cleanly.
    """
    available = available_ram_gb()
    if available == available and available < min_available_gb:  # nan-safe comparison
        raise MemoryError(
            f"only {available:.2f} GB RAM available (< {min_available_gb} GB); "
            "wait for other jobs to finish rather than starting a heavy step"
        )
