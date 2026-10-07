# Compute environment — the machine this project runs on, and how to write code for it

Measured 2026-10-07 on the development machine. **Every script in this repository should
follow the rules below**; `src/perf.py` implements them.

## Hardware

| Component | Spec | Consequence for code |
|---|---|---|
| CPU | Intel i9-13900HX — **8 P-cores + 16 E-cores** (24 physical, 32 logical), 36 MB L3 | Two core classes with very different throughput. Default 32-thread BLAS/OMP oversubscribes the 8 fast cores and *slows numeric work down*. Pin threads. |
| RAM | **15.7 GB total**, ~8.1 GB typically available, **1 GB page file**, commit limit 16.7 GB | **The binding constraint.** Committed memory is already ~10.5 GB with a normal desktop session. Budget **≤ 5 GB per Python process**. Exceeding it does not fail cleanly — it thrashes. |
| GPU | NVIDIA RTX 4060 Laptop, **8 GB VRAM**, Ada (compute capability 8.9), driver 576.57 | Real and usable, but 8 GB is small. Use for torch models; keep batch sizes modest; `float32` only (no `float64`, no `float16` training by default). |
| CUDA / torch | CUDA 12.4, `torch==2.6.0+cu124`, `torch.cuda.is_available() == True` | `torch.get_num_threads()` returns 24 by default — set it explicitly. |
| Python | 3.12.3 in `.venv` | — |
| Disk | `C:` 510 GB free (checkout + `data/`), `D:` 690 GB free | Datasets live under `C:\...\calibIllit\data\` (26 GB); no space pressure. |

## Datasets already on disk (do not re-download)

| Dataset | On disk | Detail |
|---|---|---|
| BCI Competition IV-2a (`BNCI2014_001`) | 1.17 GB | 18 `.mat` = 9 subjects × {T, E} sessions |
| BCI Competition IV-2b (`BNCI2014_004`) | (same dir) | 18 `.mat` = 9 subjects × {T, E}; 5 sessions inside |
| OpenBMI / Lee2019 (`Lee2019_MI`) | 22.5 GB | **20 of 54** subjects |
| PhysioNet MI (`PhysionetMI`) | 1.55 GB | 109 subjects (108 usable — subject 88 excluded) |
| Artifact-cleaned IV-2a | 1.01 GB | `data/cleaned/`, 18 `.fif` (subject × session) |

## The rules

### 1. Pin threads, always — first thing in the file

```python
from src.perf import setup_threads
setup_threads(8)          # before importing numpy
import numpy as np
```

Defaults to 8. On this CPU that is the sweet spot for BLAS-heavy work (CSP covariances,
eigendecompositions, PCA). Use `setup_threads(4)` when *several* processes run
concurrently, and `setup_threads(16)` only for a single genuinely CPU-bound job with no
concurrency.

**Never let numpy/sklearn/torch use all 32 threads** — measured to be slower than 8 on this
part-P-core CPU, and it starves any concurrent process.

### 2. Memory discipline

- **Process one subject at a time.** Never `[loader(s) for s in subjects]` for the big
  datasets (OpenBMI is ~600 MB/subject raw; 20 of them will not fit).
- IV-2a per subject is small (`288 × 22 × 500 float64` ≈ 25 MB per session) — all 9 subjects
  at once is fine (~450 MB).
- Use `float32` for any array expected to exceed ~500 MB, and for **all** torch tensors.
- `del` large intermediates and `gc.collect()` between subjects in a loop.
- If a script needs to hold a big intermediate across subjects, write it to
  `data/cache/*.npz` instead and reload per subject.
- `src.perf.mem_report()` prints RSS + available RAM; call it around anything heavy while
  developing.

### 3. GPU usage

```python
from src.perf import get_device
device = get_device()     # cuda if available, else cpu
```

- Worth it for: wavelet calibration (Phase 4b/4c), prototypical network (Phase 6),
  geometric correction (Module A), any new torch model.
- Not worth it for CSP/sklearn/pyriemann — transfer overhead dominates; keep those on CPU.
- 8 GB VRAM budget: keep tensors and batches small (the existing models here are tiny
  MLPs/Morlet banks — batch 32–256 on `22 × 500` trials is comfortable).
- `torch.manual_seed` **and** `torch.cuda.manual_seed_all` for determinism.

### 4. Concurrency policy (multi-agent work)

The machine cannot support several heavy jobs at once. **One dataset-loading or
model-training script at a time.** These are cheap and can overlap freely: metadata
inspection, a single-subject smoke run, plotting from already-written CSVs, and markdown
writing. When in doubt, ask before starting something heavy — a starved machine produces
`MemoryError` and, worse, silently paged-to-disk runs that look like slowness.

### 5. Determinism

Every script seeds numpy (`np.random.default_rng(RANDOM_STATE)`), sklearn (via
`random_state`), and torch. Results are written to `results/` as CSV with a stable filename.
A rerun must reproduce the committed numbers.

## Quick sanity check

```powershell
.venv\Scripts\python experiments\stage0_icp_validation.py     # ~3 s, assertion-gated
.venv\Scripts\python -m pytest -q                             # ~10 s, 40 tests
```
