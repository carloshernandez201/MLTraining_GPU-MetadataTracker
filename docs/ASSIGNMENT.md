# Assignment: GPU Training Profiler — Layer 1 (Data Collection)

## Goal

Build a profiling toolkit that monitors GPU training jobs in real time. The system collects hardware-level metrics (utilization, memory, power, thermals), instruments PyTorch training loops (step timing, throughput, bottleneck breakdown), tracks CUDA kernel execution at the microsecond level, and pulls Slurm job metadata to tie everything together. All data is output as JSON for downstream analysis.

This is the foundation layer. Everything you build in Layers 2–4 (storage, bottleneck detection, auto-tuning) consumes data produced here.

---

## File Breakdown

### 1. `collector.cpp`
**What it does:** Daemon that polls GPU hardware metrics at a configurable interval using the NVML C API. Outputs JSONL (one JSON object per sample) to stdout or a file. Handles graceful shutdown via SIGINT/SIGTERM.

**Metrics collected:** GPU utilization %, memory utilization %, memory used/total/free, temperature, power draw vs power limit, SM and memory clock speeds, PCIe throughput, running process count.

**Key concepts to learn:**
- NVML (NVIDIA Management Library) — the C API that `nvidia-smi` itself is built on
- Signal handling in C++ (`std::signal`, `std::atomic<bool>`)
- JSONL format (newline-delimited JSON for streaming data)

**Resources:**
- NVML API Reference: https://docs.nvidia.com/deploy/nvml-api/
- NVML Developer Guide: https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html
- `nvidia-smi` source behavior (it's literally a thin wrapper over NVML)
- Signal handling: https://en.cppreference.com/w/cpp/utility/program/signal

**Build:** `g++ -O2 -std=c++17 collector.cpp -o collector -lnvidia-ml -lpthread`

---

### 2. `torch_hooks.py`
**What it does:** Instruments a PyTorch training loop to measure where time is spent *within each training step*. You place markers (`step_start`, `mark_data_loaded`, `mark_forward_done`, `mark_backward_done`, `step_end`) around your training code. It computes: data loading time, forward pass time, backward pass time, optimizer time, throughput (samples/sec), gradient norms, and GPU memory usage. Outputs per-step JSONL and per-epoch summaries.

**Why this matters:** `collector.cpp` tells you "GPU is at 40% utilization." This file tells you *why* — maybe 60% of each step is waiting on the data loader.

**Key concepts to learn:**
- `torch.cuda.synchronize()` — why you need it for accurate GPU timing
- `torch.profiler` — PyTorch's built-in kernel-level profiler (optional integration)
- Gradient norm computation — useful for detecting training instability
- `time.perf_counter()` vs `time.time()` — why perf_counter is better for benchmarking

**Resources:**
- PyTorch Profiler tutorial: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- `torch.profiler` docs: https://pytorch.org/docs/stable/profiler.html
- Understanding CUDA synchronization: https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
- Why gradient norms matter: https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem

**Build:** Pure Python, no compilation. `pip install torch`

---

### 3. `cuda_profiler.hpp` + `cuda_profiler.cpp`
**What it does:** Low-level CUDA kernel timer. Wraps CUDA Events to measure individual kernel execution times with sub-microsecond precision. Tracks per-kernel statistics (count, mean, min, max, std, percentiles), computes GFLOP/s and memory bandwidth, and measures kernel launch overhead (the gap between CPU-side wall time and GPU-side execution time). Exports results to JSON. Can be built as a standalone demo or as a Python module via pybind11.

**Why this matters:** `torch_hooks.py` tells you "the forward pass took 50ms." This tells you *which CUDA kernels* inside that forward pass are slow — maybe one matmul kernel is running at 10% of peak bandwidth.

**Key concepts to learn:**
- CUDA Events (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`) — the standard way to time GPU work
- Roofline model — relating kernel performance to peak compute (FLOP/s) and memory bandwidth (GB/s)
- Kernel launch overhead — the CPU-side cost of dispatching work to the GPU
- pybind11 — exposing C++ classes to Python

**Resources:**
- CUDA Events: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
- CUDA Best Practices Guide (profiling section): https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#profiling
- Roofline model explainer: https://en.wikipedia.org/wiki/Roofline_model
- pybind11 docs: https://pybind11.readthedocs.io/en/stable/
- NVIDIA Nsight Systems (visual profiler, good for validating your results): https://developer.nvidia.com/nsight-systems

**Build:**
```
# Standalone demo
nvcc -O2 -std=c++17 -x cu cuda_profiler.cpp -o cuda_profiler_demo

# Python module
nvcc -O2 -std=c++17 -x cu -shared -Xcompiler -fPIC -DBUILD_PYBIND \
  $(python3 -m pybind11 --includes) \
  cuda_profiler.cpp -o cuda_profiler$(python3-config --extension-suffix)
```

---

### 4. `slurm_monitor.cpp`
**What it does:** Queries Slurm for job metadata by parsing output from `scontrol show job`, `sacct`, and `squeue`. Extracts: job ID, name, user, partition, node list, GPU/CPU/memory allocation, time limit, elapsed time, GPU type, and TRES (trackable resources). Also computes resource efficiency for completed jobs (memory efficiency, wall time efficiency, GPU-seconds billed). Doubles as a header-only library (`#define SLURM_MONITOR_HEADER_ONLY`) so you can use it from other C++ files.

**Why this matters:** Profiling data is useless without context. This ties metrics to "job 12345 on node gpu001, 2x A100s, 32 CPUs, 64GB RAM, ran for 4 hours."

**Key concepts to learn:**
- Slurm job scheduling — `scontrol`, `sacct`, `squeue` commands
- TRES (Trackable Resources) — how Slurm tracks GPU allocation
- Resource efficiency analysis — are you actually using what you requested?
- `popen()` — running subprocesses from C++ and capturing output

**Resources:**
- Slurm scontrol docs: https://slurm.schedmd.com/scontrol.html
- Slurm sacct docs: https://slurm.schedmd.com/sacct.html
- Slurm TRES: https://slurm.schedmd.com/tres.html
- HiPerGator Slurm guide: https://help.rc.ufl.edu/doc/SLURM_Commands
- popen: https://man7.org/linux/man-pages/man3/popen.3.html


**Build:** `g++ -O2 -std=c++17 slurm_monitor.cpp -o slurm_monitor`

---

### 5. `__init__.py`
**What it does:** Python package init. Imports and re-exports all components so you can do `from profiler import GPUCollector, TrainingProfiler, SlurmMonitor`. Tries to import the C++ CUDA profiler module, falls back gracefully if not compiled.

**Not much to learn here** — it's just wiring.

---

### 6. `examples/full_integration.py`
**What it does:** Shows how all components work together in a real training loop. Starts background GPU collection in a thread, instruments a PyTorch training loop with `TrainingProfiler`, grabs Slurm context, and saves everything to a `profile_data/` directory. Includes a dummy mode that works without a GPU for testing the profiler infrastructure.

**Use this as your test harness** when developing. Get it running with the dummy mode first, then swap in real training code on HiPerGator.

---

## Suggested Build Order

1. **`collector.cpp`** — get NVML metrics printing to console on HiPerGator. Validate against `nvidia-smi` output.
2. **`slurm_monitor.cpp`** — test inside a Slurm job, verify it picks up job metadata correctly.
3. **`torch_hooks.py`** — instrument a simple training loop (the example in `full_integration.py`), look at the JSONL output.
4. **`cuda_profiler.cpp`** — compile the demo, run it, compare kernel times against `nsys profile` output.
5. **`full_integration.py`** — wire everything together, run a real training job on HiPerGator.

Each file is independently testable. You don't need file N+1 to validate file N.
