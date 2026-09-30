# GPU Training Profiler

[![CI](https://github.com/carloshernandez201/MLTraining_GPU-MetadataTracker/actions/workflows/ci.yml/badge.svg)](https://github.com/carloshernandez201/MLTraining_GPU-MetadataTracker/actions/workflows/ci.yml)

Real-time profiling for GPU training jobs on Slurm clusters (built for UF HiPerGator).
It collects the three things you need to explain a slow or wasteful training run and
writes them to **one time-ordered JSONL stream**:

| Question | Component | Language |
|---|---|---|
| *Is the GPU busy, hot, throttled, out of memory?* | `profiler_daemon` / `GPUCollector` (NVML) | C++ / Python |
| *Why? Where does each training step spend its time?* | `TrainingProfiler` (data / forward / backward / optimizer) | Python |
| *Which CUDA kernels are slow, and how close to peak are they?* | `CudaProfiler` (CUDA Events) | C++/CUDA + pybind11 |
| *What job, node, and resources is this, and are they being used?* | `slurm_monitor` / `SlurmMonitor` | C++ / Python |

A small closed loop sits on top: `feedLLMData.py` summarises the last N seconds of
telemetry, asks an LLM for one tuning action, and writes `control.json`, which the
daemon (power limit, node drain) or the training loop (learning rate, grad clipping,
batch size, stop) applies.

```
             ┌────────────────────┐   ┌─────────────────────┐   ┌──────────────────┐
             │ profiler_daemon    │   │ training process    │   │ slurm_monitor    │
             │ NVML every N sec   │   │ TrainingProfiler    │   │ scontrol / sacct │
             └─────────┬──────────┘   └──────────┬──────────┘   └────────┬─────────┘
                       │  flock'd appends, shared envelope                │
                       ▼                         ▼                        ▼
                    logs/<SLURM_JOB_ID>/metrics.jsonl  ───►  feedLLMData.py (summary → LLM)
                       ▲                                                  │
                       └──── control.json ◄───────────────────────────────┘
```

## Quick start (no GPU required)

```bash
pip install -r requirements.txt          # or just: pip install torch
python examples/full_integration.py --dummy
```

```
gpu collector backend: dummy
training: torch on cpu
epoch 0: 20 steps, 106.9 ms/step, 1146 samples/s, loss 2.2979, bottleneck=backward (data 0% / fwd 40% / bwd 56% / opt 4%)
...
wrote profile_data/unknown/metrics.jsonl and profile_data/unknown/summary.json
```

Without torch installed the demo falls back to a simulated training loop, so the whole
pipeline can be exercised anywhere.

## Building the C++ / CUDA tools

```bash
cd profiler
make                 # bin/profiler_daemon + bin/slurm_monitor  (needs NVML: CUDA toolkit)
make MOCK=1          # same, with a synthetic GPU backend — builds on any Linux box
make test            # C++ unit tests for the Slurm parsers
make cuda            # bin/cuda_profiler_demo                   (needs nvcc)
make pybind          # profiler/cuda_profiler*.so Python module (needs nvcc + pybind11)
```

On HiPerGator: `module load cuda gcc && make`.

## Usage

**GPU daemon** — run alongside training (e.g. backgrounded in your sbatch script):

```bash
./bin/profiler_daemon --interval 1 &          # logs/<job>/metrics.jsonl, stops on SIGINT/SIGTERM
./bin/profiler_daemon -i 0.5 -d 10 --stdout   # 10 s to stdout
```

**Slurm context**:

```bash
./bin/slurm_monitor            # current job ($SLURM_JOB_ID), else all of your jobs
./bin/slurm_monitor 12345      # job info + efficiency (walltime, memory, GPU-seconds)
```

**Instrumenting a training loop**:

```python
from profiler import GPUCollector, SlurmMonitor, TrainingProfiler, ControlHandler

SlurmMonitor().snapshot()
tp = TrainingProfiler(model=model)
ctl = ControlHandler(model=model, dataloader=loader, optimizer=opt)

with GPUCollector(interval=1.0):          # background thread; or use profiler_daemon
    for epoch in range(epochs):
        tp.start_epoch(epoch)
        for x, y in loader:
            tp.step_start()
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            tp.mark_data_loaded()
            loss = criterion(model(x), y)
            tp.mark_forward_done()
            loss.backward()
            tp.mark_backward_done()
            opt.step(); opt.zero_grad()
            tp.step_end(loss=loss, batch_size=x.size(0))
            ctl.check_and_apply()             # picks up control.json commands
        print(tp.end_epoch())                 # per-epoch summary incl. bottleneck phase
```

Timing calls `torch.cuda.synchronize()` at each marker so GPU work is attributed to the
right phase (CUDA launches are asynchronous), and uses `time.perf_counter()`.

**Kernel-level timing** (after `make pybind`):

```python
from profiler import CudaProfiler
prof = CudaProfiler()
prof.start("matmul"); c = a @ b; prof.stop("matmul", flops=2 * n**3)
print(prof.to_json())   # mean/p50/p95/p99 ms, GFLOP/s, GB/s, launch overhead
```

**LLM tuning loop**:

```bash
python -m profiler.src.feedLLMData --interval 60 --dry-run   # show the summary sent to the model
OPENAI_API_KEY=... python -m profiler.src.feedLLMData --interval 60
```

## Record format

Every line in `metrics.jsonl` has the same envelope:

```json
{"timestamp": 1790811204.859, "job_id": "12345", "host": "c1000a-s17", "pid": 456,
 "source": "gpu", "data": { ... }}
```

| `source` | `data` contents |
|---|---|
| `gpu` | `devices[]` (util, mem util, mem used/total/free, temp, power/limit, SM/mem clocks, PCIe tx/rx, process count) + `avg_gpu_util`, `avg_temp_c`, `avg_mem_usage_ratio`, `avg_clock_throttle_ratio`, `total_power_w` |
| `pytorch` | `step`, `epoch`, `loss`, `step_time_ms`, `{data,forward,backward,optimizer}_ms` / `_frac`, `bottleneck`, `throughput_samples_per_s`, `grad_norm`, GPU memory |
| `pytorch_epoch` | step time mean/p50/p95, phase fractions, bottleneck, throughput, loss and grad-norm stats |
| `slurm` | `type: job_info` (job, user, partition, nodes, CPUs, memory, GPUs + type, TRES, limits) or `type: efficiency` (walltime / memory efficiency, GPU-seconds) |
| `control` | a training command that was applied |

## Control commands

Write `logs/<job>/control.json` (the LLM step does this for you):

| Action | Handled by | Fields |
|---|---|---|
| `set_lr` / `scale_lr` | training loop | `lr` / `factor` |
| `set_grad_clip` | training loop | `max_norm` |
| `set_batch_size` | training loop (next epoch) | `batch_size` |
| `stop_training` | training loop | — |
| `reduce_power_limit` | daemon (needs root) | `gpu_index`, `watts` |
| `drain_node` | daemon (needs Slurm admin) | `reason` |

## Layout

```
profiler/
  main.cpp                 profiler_daemon entry point (signals, interval loop)
  headers/                 collector.h, slurm_monitor.h, metric_writer.h, system_executor.h, cuda_profiler.hpp
  src/collector.cpp        NVML sampling (+ mock backend)
  src/slurm_monitor.cpp    scontrol / squeue / sacct parsing, standalone CLI
  src/cuda_profiler.cpp    CUDA Event timer, demo kernels, pybind11 module
  src/torch_hooks.py       TrainingProfiler + profiled_train decorator
  src/gpu_collector.py     Python GPU sampler (pynvml → nvidia-smi → dummy)
  src/slurm_monitor.py     Python Slurm context
  src/control_handler.py   applies training-side control commands
  src/feedLLMData.py       telemetry summary → LLM → control.json
  src/metric_writer.py     shared JSONL writer
  tests/                   C++ parser tests
examples/full_integration.py
tests/                     pytest suite
docs/ASSIGNMENT.md         original Layer 1 spec
```

## Tests

```bash
pytest -q tests
cd profiler && make MOCK=1 test
```

CI builds the mock daemon and runs a smoke test, compiles the real NVML daemon and the
CUDA profiler inside an `nvidia/cuda` container, and runs the Python tests plus the dummy
integration run.
