"""GPU training profiler — Layer 1 (data collection).

    from profiler import GPUCollector, TrainingProfiler, SlurmMonitor
"""
from profiler.src.control_handler import ControlHandler
from profiler.src.gpu_collector import GPUCollector
from profiler.src.metric_writer import get_log_dir, get_log_path, write_metric
from profiler.src.slurm_monitor import SlurmMonitor
from profiler.src.torch_hooks import TrainingProfiler, profiled_train

# The compiled CUDA extension is optional (`make pybind`).
try:
    from profiler.cuda_profiler import CudaProfiler  # built into profiler/ by make pybind
    HAS_CUDA_PROFILER = True
except ImportError:
    CudaProfiler = None
    HAS_CUDA_PROFILER = False

__all__ = [
    "ControlHandler",
    "CudaProfiler",
    "GPUCollector",
    "HAS_CUDA_PROFILER",
    "SlurmMonitor",
    "TrainingProfiler",
    "get_log_dir",
    "get_log_path",
    "profiled_train",
    "write_metric",
]
