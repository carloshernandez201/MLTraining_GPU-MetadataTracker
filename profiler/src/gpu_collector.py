"""Background GPU sampler for use from Python (same record schema as the C++ daemon).

Backends, tried in order with backend="auto":
  nvml        -> pynvml / nvidia-ml-py
  nvidia-smi  -> parses `nvidia-smi --query-gpu` CSV
  dummy       -> synthetic numbers, for laptops / CI without a GPU
"""
import math
import shutil
import subprocess
import threading
import time

from profiler.src.metric_writer import write_metric


def summarize(devices: list) -> dict:
    """Build the "data" object: per-device list + cross-GPU averages."""
    n = len(devices)

    def avg(key):
        return sum(d[key] for d in devices) / n if n else 0.0

    mem_ratio = [d["mem_used"] / d["mem_total"] if d["mem_total"] else 0.0 for d in devices]
    clock_ratio = [
        d["clock_sm_mhz"] / d["clock_sm_max_mhz"] if d["clock_sm_max_mhz"] else 1.0 for d in devices
    ]
    return {
        "gpu_count": n,
        "devices": devices,
        "avg_gpu_util": avg("gpu_util"),
        "avg_temp_c": avg("temp_c"),
        "avg_mem_usage_ratio": sum(mem_ratio) / n if n else 0.0,
        "avg_clock_throttle_ratio": sum(clock_ratio) / n if n else 0.0,
        "total_power_w": sum(d["power_w"] for d in devices),
    }


class _NvmlBackend:
    name = "nvml"

    def __init__(self):
        import pynvml  # provided by the nvidia-ml-py package

        self.nv = pynvml
        pynvml.nvmlInit()

    def sample(self) -> list:
        nv = self.nv
        out = []
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            util = nv.nvmlDeviceGetUtilizationRates(h)
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            name = nv.nvmlDeviceGetName(h)

            def safe(fn, default=0):
                try:
                    return fn()
                except nv.NVMLError:
                    return default

            out.append({
                "index": i,
                "name": name.decode() if isinstance(name, bytes) else name,
                "gpu_util": util.gpu,
                "mem_util": util.memory,
                "mem_used": mem.used,
                "mem_total": mem.total,
                "mem_free": mem.free,
                "temp_c": safe(lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
                "power_w": safe(lambda: nv.nvmlDeviceGetPowerUsage(h)) / 1000.0,
                "power_limit_w": safe(lambda: nv.nvmlDeviceGetPowerManagementLimit(h)) / 1000.0,
                "clock_sm_mhz": safe(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
                "clock_sm_max_mhz": safe(lambda: nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM)),
                "clock_mem_mhz": safe(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)),
                "pcie_tx_kbps": safe(lambda: nv.nvmlDeviceGetPcieThroughput(h, nv.NVML_PCIE_UTIL_TX_BYTES)),
                "pcie_rx_kbps": safe(lambda: nv.nvmlDeviceGetPcieThroughput(h, nv.NVML_PCIE_UTIL_RX_BYTES)),
                "process_count": len(safe(lambda: nv.nvmlDeviceGetComputeRunningProcesses(h), [])),
            })
        return out

    def close(self):
        self.nv.nvmlShutdown()


class _SmiBackend:
    name = "nvidia-smi"
    FIELDS = [
        "index", "name", "utilization.gpu", "utilization.memory", "memory.used", "memory.total",
        "memory.free", "temperature.gpu", "power.draw", "power.limit", "clocks.sm",
        "clocks.max.sm", "clocks.mem",
    ]

    def __init__(self):
        if not shutil.which("nvidia-smi"):
            raise RuntimeError("nvidia-smi not found")
        self.sample()  # fail fast if it doesn't work

    @staticmethod
    def _num(v):
        try:
            return float(v)
        except ValueError:  # "[N/A]", "[Not Supported]"
            return 0.0

    def sample(self) -> list:
        text = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(self.FIELDS), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        out = []
        for line in text.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < len(self.FIELDS):
                continue
            mib = 1024 * 1024
            out.append({
                "index": int(p[0]),
                "name": p[1],
                "gpu_util": self._num(p[2]),
                "mem_util": self._num(p[3]),
                "mem_used": int(self._num(p[4]) * mib),
                "mem_total": int(self._num(p[5]) * mib),
                "mem_free": int(self._num(p[6]) * mib),
                "temp_c": self._num(p[7]),
                "power_w": self._num(p[8]),
                "power_limit_w": self._num(p[9]),
                "clock_sm_mhz": self._num(p[10]),
                "clock_sm_max_mhz": self._num(p[11]),
                "clock_mem_mhz": self._num(p[12]),
                "pcie_tx_kbps": 0,
                "pcie_rx_kbps": 0,
                "process_count": 0,
            })
        return out

    def close(self):
        pass


class _DummyBackend:
    name = "dummy"

    def __init__(self, gpus=1):
        self.gpus = gpus

    def sample(self) -> list:
        t = time.time()
        out = []
        for i in range(self.gpus):
            util = 70 + 25 * math.sin(t / 5.0 + i)
            total = 80 * 1024 ** 3
            used = int(total * 0.6)
            out.append({
                "index": i, "name": "Dummy GPU", "gpu_util": round(util, 1), "mem_util": round(util / 2, 1),
                "mem_used": used, "mem_total": total, "mem_free": total - used,
                "temp_c": 55 + util / 5, "power_w": 100 + 2.8 * util, "power_limit_w": 400.0,
                "clock_sm_mhz": 1200 + util, "clock_sm_max_mhz": 1410, "clock_mem_mhz": 1593,
                "pcie_tx_kbps": int(1000 * util), "pcie_rx_kbps": int(2000 * util), "process_count": 1,
            })
        return out

    def close(self):
        pass


def _make_backend(backend: str):
    if backend == "dummy":
        return _DummyBackend()
    candidates = {"nvml": [_NvmlBackend], "nvidia-smi": [_SmiBackend], "auto": [_NvmlBackend, _SmiBackend]}
    if backend not in candidates:
        raise ValueError(f"unknown backend {backend!r}")
    errors = []
    for cls in candidates[backend]:
        try:
            return cls()
        except Exception as e:  # missing library, no driver, ...
            errors.append(f"{cls.name}: {e}")
    if backend == "auto":
        return _DummyBackend()
    raise RuntimeError("; ".join(errors))


class GPUCollector:
    """Samples GPU metrics every `interval` seconds on a daemon thread.

        with GPUCollector(interval=1.0):
            train()
    """

    def __init__(self, interval: float = 1.0, backend: str = "auto", log_path: str = None):
        self.interval = interval
        self.backend = _make_backend(backend)
        self.log_path = log_path
        self.samples = 0
        self._stop = threading.Event()
        self._thread = None

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def sample_once(self) -> dict:
        data = summarize(self.backend.sample())
        data["backend"] = self.backend.name
        write_metric("gpu", data, path=self.log_path)
        self.samples += 1
        return data

    def _run(self):
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception as e:  # keep training alive if sampling hiccups
                print(f"[GPUCollector] sample failed: {e}")
            next_tick += self.interval
            self._stop.wait(max(0.0, next_tick - time.monotonic()))

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="GPUCollector", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self.backend.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
