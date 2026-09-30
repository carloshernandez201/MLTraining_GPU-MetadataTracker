"""Python-side Slurm context (mirrors src/slurm_monitor.cpp).

Uses `scontrol show job` when available and falls back to the SLURM_* environment
variables Slurm exports into every job, so it also works on compute nodes where
the client commands are restricted.
"""
import os
import re
import shutil
import subprocess

from profiler.src.metric_writer import write_metric

_JOB_ID_RE = re.compile(r"^[0-9_+.]+$")


def parse_scontrol(text: str) -> dict:
    kv = {}
    for token in text.split():
        key, sep, val = token.partition("=")
        if sep and key:
            kv[key] = val
    return kv


def parse_mem_mb(s: str) -> float:
    s = (s or "").strip()
    if s and s[-1] in "nc":
        s = s[:-1]
    if not s:
        return 0.0
    unit = "M"
    if s[-1].isalpha():
        unit, s = s[-1].upper(), s[:-1]
    try:
        v = float(s)
    except ValueError:
        return 0.0
    return v * {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2}.get(unit, 1)


def parse_gpus_from_tres(tres: str):
    count, gpu_type = 0, ""
    for item in (tres or "").split(","):
        key, _, val = item.partition("=")
        if key == "gres/gpu":
            count = max(count, int(val or 0))
        elif key.startswith("gres/gpu:"):
            gpu_type = key[len("gres/gpu:"):]
            count = max(count, int(val or 0))
    return count, gpu_type


def job_info_from_scontrol(text: str) -> dict:
    kv = parse_scontrol(text)
    tres = kv.get("AllocTRES") or kv.get("ReqTRES", "")
    mem_mb = 0.0
    for item in tres.split(","):
        if item.startswith("mem="):
            mem_mb = parse_mem_mb(item[4:])
    if not mem_mb:
        mem_mb = parse_mem_mb(kv.get("MinMemoryNode", ""))
    num_gpus, gpu_type = parse_gpus_from_tres(tres)
    return {
        "type": "job_info",
        "slurm_job_id": kv.get("JobId", ""),
        "name": kv.get("JobName", ""),
        "user": kv.get("UserId", "").split("(")[0],
        "state": kv.get("JobState", ""),
        "partition": kv.get("Partition", ""),
        "node_list": kv.get("NodeList", ""),
        "num_nodes": int(kv.get("NumNodes", "0").split("-")[0] or 0),
        "num_cpus": int(kv.get("NumCPUs", "0") or 0),
        "mem_mb": mem_mb,
        "num_gpus": num_gpus,
        "gpu_type": gpu_type,
        "time_limit": kv.get("TimeLimit", ""),
        "elapsed": kv.get("RunTime", ""),
        "tres": tres,
        "work_dir": kv.get("WorkDir", ""),
    }


def job_info_from_env(env=None) -> dict:
    env = os.environ if env is None else env
    gpus = env.get("SLURM_GPUS_ON_NODE") or env.get("SLURM_GPUS") or ""
    return {
        "type": "job_info",
        "slurm_job_id": env.get("SLURM_JOB_ID", ""),
        "name": env.get("SLURM_JOB_NAME", ""),
        "user": env.get("SLURM_JOB_USER", env.get("USER", "")),
        "state": "RUNNING" if env.get("SLURM_JOB_ID") else "",
        "partition": env.get("SLURM_JOB_PARTITION", ""),
        "node_list": env.get("SLURM_JOB_NODELIST", ""),
        "num_nodes": int(env.get("SLURM_JOB_NUM_NODES", "0") or 0),
        "num_cpus": int(env.get("SLURM_CPUS_ON_NODE", "0") or 0),
        "mem_mb": parse_mem_mb(env.get("SLURM_MEM_PER_NODE", "")),
        "num_gpus": int(gpus) if gpus.isdigit() else 0,
        "gpu_type": "",
        "time_limit": "",
        "elapsed": "",
        "tres": "",
        "work_dir": env.get("SLURM_SUBMIT_DIR", ""),
        "source": "env",
    }


class SlurmMonitor:
    def __init__(self, job_id: str = None):
        self.job_id = job_id or os.environ.get("SLURM_JOB_ID")

    @property
    def in_slurm_job(self) -> bool:
        return bool(self.job_id)

    def job_info(self) -> dict:
        if not self.job_id:
            return {}
        if _JOB_ID_RE.match(self.job_id) and shutil.which("scontrol"):
            try:
                out = subprocess.run(
                    ["scontrol", "show", "job", self.job_id],
                    capture_output=True, text=True, timeout=10,
                ).stdout
                if "JobId=" in out:
                    return job_info_from_scontrol(out)
            except (subprocess.SubprocessError, OSError):
                pass
        return job_info_from_env()

    def snapshot(self) -> dict:
        """Write the current job's metadata to the metrics log and return it."""
        info = self.job_info()
        if info:
            write_metric("slurm", info)
        return info
