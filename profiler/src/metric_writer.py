"""Shared JSONL writer. Every component (C++ daemon included) appends records with
the same envelope to logs/<SLURM_JOB_ID>/metrics.jsonl so they can be merged by time.
"""
import json
import os
import socket
import time

try:
    import fcntl  # POSIX only; on Windows we fall back to unlocked appends
except ImportError:  # pragma: no cover
    fcntl = None


def get_job_id() -> str:
    return os.environ.get("SLURM_JOB_ID", "unknown")


def get_log_dir() -> str:
    base = os.environ.get("PROFILER_LOG_DIR", "logs")
    d = os.path.join(base, get_job_id())
    os.makedirs(d, exist_ok=True)
    return d


def get_log_path() -> str:
    return os.path.join(get_log_dir(), "metrics.jsonl")


def make_record(source: str, data: dict) -> dict:
    return {
        "timestamp": time.time(),
        "job_id": get_job_id(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "source": source,
        "data": data,
    }


def write_metric(source: str, data: dict, path: str = None) -> dict:
    record = make_record(source, data)
    line = json.dumps(record)
    path = path or get_log_path()
    with open(path, "a") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line + "\n")
        finally:
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_UN)
    return record
