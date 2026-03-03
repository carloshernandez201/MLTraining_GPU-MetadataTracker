import json
import time
import os
import socket
import fcntl

def get_log_path():
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    d = f"logs/{job_id}"
    os.makedirs(d, exist_ok=True)
    return f"{d}/metrics.jsonl"

def write_metric(source: str, data: dict):
    line = json.dumps({
        "timestamp": time.time(),
        "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "source": source,
        "data": data,
    })
    path = get_log_path()
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)