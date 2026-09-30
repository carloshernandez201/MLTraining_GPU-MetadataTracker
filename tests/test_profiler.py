import json
import os
import time

import pytest


@pytest.fixture(autouse=True)
def log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILER_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    return tmp_path / "999"


def read_records(log_dir):
    with open(log_dir / "metrics.jsonl") as f:
        return [json.loads(line) for line in f]


def test_write_metric_envelope(log_dir):
    from profiler import write_metric

    write_metric("test", {"x": 1})
    (rec,) = read_records(log_dir)
    assert rec["source"] == "test"
    assert rec["job_id"] == "999"
    assert rec["data"] == {"x": 1}
    assert {"timestamp", "host", "pid"} <= rec.keys()


def test_gpu_collector_dummy(log_dir):
    from profiler import GPUCollector

    with GPUCollector(interval=0.05, backend="dummy") as c:
        time.sleep(0.3)
    assert c.samples >= 3
    recs = [r for r in read_records(log_dir) if r["source"] == "gpu"]
    d = recs[0]["data"]
    assert d["gpu_count"] == 1
    assert 0 <= d["avg_gpu_util"] <= 100
    assert d["devices"][0]["mem_total"] > 0


def test_training_profiler_phases(log_dir):
    from profiler import TrainingProfiler

    tp = TrainingProfiler(batch_size=32)
    tp.start_epoch(0)
    for _ in range(3):
        tp.step_start()
        time.sleep(0.001)
        tp.mark_data_loaded()
        time.sleep(0.001)
        tp.mark_forward_done()
        time.sleep(0.02)  # backward dominates
        tp.mark_backward_done()
        rec = tp.step_end(loss=1.0)
    assert rec["bottleneck"] == "backward"
    assert abs(sum(rec[f"{p}_frac"] for p in ("data", "forward", "backward", "optimizer")) - 1) < 1e-6
    assert rec["throughput_samples_per_s"] > 0

    summary = tp.end_epoch()
    assert summary["steps"] == 3
    assert summary["bottleneck"] == "backward"
    sources = [r["source"] for r in read_records(log_dir)]
    assert sources.count("pytorch") == 3 and sources.count("pytorch_epoch") == 1


def test_control_handler_ignores_system_actions(log_dir):
    from profiler import ControlHandler

    h = ControlHandler()
    path = h.control_path
    with open(path, "w") as f:
        json.dump({"action": "reduce_power_limit", "gpu_index": 0, "watts": 250}, f)
    assert h.check_and_apply() is None
    assert os.path.exists(path)  # left for the C++ daemon

    with open(path, "w") as f:
        json.dump({"action": "set_grad_clip", "max_norm": 1.5}, f)
    assert h.check_and_apply()["action"] == "set_grad_clip"
    assert h.grad_clip == 1.5
    assert not os.path.exists(path)


def test_slurm_parsers():
    from profiler.src.slurm_monitor import job_info_from_env, job_info_from_scontrol, parse_mem_mb

    text = ("JobId=42 JobName=train UserId=bob(1) JobState=RUNNING Partition=gpu NodeList=n1 "
            "NumNodes=1 NumCPUs=8 TimeLimit=02:00:00 RunTime=00:10:00 "
            "AllocTRES=cpu=8,mem=32G,node=1,gres/gpu:a100=1,gres/gpu=1")
    info = job_info_from_scontrol(text)
    assert info["user"] == "bob" and info["num_gpus"] == 1 and info["gpu_type"] == "a100"
    assert info["mem_mb"] == 32768
    assert parse_mem_mb("512K") == 0.5

    env = job_info_from_env({"SLURM_JOB_ID": "7", "SLURM_GPUS_ON_NODE": "2", "SLURM_JOB_PARTITION": "gpu"})
    assert env["num_gpus"] == 2 and env["partition"] == "gpu"


def test_llm_summary_window(log_dir):
    from profiler import write_metric
    from profiler.src.feedLLMData import iter_lines_reverse, summarize_window

    for i, loss in enumerate([2.0, 1.5, 1.0]):
        write_metric("pytorch", {"step": i, "loss": loss, "bottleneck": "data", "data_frac": 0.6})
    write_metric("gpu", {"avg_gpu_util": 40, "devices": [{"index": 0}]})

    path = str(log_dir / "metrics.jsonl")
    assert len(list(iter_lines_reverse(path, chunk_size=16))) == 4  # chunk boundaries handled
    s = summarize_window(path, interval=60)
    assert s["pytorch"]["loss_trend"] == "decreasing"
    assert s["pytorch"]["most_common_bottleneck"] == "data"
    assert "devices" not in s["gpu"][0]
    assert summarize_window(path, interval=60, now=time.time() + 3600)["pytorch"] == {}


def test_training_profiler_with_torch(log_dir):
    torch = pytest.importorskip("torch")
    from profiler import TrainingProfiler

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    tp = TrainingProfiler(model=model, batch_size=8)
    x, y = torch.randn(8, 4), torch.randint(0, 2, (8,))
    tp.step_start()
    tp.mark_data_loaded()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    tp.mark_forward_done()
    loss.backward()
    tp.mark_backward_done()
    opt.step()
    rec = tp.step_end(loss=loss)
    assert isinstance(rec["loss"], float)
    assert rec["grad_norm"] > 0
