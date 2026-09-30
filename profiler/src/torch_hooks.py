"""Per-step timing for PyTorch training loops.

Two ways to use it:

1. Explicit markers (fine-grained breakdown):

    tp = TrainingProfiler(model=model)
    for epoch in range(E):
        tp.start_epoch(epoch)
        for x, y in loader:
            tp.step_start()
            x, y = x.to(dev), y.to(dev)
            tp.mark_data_loaded()
            loss = criterion(model(x), y)
            tp.mark_forward_done()
            loss.backward()
            tp.mark_backward_done()
            opt.step(); opt.zero_grad()
            tp.step_end(loss=loss, batch_size=x.size(0))
        tp.end_epoch()

2. The `profiled_train` decorator around a generator that yields per-step dicts,
   which adds torch.profiler tracing and control.json handling.

Every step is written as a "pytorch" record and every epoch as "pytorch_epoch".
"""
import functools
import math
import os
import time

from profiler.src.control_handler import ControlHandler
from profiler.src.metric_writer import write_metric

try:
    import torch
except ImportError:  # the marker API also works without torch (dummy mode)
    torch = None

PHASES = ("data", "forward", "backward", "optimizer")


def _to_float(x):
    if torch is not None and torch.is_tensor(x):
        return float(x.detach().float().cpu().item())
    return float(x)


def grad_norm(model) -> float:
    """Global L2 norm of all parameter gradients."""
    if torch is None or model is None:
        return float("nan")
    norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


class TrainingProfiler:
    def __init__(self, model=None, batch_size=None, sync_cuda=True, track_grad_norm=True,
                 log_every=1, write=True):
        self.model = model
        self.batch_size = batch_size
        self.sync_cuda = sync_cuda and torch is not None and torch.cuda.is_available()
        self.track_grad_norm = track_grad_norm and model is not None
        self.log_every = max(1, log_every)
        self.write = write

        self.epoch = 0
        self.global_step = 0
        self._marks = {}
        self._epoch_steps = []
        self._epoch_start = None
        self.epoch_summaries = []

    # -- timing ------------------------------------------------------------
    def _now(self):
        # GPU work is asynchronous: without a sync we'd time the *launch*, not the work.
        if self.sync_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def start_epoch(self, epoch=None):
        if epoch is None:
            epoch = self.epoch + 1 if self.epoch_summaries else 0
        self.epoch = epoch
        self._epoch_steps = []
        self._epoch_start = time.perf_counter()

    def step_start(self):
        if self._epoch_start is None:
            self.start_epoch()
        self._marks = {"start": self._now()}

    def mark_data_loaded(self):
        self._marks["data"] = self._now()

    def mark_forward_done(self):
        self._marks["forward"] = self._now()

    def mark_backward_done(self):
        self._marks["backward"] = self._now()
        if self.track_grad_norm:
            self._marks["grad_norm"] = grad_norm(self.model)

    def step_end(self, loss=None, batch_size=None, **extra) -> dict:
        end = self._now()
        m = self._marks
        if "start" not in m:
            raise RuntimeError("step_end() called without step_start()")

        # Missing markers collapse that phase to zero instead of failing.
        t_data = m.get("data", m["start"])
        t_fwd = m.get("forward", t_data)
        t_bwd = m.get("backward", t_fwd)
        times = {
            "data": t_data - m["start"],
            "forward": t_fwd - t_data,
            "backward": t_bwd - t_fwd,
            "optimizer": end - t_bwd,
        }
        total = end - m["start"]
        bs = batch_size or self.batch_size

        rec = {
            "epoch": self.epoch,
            "step": self.global_step,
            "step_time_ms": total * 1000,
            **{f"{k}_ms": v * 1000 for k, v in times.items()},
            **{f"{k}_frac": (v / total if total > 0 else 0.0) for k, v in times.items()},
            "bottleneck": max(times, key=times.get),
        }
        if bs:
            rec["batch_size"] = bs
            rec["throughput_samples_per_s"] = bs / total if total > 0 else 0.0
        if loss is not None:
            rec["loss"] = _to_float(loss)
        if "grad_norm" in m:
            rec["grad_norm"] = m["grad_norm"]
        if self.sync_cuda:
            rec["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated() / 2 ** 20
            rec["gpu_mem_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        for k, v in extra.items():
            rec[k] = _to_float(v) if torch is not None and torch.is_tensor(v) else v

        self._epoch_steps.append(rec)
        if self.write and self.global_step % self.log_every == 0:
            write_metric("pytorch", rec)
        self.global_step += 1
        self._marks = {}
        return rec

    def end_epoch(self) -> dict:
        steps = self._epoch_steps
        if not steps:
            return {}
        n = len(steps)
        wall = time.perf_counter() - self._epoch_start
        phase_totals = {p: sum(s[f"{p}_ms"] for s in steps) for p in PHASES}
        step_total = sum(s["step_time_ms"] for s in steps)
        losses = [s["loss"] for s in steps if "loss" in s and math.isfinite(s["loss"])]
        samples = sum(s.get("batch_size", 0) for s in steps)
        step_ms = sorted(s["step_time_ms"] for s in steps)

        summary = {
            "epoch": self.epoch,
            "steps": n,
            "wall_time_s": wall,
            "mean_step_ms": step_total / n,
            "p50_step_ms": step_ms[n // 2],
            "p95_step_ms": step_ms[min(n - 1, int(0.95 * n))],
            **{f"{p}_frac": (phase_totals[p] / step_total if step_total else 0.0) for p in PHASES},
            "bottleneck": max(phase_totals, key=phase_totals.get),
        }
        if samples:
            summary["throughput_samples_per_s"] = samples / wall if wall > 0 else 0.0
        if losses:
            summary["loss_first"] = losses[0]
            summary["loss_last"] = losses[-1]
            summary["loss_mean"] = sum(losses) / len(losses)
        norms = [s["grad_norm"] for s in steps if "grad_norm" in s and math.isfinite(s["grad_norm"])]
        if norms:
            summary["grad_norm_mean"] = sum(norms) / len(norms)
            summary["grad_norm_max"] = max(norms)

        if self.write:
            write_metric("pytorch_epoch", summary)
        self.epoch_summaries.append(summary)
        self._epoch_start = None
        return summary


def profiled_train(use_cuda=True, record_shapes=False, profile_steps=None, export_trace_path=None):
    """Decorate a generator `f(model, dataloader, criterion, optimizer, device=..., handler=...)`
    that yields a dict per step (at least {"loss": ...}). Each step is logged, control
    commands are applied between steps, and a torch.profiler trace is optionally exported.
    Returns the ControlHandler so callers can pick up a rebuilt dataloader for the next epoch.
    """
    if torch is None:
        raise ImportError("profiled_train requires torch")
    from torch.profiler import ProfilerActivity, profile

    def decorator(func):
        @functools.wraps(func)
        def wrapped(model, dataloader, criterion, optimizer, device=None, **kwargs):
            if device is None:
                device = next(model.parameters()).device

            acts = [ProfilerActivity.CPU]
            if use_cuda and torch.cuda.is_available():
                acts.append(ProfilerActivity.CUDA)

            handler = ControlHandler(job_id=os.environ.get("SLURM_JOB_ID", "unknown"),
                                     model=model, dataloader=dataloader, optimizer=optimizer)

            step_count = 0
            with profile(activities=acts, record_shapes=record_shapes) as prof:
                for metrics in func(model, dataloader, criterion, optimizer,
                                    device=device, handler=handler, **kwargs):
                    if not isinstance(metrics, dict):
                        metrics = {"value": str(metrics)}
                    metrics.setdefault("step", step_count)
                    if "loss" in metrics:
                        metrics["loss"] = _to_float(metrics["loss"])
                    write_metric("pytorch", metrics)

                    cmd = handler.check_and_apply()
                    if cmd:
                        print(f"[torch_hooks] applied: {cmd}")

                    step_count += 1
                    prof.step()
                    if handler.stop_requested or (profile_steps is not None and step_count >= profile_steps):
                        break

            if export_trace_path:
                prof.export_chrome_trace(export_trace_path)
            if handler.dataloader_dirty:
                handler.rebuild_dataloader()
            handler.profiler = prof
            return handler
        return wrapped
    return decorator
