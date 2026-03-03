import json
import time
import os
import socket
import functools
import torch
from torch.profiler import profile, ProfilerActivity
from profiler.src.metric_writer import write_metric
from profiler.src.control_handler import ControlHandler


def profiled_train(
    use_cuda=True,
    record_shapes=True,
    profile_steps=None,
    export_trace_path=None,
):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(model, dataloader, criterion, optimizer, device=None, **kwargs):
            if device is None:
                device = next(model.parameters()).device

            job_id = os.environ.get("SLURM_JOB_ID", "unknown")

            acts = [ProfilerActivity.CPU]
            if use_cuda and torch.cuda.is_available():
                acts.append(ProfilerActivity.CUDA)

            # control handler — checks control.json between steps
            handler = ControlHandler(
                job_id=job_id,
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
            )

            with profile(activities=acts, record_shapes=record_shapes) as prof:
                step_count = 0

                result = func(model, dataloader, criterion, optimizer, device=device, **kwargs)
                iterable = result if hasattr(result, "__iter__") else []

                for metrics in iterable:
                    if not isinstance(metrics, dict):
                        metrics = {"value": str(metrics)}

                    if "step" not in metrics:
                        metrics["step"] = step_count

                    if "loss" in metrics and torch.is_tensor(metrics["loss"]):
                        metrics["loss"] = float(metrics["loss"].detach().cpu().item())

                    write_metric("pytorch", metrics)

                    # check for agent commands (one os.path.exists call)
                    cmd = handler.check_and_apply()
                    if cmd:
                        print(f"[torch_hooks] applied: {cmd}")

                    # if agent changed batch size, swap dataloader
                    if handler.dataloader_dirty:
                        dataloader = handler.rebuild_dataloader()

                    step_count += 1
                    prof.step()

                    if profile_steps is not None and step_count >= profile_steps:
                        break

            if export_trace_path:
                prof.export_chrome_trace(export_trace_path)

            return prof
        return wrapped
    return decorator


@profiled_train(profile_steps=200, export_trace_path="trace.json")
def train_one_epoch(model, dataloader, criterion, optimizer, device=None):
    model.train()
    for step, (batch, target) in enumerate(dataloader):
        if isinstance(batch, (tuple, list)):
            batch = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in batch]
        elif torch.is_tensor(batch):
            batch = batch.to(device, non_blocking=True)

        if torch.is_tensor(target):
            target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        yield {"step": step, "loss": loss}