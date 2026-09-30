"""Applies *training-side* commands from logs/<job>/control.json between steps.

System-side actions (reduce_power_limit, drain_node) are left for the C++ daemon.

Supported actions (JSON object with an "action" key):
  {"action": "set_lr", "lr": 1e-4}
  {"action": "scale_lr", "factor": 0.5}
  {"action": "set_grad_clip", "max_norm": 1.0}
  {"action": "set_batch_size", "batch_size": 64}   # takes effect on the next epoch
  {"action": "stop_training", "reason": "..."}
"""
import json
import os

from profiler.src.metric_writer import get_log_dir, write_metric

TRAINING_ACTIONS = {"set_lr", "scale_lr", "set_grad_clip", "set_batch_size", "stop_training"}
SYSTEM_ACTIONS = {"reduce_power_limit", "drain_node"}
ALL_ACTIONS = TRAINING_ACTIONS | SYSTEM_ACTIONS


class ControlHandler:
    def __init__(self, job_id=None, model=None, dataloader=None, optimizer=None, control_path=None):
        self.job_id = job_id
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.control_path = control_path or os.path.join(get_log_dir(), "control.json")
        self.grad_clip = None
        self.stop_requested = False
        self.dataloader_dirty = False
        self._pending_batch_size = None

    def _read(self):
        if not os.path.exists(self.control_path):  # fast path, called every step
            return None
        try:
            with open(self.control_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def check_and_apply(self):
        """Apply a pending training command. Returns the command dict, or None."""
        cmd = self._read()
        if not isinstance(cmd, dict) or cmd.get("action") not in TRAINING_ACTIONS:
            return None
        try:
            os.remove(self.control_path)  # consume it
        except OSError:
            pass

        action = cmd["action"]
        ok = True
        if action == "set_lr" and self.optimizer is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = float(cmd["lr"])
        elif action == "scale_lr" and self.optimizer is not None:
            for g in self.optimizer.param_groups:
                g["lr"] *= float(cmd["factor"])
        elif action == "set_grad_clip":
            self.grad_clip = float(cmd["max_norm"])
        elif action == "set_batch_size":
            self._pending_batch_size = int(cmd["batch_size"])
            self.dataloader_dirty = True
        elif action == "stop_training":
            self.stop_requested = True
        else:
            ok = False

        write_metric("control", {"applied": ok, **cmd})
        return cmd

    def rebuild_dataloader(self):
        """Return a DataLoader over the same dataset with the requested batch size."""
        self.dataloader_dirty = False
        dl = self.dataloader
        if dl is None or self._pending_batch_size is None:
            return dl
        from torch.utils.data import DataLoader

        self.dataloader = DataLoader(
            dl.dataset,
            batch_size=self._pending_batch_size,
            shuffle=True,
            num_workers=dl.num_workers,
            pin_memory=dl.pin_memory,
            drop_last=dl.drop_last,
        )
        self._pending_batch_size = None
        return self.dataloader
