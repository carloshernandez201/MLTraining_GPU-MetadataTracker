"""Summarise the last N seconds of metrics and ask an LLM for one tuning action.

    python -m profiler.src.feedLLMData --interval 60 --dry-run   # just print the summary
    python -m profiler.src.feedLLMData --interval 60             # call the model, write control.json

The model's reply is validated and written to logs/<job>/control.json, where the
C++ daemon (system actions) or the Python ControlHandler (training actions) picks it up.
"""
import argparse
import json
import os
import time

from profiler.src.control_handler import ALL_ACTIONS
from profiler.src.metric_writer import get_log_dir, get_log_path


def iter_lines_reverse(path, chunk_size=8192):
    """Yield the lines of a file from last to first without reading it all into memory."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        buf = b""
        while pos > 0:
            step = min(chunk_size, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
            lines = buf.split(b"\n")
            buf = lines[0]  # possibly incomplete — keep for the next chunk
            for line in reversed(lines[1:]):
                if line.strip():
                    yield line
        if buf.strip():
            yield buf


def summarize_window(path, interval: float, now: float = None) -> dict:
    """Collect records newer than `now - interval`, condensing per-step PyTorch
    records into one trend summary to keep the prompt small."""
    cutoff = (now if now is not None else time.time()) - interval
    gpu, slurm, epochs, steps = [], [], [], []

    for raw in iter_lines_reverse(path):
        try:
            entry = json.loads(raw.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if entry.get("timestamp", 0) < cutoff:
            break
        src = entry.get("source")
        data = entry.get("data", {})
        if src == "gpu":
            gpu.append({k: v for k, v in data.items() if k != "devices"})
        elif src == "slurm":
            slurm.append(data)
        elif src == "pytorch_epoch":
            epochs.append(data)
        elif src == "pytorch":
            steps.append(data)

    # everything above was collected newest-first
    gpu.reverse(); epochs.reverse(); steps.reverse()

    pytorch = {}
    if steps:
        losses = [s["loss"] for s in steps if isinstance(s.get("loss"), (int, float))]
        pytorch = {"steps": len(steps), "step_first": steps[0].get("step"), "step_last": steps[-1].get("step")}
        if losses:
            pytorch.update({
                "loss_oldest": losses[0],
                "loss_newest": losses[-1],
                "loss_trend": "decreasing" if losses[-1] < losses[0]
                              else "increasing" if losses[-1] > losses[0] else "flat",
            })
        for key in ("step_time_ms", "data_frac", "forward_frac", "backward_frac",
                    "optimizer_frac", "throughput_samples_per_s", "grad_norm"):
            vals = [s[key] for s in steps if isinstance(s.get(key), (int, float))]
            if vals:
                pytorch[f"mean_{key}"] = sum(vals) / len(vals)
        bottlenecks = [s["bottleneck"] for s in steps if "bottleneck" in s]
        if bottlenecks:
            pytorch["most_common_bottleneck"] = max(set(bottlenecks), key=bottlenecks.count)

    return {
        "window_seconds": interval,
        "gpu": gpu[-30:],       # cap prompt size
        "pytorch": pytorch,
        "pytorch_epochs": epochs[-5:],
        "slurm": slurm[-1:] if slurm else [],
    }


SYSTEM_PROMPT = (
    "You are part of a tool that analyzes ML training telemetry (GPU metrics, PyTorch step "
    "timing, Slurm job metadata) and recommends ONE change to make training safer or more "
    "efficient. Respond ONLY with a JSON object containing an 'action', a 'reason', and the "
    "action's parameters. Allowed actions: "
    "set_lr{lr}, scale_lr{factor}, set_grad_clip{max_norm}, set_batch_size{batch_size}, "
    "stop_training{}, reduce_power_limit{gpu_index,watts}, drain_node{}, none{}. "
    "Use 'none' when nothing should change."
)


def get_ai_response(path=None, interval=60, model=None, write=True) -> dict:
    import openai  # imported lazily so the rest of the package doesn't need it

    summary = summarize_window(path or get_log_path(), interval)
    client = openai.OpenAI()  # reads OPENAI_API_KEY from env
    chat = client.chat.completions.create(
        model=model or os.environ.get("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(summary)},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    reply = json.loads(chat.choices[0].message.content)

    if write and reply.get("action") in ALL_ACTIONS:
        out = os.path.join(get_log_dir(), "control.json")
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(reply, f)
        os.replace(tmp, out)  # atomic: consumers never see a half-written file
    return reply


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None, help="metrics.jsonl (default: logs/<job>/metrics.jsonl)")
    ap.add_argument("--interval", type=float, default=60, help="seconds of history to summarise")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the summary, don't call the model")
    args = ap.parse_args()

    if args.dry_run:
        print(json.dumps(summarize_window(args.path or get_log_path(), args.interval), indent=2))
    else:
        print(json.dumps(get_ai_response(args.path, args.interval, args.model), indent=2))


if __name__ == "__main__":
    main()
