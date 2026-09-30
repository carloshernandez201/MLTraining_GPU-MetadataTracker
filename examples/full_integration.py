"""End-to-end demo: GPU sampling + per-step training breakdown + Slurm context.

    python examples/full_integration.py --dummy          # anywhere, no GPU needed
    python examples/full_integration.py --epochs 3       # real run (on a GPU node)

Output lands in profile_data/<SLURM_JOB_ID or "unknown">/:
    metrics.jsonl   every gpu / pytorch / pytorch_epoch / slurm / control record
    summary.json    per-epoch summaries + collector info
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None


def build_torch_job(device, batch_size, n_samples):
    """Small CNN on synthetic CIFAR-shaped data (no downloads)."""
    x = torch.randn(n_samples, 3, 32, 32)
    y = torch.randint(0, 10, (n_samples,))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                        num_workers=2 if device.type == "cuda" else 0,
                        pin_memory=device.type == "cuda", drop_last=True)
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, 10),
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    return model, loader, nn.CrossEntropyLoss(), opt


def train_torch(tp, handler, args, device):
    model, loader, criterion, opt = build_torch_job(device, args.batch_size, args.samples)
    tp.model, tp.track_grad_norm = model, True
    handler.model, handler.dataloader, handler.optimizer = model, loader, opt

    for epoch in range(args.epochs):
        tp.start_epoch(epoch)
        for i, (x, y) in enumerate(loader):
            if args.steps and i >= args.steps:
                break
            tp.step_start()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            tp.mark_data_loaded()
            loss = criterion(model(x), y)
            tp.mark_forward_done()
            loss.backward()
            tp.mark_backward_done()
            if handler.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), handler.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            tp.step_end(loss=loss, batch_size=x.size(0), lr=opt.param_groups[0]["lr"])

            if handler.check_and_apply():
                print(f"  applied control command, lr={opt.param_groups[0]['lr']}")
            if handler.stop_requested:
                break
        yield tp.end_epoch()
        if handler.stop_requested:
            return
        if handler.dataloader_dirty:
            loader = handler.rebuild_dataloader()


def train_simulated(tp, handler, args):
    """No torch: sleep through fake phases so the pipeline can still be exercised."""
    loss = 2.3
    for epoch in range(args.epochs):
        tp.start_epoch(epoch)
        for _ in range(args.steps or 20):
            tp.step_start()
            time.sleep(0.004)
            tp.mark_data_loaded()
            time.sleep(0.006)
            tp.mark_forward_done()
            time.sleep(0.010)
            tp.mark_backward_done()
            time.sleep(0.002)
            loss *= 0.97
            tp.step_end(loss=loss, batch_size=args.batch_size)
            handler.check_and_apply()
            if handler.stop_requested:
                break
        yield tp.end_epoch()
        if handler.stop_requested:
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dummy", action="store_true", help="no GPU: dummy GPU metrics, CPU (or simulated) training")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=0, help="max steps per epoch (0 = full epoch)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--samples", type=int, default=8192)
    ap.add_argument("--interval", type=float, default=0.5, help="GPU sampling interval (s)")
    ap.add_argument("--out", default="profile_data")
    args = ap.parse_args()

    os.environ.setdefault("PROFILER_LOG_DIR", args.out)
    from profiler import ControlHandler, GPUCollector, SlurmMonitor, TrainingProfiler, get_log_dir, get_log_path

    if args.dummy and not args.steps:
        args.steps = 20

    slurm = SlurmMonitor()
    info = slurm.snapshot()
    print(f"slurm: {'job ' + info['slurm_job_id'] + ' on ' + info['node_list'] if info else 'not in a Slurm job'}")

    use_cuda = torch is not None and torch.cuda.is_available() and not args.dummy
    device = torch.device("cuda" if use_cuda else "cpu") if torch is not None else None

    collector = GPUCollector(interval=args.interval, backend="dummy" if args.dummy else "auto")
    print(f"gpu collector backend: {collector.backend_name}")
    print(f"training: {'torch on ' + str(device) if torch is not None else 'simulated (torch not installed)'}")

    tp = TrainingProfiler(batch_size=args.batch_size)
    handler = ControlHandler()

    with collector:
        epochs = train_torch(tp, handler, args, device) if torch is not None else train_simulated(tp, handler, args)
        for s in epochs:
            print(f"epoch {s['epoch']}: {s['steps']} steps, {s['mean_step_ms']:.1f} ms/step, "
                  f"{s.get('throughput_samples_per_s', 0):.0f} samples/s, loss {s.get('loss_last', float('nan')):.4f}, "
                  f"bottleneck={s['bottleneck']} "
                  f"(data {s['data_frac']:.0%} / fwd {s['forward_frac']:.0%} / "
                  f"bwd {s['backward_frac']:.0%} / opt {s['optimizer_frac']:.0%})")

    summary_path = os.path.join(get_log_dir(), "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "slurm": info,
            "gpu_backend": collector.backend_name,
            "gpu_samples": collector.samples,
            "device": str(device) if device is not None else "simulated",
            "epochs": tp.epoch_summaries,
        }, f, indent=2)
    print(f"\nwrote {get_log_path()} and {summary_path}")


if __name__ == "__main__":
    main()
