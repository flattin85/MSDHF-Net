import argparse
import time

import torch

from msdhf.models import MSDHFNet
from msdhf.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    model = MSDHFNet(args.channels, args.pred_len, hidden_dim=args.hidden_dim).to(device).eval()
    x = torch.randn(args.batch_size, args.seq_len, args.channels, device=device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) / args.steps
    print(f"parameters {parameters}")
    print(f"inference_seconds_per_batch {elapsed:.6f}")
    if device.type == "cuda":
        print(f"peak_memory_mb {torch.cuda.max_memory_allocated(device) / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
