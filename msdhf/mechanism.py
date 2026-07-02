import argparse
from pathlib import Path

import torch

from msdhf.data import make_dataloaders
from msdhf.models import MSDHFNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--features", type=str, default="M", choices=["M", "S"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    _, _, test_loader, channels, _, _ = make_dataloaders(
        args.data, args.seq_len, args.pred_len, args.batch_size, args.num_workers, args.features, args.target
    )
    device = torch.device(args.device)
    model = MSDHFNet(channels, args.pred_len, hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(Path(args.checkpoint), map_location=device))
    model.eval()
    scale_weights = []
    path_gates = []
    for x, _ in test_loader:
        x = x.to(device)
        _, aux = model(x, return_aux=True)
        scale_weights.append(aux["scale_weights"].detach().cpu())
        path_gates.append(aux["path_gates"].mean(dim=-1).detach().cpu())
    scale_weights = torch.cat(scale_weights, dim=0).mean(dim=0)
    path_gates = torch.cat(path_gates, dim=0).mean(dim=0)
    print("cross_scale_weights " + " ".join(f"{v:.6f}" for v in scale_weights.tolist()))
    print("local_path_ratios " + " ".join(f"{v:.6f}" for v in path_gates.tolist()))
    print("global_path_ratios " + " ".join(f"{1.0 - v:.6f}" for v in path_gates.tolist()))


if __name__ == "__main__":
    main()
