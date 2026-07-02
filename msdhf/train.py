import argparse
from pathlib import Path

import torch
from torch import nn

from msdhf.data import make_dataloaders
from msdhf.models import MSDHFNet
from msdhf.utils.early_stopping import EarlyStopping
from msdhf.utils.metrics import mae, mse
from msdhf.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--features", type=str, default="M", choices=["M", "S"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-con", type=float, default=1e-3)
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=[
            "none",
            "without_ncd",
            "without_local",
            "without_global",
            "without_igf",
            "without_ccf",
            "without_lcon",
        ],
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_epoch(model, loader, optimizer, criterion, device, lambda_con):
    model.train()
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred, aux = model(x, return_aux=True)
        loss = criterion(pred, y) + lambda_con * aux["consistency_loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(model, loader, criterion, device, lambda_con):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred, aux = model(x, return_aux=True)
        loss = criterion(pred, y) + lambda_con * aux["consistency_loss"]
        total_loss += loss.item() * x.size(0)
        total_mse += mse(pred, y).item() * x.size(0)
        total_mae += mae(pred, y).item() * x.size(0)
        total_count += x.size(0)
    total_count = max(1, total_count)
    return total_loss / total_count, total_mse / total_count, total_mae / total_count


def main():
    args = parse_args()
    set_seed(args.seed)
    train_loader, val_loader, test_loader, channels, _, columns = make_dataloaders(
        args.data, args.seq_len, args.pred_len, args.batch_size, args.num_workers, args.features, args.target
    )
    print(f"fields {','.join(columns)}")
    device = torch.device(args.device)
    model = MSDHFNet(
        channels=channels,
        pred_len=args.pred_len,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        ablation=args.ablation,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    dataset_name = Path(args.data).stem
    ckpt_path = Path(args.save_dir) / f"{dataset_name}_h{args.pred_len}_{args.ablation}.pt"
    stopper = EarlyStopping(args.patience, ckpt_path)
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.lambda_con)
        val_loss, val_mse, val_mae = evaluate(model, val_loader, criterion, device, args.lambda_con)
        print(
            f"epoch {epoch:03d} train_loss {train_loss:.6f} "
            f"val_loss {val_loss:.6f} val_mse {val_mse:.6f} val_mae {val_mae:.6f}"
        )
        stopper.step(val_loss, model)
        if stopper.stopped:
            break
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_mse, test_mae = evaluate(model, test_loader, criterion, device, args.lambda_con)
    print(f"test_loss {test_loss:.6f} test_mse {test_mse:.6f} test_mae {test_mae:.6f}")


if __name__ == "__main__":
    main()
