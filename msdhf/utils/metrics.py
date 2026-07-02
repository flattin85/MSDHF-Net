import torch


def mse(pred, true):
    return torch.mean((pred - true) ** 2)


def mae(pred, true):
    return torch.mean(torch.abs(pred - true))


def metric_dict(pred, true):
    return {"mse": mse(pred, true).item(), "mae": mae(pred, true).item()}
