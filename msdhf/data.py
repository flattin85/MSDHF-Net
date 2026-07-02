from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class ForecastDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):
        x_start = index
        x_end = x_start + self.seq_len
        y_end = x_end + self.pred_len
        x = self.data[x_start:x_end]
        y = self.data[x_end:y_end]
        return torch.from_numpy(x), torch.from_numpy(y)


def _date_column(columns):
    names = {"date", "datetime", "time", "timestamp"}
    for column in columns:
        if str(column).strip().lower() in names:
            return column
    return None


def _read_csv(path, features="M", target="OT"):
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    date = _date_column(frame.columns)
    candidates = [column for column in frame.columns if column != date]
    for column in candidates:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric_columns = [
        column
        for column in candidates
        if pd.api.types.is_numeric_dtype(frame[column]) and not frame[column].isna().all()
    ]
    if not numeric_columns:
        raise ValueError("no numeric columns found in dataset")
    if features == "S":
        if target not in numeric_columns:
            raise ValueError(f"numeric target column {target} not found")
        columns = [target]
    else:
        columns = list(numeric_columns)
        if target in columns:
            columns = [column for column in columns if column != target] + [target]
    data = frame[columns].ffill().bfill().to_numpy(dtype=np.float32)
    return data, columns


def _standard_split(length):
    train_end = int(length * 0.7)
    val_end = int(length * 0.8)
    return train_end, val_end, length


def _ett_split(path, length):
    name = Path(path).stem.lower()
    if name in {"etth1", "etth2"} and length >= 17420:
        return 12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24
    if name in {"ettm1", "ettm2"} and length >= 69680:
        return 12 * 30 * 24 * 4, 16 * 30 * 24 * 4, 20 * 30 * 24 * 4
    return _standard_split(length)


def make_dataloaders(path, seq_len, pred_len, batch_size, num_workers=0, features="M", target="OT"):
    raw, columns = _read_csv(path, features, target)
    train_end, val_end, test_end = _ett_split(path, len(raw))
    scaler = StandardScaler()
    scaler.fit(raw[:train_end])
    data = scaler.transform(raw).astype(np.float32)
    train_data = data[:train_end]
    val_start = max(0, train_end - seq_len)
    test_start = max(0, val_end - seq_len)
    val_data = data[val_start:val_end]
    test_data = data[test_start:test_end]
    train_set = ForecastDataset(train_data, seq_len, pred_len)
    val_set = ForecastDataset(val_data, seq_len, pred_len)
    test_set = ForecastDataset(test_data, seq_len, pred_len)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, raw.shape[1], scaler, columns
