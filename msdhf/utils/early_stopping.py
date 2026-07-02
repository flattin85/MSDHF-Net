from pathlib import Path

import torch


class EarlyStopping:
    def __init__(self, patience, path):
        self.patience = patience
        self.path = Path(path)
        self.best = None
        self.counter = 0
        self.stopped = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def step(self, value, model):
        if self.best is None or value < self.best:
            self.best = value
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.stopped = True
        return False
