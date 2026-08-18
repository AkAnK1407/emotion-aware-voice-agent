"""Shared helpers: config loading, seeding, device selection."""
import random
import time

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Timer:
    """Context manager that prints '[label] elapsed=Xs' on exit."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"[{self.label}] elapsed={time.time() - self._t0:.2f}s")
