#!/usr/bin/env python
"""python scripts/train.py

Trains Stage A (frozen backbone) task heads on the embeddings produced by
scripts/prepare_data.py (data/processed/{split}_embeddings.npy +
data/metadata/{split}_meta.csv). Run prepare_data.py first.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import EMOTION_CLASSES, EmbeddingDataset
from src.train import train_stage_a
from src.utils import get_device, load_config


def _load_split(processed_dir, metadata_dir, split_name):
    emb_path = os.path.join(processed_dir, f"{split_name}_embeddings.npy")
    meta_path = os.path.join(metadata_dir, f"{split_name}_meta.csv")
    if not os.path.isfile(emb_path) or not os.path.isfile(meta_path):
        print(f"Missing {emb_path} or {meta_path}. Run scripts/prepare_data.py first.")
        sys.exit(1)
    return np.load(emb_path), pd.read_csv(meta_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = os.path.join(ROOT, "data", "processed")
    metadata_dir = os.path.join(ROOT, "data", "metadata")
    models_dir = os.path.join(ROOT, "models")

    train_emb, train_meta = _load_split(processed_dir, metadata_dir, "train")
    val_emb, val_meta = _load_split(processed_dir, metadata_dir, "val")

    train_ds = EmbeddingDataset(train_emb, train_meta)
    val_ds = EmbeddingDataset(val_emb, val_meta)

    device = get_device()
    print("Using device:", device)

    train_stage_a(train_ds, val_ds, embed_dim=train_emb.shape[1], emotion_classes=EMOTION_CLASSES,
                  config=config, models_dir=models_dir, device=device)


if __name__ == "__main__":
    main()
