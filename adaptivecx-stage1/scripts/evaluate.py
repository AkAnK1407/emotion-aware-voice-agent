#!/usr/bin/env python
"""python scripts/evaluate.py

Evaluates a checkpoint on the cached test-split embeddings produced by
scripts/prepare_data.py. Writes results/evaluation.json + confusion_matrix.png.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import EmbeddingDataset
from src.evaluate import evaluate_checkpoint
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "models", "best_stage1.pt"))
    args = parser.parse_args()

    processed_dir = os.path.join(ROOT, "data", "processed")
    metadata_dir = os.path.join(ROOT, "data", "metadata")
    results_dir = os.path.join(ROOT, "results")

    emb_path = os.path.join(processed_dir, "test_embeddings.npy")
    meta_path = os.path.join(metadata_dir, "test_meta.csv")
    if not os.path.isfile(emb_path) or not os.path.isfile(meta_path):
        print(f"Missing {emb_path} (no local test embeddings cached).")
        print("This machine only has the metadata CSV from the Kaggle run, not the raw")
        print("audio/embeddings needed to recompute metrics locally. Either:")
        print("  - run scripts/prepare_data.py against a local IEMOCAP copy first, or")
        print("  - just read results/evaluation.json, which already has the Kaggle-run numbers.")
        sys.exit(1)

    test_ds = EmbeddingDataset(np.load(emb_path), pd.read_csv(meta_path))
    device = get_device()
    print("Using device:", device)

    evaluate_checkpoint(args.checkpoint, test_ds, device, results_dir)


if __name__ == "__main__":
    main()
