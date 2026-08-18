#!/usr/bin/env python
"""python scripts/prepare_data.py --data-root <path to a local IEMOCAP copy>

Parses IEMOCAP, applies the 4-class emotion mapping, does the speaker-
independent Session 1-3/4/5 split, extracts + caches one emotion2vec+
embedding per utterance, and writes everything under data/.

Requires a local IEMOCAP copy (SessionX/dialog/EmoEvaluation/*.txt +
SessionX/sentences/wav/**/*.wav). The checkpoint in models/best_stage1.pt
was trained on Kaggle against IEMOCAP attached as a Kaggle input dataset;
this script reproduces the same steps locally if you have your own copy.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    EMOTION_CLASSES,
    build_metadata,
    extract_split_embeddings,
    find_iemocap_root,
    speaker_independent_split,
)
from src.model import extract_embedding, load_emotion2vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Local IEMOCAP root (or any ancestor of it)")
    args = parser.parse_args()

    processed_dir = os.path.join(ROOT, "data", "processed")
    metadata_dir = os.path.join(ROOT, "data", "metadata")
    emb_cache_dir = os.path.join(processed_dir, "embeddings")
    funasr_output_dir = os.path.join(processed_dir, "_funasr_raw_out")
    os.makedirs(metadata_dir, exist_ok=True)

    iemocap_root, session_dirs = find_iemocap_root(args.data_root)
    print("[DATA] IEMOCAP root:", iemocap_root)
    assert len(session_dirs) == 5, f"Expected 5 IEMOCAP sessions, found {len(session_dirs)}"

    meta = build_metadata(session_dirs)
    train_df, val_df, test_df = speaker_independent_split(meta)

    print("\nDataset:")
    print(f"  train:      {len(train_df)}")
    print(f"  validation: {len(val_df)}")
    print(f"  test:       {len(test_df)}")
    print("\nEmotion classes:")
    for c in EMOTION_CLASSES:
        print(" ", c)
    print("\nArousal:\n  available: YES")
    print("\nValence:\n  available: YES")
    print("\nSpeaker independent split: YES")

    backbone, checkpoint_id, source = load_emotion2vec()
    print(f"[MODEL] backbone: {checkpoint_id} (via {source})")

    def embed_fn(wav_path, out_dir):
        return extract_embedding(wav_path, backbone, out_dir)

    # Measure embedding dim once, from a real file.
    probe_emb = embed_fn(train_df.iloc[0]["wav_path"], funasr_output_dir)
    embed_dim = probe_emb.shape[0]
    print("[DATA] embedding dim (measured):", embed_dim)

    import numpy as np
    for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        emb, df = extract_split_embeddings(df, split_name, embed_fn, embed_dim, emb_cache_dir, funasr_output_dir)
        np.save(os.path.join(processed_dir, f"{split_name}_embeddings.npy"), emb)
        df.to_csv(os.path.join(metadata_dir, f"{split_name}_meta.csv"), index=False)
        print(f"[DATA] {split_name}: {emb.shape}")

    print("\n[DATA] Done. Embeddings + metadata written under", processed_dir, "and", metadata_dir)


if __name__ == "__main__":
    main()
