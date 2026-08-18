#!/usr/bin/env python
"""python scripts/predict.py --audio samples/test.wav

Works entirely offline once the emotion2vec+ checkpoint has been downloaded
and cached by funasr on first use (needs internet the very first time).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.inference import EmotionModel, print_prediction_block
from src.model import load_emotion2vec
from src.utils import Timer, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=os.path.join(ROOT, "samples", "test.wav"))
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "models", "best_stage1.pt"))
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"No such file: {args.audio}")
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(f"No such checkpoint: {args.checkpoint}")
        sys.exit(1)

    device = get_device()
    print("Using device:", device)

    with Timer("MODEL backbone load"):
        backbone, checkpoint_id, source = load_emotion2vec()
    print(f"[MODEL] backbone: {checkpoint_id} (via {source})")

    funasr_output_dir = os.path.join(ROOT, "data", "processed", "_funasr_raw_out")
    model = EmotionModel(args.checkpoint, backbone, device, funasr_output_dir)

    result = model.predict(args.audio)
    print()
    print_prediction_block(result)


if __name__ == "__main__":
    main()
