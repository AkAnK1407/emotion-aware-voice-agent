#!/usr/bin/env python
"""python scripts/check_audio.py --audio samples/test.wav"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio import inspect_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to a WAV file")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"No such file: {args.audio}")
        sys.exit(1)

    inspect_audio(args.audio)


if __name__ == "__main__":
    main()
