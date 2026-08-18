import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import EMOTION_CLASSES, EMOTION_MAP, EmbeddingDataset


def test_emotion_classes_are_the_4_target_labels():
    assert EMOTION_CLASSES == ["angry", "happy", "neutral", "sad"]


def test_emotion_map_collapses_excited_into_happy():
    assert EMOTION_MAP["hap"] == "happy"
    assert EMOTION_MAP["exc"] == "happy"


def test_embedding_dataset_shapes_and_dtypes():
    df = pd.DataFrame({
        "emotion_idx": [0, 1, 2],
        "arousal": [0.5, -0.2, 0.0],
        "valence": [-0.1, 0.3, 0.9],
    })
    embeddings = np.random.randn(3, 8).astype(np.float32)

    ds = EmbeddingDataset(embeddings, df)
    assert len(ds) == 3

    emb, emo, aro, val = ds[0]
    assert emb.shape == (8,)
    assert emo.dtype.is_floating_point is False  # long
    assert aro.dtype == val.dtype
