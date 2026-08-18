"""emotion2vec+ backbone loading + embedding extraction, and the downstream
multi-task heads trained on top of it (Sections 3/10 of the project spec).

Backbone: emotion2vec_plus_base, loaded via funasr (its actual supported
inference API -- not plain transformers.AutoModel). ModelScope is tried
first with a Hugging Face fallback, matching the verified Kaggle run.
"""
import glob
import os

import numpy as np
import torch
import torch.nn as nn

MODEL_ID_MODELSCOPE = "iic/emotion2vec_plus_base"
MODEL_ID_HF = "emotion2vec/emotion2vec_plus_base"


def load_emotion2vec():
    """Returns (model, checkpoint_id, source). Downloads/caches on first use."""
    from funasr import AutoModel

    try:
        m = AutoModel(model=MODEL_ID_MODELSCOPE, hub="ms", disable_update=True)
        print(f"[MODEL] loaded '{MODEL_ID_MODELSCOPE}' from ModelScope")
        return m, MODEL_ID_MODELSCOPE, "ModelScope"
    except Exception as e:
        print(f"[MODEL] ModelScope load failed ({e!r}); falling back to Hugging Face hub")
        m = AutoModel(model=MODEL_ID_HF, hub="hf", disable_update=True)
        print(f"[MODEL] loaded '{MODEL_ID_HF}' from Hugging Face")
        return m, MODEL_ID_HF, "HuggingFace"


def extract_embedding(wav_path, model, output_dir):
    """One utterance-level embedding vector for wav_path.

    Defensive: funasr has returned the embedding inline under 'feats' in some
    releases and only written it to output_dir as a .npy file in others.
    """
    os.makedirs(output_dir, exist_ok=True)
    res = model.generate(
        input=wav_path,
        granularity="utterance",
        extract_embedding=True,
        output_dir=output_dir,
    )
    item = res[0]
    emb = item.get("feats", None)
    if emb is None:
        key = item.get("key", os.path.splitext(os.path.basename(wav_path))[0])
        candidates = glob.glob(os.path.join(output_dir, "**", f"{key}*.npy"), recursive=True)
        if candidates:
            emb = np.load(candidates[0])
    if emb is None:
        raise RuntimeError(
            f"Could not extract an embedding for {wav_path}; raw output keys={list(item.keys())}"
        )
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim > 1:
        emb = emb.mean(axis=0)  # frame-level -> mean-pool to one utterance vector
    return emb


class EmotionHeads(nn.Module):
    """Shared trunk + three task heads on top of a frozen emotion2vec+ embedding."""

    def __init__(self, input_dim, num_emotions, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(hidden_dim, num_emotions)
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        emotion_logits = self.emotion_head(h)
        arousal = torch.tanh(self.arousal_head(h)).squeeze(-1)
        valence = torch.tanh(self.valence_head(h)).squeeze(-1)
        return emotion_logits, arousal, valence
