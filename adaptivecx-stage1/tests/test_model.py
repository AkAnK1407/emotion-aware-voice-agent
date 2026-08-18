import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import EmotionHeads


def test_forward_pass_output_shapes():
    batch_size, input_dim, num_emotions = 4, 768, 4
    heads = EmotionHeads(input_dim=input_dim, num_emotions=num_emotions)
    x = torch.randn(batch_size, input_dim)

    emotion_logits, arousal, valence = heads(x)

    assert emotion_logits.shape == (batch_size, num_emotions)
    # squeeze(-1)'d to 1-D so they pair directly with 1-D arousal/valence
    # labels in SmoothL1Loss -- the spec's illustrative [batch_size, 1] shape
    # is equivalent before the squeeze.
    assert arousal.shape == (batch_size,)
    assert valence.shape == (batch_size,)


def test_arousal_valence_bounded_by_tanh():
    heads = EmotionHeads(input_dim=32, num_emotions=4)
    x = torch.randn(16, 32) * 100  # large inputs to stress the tanh bound

    _, arousal, valence = heads(x)

    assert torch.all(arousal.abs() <= 1.0)
    assert torch.all(valence.abs() <= 1.0)


def test_checkpoint_roundtrip(tmp_path):
    heads = EmotionHeads(input_dim=16, num_emotions=4)
    heads.eval()  # dropout must be off for two forward passes to match
    ckpt_path = tmp_path / "heads.pt"
    torch.save({
        "model_state_dict": heads.state_dict(),
        "embed_dim": 16,
        "emotion_classes": ["angry", "happy", "neutral", "sad"],
    }, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    reloaded = EmotionHeads(input_dim=ckpt["embed_dim"], num_emotions=len(ckpt["emotion_classes"]))
    reloaded.load_state_dict(ckpt["model_state_dict"])
    reloaded.eval()

    x = torch.randn(2, 16)
    out_a = heads(x)
    out_b = reloaded(x)
    for a, b in zip(out_a, out_b):
        assert torch.allclose(a, b)
