"""Offline WAV inference (Section 15). This is the stable interface Phase 2
will later wrap: EmotionModel(checkpoint_path).predict(wav_path)."""
import time

import soundfile as sf
import torch

from .audio import load_and_preprocess_audio
from .model import EmotionHeads, extract_embedding


class EmotionModel:
    def __init__(self, checkpoint_path, backbone_model, device, funasr_output_dir):
        _t0 = time.time()
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.device = device
        self.emotion_classes = ckpt["emotion_classes"]
        self.funasr_output_dir = funasr_output_dir
        self.heads = EmotionHeads(input_dim=ckpt["embed_dim"], num_emotions=len(self.emotion_classes)).to(device)
        self.heads.load_state_dict(ckpt["model_state_dict"])
        self.heads.eval()
        self.backbone = backbone_model
        print(f"[MODEL] loaded in {time.time() - _t0:.2f}s")

    def predict(self, wav_path):
        raw_wav, raw_sr = sf.read(wav_path, always_2d=False)
        duration = len(raw_wav) / raw_sr
        print(f"[AUDIO] duration={duration:.2f}s")

        _t0 = time.time()
        load_and_preprocess_audio(wav_path)  # validation pass (mono/16k/NaN/empty checks)
        emb = extract_embedding(wav_path, self.backbone, self.funasr_output_dir)
        emb_t = torch.from_numpy(emb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emo_logits, aro_pred, val_pred = self.heads(emb_t)
            probs = torch.softmax(emo_logits, dim=-1).cpu().numpy()[0]

        latency = time.time() - _t0
        pred_idx = int(probs.argmax())
        result = {
            "emotion": self.emotion_classes[pred_idx],
            "arousal": float(aro_pred.cpu().numpy()[0]),
            "valence": float(val_pred.cpu().numpy()[0]),
            "emotion_probabilities": {c: float(p) for c, p in zip(self.emotion_classes, probs)},
            "latency_sec": latency,
        }

        print(f"[INFERENCE] latency={latency:.2f}s")
        print(f"[RESULT] emotion={result['emotion']}")
        print(f"[RESULT] arousal={result['arousal']:.2f}")
        print(f"[RESULT] valence={result['valence']:.2f}")
        return result


def print_prediction_block(result):
    print("=" * 40)
    print("AdaptiveCX Stage 1")
    print("=" * 40)
    print()
    print(f"Emotion: {result['emotion']}")
    print(f"Arousal: {result['arousal']:.2f}")
    print(f"Valence: {result['valence']:.2f}")
    print()
    print("Emotion probabilities:")
    for cls, p in sorted(result["emotion_probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {cls:<8}: {p:.2f}")
    print()
    print(f"Inference latency: {result['latency_sec']:.2f} sec")
    print("=" * 40)
