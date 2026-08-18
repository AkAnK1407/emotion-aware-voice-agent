"""
voice_cx_model.py — loads Stage 1 (emotion2vec+ backbone + trained heads)
and Stage 2 Version A (acoustic-formula-labeled XGBoost regressors) and runs
inference on a WAV file. Self-contained: models/ sits alongside this file.

See adaptivecx-stage1/README.md and adaptivecx-stage2/README.md (in the
main project repo) for how these were trained and what the numbers mean --
both label sources are documented bootstrap/prototype layers, not ground
truth.

All public methods are synchronous/blocking (torch, funasr, xgboost are all
sync APIs) -- main.py calls these via a thread pool executor, never awaited
directly, so a slow prediction doesn't block other requests.
"""
import glob
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("voice_cx_model")

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGE1_CKPT = os.path.join(ROOT, "models", "best_stage1.pt")
STAGE2_MODELS_DIR = os.path.join(ROOT, "models")
FUNASR_OUTPUT_DIR = os.path.join(ROOT, "_funasr_out")

CX_TARGETS = ["stress", "frustration", "urgency", "escalation_risk"]
FEATURE_COLS = [
    "emotion_angry", "emotion_happy", "emotion_neutral", "emotion_sad", "arousal", "valence",
    "pitch_mean", "pitch_std", "energy_mean", "energy_std",
    "speech_ratio", "speaking_rate", "pause_count", "pause_ratio",
]


class EmotionHeads(nn.Module):
    """Identical architecture to adaptivecx-stage1/src/model.py -- must match
    for the state_dict in best_stage1.pt to load."""

    def __init__(self, input_dim, num_emotions, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.emotion_head = nn.Linear(hidden_dim, num_emotions)
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        emotion_logits = self.emotion_head(h)
        arousal = torch.tanh(self.arousal_head(h)).squeeze(-1)
        valence = torch.tanh(self.valence_head(h)).squeeze(-1)
        return emotion_logits, arousal, valence


class VoiceCXModel:
    """Loads once (roughly 1-4 minutes, mostly the emotion2vec+ backbone).
    Call .load() once at server startup, then .predict_from_wav_path(path)
    per request."""

    def __init__(self):
        self._loaded = False
        self.backbone = None
        self.stage1_heads = None
        self.emotion_classes = None
        self.embed_dim = None
        self.stage2_models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        if self._loaded:
            return
        _t0 = time.time()
        logger.info("[VoiceCX] loading emotion2vec+ backbone (one-time, can take a few minutes)...")
        from funasr import AutoModel
        try:
            self.backbone = AutoModel(model="iic/emotion2vec_plus_base", hub="ms", disable_update=True)
        except Exception as e:
            logger.warning(f"[VoiceCX] ModelScope load failed ({e!r}), falling back to Hugging Face hub")
            self.backbone = AutoModel(model="emotion2vec/emotion2vec_plus_base", hub="hf", disable_update=True)

        if not os.path.isfile(STAGE1_CKPT):
            raise FileNotFoundError(f"Stage 1 checkpoint not found at {STAGE1_CKPT}")
        ckpt = torch.load(STAGE1_CKPT, map_location=self.device)
        self.emotion_classes = ckpt["emotion_classes"]
        self.embed_dim = ckpt["embed_dim"]
        self.stage1_heads = EmotionHeads(
            input_dim=self.embed_dim, num_emotions=len(self.emotion_classes)
        ).to(self.device)
        self.stage1_heads.load_state_dict(ckpt["model_state_dict"])
        self.stage1_heads.eval()

        from xgboost import XGBRegressor
        for target in CX_TARGETS:
            path = os.path.join(STAGE2_MODELS_DIR, f"{target}.json")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Stage 2 model not found at {path}")
            model = XGBRegressor()
            model.load_model(path)
            self.stage2_models[target] = model

        os.makedirs(FUNASR_OUTPUT_DIR, exist_ok=True)
        self._loaded = True
        logger.info(f"[VoiceCX] loaded in {time.time() - _t0:.1f}s (device={self.device})")

    def _extract_embedding(self, wav_path):
        res = self.backbone.generate(
            input=wav_path, granularity="utterance", extract_embedding=True, output_dir=FUNASR_OUTPUT_DIR
        )
        item = res[0]
        emb = item.get("feats", None)
        if emb is None:
            key = item.get("key", os.path.splitext(os.path.basename(wav_path))[0])
            candidates = glob.glob(os.path.join(FUNASR_OUTPUT_DIR, "**", f"{key}*.npy"), recursive=True)
            if candidates:
                emb = np.load(candidates[0])
        if emb is None:
            raise RuntimeError(f"Could not extract embedding for {wav_path}")
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim > 1:
            emb = emb.mean(axis=0)
        return emb

    def _acoustic_features(self, wav, sr):
        import librosa
        from scipy.signal import find_peaks

        duration = len(wav) / sr
        f0, voiced_flag, _ = librosa.pyin(
            wav, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_mean = float(np.nanmean(voiced_f0)) if voiced_f0.size else 0.0
        pitch_std = float(np.nanstd(voiced_f0)) if voiced_f0.size else 0.0

        rms = librosa.feature.rms(y=wav)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        thresh = max(energy_mean * 0.5, 1e-4)
        voiced_frames = rms > thresh
        speech_ratio = float(np.mean(voiced_frames))

        smoothed = np.convolve(rms, np.ones(5) / 5, mode="same")
        peaks, _ = find_peaks(smoothed, height=thresh, distance=3)
        speaking_rate = float(len(peaks) / duration) if duration > 0 else 0.0

        pause_count, cur_run = 0, 0
        for v in voiced_frames:
            if not v:
                cur_run += 1
            else:
                if cur_run > 0:
                    pause_count += 1
                cur_run = 0
        if cur_run > 0:
            pause_count += 1
        pause_ratio = float(1.0 - speech_ratio)

        return {
            "pitch_mean": pitch_mean, "pitch_std": pitch_std,
            "energy_mean": energy_mean, "energy_std": energy_std,
            "speech_ratio": speech_ratio, "speaking_rate": speaking_rate,
            "pause_count": float(pause_count), "pause_ratio": pause_ratio,
        }

    def predict_from_wav_path(self, wav_path):
        """Synchronous, CPU/GPU-bound. Call via run_in_executor from async code."""
        if not self._loaded:
            self.load()

        import soundfile as sf
        import librosa

        wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        if wav.size == 0 or not np.isfinite(wav).all():
            raise ValueError(f"Invalid audio: {wav_path}")
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak

        emb = self._extract_embedding(wav_path)
        emb_t = torch.from_numpy(emb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emo_logits, aro, val = self.stage1_heads(emb_t)
            probs = torch.softmax(emo_logits, dim=-1).cpu().numpy()[0]

        acoustic = self._acoustic_features(wav, sr)
        feat_row = {f"emotion_{c}": float(p) for c, p in zip(self.emotion_classes, probs)}
        feat_row["arousal"] = float(aro.cpu().numpy()[0])
        feat_row["valence"] = float(val.cpu().numpy()[0])
        feat_row.update(acoustic)

        import pandas as pd
        x = pd.DataFrame([{c: feat_row[c] for c in FEATURE_COLS}])
        result = {t: float(self.stage2_models[t].predict(x)[0]) for t in CX_TARGETS}
        pred_idx = int(probs.argmax())
        result["emotion"] = self.emotion_classes[pred_idx]
        result["arousal"] = feat_row["arousal"]
        result["valence"] = feat_row["valence"]
        return result


_instance = None


def get_instance():
    global _instance
    if _instance is None:
        _instance = VoiceCXModel()
    return _instance
