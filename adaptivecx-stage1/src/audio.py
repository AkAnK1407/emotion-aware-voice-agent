"""Audio loading, resampling, normalization, and inspection.

Ported from the verified Kaggle pipeline (AdaptiveCX_Stage1_Kaggle.ipynb,
Section 5). Pauses are intentionally NOT stripped -- prosody (including
silence) is a signal for emotion, not noise to remove.
"""
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000


def load_and_preprocess_audio(path, target_sr=TARGET_SR):
    """Mono, resampled to target_sr, peak-normalized. Raises on empty/NaN audio."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if wav.size == 0:
        raise ValueError(f"Empty audio: {path}")
    if not np.isfinite(wav).all():
        raise ValueError(f"NaN/Inf samples in audio: {path}")
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak
    return wav, sr


def inspect_audio(path, verbose=True):
    """Returns {sampling_rate, channels, duration, n_samples} for the raw file on disk."""
    raw_wav, raw_sr = sf.read(path, always_2d=False)
    channels = 1 if raw_wav.ndim == 1 else raw_wav.shape[1]
    duration = len(raw_wav) / raw_sr
    info = {
        "sampling_rate": raw_sr,
        "channels": channels,
        "duration": duration,
        "n_samples": len(raw_wav),
    }
    if verbose:
        print(f"Sampling rate    : {info['sampling_rate']}")
        print(f"Channels         : {info['channels']}")
        print(f"Duration         : {info['duration']:.3f}s")
        print(f"Number of samples: {info['n_samples']}")
    return info
