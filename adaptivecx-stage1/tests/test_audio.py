import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio import inspect_audio, load_and_preprocess_audio


def _write_tone(path, sr=8000, duration=0.5, channels=1):
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    if channels == 2:
        tone = np.stack([tone, tone], axis=1)
    sf.write(path, tone, sr)


def test_resamples_to_16k_and_mono(tmp_path):
    wav_path = tmp_path / "stereo_8k.wav"
    _write_tone(str(wav_path), sr=8000, channels=2)

    wav, sr = load_and_preprocess_audio(str(wav_path))
    assert sr == 16000
    assert wav.ndim == 1


def test_peak_normalized(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_tone(str(wav_path))

    wav, _ = load_and_preprocess_audio(str(wav_path))
    assert np.isclose(np.abs(wav).max(), 1.0, atol=1e-5)


def test_empty_audio_raises(tmp_path):
    wav_path = tmp_path / "empty.wav"
    sf.write(str(wav_path), np.zeros((0,), dtype=np.float32), 16000)

    with pytest.raises(ValueError):
        load_and_preprocess_audio(str(wav_path))


def test_inspect_audio_reports_duration(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_tone(str(wav_path), sr=16000, duration=1.0)

    info = inspect_audio(str(wav_path), verbose=False)
    assert info["sampling_rate"] == 16000
    assert info["channels"] == 1
    assert np.isclose(info["duration"], 1.0, atol=0.01)
    assert info["n_samples"] == 16000
