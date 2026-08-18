"""
voice_cx_client.py — thin async HTTP client for the remote Voice CX server
(voice-cx-server/, deployed separately on a machine with enough RAM to run
torch/funasr reliably). The live agent process stays lightweight and never
loads those heavy models itself.

Shadow mode: results feed the dashboard's experimental "Voice-Based CX"
panel only. They never drive the agent's actual response -- that stays on
the existing, faster, already-proven text-based emotion_engine.py /
policy_engine.py.

Set VOICE_CX_SERVER_URL in .env (e.g. http://<ec2-ip>:8000) to enable this.
If unset, callers should skip calling predict_from_frames entirely -- see
agent.py's use of `voice_cx_client.is_configured()`.
"""
import io
import logging
import os
import wave

import httpx

logger = logging.getLogger("voice_cx_client")


def is_configured() -> bool:
    return bool(os.getenv("VOICE_CX_SERVER_URL", "").strip())


def estimate_timeout(frames, base: float = 3.0, factor: float = 1.0,
                      floor: float = 8.0, ceiling: float = 20.0) -> float:
    """How long to wait for this turn's /predict call, scaled to how much
    audio there actually is -- a fixed timeout was the bug: the EC2 server
    runs inference at roughly realtime (plus SSH-tunnel network overhead),
    so a fixed 12s cap silently discarded every turn longer than ~9-10s of
    speech and fell back to text every time, even though the server itself
    was healthy and responding. `base` covers upload + tunnel round-trip
    for a near-empty clip; `factor` is seconds of wait per second of audio.
    """
    duration = sum(getattr(f, "duration", 0.0) for f in (frames or []))
    return min(max(base + duration * factor, floor), ceiling)


def _frames_to_wav_bytes(frames) -> bytes | None:
    if not frames:
        return None
    sample_rate = frames[0].sample_rate
    num_channels = frames[0].num_channels
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # int16 PCM, LiveKit's AudioFrame format
        wf.setframerate(sample_rate)
        for f in frames:
            wf.writeframes(bytes(f.data))
    buf.seek(0)
    return buf.read()


async def predict_from_frames(frames, timeout: float = 20.0) -> dict:
    """Sends this turn's raw audio to the remote Voice CX server and returns
    its JSON result. Raises on any failure (unreachable server, timeout,
    non-2xx, no audio) -- callers must catch and treat as non-fatal, since
    this is shadow mode and must never affect the live conversation."""
    server_url = os.getenv("VOICE_CX_SERVER_URL", "").strip().rstrip("/")
    if not server_url:
        raise RuntimeError("VOICE_CX_SERVER_URL not set")

    wav_bytes = _frames_to_wav_bytes(frames)
    if wav_bytes is None:
        raise ValueError("no audio frames captured for this turn")

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{server_url}/predict",
            files={"audio": ("turn.wav", wav_bytes, "audio/wav")},
        )
        resp.raise_for_status()
        return resp.json()
