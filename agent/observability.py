"""
observability.py — AdaptiveCX Observability Layer

Nothing here is simulated. `livekit-agents` already emits a real
`metrics_collected` event off the AgentSession for every STT / LLM / TTS /
VAD call, carrying the actual wall-clock latency and (for the LLM) actual token
counts reported by the Gemini API. This module just listens for that event,
turns each metric into a dashboard-friendly trace row, and keeps a running
session total — the same shape as a production APM's per-request + aggregate
view (Datadog/Honeycomb-style), scaled down to one process.

Cost figures are a small, clearly-labeled per-unit rate table (`_RATES`) applied
to the real token/character/audio-second counts — they are estimates of spend,
not simulated latency.
"""

import time
from dataclasses import dataclass

from livekit.agents.metrics import (
    LLMMetrics,
    STTMetrics,
    TTSMetrics,
    VADMetrics,
)

import dashboard_bridge

# ─── Illustrative Per-Unit Cost Rates ────────────────────────────────────────────
# Approximate public list prices at time of writing. Swap these for your actual
# negotiated rates — the point is that cost is *computed from real usage*, not
# hardcoded per turn.
_RATES = {
    "gemini_input_per_1k_tokens": 0.000075,    # Gemini Flash-tier, input
    "gemini_output_per_1k_tokens": 0.00030,    # Gemini Flash-tier, output
    "deepgram_per_audio_second": 0.0000433,    # Nova-2 streaming, ~$0.0026/min
    "cartesia_per_character": 0.00003,         # Sonic TTS, illustrative
}


@dataclass
class SessionTotals:
    turns: int = 0
    stt_seconds: float = 0.0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    tts_characters: int = 0
    estimated_cost_usd: float = 0.0


_totals = SessionTotals()


def _reset_session():
    global _totals
    _totals = SessionTotals()


async def handle_metrics(metrics) -> None:
    """Dispatch a `metrics_collected` event payload from VoicePipelineAgent."""
    now = time.time()

    if isinstance(metrics, STTMetrics):
        cost = metrics.audio_duration * _RATES["deepgram_per_audio_second"]
        _totals.stt_seconds += metrics.audio_duration
        _totals.estimated_cost_usd += cost
        await dashboard_bridge.broadcast_observability({
            "stage": "stt",
            "latency_ms": round(metrics.duration * 1000, 1),
            "audio_duration_s": round(metrics.audio_duration, 2),
            "estimated_cost_usd": round(cost, 6),
            "timestamp": now,
        })

    elif isinstance(metrics, LLMMetrics):
        cost = (
            (metrics.prompt_tokens / 1000) * _RATES["gemini_input_per_1k_tokens"]
            + (metrics.completion_tokens / 1000) * _RATES["gemini_output_per_1k_tokens"]
        )
        _totals.llm_prompt_tokens += metrics.prompt_tokens
        _totals.llm_completion_tokens += metrics.completion_tokens
        _totals.estimated_cost_usd += cost
        await dashboard_bridge.broadcast_observability({
            "stage": "llm",
            "ttft_ms": round(metrics.ttft * 1000, 1),
            "duration_ms": round(metrics.duration * 1000, 1),
            "prompt_tokens": metrics.prompt_tokens,
            "completion_tokens": metrics.completion_tokens,
            "total_tokens": metrics.total_tokens,
            "tokens_per_second": round(metrics.tokens_per_second, 1),
            "estimated_cost_usd": round(cost, 6),
            "timestamp": now,
        })

    elif isinstance(metrics, TTSMetrics):
        cost = metrics.characters_count * _RATES["cartesia_per_character"]
        _totals.tts_characters += metrics.characters_count
        _totals.estimated_cost_usd += cost
        await dashboard_bridge.broadcast_observability({
            "stage": "tts",
            "ttfb_ms": round(metrics.ttfb * 1000, 1),
            "duration_ms": round(metrics.duration * 1000, 1),
            "characters_count": metrics.characters_count,
            "estimated_cost_usd": round(cost, 6),
            "timestamp": now,
        })

    elif isinstance(metrics, VADMetrics):
        # High-frequency, low-signal for a demo dashboard — track but don't broadcast every tick.
        return

    else:
        return

    await dashboard_bridge.broadcast_observability_totals({
        "turns": _totals.turns,
        "stt_seconds": round(_totals.stt_seconds, 2),
        "llm_prompt_tokens": _totals.llm_prompt_tokens,
        "llm_completion_tokens": _totals.llm_completion_tokens,
        "tts_characters": _totals.tts_characters,
        "estimated_cost_usd": round(_totals.estimated_cost_usd, 4),
    })


def mark_turn_complete():
    _totals.turns += 1
