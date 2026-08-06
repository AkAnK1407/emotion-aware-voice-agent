"""
dashboard_bridge.py — AdaptiveCX Dashboard WebSocket + HTTP Server
Runs alongside the LiveKit agent and broadcasts real-time signals to the browser.
Browser connects to the dashboard WebSocket to get live events.

This module also serves two plain-HTTP endpoints on the SAME port, via the
`process_request` hook (so a single Render/cloud port covers everything the
frontend needs — no separate token server or health-check process required):
  • GET /health, GET /  (non-upgrade)  → 200 OK, for platform health checks
  • GET /token?room=&identity=          → mints a LiveKit JWT (was token_server.py)
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Set
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.asyncio.server import ServerConnection

from emotion_engine import BehaviorSignals
from policy_engine import Policy

logger = logging.getLogger("dashboard_bridge")

# Global set of connected browser clients
_connected_clients: Set[ServerConnection] = set()

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_token(room: str, identity: str) -> str:
    """Mint a minimal LiveKit JWT token (same scheme as the old token_server.py)."""
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({
        "iss": LIVEKIT_API_KEY,
        "sub": identity,
        "iat": now,
        "exp": now + 3600,
        "video": {
            "roomJoin": True,
            "room": room,
            "canPublish": True,
            "canSubscribe": True,
        },
        "metadata": "",
        "name": identity,
    }).encode())
    sig_input = f"{header}.{payload}".encode()
    sig = hmac.new(LIVEKIT_API_SECRET.encode(), sig_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url(sig)}"


def _process_request(connection: ServerConnection, request):
    """Intercept plain-HTTP requests before the WebSocket handshake.

    Returning a Response here short-circuits the upgrade; returning None lets
    the normal WebSocket handshake (and `_handler` below) proceed. Only real
    `Upgrade: websocket` requests are allowed through for "/", so Render's
    plain-HTTP health check on "/" gets a normal 200 instead of a failed
    handshake.
    """
    parsed = urlparse(request.path)
    is_ws_upgrade = (request.headers.get("Upgrade", "") or "").lower() == "websocket"

    if parsed.path == "/token":
        qs = parse_qs(parsed.query)
        room = qs.get("room", ["adaptivecx-demo-room"])[0]
        identity = qs.get("identity", ["dashboard-user"])[0]
        body = json.dumps({"token": _make_token(room, identity)})
        response = connection.respond(200, body)
        response.headers["Content-Type"] = "application/json"
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    if not is_ws_upgrade and parsed.path in ("/", "/health"):
        return connection.respond(200, "OK")

    return None  # proceed with WebSocket handshake


async def _handler(websocket: ServerConnection):
    """Handle new browser WebSocket connections.

    NOTE: as of `websockets` 11+ (this project uses 14.x), server connection
    handlers take a single `websocket` argument — the older `(websocket, path)`
    two-argument signature was removed. Using the old signature makes every
    connection crash with `TypeError: missing 1 required positional argument:
    'path'`, which looks from the browser like the agent server keeps
    disconnecting.
    """
    _connected_clients.add(websocket)
    logger.info(f"Dashboard client connected. Total: {len(_connected_clients)}")
    try:
        # Send welcome event
        await websocket.send(json.dumps({
            "type": "connected",
            "message": "AdaptiveCX Dashboard connected. Waiting for voice input...",
        }))
        # Keep connection alive, wait for disconnect
        await websocket.wait_closed()
    finally:
        _connected_clients.discard(websocket)
        logger.info(f"Dashboard client disconnected. Total: {len(_connected_clients)}")


async def broadcast(event: dict):
    """Broadcast an event dict to all connected browser clients."""
    if not _connected_clients:
        return
    message = json.dumps(event)
    dead = set()
    for ws in list(_connected_clients):
        try:
            await ws.send(message)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _connected_clients.discard(ws)


async def broadcast_transcript(text: str, is_partial: bool = False, speaker: str = "customer"):
    """Send transcript update to dashboard."""
    await broadcast({
        "type": "transcript",
        "speaker": speaker,
        "text": text,
        "is_partial": is_partial,
    })


async def broadcast_behavior(behavior: BehaviorSignals, policy: Policy):
    """Send full behavior + policy update to dashboard."""
    await broadcast({
        "type": "behavior",
        "emotion": behavior.emotion.value,
        "emotion_confidence": behavior.emotion_confidence,
        "emotion_color": behavior.emotion_color,
        "emotion_emoji": behavior.emotion_emoji,
        "stress": behavior.stress,
        "engagement": behavior.engagement,
        "trust": behavior.trust,
        "urgency": behavior.urgency,
        "conversation_score": behavior.conversation_score,
        "conversation_trend": behavior.conversation_trend,
        "conversation_label": behavior.conversation_label,
        "patience": behavior.patience,
        "prosody": {
            "speech_rate_estimate": behavior.prosody.speech_rate_estimate,
            "caps_ratio": behavior.prosody.caps_ratio,
            "exclamation_count": behavior.prosody.exclamation_count,
            "word_count": behavior.prosody.word_count,
            "repetition_score": behavior.prosody.repetition_score,
        },
        "policy": {
            "name": policy.name,
            "empathy_level": policy.empathy_level,
            "speaking_speed": policy.speaking_speed,
            "response_length": policy.response_length,
            "offer_escalation": policy.offer_escalation,
            "voice_style": policy.voice_style,
            "color": policy.color,
            "description": policy.description,
        },
    })


async def broadcast_agent_response(text: str):
    """Send agent's generated response text to dashboard."""
    await broadcast({
        "type": "agent_response",
        "text": text,
    })


async def broadcast_tts_params(params: dict):
    """Send TTS parameter values to dashboard for visualization."""
    await broadcast({
        "type": "tts_params",
        **params,
    })


async def broadcast_tool_call(tool_name: str, arguments: dict, result: str):
    """Send a completed agentic tool call (CRM/refund/ticket/etc.) to the dashboard."""
    await broadcast({
        "type": "tool_call",
        "tool_name": tool_name,
        "arguments": arguments,
        "result": result,
    })


async def broadcast_guardrail_event(checkpoint: str, result: dict):
    """Send a guardrail check result (input or output checkpoint) to the dashboard."""
    await broadcast({
        "type": "guardrail",
        "checkpoint": checkpoint,   # "input" | "output"
        **result,
    })


async def broadcast_observability(metric_event: dict):
    """Send one real STT/LLM/TTS latency+cost metric event to the dashboard."""
    await broadcast({
        "type": "observability",
        **metric_event,
    })


async def broadcast_observability_totals(totals: dict):
    """Send the running session totals (tokens, cost, latency) to the dashboard."""
    await broadcast({
        "type": "observability_totals",
        **totals,
    })


async def broadcast_evaluation(evaluation: dict):
    """Send the per-turn evaluation result (resolution/CSAT/compliance/hallucination) to the dashboard."""
    await broadcast({
        "type": "evaluation",
        **evaluation,
    })


async def broadcast_knowledge(knowledge: dict):
    """Send the knowledge-base retrieval result (or knowledge-gap flag) to the dashboard."""
    await broadcast({
        "type": "knowledge",
        **knowledge,
    })


_server_started = False


async def start_server(port: int = 8765, host: str = "0.0.0.0"):
    """Start the combined WebSocket + HTTP (health/token) server. Call as an asyncio task."""
    global _server_started
    if _server_started:
        logger.info(f"Dashboard server already running on port {port}")
        return
    _server_started = True
    logger.info(f"Dashboard WS + /token + /health server starting on {host}:{port}")
    try:
        async with websockets.serve(_handler, host, port, process_request=_process_request):
            await asyncio.Future()   # run forever
    except Exception as e:
        _server_started = False
        logger.warning(f"Dashboard WebSocket server error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(start_server())
