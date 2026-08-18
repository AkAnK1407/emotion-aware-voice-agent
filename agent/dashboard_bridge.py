"""
dashboard_bridge.py — AdaptiveCX Dashboard WebSocket + HTTP Server
Runs alongside the LiveKit agent and broadcasts real-time signals to the browser.
Browser connects to the dashboard WebSocket to get live events.

This module also serves two plain-HTTP endpoints on the SAME port, via the
`process_request` hook (so a single Render/cloud port covers everything the
frontend needs — no separate token server or health-check process required):
  • GET /health, GET /  (non-upgrade)  → 200 OK, for platform health checks
  • GET /token?room=&identity=          → mints a LiveKit JWT (was token_server.py)

Process topology note: `start_server()` is started ONCE, at worker boot,
before `cli.run_app()` (see the bottom of agent.py) — NOT inside `entrypoint()`.
It must be reachable before any room/job exists, since a browser needs /token
to even join the room that would trigger a job. But livekit-agents runs each
job's `entrypoint_fnc` in its own subprocess, so code inside entrypoint() can't
touch this module's `_connected_clients` directly — it lives in a different
process. `broadcast()` therefore always forwards events over a localhost TCP
bridge to the process actually running the WS server, which is the only place
that ever touches `_connected_clients`. This works the same way whether the
caller happens to be in-process or in a subprocess, so it's safe everywhere.
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

# Localhost-only port used to relay broadcast events from a job subprocess
# back into the process that owns the real WebSocket connections. Never
# exposed externally — only the public WS/HTTP port above is.
_BRIDGE_PORT = int(os.getenv("DASHBOARD_BRIDGE_PORT", "8790"))


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_token(room: str, identity: str) -> str:
    """Mint a minimal LiveKit JWT token (same scheme as the old token_server.py).

    Reads the API key/secret from the environment at call-time, not at import
    time — this module is imported (and its top-level code executed) by
    agent.py *before* agent.py's own `load_dotenv()` call runs, so caching
    these in module-level constants would always see empty values.
    """
    api_key = os.getenv("LIVEKIT_API_KEY", "")
    api_secret = os.getenv("LIVEKIT_API_SECRET", "")
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({
        "iss": api_key,
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
    sig = hmac.new(api_secret.encode(), sig_input, hashlib.sha256).digest()
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


async def _local_broadcast(event: dict):
    """Actually pushes to connected WS clients. Only ever called from within
    the process running start_server() — via the WS handler or the bridge
    listener below, both on that process's single event loop."""
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


async def broadcast(event: dict):
    """Broadcast an event dict to all connected browser clients.

    Always forwards over the localhost TCP bridge, even if called from within
    the same process that owns `_connected_clients` — that keeps this function
    correct regardless of whether the caller is agent.py's job subprocess or
    (in a future refactor) the same process, without needing to detect which.
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", _BRIDGE_PORT), timeout=2.0
        )
        writer.write((json.dumps(event) + "\n").encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    except Exception as e:
        logger.warning(f"[Bridge] failed to forward event to dashboard server: {e}")


async def _bridge_tcp_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Receives events forwarded by `broadcast()` (possibly from a different
    process) and pushes them to the real WS clients on this process's loop."""
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                event = json.loads(line.decode())
            except Exception as e:
                logger.warning(f"[Bridge] malformed event: {e}")
                continue
            await _local_broadcast(event)
    finally:
        writer.close()


async def broadcast_transcript(text: str, is_partial: bool = False, speaker: str = "customer"):
    """Send transcript update to dashboard."""
    await broadcast({
        "type": "transcript",
        "speaker": speaker,
        "text": text,
        "is_partial": is_partial,
    })


async def broadcast_behavior(behavior: BehaviorSignals, policy: Policy, source: str = "text"):
    """Send full behavior + policy update to dashboard.

    source: "voice" (Stage 1, validated) or "text" (formula fallback) --
    which one actually drove this turn's policy decision.
    """
    await broadcast({
        "type": "behavior",
        "source": source,
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


async def broadcast_voice_cx(result: dict):
    """Send the voice-based (Stage 1 + Stage 2) customer-state estimate to the
    dashboard. Shadow mode: for side-by-side comparison only -- does not
    drive the agent's actual response strategy."""
    await broadcast({
        "type": "voice_cx",
        **result,
    })


async def broadcast_knowledge(knowledge: dict):
    """Send the knowledge-base retrieval result (or knowledge-gap flag) to the dashboard."""
    await broadcast({
        "type": "knowledge",
        **knowledge,
    })


_server_started = False


async def start_server(port: int = 8765, host: str = "0.0.0.0"):
    """Start the combined WebSocket + HTTP (health/token) server, plus the
    internal localhost bridge listener that `broadcast()` calls target.
    Must be started once, at worker process boot — see agent.py's
    `if __name__ == "__main__"` block, not inside entrypoint()."""
    global _server_started
    if _server_started:
        logger.info(f"Dashboard server already running on port {port}")
        return
    _server_started = True
    logger.info(f"Dashboard WS + /token + /health server starting on {host}:{port}")
    try:
        bridge_server = await asyncio.start_server(_bridge_tcp_handler, "127.0.0.1", _BRIDGE_PORT)
        logger.info(f"Dashboard bridge listener starting on 127.0.0.1:{_BRIDGE_PORT}")
        async with bridge_server, websockets.serve(_handler, host, port, process_request=_process_request):
            await asyncio.Future()   # run forever
    except Exception as e:
        _server_started = False
        logger.warning(f"Dashboard server error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(start_server())
