"""
auth_server.py — AdaptiveCX Login / Signup / History API

Why this is a separate service from dashboard_bridge.py's existing
/token + /health endpoints: that server is built directly on the
`websockets` library's WebSocket-handshake parser (see `_process_request`
in dashboard_bridge.py), which is hard-coded to accept only a bodyless GET
-- `websockets/http11.py` raises ValueError on any other method or on a
Content-Length header at all. Login can't go through that without putting
passwords in a query string (which Cloudflare/any proxy in front of it
would log in plaintext -- a real leak, not just bad style), so it needs a
real HTTP framework. FastAPI is already a dependency here (agent/tools.py's
runtime pulls it in transitively; also the exact pattern voice-cx-server/
already uses for its own small service).

Runs on its own port (AUTH_PORT, default 8766) in its own background
thread, started from agent.py's `if __name__ == "__main__":` block
alongside the existing dashboard bridge thread.
"""

import asyncio
import logging
import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import storage

logger = logging.getLogger("auth-server")

app = FastAPI(title="AdaptiveCX Auth Server")

# Open CORS -- this is a demo reachable only via an ad-hoc Cloudflare quick
# tunnel URL shared for a single session, not a standing public API; matches
# the existing /token endpoint's Access-Control-Allow-Origin: * in
# dashboard_bridge.py.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignupRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""


class LoginRequest(BaseModel):
    username: str
    password: str


async def _auth_response(user_id: int, display_name: str) -> dict:
    token = await asyncio.to_thread(storage.create_session, user_id)
    verified = await asyncio.to_thread(storage.get_verified_identity, user_id)
    return {
        "token": token,
        "user_id": user_id,
        "display_name": display_name,
        "already_verified": verified is not None,
    }


@app.post("/signup")
async def signup(body: SignupRequest):
    if not body.username.strip() or not body.password:
        raise HTTPException(status_code=400, detail="Username and password are required.")
    if len(body.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters.")
    result = await asyncio.to_thread(storage.create_user, body.username, body.password, body.display_name)
    if result is None:
        raise HTTPException(status_code=409, detail="That username is already taken.")
    logger.info(f"[Auth] signup: {body.username}")
    return await _auth_response(result.user_id, result.display_name)


@app.post("/login")
async def login(body: LoginRequest):
    result = await asyncio.to_thread(storage.authenticate, body.username, body.password)
    if result is None:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    logger.info(f"[Auth] login: {body.username}")
    return await _auth_response(result.user_id, result.display_name)


async def _require_user(authorization: str | None) -> int:
    token = (authorization or "").removeprefix("Bearer ").strip()
    user_id = await asyncio.to_thread(storage.resolve_session, token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return user_id


@app.get("/me")
async def me(authorization: str | None = Header(default=None)):
    user_id = await _require_user(authorization)
    display_name, verified = await asyncio.gather(
        asyncio.to_thread(storage.get_display_name, user_id),
        asyncio.to_thread(storage.get_verified_identity, user_id),
    )
    return {"user_id": user_id, "display_name": display_name, "verified": verified is not None}


@app.get("/history")
async def history(authorization: str | None = Header(default=None)):
    user_id = await _require_user(authorization)
    turns = await asyncio.to_thread(storage.get_chat_history, user_id)
    return {"turns": turns}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    storage.init_db()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("AUTH_PORT", "8766")))
