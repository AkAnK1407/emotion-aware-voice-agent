# AdaptiveCX — System Architecture Reference

Emotionally intelligent, voice-first AI customer support agent for a bank. This document describes the full system topology, data flow, and component responsibilities in enough detail to draw a complete architecture diagram from scratch.

## 1. High-level shape

Three independent processes, one third-party real-time transport, one external GPU inference service, all fronted by ad-hoc public tunnels for demo purposes:

1. **Frontend** — static HTML/JS dashboard, served from a plain HTTP file server.
2. **Agent process** — a LiveKit Agents worker (Python). Splits into a long-lived main process and a per-call subprocess (job).
3. **Auth server** — a small FastAPI service for login/signup/history/transactions, backed by SQLite.
4. **LiveKit Cloud** — third-party WebRTC SFU. Browser and agent both connect to it as room participants; audio/voice never touches the app's own servers directly.
5. **Voice-CX server** — a separate FastAPI service running a GPU speech-emotion model (emotion2vec+ → XGBoost), hosted on a Kaggle GPU notebook and reached over an ngrok tunnel. Shadow-mode only — never drives the live conversation.

For local demo runs, three Cloudflare quick tunnels (`trycloudflare.com`) expose the frontend, the agent's WebSocket/HTTP port, and the auth server to the public internet, since quick tunnels are regenerated (new random hostname) on every restart.

## 2. Components and ports

| Component | File(s) | Port (local) | Role |
|---|---|---|---|
| Frontend static server | `frontend/index.html`, `app.js`, `style.css` | 5500 | Serves the dashboard SPA |
| Agent worker (main process) | `agent/agent.py` | — (outbound only, to LiveKit Cloud) | Registers with LiveKit Cloud, spawns a job subprocess per call |
| Dashboard bridge | `agent/dashboard_bridge.py` | 8765 (public), 8790 (localhost-only) | WebSocket to browser (live event stream) + HTTP `/token`, `/health` |
| Auth server | `agent/auth_server.py` | 8766 | REST API: signup, login, session, chat history, transactions |
| Storage | `agent/storage.py` | — | SQLite (`adaptivecx.db`, WAL mode), used by both the auth server and the agent job subprocess |
| Voice-CX server | `voice-cx-server/main.py`, `voice_cx_model.py` | 8000 (on Kaggle), tunneled via ngrok | GPU inference: speech → stress/frustration/urgency/escalation_risk |
| LiveKit Cloud | external SaaS | — | WebRTC room/SFU; carries the actual audio |
| Gemini | external API (`livekit.plugins.google`) | — | LLM turn generation, with native function-calling |
| Deepgram | external API (`livekit.plugins.deepgram`) | — | Streaming STT |
| Cartesia | external API (`livekit.plugins.cartesia`) | — | TTS, voice style parameterized per policy |
| Silero VAD | local model (`livekit.plugins.silero`) | — | Voice activity detection |

## 3. Process topology detail

`agent.py`'s `if __name__ == "__main__"` block starts, in the **main process**:
- `dashboard_bridge`'s WebSocket+HTTP server (port 8765) — this is what the browser connects to for both the live event stream and its LiveKit join token.
- `auth_server`'s FastAPI app (port 8766), in its own thread.
- The LiveKit Agents worker loop, which registers with LiveKit Cloud and waits for jobs.

Each incoming call is a **separate OS subprocess** (`entrypoint(ctx)` in `agent.py`), because livekit-agents runs every job isolated. This subprocess:
- Cannot touch the main process's in-memory WebSocket client list directly, so `dashboard_bridge.broadcast_*()` calls forward events over a **localhost TCP bridge** (127.0.0.1:8790) back to the main process, which is the only place that ever touches `_connected_clients`.
- Resolves which logged-in user (if any) joined, via the LiveKit participant identity (`user-<id>` convention) — see §6.
- Loads its own copy of `tools.py`'s mock CRM state, scoped to that call.

## 4. Per-turn processing pipeline

Everything below runs once per customer utterance, inside the job subprocess:

1. **STT** — Deepgram streams partial + final transcripts. `agent.py`'s `stt_node` override taps the raw audio frames on the way through, so the same audio that produced the transcript is also available for the Voice-CX path below — no duplicate recording.
2. **Input guardrail** — `guardrails.check_input()` scans the transcript for spoken PII (card numbers, SSNs) and prompt-injection patterns, before anything else sees it.
3. **Emotion detection (dual-path, racing)**:
   - **Text path (always runs, low latency):** `emotion_engine.py`'s `EmotionEngine` — keyword-softmax scoring over the transcript — produces `BehaviorSignals` (emotion, stress, trust, urgency, patience, prosody features, a recency-weighted session conversation score/trend).
   - **Voice path (parallel, shadow/validating):** `voice_cx_client.py` sends this turn's raw audio (WAV) to the Voice-CX server's `/predict` endpoint. A deadline (~3.5s, scaled by audio length) races this against the text path; if it returns in time its signal is treated as the validated primary source and shown as "🎙️ VOICE (validated)" on the dashboard, otherwise the pipeline silently falls back to the text-only result ("💬 TEXT (fallback)") — the voice call never blocks the conversation. See `VOICE_CX_RACING.md` for the full state machine, including the manual force-text-fallback toggle.
4. **Knowledge retrieval** — `knowledge_base.py` does keyword-overlap FAQ matching; a miss is logged as a "knowledge gap" signal.
5. **Policy selection** — `policy_engine.py` scores five candidate policies (`HIGH_EMPATHY`, `CALM`, `BALANCED`, `EFFICIENT`, `ESCALATE`) via a weighted formula over the behavior signals: `Score(P) = 0.35·emotion + 0.30·stress + 0.15·engagement + 0.15·trust + 0.05·urgency`, plus a hard override to `ESCALATE` on extreme stress + low trust. Each policy sets empathy level, speaking speed, target response length, and TTS voice style.
6. **Prompt rebuild** — `build_system_prompt()` in `agent.py` dynamically rewrites Gemini's system instructions every turn: emotion-specific tone instruction, policy-mode instruction, response-length cap, injected KB grounding text, a guardrail notice if flagged, and (for a logged-in customer) a note that identity is already verified and to greet them by name.
7. **LLM generation** — Gemini (`livekit.plugins.google`), with native function-calling against `tools.py`'s `BANKING_TOOLS` (see §5).
8. **Output guardrail + evaluation** — `tts_node` buffers the full reply, runs `guardrails.check_output()` (PII the model is about to say out loud), then `evaluation.py` scores the turn (resolution signal, predicted CSAT, policy compliance, hallucination flag) using only rule-based signals already computed above — no second LLM "judge" call.
9. **TTS** — Cartesia synthesizes speech, voice style/speed set by the selected policy.
10. **Telemetry** — `observability.py` listens to livekit-agents' native `metrics_collected` events (real STT/LLM/TTS/VAD latency, real Gemini token counts) and computes an estimated cost from a small per-unit rate table.
11. **Broadcast** — every stage above pushes a typed event (`transcript`, `behavior`, `tool_call`, `guardrail`, `knowledge`, `evaluation`, `observability`, `voice_cx`) through `dashboard_bridge` to the connected browser, which drives the dashboard UI live, turn by turn.

## 5. Agentic tools (Gemini function-calling)

Defined in `agent/tools.py`, backed by a per-user mock CRM (`banking_data.py` generates a deterministic account — tier, balance, 5 transactions — seeded from the user's numeric ID, so the same account regenerates identically with no shared state or DB table needed):

- `verify_identity(full_name, date_of_birth)` — manual fallback for guests; logged-in users skip this (see §6).
- `lookup_customer_profile()` — account tier/balance/open tickets.
- `check_recent_transactions()` — the 5 mock transactions, flags duplicates.
- `process_refund(transaction_id, reason)` — marks a transaction refunded.
- `create_support_ticket(summary, priority)` — logs a ticket.
- `escalate_to_specialist(summary, reason)` — used when the issue can't be resolved on the call or the customer stays unsatisfied; creates a priority ticket **and** returns a generated callback meeting link + time slot, which the agent is instructed to relay back to the customer verbatim.

## 6. Identity, personalization, and persistence

SQLite (`adaptivecx.db`, WAL mode) via `storage.py`, shared by the auth server and every agent job subprocess:

- `users` — username, salted+hashed password (PBKDF2), display name.
- `sessions` — bearer tokens, 7-day TTL.
- `verified_identities` — full name + date of birth, keyed by user id.
- `chat_turns` — full transcript history per user, browsable in the dashboard's History panel.

**Key flow:** at signup, the frontend collects full name + date of birth (an `<input type="date">`) alongside username/password. `auth_server.py` writes this straight into `verified_identities` — signup *is* identity verification for this demo. When that user later joins a LiveKit room, their participant identity is set to `user-<id>` (set by the frontend at token request time); `agent.py`'s `_resolve_session_user()` reads that back out of the room, and `tools.set_current_user(user_id)` hydrates the mock CRM from the saved identity — so the agent already knows their name and never asks them to re-verify. Guest sessions (no login) get an unverified, unpersonalized demo customer identity instead, and nothing is written to the DB.

The dashboard also exposes a **Transactions panel** (`GET /transactions` on the auth server) and a **History panel** (`GET /history`) so a logged-in customer can review their own account and past conversations outside of a live call — both reuse the exact same deterministic account-generation function the agent uses mid-call, so the numbers always match.

## 7. Frontend dashboard

Single-page app (`frontend/index.html` + `app.js`), connects to:
- LiveKit Cloud directly (WebRTC, via the LiveKit JS client) for audio in/out.
- The dashboard WebSocket (`dashboard_bridge.py`) for every live event listed in §4 step 11.
- The auth server (REST) for login/signup/history/transactions.

UI surfaces, left-to-right/top-to-bottom:
- **Emotion intelligence panel** — detected emotion, stress/trust/urgency meters, patience, a recency-weighted session conversation-state trend.
- **Pipeline status** — a real (event-driven, not simulated) view of each turn moving through STT → Voice-CX → Behavior Engine → Policy Engine → Gemini LLM → Cartesia TTS.
- **Emotion timeline** — a running strip chart of detected emotion over the session.
- **Problem statement / product framing card.**
- **Enterprise panel** — agentic tool call feed, guardrail badges (input/output), knowledge retrieval result, per-turn evaluation scores, and an observability card (real STT audio duration, LLM TTFT/duration/tokens, TTS TTFB/duration, running session cost).
- **Auth overlay** — login/signup (guest mode always available), History panel, Transactions panel.

## 8. Deployment / networking (demo mode)

`start-demo.ps1` orchestrates a full local run:
1. Checks the remote Voice-CX server's `/health` (URL from `.env`'s `VOICE_CX_SERVER_URL`, currently a Kaggle GPU notebook tunneled via ngrok).
2. Starts the agent worker (`agent.py dev`) and the frontend static server (port 5500).
3. Opens three Cloudflare quick tunnels (frontend, dashboard bridge, auth server) and patches the resulting hostnames into `frontend/app.js`'s `BACKEND_HOST`/`AUTH_HOST` constants, since quick-tunnel hostnames are randomly regenerated every run.

## 9. Why the Voice-CX path is a separate GPU service

The base LiveKit agent process runs comfortably on CPU (STT/LLM/TTS are all hosted APIs), but the *speech-emotion* model (emotion2vec+ backbone + trained heads + XGBoost regressors) needs real inference compute, and running it on the same low-RAM machine as the agent was too slow to stay inside the 3.5s racing deadline. Moving it to a Kaggle GPU notebook (see `voice-cx-server/kaggle_notebook.ipynb`) keeps it fast without needing a paid GPU box, at the cost of the server being ephemeral (session timeouts, new ngrok URL on restart) — acceptable for a shadow-mode, non-critical signal.
