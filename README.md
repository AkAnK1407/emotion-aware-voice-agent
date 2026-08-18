# AdaptiveCX — Emotional Voice Agent Demo

Traditional voice bots treat every customer the same — they don't care if
you're angry, sad, or happy. **AdaptiveCX** is an emotionally-intelligent
voice agent for banking support that:

- 🎤 **Listens** to your real voice (LiveKit WebRTC)
- 🧬 **Detects** emotion two ways per turn — fast text-based scoring, and a GPU
  acoustic model racing it for the "real" voice-based signal
- ⚖️ **Selects** a conversation policy (HIGH_EMPATHY / CALM / BALANCED / EFFICIENT / ESCALATE)
- 💬 **Generates** adaptive responses with Gemini — system prompt, tools, and
  tone all change based on detected emotion and who's calling
- 🔊 **Speaks back** with Cartesia TTS (tone/pace adapted to emotional state)
- 👤 **Remembers you** — log in once and the agent already knows your name,
  never re-asks who you are, and your account/transactions are real (mock) data
- 📈 **Escalates for real** — when it can't solve the problem, it books a
  callback meeting and hands you a link + time, not just "I'll pass it along"
- 📊 **Shows** everything live on a dashboard — emotion, policy, pipeline
  status, guardrails, knowledge retrieval, evaluation, cost/latency

---

## System Flow — what happens on every turn

```
Customer voice (browser mic)
        │
        ▼
   LiveKit WebRTC room
        │
        ▼
  Deepgram (STT, streaming) ─────────► transcript (partial + final)
        │                                    │
        │ (same audio tapped, unchanged)      │
        ▼                                    ▼
┌─────────────────────────┐   ┌───────────────────────────────────────────┐
│ Voice-CX server          │   │ agent/agent.py — on_user_turn_completed()  │
│ (Kaggle GPU, via ngrok)  │   │                                             │
│                          │   │  1. guardrails.py   input check (PII /     │
│ emotion2vec+ backbone    │◄──┤     prompt-injection)                      │
│  → Stage 1 heads         │   │  2. emotion_engine.py  TEXT-based emotion/ │
│  → Stage 2 XGBoost       │   │     stress/trust/urgency/prosody           │
│    (stress/frustration/  │   │  3. RACE (3.5s deadline): voice-CX result  │
│     urgency/escalation)  │   │     vs. text result — whichever answers    │
└─────────┬────────────────┘   │     first drives this turn (see           │
          │                    │     VOICE_CX_RACING.md)                    │
          │ result, if it wins │  4. knowledge_base.py  local FAQ retrieval │
          │ the race (else     │  5. policy_engine.py   scores 5 policies, │
          │ discarded, shown   │     picks the best                        │
          │ in shadow panel    │  6. rebuilds the Gemini system prompt —   │
          │ only)              │     emotion tone, policy mode, KB         │
          └───────────────────►│     grounding, verified-customer note      │
                                └───────────────────────────────────────────┘
                                        │
                                        ▼
                          Gemini (LLM) — with tool-calling against
                          agent/tools.py's BANKING_TOOLS:
                            verify_identity · lookup_customer_profile ·
                            check_recent_transactions · process_refund ·
                            create_support_ticket · escalate_to_specialist
                                        │
                                        ▼  draft reply
                    ┌───────────────────────────────────────────┐
                    │ agent/agent.py — tts_node()                 │
                    │  guardrails.py output check (PII / unsafe)  │
                    │  → evaluation.py scores the turn (resolution│
                    │    / CSAT / policy compliance / hallucination)│
                    └───────────────────────────────────────────┘
                                        │
                                        ▼
                  Cartesia (TTS, tone/speed set by the chosen policy)
                                        │
                                        ▼
                         Customer hears the response

  Throughout: dashboard_bridge.py broadcasts every step above (transcript,
  behavior/emotion, tool calls, guardrail results, knowledge hits, evaluation
  scores, real STT/LLM/TTS latency + cost, voice-CX result) over a WebSocket
  to frontend/ — the dashboard's "PIPELINE STATUS" card lights up each stage
  in real time as it actually happens, not on a fake timer.
```

**What's actually important here, in order:**
1. **The race, not a fallback chain.** Voice-CX used to be a slow "shadow"
   panel that never affected anything. It's now a genuine second signal
   racing the text path on a hard 3.5s deadline — if it wins, it *drives*
   the turn (`🎙️ VOICE (validated)` badge); if it loses, text drives
   instantly and voice's result still lands in the shadow panel a moment
   later. Full state machine: `VOICE_CX_RACING.md`.
2. **Tools are real function-calls**, not scripted text — Gemini decides
   whether to call `verify_identity`, look up the account, issue a refund,
   or escalate, based on the conversation.
3. **Escalation has a payoff.** `escalate_to_specialist` doesn't just log a
   ticket — it returns a generated meeting link + time slot the agent is
   instructed to actually tell the customer.
4. **Identity persists across calls.** A logged-in customer's name + DOB are
   saved at signup (`auth_server.py` → `storage.py`'s `verified_identities`
   table); `tools.set_current_user()` hydrates the mock CRM from that at the
   start of every call, so `verify_identity` never has to run again for them.

---

## File Structure

```
adaptivecx-demo/
├── agent/
│   ├── agent.py             ← LiveKit agent entrypoint; wires everything below together,
│   │                            including the voice/text race (RACE_TIMEOUT)
│   ├── emotion_engine.py    ← TEXT-based emotion/stress/trust/urgency
│   ├── policy_engine.py     ← Scores 5 response policies, picks the best
│   ├── voice_cx_client.py   ← Async HTTP client to voice-cx-server/, races the text path
│   ├── voice_cx_toggle.py   ← File-based flag for the dashboard's manual voice/text toggle
│   ├── tools.py             ← Agentic tools (identity verify, CRM lookup, refund, ticket,
│   │                            escalate_to_specialist) — per-user mock CRM
│   ├── banking_data.py      ← Deterministic per-user mock account (tier/balance/5 txns),
│   │                            seeded from user_id, used by both tools.py and auth_server.py
│   ├── auth_server.py       ← FastAPI: signup/login/session/chat-history/transactions API
│   ├── storage.py           ← SQLite persistence (users, sessions, verified identities, chat history)
│   ├── dashboard_bridge.py  ← WebSocket + /token + /health server for the dashboard
│   ├── guardrails.py        ← Input/output PII + prompt-injection checks
│   ├── observability.py     ← Real STT/LLM/TTS latency + token/cost tracking
│   ├── evaluation.py        ← Per-turn resolution/CSAT/compliance/hallucination scoring
│   └── knowledge_base.py    ← Local FAQ retrieval + knowledge-gap detection
│
├── frontend/
│   ├── index.html           ← Live dashboard UI (auth overlay, history + transactions panels)
│   ├── style.css             ← Dark glassmorphism design
│   └── app.js                 ← LiveKit + WebSocket client, pipeline visualizer, auth flow
│
├── adaptivecx-stage1/        ← Independent voice emotion model (emotion2vec+), trained on
│                                 IEMOCAP, validated with real WAV inference. See its README.md.
├── adaptivecx-stage2/         ← Customer-state model (stress/frustration/urgency/escalation),
│                                 XGBoost on Stage-1 + acoustic features. See its README.md.
├── voice-cx-server/            ← Deployable FastAPI service running Stage 1+2. Currently hosted
│   ├── kaggle_notebook.ipynb      on a Kaggle GPU notebook (tunneled via ngrok) — see below.
│   └── ...                        Called by agent/voice_cx_client.py. See its README.md.
│
├── start-demo.ps1              ← One-command start/restart for the whole stack (see below)
├── clear_room.py                 ← Clears stale LiveKit room state (auto-run by start-demo.ps1)
├── ARCHITECTURE.md                 ← Full component/port/data-flow reference (deployment topology
│                                       + per-turn pipeline), detailed enough to hand to an LLM to
│                                       regenerate an architecture diagram elsewhere.
├── VOICE_CX_RACING.md              ← The voice/text race state machine, in detail
├── DEMO_GUIDE.md                     ← Live demo script + talking points for presenting this
├── .env                                ← Your API keys + VOICE_CX_SERVER_URL (gitignored)
└── README.md                            ← this file
```

---

## Setup & Run

### One-time setup

1. **Fill in `.env`** with real keys:
   ```env
   LIVEKIT_URL=wss://your-project.livekit.cloud
   LIVEKIT_API_KEY=APIxxxxxxxxxx
   LIVEKIT_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxx
   DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
   GOOGLE_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxx
   CARTESIA_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
   # Voice-CX server (GPU speech-emotion, races the text path) — a Kaggle GPU
   # notebook tunneled via ngrok; see voice-cx-server/kaggle_notebook.ipynb.
   # Optional: leave blank to run text-only (no dual-path racing).
   VOICE_CX_SERVER_URL=https://your-ngrok-url.ngrok-free.dev
   ```
2. **Install dependencies** into the agent's venv:
   ```powershell
   py -3.12 -m venv .venv312
   .\.venv312\Scripts\pip install -r requirements.txt
   ```

### Every time you want to run the demo

```powershell
.\start-demo.ps1
```

This one command: kills any stale instances, checks the remote Voice-CX
server's health (skipped gracefully if `VOICE_CX_SERVER_URL` isn't set),
starts the agent worker and frontend server, clears stale LiveKit room
state, opens three Cloudflare quick tunnels (frontend, dashboard bridge,
auth server), patches `frontend/app.js` with the current tunnel hostnames,
and prints a shareable link.

Then open the printed link (or `frontend/index.html` directly if running
locally), optionally **sign up** (name + DOB — this is what lets the agent
skip identity verification and greet you by name), and click **"🎙️ Join Room"**.

---

## Demo It

Speak these and watch the dashboard react:

| What you say | Expected emotion | Expected policy |
|---|---|---|
| "I've called THREE TIMES and nobody helped me!" | 😠 ANGRY (stress ~0.85) | HIGH_EMPATHY |
| "This is so frustrating, it keeps crashing!" | 😤 FRUSTRATED | CALM |
| "Thank you so much, this is amazing!" | 😊 HAPPY | EFFICIENT |
| "I'm worried my account may have been hacked!" | 😨 FEARFUL | HIGH_EMPATHY |
| "I just want to check my balance." | 😐 NEUTRAL | BALANCED |
| "This still isn't fixed, I need a real person." | 😠/😤 (high stress, low trust) | ESCALATE → `escalate_to_specialist` returns a meeting link + time |

To see personalization: **sign up**, then open the **🧾 Transactions** panel
to see your 5 mock transactions before ever joining a call — then ask the
agent about one of them by name (e.g. *"what was that Amazon charge?"*) and
it looks up the real number instead of guessing.

For a full walkthrough with talking points, see `DEMO_GUIDE.md`.

---

## How the (currently live) text-based emotion detection works

```
Input text: "I've called THREE TIMES and nobody helped me!"

Step 1: Keyword scoring with position decay
  frustrated: "called"(0.55) + "times"(0.60) = 1.15
  angry:       "nobody"(~0) = 0.0
  ...

Step 2: Softmax normalization → P(E|T)
  frustrated: 0.71
  angry:      0.18
  neutral:    0.09

Step 3: Stress formula
  Stress = 0.40×(0.75×0.71) + 0.20×(neg_word_density) + 0.20×(CAPS ratio) + 0.20×(urgency)
         = 0.74

Step 4: Policy scoring
  Score(HIGH_EMPATHY) = 0.35×0.80 + 0.30×0.91 + 0.15×0.6 + 0.15×0.4 + 0.05×0.0
                      = 0.67 ← WINS

Selected: HIGH_EMPATHY
```

This is a hand-designed heuristic formula (documented in
`agent/emotion_engine.py`), not a trained ML model. It's always computed
first (~1ms) and used the moment the voice-CX race times out; the trained,
GPU-backed acoustic alternative (Stage 1/2, see `VOICE_CX_RACING.md`) is
what actually drives most turns when the Voice-CX server is up and fast.

---

## Verify it's working (not just printing)

1. **Speak** → Deepgram transcribes your actual voice (not hardcoded text)
2. **Dashboard emotion badge changes** → proves real detection on your words
3. **PIPELINE STATUS card** lights up STT → Voice-CX → Behavior → Policy →
   LLM → TTS in real time, driven by actual backend events
4. **Source badge** shows 🎙️ VOICE (validated) or 💬 TEXT (fallback) —
   proves the race is real, not cosmetic
5. **Agent speaks different responses** for angry vs. happy input, and
   greets a logged-in customer by name without asking who they are
6. **Ask about a real transaction** → the agent calls `check_recent_transactions`
   and states the actual mock number, never a guess
7. **Say you're still unsatisfied** → agent calls `escalate_to_specialist` and
   tells you an actual meeting link + time slot
8. **OBSERVABILITY card** shows real STT audio duration, LLM TTFT/tokens,
   TTS timing, and a running session cost estimate

This is a live, real-time, end-to-end working system, not a scripted demo.
