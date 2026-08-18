# AdaptiveCX — Emotional Voice Agent Demo

Traditional voice bots treat every customer the same — they don't care if
you're angry, sad, or happy. **AdaptiveCX** is an emotionally-intelligent
voice agent for banking support that:

- 🎤 **Listens** to your real voice (LiveKit WebRTC)
- 🧠 **Detects** emotion, stress, trust, urgency in real-time
- ⚖️ **Selects** a conversation policy (HIGH_EMPATHY / CALM / BALANCED / EFFICIENT / ESCALATE)
- 💬 **Generates** adaptive responses with Gemini (the system prompt changes based on detected emotion)
- 🔊 **Speaks back** with Cartesia TTS (tone/pace adapted to emotional state)
- 📊 **Shows** everything live on a dashboard — emotion, policy, guardrails, knowledge retrieval, evaluation, cost/latency

---

## System Flow

```
Customer voice (browser mic)
        │
        ▼
   LiveKit WebRTC room
        │
        ▼
  Deepgram (STT) ──────────────► transcript
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ agent/agent.py — on_user_turn_completed()                  │
│                                                              │
│  1. guardrails.py       input check (PII / prompt-injection)│
│  2. emotion_engine.py   TEXT-based emotion/stress/trust/    │
│                         urgency (keyword-softmax + prosody  │
│                         proxies from punctuation/caps/etc.) │
│  3. policy_engine.py    scores 5 policies, picks the best   │
│  4. knowledge_base.py   local FAQ retrieval                 │
│  5. rebuilds the Gemini system prompt for this turn         │
└───────────────────────────────────────────────────────────┘
        │
        ▼
   Gemini (LLM) ──────────────► response text
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ agent/agent.py — tts_node()                                 │
│  guardrails.py output check → evaluation.py scores the turn │
│  (resolution / CSAT / policy compliance / hallucination)    │
└───────────────────────────────────────────────────────────┘
        │
        ▼
  Cartesia (TTS, tone/speed set by the chosen policy)
        │
        ▼
   Customer hears the response

  Throughout: dashboard_bridge.py broadcasts every step above
  (transcript, emotion, policy, guardrail results, knowledge
  hits, evaluation scores, STT/LLM/TTS latency+cost) over a
  WebSocket to frontend/ in real time.
```

### Voice-based CX (Stage 1 + Stage 2) — shadow mode, experimental

The pipeline above reacts to *what the customer typed* (post-STT text).
Separately, this project also has a fully-trained **voice-based** emotion
model that reacts to *how the customer's voice actually sounds* — pitch,
tone, prosody — independent of the words:

```
Same customer audio (tapped in agent.py's stt_node, unchanged pass-through)
        │
        ▼
  SSH tunnel → voice-cx-server/ (runs on a separate EC2 instance —
        │      the local machine doesn't have enough RAM to run this
        │      reliably alongside the live agent)
        ▼
  Stage 1: emotion2vec+ backbone + trained heads
           → emotion, arousal, valence (from audio only)
        │
        ▼
  Stage 2: XGBoost regressors (acoustic-formula-labeled)
           → stress, frustration, urgency, escalation_risk
        │
        ▼
  Broadcast to the dashboard's "VOICE-BASED CX (experimental,
  shadow mode)" panel — shown for comparison only.
```

**This does not drive the agent's actual response** — the policy engine
above still runs entirely on the text-based path, which is faster and
already proven. The voice-based path runs in the background, strictly one
request at a time, and fails silently (panel just stays empty) if the
remote server is unreachable — it can never break the live conversation.
Full details, training process, and why it's a two-source *bootstrap*
signal (not real ground truth) are in `adaptivecx-stage1/README.md`,
`adaptivecx-stage2/README.md`, and `voice-cx-server/README.md`.

---

## File Structure

```
adaptivecx-demo/
├── agent/
│   ├── agent.py             ← LiveKit agent entrypoint; wires everything below together
│   ├── emotion_engine.py    ← TEXT-based emotion/stress/trust/urgency (currently drives responses)
│   ├── policy_engine.py     ← Scores 5 response policies, picks the best
│   ├── dashboard_bridge.py  ← WebSocket + /token + /health server for the dashboard
│   ├── tools.py             ← Agentic tools (identity verify, CRM lookup, refund, ticket)
│   ├── guardrails.py        ← Input/output PII + prompt-injection checks
│   ├── observability.py     ← Real STT/LLM/TTS latency + token/cost tracking
│   ├── evaluation.py        ← Per-turn resolution/CSAT/compliance/hallucination scoring
│   ├── knowledge_base.py    ← Local FAQ retrieval + knowledge-gap detection
│   └── voice_cx_client.py   ← Async HTTP client to voice-cx-server/ (shadow mode, see above)
│
├── frontend/
│   ├── index.html           ← Live dashboard UI
│   ├── style.css            ← Dark glassmorphism design
│   └── app.js                ← LiveKit + WebSocket client
│
├── adaptivecx-stage1/        ← Independent voice emotion model (emotion2vec+), trained on
│                                IEMOCAP, validated with real WAV inference. See its README.md.
├── adaptivecx-stage2/        ← Customer-state model (stress/frustration/urgency/escalation),
│                                XGBoost on Stage-1 + acoustic features. See its README.md.
├── voice-cx-server/           ← Deployable FastAPI service running Stage 1+2 remotely
│                                (EC2), called by agent/voice_cx_client.py. See its README.md.
│
├── start-demo.ps1            ← One-command start/restart for the whole stack (see below)
├── restart-demo.ps1           ← Same as start-demo.ps1, named for when something's down mid-demo
├── clear_room.py               ← Clears stale LiveKit room state (auto-run by start-demo.ps1)
├── DEMO_GUIDE.md                ← Live demo script + talking points for presenting this
├── .env                          ← Your API keys + VOICE_CX_SERVER_URL (gitignored)
├── keypair*.pem                   ← SSH keys for the EC2 voice-CX server (gitignored, never commit)
└── README.md                       ← this file
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
   # Optional -- only if voice-cx-server/ is deployed and tunneled to localhost:8000
   VOICE_CX_SERVER_URL=http://localhost:8000
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

This one command: kills any stale instances, starts the voice-CX SSH tunnel
(if `voice-cx-server/` is deployed — non-critical, skipped gracefully if
not), starts the agent worker and frontend server, clears stale LiveKit
room state, opens two Cloudflare quick tunnels, patches
`frontend/app.js` with the current tunnel hostname, and prints a
shareable link. If something goes down mid-demo, run `.\restart-demo.ps1`
(identical, just named for that moment).

Then open the printed link (or `frontend/index.html` directly if running
locally) and click **"🎙️ Join Room"**.

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
`agent/emotion_engine.py`), not a trained ML model — the trained,
audio-based alternative is Stage 1/2 (see "Voice-based CX" above), which
currently runs in shadow mode alongside this, not in place of it.

---

## Verify it's working (not just printing)

1. **Speak** → Deepgram transcribes your actual voice (not hardcoded text)
2. **Dashboard emotion badge changes** → proves real detection on your words
3. **Agent speaks different responses** for angry vs. happy input
4. **Gemini's system prompt changes** based on emotion (check terminal logs)
5. **Cartesia TTS speed** is 0.85× for HIGH_EMPATHY, 1.05× for EFFICIENT
6. **"VOICE-BASED CX" panel** updates a few seconds after each turn, independently of the text-based cards above it

This is a live, real-time, end-to-end working system, not a scripted demo.
