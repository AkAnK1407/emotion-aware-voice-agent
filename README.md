# EMOTIONALLY INTELLIGENT AI VOICE AGENT FOR CX TRANSFORMATIONS
# AdaptiveCX — Emotional Voice Agent Demo
## Setup & Run Guide

---

## What This Does (Problem Statement)

Traditional voice bots treat every customer the same — they don't care if you are
angry, sad, or happy. **AdaptiveCX** is an emotionally-intelligent voice agent that:

- 🎤 **Listens** to your real voice (via LiveKit WebRTC)
- 🧠 **Detects** emotion, stress, urgency, trust in real-time
- ⚖️ **Selects** the best conversation policy (HIGH_EMPATHY / BALANCED / EFFICIENT / ESCALATE)
- 💬 **Generates** adaptive responses using Gemini (the prompt changes based on emotion)
- 🔊 **Speaks back** with Cartesia TTS (tone adapted to emotional state)
- 📊 **Shows** everything live on a dashboard

---

## File Structure

```
adaptivecx-demo/
├── agent/
│   ├── agent.py             ← LiveKit agent (main entry point)
│   ├── emotion_engine.py    ← Keyword softmax emotion detection
│   ├── policy_engine.py     ← Policy scoring engine
│   ├── dashboard_bridge.py  ← WebSocket server for dashboard
│   ├── tools.py             ← Agentic tool calls (CRM/refund/ticket/identity, mock data)
│   ├── guardrails.py        ← Input/output PII + prompt-injection checks
│   ├── observability.py     ← Real STT/LLM/TTS latency + token/cost tracking
│   ├── evaluation.py        ← Per-turn resolution/CSAT/compliance/hallucination scoring
│   └── knowledge_base.py    ← Local FAQ retrieval + knowledge-gap detection
├── frontend/
│   ├── index.html           ← Live dashboard UI
│   ├── style.css            ← Dark glassmorphism design
│   └── app.js               ← LiveKit + WebSocket client
├── DEMO_GUIDE.md            ← Live demo script + backend explanation for instructors
├── .env                     ← Your API keys
└── README.md
```

---

## Step 1: Fill in Your API Keys

Edit `.env` and add your real keys:

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxxxx
LIVEKIT_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxx
DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxx
CARTESIA_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

Also update the LiveKit URL in `frontend/app.js` (line 8):
```js
const LIVEKIT_URL = "wss://your-project.livekit.cloud";
```

---

## Step 2: Install Dependencies

```powershell
cd "c:\Users\hp\Desktop\agent v1\adaptivecx-demo"
pip install -r requirements.txt
```

---

## Step 3: Run the Agent

```powershell
cd agent
python agent.py dev
```

You should see:
```
Dashboard WebSocket server starting on ws://localhost:8765
INFO | Connected to LiveKit room
INFO | Participant joined: ...
```

---

## Step 4: Open the Dashboard

Open `frontend/index.html` in your browser (Chrome recommended).

- The dashboard auto-connects to the agent WebSocket at `ws://localhost:8765`
- Click **"🎙️ Join Room"** to connect your microphone to the LiveKit room
- If prompted for a token, generate one at: https://cloud.livekit.io → your project → token generator

---

## Step 5: Demo It!

Speak these sentences and watch the dashboard react:

| What you say | Expected emotion | Expected policy |
|---|---|---|
| "I've called THREE TIMES and nobody helped me!" | 😠 ANGRY (stress ~0.85) | HIGH_EMPATHY |
| "This is so frustrating, it keeps crashing!" | 😤 FRUSTRATED | CALM |
| "Thank you so much, this is amazing!" | 😊 HAPPY | EFFICIENT |
| "I'm worried my account may have been hacked!" | 😨 FEARFUL | HIGH_EMPATHY |
| "I just want to check my balance." | 😐 NEUTRAL | BALANCED |

---

## How the Emotion Detection Works (For Explanation)

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

---

## Verify It's Working (Not Just Printing)

1. **Speak** → Deepgram transcribes your actual voice (not hardcoded text)
2. **Dashboard emotion badge changes** → proves real detection on your words
3. **Agent speaks different responses** for angry vs happy input
4. **Gemini system prompt changes** based on emotion (check terminal logs)
5. **Cartesia TTS speed** is 0.85× for HIGH_EMPATHY, 1.05× for EFFICIENT

This is a **live, real-time, end-to-end working system**, not a demo script.




command three terminal :
cd "C:\Users\hp\Desktop\agent v1\adaptivecx-demo"
.\.venv312\Scripts\activate.ps1
.\.venv312\Scripts\python.exe agent\agent.py dev

cd "C:\Users\hp\Desktop\agent v1\adaptivecx-demo"
.\.venv312\Scripts\activate.ps1
.\.venv312\Scripts\python.exe .\agent\dashboard_bridge.py 

cd "C:\Users\hp\Desktop\agent v1\adaptivecx-demo"
.\.venv312\Scripts\activate.ps1
.\.venv312\Scripts\python.exe .\token_server.py

