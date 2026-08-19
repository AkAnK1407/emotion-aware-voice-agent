/**
 * app.js — AdaptiveCX Dashboard
 * Connects to:
 *   1. LiveKit room (for voice input/output via WebRTC)
 *   2. Dashboard WebSocket (ws://localhost:8765) for live signals from the agent
 */

// ── Config ────────────────────────────────────────────────────────────────────
console.log("Loading AdaptiveCX Dashboard...");
console.log("window.LiveKit:", window.LiveKit);
console.log("window.LivekitClient:", window.LivekitClient);
console.log("typeof module:", typeof module !== 'undefined' ? module : 'undefined');
console.log("typeof exports:", typeof exports !== 'undefined' ? exports : 'undefined');
const LIVEKIT_URL = "wss://voice-agent-2y77bxoc.livekit.cloud";

// Backend (agent worker + dashboard WS + /token) hostname when not running
// locally. Update this whenever you restart the Cloudflare Tunnel / redeploy
// to Render — quick tunnels get a new random hostname each run.
const BACKEND_HOST = "kernel-thereafter-counsel-softball.trycloudflare.com";

// Login/signup/history API (auth_server.py) — separate service, separate
// tunnel, since dashboard_bridge's server can't accept anything but a
// bodyless GET. Same "update after restarting the tunnel" caveat as above.
const AUTH_HOST = "thats-medicines-managed-mid.trycloudflare.com";

const IS_LOCAL = location.hostname === "localhost" || location.hostname === "127.0.0.1";
const DASHBOARD_WS = IS_LOCAL ? "ws://localhost:8765" : `wss://${BACKEND_HOST}/`;
const TOKEN_URL = IS_LOCAL ? "http://localhost:7881/token" : `https://${BACKEND_HOST}/token`;
const AUTH_URL = IS_LOCAL ? "http://localhost:8766" : `https://${AUTH_HOST}`;

// ── Auto-Generate Unique Room per Tab/Session ─────────────────────────────────
// sessionStorage (not localStorage) is scoped per browser tab, not per browser
// profile -- localStorage is shared across every tab of the same browser, so
// two tabs open to this page would get the same saved room ID and collide
// into the same conversation. sessionStorage gives each tab its own room,
// while still surviving reloads of that same tab.
function getOrCreateUserId() {
  let userId = sessionStorage.getItem("adaptivecx_user_id");
  if (!userId) {
    userId = "user-" + Math.random().toString(36).substring(2, 9) + "-" + Date.now();
    sessionStorage.setItem("adaptivecx_user_id", userId);
  }
  return userId;
}

const CURRENT_USER_ID = getOrCreateUserId();
const AUTO_ROOM_NAME = "room-" + CURRENT_USER_ID;

// ── State ─────────────────────────────────────────────────────────────────────
let room = null;
let dashWs = null;
let turnCount = 0;
let emotionHistory = [];
let isConnected = false;
let interimBubbleEl = null;      // live "still speaking" bubble, replaced on final
let lastCustomerBubbleEl = null; // most recent finalized customer bubble, awaiting its emotion tag
let authSession = null;          // {token, user_id, display_name} once logged in, else null (guest)
let authMode = "login";          // "login" | "signup" -- which tab is active in the auth card
let voiceCxEnabled = true;        // voice-CX toggle state (from server)

// ── DOM References ────────────────────────────────────────────────────────────
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const statusPill = document.getElementById("statusPill");
const turnCountEl = document.getElementById("turnCount");

const emotionEmoji = document.getElementById("emotionEmoji");
const emotionName = document.getElementById("emotionName");
const emotionConf = document.getElementById("emotionConf");
const emotionBar = document.getElementById("emotionBar");
const emotionCard = document.getElementById("emotionCard");

const stressValue = document.getElementById("stressValue");
const stressFill = document.getElementById("stressFill");
const trustValue = document.getElementById("trustValue");
const trustFill = document.getElementById("trustFill");
const urgencyFill = document.getElementById("urgencyFill");
const urgencyValue = document.getElementById("urgencyValue");
const patienceVal = document.getElementById("patienceValue");
const patiencePips = document.getElementById("patiencePips");

const policyBanner = document.getElementById("policyBanner");
const policyName = document.getElementById("policyName");
const policyDesc = document.getElementById("policyDesc");
const tagEmpathy = document.getElementById("tagEmpathy");
const tagSpeed = document.getElementById("tagSpeed");
const tagEscalate = document.getElementById("tagEscalate");

const prosodyWords = document.getElementById("prosodyWords");
const prosodyExcl = document.getElementById("prosodyExcl");
const prosodyRepeat = document.getElementById("prosodyRepeat");
const prosodyCaps = document.getElementById("prosodyCaps");

const voiceStressValue = document.getElementById("voiceStressValue");
const voiceFrustrationValue = document.getElementById("voiceFrustrationValue");
const voiceUrgencyValue = document.getElementById("voiceUrgencyValue");
const voiceEscalationValue = document.getElementById("voiceEscalationValue");
const voiceEmotionValue = document.getElementById("voiceEmotionValue");
const behaviorSourceBadge = document.getElementById("behaviorSourceBadge");

const convScoreValue = document.getElementById("convScoreValue");
const convLabelSub = document.getElementById("convLabelSub");
const convTrendBadge = document.getElementById("convTrendBadge");
const convFill = document.getElementById("convFill");

const conversation = document.getElementById("conversation");
const clearBtn = document.getElementById("clearBtn");
const joinBtn = document.getElementById("joinBtn");
const leaveBtn = document.getElementById("leaveBtn");
const roomInput = document.getElementById("roomInput");
const emotionTimeline = document.getElementById("emotionTimeline");

// ── Auto-Set User's Private Room ─────────────────────────────────────────────
roomInput.value = AUTO_ROOM_NAME;
roomInput.title = `Your private room: ${AUTO_ROOM_NAME}. Each browser tab gets its own chat.`;
roomInput.readOnly = true;
roomInput.style.cursor = "not-allowed";
roomInput.style.opacity = "0.7";

const pipeSteps = {
  stt: document.getElementById("step-stt"),
  voicecx: document.getElementById("step-voicecx"),
  behavior: document.getElementById("step-behavior"),
  policy: document.getElementById("step-policy"),
  llm: document.getElementById("step-llm"),
  tts: document.getElementById("step-tts"),
};

// ── Enterprise Panel DOM References ──────────────────────────────────────────
const toolFeed = document.getElementById("toolFeed");
const guardInputBadge = document.getElementById("guardInputBadge");
const guardInputReason = document.getElementById("guardInputReason");
const guardOutputBadge = document.getElementById("guardOutputBadge");
const guardOutputReason = document.getElementById("guardOutputReason");
const kbBadge = document.getElementById("kbBadge");
const kbBody = document.getElementById("kbBody");
const evalResolutionBadge = document.getElementById("evalResolutionBadge");
const evalCsatBadge = document.getElementById("evalCsatBadge");
const evalComplianceBadge = document.getElementById("evalComplianceBadge");
const evalComplianceReason = document.getElementById("evalComplianceReason");
const evalHallucinationBadge = document.getElementById("evalHallucinationBadge");
const evalHallucinationReason = document.getElementById("evalHallucinationReason");
const obsStt = document.getElementById("obsStt");
const obsLlm = document.getElementById("obsLlm");
const obsTts = document.getElementById("obsTts");
const obsTokens = document.getElementById("obsTokens");
const obsTotalTurns = document.getElementById("obsTotalTurns");
const obsTotalCost = document.getElementById("obsTotalCost");

// ── Auth / History DOM References ────────────────────────────────────────────
const authOverlay = document.getElementById("authOverlay");
const authForm = document.getElementById("authForm");
const authError = document.getElementById("authError");
const authUsername = document.getElementById("authUsername");
const authDisplayName = document.getElementById("authDisplayName");
const authDob = document.getElementById("authDob");
const authPassword = document.getElementById("authPassword");
const authSubmitBtn = document.getElementById("authSubmitBtn");
const authCloseBtn = document.getElementById("authCloseBtn");
const authGuestBtn = document.getElementById("authGuestBtn");
const tabLogin = document.getElementById("tabLogin");
const tabSignup = document.getElementById("tabSignup");

const accountName = document.getElementById("accountName");
const loginOpenBtn = document.getElementById("loginOpenBtn");
const logoutBtn = document.getElementById("logoutBtn");
const historyBtn = document.getElementById("historyBtn");
const historyPanel = document.getElementById("historyPanel");
const historyBackdrop = document.getElementById("historyBackdrop");
const historyBody = document.getElementById("historyBody");
const transactionsBtn = document.getElementById("transactionsBtn");
const transactionsPanel = document.getElementById("transactionsPanel");
const transactionsBackdrop = document.getElementById("transactionsBackdrop");
const transactionsBody = document.getElementById("transactionsBody");
const transactionsCloseBtn = document.getElementById("transactionsCloseBtn");
const contactsBtn = document.getElementById("contactsBtn");
const contactsPanel = document.getElementById("contactsPanel");
const contactsBackdrop = document.getElementById("contactsBackdrop");
const contactsBody = document.getElementById("contactsBody");
const contactsCloseBtn = document.getElementById("contactsCloseBtn");
const historyCloseBtn = document.getElementById("historyCloseBtn");

// ── Helpers ───────────────────────────────────────────────────────────────────

function setStatus(text, state) {
  statusText.textContent = text;
  statusDot.className = "status-dot " + (state || "");
}

function pct(v) { return Math.round(v * 100) + "%"; }

// ── Audio Unlock (browser autoplay policy) ────────────────────────────────────
let _audioUnlockBtn = null;

function showAudioUnlockBtn() {
  if (_audioUnlockBtn) return;
  _audioUnlockBtn = document.createElement("div");
  _audioUnlockBtn.id = "audioUnlockOverlay";
  _audioUnlockBtn.innerHTML = `
    <div class="audio-unlock-box">
      <div class="audio-unlock-icon">🔊</div>
      <div class="audio-unlock-title">Click to Enable Agent Audio</div>
      <div class="audio-unlock-sub">Your browser requires a tap to allow sound playback.</div>
      <button id="audioUnlockBtn" class="audio-unlock-cta">Enable Audio</button>
    </div>`;
  _audioUnlockBtn.style.cssText = `
    position:fixed;bottom:28px;right:28px;z-index:9999;
    background:rgba(14,17,27,0.92);backdrop-filter:blur(16px);
    border:1px solid rgba(139,92,246,0.45);border-radius:16px;
    padding:20px 24px;text-align:center;box-shadow:0 8px 40px rgba(0,0,0,0.5);
    animation:fadeIn 0.3s ease;`;
  document.body.appendChild(_audioUnlockBtn);
  document.getElementById("audioUnlockBtn").addEventListener("click", async () => {
    try {
      if (room && room.startAudio) await room.startAudio();
      // Also play any paused audio elements
      document.querySelectorAll("audio").forEach(a => a.play().catch(() => {}));
      hideAudioUnlockBtn();
    } catch(e) { console.warn("[Audio] Unlock failed:", e); }
  });
}

function hideAudioUnlockBtn() {
  if (_audioUnlockBtn) {
    _audioUnlockBtn.remove();
    _audioUnlockBtn = null;
  }
}

// Pipeline visualizer is driven by the real dashboard events for this turn
// (transcript/voice_cx/behavior/agent_response/observability) rather than a
// fixed timer -- each step's state reflects when that stage actually ran.
const PIPE_ORDER = ["stt", "voicecx", "behavior", "policy", "llm", "tts"];

function markStep(key, status, label) {
  const el = pipeSteps[key];
  if (!el) return;
  el.className = "pipe-step" + (status === "idle" ? "" : " " + status);
  el.querySelector(".pipe-status").textContent =
    label || { idle: "Idle", active: "Processing...", done: "Done", skipped: "Skipped" }[status];
}

function resetPipeline() {
  for (const key of PIPE_ORDER) markStep(key, "idle");
}

function addBubble(speaker, text, emotionTag, emotionColor) {
  // Remove placeholder
  const placeholder = conversation.querySelector(".conv-placeholder");
  if (placeholder) placeholder.remove();

  const wrap = document.createElement("div");
  wrap.className = `bubble-wrap ${speaker}`;

  const label = document.createElement("div");
  label.className = "bubble-label";
  label.textContent = speaker === "customer" ? "🎤 You" : "🤖 AdaptiveCX";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${speaker}`;
  bubble.textContent = text;

  const meta = document.createElement("div");
  meta.className = "bubble-meta";
  meta.textContent = new Date().toLocaleTimeString();

  if (emotionTag && speaker === "customer") {
    const tag = document.createElement("span");
    tag.className = "bubble-emotion-tag";
    tag.style.background = emotionColor + "25";
    tag.style.border = `1px solid ${emotionColor}50`;
    tag.style.color = emotionColor;
    tag.textContent = emotionTag;
    meta.appendChild(tag);
  }

  wrap.appendChild(label);
  wrap.appendChild(bubble);
  wrap.appendChild(meta);
  conversation.appendChild(wrap);
  conversation.scrollTop = conversation.scrollHeight;
  return wrap;
}

// Live "customer is speaking" preview, updated in place as interim STT
// results arrive, then swapped for a real bubble once the turn is final —
// this is what makes the transcript feel real-time instead of appearing
// only once the (multi-second) emotion/policy pipeline finishes.
function updateInterimBubble(text) {
  if (!text || !text.trim()) return;
  if (!interimBubbleEl) {
    interimBubbleEl = addBubble("customer", text, null, null);
    interimBubbleEl.classList.add("interim");
  } else {
    interimBubbleEl.querySelector(".bubble").textContent = text;
  }
  conversation.scrollTop = conversation.scrollHeight;
}

function finalizeCustomerBubble(text) {
  if (interimBubbleEl) {
    interimBubbleEl.classList.remove("interim");
    interimBubbleEl.querySelector(".bubble").textContent = text;
    lastCustomerBubbleEl = interimBubbleEl;
    interimBubbleEl = null;
  } else {
    lastCustomerBubbleEl = addBubble("customer", text, null, null);
  }
}

// Called once the "behavior" event lands (after emotion/policy analysis) --
// paints the emotion tag onto the bubble that was already shown live.
function tagLastCustomerBubble(emotionTag, emotionColor) {
  if (!lastCustomerBubbleEl) return;
  const meta = lastCustomerBubbleEl.querySelector(".bubble-meta");
  if (!meta || meta.querySelector(".bubble-emotion-tag")) return;
  const tag = document.createElement("span");
  tag.className = "bubble-emotion-tag";
  tag.style.background = emotionColor + "25";
  tag.style.border = `1px solid ${emotionColor}50`;
  tag.style.color = emotionColor;
  tag.textContent = emotionTag;
  meta.appendChild(tag);
  lastCustomerBubbleEl = null;
  conversation.scrollTop = conversation.scrollHeight;
}

function updateEmotionTimeline(emotion, color, confidence) {
  emotionHistory.push({ emotion, color, confidence });
  if (emotionHistory.length > 12) emotionHistory.shift();

  // Remove placeholder
  const ph = emotionTimeline.querySelector(".timeline-placeholder");
  if (ph) ph.remove();

  emotionTimeline.innerHTML = "";
  emotionHistory.forEach(e => {
    const bar = document.createElement("div");
    bar.className = "timeline-bar";
    bar.style.background = e.color;
    bar.style.height = Math.round(e.confidence * 60) + "px";
    bar.setAttribute("data-tip", e.emotion.toUpperCase());
    emotionTimeline.appendChild(bar);
  });
}

function updatePatiencePips(patience) {
  const pips = patiencePips.querySelectorAll(".pip");
  pips.forEach((p, i) => {
    p.className = "pip";
    if (patience === "high" && i < 3) p.classList.add("active");
    else if (patience === "medium" && i < 2) p.classList.add("medium");
    else if (patience === "low" && i === 0) p.classList.add("low");
  });
}

// ── Dashboard WebSocket ───────────────────────────────────────────────────────

function connectDashboard() {
  // Tag this connection with the room we're about to join, so the server
  // only broadcasts this room's transcript/emotion/policy events to us --
  // not every other user's conversation happening on the same backend.
  const separator = DASHBOARD_WS.includes("?") ? "&" : "?";
  dashWs = new WebSocket(`${DASHBOARD_WS}${separator}room=${encodeURIComponent(AUTO_ROOM_NAME)}`);

  dashWs.onopen = () => {
    console.log("[Dashboard] Connected");
    setStatus("Agent connected. Speak to begin.", "connected");
  };

  dashWs.onerror = () => {
    console.warn("[Dashboard] Not connected — agent may not be running yet.");
    setStatus("Agent not running. Start agent.py first.", "");
  };

  dashWs.onclose = () => {
    console.log("[Dashboard] Disconnected. Retrying in 3s...");
    setStatus("Agent disconnected. Retrying...", "");
    setTimeout(connectDashboard, 3000);
  };

  dashWs.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data);
      handleDashboardEvent(data);
    } catch (e) {
      console.error("Dashboard parse error:", e);
    }
  };
}

function handleDashboardEvent(data) {
  switch (data.type) {

    case "connected":
      setStatus("Agent ready — speak to begin!", "connected");
      break;

    case "behavior":
      // ── Source badge: which path drove this turn's decision ───────────────
      if (data.source === "voice") {
        behaviorSourceBadge.textContent = "🎙️ VOICE (validated)";
        behaviorSourceBadge.style.background = "rgba(34,197,94,0.15)";
        behaviorSourceBadge.style.color = "#22c55e";
      } else {
        behaviorSourceBadge.textContent = "💬 TEXT (fallback)";
        behaviorSourceBadge.style.background = "rgba(156,163,175,0.15)";
        behaviorSourceBadge.style.color = "#9ca3af";
      }

      // ── Update emotion ────────────────────────────────────────────────────
      emotionEmoji.textContent = data.emotion_emoji;
      emotionEmoji.classList.add("pop");
      setTimeout(() => emotionEmoji.classList.remove("pop"), 500);
      tagLastCustomerBubble(data.emotion.toUpperCase(), data.emotion_color);

      emotionName.textContent = data.emotion.toUpperCase();
      emotionName.style.color = data.emotion_color;
      emotionConf.textContent = pct(data.emotion_confidence);
      emotionCard.style.borderColor = data.emotion_color + "55";
      emotionCard.style.boxShadow = `0 0 24px ${data.emotion_color}22`;
      emotionBar.style.width = pct(data.emotion_confidence);
      emotionBar.style.background = data.emotion_color;

      // ── Stress ────────────────────────────────────────────────────────────
      stressFill.style.width = pct(data.stress);
      stressValue.textContent = pct(data.stress);
      const stressRGB = data.stress > 0.7 ? "#ef4444" : data.stress > 0.4 ? "#f97316" : "#22c55e";
      stressValue.style.color = stressRGB;

      // ── Trust ─────────────────────────────────────────────────────────────
      trustFill.style.width = pct(data.trust);
      trustValue.textContent = pct(data.trust);
      trustValue.style.color = data.trust > 0.5 ? "#22c55e" : "#ef4444";

      // ── Urgency ───────────────────────────────────────────────────────────
      urgencyFill.style.width = pct(data.urgency);
      urgencyValue.textContent = data.urgency > 0.6 ? "HIGH" : data.urgency > 0.3 ? "Medium" : "Low";
      urgencyValue.style.color = data.urgency > 0.6 ? "#ef4444" : "#f97316";

      // ── Patience ──────────────────────────────────────────────────────────
      patienceVal.textContent = data.patience.charAt(0).toUpperCase() + data.patience.slice(1);
      updatePatiencePips(data.patience);

      // ── Session Conversation State (Recency-Weighted: 70% current) ───────
      if (typeof data.conversation_score !== "undefined") {
        const score = data.conversation_score;
        convScoreValue.textContent = (score >= 0 ? "+" : "") + score.toFixed(2);
        convLabelSub.textContent = data.conversation_label || "Neutral";
        convTrendBadge.textContent = data.conversation_trend || "Stable ➔";

        if (data.conversation_trend && data.conversation_trend.includes("Improving")) {
          convTrendBadge.style.color = "#22c55e";
          convTrendBadge.style.borderColor = "#22c55e55";
        } else if (data.conversation_trend && data.conversation_trend.includes("Worsening")) {
          convTrendBadge.style.color = "#ef4444";
          convTrendBadge.style.borderColor = "#ef444455";
        } else {
          convTrendBadge.style.color = "#9ca3af";
          convTrendBadge.style.borderColor = "rgba(255,255,255,0.12)";
        }

        const fillPct = Math.min(Math.max(Math.round(((score + 1.0) / 2.0) * 100), 0), 100);
        convFill.style.width = fillPct + "%";
      }

      // ── Prosody ───────────────────────────────────────────────────────────
      if (data.prosody) {
        prosodyWords.textContent = data.prosody.word_count;
        prosodyExcl.textContent = data.prosody.exclamation_count;
        prosodyRepeat.textContent = (data.prosody.repetition_score * 100).toFixed(0) + "%";
        prosodyCaps.textContent = (data.prosody.caps_ratio * 100).toFixed(0) + "%";
      }

      // ── Policy ────────────────────────────────────────────────────────────
      if (data.policy) {
        policyName.textContent = data.policy.name.replace("_", " ");
        policyDesc.textContent = data.policy.description;
        policyName.style.color = data.policy.color;
        policyBanner.style.borderColor = data.policy.color + "44";
        policyBanner.style.background = data.policy.color + "12";
        tagEmpathy.textContent = `Empathy ${pct(data.policy.empathy_level)}`;
        tagSpeed.textContent = `Speed ${data.policy.speaking_speed}×`;
        tagEscalate.style.display = data.policy.offer_escalation ? "block" : "none";
      }

      // ── Timeline ──────────────────────────────────────────────────────────
      updateEmotionTimeline(data.emotion, data.emotion_color, data.emotion_confidence);

      // ── Pipeline: behavior + policy resolved, LLM prompting starts next ──
      markStep("behavior", "done");
      markStep("policy", "done");
      markStep("llm", "active");

      // ── Turn count ────────────────────────────────────────────────────────
      turnCount++;
      turnCountEl.textContent = turnCount;
      setStatus(`Turn ${turnCount}: ${data.emotion.toUpperCase()} → ${data.policy ? data.policy.name : ""}`, "listening");
      break;

    case "transcript":
      if (data.speaker === "customer") {
        // Live as they speak: update in place, no emotion tag yet (that
        // lands separately once the "behavior" event arrives).
        if (data.is_partial) {
          if (pipeSteps.stt.className === "pipe-step") resetPipeline();
          markStep("stt", "active");
          updateInterimBubble(data.text);
        } else {
          // STT finalized this turn's transcript -- behavior engine and the
          // voice-CX shadow path both kick off from here in parallel.
          markStep("stt", "done");
          markStep("voicecx", voiceCxEnabled ? "active" : "skipped");
          markStep("behavior", "active");
          finalizeCustomerBubble(data.text);
        }
      }
      break;

    case "agent_response":
      // Gemini's text is ready -- TTS synthesis starts next.
      markStep("llm", "done");
      markStep("tts", "active");
      addBubble("agent", data.text, null, null);
      setStatus("Agent responded — listening for next input", "connected");
      break;

    case "tool_call":
      handleToolCall(data);
      break;

    case "transaction_update":
      handleTransactionUpdate(data);
      break;

    case "new_transaction":
      handleNewTransaction(data);
      break;

    case "balance_update":
      handleBalanceUpdate(data);
      break;

    case "guardrail":
      handleGuardrail(data);
      break;

    case "knowledge":
      handleKnowledge(data);
      break;

    case "evaluation":
      handleEvaluation(data);
      break;

    case "observability":
      handleObservability(data);
      break;

    case "observability_totals":
      obsTotalTurns.textContent = data.turns;
      obsTotalCost.textContent = "$" + data.estimated_cost_usd.toFixed(4);
      break;

    case "voice_cx":
      // Experimental / shadow mode: voice-only signal (Stage 1 + Stage 2),
      // shown for comparison against the text-based cards above it. Never
      // drives the agent's actual response.
      voiceStressValue.textContent = pct(data.stress);
      voiceFrustrationValue.textContent = pct(data.frustration);
      voiceUrgencyValue.textContent = pct(data.urgency);
      voiceEscalationValue.textContent = pct(data.escalation_risk);
      voiceEmotionValue.textContent = (data.emotion || "—").toUpperCase();
      markStep("voicecx", "done");
      break;

    case "voice_cx_toggle":
      // Toggle state changed (either on this client or another connected client)
      voiceCxEnabled = data.enabled;
      updateVoiceCxToggleDisplay();
      break;
  }
}

// ── Enterprise Panel Handlers ─────────────────────────────────────────────────

function setBadge(el, text, kind) {
  el.textContent = text;
  el.className = "badge " + (kind === "pass" ? "badge-pass" : kind === "flag" ? "badge-flag" : "badge-neutral");
}

function handleToolCall(data) {
  const placeholder = toolFeed.querySelector(".tool-placeholder");
  if (placeholder) placeholder.remove();

  const entry = document.createElement("div");
  entry.className = "tool-entry";
  const argsStr = Object.keys(data.arguments || {}).length
    ? JSON.stringify(data.arguments)
    : "";
  entry.innerHTML = `
    <div class="tool-entry-name">⚡ ${data.tool_name}${argsStr ? " " + argsStr : ""}</div>
    <div class="tool-entry-result"></div>`;
  entry.querySelector(".tool-entry-result").textContent = data.result;
  toolFeed.prepend(entry);

  while (toolFeed.children.length > 6) {
    toolFeed.removeChild(toolFeed.lastChild);
  }
}

function handleGuardrail(data) {
  const kind = data.category === "clean" ? "pass" : "flag";
  const label = data.category === "clean" ? "CLEAN" : data.category.toUpperCase();
  const reason = data.category === "clean"
    ? "No PII, injection, or unsafe content detected."
    : `Flags: ${(data.flags || []).join(", ")} (severity: ${data.severity})`;

  if (data.checkpoint === "input") {
    setBadge(guardInputBadge, label, kind);
    guardInputReason.textContent = reason;
  } else {
    setBadge(guardOutputBadge, label, kind);
    guardOutputReason.textContent = reason;
  }
}

function handleKnowledge(data) {
  if (data.matched) {
    setBadge(kbBadge, "MATCHED " + data.faq_id, "pass");
    kbBody.innerHTML = `
      <div class="kb-question">${data.question}</div>
      <div class="kb-answer">${data.answer}</div>
      <div class="kb-score">match score: ${data.score}</div>`;
  } else {
    setBadge(kbBadge, "KNOWLEDGE GAP", "flag");
    kbBody.innerHTML = `<div class="kb-answer">No FAQ article matched this question (best score: ${data.score}). Logged as a knowledge gap for the KB team.</div>`;
  }
}

function handleEvaluation(data) {
  setBadge(evalResolutionBadge, data.resolution_signal ? "DETECTED" : "NOT YET", data.resolution_signal ? "pass" : "neutral");
  setBadge(evalCsatBadge, Math.round(data.csat_prediction * 100) + "%", data.csat_prediction >= 0.5 ? "pass" : "flag");
  setBadge(evalComplianceBadge, data.policy_compliant ? "COMPLIANT" : "NON-COMPLIANT", data.policy_compliant ? "pass" : "flag");
  evalComplianceReason.textContent = data.compliance_reason || "";
  setBadge(evalHallucinationBadge, data.hallucination_flag ? "FLAGGED" : "CLEAR", data.hallucination_flag ? "flag" : "pass");
  evalHallucinationReason.textContent = data.hallucination_reason || "";
}

function handleObservability(data) {
  if (data.stage === "stt") {
    // Deepgram here is a streaming (websocket) STT -- livekit-agents reports
    // request `duration` as always 0.0 for streaming STT (there's no single
    // request round-trip to time), so show the real signal we do have:
    // how much audio this turn's transcript was built from.
    obsStt.textContent = data.audio_duration_s + "s audio";
  } else if (data.stage === "llm") {
    obsLlm.textContent = `${data.ttft_ms} / ${data.duration_ms} ms`;
    obsTokens.textContent = `${data.prompt_tokens}+${data.completion_tokens}=${data.total_tokens}`;
  } else if (data.stage === "tts") {
    obsTts.textContent = `${data.ttfb_ms} / ${data.duration_ms} ms`;
    markStep("tts", "done");
  }
}

// ── Auth (login / signup / guest / history) ──────────────────────────────────
// Optional by design: closing the card or clicking "Continue as guest" works
// exactly like the demo did before this existed -- a random per-tab identity,
// nothing persisted. Logging in swaps that for a stable "user-<id>" identity
// so the agent (see agent.py:_resolve_session_user) can recognize the same
// person across calls, skip re-verifying their identity, and this page can
// show their past conversations.

function loadAuthSession() {
  try {
    const raw = localStorage.getItem("adaptivecx-auth");
    authSession = raw ? JSON.parse(raw) : null;
  } catch (e) {
    authSession = null;
  }
}

function saveAuthSession(session) {
  authSession = session;
  localStorage.setItem("adaptivecx-auth", JSON.stringify(session));
}

function clearAuthSession() {
  authSession = null;
  localStorage.removeItem("adaptivecx-auth");
}

function updateAccountUI() {
  if (authSession) {
    accountName.textContent = "👋 " + authSession.display_name;
    accountName.style.display = "inline";
    historyBtn.style.display = "inline-flex";
    transactionsBtn.style.display = "inline-flex";
    contactsBtn.style.display = "inline-flex";
    logoutBtn.style.display = "inline-flex";
    loginOpenBtn.style.display = "none";
  } else {
    accountName.style.display = "none";
    historyBtn.style.display = "none";
    transactionsBtn.style.display = "none";
    contactsBtn.style.display = "none";
    logoutBtn.style.display = "none";
    loginOpenBtn.style.display = "inline-flex";
  }
}

function setAuthMode(mode) {
  authMode = mode;
  tabLogin.classList.toggle("active", mode === "login");
  tabSignup.classList.toggle("active", mode === "signup");
  authDisplayName.style.display = mode === "signup" ? "block" : "none";
  authDob.style.display = mode === "signup" ? "block" : "none";
  authDisplayName.required = mode === "signup";
  authDob.required = mode === "signup";
  authSubmitBtn.textContent = mode === "signup" ? "Sign up" : "Log in";
  authError.style.display = "none";
}

function openAuthOverlay() {
  authError.style.display = "none";
  authForm.reset();
  authOverlay.classList.remove("hidden");
  authUsername.focus();
}

function closeAuthOverlay() {
  authOverlay.classList.add("hidden");
  sessionStorage.setItem("adaptivecx-auth-dismissed", "1");
}

async function handleAuthSubmit(e) {
  e.preventDefault();
  authError.style.display = "none";
  authSubmitBtn.disabled = true;
  try {
    const path = authMode === "signup" ? "/signup" : "/login";
    const body = { username: authUsername.value.trim(), password: authPassword.value };
    if (authMode === "signup") {
      body.display_name = authDisplayName.value.trim();
      body.date_of_birth = authDob.value;
    }

    const resp = await fetch(AUTH_URL + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Something went wrong.");

    saveAuthSession({ token: data.token, user_id: data.user_id, display_name: data.display_name });
    updateAccountUI();
    closeAuthOverlay();
    setStatus(
      data.already_verified
        ? `Welcome back, ${data.display_name} — your identity is already verified.`
        : `Welcome, ${data.display_name}!`,
      "connected"
    );
  } catch (err) {
    authError.textContent = err.message || "Could not reach the auth server.";
    authError.style.display = "block";
  } finally {
    authSubmitBtn.disabled = false;
  }
}

function logout() {
  clearAuthSession();
  updateAccountUI();
  closeHistoryPanel();
}

function formatHistoryTime(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

async function openHistoryPanel() {
  if (!authSession) return;
  historyPanel.classList.remove("hidden");
  historyBackdrop.classList.remove("hidden");
  historyBody.innerHTML = '<div class="history-placeholder">Loading…</div>';
  try {
    const resp = await fetch(AUTH_URL + "/history", {
      headers: { "Authorization": "Bearer " + authSession.token },
    });
    if (!resp.ok) throw new Error("Could not load history.");
    const data = await resp.json();
    renderHistory(data.turns || []);
  } catch (err) {
    historyBody.innerHTML = `<div class="history-placeholder">${err.message}</div>`;
  }
}

function closeHistoryPanel() {
  historyPanel.classList.add("hidden");
  historyBackdrop.classList.add("hidden");
}

const TXN_STATUS_LABEL = {
  posted: "Posted",
  duplicate_flagged: "⚠️ Duplicate flagged",
  disputed_under_review: "🔎 Under review",
  approved_for_refund: "✅ Approved — refund pending",
  refund_denied: "❌ Refund denied",
  refunded: "✅ Refunded",
  escalated: "📞 Escalated to specialist",
};

// Client-side cache so a "transaction_update" event arriving mid-call can
// patch the panel live, even if it isn't open yet -- the next time the
// customer opens it, it's already current instead of waiting on a re-fetch.
let cachedTransactionsData = null;

function handleTransactionUpdate(data) {
  transactionsBtn.classList.add("has-update");
  if (!cachedTransactionsData) return;
  const t = cachedTransactionsData.transactions.find(x => x.transaction_id === data.transaction_id);
  if (!t) return;
  t.status = data.status;
  t.reason = data.reason;
  t.decided_by = data.decided_by;
  t.note = data.note;
  t.updated_at = Date.now() / 1000;
  t.history = t.history || [];
  t.history.push({
    transaction_id: data.transaction_id, status: data.status, reason: data.reason,
    decided_by: data.decided_by, note: data.note, created_at: Date.now() / 1000,
  });
  if (!transactionsPanel.classList.contains("hidden")) {
    renderTransactions(cachedTransactionsData);
  }
}

// A transfer creates a transaction the panel never had a snapshot of --
// prepend it live instead of waiting for the customer to reopen the panel.
function handleNewTransaction(data) {
  transactionsBtn.classList.add("has-update");
  if (!cachedTransactionsData) return;
  cachedTransactionsData.transactions.unshift({
    transaction_id: data.transaction_id, date: data.date, merchant: data.merchant,
    amount: data.amount, status: data.status, history: [],
  });
  if (!transactionsPanel.classList.contains("hidden")) {
    renderTransactions(cachedTransactionsData);
  }
}

// The account balance line updates the instant it actually changes (e.g.
// right after a transfer completes) instead of only on next open/refresh.
function handleBalanceUpdate(data) {
  transactionsBtn.classList.add("has-update");
  if (!cachedTransactionsData) return;
  cachedTransactionsData.balance = data.balance;
  if (!transactionsPanel.classList.contains("hidden")) {
    renderTransactions(cachedTransactionsData);
  }
}

async function openTransactionsPanel() {
  if (!authSession) return;
  transactionsPanel.classList.remove("hidden");
  transactionsBackdrop.classList.remove("hidden");
  transactionsBody.innerHTML = '<div class="history-placeholder">Loading…</div>';
  try {
    const resp = await fetch(AUTH_URL + "/transactions", {
      headers: { "Authorization": "Bearer " + authSession.token },
    });
    if (!resp.ok) throw new Error("Could not load transactions.");
    const data = await resp.json();
    cachedTransactionsData = data;
    transactionsBtn.classList.remove("has-update");
    renderTransactions(data);
  } catch (err) {
    transactionsBody.innerHTML = `<div class="history-placeholder">${err.message}</div>`;
  }
}

function closeTransactionsPanel() {
  transactionsPanel.classList.add("hidden");
  transactionsBackdrop.classList.add("hidden");
}

const DECIDED_BY_LABEL = {
  system: "🤖 Fraud monitoring (automatic)",
  review_team: "🧑‍💼 Review team",
  agent: "🎧 Your agent",
  customer: "🗣️ You",
};

function renderTransactions(data) {
  const header = document.createElement("div");
  header.className = "history-room-divider";
  const who = data.full_name ? `${data.full_name}${data.date_of_birth ? " · DOB " + data.date_of_birth : ""} — ` : "";
  header.textContent = `${who}${data.account_id} · ${data.tier} tier · Balance $${data.balance.toFixed(2)}`;
  transactionsBody.innerHTML = "";
  transactionsBody.appendChild(header);

  for (const t of data.transactions) {
    const wrap = document.createElement("div");
    wrap.className = "history-turn agent";
    const statusLabel = TXN_STATUS_LABEL[t.status] || t.status;
    const decidedByLabel = t.decided_by ? (DECIDED_BY_LABEL[t.decided_by] || t.decided_by) : "";

    let detailHtml = "";
    if (t.reason || t.note) {
      detailHtml += `<div class="h-detail">`;
      if (t.reason) detailHtml += `<div class="h-detail-line"><b>Reason given:</b> ${t.reason}</div>`;
      if (t.note) detailHtml += `<div class="h-detail-line"><b>Outcome:</b> ${t.note}</div>`;
      if (decidedByLabel) detailHtml += `<div class="h-detail-line"><b>Decided by:</b> ${decidedByLabel}</div>`;
      detailHtml += `</div>`;
    }
    if (t.history && t.history.length > 1) {
      detailHtml += `<div class="h-detail h-timeline"><b>Timeline:</b><ul>`;
      for (const ev of t.history) {
        const time = new Date(ev.created_at * 1000).toLocaleTimeString();
        detailHtml += `<li>${time} — ${TXN_STATUS_LABEL[ev.status] || ev.status}${ev.note ? " — " + ev.note : ""}</li>`;
      }
      detailHtml += `</ul></div>`;
    }

    wrap.innerHTML = `
      <div class="h-speaker">${t.transaction_id}</div>
      <div class="h-text"></div>
      <div class="h-time">${t.date} · $${t.amount.toFixed(2)} · ${statusLabel}</div>
      ${detailHtml}
    `;
    wrap.querySelector(".h-text").textContent = t.merchant;
    transactionsBody.appendChild(wrap);
  }

  const hint = document.createElement("div");
  hint.className = "history-placeholder";
  hint.textContent = "Ask the agent about any of these on your call, e.g. \"what was that Amazon charge?\"";
  transactionsBody.appendChild(hint);
}

// ── Contacts panel (valid money-transfer recipients) ─────────────────────────
// Every other registered customer -- the same set agent/tools.py:find_contact
// searches on a call. Account ID is shown (not DOB) since another customer's
// DOB has no reason to be visible here; see auth_server.py's /contacts.
async function openContactsPanel() {
  if (!authSession) return;
  contactsPanel.classList.remove("hidden");
  contactsBackdrop.classList.remove("hidden");
  contactsBody.innerHTML = '<div class="history-placeholder">Loading…</div>';
  try {
    const resp = await fetch(AUTH_URL + "/contacts", {
      headers: { "Authorization": "Bearer " + authSession.token },
    });
    if (!resp.ok) throw new Error("Could not load contacts.");
    const data = await resp.json();
    renderContacts(data.contacts);
  } catch (err) {
    contactsBody.innerHTML = `<div class="history-placeholder">${err.message}</div>`;
  }
}

function closeContactsPanel() {
  contactsPanel.classList.add("hidden");
  contactsBackdrop.classList.add("hidden");
}

function renderContacts(contacts) {
  contactsBody.innerHTML = "";
  if (!contacts.length) {
    contactsBody.innerHTML = '<div class="history-placeholder">No other registered customers yet.</div>';
    return;
  }
  for (const c of contacts) {
    const wrap = document.createElement("div");
    wrap.className = "history-turn agent";
    const dob = c.date_of_birth ? ` · DOB ${c.date_of_birth}` : "";
    wrap.innerHTML = `
      <div class="h-speaker">${c.display_name}</div>
      <div class="h-time">${c.account_id}${dob}</div>
    `;
    contactsBody.appendChild(wrap);
  }
  const hint = document.createElement("div");
  hint.className = "history-placeholder";
  hint.textContent = "Transfers can only be sent to a registered customer shown here -- ask the agent, e.g. \"send $50 to Rahul.\"";
  contactsBody.appendChild(hint);
}

function renderHistory(turns) {
  if (!turns.length) {
    historyBody.innerHTML = '<div class="history-placeholder">No conversations yet — start one!</div>';
    return;
  }
  historyBody.innerHTML = "";
  let lastRoom = null;
  for (const t of turns) {
    if (t.room !== lastRoom) {
      const divider = document.createElement("div");
      divider.className = "history-room-divider";
      divider.textContent = t.room;
      historyBody.appendChild(divider);
      lastRoom = t.room;
    }
    const wrap = document.createElement("div");
    wrap.className = "history-turn " + (t.speaker === "customer" ? "customer" : "agent");
    wrap.innerHTML = `
      <div class="h-speaker">${t.speaker === "customer" ? "You" : "AdaptiveCX"}</div>
      <div class="h-text"></div>
      <div class="h-time">${formatHistoryTime(t.created_at)}</div>
    `;
    wrap.querySelector(".h-text").textContent = t.text;
    historyBody.appendChild(wrap);
  }
  historyBody.scrollTop = historyBody.scrollHeight;
}

// ── LiveKit Room Connection ───────────────────────────────────────────────────

// Unique per browser tab/session for guests. Without this, everyone who
// opens the demo link joins as the same identity, and LiveKit treats a
// second person's join as replacing the first person's connection rather
// than adding a second participant. Logged-in visitors get a stable
// "user-<id>" identity instead (see the auth block above) so the agent can
// recognize returning callers.
function getVisitorIdentity() {
  if (authSession && authSession.user_id) return "user-" + authSession.user_id;
  let id = sessionStorage.getItem("adaptivecx-identity");
  if (!id) {
    id = "visitor-" + Math.random().toString(36).slice(2, 10);
    sessionStorage.setItem("adaptivecx-identity", id);
  }
  return id;
}

async function joinRoom() {
  const roomName = roomInput.value.trim();
  if (!roomName) { alert("Please enter a room name"); return; }

  joinBtn.disabled = true;
  setStatus("Fetching token...", "");

  try {
    // Fetch token from a token server OR use a pre-generated one
    // For demo, we construct the room URL directly
    // You need a token — if you have a token server, replace this:
    const tokenResponse = await fetch(
      `${TOKEN_URL}?room=${encodeURIComponent(roomName)}&identity=${encodeURIComponent(getVisitorIdentity())}`
    ).catch(() => null);

    let token;
    if (tokenResponse && tokenResponse.ok) {
      const data = await tokenResponse.json();
      token = data.token;
    } else {
      // Fallback: prompt user for a pre-generated token
      token = prompt(
        "Paste your LiveKit token here\n(or set up a token server at localhost:7881):"
      );
      if (!token) { joinBtn.disabled = false; return; }
    }

    // Detect the exact global namespace structure
    const LK = window.LivekitClient || window.LiveKit;
    if (!LK) {
      console.log("Global variables present:", Object.keys(window));
      throw new Error("LiveKit SDK library not loaded.");
    }
    
    // Fallbacks for namespace differences in UMD bundles
    const RoomClass = LK.Room;
    const RoomEventClass = LK.RoomEvent || LK.RoomEvents || {};
    const TrackClass = LK.Track || {};

    room = new RoomClass({
      adaptiveStream: true,
      dynacast: true,
    });

    room.on(RoomEventClass.Connected || 'connected', () => {
      isConnected = true;
      setStatus("Room joined — mic active!", "connected");
      joinBtn.style.display = "none";
      leaveBtn.style.display = "inline-flex";
    });

    room.on(RoomEventClass.Disconnected || 'disconnected', () => {
      isConnected = false;
      setStatus("Disconnected from room", "");
      joinBtn.style.display = "inline-flex";
      leaveBtn.style.display = "none";
      joinBtn.disabled = false;
    });

    room.on(RoomEventClass.TrackSubscribed || 'trackSubscribed', (track, publication, participant) => {
      const trackKindAudio = TrackClass.Kind ? TrackClass.Kind.Audio : 'audio';
      if (track.kind === trackKindAudio) {
        console.log('[Audio] Agent audio track received, attaching...');
        const element = track.attach();
        element.id = "agent-audio-" + Date.now();
        element.autoplay = true;
        element.setAttribute('playsinline', '');
        document.body.appendChild(element);
        element.play().catch(e => {
          console.warn('[Audio] Autoplay blocked, showing unlock button:', e);
          showAudioUnlockBtn();
        });
      }
    });

    room.on(RoomEventClass.AudioPlaybackStatusChanged || 'audioPlaybackChanged', () => {
      if (!room.canPlaybackAudio) {
        console.warn('[Audio] Playback blocked by browser. Showing unlock button.');
        showAudioUnlockBtn();
      } else {
        hideAudioUnlockBtn();
      }
    });

    await room.connect(LIVEKIT_URL, token);

    // Attempt to unlock audio context immediately after connect
    try {
      if (room.startAudio) {
        await room.startAudio();
        console.log('[Audio] room.startAudio() succeeded.');
      }
    } catch(e) {
      console.warn('[Audio] room.startAudio() failed, will retry on user gesture:', e);
      showAudioUnlockBtn();
    }

    await room.localParticipant.setMicrophoneEnabled(true);

  } catch (err) {
    console.error("LiveKit connection error:", err);
    setStatus("Failed to connect: " + err.message, "");
    joinBtn.disabled = false;
  }
}

async function leaveRoom() {
  if (room) {
    await room.disconnect();
    room = null;
  }
}

function updateVoiceCxToggleDisplay() {
  const voiceCxToggleBtn = document.getElementById("voiceCxToggleBtn");
  if (voiceCxEnabled) {
    voiceCxToggleBtn.textContent = "🎙️ Voice Primary: ON";
    voiceCxToggleBtn.style.background = "rgba(34,197,94,0.15)";
    voiceCxToggleBtn.style.color = "#22c55e";
  } else {
    voiceCxToggleBtn.textContent = "💬 Text Fallback (forced)";
    voiceCxToggleBtn.style.background = "rgba(156,163,175,0.15)";
    voiceCxToggleBtn.style.color = "#9ca3af";
  }
}

// ── Event Listeners ───────────────────────────────────────────────────────────

joinBtn.addEventListener("click", joinRoom);
leaveBtn.addEventListener("click", leaveRoom);
clearBtn.addEventListener("click", () => {
  conversation.innerHTML = `
    <div class="conv-placeholder">
      Conversation cleared. Speak to continue.
    </div>`;
  emotionHistory = [];
  emotionTimeline.innerHTML = `<div class="timeline-placeholder">Emotion history will appear here...</div>`;
  turnCount = 0;
  turnCountEl.textContent = "0";
});

tabLogin.addEventListener("click", () => setAuthMode("login"));
tabSignup.addEventListener("click", () => setAuthMode("signup"));
authForm.addEventListener("submit", handleAuthSubmit);
authCloseBtn.addEventListener("click", closeAuthOverlay);
authGuestBtn.addEventListener("click", closeAuthOverlay);
loginOpenBtn.addEventListener("click", openAuthOverlay);
logoutBtn.addEventListener("click", logout);
historyBtn.addEventListener("click", openHistoryPanel);
historyCloseBtn.addEventListener("click", closeHistoryPanel);
historyBackdrop.addEventListener("click", closeHistoryPanel);
transactionsBtn.addEventListener("click", openTransactionsPanel);
transactionsCloseBtn.addEventListener("click", closeTransactionsPanel);
transactionsBackdrop.addEventListener("click", closeTransactionsPanel);
contactsBtn.addEventListener("click", openContactsPanel);
contactsCloseBtn.addEventListener("click", closeContactsPanel);
contactsBackdrop.addEventListener("click", closeContactsPanel);

document.getElementById("voiceCxToggleBtn").addEventListener("click", () => {
  voiceCxEnabled = !voiceCxEnabled;
  updateVoiceCxToggleDisplay();
  // Send toggle to agent
  if (dashWs && dashWs.readyState === WebSocket.OPEN) {
    dashWs.send(JSON.stringify({
      type: "toggle_voice_cx",
      enabled: voiceCxEnabled,
    }));
  }
});

// ── Init ──────────────────────────────────────────────────────────────────────

// Connect to dashboard WebSocket immediately
connectDashboard();

// Auth: restore a saved session, or offer the login card once per tab
// (declining/guest-ing is remembered for the rest of this tab so it
// doesn't nag on every re-render).
loadAuthSession();
updateAccountUI();
if (!authSession && !sessionStorage.getItem("adaptivecx-auth-dismissed")) {
  openAuthOverlay();
}

// Show initial state
setStatus("Connecting to agent...", "");
console.log("AdaptiveCX Dashboard initialized");
console.log("Dashboard WS:", DASHBOARD_WS);
console.log("Auth URL:", AUTH_URL);
console.log("LiveKit URL:", LIVEKIT_URL);
