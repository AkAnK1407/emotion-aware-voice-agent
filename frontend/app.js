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
const LIVEKIT_URL = "wss://voice-agent-9u8rfie6.ohyderabad1a.production.livekit.cloud";

// Backend (agent worker + dashboard WS + /token) hostname when not running
// locally. Update this whenever you restart the Cloudflare Tunnel / redeploy
// to Render — quick tunnels get a new random hostname each run.
const BACKEND_HOST = "tenant-incentives-procurement-memphis.trycloudflare.com";

// Login/signup/history API (auth_server.py) — separate service, separate
// tunnel, since dashboard_bridge's server can't accept anything but a
// bodyless GET. Same "update after restarting the tunnel" caveat as above.
const AUTH_HOST = "unlimited-knows-glance-creates.trycloudflare.com";

const IS_LOCAL = location.hostname === "localhost" || location.hostname === "127.0.0.1";
const DASHBOARD_WS = IS_LOCAL ? "ws://localhost:8765" : `wss://${BACKEND_HOST}/`;
const TOKEN_URL = IS_LOCAL ? "http://localhost:7881/token" : `https://${BACKEND_HOST}/token`;
const AUTH_URL = IS_LOCAL ? "http://localhost:8766" : `https://${AUTH_HOST}`;

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

const pipeSteps = {
  stt: document.getElementById("step-stt"),
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

function animatePipe(stepId) {
  const order = ["stt", "behavior", "policy", "llm", "tts"];
  let delay = 0;
  for (const key of order) {
    pipeSteps[key].className = "pipe-step";
    pipeSteps[key].querySelector(".pipe-status").textContent = "Idle";
  }
  for (const key of order) {
    ((k, d) => {
      setTimeout(() => {
        // Activate
        pipeSteps[k].className = "pipe-step active";
        pipeSteps[k].querySelector(".pipe-status").textContent = "Processing...";
        // Then mark done after 600ms
        setTimeout(() => {
          pipeSteps[k].className = "pipe-step done";
          pipeSteps[k].querySelector(".pipe-status").textContent = "Done";
        }, 600);
      }, d);
    })(key, delay);
    delay += 350;
  }
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
  dashWs = new WebSocket(DASHBOARD_WS);

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

      // ── Pipeline Animation ────────────────────────────────────────────────
      animatePipe();

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
          updateInterimBubble(data.text);
        } else {
          finalizeCustomerBubble(data.text);
        }
      }
      break;

    case "agent_response":
      addBubble("agent", data.text, null, null);
      setStatus("Agent responded — listening for next input", "connected");
      break;

    case "tool_call":
      handleToolCall(data);
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
    obsStt.textContent = data.latency_ms + " ms";
  } else if (data.stage === "llm") {
    obsLlm.textContent = `${data.ttft_ms} / ${data.duration_ms} ms`;
    obsTokens.textContent = `${data.prompt_tokens}+${data.completion_tokens}=${data.total_tokens}`;
  } else if (data.stage === "tts") {
    obsTts.textContent = `${data.ttfb_ms} / ${data.duration_ms} ms`;
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
    logoutBtn.style.display = "inline-flex";
    loginOpenBtn.style.display = "none";
  } else {
    accountName.style.display = "none";
    historyBtn.style.display = "none";
    logoutBtn.style.display = "none";
    loginOpenBtn.style.display = "inline-flex";
  }
}

function setAuthMode(mode) {
  authMode = mode;
  tabLogin.classList.toggle("active", mode === "login");
  tabSignup.classList.toggle("active", mode === "signup");
  authDisplayName.style.display = mode === "signup" ? "block" : "none";
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
    if (authMode === "signup") body.display_name = authDisplayName.value.trim();

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
