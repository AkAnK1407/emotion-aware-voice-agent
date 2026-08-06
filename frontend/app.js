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
const BACKEND_HOST = "lancaster-smithsonian-powell-motel.trycloudflare.com";

const IS_LOCAL = location.hostname === "localhost" || location.hostname === "127.0.0.1";
const DASHBOARD_WS = IS_LOCAL ? "ws://localhost:8765" : `wss://${BACKEND_HOST}/`;
const TOKEN_URL = IS_LOCAL ? "http://localhost:7881/token" : `https://${BACKEND_HOST}/token`;

// ── State ─────────────────────────────────────────────────────────────────────
let room = null;
let dashWs = null;
let turnCount = 0;
let emotionHistory = [];
let isConnected = false;

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
      // ── Update emotion ────────────────────────────────────────────────────
      emotionEmoji.textContent = data.emotion_emoji;
      emotionEmoji.classList.add("pop");
      setTimeout(() => emotionEmoji.classList.remove("pop"), 500);

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
      if (data.speaker === "customer" && !data.is_partial) {
        const emotionLabel = emotionName.textContent;
        const emotionClr = emotionCard.style.borderColor.replace("55", "");
        addBubble("customer", data.text, emotionLabel, emotionClr || "#8b5cf6");
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

// ── LiveKit Room Connection ───────────────────────────────────────────────────

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
      `${TOKEN_URL}?room=${encodeURIComponent(roomName)}&identity=dashboard-user`
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

// ── Init ──────────────────────────────────────────────────────────────────────

// Connect to dashboard WebSocket immediately
connectDashboard();

// Show initial state
setStatus("Connecting to agent...", "");
console.log("AdaptiveCX Dashboard initialized");
console.log("Dashboard WS:", DASHBOARD_WS);
console.log("LiveKit URL:", LIVEKIT_URL);
