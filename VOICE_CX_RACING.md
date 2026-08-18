# Voice-CX Racing: Primary + Text Fallback + Manual Toggle

## Overview

**Voice is now the primary emotion signal**, racing against a **3.5-second deadline**. If voice-CX finishes within that window, it drives the turn's response. Otherwise, text-based emotion detection kicks in immediately, keeping the agent responsive while voice continues in the background.

A **dashboard toggle button** lets you force text-only mode (for demos: "show the interviewer the fallback works").

## What You See

### The Source Badge
In the top-left emotion card, watch for:
- **🎙️ VOICE (validated)** — green — voice-CX won the race and is driving this turn
- **💬 TEXT (fallback)** — gray — text-based emotion was used (voice timed out or is disabled)

### The Toggle Button
In the controls bar next to "Join Room":
- **🎙️ Voice Primary: ON** (green) — racing voice-CX each turn
- **💬 Text Fallback (forced)** (gray) — skipping voice entirely, text only

Click to toggle. The state persists across page reloads and is shared with other connected dashboards.

## How It Works

### Under the Hood

1. **Text fallback is always computed first** (~1ms) — keyword-based emotion detection
2. **Voice-CX launches as a background task** — sends audio frames to `voice-cx-server/` via HTTP
3. **Race for 3.5 seconds** — whichever finishes first wins
   - Voice wins? Use its emotion/arousal/valence to build behavior + policy
   - Voice loses? Use text emotion immediately; voice continues running in background
4. **Late-arriving voice results** still update the "Voice-Based CX (experimental)" shadow panel
5. **Manual override**: if you click the toggle button, voice-CX is skipped entirely for subsequent turns

### Key Files Changed

| File | Changes |
|---|---|
| `agent/agent.py` | Racing logic in `on_user_turn_completed` (Layer 2). Added `RACE_TIMEOUT = 3.5` constant, callbacks for background completion. |
| `agent/voice_cx_toggle.py` | **NEW** — file-based flag (`voice_cx_toggle.json`) for manual enable/disable. Both agent process and dashboard server can read/write it. |
| `agent/dashboard_bridge.py` | `_handler` now receives incoming WebSocket messages; broadcasts toggle state on connect and when changed. |
| `frontend/index.html` | New toggle button in `.controls-wrap`. |
| `frontend/app.js` | Toggle button click handler, WebSocket message dispatch for toggle state, UI update function. |

## Latency Budget

**Before (sequential wait):**
- Text emotion: ~1ms
- Voice attempt: 8–20s (audio-length-scaled timeout)
- **Total per turn: 8–20 seconds** (even on timeout)

**After (parallel race):**
- Compute text: ~1ms
- Race voice: ≤3.5s (hard deadline)
- **Total per turn: ≤3.5 seconds** regardless of voice speed
- Voice can still complete later (just doesn't block the agent reply)

## Testing

### 1. Voice CX is Fast & Works
1. Ensure `voice-cx-server` is running (check `VOICE_CX_SERVER_URL` in `.env`)
2. Join the room, speak a turn
3. **Expected:** Source badge shows 🎙️ VOICE (validated), response feels snappy (3-5s total per turn)

### 2. Voice CX is Slow or Unavailable
1. Kill the `voice-cx-server` process (or set `VOICE_CX_SERVER_URL=""` in `.env`)
2. Speak a turn
3. **Expected:** Source badge shows 💬 TEXT (fallback), response comes in ~1s (bounded by race timeout, not voice budget)
4. If voice eventually recovers, its results still land in the "Voice-Based CX" shadow panel

### 3. Manual Fallback Toggle
1. Ensure voice-CX is running and fast
2. Click the "🎙️ Voice Primary: ON" button
3. **Expected:** Button text changes to "💬 Text Fallback (forced)", color goes gray
4. Speak a turn
5. **Expected:** Source badge shows 💬 TEXT (fallback) — voice was skipped entirely
6. Reload the page
7. **Expected:** Toggle state persists (button still shows gray)
8. Open another dashboard tab
9. **Expected:** Both tabs show the same state (real-time sync via WebSocket)
10. Click toggle again on tab 1
11. **Expected:** Both tabs update immediately

### 4. Late-Arriving Voice (Optional, Advanced)
1. Add an artificial delay to `voice-cx-server/main.py` to make voice slower than 3.5s but still working (e.g., `time.sleep(5)`)
2. Join room, speak
3. **Expected:** 
   - Agent replies immediately with 💬 TEXT badge (race timeout)
   - ~5s later, the "Voice-Based CX (experimental)" card updates with voice emotion/stress/etc
   - Agent's reply is NOT re-run or changed

## Tuning

**To change the race deadline**, edit `agent/agent.py`:

```python
RACE_TIMEOUT = 3.5  # ← seconds. Try 2.5 for more text, 5.0 to give voice more time
```

Lower = favor text responsiveness, higher = favor voice accuracy.

## Architecture Notes

- **Voice-CX continues in background**: Even if it loses the race, the server doesn't know to stop. The client just stops waiting. A callback handles the eventual result.
- **Cooldown after failures**: If voice fails/times out, an 8-second cooldown is set (prevents hammering a slow server with retries).
- **Shadow panel stays independent**: The "Voice-Based CX (experimental)" panel shows all voice results, whether they drove the turn or arrived late. It's still display-only — text policy always drives the actual agent reply when voice loses the race.
- **Toggle is UI-only**: Forcing text doesn't disable the voice server; it just tells the agent to skip attempting to invoke it. Useful for demos. In production, you'd want metrics-driven auto-skip (detect SLA violations) instead.

## Commit

Commit `21d9a74`: "Voice-CX racing: make voice primary with short timeout, text fallback, & manual toggle"
