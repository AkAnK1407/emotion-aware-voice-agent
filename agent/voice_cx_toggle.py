"""
voice_cx_toggle.py — Shared file-based flag for manual voice-CX enable/disable.

Allows the dashboard (browser) to signal the agent subprocess to skip voice-CX
inference and use text-based fallback instead. Uses a simple JSON file on disk,
readable/writable from both processes.

This is deliberately minimal to match the stated design principle of the codebase:
"minimal complexity, correct solution" — no bidirectional socket protocol, no IPC
library, just a file both sides can safely read/write.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("voice_cx_toggle")

TOGGLE_FILE = Path(__file__).parent / "voice_cx_toggle.json"


def is_voice_enabled() -> bool:
    """Read current voice-CX enabled state. Defaults to True if file missing/unreadable."""
    try:
        if TOGGLE_FILE.exists():
            data = json.loads(TOGGLE_FILE.read_text())
            return data.get("enabled", True)
    except Exception as e:
        logger.debug(f"[VoiceCX Toggle] failed to read toggle file: {e}")
    return True


def set_voice_enabled(enabled: bool) -> None:
    """Write voice-CX enabled state to the toggle file."""
    try:
        TOGGLE_FILE.write_text(json.dumps({"enabled": enabled}))
        logger.info(f"[VoiceCX Toggle] set to {enabled}")
    except Exception as e:
        logger.warning(f"[VoiceCX Toggle] failed to write toggle file: {e}")
