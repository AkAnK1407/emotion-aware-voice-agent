"""
guardrails.py — AdaptiveCX Guardrail Layer

Two checkpoints around the LLM, matching where a real enterprise CX platform
would place them:

  1. `check_input()`  — runs on the customer's transcribed text, before it
     reaches the emotion engine / LLM. Flags PII the customer stated out loud
     (card numbers, SSNs, etc.) and prompt-injection / jailbreak attempts
     ("ignore previous instructions", "reveal your system prompt", ...).

  2. `check_output()` — runs on the LLM's generated reply, before it is handed
     to TTS. Flags PII the model is about to *speak back* (e.g. echoing a full
     card number) and blocks a small set of clearly unsafe response patterns,
     substituting a safe fallback line instead of speaking the flagged text.

Detection here is deliberately transparent regex/keyword matching rather than
a black-box classifier — for a demo you can point at the exact line of code
that caught a given case, which matters more than raw recall for showing an
instructor *why* it works.
"""

import re
from dataclasses import dataclass, field


# ─── Patterns ────────────────────────────────────────────────────────────────────

PII_PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
}

INJECTION_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above)?\s*instructions",
    r"disregard (your|the)\s*(system\s*)?prompt",
    r"reveal (your|the)\s*(system\s*)?prompt",
    r"you are now",
    r"act as (if|though)",
    r"jailbreak",
    r"pretend (you|to)\s*(are|be)",
    r"new instructions?:",
    r"developer mode",
    r"system:\s*override",
]
INJECTION_REGEX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# Deliberately small, explicit blocklist — a real deployment would route this
# through a moderation model; here it demonstrates the *checkpoint*, not a
# production-grade classifier.
UNSAFE_OUTPUT_KEYWORDS = [
    "kill yourself", "self harm", "suicide method", "how to make a bomb",
    "i am not an ai", "i have no restrictions",
]
UNSAFE_OUTPUT_REGEX = re.compile("|".join(re.escape(k) for k in UNSAFE_OUTPUT_KEYWORDS), re.IGNORECASE)


# ─── Result Types ────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    passed: bool
    category: str                 # "clean" | "pii" | "injection" | "unsafe"
    flags: list[str] = field(default_factory=list)
    redacted_text: str = ""
    severity: str = "none"        # "none" | "low" | "medium" | "high"


def _redact_pii(text: str) -> tuple[str, list[str]]:
    redacted = text
    found: list[str] = []
    for label, pattern in PII_PATTERNS.items():
        if pattern.search(redacted):
            found.append(label)
            redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted, found


# ─── Public API ──────────────────────────────────────────────────────────────────

def check_input(text: str) -> GuardrailResult:
    """Run on the customer's utterance before it reaches emotion/LLM analysis."""
    if INJECTION_REGEX.search(text):
        return GuardrailResult(
            passed=False,
            category="injection",
            flags=["prompt_injection_pattern"],
            redacted_text=text,
            severity="high",
        )

    redacted, pii_found = _redact_pii(text)
    if pii_found:
        return GuardrailResult(
            passed=True,   # PII in customer speech doesn't block the turn, just redacts before logging/prompting
            category="pii",
            flags=pii_found,
            redacted_text=redacted,
            severity="medium",
        )

    return GuardrailResult(passed=True, category="clean", redacted_text=text)


def mask_name(full_name: str) -> str:
    """Mask every part of the name after the first before it's returned to
    the LLM as fetched CRM data -- first name stays so the agent can still
    address the caller naturally, the rest becomes an initial + asterisks
    (e.g. "Sarah Chen" -> "Sarah C***")."""
    parts = full_name.strip().split()
    if len(parts) <= 1:
        return full_name.strip()
    first, rest = parts[0], parts[1:]
    masked_rest = " ".join(f"{p[0].upper()}{'*' * max(len(p) - 1, 1)}" for p in rest)
    return f"{first} {masked_rest}"


_ACCOUNT_ID_RE = re.compile(r"^([A-Za-z]+-?)(\d+)$")


def mask_account_id(account_id: str) -> str:
    """Mask all but the last 2 digits of an account number before it's
    returned to the LLM (e.g. "AC-10293" -> "AC-***93") -- the model needs
    the account exists and its balance, not the exact number."""
    m = _ACCOUNT_ID_RE.match(account_id.strip())
    if not m:
        return account_id
    prefix, digits = m.groups()
    if len(digits) <= 2:
        return f"{prefix}{'*' * len(digits)}"
    return f"{prefix}{'*' * (len(digits) - 2)}{digits[-2:]}"


DOB_PLACEHOLDER = "[REDACTED_DOB]"
PASSWORD_PATTERN = re.compile(r"\b(password|pwd|passcode)\s*[:=]\s*\S+", re.IGNORECASE)


def mask_dob(date_of_birth: str) -> str:
    """Fully redact a date of birth before it's returned to the LLM as
    fetched CRM data -- unlike name/account there's no natural-language need
    for the model to ever see the real value here."""
    return DOB_PLACEHOLDER if date_of_birth.strip() else date_of_birth


def mask_fetched_data(text: str) -> tuple[str, list[str]]:
    """Guardrail checkpoint for data fetched from the CRM/banking store
    (account lookups, balances, transactions) before it's handed to the LLM
    as a tool result. Catches card/SSN/email/phone (shared with the input/
    output checkpoints) plus password-style strings. Name and account ID are
    masked separately at the call site (mask_name / mask_account_id /
    mask_dob) since those need custom, not blanket, redaction."""
    redacted, flags = _redact_pii(text)
    if PASSWORD_PATTERN.search(redacted):
        redacted = PASSWORD_PATTERN.sub(lambda m: f"{m.group(1)}: [REDACTED_PASSWORD]", redacted)
        flags.append("password")
    return redacted, flags


def check_output(text: str) -> GuardrailResult:
    """Run on the LLM's draft reply before it is sent to TTS."""
    if UNSAFE_OUTPUT_REGEX.search(text):
        return GuardrailResult(
            passed=False,
            category="unsafe",
            flags=["unsafe_content_pattern"],
            redacted_text="I'm not able to help with that. Let me connect you with a specialist who can assist further.",
            severity="high",
        )

    redacted, pii_found = _redact_pii(text)
    if pii_found:
        return GuardrailResult(
            passed=False,   # never speak PII back to the caller unredacted
            category="pii",
            flags=pii_found,
            redacted_text=redacted,
            severity="medium",
        )

    return GuardrailResult(passed=True, category="clean", redacted_text=text)
