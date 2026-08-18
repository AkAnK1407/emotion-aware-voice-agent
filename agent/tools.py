"""
tools.py — AdaptiveCX Agentic Tool Layer

Gives the LLM real function-calling ability (not a text trick — this uses
livekit-agents' native `@function_tool` mechanism, the same one OpenAI/Gemini
function-calling runs through). Gemini decides, mid-conversation, whether it
needs to call one of these tools based on the function name + docstring +
argument type hints it is given — nothing here is scripted from the frontend.

Backed by an in-memory mock CRM/transaction/ticket store scoped to a single
demo customer, matching the banking demo scenario (duplicate transaction
dispute). Swapping the mock store for a real CRM/core-banking API later only
means changing `_MockBankingStore`; the tool surface (function signatures)
stays the same.

NOTE: as of livekit-agents 1.x, tools are plain `@function_tool`-decorated
functions passed to `Agent(tools=BANKING_TOOLS)` — the older
`FunctionContext`/`@ai_callable` class-based mechanism (livekit-agents 0.x)
was removed in the 1.0 API redesign.
"""

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from livekit.agents import function_tool

import dashboard_bridge
import storage

# Names + DOB that verify_identity accepts, all mapped onto the one seeded
# demo account below — lets any teammate run the demo as themselves instead
# of memorizing "Sarah Chen".
_VALID_IDENTITIES = {
    ("sarah chen", "1990-04-12"),
    ("ritik", "2000-01-01"),
    ("vanshika", "2000-01-01"),
    ("akanksha", "2000-01-01"),
    ("akash", "2000-01-01"),
}

_DOB_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y",
    "%B %d, %Y", "%B %d %Y", "%d %B %Y", "%d %B, %Y",
    "%b %d, %Y", "%b %d %Y", "%d %b %Y",
]


_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_DAY_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30,
}
_TENS_ORDINAL_PREFIX = {"twenty": 20, "thirty": 30}
_ONES_ORDINAL_SUFFIX = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9,
}
_MONTH_WORDS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTH_WORDS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})


def _spoken_day_to_digit(phrase: str) -> str | None:
    """'first' -> '1', 'twenty-first' -> '21'. Also accepts cardinal number
    words ('one', 'twenty one') since a customer might say the day either
    way."""
    words = [w for w in phrase.replace("-", " ").split() if w]
    if len(words) == 1 and words[0] in _DAY_ORDINALS:
        return str(_DAY_ORDINALS[words[0]])
    if len(words) == 1 and words[0] in _ONES:
        return str(_ONES[words[0]])
    if len(words) == 2 and words[0] in _TENS_ORDINAL_PREFIX and words[1] in _ONES_ORDINAL_SUFFIX:
        return str(_TENS_ORDINAL_PREFIX[words[0]] + _ONES_ORDINAL_SUFFIX[words[1]])
    if len(words) == 2 and words[0] in _TENS and words[1] in _ONES:
        return str(_TENS[words[0]] + _ONES[words[1]])
    return None


def _spoken_year_to_digit(phrase: str) -> str | None:
    """'two thousand' -> '2000', 'two thousand and five' -> '2005',
    'nineteen ninety two' -> '1992'. Covers the two ways people actually
    say a year out loud."""
    words = [w for w in phrase.replace("-", " ").split() if w and w != "and"]
    if not words:
        return None
    if words[0] == "two" and len(words) >= 2 and words[1] == "thousand":
        rest = words[2:]
        if not rest:
            return "2000"
        if len(rest) == 1 and rest[0] in _ONES:
            return str(2000 + _ONES[rest[0]])
        if len(rest) == 2 and rest[0] in _TENS and rest[1] in _ONES:
            return str(2000 + _TENS[rest[0]] + _ONES[rest[1]])
        if len(rest) == 1 and rest[0] in _TENS:
            return str(2000 + _TENS[rest[0]])
        return None
    if len(words) == 2 and words[0] in _ONES and 10 <= _ONES[words[0]] <= 19 and words[1] in _TENS:
        return str(_ONES[words[0]] * 100 + _TENS[words[1]])
    if (len(words) == 3 and words[0] in _ONES and 10 <= _ONES[words[0]] <= 19
            and words[1] in _TENS and words[2] in _ONES):
        return str(_ONES[words[0]] * 100 + _TENS[words[1]] + _ONES[words[2]])
    return None


def _spell_out_dob(raw: str) -> str:
    """Converts a fully-or-partly spelled-out DOB ('first of January two
    thousand', 'January first, two thousand') into digit form so the
    strptime pass below can parse it. STT + a lightweight LLM won't always
    normalize spoken numbers to digits before this tool is called -- a
    voice interface hears '2000' as the words 'two thousand', not the
    digits, and without this the customer's DOB just never matches no
    matter how many times they repeat it."""
    words = [w for w in re.sub(r"[,]", " ", raw.lower()).split() if w not in ("of", "the")]
    month_idx = next((i for i, w in enumerate(words) if w in _MONTH_WORDS), None)
    if month_idx is None:
        return raw

    month_word = words[month_idx]
    before, after = words[:month_idx], words[month_idx + 1:]

    def _digit(phrase_words, parser):
        phrase = " ".join(phrase_words)
        if not phrase:
            return None
        return phrase if phrase.strip(" -").isdigit() else parser(phrase)

    if before:
        # "first of January two thousand" -- day precedes the month.
        day_digit = _digit(before, _spoken_day_to_digit)
        year_digit = _digit(after, _spoken_year_to_digit)
    else:
        # "January first, two thousand" -- day follows the month; try every
        # split of the remaining words since there's no fixed-length marker
        # between the day phrase and the year phrase.
        day_digit = year_digit = None
        for k in range(1, len(after)):
            d = _digit(after[:k], _spoken_day_to_digit)
            y = _digit(after[k:], _spoken_year_to_digit)
            if d is not None and y is not None:
                day_digit, year_digit = d, y
                break

    if day_digit is None or year_digit is None:
        return raw

    return f"{month_word.capitalize()} {day_digit} {year_digit}"


def _normalize_dob(raw: str) -> str:
    """Best-effort normalize a spoken/typed DOB to YYYY-MM-DD, since Gemini
    may pass back '2000-01-01', 'January 1, 2000', '1st January 2000',
    'first of January two thousand', etc. depending on how it was said --
    exact string matching alone is too fragile for a voice interface."""
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", raw.strip(), flags=re.IGNORECASE)
    for candidate in (cleaned, _spell_out_dob(cleaned)):
        for fmt in _DOB_FORMATS:
            try:
                return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return raw.strip()


# ─── Mock Banking Data Store ─────────────────────────────────────────────────────

@dataclass
class Transaction:
    transaction_id: str
    date: str
    merchant: str
    amount: float
    status: str          # "posted" | "duplicate_flagged" | "refunded"


@dataclass
class CustomerProfile:
    account_id: str
    full_name: str
    date_of_birth: str
    tier: str
    balance: float
    open_tickets: int
    identity_verified: bool = False


class _MockBankingStore:
    """
    In-memory mock of a CRM + core-banking system for the demo.
    One seeded customer whose story matches the required demo scenario:
    "I've called three times already and nobody has fixed my duplicate transaction."
    """

    def __init__(self):
        self.customer = CustomerProfile(
            account_id="AC-10293",
            full_name="Sarah Chen",
            date_of_birth="1990-04-12",
            tier="priority",
            balance=2143.87,
            open_tickets=2,
        )
        now = datetime.now()
        self.transactions: list[Transaction] = [
            Transaction("TXN-8841", (now - timedelta(days=1)).strftime("%Y-%m-%d"), "Amazon", 128.50, "duplicate_flagged"),
            Transaction("TXN-8842", (now - timedelta(days=1)).strftime("%Y-%m-%d"), "Amazon", 128.50, "duplicate_flagged"),
            Transaction("TXN-8790", (now - timedelta(days=4)).strftime("%Y-%m-%d"), "Whole Foods", 64.12, "posted"),
        ]
        self.tickets: list[dict] = []
        self.refunds: list[dict] = []
        self._ticket_seq = 4400
        self._refund_seq = 900

    def next_ticket_id(self) -> str:
        self._ticket_seq += 1
        return f"TCK-{self._ticket_seq}"

    def next_refund_id(self) -> str:
        self._refund_seq += 1
        return f"RFD-{self._refund_seq}"


_store = _MockBankingStore()

# Set once per call by agent.py:entrypoint after the joining LiveKit
# participant's identity is resolved to a logged-in account (None for guest
# sessions, which behave exactly as before -- nothing persisted).
_current_user_id: int | None = None


def set_current_user(user_id: int | None) -> None:
    """Hydrates the mock CRM's identity_verified state from a prior
    verification tied to this account, so a returning logged-in customer
    never has to restate their name/DOB. Guest sessions (user_id=None) are
    untouched."""
    global _current_user_id
    _current_user_id = user_id
    if user_id is None:
        return
    prior = storage.get_verified_identity(user_id)
    if prior is not None:
        _store.customer.identity_verified = True
        _store.customer.full_name = prior.full_name
        _store.customer.date_of_birth = prior.date_of_birth


def verified_customer_note() -> str:
    """A one-line addition for the system prompt when this call already
    started out verified (see set_current_user) -- tells Gemini not to
    bother re-asking. Empty string otherwise, so build_system_prompt's
    default behavior (unverified, ask normally) is unaffected."""
    if _current_user_id is not None and _store.customer.identity_verified:
        return (
            f"\n\nThe caller is already verified as {_store.customer.full_name} from a "
            f"previous call. Do not ask them to verify their identity again unless they "
            f"explicitly want to switch accounts."
        )
    return ""


# ─── Agentic Tool Definitions ─────────────────────────────────────────────────────
# The function's docstring becomes the tool description the LLM sees, and its
# type-hinted parameters become the tool's argument schema — both are inferred
# automatically by @function_tool, matching how the old @ai_callable worked.

@function_tool
async def verify_identity(full_name: str, date_of_birth: str) -> str:
    """Verify the caller's identity before discussing account details or taking
    action on the account. Call this first for any account-specific request."""
    normalized_dob = _normalize_dob(date_of_birth)
    match = (full_name.strip().lower(), normalized_dob) in _VALID_IDENTITIES
    _store.customer.identity_verified = match
    if match:
        # Reflect whichever teammate actually verified, instead of always
        # showing the original demo customer's name.
        _store.customer.full_name = full_name.strip()
        if _current_user_id is not None:
            # Logged-in caller -- remember this so next call skips re-asking.
            storage.save_verified_identity(_current_user_id, full_name.strip(), normalized_dob)
    result = (
        f"Identity VERIFIED for {full_name}." if match
        else "Identity verification FAILED — name or date of birth does not match our records."
    )
    await dashboard_bridge.broadcast_tool_call("verify_identity", {"full_name": full_name, "date_of_birth": date_of_birth}, result)
    return result


@function_tool
async def lookup_customer_profile() -> str:
    """Look up the caller's account profile: tier, balance, and number of open support tickets."""
    c = _store.customer
    if not c.identity_verified:
        result = "Cannot retrieve profile: identity has not been verified yet. Call verify_identity first."
    else:
        result = (
            f"Account {c.account_id} — {c.full_name}, tier={c.tier}, "
            f"balance=${c.balance:,.2f}, open_tickets={c.open_tickets}"
        )
    await dashboard_bridge.broadcast_tool_call("lookup_customer_profile", {}, result)
    return result


@function_tool
async def check_recent_transactions() -> str:
    """List the caller's recent transactions and flag any duplicate or disputed charges."""
    lines = [
        f"{t.transaction_id}: {t.date} {t.merchant} ${t.amount:.2f} [{t.status}]"
        for t in _store.transactions
    ]
    result = "Recent transactions:\n" + "\n".join(lines)
    await dashboard_bridge.broadcast_tool_call("check_recent_transactions", {}, result)
    return result


@function_tool
async def process_refund(transaction_id: str, reason: str) -> str:
    """Process a refund for a specific transaction ID once a duplicate or erroneous charge is confirmed."""
    txn = next((t for t in _store.transactions if t.transaction_id == transaction_id), None)
    if txn is None:
        result = f"No transaction found with ID {transaction_id}."
    elif txn.status == "refunded":
        result = f"Transaction {transaction_id} was already refunded."
    else:
        txn.status = "refunded"
        refund_id = _store.next_refund_id()
        _store.refunds.append({"refund_id": refund_id, "transaction_id": transaction_id, "amount": txn.amount, "reason": reason})
        result = f"Refund {refund_id} of ${txn.amount:.2f} issued for {transaction_id}. Funds return in 3-5 business days."
    await dashboard_bridge.broadcast_tool_call("process_refund", {"transaction_id": transaction_id, "reason": reason}, result)
    return result


@function_tool
async def create_support_ticket(summary: str, priority: str = "normal") -> str:
    """Create a support ticket to escalate or track an issue that cannot be resolved on this call."""
    ticket_id = _store.next_ticket_id()
    _store.tickets.append({"ticket_id": ticket_id, "summary": summary, "priority": priority})
    _store.customer.open_tickets += 1
    result = f"Ticket {ticket_id} created (priority={priority}): {summary}"
    await dashboard_bridge.broadcast_tool_call("create_support_ticket", {"summary": summary, "priority": priority}, result)
    return result


BANKING_TOOLS = [
    verify_identity,
    lookup_customer_profile,
    check_recent_transactions,
    process_refund,
    create_support_ticket,
]
