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

import asyncio
import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from livekit.agents import function_tool

import banking_data
import dashboard_bridge
import guardrails
import policy_faq
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
    # "posted" -> normal, no dispute yet
    # "duplicate_flagged" -> system already detected this as a probable duplicate
    # "disputed_under_review" -> customer disputed a non-duplicate charge; needs
    #   a human to actually look at it, can't be auto-approved on this call
    # "approved_for_refund" -> cleared to refund (duplicate, or review approved it)
    # "refund_denied" -> reviewed and rejected
    # "refunded" -> money actually moved
    status: str
    dispute_reason: str = ""


@dataclass
class CustomerProfile:
    account_id: str
    full_name: str
    date_of_birth: str
    tier: str
    balance: float
    open_tickets: int
    identity_verified: bool = False
    card_frozen: bool = False
    upi_limit: float = 50000.0
    cc_limit: float = 5000.0
    cc_balance_due: float = 0.0
    cc_due_date: str = ""
    cc_min_payment: float = 0.0


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
            upi_limit=50000.0,
            cc_limit=6000.0,
            cc_balance_due=842.15,
            cc_due_date=(datetime.now() + timedelta(days=12)).strftime("%Y-%m-%d"),
            cc_min_payment=35.0,
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


# Fallback for unauthenticated (guest) sessions -- unverified, matches the
# original demo customer story exactly as before.
_GUEST_STORE = _MockBankingStore()
_store = _GUEST_STORE

# Logged-in customers each get their own account (tier/balance/5 transactions
# deterministically derived from user_id, see banking_data.py) instead of all
# sharing the one demo customer. Built lazily and cached for this process's
# lifetime -- a fresh livekit-agents job subprocess per call, so this cache
# is naturally scoped to however many distinct accounts call in during that
# process's life, not persisted across restarts (transactions are re-derived
# from user_id on demand anyway, so nothing is lost).
_user_stores: dict[int, _MockBankingStore] = {}

# Set once per call by agent.py:entrypoint after the joining LiveKit
# participant's identity is resolved to a logged-in account (None for guest
# sessions, which behave exactly as before -- nothing persisted).
_current_user_id: int | None = None


def _build_user_store(user_id: int, full_name: str, date_of_birth: str) -> _MockBankingStore:
    account = banking_data.build_account(user_id)
    # Deterministic base balance plus every transfer this account has ever
    # sent/received (see storage.balance_adjustments) -- so a customer's
    # balance is correct on every fresh load, in whichever process happens
    # to hydrate them next, not just within the process that ran the transfer.
    balance = account.balance + storage.get_balance_adjustments_total(user_id)
    store = _MockBankingStore.__new__(_MockBankingStore)
    store.customer = CustomerProfile(
        account_id=account.account_id, full_name=full_name, date_of_birth=date_of_birth,
        tier=account.tier, balance=balance, open_tickets=account.open_tickets,
        identity_verified=True,
        upi_limit=account.upi_limit, cc_limit=account.cc_limit,
        cc_balance_due=account.cc_balance_due, cc_due_date=account.cc_due_date,
        cc_min_payment=account.cc_min_payment,
    )
    store.transactions = list(account.transactions)
    store.tickets = []
    store.refunds = []
    store._ticket_seq = 4400 + user_id
    store._refund_seq = 900 + user_id
    return store


def set_current_user(user_id: int | None) -> None:
    """Points the mock CRM at this caller's own account, hydrated from their
    signup identity (name + DOB, saved at signup -- see auth_server.py) so a
    logged-in customer never has to state or verify their identity on the
    call. Guest sessions (user_id=None, or an old account with no saved
    identity) fall back to the original unverified demo customer."""
    global _current_user_id, _store
    _current_user_id = user_id
    if user_id is None:
        _store = _GUEST_STORE
        return
    prior = storage.get_verified_identity(user_id)
    if prior is None:
        _store = _GUEST_STORE
        return
    if user_id not in _user_stores:
        _user_stores[user_id] = _build_user_store(user_id, prior.full_name, prior.date_of_birth)
    _store = _user_stores[user_id]


def verified_customer_note() -> str:
    """A one-line addition for the system prompt when this call already
    started out verified (see set_current_user) -- tells Gemini not to
    bother re-asking. Empty string otherwise, so build_system_prompt's
    default behavior (unverified, ask normally) is unaffected."""
    if _current_user_id is not None and _store.customer.identity_verified:
        return (
            f"\n\nThe caller is already verified as {_store.customer.full_name} from their "
            f"account signup. Do not ask them to verify their identity again unless they "
            f"explicitly want to switch accounts. Greet them by name."
        )
    return ""


# ─── Agentic Tool Definitions ─────────────────────────────────────────────────────
# The function's docstring becomes the tool description the LLM sees, and its
# type-hinted parameters become the tool's argument schema — both are inferred
# automatically by @function_tool, matching how the old @ai_callable worked.

@function_tool
async def verify_identity(full_name: str, date_of_birth: str) -> str:
    """Verify the caller's identity before discussing account details or taking
    action on the account. Call this first for any account-specific request.
    Do NOT call this if the caller is already shown as verified from account
    login -- calling it again on an already-verified session is a no-op that
    just confirms, it can never un-verify them."""
    # A logged-in caller who arrived pre-verified (see set_current_user,
    # hydrated from their signup identity) must never be knocked back to
    # unverified by a redundant call here -- a mismatch below only means the
    # name they happened to say out loud didn't string-match, not that
    # they're not who the account login already proved they are.
    if _current_user_id is not None and _store.customer.identity_verified:
        result = f"Identity already verified for {guardrails.mask_name(_store.customer.full_name)} from account login -- no need to re-verify."
        await dashboard_bridge.broadcast_tool_call(
            "verify_identity", {"full_name": full_name, "date_of_birth": date_of_birth}, result,
            masked_fields=["last_name"],
        )
        return result

    normalized_dob = _normalize_dob(date_of_birth)
    match = (full_name.strip().lower(), normalized_dob) in _VALID_IDENTITIES
    if not match and _current_user_id is not None:
        # Also accept a match against THIS caller's own signup-time identity,
        # not just the fixed demo persona list -- otherwise any real signed-up
        # user who isn't one of the 5 baked-in demo names can never verify.
        on_file = storage.get_verified_identity(_current_user_id)
        if on_file is not None:
            match = (
                full_name.strip().lower() == on_file.full_name.strip().lower()
                and normalized_dob == on_file.date_of_birth
            )
    _store.customer.identity_verified = match
    if match:
        # Reflect whichever teammate actually verified, instead of always
        # showing the original demo customer's name.
        _store.customer.full_name = full_name.strip()
        if _current_user_id is not None:
            # Logged-in caller -- remember this so next call skips re-asking.
            storage.save_verified_identity(_current_user_id, full_name.strip(), normalized_dob)
    result = (
        f"Identity VERIFIED for {guardrails.mask_name(full_name)}." if match
        else "Identity verification FAILED — name or date of birth does not match our records."
    )
    await dashboard_bridge.broadcast_tool_call(
        "verify_identity", {"full_name": full_name, "date_of_birth": date_of_birth}, result,
        masked_fields=["last_name"] if match else [],
    )
    return result


@function_tool
async def lookup_customer_profile() -> str:
    """Look up the caller's account profile: tier, balance, and number of open support tickets."""
    c = _store.customer
    masked_fields: list[str] = []
    if not c.identity_verified:
        result = "Cannot retrieve profile: identity has not been verified yet. Call verify_identity first."
    else:
        # Data-privacy guardrail: the LLM only needs the balance/tier to
        # answer the caller, never the exact account number or full name --
        # masked here before the tool result ever reaches Gemini.
        result = (
            f"Account {guardrails.mask_account_id(c.account_id)} — {guardrails.mask_name(c.full_name)}, "
            f"tier={c.tier}, balance=${c.balance:,.2f}, open_tickets={c.open_tickets}, "
            f"card_frozen={c.card_frozen}"
        )
        masked_fields = ["account_id", "last_name"]
    result, pii_flags = guardrails.mask_fetched_data(result)
    masked_fields += pii_flags
    await dashboard_bridge.broadcast_tool_call("lookup_customer_profile", {}, result, masked_fields=masked_fields)
    return result


@function_tool
async def check_recent_transactions() -> str:
    """List the caller's recent transactions and flag any duplicate or disputed charges."""
    lines = [
        f"{t.transaction_id}: {t.date} {t.merchant} ${t.amount:.2f} [{t.status}]"
        for t in _store.transactions
    ]
    result = "Recent transactions:\n" + "\n".join(lines)
    result, masked_fields = guardrails.mask_fetched_data(result)
    await dashboard_bridge.broadcast_tool_call("check_recent_transactions", {}, result, masked_fields=masked_fields)
    return result


@function_tool
async def check_credit_card_bill() -> str:
    """Look up the caller's credit card bill: limit, current balance due, minimum payment, and due date."""
    c = _store.customer
    if not c.identity_verified:
        result = "Cannot retrieve credit card bill: identity has not been verified yet. Call verify_identity first."
    else:
        result = (
            f"Credit card — limit=${c.cc_limit:,.2f}, balance_due=${c.cc_balance_due:,.2f}, "
            f"minimum_payment=${c.cc_min_payment:,.2f}, due_date={c.cc_due_date or 'n/a'}"
        )
    result, masked_fields = guardrails.mask_fetched_data(result)
    await dashboard_bridge.broadcast_tool_call("check_credit_card_bill", {}, result, masked_fields=masked_fields)
    return result


@function_tool
async def block_card(reason: str) -> str:
    """Freeze the caller's card (lost, stolen, or suspicious activity). Ask why before calling --
    a real reason helps triage, but any stated reason is enough to freeze immediately since this
    is protective and reversible, unlike a refund or transfer."""
    c = _store.customer
    if not c.identity_verified:
        result = "Cannot block the card: identity has not been verified yet. Call verify_identity first."
    elif c.card_frozen:
        result = "Card is already frozen -- no action needed."
    else:
        c.card_frozen = True
        ticket_id = _store.next_ticket_id()
        _store.tickets.append({"ticket_id": ticket_id, "summary": f"Card frozen: {reason}", "priority": "high"})
        c.open_tickets += 1
        result = f"Card frozen (ticket {ticket_id}). No new charges can post until it's unfrozen or a replacement is issued."
    result, masked_fields = guardrails.mask_fetched_data(result)
    await dashboard_bridge.broadcast_tool_call("block_card", {"reason": reason}, result, masked_fields=masked_fields)
    return result


@function_tool
async def request_upi_limit_increase(new_limit: float) -> str:
    """Request an increase to the caller's UPI transaction limit. Never applies instantly, same as a
    refund -- opens a review ticket and tells the customer the standard turnaround."""
    c = _store.customer
    if not c.identity_verified:
        result = "Cannot request a UPI limit increase: identity has not been verified yet. Call verify_identity first."
    elif new_limit <= c.upi_limit:
        result = f"Requested limit (${new_limit:,.2f}) is not higher than the current limit (${c.upi_limit:,.2f}) -- nothing to change."
    else:
        ticket_id = _store.next_ticket_id()
        _store.tickets.append({
            "ticket_id": ticket_id, "priority": "normal",
            "summary": f"UPI limit increase request: ${c.upi_limit:,.2f} -> ${new_limit:,.2f}",
        })
        c.open_tickets += 1
        result = (
            f"Request {ticket_id} submitted to raise the UPI limit from ${c.upi_limit:,.2f} to "
            f"${new_limit:,.2f}. This goes through standard review, not applied on this call -- "
            f"tell the customer to expect a decision within 24-48 hours."
        )
    result, masked_fields = guardrails.mask_fetched_data(result)
    await dashboard_bridge.broadcast_tool_call("request_upi_limit_increase", {"new_limit": new_limit}, result, masked_fields=masked_fields)
    return result


# ─── Money transfer (registered customers only) ──────────────────────────────────
# Two distinct confirmation gates, matching the refund flow's state-machine
# shape rather than trusting the LLM to "be careful": find_contact resolves
# and surfaces WHO the money would go to (gate 1 -- the agent reads this back
# to the customer before doing anything else); initiate_transfer computes and
# HOLDS a pending transfer without moving anything; confirm_transfer is the
# only tool that can actually move money, and only on an explicit yes.

_pending_transfers: dict[str, dict] = {}
_transfer_seq = 500

# Data-privacy guardrail for the transfer flow: the LLM (and therefore the
# chat/voice transcript) must never see or say a recipient's real account
# number. find_contact hands out a short-lived opaque reference instead
# (same shape as transfer_id below) that maps server-side to the real
# account -- initiate_transfer takes that reference, not the account number,
# so the real digits never have to round-trip through the model at all, not
# even redacted-then-reconstructed.
_contact_refs: dict[str, str] = {}
_contact_ref_seq = 0


def _next_transfer_id() -> str:
    global _transfer_seq
    _transfer_seq += 1
    return f"TRF-{_transfer_seq}"


def _next_contact_ref() -> str:
    global _contact_ref_seq
    _contact_ref_seq += 1
    return f"CT-{_contact_ref_seq}"


def _account_id_to_user_id(account_id: str) -> int | None:
    """Account IDs are deterministically AC-{10000+user_id} (see
    banking_data.build_account) -- this inverts cleanly with no extra lookup
    table needed, as long as the account actually belongs to a real user."""
    try:
        candidate = int(account_id.strip().upper().removeprefix("AC-")) - 10000
    except ValueError:
        return None
    return candidate if candidate > 0 else None


@function_tool
async def find_contact(name: str) -> str:
    """Look up a registered AdaptiveCX customer by name before transferring money to them --
    ALWAYS call this first for any transfer. Transfers can only go to another real customer with
    an account here, never an arbitrary external payee. Read the result back to the customer to
    confirm it's the right person before calling initiate_transfer -- pass the reference ID from
    this result (e.g. 'CT-3'), never an account number, since account numbers are always masked
    here and can't be used directly. If more than one match comes back, ask the customer which one
    (by name) rather than guessing."""
    if not _store.customer.identity_verified:
        result = "Cannot look up a contact: identity has not been verified yet. Call verify_identity first."
        await dashboard_bridge.broadcast_tool_call("find_contact", {"name": name}, result)
        return result

    matches = storage.search_users_by_name(name, exclude_user_id=_current_user_id)
    masked_fields: list[str] = []
    if not matches:
        result = f"No registered AdaptiveCX customer found named '{name}'. Transfers can only be sent to another customer with an account here."
    elif len(matches) == 1:
        uid = matches[0]["user_id"]
        account_id = banking_data.build_account(uid).account_id
        ref = _next_contact_ref()
        _contact_refs[ref] = account_id
        # Data-privacy guardrail: the LLM only ever sees the masked account
        # number (for reading back to the customer) plus an opaque reference
        # to actually act on -- the real account_id stays server-side in
        # _contact_refs and is never spoken, shown, or sent to the model.
        result = (
            f"Found: {guardrails.mask_name(matches[0]['display_name'])}, account "
            f"{guardrails.mask_account_id(account_id)} (reference {ref}). Confirm this is the "
            f"right person with the customer, then call initiate_transfer with "
            f"recipient_reference='{ref}'."
        )
        masked_fields = ["recipient_last_name", "recipient_account_id"]
    else:
        lines = []
        for m in matches:
            account_id = banking_data.build_account(m["user_id"]).account_id
            ref = _next_contact_ref()
            _contact_refs[ref] = account_id
            lines.append(
                f"{guardrails.mask_name(m['display_name'])} — account "
                f"{guardrails.mask_account_id(account_id)} (reference {ref})"
            )
        result = (
            f"Multiple customers named '{name}' found:\n" + "\n".join(lines) +
            "\nAsk the customer which one (by name) to narrow it down -- do not guess. Then call "
            "initiate_transfer with that person's reference."
        )
        masked_fields = ["recipient_last_name", "recipient_account_id"]
    result, pii_flags = guardrails.mask_fetched_data(result)
    masked_fields += pii_flags
    await dashboard_bridge.broadcast_tool_call("find_contact", {"name": name}, result, masked_fields=masked_fields)
    return result


@function_tool
async def initiate_transfer(recipient_reference: str, amount: float) -> str:
    """Start a money transfer to a registered customer found via find_contact -- call find_contact
    first in every case and pass the reference ID it returned (e.g. 'CT-3') here, never an account
    number directly. This only HOLDS the transfer; no money moves here. Read the amount and
    recipient's name back to the customer and get an explicit yes/no, then call confirm_transfer
    with their answer."""
    if not _store.customer.identity_verified:
        result = "Cannot start a transfer: identity has not been verified yet. Call verify_identity first."
        await dashboard_bridge.broadcast_tool_call("initiate_transfer", {"recipient_reference": recipient_reference, "amount": amount}, result)
        return result

    recipient_account_id = _contact_refs.get(recipient_reference.strip().upper())
    masked_fields: list[str] = []
    if recipient_account_id is None:
        result = f"'{recipient_reference}' isn't a valid contact reference. Call find_contact to look up the recipient first."
    elif amount <= 0:
        result = "Transfer amount must be greater than zero."
    elif amount > _store.customer.balance:
        result = f"Cannot transfer ${amount:,.2f} -- balance is only ${_store.customer.balance:,.2f}."
    else:
        recipient_user_id = _account_id_to_user_id(recipient_account_id)
        recipient = storage.get_display_name(recipient_user_id) if recipient_user_id is not None else None
        if recipient is None:
            result = f"Contact reference '{recipient_reference}' no longer resolves to a registered account. Call find_contact again."
        elif recipient_user_id == _current_user_id:
            result = "Cannot transfer to your own account."
        else:
            transfer_id = _next_transfer_id()
            _pending_transfers[transfer_id] = {
                "sender_user_id": _current_user_id,
                "recipient_user_id": recipient_user_id,
                "recipient_name": recipient,
                "recipient_account_id": recipient_account_id,
                "amount": amount,
            }
            # The real account number never appears in this LLM-facing text --
            # only the masked form, same guardrail as find_contact.
            result = (
                f"Transfer {transfer_id} ready: ${amount:,.2f} to {guardrails.mask_name(recipient)} "
                f"({guardrails.mask_account_id(recipient_account_id)}). NOT sent yet. Read this back "
                f"to the customer and ask them to explicitly confirm yes or no, then call "
                f"confirm_transfer with transfer_id='{transfer_id}' and their answer."
            )
            masked_fields = ["recipient_last_name", "recipient_account_id"]
    result, pii_flags = guardrails.mask_fetched_data(result)
    masked_fields += pii_flags
    await dashboard_bridge.broadcast_tool_call(
        "initiate_transfer", {"recipient_reference": recipient_reference, "amount": amount}, result,
        masked_fields=masked_fields,
    )
    return result


@function_tool
async def confirm_transfer(transfer_id: str, confirm: bool) -> str:
    """Finish a transfer started by initiate_transfer. Only call this after the customer has
    explicitly said yes or no out loud -- never assume. confirm=False cancels it, nothing moves.
    confirm=True actually moves the money; this is the only tool that does."""
    masked_fields: list[str] = []
    pending = _pending_transfers.get(transfer_id)
    if pending is None:
        result = f"No pending transfer found with ID {transfer_id} -- it may have already been completed or cancelled."
    elif pending["sender_user_id"] != _current_user_id:
        result = "That transfer doesn't belong to this caller."
    elif not confirm:
        del _pending_transfers[transfer_id]
        result = f"Transfer {transfer_id} cancelled at the customer's request -- no money moved."
    else:
        amount = pending["amount"]
        if amount > _store.customer.balance:
            del _pending_transfers[transfer_id]
            result = f"Cannot complete transfer {transfer_id} -- balance is now only ${_store.customer.balance:,.2f}."
        else:
            del _pending_transfers[transfer_id]
            sender_id = pending["sender_user_id"]
            recipient_id = pending["recipient_user_id"]
            _store.customer.balance -= amount
            # Full names are fine in the DB-backed transaction history below
            # (that's each account holder's own private record, read only by
            # them via storage/auth_server, never handed to the LLM) -- the
            # data-privacy guardrail only applies to what gets returned to
            # Gemini in `result`, masked further down.
            storage.save_balance_adjustment(sender_id, -amount, reason=f"Transfer to {pending['recipient_name']}", counterparty_user_id=recipient_id, transfer_id=transfer_id)
            storage.save_balance_adjustment(recipient_id, amount, reason=f"Transfer from {_store.customer.full_name}", counterparty_user_id=sender_id, transfer_id=transfer_id)
            new_txn = Transaction(
                transaction_id=transfer_id,
                date=datetime.now().strftime("%Y-%m-%d"),
                merchant=f"Transfer to {pending['recipient_name']}",
                amount=amount, status="posted",
            )
            _store.transactions.insert(0, new_txn)
            await dashboard_bridge.broadcast_new_transaction({
                "transaction_id": new_txn.transaction_id, "date": new_txn.date,
                "merchant": new_txn.merchant, "amount": new_txn.amount, "status": new_txn.status,
            })
            await dashboard_bridge.broadcast_balance_update(
                _store.customer.balance, reason=f"Sent ${amount:,.2f} to {pending['recipient_name']}",
            )
            result = (
                f"Transfer {transfer_id} complete: ${amount:,.2f} sent to "
                f"{guardrails.mask_name(pending['recipient_name'])} "
                f"({guardrails.mask_account_id(pending['recipient_account_id'])}). "
                f"New balance: ${_store.customer.balance:,.2f}."
            )
            masked_fields = ["recipient_last_name", "recipient_account_id"]
    result, pii_flags = guardrails.mask_fetched_data(result)
    masked_fields += pii_flags
    await dashboard_bridge.broadcast_tool_call(
        "confirm_transfer", {"transfer_id": transfer_id, "confirm": confirm}, result,
        masked_fields=masked_fields,
    )
    return result


_UNSET = object()


async def _record_transaction_event(
    txn: "Transaction", status: str, decided_by: str, note: str,
    reason: str | None = None, user_id: int | None | object = _UNSET,
) -> None:
    """Single choke point for every transaction status change: updates the
    in-memory record, persists it (logged-in callers only -- see storage.py)
    so it survives after the call, and broadcasts it live so the dashboard's
    Transactions panel updates in real time instead of only on next fetch.

    user_id defaults to whichever caller is "current" right now, but accepts
    an explicit override -- background review tasks (see
    _schedule_review_decision) capture the user_id at dispute time rather
    than reading the global at resolution time, since by the time a delayed
    decision fires, a different call could in principle be "current" in this
    same worker process.
    """
    uid = _current_user_id if user_id is _UNSET else user_id
    txn.status = status
    if reason is not None:
        txn.dispute_reason = reason
    if uid is not None:
        storage.save_transaction_event(
            uid, txn.transaction_id, status,
            reason=reason if reason is not None else txn.dispute_reason,
            decided_by=decided_by, note=note,
        )
    await dashboard_bridge.broadcast_transaction_update({
        "transaction_id": txn.transaction_id,
        "merchant": txn.merchant,
        "amount": txn.amount,
        "status": status,
        "reason": reason if reason is not None else txn.dispute_reason,
        "decided_by": decided_by,
        "note": note,
    })


# Background review-decision tasks (see _schedule_review_decision) must be
# held onto or asyncio will silently garbage-collect them mid-sleep.
_bg_tasks: set = set()


def _create_bg_task(coro):
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


# How long a dispute stays visibly "under review" before a decision lands.
# Real banks take 1-2 business days; this demo can't make you wait that long
# to see the outcome, but it should never look instant either -- an agent
# that can refund money the moment it's asked is the exact behavior this
# whole review flow exists to prevent. A duplicate confirmed by matching it
# against another charge on the account is closer to automated reconciliation
# (fast); a decision resting on judgment of the customer's own wording is
# modeled as a slower, real human review queue.
FAST_REVIEW_SECONDS = 6.0
SLOW_REVIEW_SECONDS = 18.0

# Set once per turn by agent.py from the session's current BehaviorSignals
# (mirrors set_escalation_authorized below) -- the ONLY place dispute/refund
# decision logic is allowed to read the customer's current emotional state,
# for the one policy rule that depends on it (see policy_faq.NEGATIVE_EMOTIONS
# and its use in _schedule_review_decision).
_current_emotion: str = "neutral"


def set_current_emotion(emotion: str) -> None:
    global _current_emotion
    _current_emotion = emotion


def _check_objective_duplicate(txn: "Transaction") -> str | None:
    """Does this charge match another one on the account (same merchant +
    amount, different transaction) that isn't already refunded? That's a
    verifiable duplicate regardless of what the customer said."""
    dupes = [
        t for t in _store.transactions
        if t.transaction_id != txn.transaction_id
        and t.merchant == txn.merchant and t.amount == txn.amount
        and t.status != "refunded"
    ]
    if not dupes:
        return None
    other = dupes[0]
    return (
        f"Matches {other.transaction_id} -- same merchant ({txn.merchant}) and amount "
        f"(${txn.amount:.2f}) charged separately. Confirmed duplicate."
    )


def _check_filing_window(txn: "Transaction") -> str | None:
    """Hard policy cutoff (policy_faq.DISPUTE_FILING_WINDOW_DAYS) -- returns a
    denial note if the transaction is older than the window, else None. This
    is a mechanical rule, not a judgment call, so it's checked first and
    denies immediately rather than waiting on a review delay."""
    try:
        txn_date = datetime.strptime(txn.date, "%Y-%m-%d").date()
    except ValueError:
        return None
    days_elapsed = (datetime.now().date() - txn_date).days
    if days_elapsed > policy_faq.DISPUTE_FILING_WINDOW_DAYS:
        return (
            f"Transaction was {days_elapsed} day(s) ago, outside the "
            f"{policy_faq.DISPUTE_FILING_WINDOW_DAYS}-day dispute filing window (policy POL-01). "
            f"Disputes filed after the window are denied automatically, regardless of reason."
        )
    return None


def _review_team_decide(txn: "Transaction", reason: str) -> tuple[str, str, float]:
    """Decides approve/deny/pending, reading only from policy_faq.py -- no
    thresholds or category rules live here, so what the agent tells a
    customer about policy (see policy_faq.FAQ_ENTRIES, surfaced through
    knowledge_base.py) can never drift from what this function actually
    enforces. Returns (decision, note, delay_seconds).

    1. Filing window (mechanical, checked by the caller before this runs).
    2. Objective: an automated cross-check against the rest of the account
       (see _check_objective_duplicate) -- if it matches, that alone approves
       it, fast, like automated reconciliation would.
    3. Otherwise, the customer's stated reason is matched against
       policy_faq.DISPUTE_CATEGORIES. An eligible category is approved; the
       merchant/satisfaction category is denied (valid charge, not a bank
       error); no match goes to "pending" -- a real reviewer, not a guess.
    """
    dup_note = _check_objective_duplicate(txn)
    if dup_note:
        return "approved", dup_note, FAST_REVIEW_SECONDS

    category = policy_faq.find_category(reason)
    if category is None:
        return "pending", (
            "Reason doesn't clearly match a known dispute category and no matching duplicate "
            "charge exists on the account -- held for a human reviewer rather than guessed."
        ), 0.0
    if not category.eligible:
        return "denied", (
            f"Category: {category.label}. {category.policy_note}"
        ), SLOW_REVIEW_SECONDS
    return "approved", (
        f"Category: {category.label}. {category.policy_note or 'Approved per policy.'}"
    ), SLOW_REVIEW_SECONDS


def _maybe_authorize_escalation_on_denial() -> None:
    """Policy (see policy_faq.NEGATIVE_EMOTIONS): a denial landing while the
    customer is currently showing visible negative emotion authorizes
    escalation immediately, bypassing the normal multi-turn dissatisfaction
    streak -- a denial is the agent's own decision-making running out of
    road, and an upset customer shouldn't just be told "no" and left there."""
    if _current_emotion in policy_faq.NEGATIVE_EMOTIONS:
        set_escalation_authorized(
            True,
            reason="A dispute was just denied while the customer was visibly upset -- "
                   "policy authorizes escalation to a specialist for a final call.",
        )


def _schedule_review_decision(txn: "Transaction", decision: str, note: str, decided_by: str, delay: float, user_id) -> None:
    """Applies an already-computed approve/deny decision after `delay`
    seconds, so the Transactions panel genuinely shows "pending" for a while
    before the outcome appears -- see FAST/SLOW_REVIEW_SECONDS above. Only
    fires if the transaction is still sitting in disputed_under_review by
    then (skips silently if something else already changed it)."""
    async def _resolve():
        await asyncio.sleep(delay)
        if txn.status != "disputed_under_review":
            return
        status = "approved_for_refund" if decision == "approved" else "refund_denied"
        if decision == "denied":
            _maybe_authorize_escalation_on_denial()
        await _record_transaction_event(txn, status, decided_by=decided_by, note=note, user_id=user_id)
    _create_bg_task(_resolve())


@function_tool
async def dispute_transaction(transaction_id: str, reason: str) -> str:
    """Open a dispute on a transaction the customer says is wrong (duplicate, wrong amount,
    charge they don't recognize, etc). Always call this BEFORE process_refund -- a refund can
    only be processed once a dispute has cleared review, which never happens instantly. Ask the
    customer why they believe the charge is wrong and pass their real reason; do not invent one."""
    if not _store.customer.identity_verified:
        result = "Cannot open a dispute: identity has not been verified yet. Call verify_identity first."
        await dashboard_bridge.broadcast_tool_call("dispute_transaction", {"transaction_id": transaction_id, "reason": reason}, result)
        return result

    txn = next((t for t in _store.transactions if t.transaction_id == transaction_id), None)
    if txn is None:
        result = f"No transaction found with ID {transaction_id}."
    elif txn.status == "refunded":
        result = f"Transaction {transaction_id} was already refunded -- nothing to dispute."
    elif txn.status == "approved_for_refund":
        result = f"Transaction {transaction_id} is already approved for refund. Call process_refund to finish it."
    elif txn.status == "disputed_under_review":
        result = f"Transaction {transaction_id} is already under review from an earlier dispute -- no need to open another, tell the customer it's still pending."
    elif txn.status == "refund_denied":
        result = f"Transaction {transaction_id} was already reviewed and the refund was denied. Offer to escalate if the customer disagrees."
    elif len(reason.strip()) < 8:
        result = "Need a real reason before a dispute can be opened -- ask the customer specifically why this charge is wrong (duplicate? wrong amount? they don't recognize it?) and call this again with that reason."
    elif (window_note := _check_filing_window(txn)) is not None:
        # Mechanical policy cutoff (policy_faq.DISPUTE_FILING_WINDOW_DAYS) --
        # denied immediately, no review queue, regardless of category or
        # reason. This is the one denial that can be instant: it isn't a
        # judgment call, it's a fixed rule, so there's nothing to "review."
        await _record_transaction_event(
            txn, "refund_denied", decided_by="system", reason=reason, note=window_note,
        )
        _maybe_authorize_escalation_on_denial()
        result = (
            f"Dispute for {transaction_id} DENIED: {window_note} Tell the customer plainly that the "
            f"filing window has passed, and offer to escalate if they disagree."
        )
    else:
        # Every dispute -- even one matching a pre-existing fraud-monitoring
        # flag -- opens as "under review" first. Nothing resolves inside this
        # same tool call; a decision lands later via _schedule_review_decision
        # and shows up in the Transactions panel's timeline, same as a real
        # bank's review queue would, not an instant refund on request.
        if txn.status == "duplicate_flagged":
            decision, note, delay = "approved", (
                "Pre-flagged by fraud monitoring as a probable duplicate before the customer even "
                "called in -- confirming against the account now."
            ), FAST_REVIEW_SECONDS
        else:
            decision, note, delay = _review_team_decide(txn, reason)

        await _record_transaction_event(
            txn, "disputed_under_review", decided_by="system", reason=reason,
            note="Dispute received and queued for review." if decision != "pending" else note,
        )
        ticket_id = _store.next_ticket_id()
        _store.tickets.append({
            "ticket_id": ticket_id,
            "summary": f"Dispute on {transaction_id} ({txn.merchant}, ${txn.amount:.2f}): {reason}",
            "priority": "normal",
        })
        _store.customer.open_tickets += 1

        if decision == "pending":
            result = (
                f"Dispute {ticket_id} opened for {transaction_id} -- {note} Tell the customer it'll "
                f"be reviewed within 1-2 business days and it is NOT approved on this call."
            )
        else:
            _schedule_review_decision(txn, decision, note, "system" if delay == FAST_REVIEW_SECONDS else "review_team", delay, _current_user_id)
            eta = "shortly" if delay == FAST_REVIEW_SECONDS else "within a couple minutes"
            result = (
                f"Dispute {ticket_id} opened for {transaction_id} and sent for review -- it is under "
                f"review right now, NOT approved yet. A decision will land {eta} and show up in the "
                f"Transactions panel; do not tell the customer it's approved or that a refund is "
                f"coming until you see that decision (call check_recent_transactions to check)."
            )
    result, masked_fields = guardrails.mask_fetched_data(result)
    await dashboard_bridge.broadcast_tool_call("dispute_transaction", {"transaction_id": transaction_id, "reason": reason}, result, masked_fields=masked_fields)
    return result


@function_tool
async def process_refund(transaction_id: str) -> str:
    """Actually move the money for a transaction that dispute_transaction has already cleared
    (status approved_for_refund). Will refuse if no dispute was opened, the dispute is still
    under review, or it was denied -- call dispute_transaction first in every case."""
    if not _store.customer.identity_verified:
        result = "Cannot process a refund: identity has not been verified yet. Call verify_identity first."
    else:
        txn = next((t for t in _store.transactions if t.transaction_id == transaction_id), None)
        if txn is None:
            result = f"No transaction found with ID {transaction_id}."
        elif txn.status == "refunded":
            result = f"Transaction {transaction_id} was already refunded."
        elif txn.status != "approved_for_refund":
            result = (
                f"Cannot refund {transaction_id} yet -- status is '{txn.status}'. Call "
                f"dispute_transaction first; it must be approved before a refund can be processed."
            )
        else:
            refund_id = _store.next_refund_id()
            _store.refunds.append({
                "refund_id": refund_id, "transaction_id": transaction_id,
                "amount": txn.amount, "reason": txn.dispute_reason,
            })
            await _record_transaction_event(
                txn, "refunded", decided_by="system",
                note=f"Refund {refund_id} of ${txn.amount:.2f} issued. Funds return in 3-5 business days.",
            )
            result = f"Refund {refund_id} of ${txn.amount:.2f} issued for {transaction_id}. Funds return in 3-5 business days."
    result, masked_fields = guardrails.mask_fetched_data(result)
    await dashboard_bridge.broadcast_tool_call("process_refund", {"transaction_id": transaction_id}, result, masked_fields=masked_fields)
    return result


@function_tool
async def create_support_ticket(summary: str, priority: str = "normal") -> str:
    """Create a support ticket to escalate or track an issue that cannot be resolved on this call."""
    ticket_id = _store.next_ticket_id()
    _store.tickets.append({"ticket_id": ticket_id, "summary": summary, "priority": priority})
    _store.customer.open_tickets += 1
    result = f"Ticket {ticket_id} created (priority={priority}): {summary}"
    result, masked_fields = guardrails.mask_fetched_data(result)
    await dashboard_bridge.broadcast_tool_call("create_support_ticket", {"summary": summary, "priority": priority}, result, masked_fields=masked_fields)
    return result


_MEETING_SLOTS = ["10:00 AM", "11:30 AM", "1:30 PM", "3:45 PM", "5:15 PM"]

# Set once per turn by agent.py from the session's rolling dissatisfaction
# signal (see _update_escalation_eligibility) -- keeps escalation reserved
# for customers who are actually repeatedly unhappy, instead of the model
# being able to hand every single query straight to a human. Defaults True
# so a hard-coded high-severity call (e.g. fraud) from the agent's own
# judgment during a guest/unscored context is never silently blocked.
_escalation_authorized = True
_escalation_gate_reason = ""


def set_escalation_authorized(authorized: bool, reason: str = "") -> None:
    global _escalation_authorized, _escalation_gate_reason
    _escalation_authorized = authorized
    _escalation_gate_reason = reason


@function_tool
async def escalate_to_specialist(summary: str, reason: str, transaction_id: str = "") -> str:
    """Escalate the caller to a live human specialist -- use this when the issue is
    outside what you can resolve on this call, or the customer is still unsatisfied
    after your best effort. Creates a priority ticket and books a callback meeting
    with a real specialist, returning a meeting link and time the customer can join.
    If this escalation is about a specific transaction (e.g. a denied refund the
    customer disagrees with), pass its transaction_id so it's reflected there too."""
    if not _escalation_authorized:
        result = (
            "Escalation not authorized yet -- this customer hasn't crossed the "
            "dissatisfaction threshold for a human handoff. " + _escalation_gate_reason +
            " Keep trying to resolve this yourself: acknowledge their frustration, "
            "offer a concrete next step, and only try escalating again if they remain "
            "unsatisfied after that."
        )
        await dashboard_bridge.broadcast_tool_call("escalate_to_specialist", {"summary": summary, "reason": reason}, result)
        return result

    ticket_id = _store.next_ticket_id()
    _store.tickets.append({"ticket_id": ticket_id, "summary": summary, "priority": "high"})
    _store.customer.open_tickets += 1
    meeting_day = (datetime.now() + timedelta(days=1)).strftime("%A, %B %d")
    slot = _MEETING_SLOTS[_store._ticket_seq % len(_MEETING_SLOTS)]
    meeting_link = f"https://meet.adaptivecx.example/{ticket_id.lower()}"
    result = (
        f"Escalation ticket {ticket_id} created (reason: {reason}). A specialist is "
        f"booked to meet you on {meeting_day} at {slot}. Meeting link: {meeting_link}"
    )

    txn = next((t for t in _store.transactions if t.transaction_id == transaction_id), None) if transaction_id else None
    if txn is not None:
        await _record_transaction_event(
            txn, "escalated", decided_by="agent", reason=reason,
            note=f"Escalated to a human specialist -- ticket {ticket_id}, meeting {meeting_day} at {slot}.",
        )

    result, masked_fields = guardrails.mask_fetched_data(result)
    await dashboard_bridge.broadcast_tool_call(
        "escalate_to_specialist",
        {"summary": summary, "reason": reason, "transaction_id": transaction_id},
        result,
        masked_fields=masked_fields,
    )
    return result


BANKING_TOOLS = [
    verify_identity,
    lookup_customer_profile,
    check_recent_transactions,
    check_credit_card_bill,
    block_card,
    request_upi_limit_increase,
    find_contact,
    initiate_transfer,
    confirm_transfer,
    dispute_transaction,
    process_refund,
    create_support_ticket,
    escalate_to_specialist,
]
