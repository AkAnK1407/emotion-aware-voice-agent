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

from dataclasses import dataclass
from datetime import datetime, timedelta

from livekit.agents import function_tool

import dashboard_bridge


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


# ─── Agentic Tool Definitions ─────────────────────────────────────────────────────
# The function's docstring becomes the tool description the LLM sees, and its
# type-hinted parameters become the tool's argument schema — both are inferred
# automatically by @function_tool, matching how the old @ai_callable worked.

@function_tool
async def verify_identity(full_name: str, date_of_birth: str) -> str:
    """Verify the caller's identity before discussing account details or taking
    action on the account. Call this first for any account-specific request."""
    match = (
        full_name.strip().lower() == _store.customer.full_name.lower()
        and date_of_birth.strip() == _store.customer.date_of_birth
    )
    _store.customer.identity_verified = match
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
