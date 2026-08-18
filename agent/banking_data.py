"""
banking_data.py — deterministic per-user mock banking data.

Each logged-in user gets a stable synthetic account (tier, balance, 5
transactions) derived from their user_id via a seeded RNG -- same user_id
always regenerates the same account, so this needs no shared state or DB
table. That matters because it's called from two places that never share
memory: agent/tools.py (a separate per-call livekit-agents subprocess) and
auth_server.py (the dashboard's HTTP API, running in the main process).
"""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

_MERCHANTS = [
    "Amazon", "Starbucks", "Uber", "Netflix", "Whole Foods", "Shell Gas",
    "Target", "Apple Store", "DoorDash", "Spotify", "Costco", "Delta Airlines",
]
_TIERS = ["standard", "priority", "premium"]


@dataclass
class Transaction:
    transaction_id: str
    date: str
    merchant: str
    amount: float
    status: str  # "posted" | "duplicate_flagged" | "refunded"


@dataclass
class Account:
    account_id: str
    tier: str
    balance: float
    open_tickets: int
    transactions: list[Transaction]


def build_account(user_id: int) -> Account:
    rng = random.Random(1000 + user_id)
    tier = rng.choice(_TIERS)
    balance = round(rng.uniform(400, 12000), 2)
    now = datetime.now()

    txns = []
    for i in range(5):
        merchant = rng.choice(_MERCHANTS)
        amount = round(rng.uniform(8, 420), 2)
        days_ago = rng.randint(0, 13)
        txns.append(Transaction(
            transaction_id=f"TXN-{user_id:04d}-{i + 1}",
            date=(now - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
            merchant=merchant, amount=amount, status="posted",
        ))
    txns.sort(key=lambda t: t.date, reverse=True)

    # ~30% of accounts carry the demo's core storyline: a duplicate charge
    # the customer needs to dispute.
    if rng.random() < 0.3:
        d = txns[0]
        txns[0] = Transaction(d.transaction_id, d.date, d.merchant, d.amount, "duplicate_flagged")
        txns[1] = Transaction(txns[1].transaction_id, d.date, d.merchant, d.amount, "duplicate_flagged")

    return Account(
        account_id=f"AC-{10000 + user_id}", tier=tier, balance=balance,
        open_tickets=0, transactions=txns,
    )
