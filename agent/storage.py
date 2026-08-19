"""
storage.py — AdaptiveCX Persistence Layer

SQLite (stdlib only, no new dependency), shared by:
  • auth_server.py  — signup/login/history HTTP API
  • agent.py/tools.py — identity hydration + chat-turn logging from within
    a live call

One file (`adaptivecx.db`, WAL mode) read/written from multiple threads and
processes (the auth server's own thread, and livekit-agents' per-job
subprocess) — WAL mode plus short-lived connections per call keeps that safe
at this scale; a real deployment would swap this module for a real DB
without touching callers.

Passwords are salted + hashed with stdlib `hashlib.pbkdf2_hmac` — no bcrypt
dependency needed, consistent with this project's existing hand-rolled HMAC
JWT signing in dashboard_bridge.py rather than pulling in an auth framework.
"""

import hashlib
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

DB_PATH = os.getenv("ADAPTIVECX_DB_PATH", os.path.join(os.path.dirname(__file__), "adaptivecx.db"))

SESSION_TTL_SECONDS = 7 * 24 * 3600  # 7 days
_PBKDF2_ITERATIONS = 200_000


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS verified_identities (
                user_id INTEGER PRIMARY KEY,
                full_name TEXT NOT NULL,
                date_of_birth TEXT NOT NULL,
                verified_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                room TEXT NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                emotion TEXT,
                policy_name TEXT,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_turns_user ON chat_turns(user_id, created_at)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transaction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                transaction_id TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                decided_by TEXT,
                note TEXT,
                created_at REAL NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_txn_events_user ON transaction_events(user_id, transaction_id, created_at)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS balance_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                delta REAL NOT NULL,
                reason TEXT,
                counterparty_user_id INTEGER,
                transfer_id TEXT,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_balance_adj_user ON balance_adjustments(user_id)")


# ─── Password hashing ────────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERATIONS).hex()


# ─── Users / sessions ────────────────────────────────────────────────────────────

@dataclass
class AuthResult:
    user_id: int
    display_name: str


def create_user(username: str, password: str, display_name: str) -> Optional[AuthResult]:
    """Returns None if the username is already taken."""
    username = username.strip().lower()
    display_name = display_name.strip() or username
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    with _connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, salt, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, salt, display_name, time.time()),
            )
        except sqlite3.IntegrityError:
            return None
        return AuthResult(user_id=cur.lastrowid, display_name=display_name)


def authenticate(username: str, password: str) -> Optional[AuthResult]:
    username = username.strip().lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, password_hash, salt, display_name FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        return None
    if _hash_password(password, row["salt"]) != row["password_hash"]:
        return None
    return AuthResult(user_id=row["id"], display_name=row["display_name"])


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now, now + SESSION_TTL_SECONDS),
        )
    return token


def resolve_session(token: str) -> Optional[int]:
    """Returns the user_id for a valid, unexpired session token, else None."""
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,)
        ).fetchone()
    if row is None or row["expires_at"] < time.time():
        return None
    return row["user_id"]


def get_display_name(user_id: int) -> Optional[str]:
    with _connect() as conn:
        row = conn.execute("SELECT display_name FROM users WHERE id = ?", (user_id,)).fetchone()
    return row["display_name"] if row else None


# ─── Verified identity (banking) ─────────────────────────────────────────────────

@dataclass
class VerifiedIdentity:
    full_name: str
    date_of_birth: str


def get_verified_identity(user_id: int) -> Optional[VerifiedIdentity]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT full_name, date_of_birth FROM verified_identities WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return None
    return VerifiedIdentity(full_name=row["full_name"], date_of_birth=row["date_of_birth"])


def save_verified_identity(user_id: int, full_name: str, date_of_birth: str):
    with _connect() as conn:
        conn.execute(
            """INSERT INTO verified_identities (user_id, full_name, date_of_birth, verified_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   full_name=excluded.full_name, date_of_birth=excluded.date_of_birth, verified_at=excluded.verified_at""",
            (user_id, full_name, date_of_birth, time.time()),
        )


# ─── Chat history ─────────────────────────────────────────────────────────────────

def save_chat_turn(user_id: int, room: str, speaker: str, text: str,
                    emotion: Optional[str] = None, policy_name: Optional[str] = None):
    with _connect() as conn:
        conn.execute(
            """INSERT INTO chat_turns (user_id, room, speaker, text, emotion, policy_name, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, room, speaker, text, emotion, policy_name, time.time()),
        )


def get_chat_history(user_id: int, limit: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT room, speaker, text, emotion, policy_name, created_at
               FROM chat_turns WHERE user_id = ? ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows][::-1]  # oldest first


# ─── Transaction dispute/refund/escalation events ────────────────────────────────
# Written from inside a live call (agent/tools.py, its own subprocess) so a
# dispute/review/refund/escalation decision survives after the call ends --
# auth_server.py's /transactions endpoint reads these back and overlays them
# on top of the deterministic base transaction list (banking_data.py) so the
# dashboard's Transactions panel reflects what actually happened on a call,
# not just the account's static starting state.

def save_transaction_event(
    user_id: int, transaction_id: str, status: str,
    reason: Optional[str] = None, decided_by: Optional[str] = None, note: Optional[str] = None,
):
    with _connect() as conn:
        conn.execute(
            """INSERT INTO transaction_events
               (user_id, transaction_id, status, reason, decided_by, note, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, transaction_id, status, reason, decided_by, note, time.time()),
        )


def get_transaction_status_overrides(user_id: int) -> dict[str, dict]:
    """Latest event per transaction_id -- what the panel should show right now."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT transaction_id, status, reason, decided_by, note, created_at FROM transaction_events
               WHERE user_id = ? AND id IN (
                   SELECT MAX(id) FROM transaction_events WHERE user_id = ? GROUP BY transaction_id
               )""",
            (user_id, user_id),
        ).fetchall()
    return {r["transaction_id"]: dict(r) for r in rows}


def search_users_by_name(query: str, exclude_user_id: Optional[int] = None) -> list[dict]:
    """Case-insensitive substring match on display_name -- backs the
    money-transfer flow's recipient lookup (agent/tools.py:find_contact).
    Transfers only ever go to another real registered customer, never an
    arbitrary external payee, so this table IS the contact list."""
    like = f"%{query.strip().lower()}%"
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id AS user_id, display_name FROM users WHERE lower(display_name) LIKE ? ORDER BY display_name",
            (like,),
        ).fetchall()
    return [dict(r) for r in rows if exclude_user_id is None or r["user_id"] != exclude_user_id]


def list_all_users(exclude_user_id: Optional[int] = None) -> list[dict]:
    """Every registered customer -- backs the dashboard's Contacts panel."""
    with _connect() as conn:
        rows = conn.execute("SELECT id AS user_id, display_name FROM users ORDER BY display_name").fetchall()
    return [dict(r) for r in rows if exclude_user_id is None or r["user_id"] != exclude_user_id]


# ─── Balance ledger (money transfers) ─────────────────────────────────────────────
# A transfer's sender and recipient are near-certainly two different users,
# who in this architecture live in two different livekit-agents job
# subprocesses (see agent.py's own docs -- one process per call) with no
# shared memory. Neither side's in-memory _store can just be mutated
# directly for the other party. This ledger is the one thing both sides
# read: a customer's effective balance is always their deterministic base
# balance (banking_data.build_account) plus the sum of every adjustment ever
# recorded against their user_id, computed fresh wherever/whenever their
# store is hydrated (see tools.py:_build_user_store).

def save_balance_adjustment(
    user_id: int, delta: float, reason: Optional[str] = None,
    counterparty_user_id: Optional[int] = None, transfer_id: Optional[str] = None,
):
    with _connect() as conn:
        conn.execute(
            """INSERT INTO balance_adjustments
               (user_id, delta, reason, counterparty_user_id, transfer_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, delta, reason, counterparty_user_id, transfer_id, time.time()),
        )


def get_balance_adjustments_total(user_id: int) -> float:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(delta), 0) AS total FROM balance_adjustments WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return float(row["total"])


def get_transaction_events(user_id: int, transaction_id: Optional[str] = None) -> list[dict]:
    """Full timeline (oldest first) -- for a "what happened and when" history view."""
    with _connect() as conn:
        if transaction_id:
            rows = conn.execute(
                """SELECT transaction_id, status, reason, decided_by, note, created_at
                   FROM transaction_events WHERE user_id = ? AND transaction_id = ? ORDER BY created_at ASC""",
                (user_id, transaction_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT transaction_id, status, reason, decided_by, note, created_at
                   FROM transaction_events WHERE user_id = ? ORDER BY created_at ASC""",
                (user_id,),
            ).fetchall()
    return [dict(r) for r in rows]
