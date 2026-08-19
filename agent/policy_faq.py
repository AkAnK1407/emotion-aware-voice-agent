"""
policy_faq.py — AdaptiveCX Dispute, Refund & Escalation Policy (source of truth)

This is the ONE place the actual decision logic (tools.py's dispute/refund
review) reads its thresholds and category rules from -- and the same
structured content is exposed as FAQ entries wired into knowledge_base.py's
retrieval, so the agent can quote the real policy back to a customer instead
of describing behavior that only lives in scattered code. Keeping one
canonical source is the whole point of this file: a decision the agent makes
and an answer the agent gives about "what's your policy" can no longer drift
apart, because both read the same numbers.

NOTE for future maintainers: real US banking regulation (Regulation E for
electronic transfers, the Fair Credit Billing Act for card charges) gives
customers far longer than a day to dispute a charge -- commonly 60 days from
the statement date. DISPUTE_FILING_WINDOW_DAYS below is deliberately much
tighter for this demo, per explicit product requirement, not a claim about
real-world compliance. If that requirement changes, change it only here --
nothing else in the codebase should hardcode this number.
"""

from dataclasses import dataclass, field


# ─── Filing window ────────────────────────────────────────────────────────────

DISPUTE_FILING_WINDOW_DAYS = 1

REFUND_PROCESSING_WINDOW = "3-5 business days after approval"
REVIEW_FAST_ETA = "a few seconds to a minute (automated cross-check)"
REVIEW_SLOW_ETA = "a couple of minutes on this call for a demo; 1-2 business days for a standard case off-call"


# ─── Dispute categories ───────────────────────────────────────────────────────
# What actually decides approve/deny/pending in tools.py -- the customer's
# stated reason is matched against these categories' keywords, in order.
# `eligible=False` categories (buyer's remorse, quality complaints) are valid
# reasons to be unhappy, just not reasons a bank can refund -- those get
# redirected to the merchant, not denied silently.

@dataclass
class DisputeCategory:
    key: str
    label: str
    eligible: bool                    # can this ever be refunded by the bank?
    auto_approve_on_system_match: bool  # does an objective duplicate-match alone approve it?
    keywords: list = field(default_factory=list)
    policy_note: str = ""


DISPUTE_CATEGORIES: list[DisputeCategory] = [
    DisputeCategory(
        key="duplicate", label="Duplicate / double charge", eligible=True,
        auto_approve_on_system_match=True,
        keywords=["duplicate", "charged twice", "twice", "double charged", "same order", "two times"],
        policy_note="If it independently cross-checks against another charge on the account "
                     "(same merchant, same amount, separately posted), it's confirmed automatically "
                     "-- no human review needed.",
    ),
    DisputeCategory(
        key="unauthorized", label="Unauthorized or fraudulent charge", eligible=True,
        auto_approve_on_system_match=False,
        keywords=["don't recognize", "do not recognize", "not mine", "didn't authorize",
                   "did not authorize", "unauthorized", "stolen", "fraud", "hacked", "not authorized"],
        policy_note="Routed to human review; a fraud ticket is opened and card-freeze is offered "
                     "regardless of the refund outcome.",
    ),
    DisputeCategory(
        key="billing_error", label="Billing error (wrong amount charged)", eligible=True,
        auto_approve_on_system_match=False,
        keywords=["wrong amount", "overcharged", "billed wrong", "charged extra", "incorrect amount"],
        policy_note="Routed to human review to confirm the correct amount against the order record.",
    ),
    DisputeCategory(
        key="not_received", label="Goods or services never received", eligible=True,
        auto_approve_on_system_match=False,
        keywords=["never received", "didn't receive", "did not receive", "never arrived",
                   "never got it", "not delivered"],
        policy_note="Routed to human review; typically needs merchant/delivery confirmation, which "
                     "this line can't pull up -- expect the slower review queue.",
    ),
    DisputeCategory(
        key="merchant_quality", label="Merchant/product dispute (quality, fit, satisfaction, changed mind)",
        eligible=False, auto_approve_on_system_match=False,
        keywords=["changed my mind", "don't want it", "do not want it", "no longer need",
                   "ordered by mistake", "wrong size", "already returned", "buyer's remorse",
                   "buyers remorse", "don't like it", "do not like it", "not satisfied",
                   "poor quality", "didn't like it"],
        policy_note="The charge itself is valid -- this is a merchant-side return/satisfaction issue, "
                     "not a bank error, and is never eligible for a bank-side refund. Direct the "
                     "customer to the merchant's own return policy.",
    ),
]


def find_category(reason: str) -> DisputeCategory | None:
    """First category whose keywords appear in the customer's stated reason,
    checked in DISPUTE_CATEGORIES order (ineligible categories are checked
    last so a message like "duplicate charge, don't want it anymore" -- which
    hits both an eligible and an ineligible keyword -- resolves to the
    eligible, more specific read rather than the catch-all one)."""
    lower = reason.lower()
    eligible_first = sorted(DISPUTE_CATEGORIES, key=lambda c: c.eligible, reverse=True)
    for cat in eligible_first:
        if any(kw in lower for kw in cat.keywords):
            return cat
    return None


# ─── Escalation policy ────────────────────────────────────────────────────────
# Not every complaint goes to a human. Escalation is authorized when EITHER:
#   (a) the customer stays visibly dissatisfied for ESCALATION_STREAK_THRESHOLD
#       consecutive turns (policy_engine's own HIGH_EMPATHY/ESCALATE-worthy
#       turns, or a "Critical / Tense" conversation label), or
#   (b) a single turn is acute enough that policy_engine.select() auto-fires
#       ESCALATE outright (very high stress + very low trust), or
#   (c) a dispute comes back DENIED while the customer is showing visible
#       negative emotion (angry/frustrated/sad/fearful) -- a denial is the
#       agent's own decision-making running out of road; if the customer is
#       upset about it, policy says let a human make the final call rather
#       than the agent just repeating "no."
ESCALATION_STREAK_THRESHOLD = 2
NEGATIVE_EMOTIONS = {"angry", "frustrated", "sad", "fearful"}


# ─── FAQ entries ──────────────────────────────────────────────────────────────
# Human-readable versions of the same rules above, merged into
# knowledge_base.py's retrieval set so the agent can quote real policy back
# to a customer who asks about it, instead of describing what the code does
# from memory (which can drift) or making something up (which it must not).

FAQ_ENTRIES = [
    {
        "id": "POL-01",
        "question": "What is your policy on disputing a charge?",
        "answer": (
            f"A dispute must be filed within {DISPUTE_FILING_WINDOW_DAYS} day of the transaction date. "
            f"After that window, the charge is no longer eligible for a bank-side dispute and the "
            f"request is denied automatically, regardless of the reason given -- the customer would "
            f"need to pursue it directly with the merchant."
        ),
        "tags": ["dispute", "policy", "time limit", "deadline", "window", "how long do i have"],
    },
    {
        "id": "POL-02",
        "question": "What kinds of charges are eligible for a refund?",
        "answer": (
            "Eligible: a confirmed duplicate charge, an unauthorized/fraudulent charge, a billing "
            "error (wrong amount), or goods/services never received. NOT eligible for a bank-side "
            "refund: buyer's remorse, changed your mind, wrong size/color, or general dissatisfaction "
            "with a merchant -- those are handled through the merchant's own return policy, since the "
            "charge itself was valid and authorized."
        ),
        "tags": ["refund", "eligible", "policy", "what qualifies", "covered"],
    },
    {
        "id": "POL-03",
        "question": "How does the bank decide whether to approve or deny a dispute?",
        "answer": (
            "Two checks. First, an automatic cross-check against the rest of the account: if the "
            "same merchant and amount were genuinely charged twice, that alone confirms it, fast, "
            "no human needed. Otherwise, the customer's stated reason is matched against the dispute "
            "category it falls into (unauthorized charge, billing error, item never received, or a "
            "merchant/satisfaction issue) and a human reviewer makes the call. A vague reason that "
            "doesn't clearly fit a category is held for manual review rather than guessed either way."
        ),
        "tags": ["how", "decide", "validate", "fraud", "review", "process"],
    },
    {
        "id": "POL-04",
        "question": "How long does a refund take once approved?",
        "answer": f"Funds appear in the account {REFUND_PROCESSING_WINDOW}. No refund is issued before a decision is reached -- a dispute is never approved instantly on request.",
        "tags": ["refund", "how long", "days", "when will i get my money", "timeline"],
    },
    {
        "id": "POL-05",
        "question": "My dispute was denied -- what can I do?",
        "answer": (
            "If a dispute is denied and the customer disagrees, that's a legitimate reason to ask "
            "for escalation to a human specialist, who can review the case and make the final call "
            "independent of the automated decision. The agent should offer this rather than simply "
            "repeating the denial."
        ),
        "tags": ["denied", "denial", "rejected", "disagree", "appeal", "escalate"],
    },
    {
        "id": "POL-06",
        "question": "When do I get connected to a human specialist?",
        "answer": (
            "Escalation isn't automatic on the first complaint. It's authorized when the customer "
            "stays unhappy across multiple turns despite the agent's genuine effort to help, when a "
            "single turn is acute enough (very high stress, very low trust) to clearly need a person "
            "immediately, or when a dispute comes back denied while the customer is visibly upset "
            "about it. Outside those cases, the agent works the issue itself first."
        ),
        "tags": ["escalate", "escalation", "human", "specialist", "manager", "connect me", "talk to a person"],
    },
    {
        "id": "POL-07",
        "question": "What happens if I report my account as hacked or my card as stolen?",
        "answer": (
            "Unauthorized-charge and fraud reports are always routed to human review regardless of "
            "amount, a priority fraud ticket is opened, and the agent should offer to note the card "
            "as compromised on the account. This happens even if the specific charge in question "
            "turns out not to be refundable."
        ),
        "tags": ["hacked", "fraud", "stolen", "security", "compromised", "card"],
    },
    {
        "id": "POL-08",
        "question": "Do you give a provisional credit while a dispute is being reviewed?",
        "answer": (
            "No -- this bank's policy is that funds move only after a dispute is actually approved, "
            "never before or during review. The agent must not promise a refund, provisional or "
            "otherwise, while a dispute is still pending."
        ),
        "tags": ["provisional credit", "temporary credit", "while reviewing", "in the meantime"],
    },
    {
        "id": "POL-09",
        "question": "Who can I transfer money to?",
        "answer": (
            "Only another registered AdaptiveCX customer with an account here -- never an arbitrary "
            "external payee. The agent looks the recipient up by name first and confirms the exact "
            "account with the customer before anything is set up to send."
        ),
        "tags": ["transfer", "send money", "who", "recipient", "payee"],
    },
    {
        "id": "POL-10",
        "question": "How do you make sure a money transfer is correct before sending it?",
        "answer": (
            "Two separate checks, not one. First, the recipient is resolved by name and read back to "
            "the customer to confirm it's the right person (if more than one customer shares that "
            "name, the customer is asked which account). Second, once the recipient and amount are "
            "confirmed, the transfer is held -- nothing moves -- until the customer gives an explicit "
            "yes. A no or any hesitation cancels it with nothing sent."
        ),
        "tags": ["transfer", "confirm", "safe", "mistake", "double check", "verify"],
    },
    {
        "id": "POL-11",
        "question": "Can I raise my UPI limit right away?",
        "answer": (
            "No -- a UPI limit increase always goes through review rather than applying on the call, "
            "the same principle as a refund never being instant. Standard turnaround is 24-48 hours."
        ),
        "tags": ["upi", "limit", "increase", "raise limit"],
    },
]
