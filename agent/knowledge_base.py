"""
knowledge_base.py — AdaptiveCX Knowledge Retrieval Layer

A small local FAQ set with keyword-overlap retrieval, deliberately built the
same way as `emotion_engine.py`'s keyword-softmax scoring (no embeddings model,
no external API, fully inspectable) so the whole pipeline stays consistent in
style. When nothing clears the match threshold, that's logged as a *knowledge
gap* — a real signal a production KB team would triage ("customers keep asking
X and we have no article for it").

Retrieved text is injected into the LLM's system prompt (see agent.py) as
grounding context, and the tool layer already keeps the model from inventing
account-specific numbers — this module covers the *policy/process* knowledge
side (e.g. "how long does a dispute take") that isn't in the CRM.
"""

import re
from dataclasses import dataclass

import policy_faq

FAQ_ENTRIES = [
    {
        "id": "FAQ-03",
        "question": "What should I do if I think my account was hacked?",
        "answer": "Immediately verify identity, freeze card transactions, review the last 30 days of activity for unfamiliar charges, and open a priority fraud ticket.",
        "tags": ["hacked", "fraud", "stolen", "security", "compromised", "suspicious"],
    },
    {
        "id": "FAQ-04",
        "question": "How do I read my monthly statement?",
        "answer": "Monthly statements list all posted transactions, pending holds, interest charged, and the statement closing balance; anything marked 'pending' has not yet posted and may still change.",
        "tags": ["statement", "billing cycle", "monthly", "balance"],
    },
    {
        "id": "FAQ-05",
        "question": "What are the fees for international transfers?",
        "answer": "International transfers carry a flat $25 fee plus a 1% currency conversion spread; transfers typically settle within 1-3 business days.",
        "tags": ["international", "transfer", "wire", "fees", "abroad"],
    },
    {
        "id": "FAQ-06",
        "question": "How do I close my account?",
        "answer": "Account closure requires identity verification and a zero balance; once confirmed, closure is processed within 24 hours and a confirmation letter is sent.",
        "tags": ["close account", "closing", "cancel account"],
    },
    {
        "id": "FAQ-07",
        "question": "What are your customer service hours?",
        "answer": "Voice support is available 24/7; specialist escalations for fraud and disputes are staffed 7am-11pm local time.",
        "tags": ["hours", "open", "available", "support hours", "when"],
    },
] + policy_faq.FAQ_ENTRIES

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "i", "my", "me", "you", "your",
    "to", "of", "and", "it", "this", "that", "for", "on", "in", "do", "does",
    "how", "what", "can", "please", "have", "has", "with",
}

MATCH_THRESHOLD = 0.12


@dataclass
class KnowledgeResult:
    matched: bool
    faq_id: str = ""
    question: str = ""
    answer: str = ""
    score: float = 0.0
    knowledge_gap: bool = False


_gap_log: list[str] = []


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def retrieve(text: str) -> KnowledgeResult:
    query_tokens = _tokenize(text)
    if not query_tokens:
        _gap_log.append(text)
        return KnowledgeResult(matched=False, knowledge_gap=True)

    best_entry = None
    best_score = 0.0
    for entry in FAQ_ENTRIES:
        entry_tokens = _tokenize(entry["question"]) | {t.lower() for t in entry["tags"]}
        overlap = query_tokens & entry_tokens
        score = len(overlap) / len(query_tokens)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is not None and best_score >= MATCH_THRESHOLD:
        return KnowledgeResult(
            matched=True,
            faq_id=best_entry["id"],
            question=best_entry["question"],
            answer=best_entry["answer"],
            score=round(best_score, 3),
        )

    _gap_log.append(text)
    return KnowledgeResult(matched=False, score=round(best_score, 3), knowledge_gap=True)


def get_gap_log() -> list[str]:
    """Queries that matched nothing in the KB — candidates for new FAQ articles."""
    return list(_gap_log)
