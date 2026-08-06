"""
evaluation.py — AdaptiveCX Per-Turn Evaluation Layer

Runs immediately after each agent reply and scores the turn the way an
enterprise QA/evaluation pipeline would, using signals already computed
elsewhere in the pipeline (BehaviorSignals from emotion_engine.py, the
selected Policy, whether a tool was actually invoked this turn) plus a small
amount of text analysis of the reply itself. Every score here traces back to
an explicit, inspectable rule — there is no black-box "judge" call, which
keeps it fast enough to run on every turn and easy to defend line-by-line.
"""

import re
from dataclasses import dataclass, field

RESOLUTION_PHRASES = [
    "refund", "issued", "resolved", "ticket #", "ticket created", "booked",
    "confirmed", "verified", "reversed", "credited", "fixed",
]

EMPATHY_PHRASES = [
    "sorry", "apologize", "understand", "appreciate your patience",
    "i hear you", "that must be", "frustrating", "i know this is",
]

# A dollar amount or an ID-like token (TXN-1234 / TCK-4401 / RFD-901) being
# spoken without a tool call this turn is the cheap tell for an ungrounded
# ("hallucinated") specific — a real number the model could not have known.
SPECIFIC_FIGURE_REGEX = re.compile(r"\$\d|(?:TXN|TCK|RFD|AC)-\d+", re.IGNORECASE)


@dataclass
class EvaluationResult:
    resolution_signal: bool
    resolution_matched: list[str] = field(default_factory=list)
    csat_prediction: float = 0.0
    policy_compliant: bool = True
    compliance_reason: str = ""
    hallucination_flag: bool = False
    hallucination_reason: str = ""


def _score_csat(behavior) -> float:
    """CSAT_pred = 0.5*trust + 0.3*(1-stress) + 0.2*normalized(conversation_score)

    Weighted the same way as the platform's other formulas (policy_engine.py):
    trust dominates because a trusting customer forgives a rough turn, stress
    is inverted (lower stress -> higher satisfaction), and the session-level
    conversation_score (-1..+1) is folded in at low weight so one bad turn in
    an otherwise good call doesn't crash the prediction.
    """
    conv_norm = (behavior.conversation_score + 1.0) / 2.0
    score = 0.5 * behavior.trust + 0.3 * (1.0 - behavior.stress) + 0.2 * conv_norm
    return round(min(max(score, 0.0), 1.0), 3)


def _check_policy_compliance(agent_text: str, policy) -> tuple[bool, str]:
    lower = agent_text.lower()
    if policy.empathy_level >= 0.75:
        if any(p in lower for p in EMPATHY_PHRASES):
            return True, f"{policy.name} requires visible empathy — found empathetic language."
        return False, f"{policy.name} requires empathy_level={policy.empathy_level:.0%} but no empathetic phrasing was detected."

    if policy.response_length == "short":
        word_count = len(agent_text.split())
        if word_count <= 40:
            return True, f"{policy.name} requires a concise reply — {word_count} words is within budget."
        return False, f"{policy.name} requires a short reply but the response ran {word_count} words."

    return True, f"{policy.name} has no strict compliance rule beyond baseline tone."


def evaluate_turn(customer_text: str, agent_text: str, behavior, policy, tool_called_this_turn: bool) -> EvaluationResult:
    lower_agent = agent_text.lower()

    matched = [p for p in RESOLUTION_PHRASES if p in lower_agent]
    resolution_signal = len(matched) > 0

    csat = _score_csat(behavior)
    compliant, reason = _check_policy_compliance(agent_text, policy)

    hallucination = False
    hallucination_reason = ""
    if SPECIFIC_FIGURE_REGEX.search(agent_text) and not tool_called_this_turn:
        hallucination = True
        hallucination_reason = "Response states a specific dollar amount or record ID but no tool was called this turn to source it."

    return EvaluationResult(
        resolution_signal=resolution_signal,
        resolution_matched=matched,
        csat_prediction=csat,
        policy_compliant=compliant,
        compliance_reason=reason,
        hallucination_flag=hallucination,
        hallucination_reason=hallucination_reason,
    )
