"""
policy_engine.py — AdaptiveCX Conversation Policy Engine
Selects the best conversation strategy based on BehaviorSignals.
Formula: Score(P) = 0.35·f_emotion + 0.30·f_stress + 0.15·f_engagement + 0.15·f_trust + 0.05·f_urgency
"""

from dataclasses import dataclass
from typing import Dict
from emotion_engine import Emotion, BehaviorSignals


# ─── Policy Definitions ──────────────────────────────────────────────────────────

@dataclass
class Policy:
    name: str
    empathy_level: float      # 0.0 – 1.0
    speaking_speed: float     # multiplier (0.8 = slower, 1.1 = faster)
    response_length: str      # "short" | "medium" | "long"
    offer_escalation: bool
    voice_style: str          # passed to Cartesia TTS
    color: str                # dashboard color
    description: str          # human-readable


POLICIES: Dict[str, Policy] = {
    "HIGH_EMPATHY": Policy(
        name="HIGH_EMPATHY",
        empathy_level=0.95,
        speaking_speed=0.85,
        response_length="medium",
        offer_escalation=True,
        voice_style="calm",
        color="#ef4444",
        description="Customer is distressed. Lead with empathy, speak slowly, validate feelings.",
    ),
    "CALM": Policy(
        name="CALM",
        empathy_level=0.75,
        speaking_speed=0.90,
        response_length="medium",
        offer_escalation=False,
        voice_style="calm",
        color="#f97316",
        description="Customer is frustrated. Stay calm, reassure, solve step by step.",
    ),
    "BALANCED": Policy(
        name="BALANCED",
        empathy_level=0.50,
        speaking_speed=1.00,
        response_length="medium",
        offer_escalation=False,
        voice_style="professional",
        color="#6b7280",
        description="Customer is neutral. Professional, helpful, concise.",
    ),
    "EFFICIENT": Policy(
        name="EFFICIENT",
        empathy_level=0.25,
        speaking_speed=1.05,
        response_length="short",
        offer_escalation=False,
        voice_style="cheerful",
        color="#22c55e",
        description="Customer is happy/engaged. Be upbeat, quick, keep momentum.",
    ),
    "ESCALATE": Policy(
        name="ESCALATE",
        empathy_level=0.85,
        speaking_speed=0.80,
        response_length="long",
        offer_escalation=True,
        voice_style="calm",
        color="#8b5cf6",
        description="High stress, low trust. Offer human escalation immediately.",
    ),
}

# Emotion fit scores per policy: how well does this policy suit each emotion?
EMOTION_FIT: Dict[str, Dict[Emotion, float]] = {
    "HIGH_EMPATHY": {
        Emotion.ANGRY:      0.95,
        Emotion.FRUSTRATED: 0.80,
        Emotion.SAD:        0.85,
        Emotion.FEARFUL:    0.88,
        Emotion.NEUTRAL:    0.35,
        Emotion.SURPRISED:  0.40,
        Emotion.HAPPY:      0.10,
    },
    "CALM": {
        Emotion.ANGRY:      0.80,
        Emotion.FRUSTRATED: 0.90,
        Emotion.SAD:        0.80,
        Emotion.FEARFUL:    0.75,
        Emotion.NEUTRAL:    0.55,
        Emotion.SURPRISED:  0.45,
        Emotion.HAPPY:      0.20,
    },
    "BALANCED": {
        Emotion.ANGRY:      0.45,
        Emotion.FRUSTRATED: 0.55,
        Emotion.SAD:        0.55,
        Emotion.FEARFUL:    0.50,
        Emotion.NEUTRAL:    0.90,
        Emotion.SURPRISED:  0.70,
        Emotion.HAPPY:      0.65,
    },
    "EFFICIENT": {
        Emotion.ANGRY:      0.05,
        Emotion.FRUSTRATED: 0.15,
        Emotion.SAD:        0.10,
        Emotion.FEARFUL:    0.10,
        Emotion.NEUTRAL:    0.65,
        Emotion.SURPRISED:  0.60,
        Emotion.HAPPY:      0.95,
    },
    "ESCALATE": {
        Emotion.ANGRY:      0.85,
        Emotion.FRUSTRATED: 0.75,
        Emotion.SAD:        0.65,
        Emotion.FEARFUL:    0.80,
        Emotion.NEUTRAL:    0.15,
        Emotion.SURPRISED:  0.20,
        Emotion.HAPPY:      0.05,
    },
}


class PolicyEngine:
    """
    Selects best conversation policy using weighted scoring formula.
    """

    def select(self, behavior: BehaviorSignals) -> Policy:
        """
        Score(P) = 0.35·f_emotion(P,E) + 0.30·f_stress(P,stress)
                 + 0.15·f_engagement + 0.15·f_trust + 0.05·f_urgency
        Returns Policy with highest score.
        """
        # Auto-escalate on extreme stress + low trust
        if behavior.stress > 0.88 and behavior.trust < 0.30:
            return POLICIES["ESCALATE"]

        scores: Dict[str, float] = {}

        for policy_name, policy in POLICIES.items():
            f_emotion = EMOTION_FIT[policy_name][behavior.emotion]

            # f_stress: how well does this policy's empathy match the stress level?
            # HIGH_EMPATHY is ideal for high stress; EFFICIENT for low stress
            stress_ideal = {
                "HIGH_EMPATHY": 0.85,
                "CALM":         0.65,
                "BALANCED":     0.40,
                "EFFICIENT":    0.10,
                "ESCALATE":     0.92,
            }[policy_name]
            f_stress = 1.0 - abs(stress_ideal - behavior.stress)

            f_engagement = behavior.engagement
            f_trust      = behavior.trust
            f_urgency    = behavior.urgency

            score = (
                0.35 * f_emotion +
                0.30 * f_stress +
                0.15 * f_engagement +
                0.15 * f_trust +
                0.05 * f_urgency
            )
            scores[policy_name] = round(score, 4)

        best_policy_name = max(scores, key=scores.get)
        return POLICIES[best_policy_name]


# ─── Quick Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from emotion_engine import EmotionEngine

    emotion_engine = EmotionEngine()
    policy_engine  = PolicyEngine()

    test_cases = [
        "I've called three times and nobody helped me! This is ridiculous!",
        "Great service, thank you so much for your help!",
        "I'm a bit confused about my bill but it's not urgent.",
        "My account has been hacked PLEASE HELP ME RIGHT NOW!!!",
        "I'm sad, I lost all my data and nobody seems to care.",
    ]

    print("\n" + "=" * 70)
    print("  AdaptiveCX — Policy Engine Test")
    print("=" * 70)

    for text in test_cases:
        behavior = emotion_engine.detect(text)
        policy   = policy_engine.select(behavior)

        print(f"\nText    : {text[:65]}...")
        print(f"Emotion : {behavior.emotion_emoji} {behavior.emotion.value.upper()} "
              f"(stress={behavior.stress:.2f}, trust={behavior.trust:.2f})")
        print(f"Policy  : ► {policy.name} — {policy.description[:55]}")
        print(f"Voice   : speed={policy.speaking_speed}x  empathy={policy.empathy_level:.0%}")
