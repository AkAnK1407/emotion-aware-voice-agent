"""
emotion_engine.py — AdaptiveCX Behavior Intelligence Engine
Detects emotion, stress, prosody, trust, urgency from customer text.
Uses keyword-based softmax (reliable, fast, no external API) +
optionally HuggingFace for higher accuracy.
"""

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


# ─── Data Structures ────────────────────────────────────────────────────────────

class Emotion(Enum):
    ANGRY      = "angry"
    HAPPY      = "happy"
    FRUSTRATED = "frustrated"
    SAD        = "sad"
    NEUTRAL    = "neutral"
    FEARFUL    = "fearful"
    SURPRISED  = "surprised"


EMOTION_COLORS = {
    Emotion.ANGRY:      "#ef4444",   # red
    Emotion.FRUSTRATED: "#f97316",   # orange
    Emotion.SAD:        "#3b82f6",   # blue
    Emotion.FEARFUL:    "#8b5cf6",   # purple
    Emotion.NEUTRAL:    "#6b7280",   # grey
    Emotion.SURPRISED:  "#eab308",   # yellow
    Emotion.HAPPY:      "#22c55e",   # green
}

EMOTION_EMOJI = {
    Emotion.ANGRY:      "😠",
    Emotion.FRUSTRATED: "😤",
    Emotion.SAD:        "😢",
    Emotion.FEARFUL:    "😨",
    Emotion.NEUTRAL:    "😐",
    Emotion.SURPRISED:  "😲",
    Emotion.HAPPY:      "😊",
}


@dataclass
class ProsodyFeatures:
    """Extracted from text-level signals (works without raw audio)."""
    speech_rate_estimate: float     # estimated WPM (words per second × 60)
    caps_ratio: float               # ratio of UPPERCASE words (stress indicator)
    exclamation_count: int          # ! count
    question_count: int             # ? count
    word_count: int
    avg_word_length: float
    repetition_score: float         # word repetition ratio (frustration indicator)
    punctuation_density: float      # .,!? per word


@dataclass
class BehaviorSignals:
    """Unified behavioral intelligence output consumed by policy engine."""
    # Core emotion
    emotion: Emotion
    emotion_confidence: float       # 0.0 – 1.0
    emotion_color: str
    emotion_emoji: str

    # Derived scores (all 0.0 – 1.0)
    stress: float
    engagement: float
    trust: float
    urgency: float

    # Session Conversation State (70% current turn, 30% historical)
    conversation_score: float       # -1.0 to +1.0
    conversation_trend: str         # "Improving ↗" | "Worsening ↘" | "Stable ➔"
    conversation_label: str         # "Positive" | "Neutral" | "Critical / Tense"

    # Patience level
    patience: str                   # "low" | "medium" | "high"

    # Prosody features
    prosody: ProsodyFeatures

    # Raw text that was analyzed
    text: str


# ─── Keyword Lexicon ─────────────────────────────────────────────────────────────

KEYWORD_LEXICON: Dict[Emotion, Dict[str, float]] = {
    Emotion.ANGRY: {
        "angry": 0.90, "furious": 0.95, "hate": 0.85, "terrible": 0.80,
        "awful": 0.80, "ridiculous": 0.75, "unacceptable": 0.90,
        "outraged": 0.92, "disgusting": 0.88, "worst": 0.82,
        "how dare": 0.88, "lawsuit": 0.85, "sue": 0.80,
        "never again": 0.85, "scam": 0.90, "fraud": 0.88,
    },
    Emotion.FRUSTRATED: {
        "frustrated": 0.85, "annoyed": 0.75, "disappointed": 0.80,
        "broken": 0.70, "doesn't work": 0.75, "still": 0.55,
        "again": 0.60, "keeps": 0.60, "failing": 0.72,
        "hours": 0.65, "waiting": 0.65, "wasted": 0.75,
        "called": 0.55, "times": 0.60, "repeated": 0.70,
        "cannot": 0.55, "won't": 0.55, "useless": 0.80,
        "fix this": 0.75, "not working": 0.78, "keeps crashing": 0.82,
    },
    Emotion.HAPPY: {
        "happy": 0.90, "great": 0.80, "excellent": 0.85, "amazing": 0.90,
        "love": 0.85, "perfect": 0.85, "wonderful": 0.90, "thank": 0.75,
        "appreciate": 0.80, "fantastic": 0.88, "awesome": 0.85,
        "pleased": 0.82, "delighted": 0.90, "superb": 0.88,
        "brilliant": 0.85, "helpful": 0.78,
    },
    Emotion.SAD: {
        "sad": 0.80, "depressed": 0.85, "miserable": 0.85, "unhappy": 0.80,
        "heartbroken": 0.90, "devastated": 0.88, "upset": 0.72,
        "disappointed": 0.70, "lost": 0.65, "hopeless": 0.85,
        "crying": 0.88, "tears": 0.80, "grief": 0.90,
    },
    Emotion.FEARFUL: {
        "scared": 0.85, "afraid": 0.82, "worried": 0.75, "anxious": 0.78,
        "panic": 0.88, "terrified": 0.90, "nervous": 0.72,
        "concern": 0.65, "risk": 0.60, "danger": 0.78,
        "fraud": 0.75, "hacked": 0.85, "stolen": 0.82,
    },
    Emotion.SURPRISED: {
        "wow": 0.80, "surprised": 0.82, "unexpected": 0.75,
        "unbelievable": 0.82, "shocking": 0.80, "incredible": 0.75,
        "cannot believe": 0.82, "really": 0.50, "seriously": 0.55,
    },
}

# Urgency keywords
URGENCY_KEYWORDS = {
    "urgent": 0.90, "asap": 0.95, "immediately": 0.90, "now": 0.75,
    "emergency": 0.95, "critical": 0.88, "right now": 0.85,
    "please hurry": 0.88, "quickly": 0.72, "deadline": 0.80,
}

# Negative word density
NEGATIVE_WORDS = {
    "not", "never", "no", "cannot", "can't", "won't", "don't",
    "didn't", "doesn't", "nobody", "nothing", "without", "fail",
    "failed", "problem", "issue", "error", "bug", "broken",
}


# ─── Emotion Engine ──────────────────────────────────────────────────────────────

class EmotionEngine:
    """
    Detects emotion from customer text using softmax over keyword scores.
    Maintains session-level conversation state with recency weighting (70% current, 30% history).
    """

    def __init__(self, recency_weight: float = 0.70):
        self.recency_weight = recency_weight       # Current message weight (70%)
        self.history_weight = 1.0 - recency_weight # Previous messages weight (30%)
        self.reset_session()

    def reset_session(self):
        """Reset conversation session history."""
        self.session_turns = 0
        self.conversation_score = 0.0
        self.previous_conversation_score = 0.0
        self.conversation_trend = "Stable ➔"

    def _calculate_sentence_sentiment(self, emotion: Emotion, confidence: float, stress: float) -> float:
        """Map current emotion and stress to a numerical score between -1.0 and +1.0."""
        base_map = {
            Emotion.ANGRY:      -1.0,
            Emotion.FRUSTRATED: -0.7,
            Emotion.SAD:        -0.5,
            Emotion.FEARFUL:    -0.6,
            Emotion.SURPRISED:  +0.1,
            Emotion.NEUTRAL:     0.0,
            Emotion.HAPPY:      +1.0,
        }
        raw_val = base_map.get(emotion, 0.0) * confidence
        if raw_val < 0:
            raw_val -= stress * 0.2
        elif raw_val > 0:
            raw_val -= stress * 0.1
        return min(max(raw_val, -1.0), 1.0)

    def _score_emotion(self, text_lower: str, words: list) -> Dict[Emotion, float]:
        """Compute raw score for each emotion class."""
        scores: Dict[Emotion, float] = {e: 0.0 for e in Emotion}
        scores[Emotion.NEUTRAL] = 0.1   # baseline

        for emotion, lexicon in KEYWORD_LEXICON.items():
            score = 0.0
            for i, word in enumerate(words):
                for keyword, weight in lexicon.items():
                    if keyword in text_lower:
                        # Position decay: words earlier in sentence have more weight
                        pos_idx = text_lower.find(keyword)
                        position_decay = 1.0 / math.log(pos_idx / 10 + 2)
                        score += weight * position_decay
            scores[emotion] = score

        return scores

    def _softmax(self, scores: Dict[Emotion, float]) -> Dict[Emotion, float]:
        """Apply softmax normalization."""
        max_score = max(scores.values())
        exps = {e: math.exp(s - max_score) for e, s in scores.items()}
        total = sum(exps.values())
        return {e: v / total for e, v in exps.items()}

    def _extract_prosody(self, text: str, words: list) -> ProsodyFeatures:
        """Extract prosody features from text-level signals."""
        word_count = len(words)
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        caps_ratio = caps_words / max(word_count, 1)
        exclamation_count = text.count("!")
        question_count = text.count("?")
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)

        # Repetition score: ratio of unique words to total words
        unique_ratio = len(set(w.lower() for w in words)) / max(word_count, 1)
        repetition_score = 1.0 - unique_ratio

        punct_count = sum(1 for c in text if c in ".,!?;:")
        punctuation_density = punct_count / max(word_count, 1)

        # Estimate speaking rate from word count (assume ~2 words/second for fast speech)
        speech_rate_estimate = word_count * 2.0 * 60  # rough WPM

        return ProsodyFeatures(
            speech_rate_estimate=speech_rate_estimate,
            caps_ratio=caps_ratio,
            exclamation_count=exclamation_count,
            question_count=question_count,
            word_count=word_count,
            avg_word_length=avg_word_length,
            repetition_score=repetition_score,
            punctuation_density=punctuation_density,
        )

    def _compute_stress(
        self,
        primary_emotion: Emotion,
        confidence: float,
        prosody: ProsodyFeatures,
        text_lower: str,
    ) -> float:
        """
        Stress = 0.40×S_emotion + 0.20×S_words + 0.20×S_prosody + 0.20×S_urgency
        """
        emotion_stress_map = {
            Emotion.ANGRY:      0.95,
            Emotion.FRUSTRATED: 0.75,
            Emotion.FEARFUL:    0.80,
            Emotion.SAD:        0.60,
            Emotion.SURPRISED:  0.40,
            Emotion.NEUTRAL:    0.15,
            Emotion.HAPPY:      0.08,
        }
        s_emotion = emotion_stress_map[primary_emotion] * confidence

        # Negative word density
        words_lower = text_lower.split()
        neg_count = sum(1 for w in words_lower if w in NEGATIVE_WORDS)
        s_words = min(neg_count / max(len(words_lower), 1) * 3.0, 1.0)

        # Prosody stress (CAPS, exclamations, repetition)
        s_prosody = min(
            prosody.caps_ratio * 2.0 +
            prosody.exclamation_count * 0.15 +
            prosody.repetition_score * 0.5,
            1.0
        )

        # Urgency keywords
        urgency_score = sum(
            weight for kw, weight in URGENCY_KEYWORDS.items()
            if kw in text_lower
        )
        s_urgency = min(urgency_score, 1.0)

        stress = (
            0.40 * s_emotion +
            0.20 * s_words +
            0.20 * s_prosody +
            0.20 * s_urgency
        )
        return round(min(max(stress, 0.0), 1.0), 3)

    def _compute_urgency(self, text_lower: str) -> float:
        total = sum(
            weight for kw, weight in URGENCY_KEYWORDS.items()
            if kw in text_lower
        )
        return round(min(total, 1.0), 3)

    def _compute_trust(self, primary_emotion: Emotion, stress: float) -> float:
        emotion_trust_delta = {
            Emotion.ANGRY:      -0.40,
            Emotion.FRUSTRATED: -0.30,
            Emotion.SAD:        -0.20,
            Emotion.FEARFUL:    -0.25,
            Emotion.SURPRISED:   0.00,
            Emotion.NEUTRAL:     0.00,
            Emotion.HAPPY:      +0.30,
        }
        trust = 0.70 + emotion_trust_delta[primary_emotion] * (1.0 - stress)
        return round(min(max(trust, 0.0), 1.0), 3)

    def detect(self, text: str) -> BehaviorSignals:
        """
        Main entry point. Given a customer utterance, returns BehaviorSignals.
        Recency Weighting applied: 70% current sentence, 30% previous history.
        """
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text)

        # Step 1: Score emotions
        raw_scores = self._score_emotion(text_lower, words)

        # Step 2: Softmax → probabilities
        probabilities = self._softmax(raw_scores)

        # Step 3: Primary emotion
        primary_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[primary_emotion]

        # Step 4: Prosody features
        prosody = self._extract_prosody(text, words)

        # Step 5: Derived scores
        stress    = self._compute_stress(primary_emotion, confidence, prosody, text_lower)
        urgency   = self._compute_urgency(text_lower)
        trust     = self._compute_trust(primary_emotion, stress)
        engagement = round(min(0.4 + prosody.punctuation_density * 0.8 + confidence * 0.3, 1.0), 3)
        patience  = "low" if stress > 0.70 else ("medium" if stress > 0.40 else "high")

        # Step 6: Session Recency Weighting (70% current sentence, 30% history)
        current_sentence_sentiment = self._calculate_sentence_sentiment(primary_emotion, confidence, stress)
        if self.session_turns == 0:
            conv_score = current_sentence_sentiment
            trend = "Stable ➔"
        else:
            conv_score = (self.recency_weight * current_sentence_sentiment) + (self.history_weight * self.conversation_score)
            diff = conv_score - self.conversation_score
            if diff > 0.08:
                trend = "Improving ↗"
            elif diff < -0.08:
                trend = "Worsening ↘"
            else:
                trend = "Stable ➔"

        self.previous_conversation_score = self.conversation_score
        self.conversation_score = round(conv_score, 3)
        self.session_turns += 1
        self.conversation_trend = trend

        if conv_score > 0.30:
            conv_label = "Positive"
        elif conv_score < -0.30:
            conv_label = "Critical / Tense"
        elif conv_score < -0.05:
            conv_label = "Slightly Negative"
        else:
            conv_label = "Neutral"

        return BehaviorSignals(
            emotion=primary_emotion,
            emotion_confidence=round(confidence, 3),
            emotion_color=EMOTION_COLORS[primary_emotion],
            emotion_emoji=EMOTION_EMOJI[primary_emotion],
            stress=stress,
            engagement=engagement,
            trust=trust,
            urgency=urgency,
            conversation_score=self.conversation_score,
            conversation_trend=self.conversation_trend,
            conversation_label=conv_label,
            patience=patience,
            prosody=prosody,
            text=text,
        )


# ─── Quick Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = EmotionEngine()
    test_cases = [
        "I've called three times already and nobody has fixed my issue! This is ridiculous!",
        "Thank you so much! Your service has been absolutely wonderful today.",
        "I'm really disappointed with this. It keeps failing every time I try.",
        "This is an emergency! My account has been hacked right now, please help!",
        "I just wanted to check the status of my order.",
    ]

    print("\n" + "=" * 70)
    print("  AdaptiveCX — Emotion Engine Test")
    print("=" * 70)

    for text in test_cases:
        result = engine.detect(text)
        print(f"\nText   : {text[:60]}...")
        print(f"Emotion: {result.emotion_emoji} {result.emotion.value.upper()} (confidence={result.emotion_confidence:.2f})")
        print(f"Stress : {'█' * int(result.stress * 20)}{'░' * (20 - int(result.stress * 20))} {result.stress:.2f}")
        print(f"Trust  : {result.trust:.2f} | Urgency: {result.urgency:.2f} | Patience: {result.patience}")
