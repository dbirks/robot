"""Contextual wake detection using DOA gating + semantic similarity.

Determines if an utterance is directed at the robot without requiring
a fixed wake word. Uses sentence embeddings to compare against a bank
of "talking to the robot" reference phrases.
"""

import logging
import math
import os

import numpy as np
import requests

log = logging.getLogger(__name__)

DOA_URL = "http://localhost:8000/api/state/doa"
DOA_FRONT_MIN = math.pi / 4
DOA_FRONT_MAX = 3 * math.pi / 4
WAKE_THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.45"))

WAKE_PHRASES = [
    "wake up",
    "hey robot",
    "hello robot",
    "good morning",
    "rise and shine",
    "can you help me",
    "I have a question",
    "excuse me",
    "are you there",
    "are you awake",
    "hey can you hear me",
    "I need your help",
    "what do you think",
    "tell me something",
    "what time is it",
    "look at me",
    "hey",
    "hello",
    "hi there",
    "hey buddy",
    "time to get up",
    "I want to talk to you",
    "can I ask you something",
    "do you know",
    "what do you see",
    "who is there",
    "who do you see",
    "play peekaboo",
    "can you see me",
]


class WakeDetector:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        log.info("Loading wake detector (all-MiniLM-L6-v2)...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self._wake_embeddings = self._model.encode(WAKE_PHRASES, normalize_embeddings=True)
        log.info("Wake detector ready (%d reference phrases, threshold=%.2f)", len(WAKE_PHRASES), WAKE_THRESHOLD)

    def is_from_front(self) -> bool | None:
        """Check if speech is coming from the front hemisphere via DOA."""
        try:
            resp = requests.get(DOA_URL, timeout=0.2)
            if resp.status_code == 200:
                data = resp.json()
                if data and data.get("speech_detected"):
                    angle = data["angle"]
                    return DOA_FRONT_MIN <= angle <= DOA_FRONT_MAX
        except requests.RequestException:
            pass
        return None

    def is_directed_at_robot(self, text: str) -> tuple[bool, float]:
        """Check if the utterance is semantically directed at the robot.

        Returns (is_wake, max_similarity_score).
        """
        text_emb = self._model.encode([text], normalize_embeddings=True)
        similarities = text_emb @ self._wake_embeddings.T
        max_sim = float(similarities.max())
        return max_sim >= WAKE_THRESHOLD, max_sim

    def should_wake(self, text: str, use_doa: bool = True) -> bool:
        """Full wake detection: DOA gate + semantic classification."""
        if use_doa:
            from_front = self.is_from_front()
            if from_front is False:
                log.debug("DOA: speech from side/behind, ignoring: %s", text)
                return False

        is_wake, score = self.is_directed_at_robot(text)
        if is_wake:
            log.info("Wake detected (score=%.3f): %s", score, text)
        else:
            log.debug("Not directed at robot (score=%.3f): %s", score, text)
        return is_wake

    def classify_utterance(self, text: str) -> str:
        """Classify whether active-mode speech is directed at the robot.

        Uses fast heuristics first, falls back to semantic similarity.
        Total time: <25ms.

        Returns:
            'respond' — high confidence, respond normally
            'ignore'  — background speech, ignore silently
        """
        lower = text.lower().strip()
        words = lower.split()
        word_count = len(words)

        # Instant signals — definitely directed at robot
        robot_names = {"robot", "reachy", "reachi", "richy", "richi"}
        if robot_names & set(words):
            log.info("Robot name detected: %s", text)
            return "respond"

        # Second-person address — strong signal
        you_patterns = {"you", "your", "you're", "can you", "do you", "are you",
                        "would you", "could you", "will you", "tell me", "show me",
                        "help me", "let me"}
        if any(p in lower for p in you_patterns):
            log.info("Second-person address: %s", text)
            return "respond"

        # Questions directed outward (likely to robot if short)
        if lower.rstrip().endswith("?") and word_count < 15:
            log.info("Short question: %s", text)
            return "respond"

        # Very short utterances (1-4 words) — likely directed at robot
        if word_count <= 4:
            log.info("Short utterance: %s", text)
            return "respond"

        # DOA check — if not from front, likely background
        from_front = self.is_from_front()
        if from_front is False and word_count > 8:
            log.debug("Long speech from side/behind, ignoring: %s", text)
            return "ignore"

        # Long monologue (>25 words) with no robot cues — likely TV/podcast
        if word_count > 25:
            log.debug("Long monologue, ignoring: %s", text)
            return "ignore"

        # Medium length (5-25 words), no clear cues — use semantic classifier
        _, score = self.is_directed_at_robot(text)
        if from_front is False:
            score *= 0.5

        if score >= 0.40:
            log.info("Semantic match (score=%.3f): %s", score, text)
            return "respond"

        # Default: if it's moderate length and we're not sure, respond anyway
        # Better to respond to background speech occasionally than miss real requests
        if word_count <= 12:
            log.info("Short-ish, responding by default: %s", text)
            return "respond"

        log.debug("No match, ignoring (score=%.3f, words=%d): %s", score, word_count, text)
        return "ignore"
