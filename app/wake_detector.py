"""Contextual wake detection using DOA gating + semantic similarity.

Determines if an utterance is directed at the robot without requiring
a fixed wake word. Uses sentence embeddings to compare against a bank
of "talking to the robot" reference phrases.
"""

import logging
import math
import os
import re

import numpy as np
import requests

log = logging.getLogger(__name__)

DOA_URL = "http://localhost:8000/api/state/doa"
DOA_FRONT_MIN = math.pi / 4
DOA_FRONT_MAX = 3 * math.pi / 4

# Threshold for waking from sleep. 0.45 was far too low for all-MiniLM-L6-v2
# cosine similarity — generic short English fragments and Whisper hallucinations
# cleared it (observed false wakes at 0.46-0.48 on TV dialogue). Genuine wakes
# score much higher ("good morning" ~0.94), so 0.6 rejects the noise while
# keeping the real wakes. Explicit phrases (WAKE_KEYPHRASES) catch the common
# wakes regardless of score, so this only gates paraphrased/semantic wakes and
# can sit higher (0.65) to reject borderline sleep-themed TV dialogue.
WAKE_THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.65"))
# Threshold for the awake-mode "is this directed at me?" semantic check.
DIRECTED_THRESHOLD = float(os.getenv("DIRECTED_THRESHOLD", "0.45"))
# A semantic wake needs at least this many words. Single/short fragments
# ("Who", "No, I'm") are almost always Whisper hallucinations on noise.
# Genuine short wakes ("wake up", "good morning") are caught by WAKE_KEYPHRASES
# below regardless of length, so they bypass this guard.
WAKE_MIN_WORDS = int(os.getenv("WAKE_MIN_WORDS", "3"))

# Tight bank of true wake intents — used ONLY while asleep. Deliberately
# excludes generic conversational filler (hey/hello/who is there/what do you
# think) that matched arbitrary ambient chatter at the old 0.45 threshold.
WAKE_PHRASES = [
    "wake up",
    "wake up robot",
    "robot wake up",
    "hey robot wake up",
    "time to wake up",
    "you can wake up now",
    "are you awake",
    "are you sleeping",
    "good morning",
    "good morning robot",
    "rise and shine",
    "time to get up",
    "hey robot",
    "wake up reachy",
]

# Explicit phrases that wake the robot if they appear ANYWHERE in the
# transcript — even buried in surrounding TV dialogue. This is what lets a
# real "wake up robot" said over the top of a movie still trigger, while the
# rest of the dialogue is ignored. Kept exact/substring (not semantic) so it
# can never be diluted by surrounding words.
WAKE_KEYPHRASES = [
    "wake up",
    "good morning",
    "rise and shine",
    "hey robot",
    "time to get up",
    "are you awake",
]

# Broad bank for the AWAKE-mode ambient filter (classify_utterance). This
# preserves the previous behavior of is_directed_at_robot — it answers "is
# this speech aimed at me?" rather than "is this a wake command?", so it keeps
# the wider conversational phrases that the tight wake bank drops.
DIRECTED_PHRASES = [
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

# Cap on the number of candidate segments we embed per transcript, to bound
# CPU cost on pathologically long ambient transcripts.
_MAX_WAKE_CANDIDATES = 64


class WakeDetector:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        log.info("Loading wake detector (all-MiniLM-L6-v2)...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self._wake_embeddings = self._model.encode(WAKE_PHRASES, normalize_embeddings=True)
        self._directed_embeddings = self._model.encode(DIRECTED_PHRASES, normalize_embeddings=True)
        log.info(
            "Wake detector ready (%d wake / %d directed phrases, wake_threshold=%.2f)",
            len(WAKE_PHRASES),
            len(DIRECTED_PHRASES),
            WAKE_THRESHOLD,
        )

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

        Used by the AWAKE-mode ambient filter (classify_utterance). Compares
        against the broad DIRECTED_PHRASES bank, not the tight wake bank.

        Returns (is_directed, max_similarity_score).
        """
        text_emb = self._model.encode([text], normalize_embeddings=True)
        similarities = text_emb @ self._directed_embeddings.T
        max_sim = float(similarities.max())
        return max_sim >= DIRECTED_THRESHOLD, max_sim

    def _wake_candidates(self, text: str) -> list[str]:
        """Break a transcript into segments to test against the wake bank.

        A genuine wake phrase spoken over TV dialogue is diluted if we embed
        the whole transcript at once. So we also test individual clauses and
        short sliding windows: the wake phrase scores high on its own segment
        while the surrounding dialogue does not.
        """
        candidates = {text.strip()}
        # Clauses split on sentence/clause punctuation.
        for clause in re.split(r"[.?!,;:]+", text):
            clause = clause.strip()
            if clause:
                candidates.add(clause)
        # Short sliding windows catch a wake phrase inside an unpunctuated run.
        words = text.split()
        for size in (3, 5):
            if len(words) > size:
                for i in range(len(words) - size + 1):
                    candidates.add(" ".join(words[i : i + size]))
        out = [c for c in candidates if c]
        return out[:_MAX_WAKE_CANDIDATES]

    def _wake_score(self, text: str) -> float:
        """Max similarity of any segment of the transcript to a wake phrase."""
        candidates = self._wake_candidates(text)
        embs = self._model.encode(candidates, normalize_embeddings=True)
        similarities = embs @ self._wake_embeddings.T
        return float(similarities.max())

    def should_wake(self, text: str, use_doa: bool = True) -> bool:
        """Decide whether asleep robot should wake from this transcript.

        DOA gate -> explicit wake-phrase substring -> length guard ->
        segment-wise semantic match. Tuned to ignore ambient TV dialogue and
        Whisper hallucinations while still catching a real wake command, even
        one buried in surrounding speech.
        """
        norm = " ".join(text.lower().split())
        if not norm:
            return False

        if use_doa:
            from_front = self.is_from_front()
            if from_front is False:
                log.debug("DOA: speech from side/behind, ignoring: %s", text)
                return False

        # Explicit wake phrase anywhere in the transcript — wakes even if
        # surrounded by unrelated TV dialogue.
        for phrase in WAKE_KEYPHRASES:
            if phrase in norm:
                log.info("Wake phrase '%s' detected in: %s", phrase, text)
                return True

        # No explicit phrase: reject short fragments (Whisper hallucinations
        # on noise/near-silence like "Who", "No, I'm").
        if len(norm.split()) < WAKE_MIN_WORDS:
            log.debug("Too short, no wake phrase, ignoring: %s", text)
            return False

        # Segment-wise semantic match so a paraphrased wake intent embedded in
        # surrounding speech still scores on its own segment.
        score = self._wake_score(text)
        if score >= WAKE_THRESHOLD:
            log.info("Wake detected (score=%.3f): %s", score, text)
            return True
        log.debug("Not a wake intent (score=%.3f): %s", score, text)
        return False

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
        you_patterns = {
            "you",
            "your",
            "you're",
            "can you",
            "do you",
            "are you",
            "would you",
            "could you",
            "will you",
            "tell me",
            "show me",
            "help me",
            "let me",
        }
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
