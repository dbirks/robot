"""AttentionManager decision cascade (ADR 0003).

  1. explicit wake keyword                  -> engage
  2. active lease, same participant         -> engage (renew)
  3. deterministic spatial/visual evidence  -> respond or ignore
  4. ambiguous ambient                      -> ignore (fail-closed);
     optionally 'ask_if_addressed' in the marginal band

Rule of the house: heuristics may only RAISE the bar (e.g. TV duty cycle),
never early-return "respond". The old fail-open filter logged 274 responds
and 0 ignores because second-person phrasing short-circuited every ignore
branch; TV dialogue IS saturated with "can you...". Ignored ambient
utterances never reach the LLM at all, so they can never trigger tools.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .. import journal as J
from .lease import AttentionLease
from .participant import UNKNOWN, Observation, ResolvedParticipant
from .profiles import AttentionProfile


class Action(Enum):
    ENGAGE = "engage"  # grant/renew lease, open realtime stream
    IGNORE = "ignore"  # utterance disappears from context
    ASK_IF_ADDRESSED = "ask_if_addressed"


@dataclass
class Decision:
    action: Action
    confidence: float
    evidence: dict = field(default_factory=dict)
    renewed: bool = False
    acquired: bool = False


class AttentionManager:
    def __init__(
        self,
        profile: AttentionProfile,
        lease: AttentionLease,
        journal,
        clock=None,
        rng=random.Random(),
    ) -> None:
        self.profile = profile
        self.lease = lease
        self.journal = journal
        self._rng = rng
        # Rolling 60s speech-duty-cycle window (seconds of speech per second
        # of wall time). A TV talks continuously for minutes; people don't.
        self._speech = deque()
        self._window = 60.0
        self.clock = clock

    # ---- duty-cycle tracker ----

    def record_speech(self, seconds: float) -> None:
        import time

        now = (self.clock or time.monotonic)()
        self._speech.append((now, seconds))
        while self._speech and now - self._speech[0][0] > self._window:
            self._speech.popleft()

    @property
    def duty_cycle(self) -> float:
        import time

        if not self._speech:
            return 0.0
        now = (self.clock or time.monotonic)()
        while self._speech and now - self._speech[0][0] > self._window:
            self._speech.popleft()
        return min(1.0, sum(d for _, d in self._speech) / self._window)

    def _raise(self) -> float:
        return self.profile.duty_raise if self.duty_cycle > self.profile.duty_cycle_limit else 1.0

    # ---- decisions ----

    def on_keyword(self, participant_id: str | None, keyword: str) -> Decision:
        holder = participant_id or f"doa-{self._rng.randrange(1_000_000)}"
        was_holder = self.lease.holder
        self.lease.acquire(holder, evidence=f"kws:{keyword}")
        d = Decision(Action.ENGAGE, confidence=1.0, evidence={"keyword": keyword}, acquired=True)
        if was_holder and was_holder != holder:
            d.evidence["transferred_from"] = was_holder
        self._log(d, "keyword")
        return d

    def decide(self, resolved: ResolvedParticipant, obs: Observation) -> Decision:
        self.lease.expire_if_due()
        raise_mult = self._raise()

        # 2. active lease, same (or unidentifiable-but-compatible) participant
        if self.lease.active():
            same = (
                resolved.id == UNKNOWN
                or resolved.id == self.lease.holder
                or (obs.doa_deg is not None and self.lease.seconds_since_interaction < self.profile.recency_window_s)
            )
            if same:
                self.lease.renew(resolved.id if resolved.id != UNKNOWN else None, evidence="continued-speech")
                d = Decision(
                    Action.ENGAGE,
                    confidence=self.lease.confidence(),
                    evidence={"lease": True, **resolved.evidence},
                    renewed=True,
                )
                self._log(d, "lease")
                return d
            # someone else talking while we're engaged with a participant:
            # that's a SIDE conversation - ignore it, wake phrase still
            # reacquires via on_keyword
            d = Decision(Action.IGNORE, confidence=0.0, evidence={"side_conversation": resolved.id})
            self._log(d, "lease-held-by-other")
            return d

        # 3/4. no lease: ambient. Evidence score against the directedness bar.
        recency = self.lease.seconds_since_interaction
        score = resolved.confidence
        if recency < self.profile.recency_window_s:
            score += self.profile.w_recency * (1.0 - recency / self.profile.recency_window_s)
        bar = self.profile.ambient_directedness * raise_mult

        if score >= bar:
            holder = resolved.id if resolved.id != UNKNOWN else "unknown"
            self.lease.acquire(holder, evidence=f"ambient-score:{score:.2f}")
            d = Decision(
                Action.ENGAGE,
                confidence=min(1.0, score),
                evidence={"score": round(score, 3), "bar": round(bar, 3), **resolved.evidence},
                acquired=True,
            )
        elif score >= bar - self.profile.ask_band and self.profile.ask_if_addressed:
            d = Decision(
                Action.ASK_IF_ADDRESSED, confidence=score, evidence={"score": round(score, 3), "bar": round(bar, 3)}
            )
        else:
            d = Decision(
                Action.IGNORE,
                confidence=score,
                evidence={"score": round(score, 3), "bar": round(bar, 3), "duty_cycle": round(self.duty_cycle, 2)},
            )
        self._log(d, "ambient")
        return d

    # ---- tool gate ----

    def tool_confidence(self, _name: str) -> float:
        return self.lease.confidence()

    def _log(self, d: Decision, stage: str) -> None:
        self.journal.write(
            J.ATTENTION_DECISION,
            action=d.action.value,
            stage=stage,
            confidence=round(d.confidence, 3),
            evidence=d.evidence,
            duty_cycle=round(self.duty_cycle, 2),
            raise_applied=self._raise() > 1.0,
        )
