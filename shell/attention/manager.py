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


def _monotonic() -> float:
    import time

    return time.monotonic()


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
        # Attention, measured when the guest speaks rather than when the model gets around to
        # acting on it. See tool_confidence().
        self._turn_conf = 0.0
        self._turn_until = 0.0

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
        if participant_id is None and self.lease.active():
            # "Reachy, ..." mid-conversation: same conversation, not a new
            # participant. Renew; no transfer, no fresh wake acknowledgement.
            self.lease.renew(None, evidence=f"kws:{keyword}")
            d = Decision(Action.ENGAGE, confidence=1.0, evidence={"keyword": keyword}, acquired=False)
            self._log(d, "keyword-renew")
            return d
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

    def _arm(self, confidence: float) -> None:
        """Record the attention the guest just demonstrated, for the length of one turn.

        Staleness is the whole point of the window: an arm from a previous turn must not be
        revived for whoever is standing there now, so an expired arm is dropped before the new
        value is maxed against it.
        """
        now = (self.clock or _monotonic)()
        if now > self._turn_until:
            self._turn_conf = 0.0
        self._turn_conf = max(self._turn_conf, min(1.0, confidence))
        self._turn_until = now + self.profile.lease_renew_s

    def freeze_for_turn(self) -> None:
        """Called where an accepted user turn is recorded (`lease.note_interaction()`).

        Measured live 2026-10-13 at 13:06 in the hackday config: the wake word hit, the lease was
        live, and `set_volume` came back `attention.tool_denied` at confidence 0.7527 against the
        event profile's 0.8 - with `play_emotion` denied twice in the same second. The guest had
        spoken 3.0 s earlier. The gate read `lease.confidence()`, which is
        ``remaining_s / lease_renew_s``: with renew 15 s that needs 12 s of lease left to clear
        0.8, so a tool call had to land within ~3 s of the person finishing speech. This box
        cannot do a cold camera grab (2.7-3.0 s) inside that, let alone a whole 4B reply - so
        every state-changing tool was unreachable by voice at the event profile, and "go to bed"
        only ever worked because the deterministic intent bypasses this gate.

        Re-arms with the lease's own reading, and cannot lower an arm already held by stronger
        evidence from the same turn (the wake word's 1.0).
        """
        self._arm(self.lease.confidence())

    def tool_confidence(self, _name: str) -> float:
        live = self.lease.confidence()
        if not self.lease.active():
            # An expired lease ends the turn's claim too: decay is the mechanism that stops an
            # old participant's session licensing whoever is standing there now (ADR 0003).
            self._turn_conf = 0.0
            return 0.0
        now = (self.clock or _monotonic)()
        if self._turn_conf and now <= self._turn_until:
            return max(live, self._turn_conf)
        return live

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

        if d.action == Action.ENGAGE:
            # The evidence-derived number is what should decide an evidence gate. `on_keyword`
            # returns confidence 1.0 and an ambient engage returns its own score, while
            # `lease.confidence()` at the event profile starts at lease_base_s / lease_renew_s =
            # 10/15 = 0.667 - BELOW the 0.8 bar it is compared to. Arming here is what makes the
            # wake word mean what the journal already said it meant.
            self._arm(d.confidence)
