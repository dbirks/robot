"""Attention lease: 'Reachy believes this participant is interacting with him.'

Evidence-based renewal, never a single hard timer (ADR 0003). Expiry is
evaluated lazily via `active()` + `tick()` so it works in both async and
sync contexts without a background thread.
"""

from __future__ import annotations

import time

from .. import journal as J
from .profiles import AttentionProfile


class AttentionLease:
    def __init__(self, profile: AttentionProfile, journal, clock=time.monotonic) -> None:
        self.profile = profile
        self.journal = journal
        self.clock = clock
        self.holder: str | None = None
        self._expires = 0.0
        self._started = 0.0
        self.last_interaction = 0.0  # any accepted dialogue/turn, for recency

    @property
    def seconds_since_interaction(self) -> float:
        if not self.last_interaction:
            return float("inf")
        return self.clock() - self.last_interaction

    @property
    def remaining_s(self) -> float:
        return max(0.0, self._expires - self.clock())

    def active(self) -> bool:
        return self.holder is not None and self.clock() < self._expires

    def acquire(self, participant: str, evidence: str) -> None:
        now = self.clock()
        if self.holder is not None and self.holder != participant:
            self.journal.write(J.ATTENTION_LEASE_TRANSFERRED, to=participant, from_=self.holder, evidence=evidence)
        self.holder = participant
        self._started = now
        self._expires = now + self.profile.lease_base_s
        self.last_interaction = now
        self.journal.write(
            J.ATTENTION_LEASE_STARTED, holder=participant, evidence=evidence, base_s=self.profile.lease_base_s
        )

    def renew(self, participant: str | None, evidence: str) -> bool:
        if not self.active():
            return False
        if participant is not None and participant != self.holder:
            self.acquire(participant, evidence)
            return True
        now = self.clock()
        self._expires = min(
            self._started + self.profile.lease_max_s,
            now + self.profile.lease_renew_s,
        )
        self.last_interaction = now
        self.journal.write(
            J.ATTENTION_LEASE_RENEWED, holder=self.holder, evidence=evidence, remaining_s=round(self.remaining_s, 1)
        )
        return True

    def exchange(self, evidence: str = "reply-done") -> bool:
        """A real dialogue exchange happened (Reachy finished replying).

        Slides the max-duration window so a live conversation never hits the
        lease_max_s cap mid-flow - that cap exists to stop ambient speech
        (TV) from holding attention via speech-start renewals, not to cut off
        someone Reachy is actually talking with. Gives the user a full
        renewal window to answer from the END of Reachy's reply. Re-arms a
        lease that lapsed while he was talking (holder kept until expiry)."""
        if self.holder is None:
            return False
        now = self.clock()
        self._started = now
        self._expires = now + self.profile.lease_renew_s
        self.last_interaction = now
        self.journal.write(
            J.ATTENTION_LEASE_RENEWED, holder=self.holder, evidence=evidence, remaining_s=round(self.remaining_s, 1)
        )
        return True

    def note_interaction(self) -> None:
        self.last_interaction = self.clock()

    def expire_if_due(self) -> bool:
        """Returns True if this call just expired the lease."""
        if self.holder is None or self.clock() < self._expires:
            return False
        holder, self.holder = self.holder, None
        self.journal.write(J.ATTENTION_LEASE_EXPIRED, holder=holder, held_s=round(self.clock() - self._started, 1))
        return True

    def confidence(self) -> float:
        """Current attention confidence for tool gating: 1.0 just after an
        interaction, decaying linearly across the lease remainder, 0 once
        expired."""
        if not self.active():
            return 0.0
        base = self.profile.lease_renew_s if self.profile.lease_renew_s > 0 else 1.0
        return max(0.0, min(1.0, self.remaining_s / base))
