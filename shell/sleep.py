"""Sleep as one state, not one pose (issue #36).

Until now "asleep" was a boolean on `MotionOwner`: the head lay down and the
microphone kept feeding the realtime service, so he answered the room while
lying there. This module owns the flag instead. `MotionOwner` keeps the
physical side — it is the only caller of `goto_target` (ADR 0002) and it reads
this same flag to pause its loop — and the shell assembly gates the mic, the
attention lease and the tool gate on it too, so head and microphone can never
be told different things.

The routing rules are pure functions of that one flag plus what the audio
owners are doing, which is what makes them testable without hardware.
"""

from __future__ import annotations

import time
from typing import Callable

from . import journal as J


class SleepState:
    """The single asleep/awake truth, journaled with the reason it moved."""

    def __init__(self, journal=None, clock: Callable[[], float] = time.monotonic) -> None:
        self.journal = journal
        self.clock = clock
        self._sleeping = False
        self.reason = ""  # why he is asleep, or what just woke him ("tool", "auto_sleep", "kws:Reachy")

    @property
    def sleeping(self) -> bool:
        return self._sleeping

    def enter(self, reason: str) -> bool:
        """False when he was already asleep: no duplicate move, no duplicate event."""
        if self._sleeping:
            return False
        self._sleeping = True
        self.reason = reason
        if self.journal is not None:
            self.journal.write(J.SLEEP_ENTER, reason=reason)
        return True

    def exit(self, reason: str) -> bool:
        if not self._sleeping:
            return False
        self._sleeping = False
        self.reason = reason
        if self.journal is not None:
            self.journal.write(J.SLEEP_EXIT, reason=reason)
        return True


def feeds_service(*, sleeping: bool, engaged: bool, muted: bool = False) -> bool:
    """True = hand the realtime service this frame; False = hand it silence.

    Asleep outranks engagement: a lease that had not yet timed out when he lay
    down must not keep the mic open. `muted` is the forced-endpoint silence of
    an over-long turn (unchanged from before issue #36).
    """
    return engaged and not muted and not sleeping


def feeds_keyword_spotter(*, sleeping: bool, engaged: bool) -> bool:
    """KWS runs whenever the service does not own the mic - and ALWAYS while
    asleep, or the wake word could never end the nap."""
    return sleeping or not engaged


def should_auto_sleep(*, auto_sleep_s: float, idle_s: float, engaged: bool, sleeping: bool) -> bool:
    """Lie down after `auto_sleep_s` of silence with nobody engaged.

    0 means never, and that is the answer in every shipped environment today
    (code default, `envs/home.yaml`, `envs/hackday.yaml`): he stays awake for an
    empty room and only sleeps when someone asks him to (issue #36).
    """
    return auto_sleep_s > 0 and not sleeping and not engaged and idle_s >= auto_sleep_s
