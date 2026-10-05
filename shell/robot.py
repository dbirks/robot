"""The shell's one connection to the Reachy Mini daemon.

Duck-type compatible with app.robot_state.RobotConnection (`.mini`,
`.connected`, `.config.reachy_host`) so the legacy tool handlers run
unchanged against it during the migration window.

Connects with media_backend="no_media": this process owns audio (ADR 0002)
and the SDK contract for no_media is "the daemon releases camera and mic for
direct access". Robot absent is a normal state, never an exception: tools
then answer {"ok": False, "error": "Robot not connected"} (CLAUDE.md).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from . import journal as J

log = logging.getLogger("shell.robot")


@dataclass
class _RobotConfig:
    reachy_host: str = field(default_factory=lambda: os.getenv("REACHY_HOST", "localhost"))


class RobotLink:
    def __init__(self, journal=None, *, timeout_s: float = 5.0) -> None:
        self.config = _RobotConfig()
        self.journal = journal
        self.timeout_s = timeout_s
        self.mini = None

    @property
    def connected(self) -> bool:
        return self.mini is not None

    @property
    def daemon_url(self) -> str:
        return f"http://{self.config.reachy_host}:8000"

    def connect(self) -> bool:
        """Best effort; returns whether a robot is attached. Never raises."""
        try:
            from reachy_mini import ReachyMini

            self.mini = ReachyMini(
                host=self.config.reachy_host,
                media_backend="no_media",
                timeout=self.timeout_s,
            )
            log.info("connected to Reachy Mini daemon at %s", self.config.reachy_host)
        except Exception as e:
            self.mini = None
            log.warning("Reachy Mini not connected (%r); robot tools will report it", e)
        if self.journal is not None:
            self.journal.write(J.ROBOT_CONNECTION, connected=self.connected, host=self.config.reachy_host)
        return self.connected

    def disconnect(self) -> None:
        mini, self.mini = self.mini, None
        if mini is None:
            return
        try:
            mini.__exit__(None, None, None)
        except Exception as e:
            log.warning("robot disconnect: %r", e)
