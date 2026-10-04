"""XVF3800 capture-stall watchdog (ADR 0005).

The Reachy Mini mic array's firmware sometimes keeps the USB capture stream
running while delivering exact digital zeros. The XMOS REBOOT vendor command
recovers it in ~8 s without a replug. This decides WHEN to send it, from the
mic owner's health signal only - it never opens a stream (ADR 0002).

Rules (from the 2026-07-27 reboot-loop incident):
- never reboot when muted in-app or in PipeWire, or when mute state is unknown
- never reboot a stalled/missing stream: that is a different failure
- require sustained exact silence, and back off between reboots
"""

from __future__ import annotations

import logging
import time
from typing import Callable

log = logging.getLogger("shell.xmos")


class XmosWatchdog:
    def __init__(
        self,
        journal,
        reboot: Callable[[], None],
        *,
        silent_s: float = 30.0,
        backoff_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.journal = journal
        self.reboot = reboot
        self.silent_s = silent_s
        self.backoff_s = backoff_s
        self.clock = clock
        self._last_reboot: float | None = None
        self.reboots = 0

    def should_reboot(self, h: dict) -> bool:
        if h.get("muted") or h.get("pipewire_mute") is not False:
            return False
        if h.get("seconds_since_last_block", 0.0) > 2.0:
            return False  # stream itself is stalled: not the firmware zeros case
        if h.get("seconds_since_sound", 0.0) < self.silent_s:
            return False
        now = self.clock()
        return self._last_reboot is None or now - self._last_reboot >= self.backoff_s

    def check(self, h: dict) -> bool:
        """Reboot if warranted. Blocking (USB control transfer); call off-loop."""
        if not self.should_reboot(h):
            return False
        self._last_reboot = self.clock()
        self.reboots += 1
        silent = h.get("seconds_since_sound")
        log.warning("mic delivering exact zeros for %.0fs; sending XMOS REBOOT (#%d)", silent, self.reboots)
        try:
            self.reboot()
            self.journal.write("audio.xmos_reboot", silent_s=silent, n=self.reboots, ok=True)
        except Exception as e:
            log.error("XMOS REBOOT failed: %r", e)
            self.journal.write("audio.xmos_reboot", silent_s=silent, n=self.reboots, ok=False, error=repr(e))
        return True


def reboot_xmos() -> None:
    from reachy_mini.media.audio_control_utils import init_respeaker_usb

    dev = init_respeaker_usb()
    if dev is None:
        raise RuntimeError("Reachy Mini audio USB device not found")
    dev.write("REBOOT", [1])
