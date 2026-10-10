"""XVF3800 runtime tuning: AGC ceiling and wake-direction beam focus.

Two of the "unexploited controls" from ADR 0005:

- AGC: the firmware boots with PP_AGCMAXGAIN=64, so in a quiet room it pumps
  the TV / next room up to speech level and the VAD hears a conversation.
  We cap it (Pollen's conversation app uses 10) or turn it off (0).
- Beam focus: on a wake word both fixed beams are steered at the speaker's
  azimuth (AEC_FIXEDBEAMSONOFF), so off-axis sources are attenuated before
  they reach the pipeline. Released when the attention lease ends, so the
  wake-word spotter always listens with the normal auto-select beam. The
  array is LINEAR: a TV straight in front/behind on the same axis as the user
  is not helped (front/back ambiguity, ADR 0005).

Every setting is lost on an XMOS REBOOT (the watchdog's recovery), so
`ensure()` re-checks and re-applies; it is cheap (one ~1 ms control read).
Everything here runs off the audio threads and never raises.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger("shell.xmos")


def _int(r) -> int | None:
    # int32/uint8 reads come back as [status, value, ...]; floats as a tuple.
    return None if r is None else int(r[1])


class XmosTuner:
    def __init__(self, journal, *, agc_max_gain: float = 10.0, beam_focus: bool = True, open_device=None) -> None:
        self.journal = journal
        self.agc_max_gain = agc_max_gain
        self.beam_focus = beam_focus
        self._open = open_device or _open_respeaker
        self._dev = None
        self._lock = threading.Lock()
        self.focused: float | None = None  # azimuth (rad) the beams point at

    def _device(self):
        if self._dev is None:
            self._dev = self._open()
        return self._dev

    def _call(self, fn):
        with self._lock:
            try:
                dev = self._device()
                return None if dev is None else fn(dev)
            except Exception as e:  # stale handle after REBOOT / replug
                log.debug("xmos control failed: %r", e)
                self._drop()
                return None

    def _drop(self) -> None:
        dev, self._dev = self._dev, None
        if dev is not None:
            try:
                dev.close()
            except Exception:
                pass

    def ensure(self) -> None:
        """Apply the AGC setting if the chip lost it (boot, REBOOT)."""

        def go(dev):
            on = _int(dev.read("PP_AGCONOFF"))
            if self.agc_max_gain <= 0:
                if on != 0:
                    dev.write("PP_AGCONOFF", [0])
                    return {"agc": "off"}
                return None
            changed = {}
            if on != 1:
                dev.write("PP_AGCONOFF", [1])
                changed["agc"] = "on"
            cur = dev.read("PP_AGCMAXGAIN")
            if cur is None or abs(float(cur[0]) - self.agc_max_gain) > 0.01:
                dev.write("PP_AGCMAXGAIN", [self.agc_max_gain])
                changed["agc_max_gain"] = self.agc_max_gain
                changed["was"] = None if cur is None else round(float(cur[0]), 2)
            return changed or None

        changed = self._call(go)
        if changed:
            log.info("xmos tuned: %s", changed)
            self.journal.write("xmos.tuned", **changed)

    def focus(self, theta: float | None) -> None:
        """Point both fixed beams at `theta` (DOA radians, 0..pi)."""
        if not self.beam_focus or theta is None:
            return

        def go(dev):
            dev.write("AEC_FIXEDBEAMSAZIMUTH_VALUES", [theta, theta])
            dev.write("AEC_FIXEDBEAMSELEVATION_VALUES", [0.0, 0.0])
            dev.write("AEC_FIXEDBEAMSONOFF", [1])
            return True

        if self._call(go):
            self.focused = theta
            self.journal.write("xmos.beam_focus", azimuth_deg=round(theta * 57.29578, 1))

    def release(self) -> None:
        if self.focused is None:
            return
        if self._call(lambda dev: dev.write("AEC_FIXEDBEAMSONOFF", [0]) or True):
            self.journal.write("xmos.beam_release")
        self.focused = None  # after a failure the REBOOT path resets it anyway

    def close(self) -> None:
        self.release()
        with self._lock:
            self._drop()


def _open_respeaker():
    from reachy_mini.media.audio_control_utils import init_respeaker_usb

    return init_respeaker_usb()
