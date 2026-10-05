"""Direction of arrival: one reader, a short ring buffer, gated aiming.

The XVF3800 reports (azimuth, speech) where azimuth is 0 = left, pi/2 =
front/back, pi = right. The array is LINEAR (ADR 0005): front and back are
indistinguishable, so we assume the speaker is in front and map straight to
head yaw (positive = left).

Source: by default the shell reads DOA_VALUE_RADIANS from the XMOS over USB
itself (~1 ms vendor control transfer, no audio interface claimed). The
daemon's GET /api/state/doa holds a libusb handle opened at daemon start that
goes stale on every XMOS REBOOT (which the shell's own watchdog sends): on
2026-10-05 it returned 500 "No such device" until the daemon restarted.
Our handle is reopened with backoff after any error instead.
REACHY_DOA_SOURCE=daemon|usb|off selects; daemon remains as a fallback.

Policy (EPIC "Reaction"; brief for this change):
- samples taken while Reachy is speaking (and a short tail after) are
  dropped: that is his own voice
- on a wake word: median of the speech-flagged samples of the last ~1 s
- while the lease is active and the user is talking: re-aim only when the
  last 0.6 s of speech agrees (spread <= 20 deg), the target moved > 12 deg
  and >= 0.8 s passed since the last aim
"""

from __future__ import annotations

import logging
import math
import os
import statistics
import threading
import time
from collections import deque
from typing import Callable

from .. import journal as J

log = logging.getLogger("shell.doa")

DOA_HZ = 10.0
WINDOW_S = 2.0
WAKE_SPAN_S = 1.0
REAIM_SPAN_S = 0.6
REAIM_MIN_SAMPLES = 3
REAIM_MIN_CHANGE_DEG = 12.0
REAIM_MIN_INTERVAL_S = 0.8
REAIM_MAX_SPREAD_DEG = 20.0  # window must agree: never aim at a mix of two talkers
SPEAKER_TAIL_S = 0.3  # room echo of his own voice after playback stops
MAX_YAW_DEG = 60.0
WAKE_HOLD_S = 10.0
REAIM_HOLD_S = 8.0


def doa_to_yaw_deg(theta_rad: float, max_deg: float = MAX_YAW_DEG) -> float:
    """0 = left, pi/2 = front, pi = right  ->  head yaw (positive = left)."""
    return max(-max_deg, min(max_deg, math.degrees(math.pi / 2 - theta_rad)))


class DoaBuffer:
    """Ring buffer of (t, theta, speech); 2 s at 10 Hz is 20 samples."""

    def __init__(self, window_s: float = WINDOW_S) -> None:
        self.window_s = window_s
        self._q: deque[tuple[float, float, bool]] = deque()
        self._lock = threading.Lock()

    def add(self, t: float, theta: float, speech: bool) -> None:
        with self._lock:
            self._q.append((t, float(theta), bool(speech)))
            while self._q and self._q[0][0] < t - self.window_s:
                self._q.popleft()

    def speech_samples(self, now: float, span_s: float) -> list[float]:
        with self._lock:
            return [th for t, th, sp in self._q if sp and now - span_s <= t <= now]

    def median_speech(self, now: float, span_s: float, min_samples: int = 1) -> tuple[float | None, int]:
        s = self.speech_samples(now, span_s)
        if len(s) < min_samples:
            return None, len(s)
        return statistics.median(s), len(s)

    def latest(self) -> tuple[float, float, bool] | None:
        with self._lock:
            return self._q[-1] if self._q else None

    def clear(self) -> None:
        with self._lock:
            self._q.clear()


class AimGate:
    """Hysteresis for re-aiming: big enough change, not too often."""

    def __init__(self, min_change_deg: float = REAIM_MIN_CHANGE_DEG, min_interval_s: float = REAIM_MIN_INTERVAL_S):
        self.min_change_deg = min_change_deg
        self.min_interval_s = min_interval_s
        self.last_t: float | None = None
        self.last_yaw: float | None = None

    def allows(self, now: float, yaw_deg: float) -> bool:
        if self.last_t is None:
            return True
        return now - self.last_t >= self.min_interval_s and abs(yaw_deg - self.last_yaw) > self.min_change_deg

    def mark(self, now: float, yaw_deg: float) -> None:
        self.last_t, self.last_yaw = now, yaw_deg


# ---- sources: read() -> (theta_rad, speech) | None, never raises ----


class UsbDoaSource:
    def __init__(self, retry_s: float = 2.0, clock=time.monotonic) -> None:
        self.retry_s = retry_s
        self.clock = clock
        self._dev = None
        self._next_try = 0.0
        self.errors = 0

    def read(self):
        if self._dev is None:
            if self.clock() < self._next_try:
                return None
            try:
                from reachy_mini.media.audio_control_utils import init_respeaker_usb

                self._dev = init_respeaker_usb()
            except Exception as e:
                log.debug("respeaker init failed: %r", e)
                self._dev = None
            if self._dev is None:
                self._next_try = self.clock() + self.retry_s
                return None
            log.info("DOA: reading XMOS over USB")
        try:
            r = self._dev.read("DOA_VALUE_RADIANS")
            if r is None:
                return None
            return float(r[0]), bool(r[1])
        except Exception as e:  # stale handle after XMOS REBOOT / replug
            self.errors += 1
            if self.errors == 1 or self.errors % 100 == 0:
                log.warning("DOA read failed (%d); reopening: %r", self.errors, e)
            self.close()
            self._next_try = self.clock() + self.retry_s
            return None

    def close(self) -> None:
        dev, self._dev = self._dev, None
        if dev is not None:
            try:
                dev.close()
            except Exception:
                pass


class HttpDoaSource:
    def __init__(self, daemon_url: str = "http://localhost:8000", timeout_s: float = 0.2) -> None:
        self.url = daemon_url.rstrip("/") + "/api/state/doa"
        self.timeout_s = timeout_s
        self._session = None

    def read(self):
        try:
            import requests

            if self._session is None:
                self._session = requests.Session()
            r = self._session.get(self.url, timeout=self.timeout_s)
            if r.status_code != 200:
                return None
            d = r.json()
            if not d:
                return None
            return float(d["angle"]), bool(d["speech_detected"])
        except Exception:
            return None

    def close(self) -> None:
        pass


def make_doa_source(kind: str | None = None, daemon_url: str = "http://localhost:8000"):
    kind = (kind or os.getenv("REACHY_DOA_SOURCE", "usb")).lower()
    if kind == "off":
        return None
    if kind == "daemon":
        return HttpDoaSource(daemon_url)
    return UsbDoaSource()


class DoaTracker:
    """The one DOA reader thread (10 Hz) plus wake-turn / re-aim policy."""

    def __init__(
        self,
        source,
        motion,
        journal=None,
        *,
        is_speaking: Callable[[], bool] = lambda: False,
        lease_active: Callable[[], bool] = lambda: False,
        hz: float = DOA_HZ,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source
        self.motion = motion
        self.journal = journal
        self.is_speaking = is_speaking
        self.lease_active = lease_active
        self.hz = hz
        self.clock = clock
        self.buffer = DoaBuffer()
        self.gate = AimGate()
        self._last_playing = -1e9
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- sampling ----

    def sample(self, now: float | None = None) -> None:
        """One read + policy step (the thread calls this at `hz`)."""
        now = self.clock() if now is None else now
        if self.is_speaking():
            self._last_playing = now
            return  # his own voice: never aim at himself
        if now - self._last_playing < SPEAKER_TAIL_S:
            return
        r = self.source.read() if self.source is not None else None
        if r is None:
            return
        theta, speech = r
        self.buffer.add(now, theta, speech)
        if speech:
            self._maybe_reaim(now)

    def _maybe_reaim(self, now: float) -> None:
        if not self.lease_active() or self.motion is None or self.motion.sleeping or self.motion.busy:
            return
        samples = self.buffer.speech_samples(now, REAIM_SPAN_S)
        n = len(samples)
        if n < REAIM_MIN_SAMPLES or math.degrees(max(samples) - min(samples)) > REAIM_MAX_SPREAD_DEG:
            return
        yaw = doa_to_yaw_deg(statistics.median(samples))
        if not self.gate.allows(now, yaw):
            return
        self.gate.mark(now, yaw)
        self.motion.look_at(yaw, hold_s=REAIM_HOLD_S, source="doa")
        if self.journal is not None:
            self.journal.write(J.MOTION_DOA_AIM, yaw_deg=round(yaw, 1), samples=n)

    # ---- wake ----

    def wake_yaw(self, now: float | None = None) -> tuple[float | None, int]:
        """(yaw_deg, n_samples) from the speech of the last ~1 s, or (None, 0)."""
        now = self.clock() if now is None else now
        theta, n = self.buffer.median_speech(now, WAKE_SPAN_S, 1)
        return (None if theta is None else doa_to_yaw_deg(theta)), n

    def wake_turn(self, t_detect: float | None = None) -> float | None:
        """KWS fired: turn toward the speaker NOW (no LLM in the path).
        Called on the mic callback thread: cheap, never raises."""
        try:
            now = self.clock()
            yaw, n = self.wake_yaw(now)
            if self.motion is not None:
                self.motion.wake()  # asleep? get up first (queued exclusive)
                if yaw is not None:
                    self.motion.look_at(yaw, hold_s=WAKE_HOLD_S, source="wake")
                    self.gate.mark(now, yaw)
            if self.journal is not None:
                self.journal.write(
                    J.MOTION_WAKE_TURN,
                    yaw_deg=None if yaw is None else round(yaw, 1),
                    samples=n,
                    latency_ms=None if t_detect is None else round((self.clock() - t_detect) * 1000.0, 2),
                    sleeping=bool(getattr(self.motion, "sleeping", False)),
                )
            return yaw
        except Exception as e:
            log.warning("wake turn failed: %r", e)
            return None

    # ---- thread ----

    def start(self) -> bool:
        if self.source is None:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="doa", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.source is not None:
            self.source.close()

    def _run(self) -> None:
        interval = 1.0 / self.hz
        while not self._stop.is_set():
            t0 = self.clock()
            try:
                self.sample(t0)
            except Exception as e:  # keep the reader alive whatever happens
                log.warning("DOA sample failed: %r", e)
            self._stop.wait(max(0.0, interval - (self.clock() - t0)))
