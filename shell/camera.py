"""Camera frames for the vision tools and the greeter, without fighting the daemon.

The daemon can own the camera (its GStreamer media server). Two processes on
one UVC device is the contention class ADR 0002 exists to prevent, so nothing
here opens the device while the daemon holds it: before opening it asks
GET /api/media/status and opens V4L2 directly only when the daemon reports
its media pipeline unavailable (no media server, or released via the
no_media connection in shell/robot.py - the SDK's documented "direct access"
contract). The one-shot `grab()` (vision tools) opens per call and releases
immediately. `CameraStream` - the greeter's poll - instead holds the handle
open (`open_stream`/`read_stream`/`close_stream` on the grabber): per sample it
pays exactly one `cap.read()`, the media-status check is cached behind a TTL,
and the /sys device scan happens on (re)open only, with a reopen backoff after
failures. If the cached status ever says the daemon has taken the device back,
the stream releases it. When the daemon wants the camera while we hold it, it
gets read failures; the backoff-and-reopen loop (status re-checked fresh on
every open) hands it back. This replaced open-per-grab at the 2026-10-12
review: warmup reads + a GET + a /sys scan at 2-3 Hz is not "one cheap sample",
and docs/pins.yaml `face_detection` says exactly which costs are measured and
which are UNMEASURED PENDING BENCH.

REACHY_CAMERA=off disables capture entirely.

`CameraStream` is the continuous owner the proactive greeter needs (B0): one
thread, one device, 2 FPS idle / 3 FPS while somebody is engaged, everybody else
reading `latest()`, and NO polling at all while he sleeps (the wake word owns the
way back up; the camera is not part of it). It is a wrapper over
`CameraGrabber`, not a second way to open the device, so the greeter and the
vision tools still share exactly one thing touching /dev/videoN (ADR 0002).
When the device is not there the stream disables itself and `latest()` stays
None — the shell must not notice.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from . import journal as J

log = logging.getLogger("shell.camera")

WARMUP_FRAMES = 5  # UVC auto-exposure needs a few frames before one is usable


class FrameGrabber(Protocol):
    """What `CameraStream` polls: something that returns a BGR frame or raises a
    speakable RuntimeError. `CameraGrabber` is the only implementation that touches
    hardware, which is the point of the protocol, not ceremony around it."""

    def grab(self) -> Any: ...


def find_reachy_camera(sysfs: Path = Path("/sys/class/video4linux")) -> int | None:
    """Lowest /dev/videoN whose V4L2 name is the Reachy Mini camera."""
    best = None
    try:
        for d in sysfs.iterdir():
            try:
                name = (d / "name").read_text().strip()
            except OSError:
                continue
            if "reachy" not in name.lower() or not d.name.startswith("video"):
                continue
            n = int(d.name[5:])
            best = n if best is None else min(best, n)
    except OSError:
        return None
    return best


def daemon_holds_camera(media_status: dict | None) -> bool:
    """Fail closed: unknown status counts as 'held'."""
    if not isinstance(media_status, dict):
        return True
    if media_status.get("released"):
        return False
    return bool(media_status.get("available", True))


class CameraGrabber:
    def __init__(self, daemon_url: str = "http://localhost:8000", *, mode: str | None = None) -> None:
        self.daemon_url = daemon_url.rstrip("/")
        self.mode = (mode or os.getenv("REACHY_CAMERA", "auto")).lower()
        self._lock = threading.Lock()  # one open/read/close at a time
        self._cap = None  # the held capture, owned by CameraStream via open_stream()
        self._status: tuple[float, dict | None] | None = None  # (monotonic, payload)
        self._reopen_after = 0.0
        self._backoff_s = self.REOPEN_BACKOFF_S

    def _fetch_media_status(self) -> dict | None:
        try:
            import requests

            r = requests.get(f"{self.daemon_url}/api/media/status", timeout=0.5)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def _media_status(self, *, ttl: float = 0.0) -> dict | None:
        """GET /api/media/status, optionally cached: the held-handle poll re-asks at
        most every MEDIA_STATUS_TTL_S (a None answer fails closed to 'held')."""
        if ttl and self._status is not None and time.monotonic() - self._status[0] < ttl:
            return self._status[1]
        payload = self._fetch_media_status()
        self._status = (time.monotonic(), payload)
        return payload

    def grab(self):
        """BGR frame (numpy) or raises RuntimeError with a speakable reason."""
        if self._cap is not None:
            return self.read_stream()  # the stream holds the handle: one device, one owner
        if self.mode == "off":
            raise RuntimeError("Camera disabled (REACHY_CAMERA=off)")
        if daemon_holds_camera(self._media_status()):
            raise RuntimeError("Camera is held by the robot daemon")
        index = find_reachy_camera()
        if index is None:
            raise RuntimeError("Reachy Mini camera not found")
        import cv2

        with self._lock:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open camera /dev/video{index}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                frame = None
                for _ in range(WARMUP_FRAMES):
                    ok, f = cap.read()
                    if ok:
                        frame = f
                if frame is None:
                    raise RuntimeError("No frame available from camera")
                return frame
            finally:
                cap.release()

    # ---- the held-handle path (CameraStream's poll) ----

    MEDIA_STATUS_TTL_S = 10.0  # re-ask the daemon about the device at most this often
    REOPEN_BACKOFF_S = 2.0  # after a failed open/read: wait, doubled, capped
    REOPEN_BACKOFF_MAX_S = 30.0

    def _check(self) -> None:
        if self.mode == "off":
            raise RuntimeError("Camera disabled (REACHY_CAMERA=off)")
        if time.monotonic() < self._reopen_after:
            raise RuntimeError("camera reopen backoff")
        if daemon_holds_camera(self._media_status()):
            raise RuntimeError("Camera is held by the robot daemon")

    def _fail(self) -> None:
        self._reopen_after = time.monotonic() + self._backoff_s
        self._backoff_s = min(self._backoff_s * 2.0, self.REOPEN_BACKOFF_MAX_S)

    def open_stream(self) -> None:
        """Open and warm the device once; `read_stream()` is then one `cap.read()`
        per sample. Raises RuntimeError with a speakable reason; on failure the
        next attempt waits out an increasing backoff."""
        if self._cap is not None:
            return
        self._check()
        index = find_reachy_camera()  # the /sys scan: per open, not per sample
        if index is None:
            self._fail()
            raise RuntimeError("Reachy Mini camera not found")
        import cv2

        with self._lock:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                self._fail()
                raise RuntimeError(f"Could not open camera /dev/video{index}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            frame = None
            for _ in range(WARMUP_FRAMES):  # auto-exposure: paid once per open, not per sample
                ok, f = cap.read()
                if ok:
                    frame = f
            if frame is None:
                cap.release()
                self._fail()
                raise RuntimeError("No frame available from camera")
            self._cap = cap
            self._backoff_s = self.REOPEN_BACKOFF_S

    def read_stream(self):
        """One held-handle frame: the whole per-sample device cost."""
        if self._cap is None:
            raise RuntimeError("camera stream is not open")
        with self._lock:
            if daemon_holds_camera(self._media_status(ttl=self.MEDIA_STATUS_TTL_S)):
                self.close_stream()  # the daemon took it back: let go, re-check on open
                raise RuntimeError("Camera reclaimed by the robot daemon")
            ok, frame = self._cap.read()
        if not ok:
            with self._lock:
                self.close_stream()
            self._fail()
            raise RuntimeError("No frame available from camera")
        return frame

    def close_stream(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
            self._cap = None


class CameraStream:
    """The one continuous frame source: `latest()` for readers, one poll thread.

    Drop-in for `CameraGrabber` where a caller wants the shared cache: `grab()`
    returns the cached frame while the stream is healthy and falls back to the
    grabber when it is not (a tool asking for a fresh frame mid-greeter is still
    one device owner, because both paths are inside this object).

    Against a grabber that supports the held-handle path (`open_stream` /
    `read_stream` / `close_stream`, i.e. `CameraGrabber`) one sample is one
    `cap.read()` — the device stays open across samples and is released on stop
    or on failure, reopened behind a backoff. Against a plain `grab()`-only
    grabber (tests, anything else) it falls back to open-per-sample. No polling
    at all while `asleep()` is true; a wake word does not need frames.
    """

    def __init__(
        self,
        grabber: FrameGrabber,
        *,
        idle_fps: float = 2.0,
        engaged_fps: float = 3.0,
        journal=None,
        clock=time.monotonic,
        asleep=None,
    ) -> None:
        self.grabber = grabber
        self.idle_fps = idle_fps
        self.engaged_fps = engaged_fps
        self.journal = journal
        self.clock = clock
        self._asleep = asleep if asleep is not None else (lambda: False)
        self._frame = None
        self._lock = threading.Lock()
        self._engaged = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = False
        self._failures = 0

    @property
    def engaged(self) -> bool:
        return self._engaged.is_set()

    def set_engaged(self, engaged: bool) -> None:
        """Called by the greeter: ~3 FPS while a person is in front of him."""
        self._engaged.set() if engaged else self._engaged.clear()

    def latest(self):
        """The cached frame, or None when nothing is streaming (never raises)."""
        with self._lock:
            return self._frame

    def grab(self):
        """Fresh frame when the stream is not running, cached frame when it is."""
        frame = self.latest()
        if frame is not None and self._thread is not None:
            return frame
        return self.grabber.grab()  # raises RuntimeError with a speakable reason

    def start(self) -> bool:
        """Poll once synchronously, then keep polling: the return value says
        whether there is a camera to stream right now, so the caller can log it.
        The thread stays up either way — a camera the daemon hands back, or a
        Reachy that gets plugged in, comes up without a restart."""
        if self._thread is not None:
            return self.available
        self._stop.clear()
        self._poll_once()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self.available

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._release()  # the held handle goes back with the stream (ADR 0002)
        with self._lock:
            self._frame = None
        self.available = False

    def _release(self) -> None:
        close = getattr(self.grabber, "close_stream", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _sample(self):
        """One poll: a held-handle read when the grabber supports it (the device
        was opened once), else the grabber's own per-call contract."""
        read = getattr(self.grabber, "read_stream", None)
        open_stream = getattr(self.grabber, "open_stream", None)
        if not (callable(read) and callable(open_stream)):
            return self.grabber.grab()
        open_stream()  # idempotent; raises + backs off on a real failure
        return read()

    def _poll_once(self) -> bool:
        """One sample; False when the device said no, which disables the stream."""
        if self._asleep():
            return False  # no sample at all while napping: no read, no inference
        try:
            frame = self._sample()
        except Exception as e:
            self._release()  # never keep a handle the next sample cannot trust
            self._failures += 1
            with self._lock:
                self._frame = None
            self.available = False
            # One line, then quiet: a missing camera is a fact about the room, not
            # an event to repeat three times a second (ADR 0006 thread budgets).
            if self._failures == 1 or self._failures % 120 == 0:
                log.info("camera stream disabled: %r", e)
                if self.journal is not None:
                    self.journal.write(J.CAMERA_DISABLED, error=str(e))
            return False
        with self._lock:
            self._frame = frame
        self.available = True
        self._failures = 0
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._asleep():
                self._stop.wait(0.5)
                continue  # asleep: the poll is fully silent, not merely cheap
            fps = self.engaged_fps if self.engaged else self.idle_fps
            self._stop.wait(1.0 / max(0.1, fps))
            self._poll_once()
