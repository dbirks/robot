"""The one continuous camera owner: cache, cadence, and degrading to absent.

`FakeGrabber` stands in for /dev/videoN: no device is opened here any more than
the greeter opens one (ADR 0002 is about who touches the hardware, and the answer
has to stay "one object").
"""

from __future__ import annotations

import time

from conftest import RecJournal

from shell.camera import CameraGrabber, CameraStream, daemon_holds_camera


class FakeGrabber:
    def __init__(self, error=None):
        self.grabs = 0
        self.error = error

    def grab(self):
        self.grabs += 1
        if self.error:
            raise RuntimeError(self.error)
        return f"frame-{self.grabs}"


class FakeStreamGrabber:
    """The held-handle path (`CameraGrabber`'s open_stream/read_stream/close_stream),
    so the cost model of the poll can be asserted without /dev/videoN."""

    def __init__(self):
        self.opens = 0
        self.reads = 0
        self.closes = 0
        self.open_error = None
        self.read_errors = 0  # fail this many upcoming reads, then behave
        self._cap = None

    def open_stream(self):
        if self._cap is not None:
            return  # idempotent, like CameraGrabber's: holding the handle is not a re-open
        self.opens += 1
        if self.open_error:
            raise RuntimeError(self.open_error)
        self._cap = object()

    def read_stream(self):
        if self._cap is None:
            raise RuntimeError("camera stream is not open")
        if self.read_errors:
            self.read_errors -= 1
            raise RuntimeError("No frame available from camera")
        self.reads += 1
        return f"held-{self.reads}"

    def close_stream(self):
        self.closes += 1
        self._cap = None

    def grab(self):  # pragma: no cover - the stream never uses it when reads exist
        return "fallback"


def stream(*, error=None, **kw):
    grabber = FakeGrabber(error)
    kw.setdefault("journal", _journal())
    return CameraStream(grabber, **kw), grabber


def _journal():
    from conftest import RecJournal

    return RecJournal()


def test_poll_once_without_a_thread_does_not_start_streaming():
    s, grabber = stream()
    assert s.latest() is None
    assert s.available is False and grabber.grabs == 0


def test_start_caches_frames_and_serves_latest():
    s, grabber = stream(idle_fps=20.0, engaged_fps=20.0)
    assert s.start() is True and s.available is True
    for _ in range(50):
        if s.latest() is not None:
            break
        time.sleep(0.01)
    assert s.latest() == "frame-1"
    s.stop()
    assert s.latest() is None and s.available is False
    assert s.start() is True  # restartable
    s.stop()


def test_a_missing_camera_disables_the_stream_once_and_stays_quiet():
    s, grabber = stream(error="Reachy Mini camera not found", idle_fps=50.0)
    assert s.start() is False and s.available is False and s.latest() is None
    time.sleep(0.2)
    assert grabber.grabs >= 2  # it keeps asking, so a plugged-in camera comes up
    journal = s.journal
    assert sum(1 for t, _ in journal.events if t == "camera.disabled") == 1


def test_the_stream_recovers_when_the_daemon_hands_the_camera_back():
    s, grabber = stream(error="Camera is held by the robot daemon", idle_fps=50.0)
    s.start()
    grabber.error = None
    for _ in range(100):
        if s.latest() is not None:
            break
        time.sleep(0.01)
    assert s.available is True
    s.stop()


def test_grab_falls_back_to_the_grabber_when_nothing_is_streaming():
    s, grabber = stream()
    assert s.grab() == "frame-1" and grabber.grabs == 1  # no thread: pass through
    s._frame = "cached"
    s._thread = object()  # pretend the poll thread is up
    assert s.grab() == "cached" and grabber.grabs == 1
    s._thread = None


def test_engage_switches_the_stated_cadence_only():
    s, _ = stream(idle_fps=2.0, engaged_fps=3.0)
    assert s.engaged is False
    s.set_engaged(True)
    assert s.engaged is True
    s.set_engaged(False)
    assert s.engaged is False


def test_the_plain_grabber_still_disables_itself_with_no_camera(monkeypatch, tmp_path):
    monkeypatch.setenv("REACHY_CAMERA", "off")
    grabber = CameraGrabber("http://localhost:1")  # never contacted: mode wins
    for _ in range(5):
        try:
            grabber.grab()
        except RuntimeError as e:
            assert "Camera disabled" in str(e)
            return
    raise AssertionError("REACHY_CAMERA=off did not disable capture")


# ---- the held-handle poll: open once, read per sample, release on stop ----


def _wait(predicate, limit_s=2.0):
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_held_camera_is_opened_once_and_read_per_sample():
    grabber = FakeStreamGrabber()
    s = CameraStream(grabber, idle_fps=50.0, journal=RecJournal())
    assert s.start() is True
    assert _wait(lambda: grabber.reads >= 5)
    assert grabber.opens == 1  # no per-sample VideoCapture, no per-sample warmup
    assert s.latest() is not None
    s.stop()
    assert grabber.closes == 1 and s.latest() is None  # the handle leaves with the stream


def test_a_failed_read_releases_and_reopens_without_losing_the_stream():
    grabber = FakeStreamGrabber()
    s = CameraStream(grabber, idle_fps=50.0, journal=RecJournal())
    s.start()
    assert _wait(lambda: grabber.reads >= 2)
    grabber.read_errors = 1
    assert _wait(lambda: grabber.reads >= 4)  # recovered at the next sample
    assert grabber.closes >= 1 and s.available is True
    journal = s.journal
    assert journal is not None
    assert sum(1 for t, _ in journal.events if t == "camera.disabled") <= 1  # a fact, not an event
    s.stop()


def test_asleep_means_no_samples_at_all():
    grabber = FakeStreamGrabber()
    asleep = [True]
    s = CameraStream(grabber, idle_fps=50.0, journal=RecJournal(), asleep=lambda: asleep[0])
    assert s.start() is False  # the synchronous probe refuses to touch the device
    time.sleep(0.1)
    assert grabber.opens == 0 and grabber.reads == 0
    asleep[0] = False  # a wake word does not need frames, but the greeter afterwards does
    assert _wait(lambda: grabber.reads >= 1)
    assert s.available is True
    s.stop()


def test_media_status_is_cached_behind_a_ttl_on_the_held_path(monkeypatch):
    grabber = CameraGrabber("http://localhost:1")  # never contacted: the fetch is faked
    calls = []

    def fake_fetch():
        calls.append(1)
        return {"available": False}  # daemon not holding: the contract that lets us open

    monkeypatch.setattr(grabber, "_fetch_media_status", fake_fetch)
    for _ in range(5):  # the poll's path: one GET per TTL window, not per sample
        assert not daemon_holds_camera(grabber._media_status(ttl=60.0))
    assert len(calls) == 1
    assert not daemon_holds_camera(grabber._media_status())  # the one-shot path never caches
    assert len(calls) == 2
