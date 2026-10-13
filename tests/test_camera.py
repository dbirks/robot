"""The one continuous camera owner: cache, cadence, and degrading to absent.

`FakeGrabber` stands in for /dev/videoN: no device is opened here any more than
the greeter opens one (ADR 0002 is about who touches the hardware, and the answer
has to stay "one object"). `FakeCapture` goes one level deeper - it is the capture
behind `CameraGrabber._open`, on a fake clock, so the warm-up deadline and its reason
are the real ones while nothing sleeps and no node is touched.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest
from conftest import RecJournal

from shell import camera as camera_module
from shell.camera import (
    WARM_MAX_READS,
    WARM_POLL_GAP_S,
    CameraGrabber,
    CameraStream,
    daemon_holds_camera,
    warmup_deadline_s,
)


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


# ---- the warm-up deadline: a silent device must not occupy a tool call ----


class FakeClock:
    """Time that only moves when the fake device says it did."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class FakeCapture:
    """The `cv2.VideoCapture` behind `CameraGrabber._open`.

    `block_s` is charged to the shared clock per read, which is exactly what an
    unpowered Reachy looks like from Python: the read comes back `(False, None)` only
    after the kernel's V4L2 select() timeout. Nothing here sleeps, and nothing here
    touches a node. `_device()` wires this capture to the grabber's clock, which is the
    whole trick: the blocked read consumes the deadline it is then judged against.
    """

    def __init__(self, clock, *, block_s=0.0, frame_after=None, opened=True):
        self.clock = clock
        self.block_s = block_s
        self.frame_after = frame_after  # None => the device never yields a frame
        self.opened = opened
        self.reads = 0
        self.released = 0

    def isOpened(self):
        return self.opened

    def set(self, _prop, _value):
        pass

    def read(self):
        self.reads += 1
        self.clock.advance(self.block_s)
        if self.frame_after is not None and self.reads > self.frame_after:
            return True, f"frame-{self.reads}"
        return False, None

    def release(self):
        self.released += 1


class FakeCv2:
    CAP_V4L2 = 200
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, capture):
        self.capture = capture
        self.opens = 0

    def VideoCapture(self, _index, _api):
        self.opens += 1
        return self.capture


def _device(monkeypatch, capture, *, index=3, deadline=6.0, clock=None, sleep=None):
    """A real `CameraGrabber` whose /dev/videoN is `capture`: the real warm-up loop and
    the real reason string, with no cv2, no device and no sleeping."""
    cv2 = FakeCv2(capture)
    clock = clock or FakeClock()
    capture.clock = clock  # the fake charges its blocked reads to the grabber's clock
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(camera_module, "find_reachy_camera", lambda *a, **k: index)
    grabber = CameraGrabber(
        "http://localhost:1",
        mode="auto",
        deadline_s=deadline,
        clock=clock,
        sleep=sleep if sleep is not None else (lambda _s: None),
    )
    monkeypatch.setattr(grabber, "_fetch_media_status", lambda: {"available": False})  # daemon not holding
    return grabber, clock, cv2


def test_the_warmup_stops_at_the_first_good_frame(monkeypatch):
    cap = FakeCapture(FakeClock(), frame_after=2)
    grabber, _clock, cv2 = _device(monkeypatch, cap)
    assert grabber.grab() == "frame-3"
    assert cap.reads == 3, "it kept reading after it had a usable frame"
    assert cv2.opens == 1 and cap.released == 1  # one handle, released on the way out


def test_a_silent_device_costs_one_blocked_read_not_five(monkeypatch):
    """Tonight's ~50 s: 5 warm-up reads x the ~10 s V4L2 select() timeout, on a robot
    with no power. The deadline cannot interrupt a read, only refuse the next one."""
    cap = FakeCapture(FakeClock(), block_s=10.0)
    grabber, _clock, _cv2 = _device(monkeypatch, cap)
    with pytest.raises(RuntimeError) as e:
        grabber.grab()
    reason = str(e.value)
    assert cap.reads == 1, f"the deadline did not stop the warm-up: {reason}"
    assert reason.startswith("no frames from /dev/video3 in 10.0s (device opened; check the robot is powered")
    assert "1 read(s)" in reason and "6.0s warm-up deadline" in reason, "the reason must say what it tried"
    assert cap.released == 1


def test_an_instantly_refusing_device_spends_its_wall_clock_window_not_its_read_budget(monkeypatch):
    """A cold UVC camera answers `(False, None)` *instantly* for its first seconds - auto-exposure
    settling, measured 2.7-3.0 s to the first frame on this box. The lane that added the wall-clock
    deadline left the read cap at 5, so five free reads burned ~0 s and the shell journalled
    `camera.disabled` at start-up: no greeter, no vision tools, for the session. The name of the
    test it replaces - "uses its frame budget" - is precisely the bug.

    Reads are now paced, so the deadline is what actually expires. `clock` and `sleep` are the same
    fake clock here: that is the whole point, the retries have to cost the warm-up its seconds.
    """
    clock = FakeClock()

    def paced_sleep(seconds):
        clock.advance(seconds)

    cap = FakeCapture(clock)  # never yields a frame, and never blocks
    grabber, _clock, _cv2 = _device(monkeypatch, cap, clock=clock, sleep=paced_sleep)
    with pytest.raises(RuntimeError) as e:
        grabber.grab()
    assert cap.reads > 20, f"only {cap.reads} reads: it gave up on a free-retry device again"
    assert cap.reads < WARM_MAX_READS, "the backstop cap, not the deadline, ended the warm-up"
    assert 5.5 <= _clock.t <= 6.6, f"the 6 s deadline was not the bound: {_clock.t:.2f}s"
    assert "no frames from /dev/video3 in 6.0s" in str(e.value)
    assert cap.released == 1


def test_a_cold_camera_that_needs_three_seconds_is_not_called_dead(monkeypatch):
    """The one that would have cost the booth its greeter: first good frame at ~2.8 s, well inside
    the 6 s window and well outside the old 5-read budget."""
    clock = FakeClock()
    cap = FakeCapture(clock, block_s=0.4, frame_after=7)  # 7 x 0.4 s of settling, then a frame
    grabber, _clock, _cv2 = _device(monkeypatch, cap, clock=clock)
    assert grabber.grab() == "frame-8", "a healthy cold open was abandoned"
    assert cap.reads == 8
    assert 2.4 <= _clock.t <= 3.4, f"the frame arrived at {_clock.t:.2f}s, not the measured ~2.8s"
    assert cap.released == 1


def test_the_deadline_is_configurable_and_a_blank_override_does_not_crash(monkeypatch):
    monkeypatch.setenv("REACHY_CAMERA_WARMUP_DEADLINE_S", "2")
    assert CameraGrabber(mode="off").warmup_deadline_s == 2.0
    monkeypatch.setenv("REACHY_CAMERA_WARMUP_DEADLINE_S", "")  # the float("") class of bug
    assert CameraGrabber(mode="off").warmup_deadline_s == 6.0 == warmup_deadline_s()
    monkeypatch.setenv("REACHY_CAMERA_WARMUP_DEADLINE_S", "0")  # 0 would never read a frame
    assert CameraGrabber(mode="off").warmup_deadline_s == 6.0


def test_the_held_path_pays_the_same_deadline_and_backs_off(monkeypatch):
    cap = FakeCapture(FakeClock(), block_s=10.0)
    grabber, clock, cv2 = _device(monkeypatch, cap)
    with pytest.raises(RuntimeError) as e:
        grabber.open_stream()
    assert "check the robot is powered" in str(e.value) and cap.reads == 1
    assert cv2.opens == 1 and cap.released == 1 and grabber._cap is None
    with pytest.raises(RuntimeError, match="reopen backoff"):
        grabber.open_stream()  # the greeter's next sample: a refusal, not another blocked read
    assert cv2.opens == 1 and cap.reads == 1
    clock.advance(CameraGrabber.REOPEN_BACKOFF_S)
    with pytest.raises(RuntimeError):
        grabber.open_stream()  # expired backoff: exactly one more bounded attempt
    assert cap.reads == 2


def test_a_stalled_sample_releases_the_handle_instead_of_spinning(monkeypatch):
    cap = FakeCapture(FakeClock(), frame_after=0)  # a frame on every read
    grabber, _clock, _cv2 = _device(monkeypatch, cap)
    grabber.open_stream()
    cap.block_s = 10.0  # the robot loses power mid-visit: the next sample blocks out
    with pytest.raises(RuntimeError) as e:
        grabber.read_stream()
    assert str(e.value).startswith("late frame from /dev/video3 after 10.0s (device opened; check the robot is powered")
    assert cap.released == 1 and grabber._cap is None
    with pytest.raises(RuntimeError, match="reopen backoff"):
        grabber.open_stream()


def test_a_failed_sample_names_the_node_it_was_holding(monkeypatch):
    cap = FakeCapture(FakeClock(), frame_after=0)
    grabber, _clock, _cv2 = _device(monkeypatch, cap, index=1)
    grabber.open_stream()
    cap.frame_after = None  # quiet from here on, and quick about it
    with pytest.raises(RuntimeError) as e:
        grabber.read_stream()
    assert "no frames from /dev/video1 in 0.0s" in str(e.value)


def test_the_daemon_reclaiming_the_camera_does_not_wedge_the_handle(monkeypatch):
    """`close_stream()` takes the lock, so calling it from inside a locked `read_stream()`
    deadlocked the camera thread *holding the lock*: every later `grab()` then blocked on
    it with no timeout at all. Same pile-up as tonight, one layer down."""
    cap = FakeCapture(FakeClock(), frame_after=0)
    grabber, _clock, _cv2 = _device(monkeypatch, cap)
    grabber.open_stream()
    monkeypatch.setattr(grabber, "_media_status", lambda **_kw: {"available": True, "released": False})
    said = []

    def sample():
        try:
            grabber.read_stream()
        except RuntimeError as e:
            said.append(str(e))

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    thread.join(0.5)
    assert not thread.is_alive(), "read_stream deadlocked on its own lock"
    assert said == ["Camera reclaimed by the robot daemon"]
    assert grabber._cap is None and cap.released == 1
    assert grabber._lock.acquire(timeout=0.2), "the camera lock is still held"
    grabber._lock.release()


def test_a_warmup_timeout_is_one_journal_row_with_a_reason_not_a_traceback_per_frame(monkeypatch):
    """The greeter journals its decisions, so an unpowered robot has to read as one
    row an operator can act on, then quiet."""
    cap = FakeCapture(FakeClock(), block_s=10.0)
    grabber, clock, _cv2 = _device(monkeypatch, cap)
    journal = RecJournal()
    stream = CameraStream(grabber, journal=journal, idle_fps=50.0)
    for _ in range(6):
        clock.advance(0.5)  # the poll's own cadence: far shorter than the 2 s backoff
        assert stream._poll_once() is False
    assert cap.reads == 2, f"six polls touched the device {cap.reads} times, not the two the backoff allows"
    rows = journal.find("camera.disabled")
    assert len(rows) == 1 and "check the robot is powered" in rows[0]["error"]
    assert stream.latest() is None and stream.available is False
    clock.advance(CameraGrabber.REOPEN_BACKOFF_S * 2)  # the doubled backoff expires
    assert stream._poll_once() is False
    assert cap.reads == 3 and len(journal.find("camera.disabled")) == 1  # retried, quietly
