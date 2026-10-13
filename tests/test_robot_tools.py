"""Robot connection + vision tool wiring, hardware-free.

Nothing here may reach the live daemon or open a camera: ReachyMini is
monkeypatched and the camera is a fake.
"""

import numpy as np
import pytest

# These tests exercise the SDK-backed paths (app/robot_tools.py imports reachy_mini at
# module level; the motion owner and DOA read its pose types). CI deliberately runs the
# hardware-free suite in a minimal env without the SDK - installing it would pull the
# multi-GB torch stack the lint workflow goes out of its way to avoid (ADR 0004) - so
# skip here rather than report a failure that means "this machine has no robot library".
# On the robot box, where the SDK is installed, they run.
pytest.importorskip("reachy_mini")

from shell.camera import CameraGrabber, daemon_holds_camera, find_reachy_camera
from shell.robot import RobotLink
from shell.robot_tools import _Faces, _guard, build_handlers, make_vision_handlers


class Boom:
    def __init__(self, *a, **k):
        raise ConnectionError("no daemon")


@pytest.fixture
def no_daemon(monkeypatch):
    import reachy_mini

    monkeypatch.setattr(reachy_mini, "ReachyMini", Boom)


def test_robot_absent_degrades(no_daemon, journal):
    robot = RobotLink(journal)
    assert robot.connect() is False
    assert robot.connected is False
    assert journal.find("robot.connection") == [{"connected": False, "host": robot.config.reachy_host}]
    robot.disconnect()  # no-op, no raise


def test_motion_tools_say_not_connected(no_daemon, tmp_path):
    robot = RobotLink()
    robot.connect()
    tools, handlers = build_handlers(robot, FakeCamera(None), data_dir=tmp_path)
    names = {t["function"]["name"] for t in tools}
    for name in ("look_left", "nod", "shake_head", "play_emotion", "peekaboo", "go_to_sleep"):
        assert name in names
        assert handlers[name]() == {"ok": False, "error": "Robot not connected"}
    for name in ("take_snapshot", "describe_scene", "identify_face", "learn_face", "forget_face"):
        assert name in handlers


def test_guard_holds_for_a_callable_without_a_name():
    """The never-raise boundary must not be the thing that raises (issue #44).

    `_guard` used to read `fn.__name__` twice outside the wrapper it installs:
    once to stamp `wrapped.__name__`, which raises while the handler table is
    being BUILT, and once inside its own `except` block, which raises from the
    one place a raise escapes the contract in AGENTS.md (tools never raise,
    always return a JSON-serializable dict). Not reachable from the handlers in
    shell/robot_tools.py today - every `_guard` there decorates a nested `def` -
    but `_guard` is the module's only safety net and any injected callable can
    reach it: functools.partial and callable class instances have no `__name__`.
    """
    import functools

    def boom(scale, **_kw):
        raise RuntimeError(f"tool exploded x{scale}")

    partial = functools.partial(boom, 2)
    assert not hasattr(partial, "__name__"), "the point of the test: nameless"
    guarded = _guard(partial)  # used to raise AttributeError here
    assert guarded() == {"ok": False, "error": "tool exploded x2"}  # ...and here


def test_guard_keeps_the_wrapped_name_for_named_handlers():
    """The stamp above is what makes a warning line name the tool that failed."""

    def look_left(**_kw):
        return {"ok": True, "action": "look_left"}

    guarded = _guard(look_left)
    # getattr, not `guarded.__name__`: `_guard` is declared to return a plain
    # `Callable[..., dict]`, which does not promise a name (that is the same
    # typing artefact that made the old code un-checkable at robot_tools.py:37)
    assert getattr(guarded, "__name__", None) == "look_left"


class FakeCamera:
    def __init__(self, frame):
        self.frame = frame
        self.grabs = 0

    def grab(self):
        self.grabs += 1
        if self.frame is None:
            raise RuntimeError("Camera is held by the robot daemon")
        return self.frame


class FakeFaces(_Faces):
    """Stand-in for the lazy InsightFace wrapper.

    A `_Faces` subclass rather than an unrelated duck-type: `make_vision_handlers`
    declares `faces: _Faces | None` and reads only `.get()`, and the honest way to
    satisfy that signature is for the fake to be one (issue #44 - fixing the fake,
    not widening the parameter type). `_Faces.__init__` loads nothing; InsightFace
    is still never imported here.
    """

    def __init__(self, found):
        super().__init__()
        self.found = found
        self.known_faces = {}
        self.saved = 0

    def get(self):
        return self

    def detect(self, frame):
        return self.found

    def identify(self, emb):
        return "David" if emb[0] > 0 else None

    def register_face(self, name, emb):
        self.known_faces[name] = emb

    def _save_faces(self):
        self.saved += 1


def test_vision_tools_never_raise_without_camera(tmp_path):
    h = make_vision_handlers(FakeCamera(None), data_dir=tmp_path, faces=FakeFaces([]))
    for name in ("take_snapshot", "describe_scene", "identify_face"):
        out = h[name]()
        assert out == {"ok": False, "error": "Camera is held by the robot daemon"}
    assert h["learn_face"]()["error"] == "No name provided"


def test_snapshot_writes_jpeg(tmp_path):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    out = make_vision_handlers(FakeCamera(frame), data_dir=tmp_path)["take_snapshot"]()
    assert out["ok"] is True and out["path"].endswith(".jpg")
    assert (tmp_path / "snapshots").exists()


def test_face_tools_with_fake_tracker(tmp_path):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    found = [{"bbox": [10, 10, 50, 50], "embedding": np.array([1.0, 0.0])}]
    faces = FakeFaces(found)
    h = make_vision_handlers(FakeCamera(frame), data_dir=tmp_path, faces=faces)
    assert h["identify_face"]() == {"ok": True, "faces": [{"name": "David", "recognized": True}], "count": 1}
    assert h["learn_face"](name="Ana")["known_faces"] == ["Ana"]
    assert (tmp_path / "face_thumbnails" / "Ana.jpg").exists()
    assert h["forget_face"](name="Ana")["ok"] is True
    assert h["forget_face"](name="Ana")["ok"] is False


def test_daemon_holds_camera_fails_closed():
    assert daemon_holds_camera(None) is True
    assert daemon_holds_camera({"available": True, "released": False}) is True
    assert daemon_holds_camera({"available": False, "released": False}) is False  # no media server
    assert daemon_holds_camera({"available": False, "released": True}) is False


def test_find_reachy_camera(tmp_path):
    for n, name in [(2, "Integrated Webcam"), (1, "Reachy Mini Camera"), (0, "Reachy Mini Camera")]:
        d = tmp_path / f"video{n}"
        d.mkdir()
        (d / "name").write_text(name + "\n")
    assert find_reachy_camera(tmp_path) == 0
    assert find_reachy_camera(tmp_path / "missing") is None


def test_grabber_refuses_when_daemon_holds_camera(monkeypatch):
    cam = CameraGrabber(mode="auto")
    monkeypatch.setattr(cam, "_media_status", lambda: {"available": True, "released": False})
    with pytest.raises(RuntimeError, match="held by the robot daemon"):
        cam.grab()
    with pytest.raises(RuntimeError, match="disabled"):
        CameraGrabber(mode="off").grab()


def test_motion_tools_drive_the_owner():
    import random

    from shell.motion import MotionOwner
    from shell.robot_tools import make_motion_handlers

    class Mini:
        def goto_target(self, **k):
            pass

    class Robot:
        mini = Mini()
        connected = True

    robot = Robot()
    motion = MotionOwner(robot, rng=random.Random(0))
    h = make_motion_handlers(robot, motion, play_sound=lambda name: None, rng=random.Random(0))
    assert h["look_left"]() == {"ok": True, "action": "look_left"}
    assert motion.gaze_target_yaw_deg == pytest.approx(30)
    assert h["look_right"]()["ok"] and motion.gaze_target_yaw_deg == pytest.approx(-30)
    assert h["look_center"]()["ok"] and motion.gaze_source == "home"
    assert h["nod"]()["ok"] and h["shake_head"]()["ok"]
    assert [a.name for a in motion._anims] == ["nod", "shake_head"]
    assert h["peekaboo"]()["ok"] is True
    assert h["peekaboo"]()["ok"] is False  # busy: one exclusive move at a time
    # Issue #36: sleeping is a state, not an exclusive move. With a move still
    # queued he goes asleep anyway (the pose queues behind it) so the
    # microphone closes now instead of after the peekaboo finishes.
    assert h["go_to_sleep"]() == {"ok": True, "action": "sleeping"}
    assert motion.sleeping
    assert h["go_to_sleep"]() == {"ok": True, "action": "already_sleeping"}
    motion._jobs.clear()
    assert h["nod"]()["ok"]  # any physical tool moves the same state back
    assert not motion.sleeping
