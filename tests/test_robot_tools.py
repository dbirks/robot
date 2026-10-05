"""Robot connection + vision tool wiring, hardware-free.

Nothing here may reach the live daemon or open a camera: ReachyMini is
monkeypatched and the camera is a fake.
"""

import numpy as np
import pytest

from shell.camera import CameraGrabber, daemon_holds_camera, find_reachy_camera
from shell.robot import RobotLink
from shell.robot_tools import build_handlers, make_vision_handlers


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


class FakeCamera:
    def __init__(self, frame):
        self.frame = frame
        self.grabs = 0

    def grab(self):
        self.grabs += 1
        if self.frame is None:
            raise RuntimeError("Camera is held by the robot daemon")
        return self.frame


class FakeFaces:
    def __init__(self, found):
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
    assert h["go_to_sleep"]()["ok"] is False
    motion._jobs.clear()
    assert h["go_to_sleep"]() == {"ok": True, "action": "sleeping"}
