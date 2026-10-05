"""Robot tool handlers for the shell.

The tool schemas (TOOLS) and most handlers still come from the legacy
app/robot_tools.py during the migration window; this module overrides the
ones whose legacy implementation breaks a shell invariant:

- vision tools: the legacy grabber fell back to cv2.VideoCapture(0) and the
  legacy FaceTracker held the camera open permanently. Here frames come from
  shell/camera.py, which never opens the device while the daemon holds it.
- motion tools (added with shell/motion): every pose goes through the single
  MotionOwner, never a direct goto_target racing the control loop.

Every handler returns a JSON-serializable dict and never raises.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("shell.robot_tools")

NOT_CONNECTED = {"ok": False, "error": "Robot not connected"}


def _guard(fn: Callable[..., dict]) -> Callable[..., dict]:
    def wrapped(**kwargs: Any) -> dict:
        try:
            return fn(**kwargs)
        except Exception as e:  # the never-raise contract, enforced once
            log.warning("tool %s failed: %r", fn.__name__, e)
            return {"ok": False, "error": str(e)}

    wrapped.__name__ = fn.__name__
    return wrapped


class _Faces:
    """Lazy InsightFace (CPU, ~1-2 s to load): built on the first face tool
    call, never at startup, and never given the camera."""

    def __init__(self) -> None:
        self._tracker = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._tracker is None:
                from app.face_tracker import FaceTracker

                self._tracker = FaceTracker()
            return self._tracker


def make_vision_handlers(
    camera,
    *,
    data_dir: Path = Path("data"),
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    faces: _Faces | None = None,
) -> dict[str, Callable[..., dict]]:
    faces = faces or _Faces()
    base_url = llm_base_url or os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
    model = llm_model or os.getenv("LLM_MODEL", "qwen3.5-4b")
    face_lock = threading.Lock()

    def _frame():
        return camera.grab()  # raises RuntimeError with a speakable reason

    @_guard
    def take_snapshot(**_kw: Any) -> dict:
        import cv2

        frame = _frame()
        out_dir = data_dir / "snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"snapshot_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        return {"ok": True, "path": str(path)}

    @_guard
    def describe_scene(question: str = "", **_kw: Any) -> dict:
        # Needs a vision-capable model behind LLM_BASE_URL: llama-server with
        # --mmproj (GET /props -> modalities.vision). Goes straight to
        # llama-server, not through the name-fixing proxy (no speech here).
        import cv2
        from openai import OpenAI

        frame = _frame()
        h, w = frame.shape[:2]
        if w > 448:
            frame = cv2.resize(frame, (448, int(h * 448 / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return {"ok": False, "error": "Could not encode camera frame"}
        b64 = base64.b64encode(buf.tobytes()).decode()
        prompt = question or "Describe what you see briefly in 1-2 sentences."
        client = OpenAI(base_url=base_url, api_key=os.getenv("LLM_API_KEY", "not-needed"), timeout=30.0)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=128,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return {"ok": True, "description": resp.choices[0].message.content or ""}

    @_guard
    def identify_face(**_kw: Any) -> dict:
        frame = _frame()
        with face_lock:
            ft = faces.get()
            found = ft.detect(frame)
            if not found:
                return {"ok": False, "error": "No face detected in frame"}
            results = []
            for f in found:
                name = ft.identify(f["embedding"])
                results.append({"name": name, "recognized": name is not None})
        return {"ok": True, "faces": results, "count": len(results)}

    @_guard
    def learn_face(name: str = "", **_kw: Any) -> dict:
        if not name:
            return {"ok": False, "error": "No name provided"}
        frame = _frame()
        with face_lock:
            ft = faces.get()
            found = ft.detect(frame)
            if not found:
                return {"ok": False, "error": "No face detected in frame"}
            face = found[0]  # largest
            ft.register_face(name, face["embedding"])
            known = list(ft.known_faces.keys())
        try:
            import cv2

            thumbs = data_dir / "face_thumbnails"
            thumbs.mkdir(parents=True, exist_ok=True)
            x1, y1, x2, y2 = face["bbox"]
            h, w = frame.shape[:2]
            crop = frame[max(0, y1 - 30) : min(h, y2 + 30), max(0, x1 - 30) : min(w, x2 + 30)]
            cv2.imwrite(str(thumbs / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as e:
            log.warning("face thumbnail for %s failed: %r", name, e)
        return {"ok": True, "name": name, "known_faces": known}

    @_guard
    def forget_face(name: str = "", **_kw: Any) -> dict:
        if not name:
            return {"ok": False, "error": "No name provided"}
        with face_lock:
            ft = faces.get()
            if name not in ft.known_faces:
                return {"ok": False, "error": f"No face named '{name}' found", "known_faces": list(ft.known_faces)}
            del ft.known_faces[name]
            ft._save_faces()
            return {"ok": True, "forgotten": name, "known_faces": list(ft.known_faces)}

    return {
        "take_snapshot": take_snapshot,
        "describe_scene": describe_scene,
        "identify_face": identify_face,
        "learn_face": learn_face,
        "forget_face": forget_face,
    }


LOOK_ANGLE_DEG = 30.0
LOOK_HOLD_S = 6.0  # then the gaze drifts home (or back to whoever wakes him)


def load_sdk_sound(name: str, rate: int = 16000):
    """An SDK asset (e.g. wake_up.wav) as mono int16 at `rate`, or None."""
    import importlib.resources

    import numpy as np
    import soundfile as sf

    path = Path(str(importlib.resources.files("reachy_mini") / "assets" / name))
    if not path.exists():
        return None
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    if sr != rate:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, rate)
        data = resample_poly(data, rate // g, sr // g)
    return np.clip(data * 32767.0, -32768, 32767).astype(np.int16)


def make_motion_handlers(robot, motion, *, play_sound: Callable[[str], None] | None = None, rng=None):
    """Motion tools on the single MotionOwner. Fire-and-forget: they return
    as soon as the intent is queued; the owner plays it out."""
    import random

    from .motion import nod_animation, shake_animation
    from .motion.kinematics import pose6, pose_matrix
    from .motion.owner import SLEEP_ANTENNAS, SLEEP_HEAD_POSE

    rng = rng or random.Random()

    def _ready() -> dict | None:
        if not robot.connected or motion is None:
            return dict(NOT_CONNECTED)
        return None

    def _woken() -> None:
        motion.wake()  # asked to move while asleep: get up first

    @_guard
    def look_left(**_kw: Any) -> dict:
        if err := _ready():
            return err
        _woken()
        motion.look_at(LOOK_ANGLE_DEG, hold_s=LOOK_HOLD_S, source="tool")
        return {"ok": True, "action": "look_left"}

    @_guard
    def look_right(**_kw: Any) -> dict:
        if err := _ready():
            return err
        _woken()
        motion.look_at(-LOOK_ANGLE_DEG, hold_s=LOOK_HOLD_S, source="tool")
        return {"ok": True, "action": "look_right"}

    @_guard
    def look_center(**_kw: Any) -> dict:
        if err := _ready():
            return err
        _woken()
        motion.center()
        return {"ok": True, "action": "look_center"}

    @_guard
    def nod(**_kw: Any) -> dict:
        if err := _ready():
            return err
        _woken()
        motion.play(nod_animation())
        return {"ok": True, "action": "nod"}

    @_guard
    def shake_head(**_kw: Any) -> dict:
        if err := _ready():
            return err
        _woken()
        motion.play(shake_animation())
        return {"ok": True, "action": "shake_head"}

    @_guard
    def peekaboo(**_kw: Any) -> dict:
        if err := _ready():
            return err
        hide_s = rng.uniform(1.0, 4.0)  # the suspense

        def sequence(mini) -> None:
            mini.goto_target(head=SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS, duration=1.0)
            time.sleep(hide_s)
            if play_sound is not None:
                play_sound("wake_up.wav")  # lands ON the pop-up: enqueue is instant
            mini.goto_target(head=pose_matrix(pose6(pitch=-10)), antennas=[0.5, 0.5], duration=0.2)
            time.sleep(0.8)
            mini.goto_target(head=pose_matrix(pose6()), antennas=[-0.1745, 0.1745], duration=0.8)

        _woken()
        if not motion.exclusive("peekaboo", sequence):
            return {"ok": False, "error": "Busy with another move - wait for it to finish"}
        return {"ok": True, "action": "peekaboo"}

    @_guard
    def go_to_sleep(**_kw: Any) -> dict:
        if err := _ready():
            return err
        if motion.sleeping:
            return {"ok": True, "action": "already_sleeping"}
        if not motion.sleep():
            return {"ok": False, "error": "Busy with another move - wait for it to finish"}
        return {"ok": True, "action": "sleeping"}

    return {
        "look_left": look_left,
        "look_right": look_right,
        "look_center": look_center,
        "nod": nod,
        "shake_head": shake_head,
        "peekaboo": peekaboo,
        "go_to_sleep": go_to_sleep,
    }


def build_handlers(
    robot, camera, *, data_dir: Path = Path("data"), motion=None, play_sound=None
) -> tuple[list[dict], dict]:
    """(TOOLS, handlers) for the router. Never raises: a missing robot SDK or
    legacy package still yields a running shell with fewer tools."""
    try:
        from app.robot_tools import TOOLS, make_handlers
    except Exception as e:
        log.warning("legacy robot tools unavailable: %r", e)
        return [], {}
    handlers: dict = {}
    try:
        handlers = dict(make_handlers(robot, movement=motion))
    except Exception as e:
        log.warning("legacy robot tool construction failed: %r", e)
    handlers.update(make_vision_handlers(camera, data_dir=data_dir))
    # play_emotion stays legacy: it feeds emotion keyframes to
    # motion.queue_animation (the MovementManager-compatible adapter).
    handlers.update(make_motion_handlers(robot, motion, play_sound=play_sound))
    return TOOLS, handlers
