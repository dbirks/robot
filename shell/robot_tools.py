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
    # The name is resolved once, here, and never read inside `wrapped`:
    # `functools.partial` and callable class instances have no __name__, and
    # reading it in the `except` block would raise from the one place where a
    # raise escapes the never-raise contract, while reading it to stamp the
    # wrapper raises before the wrapper exists at all.
    name = getattr(fn, "__name__", None)
    label = name if isinstance(name, str) else repr(fn)

    def wrapped(**kwargs: Any) -> dict:
        try:
            return fn(**kwargs)
        except Exception as e:  # the never-raise contract, enforced once
            log.warning("tool %s failed: %r", label, e)
            return {"ok": False, "error": str(e)}

    wrapped.__name__ = label
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
    # A text-only chat LLM (Granite) pairs with a small separate VLM server:
    # VISION_BASE_URL/VISION_MODEL; unset = the chat LLM must have --mmproj.
    base_url = llm_base_url or os.getenv("VISION_BASE_URL") or os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
    model = llm_model or os.getenv("VISION_MODEL") or os.getenv("LLM_MODEL", "qwen3.5-4b")
    face_lock = threading.Lock()

    def _frame():
        # Frames come from the one camera owner: when a CameraStream is running for
        # the greeter, `latest()` is its cached frame and this module never opens
        # the device (ADR 0002). With no stream, the grabber's open-per-grab path
        # still applies, and its speakable RuntimeError is the contract.
        latest = getattr(camera, "latest", None)
        frame = latest() if callable(latest) else None
        if frame is not None:
            return frame
        return camera.grab()

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
        # Needs a vision-capable model behind base_url: llama-server with
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


def make_motion_handlers(robot, motion, *, play_sound: Callable[[str], None] | None = None, rng=None, clock=None):
    """Motion tools on the single MotionOwner. Fire-and-forget: they return
    as soon as the intent is queued; the owner plays it out.

    `clock` exists for the dance journal rate-limit, which is the one behaviour here that
    depends on wall time; the tests inject a fake rather than sleeping (same reason as `rng`).
    """
    import random

    from . import journal as J
    from .motion import dances, emotes, nod_animation, shake_animation
    from .motion.kinematics import pose6, pose_matrix
    from .motion.owner import SLEEP_ANTENNAS, SLEEP_HEAD_POSE

    rng = rng or random.Random()
    clock = clock or time.monotonic

    def _ready() -> dict | None:
        if not robot.connected or motion is None:
            return dict(NOT_CONNECTED)
        return None

    def _woken() -> None:
        # Asleep is one state, not a pose (issue #36): any request that needs
        # the head moves that same state, so being asked to nod also reopens
        # the microphone. MotionOwner.wake() is the only path back up.
        motion.wake()

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
        # The flag moves even when the head is mid-move: the pose queues behind
        # it, but the microphone closes now, because he must never keep
        # answering the room lying down (issue #36).
        if not motion.sleep(reason="tool"):
            return {"ok": True, "action": "already_sleeping"}
        return {"ok": True, "action": "sleeping"}

    @_guard
    def play_emotion(emotion: str = "", **_kw: Any) -> dict:
        """Whole-body gesture from the emotion library, on the one motion owner.

        Replaces the legacy bridge handler (issue #22 deletes `app/robot_tools.py`, whose version
        also had to load keyframes through `app/emotion_loader.py` and the legacy keyframe type).
        Three behaviours changed on the way over, each because the old one would have hurt us at
        the booth:

        * an unreadable library no longer reports `Unknown emotion`. The old message told the
          model the word it used does not exist, which invites a retry - and a retry queues
          behind an animation that is already in flight, the same shape as tonight's snapshot
          pile-in;
        * it does not preempt. `preempt=True, priority=1` used to yank the head off whoever he was
          looking at, mid-sentence;
        * the result says what was reserved (keyframes, real duration, whether it woke him), and
          the failure path is offline-only by construction - "ok: true" over an unpowered servo
          bus is exactly how the camera tool lied earlier tonight.
        """
        if err := _ready():
            return err
        name = str(emotion or "").strip().lower()
        if not name:
            return {
                "ok": False,
                "error": "No emotion specified",
                "available": list(emotes.EMOTE_NAMES),
            }
        journal = getattr(motion, "journal", None)
        try:
            anim = emotes.load(name)
        except emotes.UnknownEmote:
            # Says nothing about the library, because it isn't the library's problem; the model
            # just picks a word from `available`.
            return {
                "ok": False,
                "error": f"Unknown emotion: {name}",
                "available": list(emotes.EMOTE_NAMES),
            }
        except emotes.EmoteUnavailable as e:
            # Names the real fault and forbids the retry: this is a box-setup problem
            # (scripts/fetch_emotions.py), not a wrong word, and no other emotion will work
            # either while the library cannot be read.
            error = (
                f"The emotion library is not readable on this box ({e.name}); do not try another "
                "emotion - answer in words instead"
            )
            if journal is not None:
                journal.write(J.EMOTION_PLAYED, name=name, ok=False, error=error[:120])
            return {"ok": False, "error": error, "retry": False}

        was_asleep = bool(getattr(motion, "sleeping", False))
        _woken()  # issue #36: a request that needs the head reopens the microphone as well
        motion.play(anim)
        seconds = round(sum(kf.duration for kf in anim.keyframes), 2)
        if journal is not None:
            journal.write(
                J.EMOTION_PLAYED,
                name=name,
                ok=True,
                keyframes=len(anim.keyframes),
                duration_s=seconds,
                woke=was_asleep,
            )
        return {
            "ok": True,
            "action": "play_emotion",
            "emotion": name,
            "duration_s": seconds,
            "woke": was_asleep,
        }

    _dance_journal_t: dict[str, float] = {}

    @_guard
    def dance(move: str = "", **_kw: Any) -> dict:
        """A dance routine from the vendor's dances library, on the same owner and the same
        keyframe math as `play_emotion`.

        Deliberately NOT in `shell.tools.STATE_CHANGING` - see the comment on that set. The
        containment is physical rather than attentional and all of it is inherited from the
        emote path: priority 0 with no preemption (it overlays the gaze instead of yanking it),
        `REACHY_DANCE_MAX_S` scaling a too-long routine, and `tools.exclude: [dance]` per room
        for an operator who wants the option gone entirely.
        """
        if err := _ready():
            return err
        name = str(move or "").strip().lower()
        if not name:
            return {
                "ok": False,
                "error": "No dance specified",
                "available": list(dances.DANCE_NAMES),
            }
        journal = getattr(motion, "journal", None)
        try:
            anim = dances.load(name)
        except dances.UnknownDance:
            return {
                "ok": False,
                "error": f"Unknown dance: {name}",
                "available": list(dances.DANCE_NAMES),
            }
        except dances.DanceUnavailable as e:
            # The same lie must not be available here twice: an unreadable pack reported as a bad
            # word gets retried, and a retry queues behind the animation already in flight.
            error = (
                f"The dance library is not readable on this box ({e.name}); do not try another "
                "dance - answer in words instead"
            )
            if journal is not None:
                # Never rate-limited: "the pack is missing" arriving once is exactly the row
                # someone needs to see at the table.
                journal.write(J.DANCE_PLAYED, name=name, ok=False, error=error[:120])
            return {"ok": False, "error": error, "retry": False}

        was_asleep = bool(getattr(motion, "sleeping", False))
        _woken()  # issue #36: a request that needs the head reopens the microphone as well
        motion.play(anim)  # no preempt: it queues behind the live gaze
        seconds = round(sum(kf.duration for kf in anim.keyframes), 2)
        now = clock()
        if journal is not None and now - _dance_journal_t.get(name, -1e9) >= dances.DANCE_JOURNAL_S:
            # Rate-limited per name, the way MOTION_SCAN and the greeter suppression row are.
            _dance_journal_t[name] = now
            journal.write(
                J.DANCE_PLAYED,
                name=name,
                ok=True,
                keyframes=len(anim.keyframes),
                duration_s=seconds,
                woke=was_asleep,
            )
        return {
            "ok": True,
            "action": "dance",
            "move": name,
            "duration_s": seconds,
            "woke": was_asleep,
        }

    return {
        "look_left": look_left,
        "look_right": look_right,
        "look_center": look_center,
        "nod": nod,
        "shake_head": shake_head,
        "peekaboo": peekaboo,
        "go_to_sleep": go_to_sleep,
        "play_emotion": play_emotion,
        "dance": dance,
    }


def build_handlers(
    robot, camera, *, data_dir: Path = Path("data"), motion=None, play_sound=None, memories=None
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
    tools = list(TOOLS)
    if memories is not None:
        # shell-native memory replaces the legacy `remember` (it needed the
        # old agent object and answered "Agent not available" here)
        from .memories import MEMORY_TOOLS, make_memory_handlers

        names = {t["function"]["name"] for t in MEMORY_TOOLS}
        tools = [t for t in tools if t.get("function", {}).get("name") not in names] + MEMORY_TOOLS
        handlers.update(make_memory_handlers(memories))
    # play_emotion is shell-native now - it was the last tool still leaning on
    # app/emotion_loader.py and the legacy MovementManager keyframe type. Same replacement dance
    # as the memory tools: drop the legacy schema, add ours, and keep exactly ONE definition of
    # the name in the array. Two entries for one tool means the service picks whichever it feels
    # like, and the two descriptions disagree about the vocabulary.
    from .motion import emotes as _emotes

    names = {t["function"]["name"] for t in _emotes.EMOTE_TOOLS}
    tools = [t for t in tools if t.get("function", {}).get("name") not in names] + _emotes.EMOTE_TOOLS
    # Same replacement dance for `dance`, which has no legacy schema to drop - it is purely
    # additive, and `make_motion_handlers` already supplies the handler above.
    from .motion import dances as _dances

    dance_names = {t["function"]["name"] for t in _dances.DANCE_TOOLS}
    tools = [t for t in tools if t.get("function", {}).get("name") not in dance_names] + _dances.DANCE_TOOLS
    return tools, handlers
