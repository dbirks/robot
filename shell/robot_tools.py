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


def build_handlers(robot, camera, *, data_dir: Path = Path("data"), motion=None) -> tuple[list[dict], dict]:
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
    return TOOLS, handlers
