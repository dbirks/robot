"""Camera frames for the vision tools, without fighting the daemon.

The daemon can own the camera (its GStreamer media server). Two processes on
one UVC device is the contention class ADR 0002 exists to prevent, so this
NEVER opens the device while the daemon holds it: before every grab it asks
GET /api/media/status and opens V4L2 directly only when the daemon reports
its media pipeline unavailable (no media server, or released via the
no_media connection in shell/robot.py - the SDK's documented "direct access"
contract). The device is opened per grab and released immediately: no
persistent handle, no background capture thread (ADR 0006).

REACHY_CAMERA=off disables capture entirely.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("shell.camera")

WARMUP_FRAMES = 5  # UVC auto-exposure needs a few frames before one is usable


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
        self._lock = threading.Lock()  # one grab at a time

    def _media_status(self) -> dict | None:
        try:
            import requests

            r = requests.get(f"{self.daemon_url}/api/media/status", timeout=0.5)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def grab(self):
        """BGR frame (numpy) or raises RuntimeError with a speakable reason."""
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
