"""Direction of Arrival (DOA) tracker using the Reachy Mini's microphone array.

Polls the daemon's DOA endpoint and feeds head yaw targets to the
MovementManager when no face is being tracked.
"""

import logging
import math
import threading
import time

import requests
from reachy_mini.utils import create_head_pose

log = logging.getLogger(__name__)

DOA_POLL_HZ = 5
DOA_DAEMON_URL = "http://localhost:8000/api/state/doa"
DOA_TIMEOUT = 0.1
DOA_HEAD_YAW_MAX = 35
DOA_BODY_YAW_MAX = math.radians(30)


def doa_to_yaw(doa_angle_rad: float) -> float:
    """Convert DOA angle (0=left, pi/2=front, pi=right) to yaw in degrees.

    Head yaw: positive=left, negative=right, 0=center.
    """
    offset_rad = math.pi / 2 - doa_angle_rad
    return max(-DOA_HEAD_YAW_MAX, min(DOA_HEAD_YAW_MAX, math.degrees(offset_rad)))


class DoATracker:
    """Polls DOA from the daemon and sets head/body targets on the MovementManager."""

    def __init__(self, movement_manager, robot_mini=None):
        self._movement = movement_manager
        self._robot = robot_mini
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="doa-tracker", daemon=True)
        self._thread.start()
        log.info("DoATracker started at %d Hz", DOA_POLL_HZ)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("DoATracker stopped")

    def _poll_loop(self):
        interval = 1.0 / DOA_POLL_HZ
        while not self._stop.is_set():
            try:
                resp = requests.get(DOA_DAEMON_URL, timeout=DOA_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    if data and data.get("speech_detected"):
                        angle_rad = data["angle"]
                        yaw_deg = doa_to_yaw(angle_rad)
                        pose = create_head_pose(yaw=yaw_deg, degrees=True)
                        self._movement.set_doa_target(pose)
            except requests.RequestException:
                pass
            except Exception:
                log.exception("DOA poll error")

            self._stop.wait(interval)
