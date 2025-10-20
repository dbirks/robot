"""Motion tools for Reachy Mini robot control."""

import asyncio
import math
from typing import Optional
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


class Robot:
    """Lightweight robot wrapper for reusing a single connection."""

    def __init__(self, host: Optional[str] = None):
        """Initialize robot connection.

        Args:
            host: Robot host URL. None for localhost daemon (default).
                  For wireless: "http://<pi-ip>:8000"
        """
        self._host = host
        self._rm: ReachyMini | None = None

    def __enter__(self):
        """Enter context manager, establish connection."""
        # Use default_no_video to skip camera in sim mode (avoids camera errors)
        self._rm = ReachyMini(host=self._host, media_backend="default_no_video") if self._host else ReachyMini(media_backend="default_no_video")
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context manager, close connection."""
        if self._rm:
            self._rm.__exit__(exc_type, exc, tb)

    def _goto_head(self, yaw=0, pitch=0, roll=0, z=0, duration=0.25):
        """Low-level helper to move head to specified pose.

        Args:
            yaw: Head rotation left/right in degrees
            pitch: Head rotation up/down in degrees
            roll: Head tilt in degrees
            z: Height adjustment in mm
            duration: Movement duration in seconds
        """
        pose = create_head_pose(y=yaw, x=pitch, roll=roll, z=z, degrees=True, mm=True)
        self._rm.goto_target(head=pose, duration=duration)

    async def nod(self, times: int = 1):
        """Nod head up/down a specified number of times.

        Args:
            times: Number of nods (clamped to 1-5)
        """
        times = max(1, min(5, int(times)))
        for _ in range(times):
            self._goto_head(pitch=10)
            await asyncio.sleep(0.35)
            self._goto_head(pitch=-10)
            await asyncio.sleep(0.35)
        self._goto_head()  # Return to neutral

    async def shake(self, times: int = 1):
        """Shake head left/right a specified number of times.

        Args:
            times: Number of shakes (clamped to 1-5)
        """
        times = max(1, min(5, int(times)))
        for _ in range(times):
            self._goto_head(yaw=15)
            await asyncio.sleep(0.35)
            self._goto_head(yaw=-15)
            await asyncio.sleep(0.35)
        self._goto_head()  # Return to neutral

    async def look_at(self, x_deg: float, y_deg: float):
        """Look at absolute angles in degrees.

        Args:
            x_deg: Yaw angle (+right/-left), clamped to ±35
            y_deg: Pitch angle (+up/-down), clamped to ±20
        """
        yaw = max(-35.0, min(35.0, float(x_deg)))
        pitch = max(-20.0, min(20.0, float(y_deg)))
        self._goto_head(yaw=yaw, pitch=pitch)

    async def antenna_wiggle(self, seconds: int = 2):
        """Wiggle antennas for specified duration.

        Args:
            seconds: Duration in seconds (clamped to 1-10)
        """
        secs = max(1, min(10, int(seconds)))
        end = asyncio.get_event_loop().time() + secs
        phase = 0.0

        while asyncio.get_event_loop().time() < end:
            amp = 10.0 * math.sin(phase)
            # Antenna control via goto_target - format: [right, left] in radians
            self._rm.goto_target(
                antennas=[-amp, amp],
                duration=0.15
            )
            await asyncio.sleep(0.15)
            phase += 0.8

        # Return antennas to neutral
        self._rm.goto_target(antennas=[0.0, 0.0], duration=0.2)
