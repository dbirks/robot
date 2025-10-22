"""Motion tools for Reachy Mini robot control."""

import asyncio
import math
import numpy as np
from typing import Optional
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES


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

        For subtle/cute wiggles, use smaller amplitudes:
        - Subtle: 20° (0.35 rad) - like yeah_nod dance
        - Moderate: 40° (0.70 rad) - like headbanger_combo
        - Default: 10° amplitude with sine wave

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

    async def _run_dance(
        self,
        move_name: str,
        duration_seconds: float,
        bpm: float = 120.0,
        amplitude_scale: float = 1.0,
        neutral_pos: np.ndarray = None,
        neutral_eul: np.ndarray = None,
    ):
        """Execute a dance move from the library.

        Args:
            move_name: Name of the dance move from AVAILABLE_MOVES
            duration_seconds: How long to perform the dance
            bpm: Beats per minute (tempo)
            amplitude_scale: Scale factor for all amplitudes (0.5 = half intensity)
            neutral_pos: Neutral position offset [x, y, z] in meters
            neutral_eul: Neutral orientation [roll, pitch, yaw] in radians
        """
        if neutral_pos is None:
            neutral_pos = np.array([0.0, 0.0, 0.0])
        if neutral_eul is None:
            neutral_eul = np.zeros(3)

        # Wake up robot for set_target to work properly
        self._rm.wake_up()

        move_fn, base_params, _ = AVAILABLE_MOVES[move_name]

        # Scale amplitudes
        params = base_params.copy()
        for key in params:
            if "amplitude" in key or "_amp" in key:
                params[key] *= amplitude_scale

        # Time loop
        control_ts = 0.01  # 100 Hz control loop
        t_beats = 0.0
        end_time = asyncio.get_event_loop().time() + duration_seconds

        while asyncio.get_event_loop().time() < end_time:
            loop_start = asyncio.get_event_loop().time()

            # Calculate offsets from dance move
            offsets = move_fn(t_beats, **params)

            # Apply offsets to neutral pose
            final_pos = neutral_pos + offsets.position_offset
            final_eul = neutral_eul + offsets.orientation_offset
            final_ant = offsets.antennas_offset

            # Send to robot (match dance_demo.py exactly)
            pose = create_head_pose(*final_pos, *final_eul, degrees=False)
            self._rm.set_target(pose, antennas=final_ant)

            # Update time in beats
            beats_per_second = bpm / 60.0
            t_beats += control_ts * beats_per_second

            # Sleep to maintain control loop rate
            elapsed = asyncio.get_event_loop().time() - loop_start
            await asyncio.sleep(max(0, control_ts - elapsed))

        # Return to neutral
        pose = create_head_pose(*neutral_pos, *neutral_eul, degrees=False)
        self._rm.set_target(pose, antennas=np.zeros(2))

    async def yeah_nod(self, duration: int = 4, bpm: int = 120):
        """Perform an enthusiastic 'yeah' nod with subtle antenna wiggle.

        This is an emphatic two-part nod gesture with both antennas moving
        together in a cute, subtle 20° wiggle.

        Args:
            duration: Duration in seconds (clamped to 2-10)
            bpm: Beats per minute / tempo (clamped to 60-180)
        """
        duration = max(2, min(10, int(duration)))
        bpm = max(60, min(180, int(bpm)))
        await self._run_dance("yeah_nod", duration, bpm=bpm)

    async def headbanger_combo(self, duration: int = 4, bpm: int = 120, intensity: float = 1.0):
        """Perform high-energy headbanging with vertical bounce.

        Combines a strong 30° pitch nod with vertical body bounce and
        synchronized antenna movement (40° amplitude).

        Args:
            duration: Duration in seconds (clamped to 2-10)
            bpm: Beats per minute / tempo (clamped to 60-180)
            intensity: Movement intensity multiplier (clamped to 0.3-2.0)
        """
        duration = max(2, min(10, int(duration)))
        bpm = max(60, min(180, int(bpm)))
        intensity = max(0.3, min(2.0, float(intensity)))
        await self._run_dance("headbanger_combo", duration, bpm=bpm, amplitude_scale=intensity)

    async def dizzy_spin(self, duration: int = 6, bpm: int = 100):
        """Perform a circular, dizzying head motion.

        Creates a slow, circular head motion by combining roll and pitch
        movements with opposing antenna wiggle (45° amplitude).

        Args:
            duration: Duration in seconds (clamped to 3-15)
            bpm: Beats per minute / tempo (clamped to 40-140)
        """
        duration = max(3, min(15, int(duration)))
        bpm = max(40, min(140, int(bpm)))
        await self._run_dance("dizzy_spin", duration, bpm=bpm)
