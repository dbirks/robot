"""Motion tools for Reachy Mini robot control."""

import asyncio
import math
import time
from typing import Optional

import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES


class Robot:
    """Lightweight robot wrapper for reusing a single connection."""

    def __init__(self, host: Optional[str] = None):
        """Initialize robot wrapper (connection created later via init()).

        Args:
            host: Robot host URL. None for localhost daemon (default).
                  For wireless: "http://<pi-ip>:8000"
        """
        self._host = host
        self._rm: ReachyMini | None = None
        self._emotions: RecordedMoves | None = None

    def init(self):
        """Initialize robot connection and wake up robot.

        Should be called ONCE at startup before any movements.
        """
        if self._rm is not None:
            return  # Already initialized

        # Use default_no_video to skip camera in sim mode (avoids camera errors)
        self._rm = (
            ReachyMini(host=self._host, media_backend="default_no_video")
            if self._host
            else ReachyMini(media_backend="default_no_video")
        )
        # Wake up robot ONCE - enables motors and sets initial position
        try:
            self._rm.wake_up()
        except Exception as e:
            # Audio device may not be available in headless/remote setups
            print(f"⚠️  Wake-up sound failed (device unavailable): {e}")
            print("🤖 Robot initialized (silent wake)")
        else:
            print("🤖 Robot initialized and awake")

        # Load emotions library from HuggingFace
        try:
            print("📦 Loading emotions library from HuggingFace...")
            self._emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
            print(f"😊 Loaded {len(self._emotions.list_moves())} emotions")
        except Exception as e:
            print(f"⚠️  Failed to load emotions library: {e}")
            self._emotions = None

    def cleanup(self):
        """Clean up robot connection."""
        if self._rm:
            try:
                self._rm.__exit__(None, None, None)
                print("🤖 Robot connection closed")
            except Exception as e:
                print(f"⚠️  Error closing robot: {e}")
            finally:
                self._rm = None

    def __enter__(self):
        """Enter context manager (for backward compatibility, but not recommended)."""
        if self._rm is None:
            self.init()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context manager (for backward compatibility)."""
        # Don't cleanup here - connection should persist across movements
        pass

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
        - Subtle: 20° (0.35 rad)
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
            self._rm.goto_target(antennas=[-amp, amp], duration=0.15)
            await asyncio.sleep(0.15)
            phase += 0.8

        # Return antennas to neutral
        self._rm.goto_target(antennas=[0.0, 0.0], duration=0.2)

    def _run_dance_sync(
        self,
        move_name: str,
        duration_seconds: float,
        bpm: float = 120.0,
        amplitude_scale: float = 1.0,
        neutral_pos: np.ndarray = None,
        neutral_eul: np.ndarray = None,
    ):
        """Execute a dance move from the library (synchronous blocking version).

        This uses blocking time.sleep() to match the official demo pattern for
        precise 100Hz control timing. Run via _run_dance async wrapper.

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

        move_fn, base_params, _ = AVAILABLE_MOVES[move_name]

        # Scale amplitudes
        params = base_params.copy()
        for key in params:
            if "amplitude" in key or "_amp" in key:
                params[key] *= amplitude_scale

        # Time loop - BLOCKING (matches official demo pattern)
        control_ts = 0.01  # 100 Hz control loop
        t_beats = 0.0
        end_time = time.time() + duration_seconds

        while time.time() < end_time:
            loop_start = time.time()

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

            # BLOCKING sleep to maintain precise 100Hz control loop rate
            elapsed = time.time() - loop_start
            time.sleep(max(0, control_ts - elapsed))

        # Return to neutral
        pose = create_head_pose(*neutral_pos, *neutral_eul, degrees=False)
        self._rm.set_target(pose, antennas=np.zeros(2))

    def _play_move_blocking_100hz(self, move, with_sound: bool = True):
        """Universal 100Hz blocking playback for any Move (RecordedMove or DanceMove).

        Uses precise time.sleep() for smooth motion, matching the dance timing.
        This method is intended to be run in an executor via run_in_executor().

        Args:
            move: A Move object (RecordedMove from emotions or DanceMove)
            with_sound: Whether to play accompanying audio
        """
        control_ts = 0.01  # 100 Hz control loop (10ms per frame)

        # Play sound if available (emotions have sound_path, dances don't)
        if hasattr(move, 'sound_path') and move.sound_path and with_sound:
            try:
                self._rm.media_manager.play_sound(str(move.sound_path))
            except Exception:
                pass  # Ignore audio errors (device issues are common)

        t0 = time.time()
        while time.time() - t0 < move.duration:
            loop_start = time.time()
            t = min(time.time() - t0, move.duration - 1e-2)

            # Evaluate move at current time
            head, antennas, body_yaw = move.evaluate(t)

            # Send commands to robot
            if head is not None:
                self._rm.set_target_head_pose(head)
            if body_yaw is not None:
                self._rm.set_target_body_yaw(body_yaw)
            if antennas is not None:
                self._rm.set_target_antenna_joint_positions(list(antennas))

            # Precise sleep to maintain 100Hz
            elapsed = time.time() - loop_start
            time.sleep(max(0, control_ts - elapsed))

    async def _run_dance(
        self,
        move_name: str,
        duration_seconds: float,
        bpm: float = 120.0,
        amplitude_scale: float = 1.0,
        neutral_pos: np.ndarray = None,
        neutral_eul: np.ndarray = None,
    ):
        """Execute a dance move from the library (async wrapper).

        Runs the synchronous blocking dance loop in a thread pool executor
        to avoid blocking the async event loop while maintaining precise timing.

        Args:
            move_name: Name of the dance move from AVAILABLE_MOVES
            duration_seconds: How long to perform the dance
            bpm: Beats per minute (tempo)
            amplitude_scale: Scale factor for all amplitudes (0.5 = half intensity)
            neutral_pos: Neutral position offset [x, y, z] in meters
            neutral_eul: Neutral orientation [roll, pitch, yaw] in radians
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._run_dance_sync,
            move_name,
            duration_seconds,
            bpm,
            amplitude_scale,
            neutral_pos,
            neutral_eul,
        )

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

    async def play_emotion(self, emotion_name: str, with_sound: bool = True):
        """Play an emotion from the emotions library with smooth 100Hz timing.

        Uses blocking loop in executor (same as dances) for precise motion control.

        Args:
            emotion_name: Name of the emotion (e.g., "laughing1", "surprised1")
            with_sound: Whether to play the audio that accompanies the emotion

        Raises:
            ValueError: If emotion not found or emotions library not loaded
        """
        if self._emotions is None:
            raise ValueError("Emotions library not loaded")

        try:
            move = self._emotions.get(emotion_name)

            # Use blocking 100Hz loop in executor for smooth motion
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._play_move_blocking_100hz,
                move,
                with_sound,
            )
        except ValueError as e:
            raise ValueError(f"Emotion '{emotion_name}' not found") from e
