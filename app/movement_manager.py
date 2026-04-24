import logging
import math
import threading
import time

import numpy as np
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset

log = logging.getLogger(__name__)

CONTROL_HZ = 60

# Face tracking smoothing (moved from face_tracker.py)
FACE_EMA_ALPHA = 0.25
FACE_DECAY_ALPHA = 0.03
FACE_LOST_TIMEOUT = 2.0

# Breathing (idle)
BREATHING_DELAY = 0.3
BREATHING_BLEND_DURATION = 1.0
BREATHING_Z_AMP = 0.005  # 5mm
BREATHING_Z_FREQ = 0.1
BREATHING_ANTENNA_AMP = math.radians(15)
BREATHING_ANTENNA_FREQ = 0.5

# Thinking (during LLM inference)
THINKING_RAMP_DURATION = 0.5
THINKING_YAW_AMP = math.radians(12)
THINKING_YAW_FREQ = 0.15
THINKING_PITCH_BASE = math.radians(6)
THINKING_PITCH_AMP = math.radians(3)
THINKING_PITCH_FREQ = 0.2
THINKING_Z_AMP = 0.003
THINKING_Z_FREQ = 0.12
THINKING_ANTENNA_AMP = math.radians(20)
THINKING_ANTENNA_FREQ = 0.4
THINKING_ANTENNA_PHASE = 1.2

# Antenna unfreeze
ANTENNA_UNFREEZE_DURATION = 0.4


class MovementManager:
    """60Hz control loop composing primary head pose with secondary offsets.

    Primary pose: face tracking target (smoothed via EMA), decays to center.
    Secondary offsets: speech wobble + thinking animation.
    Antennas: breathing sway, freeze during listening, thinking scan.
    """

    def __init__(self, robot_mini):
        self._robot = robot_mini
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._center = create_head_pose()

        # Face tracking state
        self._face_target: np.ndarray | None = None
        self._face_last_seen = 0.0

        # DOA (direction of arrival) state — lower priority than face
        self._doa_target: np.ndarray | None = None
        self._doa_last_seen = 0.0

        # Secondary offsets (x, y, z, roll, pitch, yaw) in meters/radians
        self._speech_offsets = np.zeros(6)

        # State flags
        self._listening = False
        self._processing = False
        self._processing_start = 0.0
        self._processing_amplitude = 0.0

        # Antenna freeze
        self._frozen_antennas: tuple[float, float] | None = None
        self._unfreeze_start: float | None = None
        self._unfreeze_from: tuple[float, float] = (0.0, 0.0)

        # Activity tracking for breathing
        self._last_activity = time.monotonic()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="movement-manager", daemon=True)
        self._thread.start()
        log.info("MovementManager started at %d Hz", CONTROL_HZ)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("MovementManager stopped")

    def set_face_target(self, pose: np.ndarray):
        with self._lock:
            self._face_target = pose
            self._face_last_seen = time.monotonic()
            self._last_activity = time.monotonic()

    def clear_face_target(self):
        with self._lock:
            self._face_target = None

    def set_doa_target(self, pose: np.ndarray):
        with self._lock:
            self._doa_target = pose
            self._doa_last_seen = time.monotonic()
            self._last_activity = time.monotonic()

    def set_speech_offsets(self, offsets: tuple[float, ...]):
        with self._lock:
            self._speech_offsets = np.array(offsets)
            self._last_activity = time.monotonic()

    def set_listening(self, listening: bool):
        with self._lock:
            self._listening = listening
            if listening:
                self._last_activity = time.monotonic()

    def set_processing(self, processing: bool):
        with self._lock:
            if processing and not self._processing:
                self._processing_start = time.monotonic()
            self._processing = processing
            self._last_activity = time.monotonic()

    def _run_loop(self):
        smooth_primary = self._center.copy()
        current_antennas = [0.0, 0.0]
        interval = 1.0 / CONTROL_HZ

        while not self._stop.is_set():
            tick_start = time.monotonic()

            with self._lock:
                face_target = self._face_target.copy() if self._face_target is not None else None
                face_last_seen = self._face_last_seen
                doa_target = self._doa_target.copy() if self._doa_target is not None else None
                doa_last_seen = self._doa_last_seen
                speech_offsets = self._speech_offsets.copy()
                listening = self._listening
                processing = self._processing
                processing_start = self._processing_start
                last_activity = self._last_activity

            now = time.monotonic()

            # --- Primary pose: face tracking > DOA > decay to center ---
            face_active = face_target is not None and (now - face_last_seen) < FACE_LOST_TIMEOUT
            doa_active = not face_active and doa_target is not None and (now - doa_last_seen) < FACE_LOST_TIMEOUT
            if face_active:
                smooth_primary = FACE_EMA_ALPHA * face_target + (1 - FACE_EMA_ALPHA) * smooth_primary
            elif doa_active:
                smooth_primary = FACE_DECAY_ALPHA * doa_target + (1 - FACE_DECAY_ALPHA) * smooth_primary
            else:
                smooth_primary = FACE_DECAY_ALPHA * self._center + (1 - FACE_DECAY_ALPHA) * smooth_primary

            # --- Secondary offsets ---
            total_offsets = np.zeros(6)

            # Speech wobble
            total_offsets += speech_offsets

            # Thinking animation
            if processing:
                elapsed = now - processing_start
                ramp = min(1.0, elapsed / THINKING_RAMP_DURATION)
                self._processing_amplitude = ramp
            elif self._processing_amplitude > 0:
                self._processing_amplitude = max(0.0, self._processing_amplitude - 2.0 / CONTROL_HZ)

            if self._processing_amplitude > 0:
                t = now
                amp = self._processing_amplitude
                total_offsets[2] += amp * THINKING_Z_AMP * math.sin(2 * math.pi * THINKING_Z_FREQ * t)
                total_offsets[4] += amp * (
                    THINKING_PITCH_BASE + THINKING_PITCH_AMP * math.sin(2 * math.pi * THINKING_PITCH_FREQ * t)
                )
                total_offsets[5] += amp * THINKING_YAW_AMP * math.sin(2 * math.pi * THINKING_YAW_FREQ * t)

            # --- Compose final head pose ---
            if np.any(total_offsets != 0):
                offset_pose = create_head_pose(
                    x=total_offsets[0],
                    y=total_offsets[1],
                    z=total_offsets[2],
                    roll=math.degrees(total_offsets[3]),
                    pitch=math.degrees(total_offsets[4]),
                    yaw=math.degrees(total_offsets[5]),
                    degrees=True,
                )
                final_head = compose_world_offset(smooth_primary, offset_pose)
            else:
                final_head = smooth_primary

            # --- Antennas ---
            idle_duration = now - last_activity
            is_idle = (
                not face_active
                and not processing
                and idle_duration > BREATHING_DELAY
                and np.allclose(speech_offsets, 0)
            )

            target_antennas = [0.0, 0.0]

            if self._processing_amplitude > 0:
                amp = self._processing_amplitude
                t = now
                target_antennas[0] = amp * THINKING_ANTENNA_AMP * math.sin(2 * math.pi * THINKING_ANTENNA_FREQ * t)
                target_antennas[1] = (
                    amp
                    * THINKING_ANTENNA_AMP
                    * math.sin(2 * math.pi * THINKING_ANTENNA_FREQ * t + THINKING_ANTENNA_PHASE)
                )
            elif is_idle:
                breathing_t = now
                sway = BREATHING_ANTENNA_AMP * math.sin(2 * math.pi * BREATHING_ANTENNA_FREQ * breathing_t)
                target_antennas = [sway, -sway]

                # Breathing Z bob on primary
                z_bob = BREATHING_Z_AMP * math.sin(2 * math.pi * BREATHING_Z_FREQ * breathing_t)
                breathing_offset = create_head_pose(z=z_bob, degrees=True)
                final_head = compose_world_offset(final_head, breathing_offset)

            # Antenna freeze/unfreeze during listening
            with self._lock:
                if listening:
                    if self._frozen_antennas is None:
                        self._frozen_antennas = tuple(current_antennas)
                    target_antennas = list(self._frozen_antennas)
                elif self._frozen_antennas is not None:
                    if self._unfreeze_start is None:
                        self._unfreeze_start = now
                        self._unfreeze_from = self._frozen_antennas
                    blend = min(1.0, (now - self._unfreeze_start) / ANTENNA_UNFREEZE_DURATION)
                    target_antennas = [
                        self._unfreeze_from[i] * (1 - blend) + target_antennas[i] * blend for i in range(2)
                    ]
                    if blend >= 1.0:
                        self._frozen_antennas = None
                        self._unfreeze_start = None

            # Smooth antenna movement
            antenna_alpha = 0.3
            current_antennas = [
                antenna_alpha * target_antennas[i] + (1 - antenna_alpha) * current_antennas[i] for i in range(2)
            ]

            # --- Issue command ---
            self._robot.set_target(head=final_head, antennas=current_antennas)

            elapsed = time.monotonic() - tick_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        log.info("MovementManager loop stopped")
