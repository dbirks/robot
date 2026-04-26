import collections
import logging
import math
import threading
import time
from dataclasses import dataclass, field

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


def _smooth_step(t: float) -> float:
    """Hermite smooth step: t*t*(3 - 2*t), clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _minimum_jerk(t: float) -> float:
    """Minimum-jerk interpolation parameter for t in [0, 1].

    Returns s = 10*t^3 - 15*t^4 + 6*t^5, which has zero velocity
    and acceleration at both endpoints.
    """
    t = max(0.0, min(1.0, t))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def _lerp_pose(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two 4x4 matrices (good enough at 60 Hz)."""
    return (1.0 - t) * a + t * b


@dataclass
class AnimationKeyframe:
    """A single keyframe in an animation sequence."""

    pose: np.ndarray  # 4x4 head matrix
    antennas: list[float] | None = None  # optional antenna positions
    duration: float = 0.3  # seconds to reach this keyframe


@dataclass
class QueuedAnimation:
    """An animation ready to be played through the MovementManager."""

    keyframes: list[AnimationKeyframe] = field(default_factory=list)
    priority: int = 0  # higher = more important, can preempt lower
    blend_in: float = 0.2  # seconds to ramp weight 0 -> 1
    blend_out: float = 0.2  # seconds to ramp weight 1 -> 0


class MovementManager:
    """60Hz control loop composing primary head pose with secondary offsets.

    Primary pose: face tracking target (smoothed via EMA), decays to center.
    Animations: queued keyframe animations that blend with the tracking pose.
    Secondary offsets: speech wobble + thinking animation (always layered on top).
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

        # --- Animation system ---
        self._animation_queue: collections.deque[QueuedAnimation] = collections.deque()
        self._active_animation: QueuedAnimation | None = None
        self._animation_time: float = 0.0  # time into current animation
        self._animation_weight: float = 0.0  # current blend weight (0-1)
        self._animation_pose: np.ndarray = self._center.copy()  # current interpolated pose
        self._animation_antennas: list[float] | None = None  # current interpolated antennas
        self._animation_keyframe_idx: int = 0  # current keyframe index
        self._animation_keyframe_time: float = 0.0  # time into current keyframe segment
        self._animation_phase: str = "idle"  # idle, blend_in, playing, blend_out
        self._animation_start_pose: np.ndarray = self._center.copy()  # pose when animation started (for blend_in)
        self._animation_start_antennas: list[float] | None = None

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

    # --- Animation API ---

    def queue_animation(
        self,
        keyframes: list[AnimationKeyframe],
        priority: int = 0,
        blend_in: float = 0.2,
        blend_out: float = 0.2,
        preempt: bool = False,
    ):
        """Add an animation to the queue.

        If preempt=True and priority >= active animation's priority,
        immediately start blending to this animation.
        """
        anim = QueuedAnimation(
            keyframes=keyframes,
            priority=priority,
            blend_in=blend_in,
            blend_out=blend_out,
        )
        with self._lock:
            if preempt and self._active_animation is not None and priority >= self._active_animation.priority:
                # Preempt: start blend_out on current, put new one at front of queue
                self._animation_phase = "blend_out"
                self._animation_time = 0.0
                self._animation_queue.appendleft(anim)
                log.info("Animation preempted (priority %d >= %d)", priority, self._active_animation.priority)
            elif self._active_animation is None and self._animation_phase == "idle":
                # No active animation, start immediately
                self._animation_queue.append(anim)
            else:
                self._animation_queue.append(anim)
            self._last_activity = time.monotonic()

    def cancel_animation(self):
        """Smoothly blend out the current animation and clear the queue."""
        with self._lock:
            self._animation_queue.clear()
            if self._active_animation is not None and self._animation_phase in ("blend_in", "playing"):
                self._animation_phase = "blend_out"
                self._animation_time = 0.0
                log.info("Animation cancelled, blending out")

    def _start_next_animation(self, current_pose: np.ndarray, current_antennas: list[float]):
        """Pop the next animation from the queue and start it.

        Must be called with self._lock held.
        """
        if not self._animation_queue:
            self._active_animation = None
            self._animation_phase = "idle"
            self._animation_weight = 0.0
            return

        self._active_animation = self._animation_queue.popleft()
        self._animation_phase = "blend_in"
        self._animation_time = 0.0
        self._animation_keyframe_idx = 0
        self._animation_keyframe_time = 0.0
        self._animation_start_pose = current_pose.copy()
        self._animation_start_antennas = list(current_antennas)
        # Initialize animation pose to the start pose
        self._animation_pose = current_pose.copy()
        self._animation_antennas = None
        self._animation_weight = 0.0
        log.info(
            "Starting animation with %d keyframes (priority=%d, blend_in=%.2f, blend_out=%.2f)",
            len(self._active_animation.keyframes),
            self._active_animation.priority,
            self._active_animation.blend_in,
            self._active_animation.blend_out,
        )

    def _step_animation(self, dt: float, tracking_pose: np.ndarray, current_antennas: list[float]):
        """Advance the animation state machine by dt seconds.

        Must be called with self._lock held.
        Returns (animation_weight, animation_pose, animation_antennas).
        """
        if self._animation_phase == "idle":
            # Check if there's something in the queue to start
            if self._animation_queue:
                self._start_next_animation(tracking_pose, current_antennas)
            else:
                return 0.0, tracking_pose, None

        anim = self._active_animation
        if anim is None:
            return 0.0, tracking_pose, None

        if self._animation_phase == "blend_in":
            self._animation_time += dt
            if anim.blend_in > 0:
                raw_t = self._animation_time / anim.blend_in
                self._animation_weight = _smooth_step(raw_t)
            else:
                self._animation_weight = 1.0

            # During blend_in, also advance keyframe interpolation
            self._advance_keyframes(dt, anim)

            if self._animation_time >= anim.blend_in:
                self._animation_phase = "playing"
                self._animation_time = 0.0
                self._animation_weight = 1.0

        elif self._animation_phase == "playing":
            self._animation_weight = 1.0
            finished = self._advance_keyframes(dt, anim)
            if finished:
                self._animation_phase = "blend_out"
                self._animation_time = 0.0

        elif self._animation_phase == "blend_out":
            self._animation_time += dt
            if anim.blend_out > 0:
                raw_t = self._animation_time / anim.blend_out
                self._animation_weight = 1.0 - _smooth_step(raw_t)
            else:
                self._animation_weight = 0.0

            if self._animation_time >= anim.blend_out:
                self._animation_weight = 0.0
                self._active_animation = None
                self._animation_phase = "idle"
                # Try to start next animation immediately
                if self._animation_queue:
                    self._start_next_animation(tracking_pose, current_antennas)

        return self._animation_weight, self._animation_pose, self._animation_antennas

    def _advance_keyframes(self, dt: float, anim: QueuedAnimation) -> bool:
        """Advance keyframe interpolation by dt seconds.

        Returns True if the animation has finished its last keyframe.
        Must be called with self._lock held.
        """
        if not anim.keyframes:
            return True

        self._animation_keyframe_time += dt
        kf_idx = self._animation_keyframe_idx
        kf = anim.keyframes[kf_idx]

        # Determine the start pose for this keyframe segment
        if kf_idx == 0:
            start_pose = self._animation_start_pose
            start_antennas = self._animation_start_antennas
        else:
            prev_kf = anim.keyframes[kf_idx - 1]
            start_pose = prev_kf.pose
            start_antennas = prev_kf.antennas

        # Interpolation progress within this keyframe
        if kf.duration > 0:
            t = self._animation_keyframe_time / kf.duration
        else:
            t = 1.0
        s = _minimum_jerk(min(t, 1.0))

        # Interpolate pose
        self._animation_pose = _lerp_pose(start_pose, kf.pose, s)

        # Interpolate antennas if the keyframe specifies them
        if kf.antennas is not None:
            if start_antennas is not None:
                self._animation_antennas = [
                    (1.0 - s) * start_antennas[i] + s * kf.antennas[i] for i in range(len(kf.antennas))
                ]
            else:
                self._animation_antennas = [kf.antennas[i] * s for i in range(len(kf.antennas))]
        elif start_antennas is not None and kf_idx == 0:
            # First keyframe has no antennas but we have a start — keep start
            self._animation_antennas = start_antennas
        else:
            self._animation_antennas = None

        # Check if we should advance to the next keyframe
        if t >= 1.0:
            if kf_idx < len(anim.keyframes) - 1:
                self._animation_keyframe_idx += 1
                self._animation_keyframe_time -= kf.duration
                return False
            else:
                return True  # finished last keyframe

        return False

    def _run_loop(self):
        smooth_primary = self._center.copy()
        current_antennas = [0.0, 0.0]
        interval = 1.0 / CONTROL_HZ
        last_tick = time.monotonic()

        while not self._stop.is_set():
            tick_start = time.monotonic()
            dt = tick_start - last_tick
            last_tick = tick_start

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

            # --- Animation system ---
            with self._lock:
                anim_weight, anim_pose, anim_antennas = self._step_animation(dt, smooth_primary, current_antennas)

            # Blend between tracking pose and animation pose
            if anim_weight > 0.0:
                primary_pose = _lerp_pose(smooth_primary, anim_pose, anim_weight)
            else:
                primary_pose = smooth_primary

            # --- Secondary offsets (ALWAYS layer on top) ---
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
                final_head = compose_world_offset(primary_pose, offset_pose)
            else:
                final_head = primary_pose

            # --- Antennas ---
            idle_duration = now - last_activity
            is_idle = (
                not face_active
                and not processing
                and idle_duration > BREATHING_DELAY
                and np.allclose(speech_offsets, 0)
                and anim_weight == 0.0
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

            # Blend antenna positions from animation
            if anim_weight > 0.0 and anim_antennas is not None:
                target_antennas = [
                    (1.0 - anim_weight) * target_antennas[i] + anim_weight * anim_antennas[i] for i in range(2)
                ]

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
