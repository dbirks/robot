"""The single motion owner: the ONLY caller of set_target in the shell.

Same rule as the audio owners (ADR 0002), applied to the head: one thread,
one control loop, everything else communicates by setting intent. The legacy
app had a 30 Hz MovementManager but peekaboo/sleep called goto_target
directly and fought it; here scripted SDK moves run via `exclusive()` ON the
motion thread, so the loop is paused by construction while they play and
resumes with a blend from wherever the robot actually ended up.

Composition per tick (all poses are [x, y, z, roll, pitch, yaw] 6-vectors):

    base gaze     yaw/pitch target (tool look, wake turn, DOA) through a
                  critically damped spring; idle gaze aversion is added to
                  the spring TARGET so it is smoothed too
    animation     keyframe override (nod, shake, emotions) blended in/out
                  over the base; `relative` animations ride on the gaze
    additive      thinking look-away, speech sway, idle breathing (z bob)
    antennas      breathing sway when idle, thinking sway, animation
                  antennas; frozen while the user is talking
    resume        min-jerk blend from the measured pose after an exclusive
                  move, so control never jumps

`tick()` is pure given its inputs (clock injected), so all of this is unit
tested without hardware; `_run()` just calls it at `hz` and sends.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .. import journal as J
from .kinematics import exp_alpha, minimum_jerk, pose6, pose6_from_matrix, pose_matrix, smooth_step, spring_update

log = logging.getLogger("shell.motion")

CONTROL_HZ = 50

# Base gaze
GAZE_HALFLIFE = 0.35  # toward a person / tool target
HOME_HALFLIFE = 0.8  # drifting back to centre
MAX_YAW_DEG = 60.0  # head stays within the 65 deg head/body relative limit
MAX_PITCH_DEG = 25.0
PERSON_SOURCES = {"wake", "doa"}

# Idle gaze aversion while attending a person (prevents staring)
GAZE_HOLD_S = (3.0, 6.0)
GAZE_AWAY_S = (0.5, 1.5)
GAZE_AWAY_YAW = math.radians(12)
GAZE_AWAY_PITCH = math.radians(5)

# Breathing (idle)
BREATHING_DELAY = 0.3
BREATHING_BLEND_S = 1.0
BREATHING_Z_AMP = 0.005  # 5 mm
BREATHING_Z_FREQ = 0.1
BREATHING_ANTENNA_AMP = math.radians(15)
BREATHING_ANTENNA_FREQ = 0.5

# Thinking look-away (between the user's transcript and Reachy's first audio)
THINKING_AVERT_YAW = math.radians(10)
THINKING_AVERT_PITCH = math.radians(3)
THINKING_AVERT_S = 0.5
THINKING_RETURN_S = 0.3
THINKING_MICRO_YAW_AMP = math.radians(1.5)
THINKING_MICRO_YAW_FREQ = 0.08
THINKING_MICRO_PITCH_AMP = math.radians(0.8)
THINKING_MICRO_PITCH_FREQ = 0.12
THINKING_Z_AMP = 0.002
THINKING_Z_FREQ = 0.10
THINKING_RAMP_S = 1.0
THINKING_ANTENNA_AMP = math.radians(15)
THINKING_ANTENNA_FREQ = 0.35
THINKING_ANTENNA_PHASE = 1.2

# Antennas / body
ANTENNA_TAU = 0.2  # ~= legacy alpha 0.15 at 30 Hz
ANTENNA_UNFREEZE_S = 0.4
BODY_TAU = 0.3
RESUME_S = 0.8

# Scripted poses (from app/robot_tools.py, measured on the robot)
SLEEP_HEAD_POSE = np.array(
    [
        [0.911, 0.004, 0.413, -0.021],
        [-0.004, 1.0, -0.001, 0.001],
        [-0.413, -0.001, 0.911, -0.044],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
SLEEP_ANTENNAS = [-3.05, 3.05]
INIT_ANTENNAS = [-0.1745, 0.1745]


@dataclass
class Keyframe:
    pose: np.ndarray  # 6-vector (absolute, or offset when the animation is relative)
    antennas: tuple[float, float] | None = None
    body_yaw: float | None = None
    duration: float = 0.3


@dataclass
class Animation:
    name: str
    keyframes: list[Keyframe]
    priority: int = 0
    blend_in: float = 0.2
    blend_out: float = 0.2
    relative: bool = False  # keyframes are offsets riding on the live gaze


@dataclass
class Command:
    pose: np.ndarray
    antennas: tuple[float, float]
    body_yaw: float = 0.0


@dataclass
class _AnimState:
    anim: Animation
    phase: str = "blend_in"  # blend_in -> playing -> blend_out
    t: float = 0.0
    kf_idx: int = 0
    kf_t: float = 0.0
    weight: float = 0.0
    start_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    start_ant: tuple[float, float] = (0.0, 0.0)
    pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ant: tuple[float, float] | None = None
    body: float | None = None


class MotionOwner:
    def __init__(
        self,
        robot=None,  # RobotLink-like: .mini (ReachyMini | None)
        journal=None,
        *,
        hz: float = CONTROL_HZ,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        is_speaking: Callable[[], bool] | None = None,
    ) -> None:
        self.robot = robot
        self.journal = journal
        self.hz = hz
        self.clock = clock
        self._rng = rng or random.Random()
        self._is_speaking = is_speaking or (lambda: False)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jobs: deque[tuple[str, Callable]] = deque()
        self.job_running: str | None = None
        now = clock()

        # base gaze: [yaw, pitch] in rad
        self._gaze_target = np.zeros(2)
        self._gaze_until: float | None = None
        self.gaze_source = "home"
        self._gaze_pos = np.zeros(2)
        self._gaze_vel = np.zeros(2)

        self._averting = False
        self._avert = np.zeros(2)
        self._avert_next = now + self._rng.uniform(*GAZE_HOLD_S)
        self._avert_end = 0.0

        self._anims: deque[Animation] = deque()
        self._anim: _AnimState | None = None

        self._speech = np.zeros(6)
        self._listening = False
        self._thinking = False
        self._thinking_t0 = now
        self._think_progress = 0.0
        self._think_dir = 1.0
        self._think_amp = 0.0
        self._last_activity = now
        self._idle_w = 0.0

        self._ant = np.zeros(2)
        self._frozen: np.ndarray | None = None
        self._unfreeze_t0: float | None = None
        self._unfreeze_from = np.zeros(2)
        self._body = 0.0

        self._resume: tuple[np.ndarray, np.ndarray, float] | None = None
        self.sleeping = False
        self._last_tick: float | None = None
        self.last_command: Command | None = None
        self._errors = 0

    # ---- intent API (any thread; cheap; never raises on bad input) ----

    def look_at(self, yaw_deg: float, pitch_deg: float = 0.0, *, hold_s: float | None = None, source: str = "tool"):
        yaw = math.radians(max(-MAX_YAW_DEG, min(MAX_YAW_DEG, float(yaw_deg))))
        pitch = math.radians(max(-MAX_PITCH_DEG, min(MAX_PITCH_DEG, float(pitch_deg))))
        with self._lock:
            now = self.clock()
            self._gaze_target = np.array([yaw, pitch])
            self._gaze_until = None if hold_s is None else now + hold_s
            self.gaze_source = source
            self._last_activity = now
            self._end_aversion(now)

    def center(self) -> None:
        with self._lock:
            self._gaze_target = np.zeros(2)
            self._gaze_until = None
            self.gaze_source = "home"
            self._last_activity = self.clock()

    @property
    def gaze_yaw_deg(self) -> float:
        """Current (smoothed) gaze yaw, for re-aim gating."""
        with self._lock:
            return math.degrees(self._gaze_pos[0])

    @property
    def gaze_target_yaw_deg(self) -> float:
        with self._lock:
            return math.degrees(self._gaze_target[0])

    def play(self, anim: Animation, *, preempt: bool = False) -> None:
        with self._lock:
            cur = self._anim
            if preempt and cur is not None and anim.priority >= cur.anim.priority:
                if cur.phase != "blend_out":
                    cur.phase, cur.t = "blend_out", 0.0
                self._anims.appendleft(anim)
            else:
                self._anims.append(anim)
            self._last_activity = self.clock()

    def queue_animation(self, keyframes, priority: int = 0, blend_in=0.2, blend_out=0.2, preempt=False) -> None:
        """Legacy MovementManager API (app/robot_tools.py, emotion_loader):
        keyframes carry 4x4 head matrices; converted once here."""
        kfs = []
        for kf in keyframes:
            pose = np.asarray(kf.pose, dtype=np.float64)
            kfs.append(
                Keyframe(
                    pose=pose6_from_matrix(pose) if pose.shape == (4, 4) else pose,
                    antennas=tuple(kf.antennas) if getattr(kf, "antennas", None) is not None else None,
                    body_yaw=getattr(kf, "body_yaw", None),
                    duration=float(getattr(kf, "duration", 0.3)),
                )
            )
        self.play(Animation("legacy", kfs, priority, blend_in, blend_out), preempt=preempt)

    def cancel_animation(self) -> None:
        with self._lock:
            self._anims.clear()
            if self._anim is not None and self._anim.phase != "blend_out":
                self._anim.phase, self._anim.t = "blend_out", 0.0

    def set_speech_offsets(self, offsets) -> None:
        with self._lock:
            self._speech = np.asarray(offsets, dtype=np.float64)

    def set_listening(self, listening: bool) -> None:
        with self._lock:
            self._listening = bool(listening)
            self._last_activity = self.clock()

    def set_thinking(self, thinking: bool) -> None:
        with self._lock:
            if thinking and not self._thinking:
                self._thinking_t0 = self.clock()
                self._think_dir = -self._think_dir  # alternate sides
            self._thinking = bool(thinking)
            self._last_activity = self.clock()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self.job_running is not None or bool(self._jobs)

    def exclusive(self, name: str, fn: Callable) -> bool:
        """Run fn(mini) on the motion thread with the loop paused (scripted
        SDK moves: goto_target, sleep, peekaboo). Returns False when there is
        no robot or another exclusive move is pending/running."""
        if self.robot is None or getattr(self.robot, "mini", None) is None:
            return False
        with self._lock:
            if self.job_running is not None or self._jobs:
                return False
            self._jobs.append((name, fn))
        return True

    def sleep(self) -> bool:
        def go(mini) -> None:
            mini.goto_target(head=SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS, duration=2.0)
            with self._lock:
                self.sleeping = True
                self._anims.clear()
                self._anim = None

        return self.exclusive("sleep", go)

    def wake(self) -> bool:
        """Leave the sleep pose (no-op when awake)."""
        if not self.sleeping:
            return False

        def go(mini) -> None:
            mini.goto_target(head=np.eye(4), antennas=INIT_ANTENNAS, duration=1.0)
            with self._lock:
                self.sleeping = False

        return self.exclusive("wake_up", go)

    # ---- control loop ----

    def start(self) -> bool:
        if self.robot is None or getattr(self.robot, "mini", None) is None:
            return False

        def startup(mini) -> None:  # from wherever it is (often the sleep pose)
            mini.goto_target(head=np.eye(4), antennas=INIT_ANTENNAS, duration=1.5)

        self._jobs.append(("startup", startup))
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="motion", daemon=True)
        self._thread.start()
        log.info("motion owner started at %.0f Hz", self.hz)
        if self.journal is not None:
            self.journal.write(J.MOTION_STARTED, hz=self.hz)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4.0)
            self._thread = None

    def _run(self) -> None:
        interval = 1.0 / self.hz
        next_t = self.clock()
        while not self._stop.is_set():
            job = None
            with self._lock:
                if self._jobs:
                    job = self._jobs.popleft()
                    self.job_running = job[0]
            if job is not None:
                self._run_job(*job)
                next_t = self.clock()
                continue
            cmd = self.tick()
            if cmd is not None:
                self._send(cmd)
            next_t += interval
            delay = next_t - self.clock()
            if delay < -interval:
                next_t = self.clock()  # fell behind (GC, CPU starved): don't burst
            elif delay > 0:
                self._stop.wait(delay)

    def _run_job(self, name: str, fn: Callable) -> None:
        mini = getattr(self.robot, "mini", None)
        t0 = self.clock()
        ok, err = True, None
        try:
            if mini is not None:
                fn(mini)
        except Exception as e:
            ok, err = False, repr(e)
            log.warning("exclusive move %s failed: %r", name, e)
        finally:
            self._capture_resume(mini)
            with self._lock:
                self.job_running = None
                self._last_tick = None
            if self.journal is not None:
                self.journal.write(J.MOTION_EXCLUSIVE, name=name, ok=ok, error=err, seconds=self.clock() - t0)

    def _capture_resume(self, mini) -> None:
        if mini is None or self.sleeping:
            return
        try:
            pose = pose6_from_matrix(mini.get_current_head_pose())
            ant = np.asarray(mini.get_present_antenna_joint_positions(), dtype=np.float64)
        except Exception as e:
            log.warning("could not read pose after exclusive move: %r", e)
            return
        with self._lock:
            self._resume = (pose, ant, self.clock())
            self._ant = ant.copy()

    def _send(self, cmd: Command) -> None:
        mini = getattr(self.robot, "mini", None)
        if mini is None:
            return
        try:
            mini.set_target(head=pose_matrix(cmd.pose), antennas=list(cmd.antennas), body_yaw=float(cmd.body_yaw))
        except Exception as e:
            self._errors += 1
            if self._errors == 1 or self._errors % 500 == 0:
                log.warning("set_target failed (%d so far): %r", self._errors, e)
                if self.journal is not None:
                    self.journal.write(J.MOTION_ERROR, error=repr(e), count=self._errors)

    def tick(self, now: float | None = None) -> Command | None:
        """One control step. Returns None while asleep."""
        with self._lock:
            now = self.clock() if now is None else now
            if self.sleeping:
                self._last_tick = now
                return None
            dt = 1.0 / self.hz if self._last_tick is None else max(0.0, min(0.1, now - self._last_tick))
            self._last_tick = now
            speaking = bool(self._is_speaking())
            engaged = self._listening or self._thinking or speaking
            animating = self._anim is not None or bool(self._anims)
            if engaged or animating:
                self._last_activity = now

            # -- base gaze --
            if self._gaze_until is not None and now >= self._gaze_until and not engaged:
                self._gaze_target = np.zeros(2)
                self._gaze_until = None
                self.gaze_source = "home"
            self._update_aversion(now, self.gaze_source in PERSON_SOURCES and not engaged and not animating)
            halflife = HOME_HALFLIFE if self.gaze_source == "home" else GAZE_HALFLIFE
            self._gaze_pos, self._gaze_vel = spring_update(
                self._gaze_pos, self._gaze_vel, self._gaze_target + self._avert, halflife, dt
            )
            base = pose6(yaw=self._gaze_pos[0], pitch=self._gaze_pos[1], degrees=False)

            # -- animation override --
            w, apose, aant, abody = self._step_animation(dt, base)
            pose = base + w * (apose - base) if w > 0.0 else base.copy()

            # -- additive layers --
            self._apply_thinking(now, dt, pose)
            pose += self._speech
            idle = not engaged and not animating and now - self._last_activity > BREATHING_DELAY
            self._idle_w = max(0.0, min(1.0, self._idle_w + (dt if idle else -dt) / BREATHING_BLEND_S))
            idle_w = smooth_step(self._idle_w)
            pose[2] += idle_w * BREATHING_Z_AMP * math.sin(2 * math.pi * BREATHING_Z_FREQ * now)

            # -- antennas --
            sway = BREATHING_ANTENNA_AMP * math.sin(2 * math.pi * BREATHING_ANTENNA_FREQ * now)
            target_ant = idle_w * np.array([sway, -sway])
            if self._think_amp > 0.0:
                a = self._think_amp * THINKING_ANTENNA_AMP
                ph = 2 * math.pi * THINKING_ANTENNA_FREQ * now
                target_ant = target_ant + np.array([a * math.sin(ph), a * math.sin(ph + THINKING_ANTENNA_PHASE)])
            if w > 0.0 and aant is not None:
                target_ant = (1.0 - w) * target_ant + w * np.asarray(aant)
            target_ant = self._apply_freeze(now, target_ant)
            self._ant = self._ant + exp_alpha(dt, ANTENNA_TAU) * (target_ant - self._ant)

            body_target = w * abody if (w > 0.0 and abody is not None) else 0.0
            self._body += exp_alpha(dt, BODY_TAU) * (body_target - self._body)

            # -- resume blend after an exclusive move --
            ant_out = self._ant.copy()
            if self._resume is not None:
                rpose, rant, t0 = self._resume
                s = minimum_jerk((now - t0) / RESUME_S)
                pose = rpose + s * (pose - rpose)
                ant_out = rant + s * (ant_out - rant)
                if s >= 1.0:
                    self._resume = None

            cmd = Command(pose=pose, antennas=(float(ant_out[0]), float(ant_out[1])), body_yaw=float(self._body))
            self.last_command = cmd
            return cmd

    # ---- tick helpers (lock held) ----

    def _end_aversion(self, now: float) -> None:
        self._averting = False
        self._avert = np.zeros(2)
        self._avert_next = now + self._rng.uniform(*GAZE_HOLD_S)

    def _update_aversion(self, now: float, allowed: bool) -> None:
        if not allowed:
            if self._averting or now >= self._avert_next:
                self._end_aversion(now)
            return
        if not self._averting and now >= self._avert_next:
            self._averting = True
            self._avert_end = now + self._rng.uniform(*GAZE_AWAY_S)
            side = self._rng.choice((-1.0, 1.0))
            self._avert = np.array(
                [GAZE_AWAY_YAW * side * self._rng.uniform(0.6, 1.0), GAZE_AWAY_PITCH * self._rng.uniform(0.5, 1.0)]
            )
        elif self._averting and now >= self._avert_end:
            self._end_aversion(now)

    def _apply_thinking(self, now: float, dt: float, pose: np.ndarray) -> None:
        if self._thinking:
            self._think_progress = min(1.0, self._think_progress + dt / THINKING_AVERT_S)
            self._think_amp = smooth_step((now - self._thinking_t0) / THINKING_RAMP_S)
        else:
            self._think_progress = max(0.0, self._think_progress - dt / THINKING_RETURN_S)
            self._think_amp = max(0.0, self._think_amp - dt / THINKING_RAMP_S)
        if self._think_progress <= 0.0:
            return
        k = minimum_jerk(self._think_progress)
        t = now - self._thinking_t0
        micro = max(0.0, (k - 0.7) / 0.3)
        pose[5] += k * THINKING_AVERT_YAW * self._think_dir + micro * THINKING_MICRO_YAW_AMP * math.sin(
            2 * math.pi * THINKING_MICRO_YAW_FREQ * t
        )
        pose[4] += k * THINKING_AVERT_PITCH + micro * THINKING_MICRO_PITCH_AMP * math.sin(
            2 * math.pi * THINKING_MICRO_PITCH_FREQ * t
        )
        pose[2] += k * THINKING_Z_AMP * math.sin(2 * math.pi * THINKING_Z_FREQ * t)

    def _apply_freeze(self, now: float, target: np.ndarray) -> np.ndarray:
        if self._listening:
            if self._frozen is None:
                self._frozen = self._ant.copy()
            self._unfreeze_t0 = None
            return self._frozen.copy()
        if self._frozen is None:
            return target
        if self._unfreeze_t0 is None:
            self._unfreeze_t0 = now
            self._unfreeze_from = self._frozen
        b = min(1.0, (now - self._unfreeze_t0) / ANTENNA_UNFREEZE_S)
        out = (1.0 - b) * self._unfreeze_from + b * target
        if b >= 1.0:
            self._frozen = None
            self._unfreeze_t0 = None
        return out

    def _start_next(self, base: np.ndarray) -> None:
        if not self._anims:
            self._anim = None
            return
        anim = self._anims.popleft()
        start = np.zeros(6) if anim.relative else base.copy()
        self._anim = _AnimState(
            anim=anim,
            start_pose=start,
            start_ant=(float(self._ant[0]), float(self._ant[1])),
            pose=start.copy(),
        )

    def _step_animation(self, dt: float, base: np.ndarray):
        if self._anim is None:
            if not self._anims:
                return 0.0, base, None, None
            self._start_next(base)
        st = self._anim
        a = st.anim
        if st.phase == "blend_in":
            st.t += dt
            st.weight = smooth_step(st.t / a.blend_in) if a.blend_in > 0 else 1.0
            done = self._advance_keyframes(dt, st)
            if st.t >= a.blend_in:
                st.phase, st.t, st.weight = ("blend_out", 0.0, 1.0) if done else ("playing", 0.0, 1.0)
        elif st.phase == "playing":
            st.weight = 1.0
            if self._advance_keyframes(dt, st):
                st.phase, st.t = "blend_out", 0.0
        else:  # blend_out
            st.t += dt
            st.weight = 1.0 - smooth_step(st.t / a.blend_out) if a.blend_out > 0 else 0.0
            if st.t >= a.blend_out:
                self._anim = None
                if self._anims:
                    self._start_next(base)
                return 0.0, base, None, None
        apose = base + st.pose if a.relative else st.pose
        return st.weight, apose, st.ant, st.body

    @staticmethod
    def _advance_keyframes(dt: float, st: _AnimState) -> bool:
        """Interpolate within the current keyframe; True when the last one is done."""
        kfs = st.anim.keyframes
        if not kfs:
            return True
        if st.kf_idx >= len(kfs):
            return True
        st.kf_t += dt
        while True:
            kf = kfs[st.kf_idx]
            if st.kf_idx == 0:
                p0, a0, b0 = st.start_pose, st.start_ant, 0.0
            else:
                prev = kfs[st.kf_idx - 1]
                p0, a0, b0 = prev.pose, prev.antennas, prev.body_yaw or 0.0
            t = st.kf_t / kf.duration if kf.duration > 0 else 1.0
            if t >= 1.0 and st.kf_idx < len(kfs) - 1:
                st.kf_t -= kf.duration
                st.kf_idx += 1
                continue
            s = minimum_jerk(t)
            st.pose = p0 + s * (np.asarray(kf.pose) - p0)
            if kf.antennas is not None:
                a0 = a0 if a0 is not None else (st.ant or st.start_ant)
                st.ant = tuple((1.0 - s) * a0[i] + s * kf.antennas[i] for i in range(2))
            if kf.body_yaw is not None:
                st.body = (1.0 - s) * b0 + s * kf.body_yaw
            return t >= 1.0


def nod_animation() -> Animation:
    d = 0.3
    return Animation(
        "nod",
        [Keyframe(pose6(pitch=-15), duration=d), Keyframe(pose6(pitch=15), duration=d), Keyframe(pose6(), duration=d)],
        blend_in=0.1,
        relative=True,
    )


def shake_animation() -> Animation:
    d = 0.3
    return Animation(
        "shake_head",
        [
            Keyframe(pose6(yaw=20), duration=d),
            Keyframe(pose6(yaw=-20), duration=d),
            Keyframe(pose6(yaw=20), duration=d),
            Keyframe(pose6(), duration=d),
        ],
        blend_in=0.1,
        relative=True,
    )
