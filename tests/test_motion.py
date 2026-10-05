"""MotionOwner + pose math, hardware-free (fake clock, fake ReachyMini)."""

import math
import random
import time

import numpy as np
import pytest

from shell.motion import Animation, Keyframe, MotionOwner, nod_animation
from shell.motion.kinematics import minimum_jerk, pose6, pose6_from_matrix, pose_matrix, smooth_step, spring_update
from shell.motion.owner import GAZE_HOLD_S, RESUME_S


class FakeMini:
    def __init__(self):
        self.gotos = []
        self.targets = []
        self.pose = np.eye(4)
        self.ant = [0.0, 0.0]

    def goto_target(self, head=None, antennas=None, duration=0.5, **_):
        self.gotos.append((head, antennas, duration))
        if head is not None:
            self.pose = np.asarray(head)
        if antennas is not None:
            self.ant = list(antennas)

    def set_target(self, head=None, antennas=None, body_yaw=None):
        self.targets.append((head, antennas, body_yaw))

    def get_current_head_pose(self):
        return self.pose

    def get_present_antenna_joint_positions(self):
        return self.ant


class FakeRobot:
    def __init__(self, mini=None):
        self.mini = mini

    @property
    def connected(self):
        return self.mini is not None


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(journal=None, speaking=None, mini=None):
    clock = Clock()
    m = MotionOwner(
        FakeRobot(mini),
        journal,
        clock=clock,
        rng=random.Random(1),
        is_speaking=(lambda: speaking[0]) if speaking else None,
    )
    return m, clock


def run(m, clock, seconds, hz=50):
    out = []
    for _ in range(int(seconds * hz)):
        clock.t += 1.0 / hz
        out.append(m.tick())
    return out


# ---- pose math ----


def test_pose_matrix_matches_sdk_convention():
    from reachy_mini.utils import create_head_pose

    args = dict(x=0.01, y=-0.02, z=0.005, roll=7.0, pitch=-12.0, yaw=33.0)
    ours = pose_matrix(pose6(**args))
    sdk = create_head_pose(**args, degrees=True)
    assert np.allclose(ours, sdk, atol=1e-9)
    assert np.allclose(pose6_from_matrix(sdk), pose6(**args), atol=1e-9)


def test_spring_converges_without_overshoot():
    pos, vel = 0.0, 0.0
    seen = []
    for _ in range(200):
        pos, vel = spring_update(pos, vel, 1.0, 0.35, 0.02)
        seen.append(pos)
    assert max(seen) <= 1.0 + 1e-9
    assert seen[-1] == pytest.approx(1.0, abs=1e-3)
    assert all(b >= a - 1e-12 for a, b in zip(seen, seen[1:]))  # monotonic


def test_interpolators_hit_endpoints():
    for f in (minimum_jerk, smooth_step):
        assert f(0.0) == 0.0 and f(1.0) == 1.0 and f(-1) == 0.0 and f(2) == 1.0
        assert f(0.5) == pytest.approx(0.5)


# ---- gaze ----


def test_look_at_springs_there_then_drifts_home():
    m, clock = make()
    m.look_at(30, hold_s=2.0, source="tool")
    cmds = run(m, clock, 1.9)
    yaws = [math.degrees(c.pose[5]) for c in cmds]
    assert yaws[-1] == pytest.approx(30, abs=0.5)
    assert max(yaws) <= 30 + 1e-6
    run(m, clock, 4.0)  # hold expired, nobody engaged
    assert m.gaze_source == "home"
    assert math.degrees(m.last_command.pose[5]) == pytest.approx(0, abs=0.5)


def test_hold_does_not_expire_while_speaking():
    speaking = [True]
    m, clock = make(speaking=speaking)
    m.look_at(25, hold_s=0.5, source="wake")
    run(m, clock, 2.0)
    assert m.gaze_source == "wake"
    speaking[0] = False
    run(m, clock, 3.0)
    assert m.gaze_source == "home"


def test_look_at_clamps():
    m, clock = make()
    m.look_at(170, pitch_deg=-80)
    assert m.gaze_target_yaw_deg == pytest.approx(60)
    run(m, clock, 3.0)
    assert math.degrees(m.last_command.pose[4]) == pytest.approx(-25, abs=0.5)


def test_idle_gaze_aversion_only_when_attending_a_person():
    m, clock = make()
    m.look_at(0, source="tool")
    run(m, clock, GAZE_HOLD_S[1] + 2)
    assert not m._averting  # a tool look is not a person: no aversion
    m.look_at(0, source="doa")
    yaws = [abs(math.degrees(c.pose[5])) for c in run(m, clock, 2 * GAZE_HOLD_S[1] + 3)]
    assert max(yaws) > 3.0  # looked away at some point
    assert max(yaws) < 13.0


# ---- animation ----


def test_relative_nod_rides_on_gaze_and_finishes():
    m, clock = make()
    m.look_at(30, source="tool")
    run(m, clock, 2.0)
    m.play(nod_animation())
    cmds = run(m, clock, 0.6)
    pitches = [math.degrees(c.pose[4]) for c in cmds]
    yaws = [math.degrees(c.pose[5]) for c in cmds]
    assert min(pitches) < -8 and max(yaws) == pytest.approx(30, abs=1.0) and min(yaws) > 28
    run(m, clock, 1.5)
    assert m._anim is None
    assert math.degrees(m.last_command.pose[4]) == pytest.approx(0, abs=0.5)


def test_absolute_animation_blends_from_base_and_back():
    m, clock = make()
    m.play(Animation("tilt", [Keyframe(pose6(roll=20), duration=0.5)], blend_in=0.1, blend_out=0.2))
    rolls = [math.degrees(c.pose[3]) for c in run(m, clock, 0.6)]
    assert rolls[0] < 2.0  # no jump on the first tick
    assert max(rolls) == pytest.approx(20, abs=0.5)
    run(m, clock, 0.5)
    assert math.degrees(m.last_command.pose[3]) == pytest.approx(0, abs=0.1)


def test_preempt_replaces_lower_priority():
    m, clock = make()
    m.play(Animation("long", [Keyframe(pose6(yaw=20), duration=5.0)], priority=0))
    run(m, clock, 0.5)
    m.play(Animation("urgent", [Keyframe(pose6(pitch=10), duration=0.3)], priority=1), preempt=True)
    run(m, clock, 0.3)  # old one blends out
    run(m, clock, 0.2)
    assert m._anim is not None and m._anim.anim.name == "urgent"


def test_legacy_queue_animation_accepts_matrices():
    class KF:  # app.movement_manager.AnimationKeyframe shape
        def __init__(self, pose, duration):
            self.pose, self.duration, self.antennas, self.body_yaw = pose, duration, [0.3, -0.3], None

    m, clock = make()
    m.queue_animation([KF(pose_matrix(pose6(yaw=15)), 0.3)], blend_in=0.1, preempt=True, priority=1)
    run(m, clock, 0.3)  # end of the keyframe, before blend-out
    assert math.degrees(m.last_command.pose[5]) == pytest.approx(15, abs=1.0)
    assert m.last_command.antennas[0] > 0.1


# ---- additive layers ----


def test_thinking_looks_away_alternating_sides_and_returns():
    m, clock = make()
    m.set_thinking(True)
    a = math.degrees(run(m, clock, 0.6)[-1].pose[5])
    m.set_thinking(False)
    run(m, clock, 0.5)
    assert abs(math.degrees(m.last_command.pose[5])) < 0.5
    m.set_thinking(True)
    b = math.degrees(run(m, clock, 0.6)[-1].pose[5])
    assert abs(a) > 8 and abs(b) > 8 and a * b < 0


def test_breathing_only_when_idle():
    speaking = [True]
    m, clock = make(speaking=speaking)
    zs = [c.pose[2] for c in run(m, clock, 3.0)]
    assert max(abs(z) for z in zs) < 1e-6
    speaking[0] = False
    ants = [c.antennas[0] for c in run(m, clock, 4.0)]
    assert max(ants) > math.radians(5)  # breathing sway kicked in


def test_antennas_freeze_while_listening():
    m, clock = make()
    run(m, clock, 3.0)  # breathing sway running
    m.set_listening(True)
    run(m, clock, 0.1)
    held = m.last_command.antennas
    for c in run(m, clock, 2.0):
        assert c.antennas == pytest.approx(held, abs=1e-3)
    m.set_listening(False)
    run(m, clock, 2.0)
    assert m._frozen is None


def test_speech_offsets_add_on_top():
    m, clock = make()
    m.look_at(10, source="tool")
    run(m, clock, 2.0)
    m.set_speech_offsets((0, 0, 0, 0, 0, math.radians(5)))
    assert math.degrees(run(m, clock, 0.02)[-1].pose[5]) == pytest.approx(15, abs=0.3)


# ---- exclusive moves / sleep ----


def test_exclusive_requires_robot_and_is_single():
    m, _ = make()
    assert m.exclusive("x", lambda mini: None) is False  # no robot
    m, _ = make(mini=FakeMini())
    assert m.exclusive("a", lambda mini: None) is True
    assert m.exclusive("b", lambda mini: None) is False  # one at a time


def test_exclusive_resumes_from_measured_pose(journal):
    mini = FakeMini()
    m, clock = make(journal=journal, mini=mini)
    m.exclusive("peek", lambda mn: mn.goto_target(head=pose_matrix(pose6(pitch=20)), antennas=[0.5, 0.5], duration=1))
    name, fn = m._jobs.popleft()
    m._run_job(name, fn)
    first = run(m, clock, 0.02)[0]
    assert math.degrees(first.pose[4]) == pytest.approx(20, abs=1.0)  # no jump back to the loop's pose
    run(m, clock, RESUME_S + 0.5)
    assert abs(math.degrees(m.last_command.pose[4])) < 1.0
    assert journal.find("motion.exclusive")[0]["ok"] is True


def test_sleep_pauses_loop_until_wake(journal):
    mini = FakeMini()
    m, clock = make(journal=journal, mini=mini)
    assert m.sleep() is True
    m._run_job(*m._jobs.popleft())
    assert m.sleeping and m.tick() is None
    assert m.wake() is True
    m._run_job(*m._jobs.popleft())
    assert not m.sleeping and m.tick() is not None
    assert len(mini.gotos) == 2


def test_failing_exclusive_is_journaled_not_raised(journal):
    m, _ = make(journal=journal, mini=FakeMini())

    def bad(mini):
        raise RuntimeError("daemon gone")

    m.exclusive("bad", bad)
    m._run_job(*m._jobs.popleft())
    ev = journal.find("motion.exclusive")[0]
    assert ev["ok"] is False and "daemon gone" in ev["error"]
    assert m.job_running is None


def test_loop_thread_sends_targets():
    mini = FakeMini()
    m = MotionOwner(FakeRobot(mini), None, hz=100)
    assert m.start() is True
    time.sleep(0.25)
    m.stop()
    assert mini.gotos, "startup move should run first"
    assert len(mini.targets) > 5
    head, ants, body = mini.targets[-1]
    assert head.shape == (4, 4) and len(ants) == 2 and isinstance(body, float)


def test_start_without_robot_is_noop():
    m = MotionOwner(FakeRobot(None))
    assert m.start() is False
