"""The motion thread must not be able to die quietly (found by the type lane, 2026-10-13).

`_run_job` has always journalled its own failures and `_send` catches `set_target`
exceptions, but `tick()` sat bare in the loop: one unexpected value - a short keyframe
list, a NaN pose, a lock held by something that died - and the thread raised, the daemon
thread died with it, and the robot kept hearing and answering with a frozen head and zero
telemetry. These drive `_run` on the current thread, so the old bug shows up as the
exception propagating out of the test rather than as a race.

No reachy_mini needed, so this runs in the bare CI environment too.
"""

from __future__ import annotations

import numpy as np

from shell.motion.owner import MotionOwner


class Journal:
    def __init__(self):
        self.rows: list[tuple[str, dict]] = []

    def write(self, kind, **fields):
        self.rows.append((kind, fields))

    def ticks(self):
        return [f for k, f in self.rows if k == "motion.error" and f.get("where") == "tick"]


class FakeRobot:
    mini = None  # _send returns early; the loop's own contract is under test here


def drive(journal, tick, iterations, step=0.05):
    """Run the loop body `iterations` times with a clock that goes nowhere near real time."""
    now = [0.0]

    class Stop:
        def __init__(self):
            self.n = 0
            self.waits: list[float] = []

        def is_set(self):
            return self.n >= iterations

        def set(self):
            self.n = self.max_big = iterations

        def wait(self, seconds=None):
            self.waits.append(0.0 if seconds is None else seconds)
            self.n += 1
            now[0] += step

    stop = Stop()
    owner = MotionOwner(FakeRobot(), journal, hz=20.0, clock=lambda: now[0])
    sent: list = []
    owner._send = sent.append
    owner._stop = stop

    def counting_tick():
        if stop.n >= iterations:
            return None
        return tick()

    owner.tick = counting_tick
    owner._run()  # an unguarded raise propagates here and fails the test loudly
    return owner, stop, sent


def test_a_failing_tick_does_not_kill_the_motion_loop():
    journal = Journal()
    calls = [0]

    def tick():
        calls[0] += 1
        if calls[0] <= 3:
            raise RuntimeError("short keyframe list")
        return None

    owner, stop, sent = drive(journal, tick, 20)
    assert calls[0] == 20, "the loop stopped turning - that is the silent-death bug"
    assert owner.tick_errors == 3
    rows = journal.ticks()
    failures = [f for f in rows if f["error"] != "recovered"]
    assert len(failures) == 1, "a transient failure must not cost one journal row per tick"
    assert failures[0]["streak"] in (1, 2, 3), "streak must say how many ticks failed first"
    assert rows[-1]["error"] == "recovered", "the journal showed a failure and never its end"


def test_a_permanently_failing_tick_backs_off_instead_of_hot_looping():
    """50 Hz exception handling on a box also running llama.cpp is its own outage."""
    journal = Journal()
    calls = [0]

    def tick():
        calls[0] += 1
        raise RuntimeError("daemon gone")

    owner, stop, sent = drive(journal, tick, 200)
    assert calls[0] == 200, "gave up on the loop instead of backing off"
    assert owner.tick_errors == 200
    assert len([f for f in journal.ticks() if f["error"] != "recovered"]) <= 4, (
        "50 exceptions a second, journalled: log flood"
    )
    one_second = [w for w in stop.waits if w == MotionOwner.TICK_ERROR_BACKOFF_S]
    assert len(one_second) > 100, "never reached the 1 Hz retry cadence"
    assert not any(f["error"] == "recovered" for f in journal.ticks())


def test_motion_resumes_by_itself_when_the_bus_comes_back():
    """Why this parks nothing: a USB flap that outlasts a park leaves a headless robot for
    the rest of the evening, which is the failure we are trading away."""
    journal = Journal()
    calls = [0]
    pose = np.zeros(6, dtype=float)

    def tick():
        calls[0] += 1
        if calls[0] <= 40:  # past the 20-failure backoff threshold
            raise RuntimeError("usb flap")
        return type("Cmd", (), {"pose": pose, "antennas": (0.0, 0.0)})()

    owner, stop, sent = drive(journal, tick, 60)
    assert sent, "motion never resumed after the transient failure"
    rows = [f for f in journal.ticks() if f["error"] == "recovered"]
    assert len(rows) == 1 and rows[0]["streak"] == 40 and rows[0]["total"] == 40
