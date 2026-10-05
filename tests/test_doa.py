"""DOA buffer, gating and wake turn - no USB, no daemon."""

import math
import random

import pytest

from shell.motion import MotionOwner
from shell.motion.doa import (
    AimGate,
    DoaBuffer,
    DoaTracker,
    HttpDoaSource,
    UsbDoaSource,
    doa_to_yaw_deg,
    make_doa_source,
)


class Clock:
    def __init__(self, t=500.0):
        self.t = t

    def __call__(self):
        return self.t


class ScriptedSource:
    def __init__(self):
        self.next = None
        self.reads = 0

    def read(self):
        self.reads += 1
        return self.next

    def close(self):
        pass


def deg(d):  # azimuth for a talker at head-yaw d (positive = left)
    return math.pi / 2 - math.radians(d)


def make(lease=True, speaking=None, clock=None):
    clock = clock or Clock()
    motion = MotionOwner(None, clock=clock, rng=random.Random(0))
    src = ScriptedSource()
    speaking = speaking or [False]
    lease_on = [lease]
    tr = DoaTracker(src, motion, None, is_speaking=lambda: speaking[0], lease_active=lambda: lease_on[0], clock=clock)
    return tr, motion, src, clock, speaking, lease_on


def feed(tr, src, clock, yaw_deg, n, speech=True, hz=10):
    for _ in range(n):
        clock.t += 1.0 / hz
        src.next = (deg(yaw_deg), speech)
        tr.sample()


def test_doa_to_yaw_mapping_and_clamp():
    assert doa_to_yaw_deg(math.pi / 2) == pytest.approx(0)
    assert doa_to_yaw_deg(math.radians(64)) == pytest.approx(26)  # measured live: 64 deg -> 26 deg left
    assert doa_to_yaw_deg(0.0) == 60.0  # far left, clamped
    assert doa_to_yaw_deg(math.pi) == -60.0


def test_buffer_window_and_median_of_speech_only():
    b = DoaBuffer(window_s=2.0)
    for i, (th, sp) in enumerate([(0.5, True), (3.0, False), (1.0, True), (1.5, True)]):
        b.add(100.0 + 0.1 * i, th, sp)
    assert b.median_speech(100.3, 1.0) == (1.0, 3)
    assert b.median_speech(100.3, 1.0, min_samples=4) == (None, 3)
    b.add(103.0, 2.0, True)  # everything older than 2 s falls out
    assert b.median_speech(103.0, 5.0) == (2.0, 1)


def test_aim_gate_needs_change_and_interval():
    g = AimGate(12.0, 0.8)
    assert g.allows(0.0, 0.0)
    g.mark(0.0, 0.0)
    assert not g.allows(0.5, 30.0)  # too soon
    assert not g.allows(1.0, 10.0)  # too small
    assert g.allows(1.0, 13.0)


def test_wake_turn_uses_last_second_of_speech(journal):
    tr, motion, src, clock, *_ = make(lease=False)
    tr.journal = journal
    feed(tr, src, clock, -40, 10, speech=False)  # TV noise, not speech
    feed(tr, src, clock, 25, 3)
    feed(tr, src, clock, 31, 2)
    yaw = tr.wake_turn(t_detect=clock.t)
    assert yaw == pytest.approx(25, abs=0.01)  # median of 25,25,25,31,31
    assert motion.gaze_source == "wake" and motion.gaze_target_yaw_deg == pytest.approx(25, abs=0.01)
    ev = journal.find("motion.wake_turn")[0]
    assert ev["samples"] == 5 and ev["yaw_deg"] == 25.0 and ev["latency_ms"] is not None


def test_wake_turn_without_speech_samples_does_not_move(journal):
    tr, motion, src, clock, *_ = make(lease=False)
    tr.journal = journal
    assert tr.wake_turn() is None
    assert motion.gaze_source == "home"
    assert journal.find("motion.wake_turn")[0]["yaw_deg"] is None


def test_samples_ignored_while_reachy_speaks():
    speaking = [True]
    tr, motion, src, clock, *_ = make(speaking=speaking)
    feed(tr, src, clock, 30, 10)
    assert src.reads == 0 and tr.buffer.latest() is None
    speaking[0] = False
    clock.t += 0.1  # inside the echo tail
    tr.sample()
    assert src.reads == 0
    feed(tr, src, clock, 30, 5)
    assert src.reads > 0


def test_reaim_only_with_lease_and_hysteresis(journal):
    tr, motion, src, clock, _, lease_on = make(lease=False)
    tr.journal = journal
    feed(tr, src, clock, 30, 10)
    assert motion.gaze_source == "home"  # no lease: ambient talk never steers the head
    lease_on[0] = True
    feed(tr, src, clock, 30, 3)
    assert motion.gaze_source == "doa" and motion.gaze_target_yaw_deg == pytest.approx(30, abs=0.1)
    feed(tr, src, clock, 38, 10)  # 8 deg: below threshold
    assert motion.gaze_target_yaw_deg == pytest.approx(30, abs=0.1)
    feed(tr, src, clock, -10, 10)  # big move, well past 0.8 s
    assert motion.gaze_target_yaw_deg == pytest.approx(-10, abs=0.1)
    assert len(journal.find("motion.doa_aim")) == 2


def test_no_aim_at_a_mixture_of_talkers():
    tr, motion, src, clock, *_ = make()
    for i in range(20):  # two people alternating at +40 / -40
        feed(tr, src, clock, 40 if i % 2 else -40, 1)
    assert motion.gaze_source == "home"


def test_sources_never_raise(monkeypatch):
    import reachy_mini.media.audio_control_utils as acu

    class Dead:
        def read(self, name):
            raise OSError("[Errno 19] No such device")

        def close(self):
            pass

    monkeypatch.setattr(acu, "init_respeaker_usb", lambda: Dead())
    clock = Clock()
    src = UsbDoaSource(retry_s=2.0, clock=clock)
    assert src.read() is None and src.errors == 1
    assert src.read() is None and src.errors == 1  # backing off, not hammering
    clock.t += 2.5
    assert src.read() is None and src.errors == 2
    assert HttpDoaSource("http://127.0.0.1:9", timeout_s=0.05).read() is None


def test_source_selection(monkeypatch):
    assert make_doa_source("off") is None
    assert isinstance(make_doa_source("daemon"), HttpDoaSource)
    assert isinstance(make_doa_source("usb"), UsbDoaSource)
