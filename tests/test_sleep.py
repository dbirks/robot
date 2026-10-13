"""Sleep is one state, not one pose (issue #36), hardware-free.

The seam under test is `shell/sleep.py`: the flag plus the three routing rules
`shell/__main__.py`'s `on_pcm` / `attention_tick` apply to every frame. Those
closures need a live mic and a robot, so the tests drive the same pure
functions with the same arguments (a frame is fed to the service iff
`feeds_service(...)`, KWS runs iff `feeds_keyword_spotter(...)`) over real
`SleepState` / `MotionOwner` / `AttentionLease` objects. What the head and the
microphone do with the flag is covered here: fake `ReachyMini`, no audio device.
"""

import random

import numpy as np

from shell.attention.lease import AttentionLease
from shell.attention.profiles import load_profile
from shell.motion.owner import MotionOwner
from shell.robot_tools import make_motion_handlers
from shell.sleep import SleepState, feeds_keyword_spotter, feeds_service, should_auto_sleep


class FakeMini:
    def __init__(self):
        self.gotos = []

    def goto_target(self, head=None, antennas=None, duration=0.5, **_):
        self.gotos.append((head, antennas, duration))

    def set_target(self, **_):
        pass


class FakeRobot:
    connected = True

    def __init__(self, mini=None):
        self.mini = mini


class FakeKws:
    """Stands in for sherpa-onnx: hands back the frames it was given."""

    def __init__(self):
        self.heard: list[bytes] = []

    def process(self, pcm):
        self.heard.append(pcm)
        return []


def rig(journal, mini=None):
    """(sleep, motion, lease, kws) wired the way the shell assembly wires them."""
    state = SleepState(journal)
    motion = MotionOwner(FakeRobot(mini), journal, rng=random.Random(0), sleep_state=state)
    lease = AttentionLease(load_profile("quiet"), journal, asleep=lambda: state.sleeping)
    return state, motion, lease, FakeKws()


def pump(mic_frames, state, lease, kws, speaker_playing=False, client=None):
    """The `on_pcm` body's two gates, per frame."""
    for pcm in mic_frames:
        engaged = lease.active() or speaker_playing
        if client is not None:
            client.append("real" if feeds_service(sleeping=state.sleeping, engaged=engaged) else "silence")
        if feeds_keyword_spotter(sleeping=state.sleeping, engaged=engaged):
            kws.process(pcm)


def test_asleep_the_client_gets_only_silence_but_kws_still_hears(journal):
    state, motion, lease, kws = rig(journal, FakeMini())
    frames = [b"\x01\x00" * 512, b"\x02\x00" * 512]
    client: list[str] = []

    lease.acquire("david", "kws:Reachy")
    pump(frames, state, lease, kws, client=client)
    assert client == ["real", "real"]  # engaged: the service owns the mic, KWS stands down
    assert kws.heard == []

    client.clear()
    state.enter("tool")  # via motion.sleep() in production; the flag is the same object
    pump(frames, state, lease, kws, client=client)
    assert client == ["silence", "silence"]  # even on a lease that was still live
    assert kws.heard == frames  # and the wake word is still audible (issue #36)


def test_wake_word_clears_sleep_and_feeding_resumes(journal):
    state, motion, lease, _ = rig(journal, FakeMini())
    motion.sleep(reason="tool")
    assert lease.acquire("david", "kws:Reachy") is False  # attention is not available
    assert feeds_service(sleeping=state.sleeping, engaged=True) is False

    # `on_pcm`'s KWS branch: the hit is the alarm clock, then normal attention.
    assert motion.wake(reason="kws:Reachy") is True
    assert not state.sleeping
    assert lease.acquire("david", "kws:Reachy") is True
    assert feeds_service(sleeping=state.sleeping, engaged=lease.active()) is True
    assert journal.find("sleep.enter") == [{"reason": "tool"}]
    assert journal.find("sleep.exit") == [{"reason": "kws:Reachy"}]


def test_wake_word_pose_is_queued_not_dropped(journal):
    mini = FakeMini()
    state, motion, _, _ = rig(journal, mini)
    motion.sleep(reason="auto_sleep")
    name, fn = motion._jobs.popleft()
    fn(mini)  # the motion thread plays it; loop stays paused while asleep
    assert name == "sleep" and len(mini.gotos) == 1
    assert motion.tick() is None  # asleep: no control commands, the pose holds

    assert motion.wake(reason="kws:Reachy") is True
    name, fn = motion._jobs.popleft()
    fn(mini)
    assert name == "wake_up" and len(mini.gotos) == 2
    assert motion.tick() is not None  # control loop resumes from the sleep pose


def test_a_physical_tool_clears_sleep_and_feeding_resumes(journal):
    state, motion, lease, _ = rig(journal, FakeMini())
    handlers = make_motion_handlers(FakeRobot(FakeMini()), motion, play_sound=lambda name: None, rng=random.Random(0))
    handlers["go_to_sleep"]()
    assert state.sleeping and feeds_service(sleeping=state.sleeping, engaged=True) is False

    handlers = make_motion_handlers(FakeRobot(FakeMini()), motion, play_sound=lambda name: None, rng=random.Random(0))
    assert handlers["nod"]()["ok"] is True
    assert not state.sleeping  # head and mic moved together (issue #36)
    assert journal.find("sleep.exit") == [{"reason": "tool"}]


def test_sleep_and_wake_events_carry_a_reason(journal):
    state = SleepState(journal)
    assert state.enter("tool") is True
    assert state.enter("tool") is False  # no duplicate row for a duplicate ask
    assert state.exit("kws:Reachy") is True
    assert state.exit("kws:Reachy") is False
    assert journal.types() == ["sleep.enter", "sleep.exit"]
    assert state.reason == "kws:Reachy"


def test_auto_sleep_only_fires_on_an_idle_awake_shell():
    assert should_auto_sleep(auto_sleep_s=300.0, idle_s=299.0, engaged=False, sleeping=False) is False
    assert should_auto_sleep(auto_sleep_s=300.0, idle_s=301.0, engaged=False, sleeping=False) is True
    assert should_auto_sleep(auto_sleep_s=300.0, idle_s=9999.0, engaged=True, sleeping=False) is False
    assert should_auto_sleep(auto_sleep_s=300.0, idle_s=9999.0, engaged=False, sleeping=True) is False
    assert should_auto_sleep(auto_sleep_s=0.0, idle_s=1e9, engaged=False, sleeping=False) is False  # home


def test_the_silence_buffer_is_what_an_unengaged_client_gets():
    # `feeds_service` only; the buffer itself is on_pcm's, unchanged by issue #36.
    assert feeds_service(sleeping=False, engaged=False) is False
    assert feeds_service(sleeping=False, engaged=True, muted=True) is False
    assert feeds_service(sleeping=True, engaged=True, muted=False) is False
    assert np.zeros(4).tobytes() != b""  # the frame shape the gates are handed


class RecordingJournal:
    def __init__(self):
        self.rows = []

    def write(self, type_, **payload):
        self.rows.append((type_, payload))


def test_asleep_refuses_speech_but_still_allows_acks():
    """The invariant the operator asked for: going to sleep turns the talking off.

    Measured 2026-10-12 23:09:21-23:09:27 - go_to_sleep returned ok:true, the head
    folded, and he then said "Let me be clear: I don't have a sleep function". The
    sleep flag reached the microphone and the motion thread but not the speaker queue.
    """
    from shell.audio.speaker import SpeakerOwner

    nap = SleepState()
    j = RecordingJournal()
    sp = SpeakerOwner(rate=16000, block=512, journal=j, asleep=lambda: nap.sleeping)
    gen = sp.begin_generation()
    assert sp.enqueue(gen, "tts", np.zeros(64, dtype=np.int16)) is True  # awake: speech flows

    nap.enter("tool")
    queued = sp.pending()  # the awake chunk from above is still in line
    gen2 = sp.begin_generation()
    assert sp.enqueue(gen2, "tts", np.zeros(64, dtype=np.int16)) is False
    assert sp.pending() == queued, "asleep but audio still queued: he would talk lying down"
    assert sp.enqueue(gen2, "ack", np.zeros(64, dtype=np.int16)) is True, "acks are not speech"
    assert sum(1 for t, _ in j.rows if t == "tts.refused_asleep") == 1, "one row per generation, not per chunk"

    nap.exit("kws:Reachy")
    gen3 = sp.begin_generation()
    assert sp.enqueue(gen3, "tts", np.zeros(64, dtype=np.int16)) is True


def test_sleep_pose_reports_where_the_antennas_actually_went():
    """ "The antennas didn't go back" vs "they went to -3.05 rad" must be a number.

    Our SLEEP_ANTENNAS is byte-identical to the SDK's SLEEP_ANTENNAS_JOINT_POSITIONS and
    the exclusive move logged ok:true in 2.009 s, so the only way to settle the
    difference is to journal the present positions either side of the move.
    """
    from shell.motion.owner import SLEEP_ANTENNAS, MotionOwner

    class FakeMini:
        def __init__(self):
            self.antennas = [-0.17, 0.17]
            self.moves = 0

        def goto_target(self, head=None, antennas=None, duration=1.0):
            self.moves += 1
            self.antennas = [float(a) for a in (antennas or [])]

        def get_present_antenna_joint_positions(self):
            return list(self.antennas)

    mini_holder = {}

    class FakeRobot:
        connected = True

        @property
        def mini(self):
            return mini_holder["mini"]

    fake = FakeMini()
    mini_holder["mini"] = fake
    j = RecordingJournal()
    owner = MotionOwner(FakeRobot(), journal=j)
    assert owner.sleep(reason="tool") is True
    # run the queued exclusive move the way the motion thread would
    name, fn = owner._jobs.popleft()
    fn(fake)
    assert fake.moves == 1
    rows = [p for name, p in j.rows if name == "motion.sleep_pose"]
    assert rows, "sleep pose wrote no telemetry"
    assert rows[-1]["antennas_before"] == [-0.17, 0.17]
    assert rows[-1]["antennas_after"] == [round(float(a), 3) for a in SLEEP_ANTENNAS]
    assert rows[-1]["move"] == "sleep"
