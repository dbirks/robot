"""The greeter's policy, on a fake clock with fake frames.

Nothing here opens a camera, a device or a model: the source hands back
`object()`, the tracker hands back hand-built `Face`s, the clock is a variable.
What is under test is exactly the part that can hurt a room — when he says hello.
"""

from __future__ import annotations

import math
from pathlib import Path

from conftest import RecJournal
from test_vision_gaze import bbox_for, kps

from shell import journal as J
from shell.vision import greeter as G
from shell.vision.gaze import Gate
from shell.vision.greeter import GROUP_CUE, SOLO_CUE, Greeter
from shell.vision.tracker import Face


class Source:
    """The one-camera-owner stand-in: latest()/grab(), and the engage flag."""

    def __init__(self, frame=object()):
        self.frame = frame
        self.engaged = None
        self.error = None

    def latest(self):
        if self.error:
            raise RuntimeError(self.error)
        return self.frame

    def set_engaged(self, engaged):
        self.engaged = engaged


class Tracker:
    def __init__(self):
        self.faces: list[Face] = []
        self.error = None
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)
        return list(self.faces)


class Rec:
    def __init__(self):
        self.cues = []

    def __call__(self, cue):
        self.cues.append(cue)

    @property
    def kinds(self):
        return [c.kind for c in self.cues]

    @property
    def texts(self):
        return [c.text for c in self.cues]


def face(*, share=0.05, cx=640.0, score=0.9, azimuth=0.0, yaw=0.0, pitch=0.0, roll=0.0):
    """A hand-built detection: bbox gives the share, kps gives the angles."""
    return Face(
        bbox=bbox_for(share, cx),
        kps=kps(cx=cx, yaw=yaw, pitch=pitch, roll=roll),
        det_score=score,
        share=share,
        azimuth_deg=azimuth,
    )


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def build(source=None, tracker=None, rec=None, journal=None, **kw):
    kw.setdefault("clock", Clock())
    return Greeter(
        source=source or Source(),
        tracker=tracker or Tracker(),
        journal=journal,
        on_greet=rec if rec is not None else Rec(),
        **kw,
    )


def dwell(g, tracker, seconds, step=0.5, start=0.0):
    """Hold the tracker's faces for `seconds` of fake clock, sampling as we go.
    Returns the last verdict, which is the one that matters after a dwell."""
    now, verdict = start, "nobody-looking"
    while now <= start + seconds + 1e-9:
        verdict = g.sample(now)
        now += step
    return verdict


# ---- the dwell rule ----


def test_dwell_below_two_seconds_does_not_greet():
    rec, tracker = Rec(), Tracker()
    g = build(rec=rec, tracker=tracker, min_spacing_s=0.0)
    tracker.faces = [face()]
    for t in (0.0, 0.5, 1.0, 1.5, 1.9):
        assert g.sample(t) == "nobody-looking", t
    assert rec.cues == []


def test_dwell_of_two_seconds_greets_once():
    rec, tracker = Rec(), Tracker()
    g = build(rec=rec, tracker=tracker, min_spacing_s=0.0)
    tracker.faces = [face()]
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.1):
        g.sample(t)
    assert rec.kinds == ["solo"]
    assert rec.cues[0].text == SOLO_CUE


def test_engaged_flag_reaches_the_camera_owner():
    rec, source, tracker = Rec(), Source(), Tracker()
    g = build(source=source, rec=rec, tracker=tracker, min_spacing_s=0.0)
    tracker.faces = []
    g.sample(0.0)
    assert source.engaged is False
    tracker.faces = [face()]
    g.sample(0.5)
    assert source.engaged is True


# ---- the gate rejects, fail-closed ----


def test_a_face_scanning_the_room_never_dwels():
    rec, tracker = Rec(), Tracker()
    g = build(rec=rec, tracker=tracker, min_spacing_s=0.0)
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        tracker.faces = [face(yaw=40.0)] if t % 1.0 else [face()]  # keeps turning away
        g.sample(t)
    assert rec.cues == []


def test_off_axis_and_distant_faces_are_not_greeted():
    for label, kw in [
        ("profile", {"yaw": 25.0}),
        ("head down", {"pitch": -20.0}),
        ("across the room", {"share": 0.02}),
        ("too close", {"share": 0.7}),
        ("at the frame edge", {"cx": 60.0}),
        ("low confidence", {"score": 0.6}),
    ]:
        rec, tracker = Rec(), Tracker()
        g = build(rec=rec, tracker=tracker, min_spacing_s=0.0)
        tracker.faces = [face(**kw)]
        verdicts = {g.sample(t) for t in (0.0, 1.0, 2.0, 3.0, 4.0)}
        assert rec.cues == [], label
        assert verdicts == {"nobody-looking"}, (label, verdicts)


# ---- one greeting per presence ----


def test_one_greeting_per_presence_however_long_they_stand_there():
    rec, tracker = Rec(), Tracker()
    g = build(rec=rec, tracker=tracker, min_spacing_s=0.0)
    tracker.faces = [face()]
    dwell(g, tracker, 2.5)
    assert rec.kinds == ["solo"]
    dwell(g, tracker, 30.0, start=3.0)
    assert rec.kinds == ["solo"], "the same presence greeted twice"


def test_the_track_follows_someone_who_shifts_or_leans():
    rec, tracker = Rec(), Tracker()
    g = build(rec=rec, tracker=tracker, min_spacing_s=0.0)
    tracker.faces = [face()]
    g.sample(0.0)
    tracker.faces = [face(azimuth=8.0, share=0.06)]  # they leaned in, same bearing
    g.sample(0.5)
    tracker.faces = [face(azimuth=8.0, share=0.06)]
    g.sample(2.2)
    assert rec.kinds == ["solo"]
    assert len(g.tracks) == 1, "a lean created a second presence"


def test_leaving_the_frame_ends_the_presence_entirely():
    rec, tracker = Rec(), Tracker()
    g = build(rec=rec, tracker=tracker, min_spacing_s=0.0)
    tracker.faces = [face()]
    dwell(g, tracker, 2.5)
    assert rec.kinds == ["solo"]
    tracker.faces = []
    g.sample(10.0)  # they walked off
    assert g.tracks == {}, "a face was remembered after the presence ended"


# ---- room rate limits ----


def test_global_spacing_blocks_the_next_person():
    rec, tracker, clock = Rec(), Tracker(), Clock(0.0)
    g = build(rec=rec, tracker=tracker, clock=clock, min_spacing_s=120.0)
    tracker.faces = [face()]
    dwell(g, tracker, 2.5)
    assert rec.kinds == ["solo"]
    tracker.faces = []
    g.sample(20.0)  # first presence over
    tracker.faces = [face(azimuth=45.0)]  # a different person walks up
    assert dwell(g, tracker, 2.5, start=20.5) == "suppressed:spacing"
    assert rec.kinds == ["solo"]


def test_global_spacing_elapses_and_the_next_person_is_greeted():
    rec, tracker, clock = Rec(), Tracker(), Clock(0.0)
    g = build(rec=rec, tracker=tracker, clock=clock, min_spacing_s=120.0)
    tracker.faces = [face()]
    dwell(g, tracker, 2.5)
    tracker.faces = []
    g.sample(20.0)
    tracker.faces = [face(azimuth=45.0)]
    dwell(g, tracker, 2.5, start=130.0)
    assert rec.kinds == ["solo", "solo"]


def test_hourly_cap_is_hard():
    rec, tracker = Rec(), Tracker()
    g = build(rec=rec, tracker=tracker, min_spacing_s=0.0, max_per_hour=2)
    tracker.faces = [face()]
    dwell(g, tracker, 2.5)
    for i, t in enumerate((10.0, 20.0, 30.0)):
        tracker.faces = []
        g.sample(t)  # end the presence
        tracker.faces = [face(azimuth=20.0 * (i + 1))]
        dwell(g, tracker, 2.5, start=t + 0.5)
    assert rec.kinds == ["solo", "solo"], "the hourly cap let a third greeting through"
    assert g.sample(33.5) == "suppressed:hourly-cap"


def test_the_cap_resets_after_an_hour():
    rec, tracker, clock = Rec(), Tracker(), Clock(0.0)
    g = build(rec=rec, tracker=tracker, clock=clock, min_spacing_s=0.0, max_per_hour=1)
    tracker.faces = [face()]
    dwell(g, tracker, 2.5)
    tracker.faces = []
    g.sample(100.0)
    tracker.faces = [face(azimuth=40.0)]
    dwell(g, tracker, 2.5, start=100.5)
    assert rec.kinds == ["solo"]
    tracker.faces = []
    g.sample(3700.0)
    tracker.faces = [face(azimuth=40.0)]
    dwell(g, tracker, 2.5, start=3700.5)
    assert rec.kinds == ["solo", "solo"]


# ---- he never talks over himself, or over a conversation ----


def test_no_greeting_while_his_own_audio_is_playing():
    rec, tracker = Rec(), Tracker()
    playing = [False]
    g = build(rec=rec, tracker=tracker, min_spacing_s=0.0, is_playing=lambda: playing[0])
    tracker.faces = [face()]
    dwell(g, tracker, 2.5)
    assert rec.kinds == ["solo"]
    rec.cues.clear()
    tracker.faces = []
    g.sample(10.0)
    playing[0] = True
    tracker.faces = [face(azimuth=50.0)]
    assert dwell(g, tracker, 2.5, start=10.5) == "suppressed:speaking"
    assert rec.cues == []


def test_solo_greeting_is_suppressed_mid_conversation():
    rec, tracker = Rec(), Tracker()
    g = build(rec=rec, tracker=tracker, min_spacing_s=0.0, lease_active=lambda: True)
    tracker.faces = [face()]
    dwell(g, tracker, 2.5)
    assert rec.cues == []
    assert dwell(g, tracker, 1.0, start=3.0) == "suppressed:conversation"


def test_new_people_arriving_mid_conversation_get_one_notice_with_its_own_cooldown():
    rec, tracker, clock = Rec(), Tracker(), Clock(0.0)
    g = build(
        rec=rec,
        tracker=tracker,
        clock=clock,
        min_spacing_s=120.0,
        group_min_spacing_s=300.0,
        lease_active=lambda: True,
    )
    tracker.faces = [face(azimuth=-30.0), face(azimuth=30.0)]
    dwell(g, tracker, 2.5)
    assert rec.kinds == ["group"] and rec.texts == [GROUP_CUE]

    # 200 s later the global spacing has elapsed, the group cooldown has not
    tracker.faces = []
    g.sample(100.0)
    tracker.faces = [face(azimuth=-30.0), face(azimuth=30.0)]
    assert dwell(g, tracker, 2.5, start=200.5) == "suppressed:group-spacing"
    assert rec.kinds == ["group"]

    # after the group cooldown both are noticed once, together
    tracker.faces = []
    g.sample(350.0)
    tracker.faces = [face(azimuth=-30.0), face(azimuth=30.0)]
    dwell(g, tracker, 2.5, start=350.5)
    assert rec.kinds == ["group", "group"]


# ---- no attention is granted by a face (ADR 0003) ----


def test_the_greeter_touches_no_lease_and_imports_no_attention():
    source = Path(G.__file__).read_text()
    assert ".acquire(" not in source and ".renew(" not in source
    assert "from ..attention" not in source and "lease.ac" not in source
    fields = set(Greeter.__dataclass_fields__)
    assert not [f for f in fields if "lease" in f and f != "lease_active"], fields


def test_greet_hook_only_inserts_a_transcript():
    class Loop:
        def __init__(self):
            self.fns = []

        def call_soon_threadsafe(self, fn):
            self.fns.append(fn)

    class Client:
        def __init__(self):
            self.transcripts = []
            self.leases = []

        def insert_user_transcript(self, text, *, origin="user"):
            self.transcripts.append((text, origin))

        def acquire(self):  # pragma: no cover - must never be reached
            self.leases.append("acquire")

    loop, client = Loop(), Client()
    hook = G.greet_hook(client, loop)
    hook(G.GreetCue(text=SOLO_CUE, kind="solo", track_id=1))
    for fn in loop.fns:
        fn()
    # the marker is the ADR 0003 guard: greeter-originated replies exchange nothing
    assert client.transcripts == [(SOLO_CUE, "greeter")]
    assert client.leases == []


def test_a_failing_hook_does_not_kill_the_poll():
    calls = []

    def boom(cue):
        calls.append(cue)
        raise RuntimeError("loop is gone")

    tracker = Tracker()
    g = build(tracker=tracker, rec=boom, min_spacing_s=0.0)
    tracker.faces = [face()]
    assert dwell(g, tracker, 2.5) == "suppressed:presence-greeted"
    assert len(calls) == 1  # it did fire, and the poll thread is still here
    assert g.sample(3.5) == "suppressed:presence-greeted"


# ---- degrade: no camera at all ----


def test_a_broken_tracker_is_warned_about_once_and_stands_down(caplog):
    import logging

    class BrokenTracker:
        def __init__(self):
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            raise RuntimeError("insightface import exploded")

    caplog.set_level(logging.WARNING, logger="shell.vision.greeter")
    tracker = BrokenTracker()
    g = build(tracker=tracker, rec=Rec())
    for t in range(6):
        g.sample(float(t))
    assert tracker.calls == 3  # tried, failed three times in a row, stood down
    assert caplog.text.count("stood down") == 1
    g.sample(10.0)
    assert tracker.calls == 3  # never called again: no warning storm at poll rate


def test_sleeping_he_reads_no_frames_and_detects_nothing():
    tracker = Tracker()
    tracker.faces = [face()]
    rec = Rec()
    g = build(tracker=tracker, rec=rec, asleep=lambda: True)
    for t in range(6):
        assert g.sample(float(t)) == "asleep"
    assert tracker.calls == 0 and rec.cues == []


def test_no_camera_degrades_to_disabled_and_says_nothing():
    from shell.camera import CameraStream

    class DeadGrabber:
        def grab(self):
            raise RuntimeError("Reachy Mini camera not found")

    stream = CameraStream(DeadGrabber())
    assert stream.start() is False
    assert stream.available is False and stream.latest() is None
    rec, tracker = Rec(), Tracker()
    tracker.faces = [face()]
    g = build(source=stream, tracker=tracker, rec=rec, min_spacing_s=0.0)
    assert g.sample(0.0) == "no-camera"
    assert g.sample(5.0) == "no-camera"
    assert rec.cues == []
    stream.stop()


def test_no_source_and_no_tracker_are_both_disabled_not_errors():
    rec = Rec()
    assert Greeter(source=None, tracker=Tracker(), on_greet=rec).sample(0.0) == "no-camera"
    tracker = Tracker()
    tracker.error = "model load failed"
    assert build(rec=rec, tracker=tracker, min_spacing_s=0.0).sample(0.0) == "nobody-looking"
    assert rec.cues == []


def test_greeter_disabled_by_configuration_does_not_even_read_the_camera():
    source = Source()
    g = Greeter(source=source, tracker=Tracker(), enabled=False)
    assert g.sample(0.0) == "disabled"
    assert g.start() is False


# ---- journal rows (ADR 0007: "why did he not say hello" must be answerable) ----


def test_the_journal_sees_the_face_the_decision_and_the_verdict():
    journal = RecJournal()
    tracker = Tracker()
    g = build(tracker=tracker, journal=journal, min_spacing_s=0.0, lease_active=lambda: True)
    tracker.faces = [face()]
    dwell(g, tracker, 2.5)
    assert J.FACE_PRESENT in journal.types()
    assert J.GREETER_SUPPRESSED in journal.types()
    assert journal.find(J.GREETER_SUPPRESSED)[0]["reason"] == "conversation"
    assert J.GREETER_FIRED not in journal.types()


# ---- the head hook (REACHY_GREETER_FACE_CENTERING) ----


class Motion:
    """The `MotionOwner` stand-in: records aims, and can say "busy" or "asleep".

    A real `look_at` clamps and smooths; recording the raw call is the point — the
    greeter is the caller under test, and this is the only seam where the head is
    asked to move for a face (shell/vision/greeter.py `_center`).
    """

    def __init__(self, *, busy=False, sleeping=False):
        self.busy = busy
        self.sleeping = sleeping
        self.aims: list[tuple[float, float, float, str]] = []

    def look_at(self, yaw_deg, pitch_deg=0.0, *, hold_s=None, source="tool"):
        self.aims.append((round(yaw_deg, 2), round(pitch_deg, 2), hold_s, source))


def test_face_centering_off_is_the_default_and_the_head_never_moves():
    """Tonight's live state in both rooms: the greet happens, the head does not."""
    rec, tracker, motion = Rec(), Tracker(), Motion()
    g = build(rec=rec, tracker=tracker, motion=motion, min_spacing_s=0.0)
    assert g.face_centering is False
    tracker.faces = [face(azimuth=15.0)]
    dwell(g, tracker, 2.5)
    assert rec.kinds == ["solo"]
    assert motion.aims == []


def test_face_centering_aims_at_the_dwelled_face_and_keeps_re_aiming():
    """The ask is lock-on-and-track, so the hook is not a one-shot nod toward the
    person: it re-aims at poll rate for as long as the presence holds (2 FPS idle /
    3 FPS engaged, `REAIM_HOLD_S` per aim)."""
    rec, tracker, motion = Rec(), Tracker(), Motion()
    g = build(rec=rec, tracker=tracker, motion=motion, min_spacing_s=0.0, face_centering=True)
    tracker.faces = [face(azimuth=15.0)]
    dwell(g, tracker, 2.5)
    assert motion.aims[0] == (15.0, 0.0, G.REAIM_HOLD_S, "face")
    first = len(motion.aims)
    assert rec.kinds == ["solo"]
    # The greeting is already spent, so every aim after this row is tracking, not
    # greeting: "presence-greeted" must not stop the head.
    assert dwell(g, tracker, 2.0, start=2.5) == "suppressed:presence-greeted"
    assert len(motion.aims) > first


def test_tracking_stops_when_the_face_leaves_the_frame():
    """There is no lock-on that survives an absence: the track evaporates after
    `TRACK_LOST_S` unseen and with it the only thing we knew about the person."""
    tracker, motion = Tracker(), Motion()
    g = build(tracker=tracker, motion=motion, min_spacing_s=0.0, face_centering=True)
    tracker.faces = [face()]
    dwell(g, tracker, 2.5)
    aims = len(motion.aims)
    assert aims > 0
    tracker.faces = []
    # A track lingers for TRACK_LOST_S after the last detection (one dropped poll
    # must not reset a dwell), and a stale candidate is still re-aimed in that
    # grace. Past the grace there is nobody left to aim at, ever.
    dwell(g, tracker, 2.0, start=2.5)
    settled = len(motion.aims)
    dwell(g, tracker, 3.0, start=4.5)
    assert len(motion.aims) == settled


def test_face_centering_stands_down_for_an_exclusive_move_or_the_nap():
    """`MotionOwner.busy` is a scripted `exclusive()` move on the motion thread and
    `sleeping` is the shell's one asleep flag; neither may be overwritten by a face.
    Note what this does NOT cover: a `doa`/`wake` gaze target has no busy flag, so a
    face aim and a wake turn both write `_gaze_target` and the last writer wins."""
    for kw in ({"busy": True}, {"sleeping": True}):
        tracker, motion = Tracker(), Motion(**kw)
        g = build(tracker=tracker, motion=motion, min_spacing_s=0.0, face_centering=True)
        tracker.faces = [face()]
        dwell(g, tracker, 2.5)
        assert motion.aims == [], kw


# ---- the gate numbers are the ones the plan measured ----


def test_dwell_and_gate_defaults_are_the_decided_numbers():
    g = Greeter()
    assert g.dwell_s == 2.0 and g.min_spacing_s == 120.0
    assert g.group_min_spacing_s == 300.0 and g.max_per_hour == 6
    assert (g.idle_fps, g.engaged_fps) == (2.0, 3.0)
    assert Gate().min_area_share == 0.04 and math.isclose(g.gate.hysteresis, 0.7)
