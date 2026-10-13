"""Lock-on: the head aims at a face and *stays* aimed at it, so the dwell can close.

The chain the booth needs is scan -> see a person -> lock on (aim and keep aiming) ->
~2 s of dwell -> greet. `shell/vision/greeter.py` had the last step and a broken version
of the middle one. Three claims live here, each with the pre-fix build run against the
same fake room so "it was broken" is a number rather than an accusation:

1. **The sign.** `Face.azimuth_deg` is positive to the right of the image
   (`shell/vision/gaze.py`), while `MotionOwner.look_at` is positive to the robot's LEFT
   (`look_left` sends +LOOK_ANGLE_DEG, `look_right` sends -): shell/robot_tools.py, pinned
   at tests/test_robot_tools.py:205-206. The legacy loop kept that straight with
   `yaw = -(cx - 0.5) * 2 * MAX_YAW` (app/face_tracker.py:126-129). The port dropped the
   minus, so centring turned him AWAY from the face it was tracking.
2. **The feedback.** The camera is head-mounted, so the measurement already contains our
   own turn: for a person at body-relative bearing `w`, the detector reports
   `azimuth = w + gaze_yaw`. Commanding that reading raw is positive feedback - the head
   chases an azimuth its own turn invented, `TRACK_MATCH_AZIMUTH` splits one person into
   several, and the dwell resets. `look_at_image` gave the legacy loop the compensation for
   free (app/face_tracker.py:141, and it is why the SDK call was in there at all); the port
   kept the spring and lost it. `aim = gaze_yaw - GAIN * azimuth` puts it back.
3. **When it starts.** Centring ran only on the dwell-qualified candidate, i.e. two
   seconds after somebody already inside the acceptance band looked at him. `centre_band`
   refuses any face more than ~24 deg off dead ahead, so a peripheral person could never
   be pulled in: no aim, no dwell, no greeting - and no sweep either, because `_scan`
   stands down while any track lives. `test_the_pre_fix_hook_...` runs that room: it is
   tonight's journal, where the only `greeter.fired` row exists for a face at 0.5 deg and
   nobody off-centre was ever greeted.

Hardware-free: the room, the tracker and the head are all fakes. The responsive room
(`Room`) is the one that matters - it moves the face in the frame when the head moves,
which is the only way to test a feedback loop without a servo.
"""

from __future__ import annotations

import math

from conftest import RecJournal
from test_vision_gaze import bbox_for, kps
from test_vision_greeter import Rec, Source, Tracker

from shell import journal as J
from shell.vision import greeter as G
from shell.vision.gaze import AZIMUTH_HALF_DEG, Gate
from shell.vision.greeter import FACE_FOLLOW_DEADBAND_DEG, FACE_FOLLOW_GAIN, Greeter
from shell.vision.tracker import Face

FRAME = (720, 1280)  # shell/camera.py's grab size == Greeter.frame_shape
FULL_VIEW = 2 * AZIMUTH_HALF_DEG  # deg of head-relative view across the whole frame
BAND_EDGE_DEG = Gate().centre_band / 2.0 * FULL_VIEW  # how off-centre a face may be: 24 deg


def face_at(azimuth: float, *, share: float = 0.05, score: float = 0.9) -> Face:
    """A detection whose bbox centre and `azimuth_deg` agree, as `FaceTracker` builds them.

    `gaze.azimuth_deg` is a linear map of the bbox centre, so a test that sets one without
    the other is describing a face the detector could not produce.
    """
    cx = (0.5 + azimuth / FULL_VIEW) * FRAME[1]
    return Face(
        bbox=bbox_for(share, cx),
        kps=kps(cx=cx),
        det_score=score,
        share=share,
        azimuth_deg=azimuth,
    )


class Head:
    """A `MotionOwner` stand-in that actually goes where it is told.

    `advance()` lands the last command before the next sample, i.e. no spring lag to hide
    behind — the worst case for a compensation term. `gaze_source` behaves like the real
    one for the only two things the greeter asks about: who holds the head, and (in
    `test_lock_on_resumes_...`) when a hold is let go.
    """

    def __init__(self, *, busy=False, sleeping=False):
        self.gaze_source = "home"
        self.gaze_yaw_deg = 0.0
        self.busy = busy
        self.sleeping = sleeping
        self._pending: list[float] = []
        self.aims: list[tuple[float, float, float | None, str]] = []

    def look_at(self, yaw_deg, pitch_deg=0.0, *, hold_s: float | None = None, source: str = "tool"):
        self.aims.append((round(float(yaw_deg), 2), round(float(pitch_deg), 2), hold_s, source))
        self.gaze_source = source
        self._pending = [float(yaw_deg)]

    def advance(self):
        if self._pending:
            self.gaze_yaw_deg = self._pending.pop(0)


class Room:
    """One still person, seen from a head that moves: the feedback loop, in a fake.

    `bearing_deg` is body-relative (positive = the robot's right) and does not change when
    he turns his own head. The camera is head-mounted, so what the detector reports is
    `bearing + gaze_yaw`; a face whose centre leaves the frame is not detected at all.
    """

    def __init__(self, head: Head, bearing_deg: float, *, share=0.05, score=0.9):
        self.head = head
        self.bearing = float(bearing_deg)
        self.share, self.score = share, score

    def faces(self) -> list[Face]:
        azimuth = self.bearing + self.head.gaze_yaw_deg
        if abs(azimuth) >= AZIMUTH_HALF_DEG:
            return []
        return [face_at(azimuth, share=self.share, score=self.score)]


class RoomTracker:
    def __init__(self, room: Room):
        self.room = room

    def detect(self, frame):
        return self.room.faces()


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def build(*, bearing=None, faces=None, cls=Greeter, **kw):
    """(greeter, head, journal, cues, clock) around one still person or one fixed frame.

    `bearing` gets the responsive room; `faces` gets the inert one from the other greeter
    tests (the head may move all it likes, the face stays where the test put it). `cls`
    lets the counterfactual hook be run on the identical rig.
    """
    clock, head, journal, cues = Clock(), Head(), RecJournal(), Rec()
    tracker: RoomTracker | Tracker
    if bearing is not None:
        tracker = RoomTracker(Room(head, bearing, share=kw.pop("share", 0.05), score=kw.pop("score", 0.9)))
    else:
        inert = Tracker()
        inert.faces = list(faces or [])
        tracker = inert
    g = cls(
        source=Source(),
        tracker=tracker,
        motion=head,
        journal=journal,
        on_greet=cues,
        clock=clock,
        **kw,
    )
    return g, head, journal, cues, clock


def walk(g, head, clock, seconds, *, step=0.5):
    """Sample `seconds` of fake time, letting the fake head arrive between samples."""
    out = []
    end = clock.t + seconds
    while clock.t <= end + 1e-9:
        head.advance()
        out.append((round(clock.t, 2), g.sample(clock.t)))
        clock.t += step
    return out


class PreFixHook(Greeter):
    """Tonight's head hook, verbatim, so the counterfactual runs on the same room:
    aim at the raw head-relative azimuth, and only once the dwell gate has closed."""

    def _aim_target(self, live: list[G.Track]) -> G.Track | None:
        return self._candidate(self.clock())

    def _center(self, track: G.Track) -> None:
        self.motion.look_at(track.azimuth_deg, 0.0, hold_s=G.REAIM_HOLD_S, source=G.FACE_SOURCE)


# ---- 1. the sign: a face on the right of the frame gets a right turn ----


def test_a_face_on_the_right_of_the_frame_turns_the_head_right():
    """The bug the whole feature hinged on: the greeter's azimuth is image-relative and
    `MotionOwner`'s is left-positive. Using one as the other turns him away from the room."""
    for azimuth, expect in ((20.0, -12.0), (-20.0, 12.0), (35.0, -21.0)):
        g, head, _j, _c, clock = build(faces=[face_at(azimuth)], face_centering=True, min_spacing_s=0.0)
        g.sample(clock.t)
        assert head.aims == [(expect, 0.0, G.REAIM_HOLD_S, G.FACE_SOURCE)], azimuth


def test_the_aim_is_where_the_head_already_points_plus_a_damped_correction():
    """aim = gaze_yaw - GAIN * azimuth: the same person in the same place produces the
    same world-directed command whatever the head is mid-turn through."""
    g, head, _j, _c, clock = build(faces=[face_at(20.0)], face_centering=True)
    head.gaze_yaw_deg = -10.0  # already panned 10 deg right
    g.sample(clock.t)
    assert head.aims[-1] == (-22.0, 0.0, G.REAIM_HOLD_S, G.FACE_SOURCE)  # -10 + 0.6 * -20
    assert -22.0 < -10.0, "the correction went the wrong way from a head that was already turned"


def test_the_correction_is_deadbanded_so_a_settled_face_stops_nudging():
    """Inside the deadband the correction is smaller than a twitch: the hold is refreshed
    where the head already is instead of re-commanding 0.6 of a sub-degree every poll."""
    for azimuth, expect in ((3.0, 0.0), (6.0, -3.6)):
        g, head, _j, _c, clock = build(faces=[face_at(azimuth)], face_centering=True)
        g.sample(clock.t)
        assert head.aims[-1][0] == expect, azimuth


# ---- 2. the feedback: a still person stays one track and one greeting ----


def test_lock_on_converges_on_a_still_person_instead_of_chasing_him():
    """The responsive room, which is the only room that can tell these two apart:
    one person, one track, one hello, and the head parked on him within the deadband."""
    g, head, journal, cues, clock = build(bearing=18.0, face_centering=True, min_spacing_s=0.0)
    walk(g, head, clock, 6.0)
    assert cues.kinds == ["solo"]
    assert len(g.tracks) == 1, f"one person became {len(g.tracks)} tracks: {list(g.tracks)}"
    assert len(journal.find(J.FACE_PRESENT)) == 1, "the dwell did not survive the pan"
    assert abs(head.gaze_yaw_deg + 18.0) <= FACE_FOLLOW_DEADBAND_DEG / FACE_FOLLOW_GAIN + 1e-9
    steps = [abs(b[0] - a[0]) for a, b in zip(head.aims, head.aims[1:])]
    assert max(steps) <= G.TRACK_MATCH_AZIMUTH, f"a re-aim wider than the match window: {steps}"
    assert all(a[0] <= 0.0 for a in head.aims), "crossed the person he was tracking"


def test_the_pre_fix_hook_turns_away_from_the_person_it_greeted():
    """The counterfactual one parameter away, same room: raw azimuth at the same poll rate.
    He greets the face that was already centred, then turns away from it — the bearing
    grows, the gate drops him, and the dwell cannot restart while the head keeps moving.
    This is the chase `Greeter._scan` was written to avoid, loose on the follow path."""
    g, head, journal, cues, clock = build(bearing=18.0, face_centering=True, min_spacing_s=0.0, cls=PreFixHook)
    walk(g, head, clock, 8.0)
    assert cues.kinds == ["solo"], "one hello, exactly as much as tonight managed"
    assert head.gaze_yaw_deg == 18.0, head.aims  # 18 deg of LEFT, away from a person 18 deg right
    assert abs(g.tracks[1].azimuth_deg) > BAND_EDGE_DEG, g.tracks  # he is out of the band now
    assert all(t.qualified is False for t in g.tracks.values()), "the dwell restarted, unlike on the lock-on path"


def test_the_pre_fix_hook_never_greets_a_peripheral_face():
    """Same room as `test_a_peripheral_face_is_pulled_into_the_band_and_then_greeted`,
    tonight's code: the person is 30 deg off, `centre_band` refuses them, centring waits
    for a dwell that therefore never starts, and the sweep stands down because a track
    exists. No hello and a head that never moves is exactly tonight's journal."""
    g, head, journal, cues, clock = build(bearing=30.0, face_centering=True, min_spacing_s=0.0, cls=PreFixHook)
    verdicts = [v for _, v in walk(g, head, clock, 6.0)]
    assert cues.cues == [] and J.GREETER_FIRED not in journal.types(), verdicts
    assert head.aims == [], "the pre-fix hook moved the head, which is what this pair pins"
    assert g.tracks and all(t.azimuth_deg > BAND_EDGE_DEG for t in g.tracks.values()), g.tracks


# ---- 3. lock-on owns the head before the dwell, and hands it back when it must ----


def test_the_head_locks_on_at_first_sighting_not_when_the_dwell_closes():
    g, head, _j, _c, clock = build(faces=[face_at(25.0)], face_centering=True, min_spacing_s=0.0)
    assert g.sample(clock.t) == "nobody-looking", "the verdict string is the greet verdict"
    assert head.aims == [(0.0 - FACE_FOLLOW_GAIN * 25.0, 0.0, G.REAIM_HOLD_S, G.FACE_SOURCE)]
    assert g.tracks and not any(t.qualified for t in g.tracks.values())


def test_a_peripheral_face_is_pulled_into_the_band_and_then_greeted():
    """The chain end to end on a person who starts outside the acceptance band — the case
    tonight's build cannot reach at all, and the reason to prefer the ported mechanism."""
    assert BAND_EDGE_DEG == 24.0  # a face further off than this is not a face he can greet
    g, head, journal, cues, clock = build(bearing=30.0, face_centering=True, min_spacing_s=0.0)
    verdicts = walk(g, head, clock, 4.0)
    assert cues.kinds == ["solo"], verdicts
    assert journal.find(J.GREETER_FIRED)[0]["dwell_s"] == 2.0
    assert head.gaze_yaw_deg < -20.0, "he never turned toward him"


def test_a_face_too_weak_to_greet_is_not_chased():
    """Pull-in is not a licence to chase wallpaper: the fallback still demands the gate's
    own detection confidence, and a face too small to ever qualify is left alone."""
    for kw in ({"score": 0.5}, {"share": 0.004}):
        g, head, _j, _c, clock = build(faces=[face_at(30.0, **kw)], face_centering=True)
        assert g.sample(clock.t) == "nobody-looking"
        assert head.aims == [], kw
        assert g._worth_turning_toward(list(g.tracks.values())[0]) is False, kw


def test_lock_on_gives_the_head_to_a_wake_turn_or_a_tool_look_and_does_not_take_it_back():
    """`look_at` has no priority argument: whoever wrote `_gaze_target` last owns the neck.
    A person who just said the wake word is being answered by voice, so the face must not
    steal him back every poll — the rule the sweep already lives by, now applied to faces."""
    for source in ("wake", "doa", "tool"):
        g, head, _j, _c, clock = build(bearing=20.0, face_centering=True, min_spacing_s=0.0)
        head.gaze_source = source
        walk(g, head, clock, 2.0)
        assert head.aims == [], source
        assert head.gaze_yaw_deg == 0.0, source  # the wake turn's target was never overwritten


def test_lock_on_resumes_the_moment_the_holder_lets_go():
    g, head, _j, _c, clock = build(bearing=20.0, face_centering=True, min_spacing_s=0.0)
    head.gaze_source = "wake"  # doa.py's wake turn holds the head for 10 s
    g.sample(clock.t)
    assert head.aims == []  # a wake turn is holding the head
    clock.t += 0.5
    head.gaze_source = "home"  # the hold expired and tick() sprang the gaze home
    g.sample(clock.t)
    assert head.aims[-1][3] == G.FACE_SOURCE


def test_the_sweep_and_the_lock_on_never_both_hold_the_head():
    """Empty room: the sweep pans. A face appears: the sweep stops and the aim takes over,
    so one head has one owner per poll and the azimuth is only ever moved by one of them."""
    g, head, _j, _c, clock = build(face_centering=True, scan=True, min_spacing_s=0.0)
    walk(g, head, clock, 3.0)
    assert [a[3] for a in head.aims] == [G.SCAN_SOURCE] * 3, head.aims
    g.tracker.faces = [face_at(20.0)]  # inert from here: he has found the person
    walk(g, head, clock, 3.0)
    sources = [a[3] for a in head.aims]
    assert sources[-6:] == [G.FACE_SOURCE] * 6, sources
    assert G.SCAN_SOURCE not in sources[3:], "the sweep kept stepping on a face he had locked onto"


def test_nothing_moves_at_all_with_the_knob_off():
    """`REACHY_GREETER_FACE_CENTERING=0` (home) stays byte-identical to before lock-on."""
    g, head, _j, _c, clock = build(bearing=25.0, face_centering=False, min_spacing_s=0.0)
    walk(g, head, clock, 3.0)
    assert head.aims == []


# ---- the ported numbers are named, not folklore ----


def test_the_follow_gains_are_the_legacy_ones():
    """Both come off the bench in `app/`, the code the operator remembers as locking on.
    If either moves, it moves with a measurement and a knobs.md row, not a hunch."""
    assert FACE_FOLLOW_GAIN == 0.6  # app/face_tracker.py OVERSHOOT_SCALE
    assert FACE_FOLLOW_DEADBAND_DEG == 2.5  # app/movement_manager.py FACE_DEAD_ZONE (radians there)
    assert math.isclose(G.REAIM_HOLD_S, 3.0)
    assert G.FOLLOW_SOURCES == ("home", G.SCAN_SOURCE, G.FACE_SOURCE)
