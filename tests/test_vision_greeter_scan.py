"""Proof that the greeter chain can fire at all, plus the idle search sweep.

Two things live in here, and they belong together because they are the same claim:
`shell/vision/greeter.py` has produced **zero** `greeter.fired` rows in production
since it was written, so the risk at the booth is not whether the sweep looks silly —
it is whether detection -> dwell 2.0 s -> lock-on -> greet can fire end to end at all.
This file answers that with a fake tracker, a fake clock and a fake motion owner, and
no camera, no sounddevice, no `ReachyMini()` and no model file (hardware-free like the
rest of tests/, so it runs in the bare `uvx` env CI uses).

The sweep (`REACHY_GREETER_SCAN`, default OFF) is the second half: what he does with an
empty room. Most of its assertions are about when it must NOT move — a live track, a
sighting inside the settle window, a wake turn that already took the head — because
`Face.azimuth_deg` is head-relative and panning under a track is how one person turns
into several.

The golden trace in `test_scan_off_is_byte_identical_to_today` is what makes merging
this safe: with the knob off, the verdict strings, every journal row and every head aim
must be exactly what they were before the knob existed.
"""

from __future__ import annotations

import pytest
from conftest import RecJournal
from test_vision_greeter import Rec, Source, Tracker, face

from shell import journal as J
from shell.realtime.client import ORIGIN_GREETER
from shell.vision import greeter as G
from shell.vision.greeter import SOLO_CUE, Greeter


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class Motion:
    """A `MotionOwner` stand-in that models what the sweep actually negotiates.

    The real owner has one gaze target and one label for it (`gaze_source`), and a
    target set with `hold_s` springs home when the hold expires. That is the whole
    protocol the sweep cares about: it may hold the head while nobody else wants it,
    and it gives it up the moment somebody does — `source="wake"` / `"doa"` from
    `shell/motion/doa.py`, `"tool"` for look_left/right, `"face"` for centring. There
    is no priority argument on `look_at`, so yielding is the only non-fighting option.
    """

    def __init__(self, clock, *, busy=False, sleeping=False):
        self.clock = clock
        self.busy = busy
        self.sleeping = sleeping
        self.gaze_source = "home"
        self.target_deg = 0.0
        self.hold_until: float | None = None
        self.aims: list[tuple[float, float, float, str]] = []

    def look_at(self, yaw_deg, pitch_deg=0.0, *, hold_s=None, source="tool"):
        self.aims.append((round(float(yaw_deg), 2), round(float(pitch_deg), 2), hold_s, source))
        self.gaze_source = source
        self.target_deg = float(yaw_deg)
        self.hold_until = None if hold_s is None else self.clock.t + hold_s

    def tick(self):
        """One control-loop step: an expired hold returns the gaze to home."""
        if self.hold_until is not None and self.clock.t >= self.hold_until:
            self.hold_until = None
            self.gaze_source = "home"
            self.target_deg = 0.0


def make(*, scan=False, faces=None, **kw):
    """(greeter, tracker, motion, clock, journal, cues) — the whole rig, no hardware."""
    clock = kw.pop("clock", None) or Clock()
    tracker = kw.pop("tracker", None) or Tracker()
    motion = kw.pop("motion", None) or Motion(clock)
    journal = kw.pop("journal", None) or RecJournal()
    cues = kw.pop("cues", None) or Rec()
    g = Greeter(
        source=kw.pop("source", None) or Source(),
        tracker=tracker,
        motion=motion,
        journal=journal,
        on_greet=cues,
        clock=clock,
        scan=scan,
        **kw,
    )
    if faces is not None:
        tracker.faces = list(faces)
    return g, tracker, motion, clock, journal, cues


def pump(g, tracker, motion, clock, seconds, *, step=0.5, faces=()):
    """Sample a fixed set of detections for `seconds` of fake clock, ticking the fake
    motion owner between samples like the 50 Hz loop would. `clock.t` ends on the first
    unsampled instant, so back-to-back pumps never sample one instant twice."""
    tracker.faces = list(faces)
    end = clock.t + seconds
    verdict = "nobody-looking"
    while clock.t <= end + 1e-9:
        motion.tick()
        verdict = g.sample(clock.t)
        clock.t += step
    motion.tick()
    return verdict


def scan_aims(motion):
    return [a for a in motion.aims if a[3] == G.SCAN_SOURCE]


# ============ the chain: can he actually greet? ============


def test_the_chain_fires_end_to_end_and_the_hello_is_a_greeter_transcript():
    """The one that matters tonight: detection -> dwell 2.0 s -> lock-on -> greet,
    through the real `greet_hook` and the real verdict, with the shipped booth numbers."""
    loop, client = Loop(), Client()
    g, tracker, motion, clock, journal, _cues = make(
        journal=RecJournal(),
        face_centering=True,
        dwell_s=2.0,
        min_spacing_s=120.0,
        max_per_hour=6,
    )
    g.on_greet = G.greet_hook(client, loop, journal=journal)
    tracker.faces = [face(azimuth=15.0, share=0.05)]
    for t in (0.0, 0.5, 1.0, 1.5):
        assert g.sample(t) == "nobody-looking"
        assert loop.fns == [], "nothing may be queued before the dwell closes"
    assert g.sample(2.0) == "fire:solo"

    assert loop.drain() == 1
    assert client.transcripts == [(SOLO_CUE, ORIGIN_GREETER)]
    assert ORIGIN_GREETER == "greeter", "the shell refuses the lease exchange on this string"
    assert client.leases == [], "a face buys a hello, not a window into the room (ADR 0003)"

    assert journal.find(J.GREETER_FIRED) == [
        {"kind": "solo", "track": 1, "dwell_s": 2.0, "azimuth_deg": 15.0, "area_share": 0.05}
    ]
    assert journal.types() == [J.FACE_PRESENT, J.GREETER_FIRED]
    # lock-on runs on the way to the verdict: the head turns toward the face it dwelled on.
    # Negative because `Face.azimuth_deg` is positive to the right of the image and
    # `MotionOwner.look_at` is positive to the robot's LEFT (`look_left` sends +30,
    # `look_right` -30: shell/robot_tools.py, tests/test_robot_tools.py:205-206), and
    # 0.6 of the way because of `FACE_FOLLOW_GAIN`. See tests/test_vision_greeter_lockon.py.
    assert motion.aims[0] == (-9.0, 0.0, G.REAIM_HOLD_S, "face")


def test_dwell_under_the_gate_leaves_the_room_ungreeted_but_journalled():
    loop, client = Loop(), Client()
    g, tracker, _motion, clock, journal, _cues = make(dwell_s=2.0)
    g.on_greet = G.greet_hook(client, loop)
    tracker.faces = [face()]
    for t in (0.0, 0.5, 1.0, 1.5):  # 1.5 s of looking, then they walk off
        assert g.sample(t) == "nobody-looking"
    tracker.faces = []
    assert g.sample(2.0) == "nobody-looking"
    assert client.transcripts == [] and loop.fns == []
    assert J.GREETER_FIRED not in journal.types()
    assert len(journal.find(J.FACE_PRESENT)) == 1, "a face was there; the log should say so"


def test_the_head_centres_a_face_he_then_declines_to_greet():
    """Verified, not wished: `_center` runs before `_verdict`, so spacing suppression
    does not stop the turn. Pinned because it is the one motion he makes for somebody
    he will not say hello to — the operator should know that is what he is watching."""
    g, tracker, motion, clock, _journal, cues = make(face_centering=True, min_spacing_s=120.0)
    g._last_any = 0.0  # a greeting landed one second ago
    tracker.faces = [face(azimuth=20.0)]
    for t in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        verdict = g.sample(t)
    assert verdict == "suppressed:spacing"
    assert cues.cues == []
    assert motion.aims[0] == (-12.0, 0.0, G.REAIM_HOLD_S, "face")


def test_120s_spacing_blocks_the_next_person_and_the_hourly_cap_is_the_backstop():
    g, tracker, _motion, _clock, journal, cues = make(min_spacing_s=120.0, max_per_hour=6)
    pump(g, tracker, _motion, _clock, 2.5, faces=[face()])
    assert cues.kinds == ["solo"]

    # Person two, 40 s later: the global spacing owns it, and the journal says which
    _clock.t = 10.0
    pump(g, tracker, _motion, _clock, 0.5)  # the room empties
    _clock.t = 10.5
    pump(g, tracker, _motion, _clock, 2.0, faces=[face(azimuth=-40.0)])
    assert cues.kinds == ["solo"], "spacing let a second hello through"
    assert journal.find(J.GREETER_SUPPRESSED)[-1]["reason"] == "spacing"

    # Six hellos in the hour is the ceiling; the seventh is refused, and the window moves
    capped = Greeter(max_per_hour=6)
    capped._fired.extend([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    assert capped._cap_reached(60.0) is True
    assert capped._cap_reached(3700.0) is False


class Loop:
    """The asyncio-loop seam `greet_hook` marshals onto."""

    def __init__(self):
        self.fns = []

    def call_soon_threadsafe(self, fn):
        self.fns.append(fn)

    def drain(self):
        out, self.fns = list(self.fns), []
        for fn in out:
            fn()
        return len(out)


class Client:
    """The realtime client seam: the greeting is one transcript, nothing else."""

    def __init__(self):
        self.transcripts: list[tuple[str, str]] = []
        self.leases: list[str] = []

    def insert_user_transcript(self, text, *, origin="user"):
        self.transcripts.append((text, origin))

    def acquire(self):  # pragma: no cover - must never be reached
        self.leases.append("acquire")


# ============ the sweep: what an empty room gets ============


def test_scan_is_off_by_default_and_moves_nothing_when_off():
    assert Greeter().scan is False
    g, tracker, motion, clock, journal, _cues = make()
    pump(g, tracker, motion, clock, 20.0)
    assert motion.aims == [], "the sweep moved the head with the knob off"
    assert J.MOTION_SCAN not in journal.types()


def test_an_empty_room_starts_sweeping_from_dead_ahead():
    g, _tracker, motion, clock, _journal, _cues = make(scan=True)
    assert g.sample(clock.t) == "nobody-looking"
    assert motion.aims[0] == (-10.0, 0.0, G.SCAN_HOLD_S, G.SCAN_SOURCE)  # left first, 10 deg
    assert motion.gaze_source == G.SCAN_SOURCE


def test_the_sweep_is_10_degree_detents_clamped_at_35_degrees():
    g, tracker, motion, clock, _journal, _cues = make(scan=True)
    pump(g, tracker, motion, clock, 24.0)
    yaw = [a[0] for a in scan_aims(motion)]
    assert yaw[0] == -10.0
    assert all(abs(b - a) <= G.SCAN_STEP_DEG + 1e-9 for a, b in zip(yaw, yaw[1:])), yaw
    assert min(yaw) == -35.0 and max(yaw) == 35.0, yaw  # the clamp is what turns him around
    assert all(abs(a) <= g.scan_limit_deg for a in yaw), yaw
    assert all(a[2] == G.SCAN_HOLD_S for a in scan_aims(motion)), "the hold must outlive the step"


def test_the_step_rate_does_not_become_the_poll_rate():
    """2 FPS idle / 3 FPS engaged must not turn into 2 or 3 degrees per second: at
    least one second per detent, whatever `sample()` is being called at."""
    for cadence in (0.25, 0.5, 1.0):
        g, tracker, motion, clock, _journal, _cues = make(scan=True)
        pump(g, tracker, motion, clock, 12.0, step=cadence)
        assert len(scan_aims(motion)) <= 12.0 / max(1.0, g.scan_step_s) + 1, (cadence, motion.aims)


def test_a_live_track_stops_the_sweep_because_azimuth_is_head_relative():
    """The hazard, designed around: panning under a track rewrites the bearing
    `TRACK_MATCH_AZIMUTH` matches on, so one person becomes several dwell-less tracks
    and the head chases an azimuth its own turn invented. A face that is in frame but
    not looking at him is exactly this case: no candidate, but a live track."""
    g, tracker, motion, clock, _journal, _cues = make(scan=True)
    away = [face(yaw=40.0)]
    assert pump(g, tracker, motion, clock, 4.0, faces=away) == "nobody-looking"
    assert g.tracks, "the test's premise: a track exists with no candidate"
    assert scan_aims(motion) == []
    assert pump(g, tracker, motion, clock, 4.0, faces=away) == "nobody-looking"
    assert scan_aims(motion) == [], "panned the whole life of a track"


def test_the_sweep_waits_for_the_settle_after_the_last_sighting():
    """A track dies 1.5 s after the last sighting and the settle is longer than that:
    the sweep must not start while a just-left presence could still be matched."""
    g, tracker, motion, clock, _journal, _cues = make(scan=True)
    pump(g, tracker, motion, clock, 0.5, faces=[face(yaw=40.0)])  # seen at t=0.0, 0.5
    pump(g, tracker, motion, clock, g.scan_settle_s - 1.0)  # room empties, settle still running
    assert scan_aims(motion) == [], "panned inside the settle window"
    pump(g, tracker, motion, clock, 1.0)
    assert scan_aims(motion), "the sweep never resumed once the room was properly empty"


def test_the_sweep_stops_the_instant_a_candidate_appears():
    g, tracker, motion, clock, _journal, _cues = make(scan=True, face_centering=True, min_spacing_s=0.0)
    pump(g, tracker, motion, clock, 6.0)
    before = len(scan_aims(motion))
    assert before >= 4
    pump(g, tracker, motion, clock, 3.0, faces=[face(azimuth=25.0)])
    assert len(scan_aims(motion)) == before, "the sweep kept stepping on a face it had locked onto"
    assert any(a[3] == "face" for a in motion.aims)


def test_a_wake_or_doa_turn_owns_the_head_and_the_sweep_yields_rather_than_fighting():
    g, tracker, motion, clock, _journal, _cues = make(scan=True)
    motion.gaze_source, motion.hold_until = "wake", clock.t + 5.0  # doa.py's wake turn holds it
    pump(g, tracker, motion, clock, 4.0)
    assert scan_aims(motion) == [], "the sweep stole the head back from the wake turn"
    pump(g, tracker, motion, clock, 4.0)  # the hold expires, gaze returns home, search resumes
    assert scan_aims(motion)


def test_an_exclusive_move_or_the_nap_stands_the_sweep_down():
    for kw in ({"busy": True}, {"sleeping": True}):
        clock = Clock()
        g, tracker, _m, _c, _j, _cue = make(scan=True, motion=Motion(clock, **kw), clock=clock)
        pump(g, tracker, _m, clock, 5.0)
        assert scan_aims(_m) == [], kw


def test_the_sweep_does_not_run_while_he_is_busy_with_a_person():
    """Asleep, speaking, holding a lease or owed a reply: the same fail-closed list
    `_verdict` uses. An aware-mode search is for an empty room, not a tic mid-turn."""
    for kw in (
        {"is_playing": lambda: True},
        {"lease_active": lambda: True},
        {"awaiting_reply": lambda: True},
    ):
        g, tracker, motion, clock, _journal, _cues = make(scan=True, **kw)
        pump(g, tracker, motion, clock, 5.0)
        assert scan_aims(motion) == [], kw


def test_a_sweep_journals_one_rate_limited_motion_row_for_the_booth_card():
    g, tracker, motion, clock, journal, _cues = make(scan=True)
    pump(g, tracker, motion, clock, 90.0)
    rows = journal.find(J.MOTION_SCAN)
    assert rows, "the booth card needs to see that he is looking around"
    assert len(rows) <= 90.0 / G.SCAN_JOURNAL_S + 1, rows
    assert set(rows[0]) == {"azimuth_deg", "direction", "limit_deg", "hold_s"}
    assert rows[0]["direction"] in ("left", "right")
    assert J.MOTION_SCAN == "motion.scan"


def test_the_sweep_survives_no_motion_owner_and_a_raising_one():
    clock = Clock()
    g = Greeter(source=Source(), tracker=Tracker(), scan=True, clock=clock)
    for t in (0.0, 1.0, 2.0):
        assert g.sample(t) == "nobody-looking"  # motion=None: nothing to move, nothing raised

    class BadMotion(Motion):
        def look_at(self, *a, **kw):
            raise RuntimeError("daemon is gone")

    g, tracker, motion, clock, _journal, _cues = make(scan=True, motion=BadMotion(clock=Clock()))
    assert pump(g, tracker, motion, clock, 3.0) == "nobody-looking"  # the poll thread outlives it


# ============ the contract that makes merging this safe ============

# One synthetic room, replayed: empty, a passer-by who never looks, a face that dwells
# and is greeted, the greeting spent, the presence ending, then a second person arriving
# inside the 120 s spacing window — every branch the booth walks through in a minute.
SCRIPT = [
    (0.0, []),
    (0.5, []),
    (1.0, [face(yaw=40.0)]),  # in frame, head turned away: never qualifies
    (1.5, [face(yaw=40.0)]),
    (2.0, [face(azimuth=12.0, share=0.05)]),  # they turn toward him
    (2.5, [face(azimuth=12.0, share=0.05)]),
    (3.0, [face(azimuth=12.0, share=0.05)]),
    (3.5, [face(azimuth=12.0, share=0.05)]),
    (4.0, [face(azimuth=12.0, share=0.05)]),
    (4.5, [face(azimuth=12.0, share=0.05)]),
    (5.0, [face(azimuth=12.0, share=0.05)]),
    (5.5, []),  # they leave
    (8.0, []),
    (11.0, [face(azimuth=-35.0, share=0.06)]),  # a different person, inside the spacing
    (11.5, [face(azimuth=-35.0, share=0.06)]),
    (13.5, [face(azimuth=-35.0, share=0.06)]),
]


def replay(**kw):
    """Run SCRIPT through one greeter and record everything an operator could read."""
    clock, tracker, journal, cues = Clock(), Tracker(), RecJournal(), Rec()
    motion = Motion(clock)
    g = Greeter(
        source=Source(),
        tracker=tracker,
        motion=motion,
        journal=journal,
        on_greet=cues,
        clock=clock,
        min_spacing_s=120.0,
        **kw,
    )
    verdicts = []
    for t, faces in SCRIPT:
        clock.t = t
        tracker.faces = list(faces)
        motion.tick()
        verdicts.append(g.sample(t))
    return {
        "verdicts": verdicts,
        "journal": [(type_, payload) for type_, payload in journal.events],
        "aims": list(motion.aims),
        "cues": cues.texts,
    }


# Recorded from `main` with the sweep nonexistent, i.e. exactly what tonight's booth
# does: the greet fires at 4.0 s (two seconds of looking after the passer-by turns
# toward him), the same presence is never greeted twice, the next person inside the
# spacing window gets nothing, and the only head movement is the live lock-on below.
GOLDEN_OFF = {
    "verdicts": [
        "nobody-looking",
        "nobody-looking",
        "nobody-looking",
        "nobody-looking",
        "nobody-looking",
        "nobody-looking",
        "nobody-looking",
        "nobody-looking",
        "fire:solo",
        "suppressed:presence-greeted",
        "suppressed:presence-greeted",
        "suppressed:presence-greeted",
        "nobody-looking",
        "nobody-looking",
        "nobody-looking",
        "suppressed:spacing",
    ],
    "journal": [
        (J.FACE_PRESENT, {"track": 1, "azimuth_deg": 12.0, "area_share": 0.05}),
        (J.GREETER_FIRED, {"kind": "solo", "track": 1, "dwell_s": 2.0, "azimuth_deg": 12.0, "area_share": 0.05}),
        (J.GREETER_SUPPRESSED, {"reason": "presence-greeted", "track": 1, "dwell_s": 2.5}),
        (J.FACE_PRESENT, {"track": 2, "azimuth_deg": -35.0, "area_share": 0.06}),
    ],
    "aims": [],
    "cues": [SOLO_CUE],
}

# `behaviour.greeter_face_centering: true` is what hackday actually runs: the head locks on
# the face and keeps tracking it. The numbers are the lock-on command, not the raw azimuth:
# `-0.6 * azimuth` (sign = the motion owner's positive-left convention, gain = the legacy
# OVERSHOOT_SCALE) added to where the head already points, which this fake never moves.
# The first two rows are the passer-by: an unqualified face is still worth turning toward,
# and at azimuth 0.0 the correction is inside the deadband, so the aim is where the head
# already is (a hold refresh, no movement). Seven rows for the greeted person because the
# aim now runs from the first sighting, not from the close of the dwell; the three at +21.0
# are the second person on the OTHER side of the frame, and `suppressed:spacing` never
# stopped the head.
GOLDEN_CENTERING = dict(
    GOLDEN_OFF,
    aims=[(0.0, 0.0, 3.0, "face")] * 2 + [(-7.2, 0.0, 3.0, "face")] * 7 + [(21.0, 0.0, 3.0, "face")] * 3,
)


def test_scan_off_is_byte_identical_to_today():
    """OFF must be the previous commit's behaviour: same verdict strings, same journal
    rows and same head aims — for the plain greeter and for the lock-on booth config."""
    off = replay(scan=False)
    assert off == replay(), "the dataclass default is not off"
    assert off["aims"] == [], "no head movement whatsoever with the sweep off"
    assert J.MOTION_SCAN not in [t for t, _ in off["journal"]]
    assert off == GOLDEN_OFF, off
    centred = replay(scan=False, face_centering=True)
    assert centred == GOLDEN_CENTERING, centred["aims"]


def test_scan_on_adds_sweep_rows_and_no_greeting_change():
    """The other half of the byte-identity claim: turning the sweep on must not alter
    any verdict, greeting, lock-on aim or existing journal row — it only adds its own."""
    off, on = replay(scan=False), replay(scan=True)
    assert off["verdicts"] == on["verdicts"] == GOLDEN_OFF["verdicts"]
    assert off["cues"] == on["cues"] == [SOLO_CUE]
    assert [(t, p) for t, p in on["journal"] if t != J.MOTION_SCAN] == off["journal"]

    centred = replay(scan=True, face_centering=True)
    assert [a for a in centred["aims"] if a[3] == "face"] == GOLDEN_CENTERING["aims"]
    assert [a for a in centred["aims"] if a[3] == G.SCAN_SOURCE], "the sweep never ran in the booth config"
