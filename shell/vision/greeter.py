"""The proactive greeter: a dwell gate, a rate limiter, and a stage direction.

One daemon poll thread (mirroring shell/motion/doa.py) reading the one camera
owner, nothing else. It owns no device, opens no audio, writes nothing to disk,
and holds no reference to the attention lease: a face is *never* attention
(ADR 0003). What it emits is a `GreetCue`, and the shell turns that into one
`insert_user_transcript` call on the asyncio loop, so the hello is the persona
speaking rather than a sentence we pasted in — a prerendered wav cannot vary by
room, and the hackday persona is where the "welcome to hack day" wording lives
(docs/environments.md hard rule 2: a greeting is speech, not a state change, so
it needs no `tool_confidence` and grants nothing).

Identity is geometric: `Track` matches on rough azimuth plus bbox size plus
dwell, so "one greeting per person" means one greeting per continuous presence.
Nobody is remembered between visits because nothing here is persistable — when
they leave the frame the track expires and with it the only thing we knew about
them (docs/environments.md B1 is recognition, and it is explicitly not this).

Cost, from docs/pins.yaml `face_detection`: detection-only is 2.9-3.7 ms/frame at
320 px, so 2 FPS idle / 3 FPS engaged is 0.6-0.9 % of one core on inference alone
(the plan's 2.4 %-of-a-core figure is the recognition pipeline, which this does not
run). The frame read, the warm-up and the camera open are honest gaps in that
number until measured with the hardware attached - see pins `UNMEASURED PENDING BENCH`.
The thread cap is per onnxruntime session (vision/tracker.py); `OMP_NUM_THREADS`
is the ADR 0006 landmine and is never set here.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .. import journal as J
from ..journal import JournalLike
from ..realtime.client import ORIGIN_GREETER
from .gaze import Gate, looking_at_me
from .tracker import Face

log = logging.getLogger("shell.vision.greeter")

# The two lines the persona is handed. Stage directions, not fake user speech:
# the cue says what is true and lets the environment's persona find the words.
SOLO_CUE = "*(someone across from you is looking at you and waiting - say a short hello, one sentence, no pitch)*"
GROUP_CUE = (
    "*(a few more people just walked up and are looking at you - briefly notice them, "
    'like "hey, I noticed there are a few more people here", then carry on)*'
)

REAIM_HOLD_S = 3.0  # lock-on target lifetime, refreshed every poll while a face is live
TRACK_LOST_S = 1.5  # unseen this long and the presence is over (identity evaporates)

# Lock-on gains, ported from the legacy loop rather than invented here. Both were
# bench-numbered in `app/face_tracker.py` / `app/movement_manager.py`, which is the
# code the operator remembers as "locking on":
#   FACE_FOLLOW_GAIN      = OVERSHOOT_SCALE (app/face_tracker.py:13, used at :153) —
#     aim 60% of the way to the face, so the re-aim converges instead of overshooting
#     into a wobble, and the per-poll head move stays inside `TRACK_MATCH_AZIMUTH`.
#   FACE_FOLLOW_DEADBAND  = FACE_DEAD_ZONE (app/movement_manager.py:20, 2.5 deg) —
#     inside 2.5 deg of the aim there is nothing to correct: refresh the hold, do not
#     twitch. At 3 FPS an undamped follow never stops nudging (0.6^n of the original
#     offset, forever) and the servos pay for every sub-degree of it (ADR 0006).
FACE_FOLLOW_GAIN = 0.6
FACE_FOLLOW_DEADBAND_DEG = 2.5
FACE_SOURCE = "face"  # MotionOwner gaze_source for the lock-on (knobs.md names it)

# The idle search sweep (`REACHY_GREETER_SCAN`, default OFF). The servos have had
# about one minute of powered time in their life, so every number here is the
# conservative one, and the whole feature is gated behind `Greeter.scan`.
SCAN_LIMIT_DEG = 35.0  # clamp: well inside motion's MAX_YAW_DEG (60)
SCAN_STEP_DEG = 10.0  # one detent per step, so the sweep is a pan, not a saccade
SCAN_STEP_S = 1.2  # >= 1 s per step: the spring (GAZE_HALFLIFE 0.35) lands each detent
SCAN_SETTLE_S = 3.0  # quiet after the last sighting before the sweep may move again
SCAN_HOLD_S = 2.5  # > SCAN_STEP_S, so the target never expires between detents
SCAN_JOURNAL_S = 30.0  # the booth card wants "he is looking around", not 1 row/step
SCAN_SOURCE = "scan"  # MotionOwner gaze_source; the bottom of the stack (see `_scan`)
TRACK_GRACE_S = 0.6  # one dropped poll must not reset a dwell (2 FPS is 0.5 s apart)
TRACK_MATCH_AZIMUTH = 18.0  # deg: coarse, because a person leans
TRACK_MATCH_SHARE = 0.75  # multiplicative tolerance on bbox size

# The gaze sources the lock-on may take the head from. Anything else — "wake"/"doa"
# from shell/motion/doa.py, "tool" from shell/robot_tools.py — is somebody already
# being answered with the head, and `MotionOwner.look_at` has no priority argument,
# so yielding is the only way not to fight them (the rule `SCAN_SOURCE` already lives
# by; see `_center`).
FOLLOW_SOURCES = ("home", SCAN_SOURCE, FACE_SOURCE)


@dataclass(frozen=True)
class GreetCue:
    """What the greeter hands the shell. `yaw_deg` is the head-relative bearing the
    track was matched on — telemetry for the bench console, not an aim: the head is
    already on this person by the time the cue is built (`_center`)."""

    text: str
    kind: str  # "solo" | "group"
    track_id: int
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0


@dataclass
class Track:
    """One continuous presence: azimuth + bbox size + dwell, no face signature."""

    id: int
    first_seen: float
    last_seen: float
    qualified_since: float | None = None
    greeted: bool = False
    azimuth_deg: float = 0.0
    share: float = 0.0
    margins: dict[str, float] = field(default_factory=dict)

    @property
    def dwell_s(self) -> float:
        return 0.0 if self.qualified_since is None else self.last_seen - self.qualified_since

    @property
    def qualified(self) -> bool:
        return self.qualified_since is not None

    def matches(self, azimuth_deg: float, share: float) -> bool:
        """Coarse geometric identity: same bearing, comparable size. A person who
        walks off and back is a new track, which is exactly the intent."""
        same_bearing = abs(azimuth_deg - self.azimuth_deg) <= TRACK_MATCH_AZIMUTH
        same_size = (
            self.share <= 0 or share <= 0 or max(share / self.share, self.share / share) <= 1.0 / TRACK_MATCH_SHARE
        )
        return same_bearing or (same_size and abs(azimuth_deg - self.azimuth_deg) <= 2 * TRACK_MATCH_AZIMUTH)


@dataclass
class Greeter:
    """Dwell gate + rate limiter + the head lock-on + (off by default) the idle search
    sweep, in that order of dependency: the sweep finds a face, `_center` keeps the head
    on it, the dwell decides whether to speak. `sample(now)` is the whole policy, callable
    from a test with a fake clock; `start()` only calls it on a thread."""

    # The collaborators are duck-typed on purpose: the greeter must be assemblable
    # without a camera, a model or a robot (see the degrade tests), and shell/camera.py
    # imports nothing from here.
    source: Any = None  # shell.camera.CameraStream | CameraGrabber: latest()/grab()
    tracker: Any = None  # vision.tracker.FaceTracker: detect(frame) -> list[Face]
    journal: JournalLike | None = None
    on_greet: Callable[[GreetCue], None] | None = None
    motion: Any = None  # shell.motion.MotionOwner; only ever asked to look_at
    enabled: bool = True
    face_centering: bool = False  # lock-on: keep the head aimed at a face (see `_center`)
    scan: bool = False  # the idle search sweep, off until it has been watched on the bench
    is_playing: Callable[[], bool] = lambda: False
    asleep: Callable[[], bool] = lambda: False  # while napping: no frames, no inference at all
    lease_active: Callable[[], bool] = lambda: False
    awaiting_reply: Callable[[], bool] = lambda: False
    gate: Gate = field(default_factory=Gate)
    clock: Callable[[], float] = time.monotonic
    dwell_s: float = 2.0  # David's call: about two seconds of looking, no less
    scan_limit_deg: float = SCAN_LIMIT_DEG  # sweep clamp, either side of dead ahead
    scan_step_deg: float = SCAN_STEP_DEG
    scan_step_s: float = SCAN_STEP_S
    scan_settle_s: float = SCAN_SETTLE_S
    min_spacing_s: float = 120.0  # global: never nag the room (docs/environments.md)
    group_min_spacing_s: float = 300.0  # the in-conversation arrival notice is rarer still
    max_per_hour: int = 6
    idle_fps: float = 2.0
    engaged_fps: float = 3.0
    debug: bool = False
    frame_shape: tuple[int, int] = (720, 1280)  # what shell/camera.py grabs at

    def __post_init__(self) -> None:
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self._fired: deque[float] = deque()
        self._last_any = -1e9
        self._last_group = -1e9
        self._last_journal = -1e9
        self._last_sighting = -1e9  # last sample that detected any face (the sweep's settle clock)
        self._scan_deg = 0.0  # last absolute head yaw the sweep commanded
        self._scan_dir = -1.0  # ping-pong direction, so the search starts by looking left
        self._scan_next = -1e9  # earliest next detent
        self._last_scan_journal = -1e9
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned_no_frames = False
        self._tracker_failures = 0
        self._tracker_dead = False  # P2-E: a broken tracker logs once, then stands down

    # ---- the poll ----

    def sample(self, now: float | None = None) -> str:
        """One read + verdict. Returns a short reason string for tests and the log."""
        now = self.clock() if now is None else now
        if not self.enabled:
            return "disabled"
        if self.asleep():
            return self._quiet("asleep")  # the wake word owns the way back up, not the camera
        frame = self._read()
        if frame is None:
            self._expire(now)
            return self._quiet("no-camera")
        live = self._update(now, self._detect(frame))
        # ~3 FPS while somebody qualified is in frame, ~2 FPS on an empty room
        # (docs/pins.yaml `face_detection`: 3.7 ms/frame detect-only either way).
        self._set_engaged(any(t.qualified for t in self.tracks.values()))
        cand = self._candidate(now)
        # Lock-on runs on what is in frame *this poll*, before the dwell gate closes:
        # turning toward a face is what brings it inside the centre band, and the
        # dwell gate decides who gets spoken to, not who gets looked at.
        aim = self._aim_target(live) if self.face_centering else None
        if cand is None and aim is None:
            if self.debug:
                self._log_faces(self.tracks.values())
            self._scan(now)  # the only caller: an empty room is the only thing to search
            return self._quiet("nobody-looking")
        self._end_scan()  # a face owns the head from this instant, so the sweep stands down
        if self.debug:
            self._log_faces([cand] if cand is not None else [aim])
        if aim is not None:
            self._center(aim)
        if cand is None:
            # Lock-on, no dwell yet. The verdict string is the *greeting* verdict and
            # part of the bench contract (`scripts/smoke_greeter.py`, `LOG_ONLY_REASONS`
            # in `scripts/bench_greeter.py`), so it stays what it has always been here.
            return self._quiet("nobody-looking")
        verdict = self._verdict(now, cand, live)
        if verdict == "fire:solo" or verdict == "fire:group":
            self._fire(now, cand, live, kind=verdict.split(":", 1)[1])
        else:
            self._suppress(now, verdict, cand)
        return verdict

    # ---- the policy ----

    def _verdict(self, now: float, cand: Track, live: list[Track]) -> str:
        if self.is_playing():
            return "suppressed:speaking"  # never talk over his own voice
        if self._cap_reached(now):
            return "suppressed:hourly-cap"
        if cand.greeted:
            return "suppressed:presence-greeted"
        if now - self._last_any < self.min_spacing_s:
            return "suppressed:spacing"
        if self.lease_active() or self.awaiting_reply():
            # Mid-conversation he only notices people *arriving* — and only when he
            # can count more than the one person he is already talking to.
            if len([t for t in live if t.qualified and not t.greeted]) < 2:
                return "suppressed:conversation"
            if now - self._last_group < self.group_min_spacing_s:
                return "suppressed:group-spacing"
            return "fire:group"
        return "fire:solo"

    def _cap_reached(self, now: float) -> bool:
        while self._fired and now - self._fired[0] > 3600.0:
            self._fired.popleft()
        return len(self._fired) >= self.max_per_hour

    def _fire(self, now: float, cand: Track, live: list[Track], *, kind: str) -> None:
        cue = GreetCue(
            text=GROUP_CUE if kind == "group" else SOLO_CUE,
            kind=kind,
            track_id=cand.id,
            yaw_deg=cand.azimuth_deg,
        )
        self._fired.append(now)
        self._last_any = now
        if kind == "group":
            self._last_group = now
        cand.greeted = True
        for track in live:
            if kind == "group" and track.qualified:
                track.greeted = True  # the arrivals were all acknowledged together
        if self.journal is not None:
            self.journal.write(
                J.GREETER_FIRED,
                kind=kind,
                track=cand.id,
                dwell_s=round(cand.dwell_s, 2),
                azimuth_deg=round(cand.azimuth_deg, 1),
                area_share=round(cand.share, 4),
            )
        log.info("greeter fired %s on track %d after %.1fs dwell", kind, cand.id, cand.dwell_s)
        if self.on_greet is None:
            return
        try:
            self.on_greet(cue)
        except Exception as e:  # a greeting must never take the poll thread down
            log.warning("greet hook failed: %r", e)

    def _suppress(self, now: float, verdict: str, cand: Track) -> None:
        reason = verdict.split(":", 1)[1]
        if self.debug:
            log.info("greeter suppressed %s dwell=%.1fs %s", reason, cand.dwell_s, cand.margins)
        if self.journal is None:
            return
        # Rate-limited, not skipped: "a person dwelled and we said nothing" is the
        # row the bench pass needs, and 2 FPS of near-misses would flush the ring.
        if now - self._last_journal < 10.0:
            return
        self._last_journal = now
        self.journal.write(J.GREETER_SUPPRESSED, reason=reason, track=cand.id, dwell_s=round(cand.dwell_s, 2))

    # ---- tracking ----

    def _update(self, now: float, faces: list[Face]) -> list[Track]:
        seen: set[int] = set()
        if faces:
            self._last_sighting = now  # the sweep's settle clock runs on sightings, not tracks
        for face in faces:
            engaged = any(t.qualified and t.matches(face.azimuth_deg, face.share) for t in self.tracks.values())
            ok, margins = looking_at_me(
                face.kps, face.bbox, self.frame_shape, face.det_score, gate=self.gate, engaged=engaged
            )
            track = self._assign(face, now)
            if ok:
                track.qualified_since = now if track.qualified_since is None else track.qualified_since
                if track.qualified_since == now and self.journal is not None:
                    # A face is a fact about the room, not an attention grant: this row
                    # exists so "why did he not say hello" is answerable tomorrow.
                    self.journal.write(
                        J.FACE_PRESENT,
                        track=track.id,
                        azimuth_deg=round(face.azimuth_deg, 1),
                        area_share=round(face.share, 4),
                    )
            else:
                track.qualified_since = None
            track.greeted = track.greeted and track.qualified
            track.last_seen = now
            track.azimuth_deg = face.azimuth_deg
            track.share = face.share
            track.margins = margins
            seen.add(track.id)
        self._expire(now, keep=seen)
        return [self.tracks[i] for i in sorted(seen)]

    def _assign(self, face: Face, now: float) -> Track:
        best: tuple[float, Track] | None = None
        for track in self.tracks.values():
            if now - track.last_seen > TRACK_LOST_S + TRACK_GRACE_S:
                continue
            if not track.matches(face.azimuth_deg, face.share):
                continue
            gap = now - track.last_seen
            if best is None or gap < best[0]:
                best = (gap, track)
        if best is not None:
            return best[1]
        track = Track(id=self._next_id, first_seen=now, last_seen=now, share=face.share, azimuth_deg=face.azimuth_deg)
        self._next_id += 1
        self.tracks[track.id] = track
        return track

    def _expire(self, now: float, keep: set[int] | None = None) -> None:
        """An ended presence is forgotten outright: no store, no cross-visit memory."""
        keep = keep or set()
        for track_id in [i for i, t in self.tracks.items() if i not in keep and now - t.last_seen > TRACK_LOST_S]:
            del self.tracks[track_id]

    def _candidate(self, now: float) -> Track | None:
        ready = [t for t in self.tracks.values() if t.qualified and t.dwell_s >= self.dwell_s]
        if not ready:
            return None
        return max(ready, key=lambda t: (t.dwell_s, t.share))

    def _aim_target(self, live: list[Track]) -> Track | None:
        """Who gets the head this poll — the lock-on half of the chain, and deliberately
        not the dwell gate.

        `live` is the tracks seen on this sample, because the aim is a thing you do to a
        face you can currently see (a track in its `TRACK_LOST_S` grace still has a dwell,
        but there is nothing in frame to aim at). Prefer a qualified presence, largest
        dwell first. With nobody qualified, aim at the biggest face that is *real and near
        enough to greet* — every part of `gaze.looking_at_me` except the angles and the
        centre band, i.e. exactly the parts turning the head cannot fix. That pull-in is
        the cheapest legacy mechanism (`app/face_tracker.py` aimed at the largest detection
        and nothing else), and it is what makes a peripheral person reachable at all:
        `centre_band` refuses any face more than ~24° off dead ahead, and the journal cannot
        say how many those are — `face.present` is only written for a face that already
        passed that band, so the whole population outside it is invisible by construction.
        It is also what stops the search dead-ing on somebody he can see but cannot read:
        `_scan` stands down while any track lives, so with no aim for an unqualified face
        there is neither a turn nor a sweep until that person moves.
        """
        ready = [t for t in live if t.qualified]
        if ready:
            return max(ready, key=lambda t: (t.dwell_s, t.share))
        near = [t for t in live if self._worth_turning_toward(t)]
        return max(near, key=lambda t: t.share) if near else None

    def _worth_turning_toward(self, track: Track) -> bool:
        """A real face at greeting distance, judged on the parts of the gate the head
        cannot fix: detection confidence and size. The area floor is the most relaxed
        one the gate will ever apply (`Gate.relaxed`), so a face too small to be
        qualified is never chased across the room."""
        return (
            track.margins.get("det_score", 0.0) >= self.gate.min_det_score
            and self.gate.min_area_share * self.gate.hysteresis <= track.share <= self.gate.max_area_share
        )

    # ---- frames ----

    def _read(self) -> Any:
        """Latest shared frame, else one grab. Never raises: no camera, the daemon
        holding the device, or REACHY_CAMERA=off all degrade to `None`, and only the
        greeter notices — the rest of the shell keeps running."""
        if self.source is None:
            return None
        try:
            latest = getattr(self.source, "latest", None)
            frame = latest() if callable(latest) else None
            if frame is not None:
                return frame
            grab = getattr(self.source, "grab", None)
            return grab() if callable(grab) else None
        except Exception as e:
            if self.debug:
                log.info("greeter has no frames: %r", e)
            return None

    def _detect(self, frame: object) -> list[Face]:
        if self.tracker is None or self._tracker_dead:
            return []
        try:
            faces = list(self.tracker.detect(frame) or [])
        except Exception as e:
            # Import/init failures raise every sample. Three in a row is not a
            # bad frame: log once, stand down, and let the (now inference-free)
            # poll idle instead of warning at 2 FPS forever (ADR 0006).
            self._tracker_failures += 1
            if self._tracker_failures >= 3:
                self._tracker_dead = True
                log.warning("face detection stood down after %d failures: %r", self._tracker_failures, e)
            return []
        self._tracker_failures = 0
        return faces

    def _center(self, track: Track) -> None:
        """Lock on: keep the head aimed at a face, at poll rate, from the first sample
        that sees it until it leaves the frame (B0 'look them in the eye').

        The aim is *not* `track.azimuth_deg`. That number is head-relative — the camera
        is head-mounted, so it already contains our own turn — and commanding it is
        positive feedback: turning toward a face moves the bearing we match identity on
        (`TRACK_MATCH_AZIMUTH`), which is the chase `_scan` was written to avoid, and it is
        also the wrong sign. `MotionOwner.look_at` takes a raw yaw in the owner's own
        convention (positive = the robot's left: `shell/robot_tools.py` `look_left` sends
        +LOOK_ANGLE_DEG, `look_right` sends -), while `gaze.azimuth_deg` is positive to
        the *right of the image*. So the correction is negative of the reading, and it is
        added to where the head already points, which is the head-pose compensation the
        legacy loop got for free from `look_at_image` (`app/face_tracker.py:141`) and the
        reason dropping it made the follow chase instead of converge:

            aim = gaze_yaw_deg - FACE_FOLLOW_GAIN * azimuth_deg      [deg, owner convention]

        With a fixed person at body-relative bearing w, the measurement is
        `azimuth_deg = w + gaze_yaw`, so the aim converges on `-w` geometrically (ratio
        1 - GAIN per poll) and the face settles at image centre instead of oscillating
        around it. `MotionOwner.gaze_yaw_deg` is the smoothed gaze, i.e. our best model of
        where the camera was when the frame was exposed; an owner that will not say (a
        duck-typed stand-in) reads 0.0 and the aim degrades to uncorrected, still
        converging, just less precisely. Yaw only: `Face` has no vertical bearing (the
        pitch in `margins` is head orientation, not elevation), so pitch stays 0 as it was.

        Stands down for an exclusive move, the nap, and any gaze holder that is not ours
        — a `wake`/`doa` turn is a person who already has his attention by voice, and
        `look_at` has no priority, so taking the head back every poll would fight it.
        """
        motion = self.motion
        if motion is None or getattr(motion, "sleeping", False) or getattr(motion, "busy", False):
            return
        if getattr(motion, "gaze_source", "home") not in FOLLOW_SOURCES:
            return
        try:
            aim = self._aim_deg(motion, track.azimuth_deg)
            motion.look_at(aim, 0.0, hold_s=REAIM_HOLD_S, source=FACE_SOURCE)
        except Exception as e:
            log.warning("face lock-on failed: %r", e)
            return
        if self.debug:
            log.info(
                "greeter lock-on track=%d azimuth=%.1f aim=%.1f dwell=%.2f/%.1f",
                track.id,
                track.azimuth_deg,
                aim,
                track.dwell_s,
                self.dwell_s,
            )

    @staticmethod
    def _aim_deg(motion: Any, azimuth_deg: float) -> float:
        """Head-relative azimuth -> a gaze command, damped and deadbanded."""
        head = float(getattr(motion, "gaze_yaw_deg", 0.0) or 0.0)
        correction = -FACE_FOLLOW_GAIN * float(azimuth_deg)
        if abs(correction) <= FACE_FOLLOW_DEADBAND_DEG:
            return head  # already on him: re-command where we are to refresh the hold
        return head + correction

    def _scan(self, now: float) -> None:
        """The idle search: a slow ping-pong sweep for somebody to look at.

        Reached only when this sample saw no face at all — with `face_centering` on, a
        face anywhere in frame owns the head through `_center` instead, and the sweep
        stands down. That restriction is the whole hazard of this feature: `Face.azimuth_deg` is **head-relative** (gaze.py
        `azimuth_deg` measures the bbox centre against the image centre and the
        camera is head-mounted), so panning while a track exists moves the bearing
        we match identity on. One person walking past a still head becomes three
        tracks (`TRACK_MATCH_AZIMUTH`), their dwell resets, and the head starts
        chasing the azimuth its own turn invented. So a live track — or a sighting
        inside `scan_settle_s`, and a track lingers `TRACK_LOST_S` after the last
        detection — is a hard stop, not a hint.

        It is a gaze target, which is the bottom of the motion stack, so anything
        else that wants the head simply takes it: a `wake`/`doa` turn or a tool look
        writes `_gaze_target` over ours, and an emote (Animation priority 0, played
        without preemption) blends over the base gaze while it runs. We never fight
        back for the gaze — while a source other than `home`/`scan` holds it, the
        sweep stands down and re-centres for the next attempt.
        """
        motion = self.motion
        if not self.scan or motion is None:
            return
        if self.tracks or now - self._last_sighting < self.scan_settle_s:
            self._end_scan()
            return
        if self.is_playing() or self.lease_active() or self.awaiting_reply():
            return  # engaged with somebody: this is an idle-room search, not a tic
        if getattr(motion, "busy", False) or getattr(motion, "sleeping", False):
            return  # scripted move or the nap — the same rule `_center` keeps
        if getattr(motion, "gaze_source", "home") not in (SCAN_SOURCE, "home"):
            self._end_scan()  # a wake turn / tool look / face aim holds the head: yield
            return
        if now < self._scan_next:
            return
        lim, step = max(0.0, self.scan_limit_deg), abs(self.scan_step_deg)
        target = self._sweep_to(self._scan_deg + self._scan_dir * step, lim)
        if target == self._scan_deg:  # parked on the limit: this end of the room is done
            self._scan_dir = -self._scan_dir
            target = self._sweep_to(self._scan_deg + self._scan_dir * step, lim)
        if target == self._scan_deg:
            return  # degenerate knobs (a zero step): never re-command the same detent
        self._scan_deg = target
        self._scan_next = now + max(1.0, self.scan_step_s)  # at least one second per step
        try:
            motion.look_at(target, 0.0, hold_s=SCAN_HOLD_S, source=SCAN_SOURCE)
        except Exception as e:  # the sweep must never take the poll thread down
            log.warning("greeter scan failed: %r", e)
            return
        if self.debug:
            log.info("greeter scan azimuth=%.1f dir=%s", target, "left" if self._scan_dir < 0 else "right")
        if self.journal is None or now - self._last_scan_journal < SCAN_JOURNAL_S:
            return
        # Rate-limited like the suppression row: the booth card wants to know he is
        # looking around, and a row per detent would flush the ring in minutes.
        self._last_scan_journal = now
        self.journal.write(
            J.MOTION_SCAN,
            azimuth_deg=round(target, 1),
            direction="left" if self._scan_dir < 0 else "right",
            limit_deg=round(lim, 1),
            hold_s=SCAN_HOLD_S,
        )

    @staticmethod
    def _sweep_to(deg: float, lim: float) -> float:
        return max(-lim, min(lim, deg))

    def _end_scan(self) -> None:
        """Give the head back. Nothing is commanded here: `SCAN_HOLD_S` expires on
        its own and the motion owner springs the gaze home, so abandoning the sweep
        cannot snap the neck any harder than finishing it."""
        self._scan_deg = 0.0
        self._scan_dir = -1.0
        self._scan_next = -1e9

    # ---- reporting ----

    def _quiet(self, reason: str) -> str:
        if self.debug:
            log.info("greeter %s", reason)
        return reason

    def _log_faces(self, tracks) -> None:
        if not self.debug:
            return
        for track in tracks:
            if not hasattr(track, "dwell_s"):
                continue
            yaw, pitch, roll = (track.margins.get(k, 0.0) for k in ("yaw_deg", "pitch_deg", "roll_deg"))
            log.info(
                "greeter track=%d dwell=%.2f/%.1f yaw=%.1f pitch=%.1f roll=%.1f share=%.3f band=%.0f/%.0f verdict=%s",
                track.id,
                track.dwell_s,
                self.dwell_s,
                yaw,
                pitch,
                roll,
                track.share,
                track.margins.get("max_yaw_deg", 0.0),
                track.margins.get("min_area_share", 0.0),
                "looking" if track.qualified else "not-looking",
            )

    # ---- thread ----

    def start(self) -> bool:
        if not self.enabled or self.source is None or self.tracker is None or self._thread is not None:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="greeter", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = self.clock()
            if self.asleep():
                self._stop.wait(0.5)  # no samples while napping (docs/pins.yaml face_detection)
                continue
            try:
                self.sample(t0)
            except Exception as e:  # the poll thread outlives any one bad frame
                log.warning("greeter sample failed: %r", e)
            fps = self.engaged_fps if any(t.qualified for t in self.tracks.values()) else self.idle_fps
            self._stop.wait(max(0.0, 1.0 / fps - (self.clock() - t0)))

    def _set_engaged(self, engaged: bool) -> None:
        setter = getattr(self.source, "set_engaged", None)
        if callable(setter):
            try:
                setter(engaged)
            except Exception:  # the stream's own cadence is best-effort
                pass


def greet_hook(client, loop, *, journal=None):
    """cue -> `client.insert_user_transcript`, marshalled onto the asyncio loop.

    `insert_user_transcript` schedules a task on the running loop, so it must not
    be called from the poll thread. The greeting stays our own speech: it prompts
    a response through the persona and deliberately touches nothing about the
    attention lease — a face does not make somebody a participant (ADR 0003), and
    the hello does not open a window in which the room's speech becomes history.
    The `origin=greeter` marker is what enforces that: the shell refuses to
    exchange the lease when the reply to this directive finishes.
    """

    def on_greet(cue: GreetCue) -> None:
        try:
            loop.call_soon_threadsafe(lambda t=cue.text: client.insert_user_transcript(t, origin=ORIGIN_GREETER))
        except Exception as e:
            log.warning("could not deliver greeting: %r", e)
            if journal is not None:
                journal.write(J.GREETER_SUPPRESSED, reason="delivery-failed")

    return on_greet
