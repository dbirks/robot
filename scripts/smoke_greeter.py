#!/usr/bin/env python3
"""Watch the eye-contact greeter decide, one second at a time, over real frames.

Why this exists (docs/environments.md B3): the journal holds zero greeter-family rows out
of ~91k events, and that is expected rather than damning - `greeter_enabled` is false in
`home` (envs/home.yaml:42) and true only in `hackday` (envs/hackday.yaml:91), and the
hackday room has never been run live. So the greeter is *unproven*, which is neither
healthy nor broken. This settles what can be settled with no audio, no motion, no journal
and nobody standing in front of him for the wrong reasons:

  * does the one camera owner get frames on this box right now, without fighting the
    daemon (`shell/camera.py` asks GET /api/media/status and fails closed)?
  * does the detector load, and does it return anything from those frames?
  * does the geometric gate (`shell/vision/gaze.py`) ever call a face "looking at me",
    and by how much does it miss when it does not?
  * does dwell accumulate to the resolved limit, and what does the policy
    (`shell/vision/greeter.py`) decide at that instant - fire, or which suppression?

The three answers that matter are three different states, and collapsing them is the
failure mode this script was written against: NO FACES AT ALL (nobody there, or lighting),
FACES BUT NONE CENTRED (the gate, tunable in `shell/vision/gaze.py`), and DETECTION BROKEN
(the model or the code - `Greeter._detect` swallows the exception, so without the
`CountingTracker` below a dead detector looks exactly like an empty room).

    scripts/smoke_greeter.py                           # 10 s of the live camera
    scripts/smoke_greeter.py --seconds 30
    scripts/smoke_greeter.py --frames-file bench/stills # CI-safe: stills, no device opened
    scripts/smoke_greeter.py --env home                 # "disabled in this room", exit 2

One line per second, straight off `Greeter.sample()`:

    t+ 2.0s samples=3 frames=3 faces=1 centred=yes dwell=2.00/2.0s gate=FIRE(solo)  yaw=...

Limits are never literals here: they come from the resolved room, the same JSON the shell
boots with (`REACHY_ENV=<env> python -m shell --print-env`, reused from
`scripts/bench_greeter.py:resolve_limits`). If that subprocess cannot run, the smoke
refuses to guess and exits 3 rather than bench against constants.

Hard rules, enforced by construction rather than by care: no audio device is opened
(nothing in `shell/camera.py` or `shell/vision/` touches sounddevice), nothing is written
to the journal (`journal=None`), no cue reaches anybody (`on_greet` is a list append, not
`greet_hook`), the head cannot move (`motion=None`), the camera window is exactly
`--seconds` (the `CameraStream` is stopped in a `finally`), one grab is in flight at a
time, and Ctrl-C stops sampling and releases the device.

Exit codes: 0 the decision path ran end to end (it fired, or a face dwelled and the rate
limiter answered), 1 it ran but never reached the dwell gate (inconclusive, nothing proven
broken), 2 the decision path could not run at all (no frames, detection raising, greeter
disabled in this room), 3 the room's resolved limits could not be read.

What stays untested from any script, and the 10-minute live hackday bench that closes it,
is printed at the end of every run - see `WHAT_STAYS_UNTESTED`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # `python scripts/smoke_greeter.py` puts scripts/ on the path, not the repo
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))  # bench_greeter sits beside this file

from bench_greeter import Limits, pins_ms_per_frame, resolve_limits  # noqa: E402

from shell.camera import CameraGrabber, CameraStream
from shell.robot import RobotLink
from shell.vision import FaceTracker, Greeter

STILL_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# What each suppression costs the operator. The same vocabulary
# `scripts/bench_greeter.py` prints, kept to one line because it is a tail, not a report.
WHY = {
    "speaking": "he is mid-sentence (pinned open in this script, so unreachable here)",
    "hourly-cap": "the per-hour ceiling is spent",
    "presence-greeted": "this presence already got its hello",
    "spacing": "another greeting is inside the min-spacing window",
    "conversation": "lease held with fewer than two un-greeted faces (pinned open here)",
    "group-spacing": "the in-conversation arrival notice is still inside its window",
    "delivery-failed": "the cue never reached the realtime client (cannot happen here)",
}

WHAT_STAYS_UNTESTED = [
    "WHAT THIS SCRIPT CANNOT SETTLE: the greeter is a thread inside the running shell.",
    "`Greeter` is assembled at shell/__main__.py:319-341 and wired to a live",
    "RealtimeClient + asyncio loop (`greet_hook`), MotionOwner, the attention lease, the",
    "nap and the journal. A script that must not open audio, move the robot or write",
    "events cannot reach any of that, so these stay UNTESTED until somebody stands in",
    "front of him in the hackday room for ten minutes, on purpose:",
    "  0. None of this is a mid-demo fix: restarting units restarts the conversation.",
    "     Do it before the doors, and note that restarting reachy-mini-daemon stops",
    "     reachy-shell and it stays stopped (BindsTo=, docs/knobs.md).",
    "  1. Give the shell the debug knob: the real name is REACHY_GREETER_DEBUG",
    "     (docs/knobs.md:42-50, read at shell/config.py:227; it logs every track's",
    "     margins and every suppression and changes no behaviour). It is process env, so",
    "     it needs a drop-in plus a restart of the unit:",
    "       systemctl --user edit reachy-shell.service",
    "         [Service]",
    "         Environment=REACHY_GREETER_DEBUG=1",
    "       systemctl --user restart reachy-shell.service",
    "     Never put it in .env for the venue: it is 2-3 log lines per second on a 4-core",
    "     box and the shell has a latency budget (ADR 0006).",
    "  2. In a second terminal: scripts/bench_greeter.py --watch --max-seconds 600",
    "     (read-only over data/events.db; it is the only witness for the fired/suppressed",
    "     rows this script deliberately never writes).",
    "  3. Stand 1-2 m out, off-centre, then dead centre for 3 s, then step out and back",
    "     in. `journalctl --user -u reachy-shell -f` should show `greeter track=...`",
    "     margins the whole way, and `face.present` -> `greeter.fired` in the DB.",
    "  4. The four things only the live shell can still prove:",
    "     - the hello is SPEECH: greeter -> insert_user_transcript(origin=greeter) ->",
    "       persona -> TTS. Nothing run from a console can tell you he said it, in words",
    "       this room's persona chose (docs/environments.md hard rule 2).",
    "     - the ADR 0003 interlock: no solo hello over a live attention lease. This",
    "       script pins is_playing/lease_active/awaiting_reply/asleep open on purpose -",
    "       measuring the gate and the interlock at once tells you nothing about either.",
    "     - the nap: asleep means no frames and no inference at all",
    "       (shell/camera.py `_asleep`, shell/vision/greeter.py `_run`), exercised only",
    "       by the running shell.",
    "     - REACHY_GREETER_FACE_CENTERING=1 -> `Greeter._center`: the head hook. Off in",
    "       both rooms, unreachable from here (motion=None), and the one part of B0 you",
    "       have to watch with your eyes rather than in the journal.",
    "  5. Take the debug knob back out afterwards.",
]

# ---- what one sample saw, and what a second of them means ----


@dataclass(frozen=True)
class Observation:
    """One `Greeter.sample()` call, as the console sees it.

    `sample()` returns a reason string and nothing else: enough for the shell log, not
    enough for a bench. "He said nothing" is three different facts (nobody was there /
    nobody was centred / the gate held) and only the last is a rate limiter. `frames` is
    0 when no frame was read at all, which is why "no faces" here can never mean "no
    camera" - and why a raising detector needs `CountingTracker` to be visible at all.
    """

    t_s: float
    verdict: str
    frames: int = 1
    faces: int = 0
    centred: int = 0
    tracks: int = 0
    dwell_s: float = 0.0
    samples: int = 1
    margins: dict[str, float] = field(default_factory=dict)

    @property
    def fired(self) -> bool:
        return self.verdict.startswith("fire:")

    @property
    def kind(self) -> str:
        return self.verdict.split(":", 1)[1] if self.fired else ""

    @property
    def suppression(self) -> str:
        return self.verdict.split(":", 1)[1] if self.verdict.startswith("suppressed:") else ""


def verdict_rank(verdict: str) -> int:
    """Which verdict wins a second: fire > suppressed > idle > asleep > disabled > no frames."""
    if verdict.startswith("fire"):
        return 0
    if verdict.startswith("suppressed"):
        return 1
    return {"nobody-looking": 2, "asleep": 3, "disabled": 4, "no-camera": 5}.get(verdict, 6)


def fold_second(observed: Sequence[Observation]) -> Observation:
    """Collapse a second of samples into the one printed line.

    The poll runs at the resolved cadence (2 FPS idle / 3 FPS engaged), so "per second"
    is an aggregate: the most detections, the most centred faces, the longest dwell and
    the most consequential verdict in the window. Nothing an operator needs is averaged
    away - the per-sample verdicts are still in `classify`'s counts.
    """
    first = observed[0]
    best = min(observed, key=lambda o: verdict_rank(o.verdict))
    return Observation(
        t_s=first.t_s,
        verdict=best.verdict,
        frames=sum(o.frames for o in observed),
        faces=max(o.faces for o in observed),
        centred=max(o.centred for o in observed),
        tracks=max(o.tracks for o in observed),
        dwell_s=max(o.dwell_s for o in observed),
        samples=len(observed),
        margins=best.margins,
    )


def gate_said(obs: Observation) -> str:
    """The verdict in the words the operator asked for: fire, or why not, or idle - and
    idle says *which* idle, because "nobody in frame" and "a face the gate will not
    accept" and "a face that has not dwelled long enough" are three different things to
    walk out of a room and check."""
    if obs.fired:
        return f"FIRE({obs.kind})"
    if obs.suppression:
        return f"suppressed-because({obs.suppression})"
    if obs.verdict == "nobody-looking":
        if obs.faces == 0:
            return "idle(no-face-in-frame)"
        if obs.centred == 0:
            return "idle(face-not-centred)"
        return "idle(dwelling, not yet at the limit)"
    return f"idle({obs.verdict})"


def format_margins(m: dict[str, float]) -> str:
    """The gate's own numbers against the band actually in force (`gaze.looking_at_me`)."""
    if not m:
        return ""
    return (
        f"yaw={m['yaw_deg']:+.1f}/{m['max_yaw_deg']:.0f} pitch={m['pitch_deg']:+.1f}/{m['max_pitch_deg']:.0f} "
        f"roll={m['roll_deg']:+.1f}/{m['max_roll_deg']:.0f} "
        f"share={m['area_share']:.3f}/{m['min_area_share']:.3f}-{m['max_area_share']:.3f} "
        f"edge={m['edge_offset']:.3f}/{m['centre_band'] / 2.0:.3f} det={m['det_score']:.2f} "
        f"band={'relaxed' if m['hysteresis'] < 1.0 else 'strict'}"
    )


def format_observation(obs: Observation, limits: Limits) -> str:
    line = (
        f"t+{obs.t_s:4.1f}s samples={obs.samples} frames={obs.frames} "
        f"faces={obs.faces} centred={'yes' if obs.centred else 'no'} "
        f"dwell={obs.dwell_s:.2f}/{limits.dwell_s:g}s gate={gate_said(obs)}"
    )
    tail = format_margins(obs.margins)
    return f"{line}  {tail}" if tail else line


# ---- what a whole run means ----

EXIT_PROVEN = 0  # the decision path ran end to end
EXIT_INCONCLUSIVE = 1  # ran, never reached the dwell gate: nothing proven broken
EXIT_CANNOT_RUN = 2  # no frames / detection raising / greeter off in this room
EXIT_NO_LIMITS = 3  # the room's resolved limits could not be read

EXIT_CODES = {
    "gate-fired": EXIT_PROVEN,
    "gate-held": EXIT_PROVEN,
    "no-faces": EXIT_INCONCLUSIVE,
    "faces-never-centred": EXIT_INCONCLUSIVE,
    "not-enough-dwell": EXIT_INCONCLUSIVE,
    "no-samples": EXIT_CANNOT_RUN,
    "no-frames": EXIT_CANNOT_RUN,
    "camera-unavailable": EXIT_CANNOT_RUN,
    "detection-broken": EXIT_CANNOT_RUN,
    "disabled-in-this-room": EXIT_CANNOT_RUN,
}


@dataclass(frozen=True)
class Context:
    """What the verdict is judged against, plus anything that went wrong on the way."""

    limits: Limits
    device_error: str | None = None
    refusals: int = 0
    tracker_errors: Sequence[str] = ()
    detect_ms: Sequence[float] = ()
    lum: Sequence[tuple[float, float]] = ()
    warnings: Sequence[str] = ()


@dataclass(frozen=True)
class Result:
    state: str
    detail: str

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.state]


def classify(observations: Sequence[Observation], ctx: Context) -> Result:
    """The three-way split the whole exercise is for, checked in priority order.

    A detector that raises yields zero faces, exactly like an empty room; the only thing
    that tells them apart is the exception `Greeter._detect` swallowed, which
    `CountingTracker` caught. So detection is checked before "no faces", and "no frames"
    before both - a dead camera is not an empty room either.
    """
    limits = ctx.limits
    if not observations:
        return Result("no-samples", "the poll ran 0 samples (window shorter than one frame?)")
    if all(o.verdict == "disabled" for o in observations):
        return Result(
            "disabled-in-this-room",
            f"greeter_enabled={limits.enabled} in the room this run resolved ({limits.source}): "
            f"there is nothing to bench. Re-run with --env hackday",
        )
    frames_read = sum(o.frames for o in observations)
    face_samples = sum(1 for o in observations if o.faces)
    centred_seen = max(o.centred for o in observations)
    max_dwell = max(o.dwell_s for o in observations)
    if ctx.tracker_errors and face_samples == 0:
        return Result(
            "detection-broken",
            f"{frames_read} frame(s) read and face detection raised {len(ctx.tracker_errors)}x; the "
            f"greeter stood down silently, which is indistinguishable from an empty room in the "
            f"journal. First error: {ctx.tracker_errors[0]}",
        )
    if frames_read == 0:
        return Result(
            "no-frames",
            f"the one camera owner handed back no frame in {len(observations)} samples "
            f"({ctx.refusals} device refusal(s)): {ctx.device_error or 'reason unrecorded'}. Not "
            f"'nobody is there' - the decision path never ran.",
        )
    if face_samples == 0:
        return Result(
            "no-faces",
            f"{frames_read} frame(s) read, 0 faces found: nobody in frame, or the lighting/angle "
            f"is under `min_det_score` (shell/vision/gaze.py Gate). Camera path and detector are "
            f"both alive, which is not nothing.",
        )
    if centred_seen == 0:
        closest = max(observations, key=lambda o: (o.faces, o.dwell_s))
        return Result(
            "faces-never-centred",
            f"faces found ({face_samples} sample(s)) and the gate never called one 'looking at "
            f"me'. Closest face: {format_margins(closest.margins) or 'no margins recorded'} - "
            f"these thresholds are calibrated against nothing yet (shell/vision/gaze.py)",
        )
    if max_dwell < limits.dwell_s:
        return Result(
            "not-enough-dwell",
            f"a face was centred but never for {limits.dwell_s:g}s (max dwell {max_dwell:.2f}s). "
            f"Either nobody stayed, or the track evaporates and re-forms (greeter.Track.matches, "
            f"TRACK_LOST_S) and the dwell restarts",
        )
    fired = [o for o in observations if o.fired]
    if fired:
        kinds = ",".join(sorted({o.kind for o in fired}))
        return Result("gate-fired", f"a centred face dwelled {max_dwell:.2f}s and the gate said fire ({kinds})")
    held = max((o for o in observations if o.dwell_s >= limits.dwell_s), key=lambda o: o.t_s)
    return Result(
        "gate-held",
        f"a centred face dwelled {held.dwell_s:.2f}s and the policy answered {held.verdict} "
        f"[{WHY.get(held.suppression, held.suppression)}]",
    )


def format_result(result: Result, ctx: Context, cues: Sequence[Any]) -> list[str]:
    out = ["", f"RESULT {result.state} (exit {result.exit_code}) - {result.detail}"]
    for warning in ctx.warnings:
        out.append(f"  NOTE {warning}")
    if result.state == "gate-fired":
        out.append("  The cue the shell would have handed the persona. Recorded only: no client,")
        out.append("  no loop, no TTS, no journal row (that is the live bench's job).")
        for cue in cues:
            out.append(f'    {cue.kind} track={cue.track_id} bearing={cue.yaw_deg:+.1f}deg: "{cue.text}"')
    elif result.state == "gate-held":
        out.append("  Silent, correctly. The rate limiter is the one part that cannot be judged")
        out.append("  on a bench where nobody else has been greeted in the last two minutes.")
    if result.exit_code == EXIT_INCONCLUSIVE:
        out.append("  UNTESTED, not broken: what is above is proven, the dwell gate is not.")
    if ctx.lum:
        out.append(f"  {frame_stats(ctx.lum)}")
    if ctx.detect_ms:
        ordered = sorted(ctx.detect_ms)
        p50, p95, pins_src = pins_ms_per_frame()
        out.append(
            f"  detect cost: n={len(ordered)} first={ctx.detect_ms[0]:.1f}ms median={ordered[len(ordered) // 2]:.1f}ms "
            f"max={ordered[-1]:.1f}ms (first includes the onnx session load; {pins_src} measures "
            f"{p50}-{p95} ms/frame for inference alone)"
        )
    return out


# ---- the collaborators, assembled the way shell/__main__.py assembles them ----


class CountingTracker:
    """Forwards `detect()` to the real `FaceTracker` and remembers what came back.

    The reason it exists: `Greeter._detect` swallows detector exceptions by design (a
    poll thread must not die on one bad frame), so a model that will not load and a room
    with nobody in it are the same sighting. Recording the error here - and re-raising it,
    so the greeter's own stand-down still runs - is what lets the smoke tell BROKEN apart
    from NOBODY THERE.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.faces: list[Any] = []
        self.calls = 0
        self.errors: list[str] = []
        self.ms: list[float] = []
        self.lum: list[tuple[float, float]] = []

    def detect(self, frame: Any) -> list[Any]:
        """`FaceTracker.detect` plus the two things the greeter cannot see: what it
        raised, and what it cost. `docs/pins.yaml` `face_detection` measures the model
        and says the per-sample read and the warm-up are UNMEASURED PENDING BENCH, and
        the console is where somebody can act on that."""
        t0 = time.perf_counter()
        self.calls += 1
        self.lum.append(_lum(frame))
        try:
            self.faces = list(self.inner.detect(frame) or [])
        except Exception as e:
            self.faces = []
            self.errors.append(f"{type(e).__name__}: {e}")
            raise
        finally:
            self.ms.append((time.perf_counter() - t0) * 1000.0)
        return self.faces


def _lum(frame: Any) -> tuple[float, float]:
    """(mean, std) of a frame, for "is there even a picture here"."""
    try:
        return (float(frame.mean()), float(frame.std()))
    except Exception:  # a fake/test frame, or nothing decodable
        return (float("nan"), float("nan"))


def frame_stats(lum: Sequence[tuple[float, float]]) -> str:
    """Are the frames pictures? Zero faces in a black frame is not "nobody is there".

    A UVC stream that opened and stalled, or a lens with the shutter on, reads exactly
    like an empty room to a detector. The mean range says lighting; the distinct count
    says the stream is moving; a flat near-zero mean says the picture is not there at
    all, and that is a broken-camera answer, not a nobody-there one.
    """
    means = [m for m, _ in lum if m == m]  # NaN-dropped
    if not means:
        return "frame brightness: not measurable from these frames"
    stds = [s for _, s in lum if s == s]
    distinct = len({round(m, 3) for m in means})
    line = (
        f"frame brightness: mean {min(means):.1f}-{max(means):.1f} (max std {max(stds, default=0.0):.1f}) "
        f"over {len(means)} frame(s), {distinct} distinct"
    )
    if max(means) < 2.0:
        return line + " - BLACK FRAMES: the stream opened and gave nothing to see"
    if distinct == 1 and len(means) > 2:
        return line + " - the same picture every read (a stalled stream live; one repeated still with --frames-file)"
    return line + " - real, moving image data"


class StillsSource:
    """The `--frames-file` frame owner: `latest()` out of a list of stills, wrapping.

    Shaped like `CameraStream` for everything `Greeter` touches (`latest()`,
    `set_engaged()`) and touches no device. `repeats` holds each still for that many
    samples so a directory of stills can be paced; wrapping means one still of one person
    is enough to walk a dwell across its own limit.
    """

    def __init__(self, frames: Sequence[Any], *, repeats: int = 1) -> None:
        self.frames = list(frames)
        self.repeats = max(1, int(repeats))
        self.engaged: bool | None = None
        self.reads = 0

    def __len__(self) -> int:
        return len(self.frames)

    def latest(self) -> Any:
        if not self.frames:
            return None
        frame = self.frames[(self.reads // self.repeats) % len(self.frames)]
        self.reads += 1
        return frame

    def set_engaged(self, engaged: bool) -> None:
        self.engaged = engaged


class ProbeGrabber(CameraGrabber):
    """`CameraGrabber` that keeps the reason the device said no.

    `CameraStream._poll_once` logs it and degrades quietly - correct for the shell, which
    owns this camera for real - and that same quietness would leave this script reporting
    "no faces". Same open/read/close path, so ADR 0002 holds: one object touches
    /dev/videoN, and it is still `shell/camera.py`'s.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.error: str | None = None
        self.refusals = 0

    def _refused(self, e: BaseException) -> None:
        self.refusals += 1
        self.error = f"{type(e).__name__}: {e}"

    def open_stream(self) -> None:
        try:
            super().open_stream()
        except Exception as e:
            self._refused(e)
            raise

    def read_stream(self) -> Any:
        try:
            return super().read_stream()
        except Exception as e:
            self._refused(e)
            raise

    def grab(self) -> Any:
        try:
            return super().grab()
        except Exception as e:
            self._refused(e)
            raise


def build_greeter(limits: Limits, source: Any, tracker: Any, cues: list[Any]) -> Greeter:
    """The production `Greeter`, assembled as shell/__main__.py:319-341 assembles it.

    Same collaborators, same resolved numbers. Four differences, all deliberate and all
    printed: `journal=None` (this run must leave no trace in data/events.db),
    `on_greet=cues.append` instead of `greet_hook(client, loop)` (the cue is recorded; no
    speech is possible), `motion=None` (the head cannot move even if `face_centering`
    resolves true), and the fail-closed interlocks (`is_playing`, `asleep`,
    `lease_active`, `awaiting_reply`) pinned open - the point is to measure the
    detection/dwell/gate path with nothing else in the way. Those interlocks belong to
    the running shell: tests/test_vision_greeter.py pins them, and
    scripts/bench_greeter.py judges them from the journal after a live run.
    """
    return Greeter(
        source=source,
        tracker=tracker,
        journal=None,
        on_greet=cues.append,
        motion=None,
        enabled=limits.enabled,
        face_centering=limits.face_centering,
        is_playing=lambda: False,
        asleep=lambda: False,
        lease_active=lambda: False,
        awaiting_reply=lambda: False,
        dwell_s=limits.dwell_s,
        min_spacing_s=limits.min_spacing_s,
        group_min_spacing_s=limits.group_min_spacing_s,
        max_per_hour=limits.max_per_hour,
        idle_fps=limits.idle_fps,
        engaged_fps=limits.engaged_fps,
        debug=limits.debug,
    )


def live_owner(limits: Limits) -> tuple[ProbeGrabber, CameraStream]:
    """`CameraGrabber` + `CameraStream` exactly as shell/__main__.py:121-131 wires them.

    `RobotLink` is constructed for its `daemon_url` property alone: `connect()` is what
    instantiates `ReachyMini`, and a second SDK client while the live shell holds the
    robot hangs, so it is never called here.
    """
    robot = RobotLink()  # no connect(): daemon_url is a plain property (shell/robot.py:41)
    grabber = ProbeGrabber(robot.daemon_url)
    stream = CameraStream(
        grabber,
        idle_fps=limits.idle_fps,
        engaged_fps=limits.engaged_fps,
        journal=None,  # the production stream journals `camera.disabled`; this one must not
        asleep=lambda: False,
    )
    return grabber, stream


# ---- the poll: sample() in the foreground, one grab at a time ----


def observe(greeter: Greeter, seen: CountingTracker, *, calls_before: int, t_s: float, verdict: str) -> Observation:
    """Read the room back out of the greeter after one `sample()`.

    Detection calls are how we know a frame was actually read (a `no-camera` sample never
    reaches the tracker), and the margins come from the largest live presence - centred or
    not - because when the gate says no, the numbers that missed are the report.
    """
    read = seen.calls > calls_before
    tracks = list(greeter.tracks.values())
    qualified = [t for t in tracks if t.qualified]
    nearest = max(qualified or tracks, key=lambda t: t.share, default=None)
    return Observation(
        t_s=t_s,
        verdict=verdict,
        frames=1 if read else 0,
        faces=len(seen.faces) if read else 0,
        centred=len(qualified),
        tracks=len(tracks),
        dwell_s=max((t.dwell_s for t in qualified), default=0.0),
        margins=dict(nearest.margins) if nearest else {},
    )


def drive(
    greeter: Greeter,
    seen: CountingTracker,
    out: list[Observation],
    *,
    seconds: float,
    limits: Limits,
    clock: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
    on_line: Callable[[Observation], None] | None = None,
) -> list[Observation]:
    """`Greeter.sample()` for `seconds`, one sample in flight, one printed line per second.

    Deliberately not `greeter.start()`: the production poll thread is a wait-loop around
    this same `sample()` (`Greeter._run`), and running it here would put thread jitter on
    the dwell boundary that has to be measured exactly - which is how
    tests/test_smoke_greeter.py pins 2.0 s against 1.9 s on a fake clock - and would make
    "the camera window ends when I said so" a courtesy instead of a `finally`. `out` is
    the caller's list, so a Ctrl-C mid-window still has its samples to report on.
    """
    bucket: list[Observation] = []
    started = clock()

    def emit(window: Sequence[Observation]) -> None:
        if on_line is not None and window:
            on_line(fold_second(window))

    while True:
        now = clock()
        t = now - started
        if t >= seconds:
            break
        if bucket and int(bucket[0].t_s) != int(t):
            emit(bucket)
            bucket = []
        calls_before = seen.calls
        verdict = greeter.sample(now)
        obs = observe(greeter, seen, calls_before=calls_before, t_s=t, verdict=verdict)
        out.append(obs)
        bucket.append(obs)
        fps = limits.engaged_fps if obs.centred else limits.idle_fps
        wait(max(0.0, 1.0 / fps - (clock() - now)))
    emit(bucket)
    return out


# ---- stills ----


def stills_in(directory: Path) -> list[Path]:
    """The stills in `directory`, in a stable order (reproducibility is the whole point)."""
    if not directory.is_dir():
        raise SystemExit(f"--frames-file {directory}: not a directory")
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in STILL_SUFFIXES)


def load_stills(paths: Sequence[Path]) -> list[Any]:
    """BGR arrays through the repo's own decoder. `cv2` is imported lazily - CI runs the
    suite without opencv, and only this path needs it, exactly as shell/camera.py defers
    it for the same reason."""
    import cv2

    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"--frames-file {path}: did not decode")
        frames.append(frame)
    return frames


def shape_warnings(frames: Sequence[Any], expected: tuple[int, int]) -> list[str]:
    """The gate is pinned to what `shell/camera.py` grabs at (`Greeter.frame_shape`).

    A still of another size is a different physical face at the same `area_share`, so say
    so rather than quietly benching a geometry nobody will ever stand in.
    """
    shapes = sorted({tuple(int(v) for v in frame.shape[:2]) for frame in frames})
    wrong = [s for s in shapes if s != tuple(expected)]
    return [
        f"stills are {list(s)}, the gate's frame_shape is {list(expected)}: share/edge are not comparable"
        for s in wrong
    ]


# ---- entry ----


def _run(args: argparse.Namespace, limits: Limits) -> int:
    cues: list[Any] = []
    observations: list[Observation] = []
    warnings: list[str] = []
    grabber: ProbeGrabber | None = None
    stream: CameraStream | None = None

    if args.frames_file:
        directory = Path(args.frames_file)
        frames = load_stills(stills_in(directory))
        if not frames:
            raise SystemExit(f"--frames-file {directory}: no {'/'.join(STILL_SUFFIXES)} files in it")
        source: Any = StillsSource(frames, repeats=args.repeats)
    else:
        grabber, stream = live_owner(limits)
        source = stream

    tracker = CountingTracker(FaceTracker())
    greeter = build_greeter(limits, source, tracker, cues)
    if args.frames_file:
        warnings += shape_warnings(source.frames, greeter.frame_shape)
        print(
            f"frames: {len(source.frames)} still(s) from {directory}, held {args.repeats} "
            f"sample(s) each, wrapping; camera not opened"
        )
    if limits.face_centering:
        warnings.append("face_centering resolved True; motion=None here, so the head hook cannot run from a script")
    else:
        warnings.append("head-centring off, as in both rooms (REACHY_GREETER_FACE_CENTERING is a live-bench knob)")

    try:
        if stream is not None and grabber is not None:
            if not stream.start():  # one synchronous poll: is there a camera on this box?
                print(
                    f"\nRESULT camera-unavailable (exit {EXIT_CODES['camera-unavailable']}) - "
                    f"{grabber.error or 'the daemon or the device said no and no reason was recorded'}"
                )
                print("  The decision path did not run. No synthetic frame is stood in for a real")
                print("  one: a smoke that passes on a painted frame is the exact thing that would")
                print("  have hidden this all week.")
                return EXIT_CODES["camera-unavailable"]
            print(f"camera: holding /dev/videoN for {args.seconds:g}s through CameraStream (daemon released it)")
        interrupted = False
        try:
            drive(
                greeter,
                tracker,
                observations,
                seconds=args.seconds,
                limits=limits,
                on_line=lambda obs: print(format_observation(obs, limits), flush=True),
            )
        except KeyboardInterrupt:
            interrupted = True
        if interrupted and observations:
            print(f"(interrupted at t+{observations[-1].t_s:.1f}s; releasing the camera)")
        ctx = Context(
            limits=limits,
            device_error=grabber.error if grabber else None,
            refusals=grabber.refusals if grabber else 0,
            tracker_errors=tuple(tracker.errors),
            detect_ms=tuple(tracker.ms),
            lum=tuple(tracker.lum),
            warnings=tuple(warnings),
        )
        result = classify(observations, ctx)
        for line in format_result(result, ctx, cues):
            print(line)
        counts = Counter(o.verdict for o in observations)
        print("  verdicts: " + ", ".join(f"{v} x{n}" for v, n in counts.most_common()))
        return result.exit_code
    finally:
        if stream is not None:
            stream.stop()  # the held handle goes back with the stream (ADR 0002)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Bounded smoke for the eye-contact greeter: real frames, no audio, no journal, no motion."
    )
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to hold the camera (default 10)")
    ap.add_argument("--env", default="hackday", help="which room's resolved limits to run (default hackday)")
    ap.add_argument("--frames-file", default=None, help="directory of stills instead of the live camera (CI-safe)")
    ap.add_argument("--repeats", type=int, default=1, help="with --frames-file: samples to hold each still for")
    args = ap.parse_args(argv)

    limits = resolve_limits(args.env)
    if limits.fallback:
        print(f"RESULT limits-unresolved (exit {EXIT_NO_LIMITS}) - {limits.source}")
        print("  Refusing to smoke against constants: the whole point is the numbers this room")
        print(f"  resolves. Run `REACHY_ENV={args.env} python -m shell --print-env` by hand first.")
        return EXIT_NO_LIMITS
    print(
        f"resolved ({limits.source}): enabled={limits.enabled} dwell={limits.dwell_s:g}s "
        f"spacing={limits.min_spacing_s:g}s group={limits.group_min_spacing_s:g}s cap={limits.max_per_hour}/h "
        f"fps={limits.idle_fps:g}/{limits.engaged_fps:g} face_centering={limits.face_centering} "
        f"debug={limits.debug}"
    )
    if limits.debug:  # the greeter's own margin log, in the shape journalctl shows
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-18s %(levelname)-5s %(message)s")
    try:
        return _run(args, limits)
    finally:
        print()
        for line in WHAT_STAYS_UNTESTED:
            print(line)


if __name__ == "__main__":
    sys.exit(main())
