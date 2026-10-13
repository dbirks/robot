"""The greeter smoke's decision path, on synthetic frames and a fake clock.

`scripts/smoke_greeter.py` is what stands in front of the venue's one untested lane, so
the arithmetic it prints has to be pinned here rather than discovered at the booth: a
face that dwells 2.0 s accumulates and fires, a face that leaves at 1.9 s does not, and
- the reason the script exists at all - "no faces at all", "faces but none centred",
"the detector is broken" and "there is no camera" are four different answers and must
never collapse into one line.

Frames are real numpy arrays and the tracker is the real `FaceTracker`; only the loaded
onnx app is scripted (the same seam tests/test_vision_greeter.py uses for detections), so
share/azimuth/keypoint maths is exercised and no model file or opencv is needed - which is
what makes `--frames-file` safe in CI. Time is a variable: the dwell boundary is the thing
under test, and thread jitter would put it in doubt.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_greeter as sg  # noqa: E402
from test_vision_gaze import bbox_for, kps  # noqa: E402

from shell.vision.greeter import SOLO_CUE  # noqa: E402
from shell.vision.tracker import FaceTracker  # noqa: E402

FRAME = (720, 1280)  # what shell/camera.py grabs at, and what `bbox_for` shares use

LIMITS = sg.Limits(
    enabled=True,
    dwell_s=2.0,
    min_spacing_s=120.0,
    group_min_spacing_s=300.0,
    max_per_hour=6,
    idle_fps=2.0,
    engaged_fps=2.0,  # 0.5 s steps: binary-exact, so "exactly 2.0 s of dwell" is arithmetic
    face_centering=False,
    debug=False,
    source="test: hand-built limits",
)


class Clock:
    """Fake clock, and `time.sleep` stand-in: `drive()`'s pacing *is* the elapsed time."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class ScriptedEngine:
    """`insightface.app.FaceAnalysis` stand-in, keyed off the frame's own pixels.

    `FaceTracker` asks a loaded app for `.get(frame)` and reads `bbox`/`kps`/`det_score`
    off each result (shell/vision/tracker.py:88-104), which is all this implements.
    `raises` is the broken-install path: the greeter swallows it, so it is the case
    `CountingTracker` exists to make visible.
    """

    def __init__(self, by_label: dict[int, list[dict]], *, raises: Exception | None = None) -> None:
        self.by_label = by_label
        self.raises = raises
        self.calls = 0

    def get(self, frame):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return list(self.by_label.get(int(frame[0, 0, 0]), []))


def paint(label: int, *, shape: tuple[int, int] = FRAME) -> np.ndarray:
    """A synthetic camera frame; the label sits in pixel (0,0) for the engine to read."""
    frame = np.zeros(shape + (3,), dtype=np.uint8)
    frame[0, 0, 0] = label
    return frame


def detection(*, share: float = 0.05, cx: float = 640.0, score: float = 0.9, yaw: float = 0.0) -> dict:
    """One detection in the dict shape insightface returns: box gives the share, kps the angles."""
    return {"bbox": list(bbox_for(share, cx)), "kps": kps(cx=cx, yaw=yaw), "det_score": score}


FRONT = paint(1)  # a centred, gate-accepting face
NOBODY = paint(0)  # an empty room
EDGE = paint(2)  # a face at the frame's right edge: detected, never centred
AWAY = paint(3)  # a second presence, elsewhere in frame and much closer: a new track

LABELS = {
    1: [detection()],
    0: [],
    2: [detection(cx=1150.0)],  # edge offset 0.398 > centre_band/2 = 0.30
    3: [detection(share=0.10, cx=940.0)],  # 18.75 deg off and twice the area: `Track.matches` says other person
}


def run(source, *, limits=LIMITS, seconds=4.0, raises=None):
    """`drive()` on the fake clock: sample n lands at n/fps, one grab at a time."""
    tracker = sg.CountingTracker(FaceTracker(app=ScriptedEngine(LABELS, raises=raises)))
    cues: list = []
    greeter = sg.build_greeter(limits, source, tracker, cues)
    clock = Clock()
    observations: list[sg.Observation] = []
    lines: list[sg.Observation] = []
    sg.drive(
        greeter,
        tracker,
        observations,
        seconds=seconds,
        limits=limits,
        clock=clock,
        wait=clock.advance,
        on_line=lines.append,
    )
    return observations, greeter, cues, tracker, lines


def ctx(limits=LIMITS, **kw):
    return sg.Context(limits=limits, **kw)


# ---- the dwell rule, on frames rather than on hand-built detections ----


def test_a_face_that_dwells_two_seconds_fires_and_the_cue_is_recorded_not_sent():
    obs, greeter, cues, tracker, _ = run(sg.StillsSource([FRONT]), seconds=2.5)
    assert [round(o.dwell_s, 2) for o in obs[:5]] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert [o.verdict for o in obs[:4]] == ["nobody-looking"] * 4
    assert obs[4].verdict == "fire:solo"
    assert len(cues) == 1 and cues[0].kind == "solo" and cues[0].text == SOLO_CUE
    # one greeting per presence, however long they stand there
    assert sum(1 for o in obs if o.fired) == 1
    assert greeter.journal is None and greeter.motion is None
    assert len(tracker.errors) == 0
    result = sg.classify(obs, ctx())
    assert (result.state, result.exit_code) == ("gate-fired", 0)


def test_a_face_that_leaves_at_1_9s_never_accumulates():
    fine = dataclasses.replace(LIMITS, idle_fps=10.0, engaged_fps=10.0)
    stills = [FRONT] * 20 + [NOBODY] * 20  # centred through t=1.9, then gone
    obs, _, cues, _, _ = run(sg.StillsSource(stills), limits=fine, seconds=4.0)
    dwell = max(o.dwell_s for o in obs)
    assert 1.8 < dwell < 2.0, dwell
    assert cues == []
    assert not any(o.fired for o in obs)
    result = sg.classify(obs, ctx(fine))
    assert (result.state, result.exit_code) == ("not-enough-dwell", sg.EXIT_INCONCLUSIVE)


def test_the_printed_dwell_uses_the_resolved_limit_not_a_literal():
    slow = dataclasses.replace(LIMITS, dwell_s=3.0)
    obs, _, cues, _, lines = run(sg.StillsSource([FRONT]), limits=slow, seconds=2.5)
    assert cues == []
    assert "dwell=2.00/3s" in sg.format_observation(lines[-1], slow)
    assert sg.classify(obs, ctx(slow)).state == "not-enough-dwell"


# ---- the three answers that must not collapse into one ----


def test_no_faces_at_all_is_reported_apart_from_faces_but_none_centred():
    obs_empty, _, _, _, _ = run(sg.StillsSource([NOBODY]), seconds=1.5)
    assert {o.faces for o in obs_empty} == {0}
    assert sg.classify(obs_empty, ctx()).state == "no-faces"

    obs_side, _, _, _, _ = run(sg.StillsSource([EDGE]), seconds=1.5)
    assert {o.faces for o in obs_side} == {1}  # the detector sees a face...
    assert {o.centred for o in obs_side} == {0}  # ...and the gate refuses it
    result = sg.classify(obs_side, ctx())
    assert result.state == "faces-never-centred"
    assert "edge=" in result.detail and "share=" in result.detail
    # the point of the whole script: `sample()`'s own verdict cannot tell these apart,
    # so the console would be reading "nobody-looking" for both of them.
    assert obs_empty[0].verdict == obs_side[0].verdict == "nobody-looking"


def test_a_broken_detector_is_broken_and_not_an_empty_room():
    obs, _, cues, tracker, _ = run(
        sg.StillsSource([FRONT]), seconds=4.0, raises=ImportError("libonnxruntime.so: cannot open shared object")
    )
    assert {o.faces for o in obs} == {0}  # what an empty room looks like, too
    assert len(tracker.errors) == 3  # `Greeter._detect` stands down after three
    assert cues == []
    result = sg.classify(obs, ctx(tracker_errors=tuple(tracker.errors)))
    assert (result.state, result.exit_code) == ("detection-broken", sg.EXIT_CANNOT_RUN)
    assert "ImportError" in result.detail


def test_no_camera_is_reported_as_no_frames_and_names_the_reason():
    class HeldSource:
        """What `CameraStream` looks like to the greeter when the daemon kept the device."""

        def latest(self):
            return None

        def grab(self):
            raise RuntimeError("Camera is held by the robot daemon")

        def set_engaged(self, engaged):
            pass

    obs, _, _, tracker, _ = run(HeldSource(), seconds=1.5)
    assert [o.verdict for o in obs] == ["no-camera"] * len(obs)
    assert tracker.calls == 0  # no frame, no inference at all
    result = sg.classify(obs, ctx(device_error="RuntimeError: Camera is held by the robot daemon", refusals=1))
    assert (result.state, result.exit_code) == ("no-frames", sg.EXIT_CANNOT_RUN)
    assert "daemon" in result.detail and "never ran" in result.detail


def test_a_dwelling_face_the_limiter_held_is_still_a_proven_path():
    """Yesterday's guest is why today's got nothing - and that is still a result.

    Two windows on one `Greeter`, the way a room actually arrives: A dwells and fires, then
    leaves and B walks up inside the resolved 120 s spacing. The second window contains no
    fire at all, so it must read as "the gate ran and the limiter held", not as a failure
    and not as an empty room. `presence-greeted` also shows up in it, which is the same
    verdict one second earlier and the reason `classify` reports the last one it saw.
    """
    tracker = sg.CountingTracker(FaceTracker(app=ScriptedEngine(LABELS)))
    source = sg.StillsSource([FRONT])
    greeter = sg.build_greeter(LIMITS, source, tracker, [])
    clock = Clock()
    first: list[sg.Observation] = []
    sg.drive(greeter, tracker, first, seconds=2.5, limits=LIMITS, clock=clock, wait=clock.advance)
    assert [o.verdict for o in first if o.fired] == ["fire:solo"]

    greeter.source = sg.StillsSource([AWAY])  # A walks off, B walks up
    second: list[sg.Observation] = []
    sg.drive(greeter, tracker, second, seconds=4.0, limits=LIMITS, clock=clock, wait=clock.advance)
    assert not any(o.fired for o in second)
    assert max(o.dwell_s for o in second) >= LIMITS.dwell_s
    assert "spacing" in {o.suppression for o in second if o.suppression}
    result = sg.classify(second, ctx())
    assert (result.state, result.exit_code) == ("gate-held", sg.EXIT_PROVEN)
    assert "min-spacing" in result.detail


# ---- the printed second ----


def test_one_line_per_second_says_faces_centred_dwell_and_the_gate():
    obs, _, _, _, lines = run(sg.StillsSource([FRONT]), seconds=3.0)
    assert [round(o.t_s, 1) for o in lines] == [0.0, 1.0, 2.0]  # 6 samples, 3 printed lines
    first, _, last = lines
    assert first.samples == 2 and first.frames == 2
    text = [sg.format_observation(o, LIMITS) for o in lines]
    assert "faces=1 centred=yes dwell=0.50/2s gate=idle(dwelling, not yet at the limit)" in text[0]
    assert "gate=FIRE(solo)" in text[2] and "dwell=2.50/2s" in text[2]
    assert "band=strict" in text[0]  # the margins of the gate that decided, not just the bool


def test_idle_names_which_kind_of_idle_it_was():
    idle = [
        (sg.Observation(t_s=0.0, verdict="nobody-looking", frames=1, faces=0), "idle(no-face-in-frame)"),
        (sg.Observation(t_s=0.0, verdict="nobody-looking", frames=1, faces=1, centred=0), "idle(face-not-centred)"),
        (
            sg.Observation(t_s=0.0, verdict="nobody-looking", frames=1, faces=1, centred=1),
            "idle(dwelling, not yet at the limit)",
        ),
        (
            sg.Observation(t_s=0.0, verdict="suppressed:hourly-cap", faces=1, centred=1),
            "suppressed-because(hourly-cap)",
        ),
        (sg.Observation(t_s=0.0, verdict="fire:group", faces=1, centred=1), "FIRE(group)"),
    ]
    assert [(sg.gate_said(o), want) for o, want in idle] == [(want, want) for _, want in idle]


def test_a_second_of_samples_folds_to_the_line_that_matters():
    obs, _, _, _, lines = run(sg.StillsSource([FRONT]), seconds=3.0)
    assert lines == [sg.fold_second(g) for g in ([obs[0], obs[1]], [obs[2], obs[3]], [obs[4], obs[5]])]
    assert max(o.centred for o in obs) == 1  # the fold keeps the peak, not the mean


# ---- the assembly and the frame owner ----


def test_the_assembly_copies_every_resolved_number_and_owns_nothing():
    limits = dataclasses.replace(
        LIMITS,
        dwell_s=3.5,
        min_spacing_s=11.0,
        group_min_spacing_s=22.0,
        max_per_hour=4,
        idle_fps=1.5,
        engaged_fps=2.5,
        face_centering=True,
        debug=True,
        enabled=False,
    )
    tracker = sg.CountingTracker(FaceTracker(app=ScriptedEngine(LABELS)))
    g = sg.build_greeter(limits, sg.StillsSource([FRONT]), tracker, [])
    assert (g.dwell_s, g.min_spacing_s, g.group_min_spacing_s, g.max_per_hour) == (3.5, 11.0, 22.0, 4)
    assert (g.idle_fps, g.engaged_fps, g.face_centering, g.debug, g.enabled) == (1.5, 2.5, True, True, False)
    assert g.journal is None and g.motion is None  # no events, no head: by construction
    assert not any((g.is_playing(), g.asleep(), g.lease_active(), g.awaiting_reply()))


def test_the_grabber_keeps_the_reason_the_device_said_no():
    grabber = sg.ProbeGrabber("http://localhost:8000", mode="off")
    try:
        grabber.grab()
    except RuntimeError as e:
        assert "REACHY_CAMERA=off" in str(e)
    else:
        raise AssertionError("ProbeGrabber swallowed the refusal instead of reporting it")
    assert grabber.refusals == 1
    assert grabber.error == "RuntimeError: Camera disabled (REACHY_CAMERA=off)"


def test_stills_are_paced_and_wrap():
    source = sg.StillsSource([FRONT, EDGE], repeats=3)
    assert [int(source.latest()[0, 0, 0]) for _ in range(9)] == [1, 1, 1, 2, 2, 2, 1, 1, 1]
    assert sg.StillsSource([]).latest() is None
    source.set_engaged(True)
    assert source.engaged is True


def test_stills_in_is_sorted_and_ignores_everything_that_is_not_an_image(tmp_path):
    for name in ("b.png", "a.jpg", "notes.txt", "c.JPEG", "sub.webp"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "dir").mkdir()
    assert sg.stills_in(tmp_path) == [
        tmp_path / "a.jpg",
        tmp_path / "b.png",
        tmp_path / "c.JPEG",
        tmp_path / "sub.webp",
    ]


def test_a_still_that_is_not_camera_shaped_is_called_out():
    assert sg.shape_warnings([FRONT, paint(1, shape=(480, 640))], (720, 1280)) == [
        "stills are [480, 640], the gate's frame_shape is [720, 1280]: share/edge are not comparable"
    ]
    assert sg.shape_warnings([FRONT], (720, 1280)) == []
