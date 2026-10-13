"""The geometric looking-at-me gate: pure functions, hand-placed keypoints.

No camera, no model, no numpy: `kps` is five [x, y] pairs, exactly what
insightface's detector hands us, so these numbers are the ones the greeter runs on.
"""

from __future__ import annotations

import math

from shell.vision.gaze import Gate, area_share, azimuth_deg, estimate_orientation, looking_at_me

FRAME = (720, 1280)  # what shell/camera.py grabs at


def kps(*, cx=640.0, eye_y=300.0, iod=80.0, yaw=0.0, pitch=0.0, roll=0.0):
    """Five keypoints: eyes at (cx±iod/2, eye_y), mouth 90 px below, nose between.

    The nose offsets reproduce the derivation (yaw from the eye midpoint in
    inter-ocular units, pitch from the eye-mouth midpoint), and the eye axis is
    rotated for roll.
    """
    e0 = (cx - iod / 2.0, eye_y)
    e1 = (cx + iod / 2.0, eye_y)
    if roll:
        a = math.radians(roll)
        e0 = (cx + (e0[0] - cx) * math.cos(a) - (e0[1] - eye_y) * math.sin(a), eye_y + (e0[0] - cx) * math.sin(a))
        e1 = (cx + (e1[0] - cx) * math.cos(a) - (e1[1] - eye_y) * math.sin(a), eye_y + (e1[0] - cx) * math.sin(a))
    mouth_y = eye_y + 90.0
    neutral_y = eye_y + 45.0
    nose = (cx + iod * math.tan(math.radians(yaw / 1.5)), neutral_y - 45.0 * math.tan(math.radians(pitch)))
    return [e0, e1, nose, (cx - 30.0, mouth_y), (cx + 30.0, mouth_y)]


def bbox_for(share: float, cx: float = 640.0) -> tuple[float, float, float, float]:
    """A square face box with the requested share of FRAME."""
    side = math.sqrt(share * FRAME[0] * FRAME[1])
    return (cx - side / 2.0, 360.0 - side / 2.0, cx + side / 2.0, 360.0 + side / 2.0)


def test_frontal_face_is_zero_zero_zero():
    yaw, pitch, roll = estimate_orientation(kps(), (0, 0, 100, 100), FRAME)
    assert abs(yaw) < 0.2 and abs(pitch) < 0.2 and abs(roll) < 0.2


def test_orientation_signs_follow_the_image():
    assert estimate_orientation(kps(yaw=20.0), None, None)[0] > 15  # nose to frame right
    assert estimate_orientation(kps(yaw=-20.0), None, None)[0] < -15
    assert estimate_orientation(kps(pitch=20.0), None, None)[1] > 15  # head tipped up
    assert estimate_orientation(kps(pitch=-20.0), None, None)[1] < -15  # head down at a phone
    assert estimate_orientation(kps(roll=15.0), None, None)[2] > 10


def test_roll_is_normalised_whatever_order_the_eyes_arrive_in():
    """The eye axis wraps into (-90, 90]: a mirrored pair is not a 180 deg head."""
    level = estimate_orientation(kps(), None, None)[2]
    mirrored = estimate_orientation(kps(roll=180.0), None, None)[2]
    assert abs(level) < 0.2 and abs(mirrored) < 0.2


def test_share_and_azimuth_are_frame_relative():
    assert abs(area_share(bbox_for(0.04), FRAME) - 0.04) < 1e-9
    assert azimuth_deg(640, 1280) == 0.0
    assert azimuth_deg(1280, 1280) > 30 and azimuth_deg(0, 1280) < -30


def test_frontal_face_at_desk_range_passes():
    ok, margins = looking_at_me(kps(), bbox_for(0.05), FRAME, 0.9)
    assert ok, margins
    assert abs(margins["yaw_deg"]) < 1 and margins["area_share"] > 0.04


def test_the_reject_matrix_is_all_fail_closed():
    for what, face_kps, box, score, engaged in [
        ("profile at a laptop", kps(yaw=25.0), bbox_for(0.05), 0.9, False),
        ("head down at a phone", kps(pitch=-20.0), bbox_for(0.05), 0.9, False),
        ("sideways head", kps(roll=30.0), bbox_for(0.05), 0.9, False),
        ("far side of the room", kps(), bbox_for(0.02), 0.9, False),
        ("nose in frame", kps(), bbox_for(0.7), 0.9, False),
        ("frame edge", kps(), bbox_for(0.05, cx=60.0), 0.9, False),
        ("uncertain detection", kps(), bbox_for(0.05), 0.65, False),
    ]:
        ok, _margins = looking_at_me(face_kps, box, FRAME, score, engaged=engaged)
        assert not ok, f"{what} should not read as looking at me"


def test_hysteresis_holds_a_face_the_strict_gate_would_drop():
    """Just outside the strict band: rejected cold, kept once a track holds it."""
    marginal = kps(yaw=21.0)  # outside the strict 18 deg, inside the relaxed 25.7 deg
    box = bbox_for(0.05)
    assert not looking_at_me(marginal, box, FRAME, 0.9)[0]
    assert looking_at_me(marginal, box, FRAME, 0.9, engaged=True)[0]
    # 30 deg is outside even the relaxed band: a face scanning the room never dwels
    assert not looking_at_me(kps(yaw=30.0), box, FRAME, 0.9, engaged=True)[0]


def test_gate_constants_are_the_plan_numbers():
    gate = Gate()
    assert (gate.max_yaw_deg, gate.max_pitch_deg, gate.max_roll_deg) == (18.0, 15.0, 20.0)
    # 0.04 was the plan's guess; 0.025 is tonight's measurement. In data/events.db, 13
    # `face.present` rows from the live hackday room read min 0.0227 / median 0.0347 / max
    # 0.0703 - 7 of 13 fell under the plan number, which is why `greeter.fired` and
    # `greeter.suppressed` were both zero all evening: no face ever got a qualified track,
    # so there was never a verdict to suppress. False positive = one extra hello (capped 6/h,
    # 120 s apart); false negative = a robot that ignores the room it is standing in.
    assert gate.min_area_share == 0.025 and gate.min_det_score == 0.70
    assert gate.relaxed().max_yaw_deg > gate.max_yaw_deg  # hysteresis loosens, never tightens
    assert gate.relaxed().min_area_share < gate.min_area_share
