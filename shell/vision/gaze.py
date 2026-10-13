"""The numeric "they are looking at me" gate: geometry only, no identity, no storage.

`buffalo_sc` does not populate `face.pose` (verified in the B0/B3 scout: `'pose' in
face` is False), but its detector does return five keypoints (both eyes, nose tip,
both mouth corners), so yaw/pitch/roll come from their geometry with zero extra
model and zero extra milliseconds — a gaze model (OpenGaze/L2CS-Net) would be a
second resident model and degrades at desk distance, which the brief refuses.

The gains below are proxies for head orientation, calibrated against nothing yet:
the camera is head-mounted, so the sign/scale depend on how the mount sits (plan
§3, open question 4). They are constants here so the bench pass tunes one place
with `REACHY_GREETER_DEBUG=1`, which logs the margins of every gate. Until then
every threshold is fail-closed, i.e. an unreadable face is not a greeting.

Hysteresis (`Gate.hysteresis`) relaxes the angular and area bounds for a face this
poll already holds a track on, so a person who is looking at him and blinks or
shifts does not lose their dwell; a face scanning a laptop across the room never
gets the relaxed band because it never gets a qualified track in the first place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

# insightface/SCRFD keypoint order
KP_LEFT_EYE, KP_RIGHT_EYE, KP_NOSE, KP_LEFT_MOUTH, KP_RIGHT_MOUTH = 0, 1, 2, 3, 4

# kps -> degrees gains (bench-tunable, see the module docstring)
YAW_GAIN = 1.5  # nose offset from the eye midpoint, in inter-ocular units
PITCH_GAIN = 1.0  # nose tip above/below the eye-mouth midpoint

# bbox centre -> the azimuth we greet/steer toward. The frame's horizontal edge is
# taken as AZIMUTH_HALF_DEG off the nose, matching motion's MAX_YAW_DEG ballpark.
AZIMUTH_HALF_DEG = 40.0

Point = Sequence[float]


@dataclass(frozen=True)
class Gate:
    """Every number in the acceptance rule (plan §3), as data so tests and the
    debug log can see one source of truth."""

    min_det_score: float = 0.70
    max_yaw_deg: float = 18.0
    max_pitch_deg: float = 15.0
    max_roll_deg: float = 20.0
    # 0.04 was a guess about desk range that cost the booth every greeting it could have
    # made. Measured live, hackday, 2026-10-13 (13 `face.present` rows in data/events.db):
    # area_share min 0.0227, MEDIAN 0.0347, max 0.0703 - so 7 of 13 real faces seen in the
    # room that evening sat below the old floor and never got a qualified track at all, which
    # is why `greeter.fired` and `greeter.suppressed` were BOTH zero all evening: not one turn
    # ever reached the verdict. 0.025 admits 11 of those 13 samples and still rejects the two
    # most peripheral. Asymmetry is deliberate: a false positive here is one extra hello, capped
    # at 6/hour with 120 s spacing; a false negative is a robot that ignores the room. The
    # max_area_share guard below is what stops an operator's nose in frame from counting.
    min_area_share: float = 0.025  # measured median 0.0347; area is the only distance proxy
    max_area_share: float = 0.60  # not an operator's nose in frame
    centre_band: float = 0.60  # bbox centre inside the central 60 % of the frame
    hysteresis: float = 0.70  # relax multiplier applied to a track already locked on

    def relaxed(self) -> Gate:
        """The engaged band: every bound loosened by `hysteresis` (0.7 -> 1/0.7 =
        1.43x wider angles, 0.7x the area floor). A face the poll already holds a
        qualified track on may shift, lean or blink without losing its dwell; a
        face that was never qualified never gets this band, so a head scanning the
        room still cannot accumulate two seconds."""
        h = self.hysteresis
        return Gate(
            min_det_score=self.min_det_score,
            max_yaw_deg=self.max_yaw_deg / h,
            max_pitch_deg=self.max_pitch_deg / h,
            max_roll_deg=self.max_roll_deg / h,
            min_area_share=self.min_area_share * h,
            max_area_share=self.max_area_share / h,
            centre_band=self.centre_band / h,
            hysteresis=h,
        )


def _xy(kps: Sequence[Any], index: int) -> tuple[float, float]:
    point = kps[index]
    return float(point[0]), float(point[1])


def estimate_orientation(
    kps: Sequence[Any], bbox: Sequence[float] | None = None, frame_shape: Sequence[int] | None = None
) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from the five keypoints.

    Sign conventions, as seen in the image: yaw positive = nose points to the
    right of the frame, pitch positive = head tipped up, roll positive = the
    right eye sits lower than the left. `bbox`/`frame_shape` are accepted for
    call-site symmetry with `looking_at_me` and are not needed here.
    """
    e0 = _xy(kps, KP_LEFT_EYE)
    e1 = _xy(kps, KP_RIGHT_EYE)
    nose = _xy(kps, KP_NOSE)
    m0 = _xy(kps, KP_LEFT_MOUTH)
    m1 = _xy(kps, KP_RIGHT_MOUTH)

    dx, dy = e1[0] - e0[0], e1[1] - e0[1]
    roll = math.degrees(math.atan2(dy, dx))
    if roll > 90.0:  # the two eyes are not guaranteed to arrive left-to-right
        roll -= 180.0
    elif roll < -90.0:
        roll += 180.0
    interocular = math.hypot(dx, dy) or 1.0

    eye_mid_x, eye_mid_y = (e0[0] + e1[0]) / 2.0, (e0[1] + e1[1]) / 2.0
    yaw = YAW_GAIN * math.degrees(math.atan2(nose[0] - eye_mid_x, interocular))

    mouth_mid_x, mouth_mid_y = (m0[0] + m1[0]) / 2.0, (m0[1] + m1[1]) / 2.0
    vertical = abs(mouth_mid_y - eye_mid_y) or 1.0
    neutral_y = (mouth_mid_y + eye_mid_y) / 2.0  # where an unpitched nose tip sits
    pitch = PITCH_GAIN * math.degrees(math.atan2(neutral_y - nose[1], vertical / 2.0))
    return yaw, pitch, roll


def azimuth_deg(center_x: float, frame_width: float) -> float:
    """Bbox centre to a head/body azimuth: 0 = dead ahead, positive = to the right."""
    if frame_width <= 0:
        return 0.0
    offset = (float(center_x) / float(frame_width)) - 0.5
    return max(-90.0, min(90.0, 2.0 * AZIMUTH_HALF_DEG * offset))


def area_share(bbox: Sequence[float], frame_shape: Sequence[int]) -> float:
    """Face area over frame area: the only distance proxy there is (no depth sensor)."""
    height, width = float(frame_shape[0]), float(frame_shape[1])
    if height <= 0 or width <= 0:
        return 0.0
    face_area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    return face_area / (height * width)


def looking_at_me(
    kps: Sequence[Any],
    bbox: Sequence[float],
    frame_shape: Sequence[int],
    det_score: float,
    *,
    gate: Gate | None = None,
    engaged: bool = False,
) -> tuple[bool, dict[str, float]]:
    """The gate, and the margins that produced the verdict (for the debug log).

    `engaged` applies the hysteresis band; `margins` always reports against the
    band actually in force, so `REACHY_GREETER_DEBUG=1` shows how far a near-miss
    was rather than hiding it behind a bool.
    """
    gate = gate or Gate()
    active = gate.relaxed() if engaged else gate
    yaw, pitch, roll = estimate_orientation(kps, bbox, frame_shape)
    share = area_share(bbox, frame_shape)
    centre_x = (float(bbox[0]) + float(bbox[2])) / 2.0
    edge = abs(centre_x / float(frame_shape[1]) - 0.5) if frame_shape[1] else 1.0
    margins = {
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "area_share": share,
        "edge_offset": edge,
        "det_score": float(det_score),
        "max_yaw_deg": active.max_yaw_deg,
        "max_pitch_deg": active.max_pitch_deg,
        "max_roll_deg": active.max_roll_deg,
        "min_area_share": active.min_area_share,
        "max_area_share": active.max_area_share,
        "centre_band": active.centre_band,
        "hysteresis": 1.0 if not engaged else active.hysteresis,
    }
    ok = (
        det_score >= active.min_det_score
        and abs(yaw) <= active.max_yaw_deg
        and abs(pitch) <= active.max_pitch_deg
        and abs(roll) <= active.max_roll_deg
        and active.min_area_share <= share <= active.max_area_share
        and edge <= active.centre_band / 2.0
    )
    return ok, margins
