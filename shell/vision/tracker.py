"""Detection-only face tracking for the shell: numpy in, `Face`s out.

Deliberately smaller than `app/face_tracker.py`, which it replaces for the new
stack. Three differences are load-bearing, not tidiness:

- it owns nothing. No `cv2.VideoCapture`, no thread, no robot: frames come from
  the one camera owner (shell/camera.py `CameraStream`, ADR 0002) and go back out
  as data. The legacy tracker held the device open permanently, which is exactly
  the contention the shell's camera contract exists to prevent.
- it detects, never recognises. `allowed_modules=["detection"]` means the ArcFace
  recognizer is not even loaded (measured: the model file is ignored at prepare),
  so no embedding of anybody is ever computed, held or written. Identity for the
  greeter is a geometric track (see greeter.py), per docs/environments.md B1.
- its threads are capped through onnxruntime session options, never
  `OMP_NUM_THREADS`. app/face_tracker.py:24-31 records why the env-var version of
  that fix was wrong (process-global, import-order dependent, landed on torch);
  ADR 0006 is the budget it protects.

Measured on this box beside the live stack (docs/pins.yaml `face_detection`):
detect-only 2.9-3.7 ms/frame at 320 px with 2 intra-op threads, so the 2 FPS idle
poll is well under 1 % of one core; the full pipeline (detect + recognise) would
have been 10.5-12 ms and is not paid at all here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .gaze import area_share, azimuth_deg

log = logging.getLogger("shell.vision.tracker")

DET_SIZE = (320, 320)  # measured: 160 px drops desk-range recall (plan §2), do not lower
INTRA_OP_THREADS = 2  # per-session cap (ADR 0006), never OMP_NUM_THREADS


@dataclass(frozen=True)
class Face:
    """One detection. No embedding field exists anywhere in shell/."""

    bbox: tuple[float, float, float, float]
    kps: tuple[tuple[float, float], ...]
    det_score: float
    share: float = 0.0  # bbox area / frame area (distance proxy)
    azimuth_deg: float = 0.0  # rough bearing of the bbox centre, 0 = dead ahead

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


@dataclass
class FaceTracker:
    """Lazy InsightFace `buffalo_sc` detector. `app` is injectable so the tests
    never load a model, and so a missing camera never costs ~1-2 s of import."""

    det_size: tuple[int, int] = DET_SIZE
    intra_op_threads: int = INTRA_OP_THREADS
    app: object | None = field(default=None, repr=False)

    def _engine(self):
        if self.app is None:
            from insightface.app import FaceAnalysis

            log.info("loading insightface buffalo_sc (detection only)")
            self.app = FaceAnalysis(
                name="buffalo_sc",
                providers=[
                    (
                        "CPUExecutionProvider",
                        {"intra_op_num_threads": self.intra_op_threads, "inter_op_num_threads": 1},
                    )
                ],
                allowed_modules=["detection"],
            )
            self.app.prepare(ctx_id=-1, det_size=self.det_size)
        return self.app

    def detect(self, frame: np.ndarray) -> list[Face]:
        """Faces, largest first. Empty list is a normal answer; a broken frame raises
        and the caller (the greeter poll) swallows it."""
        height, width = int(frame.shape[0]), int(frame.shape[1])
        faces: list[Face] = []
        raw: Sequence = self._engine().get(frame) or []
        for found in raw:
            box = [float(v) for v in found["bbox"]]
            bbox = (box[0], box[1], box[2], box[3])
            faces.append(
                Face(
                    bbox=bbox,
                    kps=tuple((float(p[0]), float(p[1])) for p in found["kps"]),
                    det_score=float(found["det_score"]),
                    share=area_share(bbox, (height, width)),
                    azimuth_deg=azimuth_deg((bbox[0] + bbox[2]) / 2.0, width),
                )
            )
        return sorted(faces, key=lambda f: -f.area)
