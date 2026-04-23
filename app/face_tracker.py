import logging
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis

log = logging.getLogger(__name__)

FRAME_CENTER_X = 0.5
FRAME_CENTER_Y = 0.4
DEAD_ZONE = 0.08
MAX_YAW = 35
MAX_PITCH = 20


class FaceTracker:
    """Tracks the most prominent face in camera frames using InsightFace.

    Provides head yaw/pitch targets to keep the face centered in frame.
    Optionally stores face embeddings for recognition.
    """

    def __init__(self, camera_index: int = 0, det_size: tuple[int, int] = (640, 640)):
        log.info("Loading InsightFace buffalo_sc...")
        self.app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=det_size)
        self.camera_index = camera_index
        self.known_faces: dict[str, np.ndarray] = {}
        self._cap: cv2.VideoCapture | None = None

    def open_camera(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            log.error("Failed to open camera %d", self.camera_index)
            self._cap = None

    def close_camera(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def grab_frame(self) -> np.ndarray | None:
        if self._cap is None:
            self.open_camera()
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def detect(self, frame: np.ndarray) -> list[dict]:
        faces = self.app.get(frame)
        h, w = frame.shape[:2]
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            cx = (bbox[0] + bbox[2]) / 2 / w
            cy = (bbox[1] + bbox[3]) / 2 / h
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (w * h)
            results.append(
                {
                    "bbox": bbox.tolist(),
                    "center": (cx, cy),
                    "area": area,
                    "score": float(face.det_score),
                    "embedding": face.embedding,
                }
            )
        results.sort(key=lambda f: f["area"], reverse=True)
        return results

    def compute_head_target(self, face: dict) -> tuple[float, float]:
        """Return (yaw, pitch) in degrees to center the face in frame."""
        cx, cy = face["center"]
        dx = cx - FRAME_CENTER_X
        dy = cy - FRAME_CENTER_Y

        if abs(dx) < DEAD_ZONE:
            dx = 0
        if abs(dy) < DEAD_ZONE:
            dy = 0

        yaw = -dx * MAX_YAW * 2
        pitch = -dy * MAX_PITCH * 2
        return (np.clip(yaw, -MAX_YAW, MAX_YAW), np.clip(pitch, -MAX_PITCH, MAX_PITCH))

    def register_face(self, name: str, embedding: np.ndarray):
        self.known_faces[name] = embedding / np.linalg.norm(embedding)
        log.info("Registered face: %s", name)

    def identify(self, embedding: np.ndarray, threshold: float = 0.4) -> str | None:
        if not self.known_faces:
            return None
        emb_norm = embedding / np.linalg.norm(embedding)
        best_name, best_sim = None, -1.0
        for name, known_emb in self.known_faces.items():
            sim = float(np.dot(emb_norm, known_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= threshold:
            return best_name
        return None
