import json
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

log = logging.getLogger(__name__)

OVERSHOOT_SCALE = 0.6
MAX_YAW = 45
MAX_PITCH = 25
FACES_PATH = Path("data/known_faces.json")


class FaceTracker:
    """Tracks the most prominent face in camera frames using InsightFace.

    Provides head yaw/pitch targets to keep the face centered in frame.
    Stores face embeddings for recognition, persisted to disk.
    """

    def __init__(self, camera_index: int = 0, det_size: tuple[int, int] = (640, 640)):
        log.info("Loading InsightFace buffalo_sc...")
        self.app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=det_size)
        self.camera_index = camera_index
        self.known_faces: dict[str, np.ndarray] = {}
        self._cap: cv2.VideoCapture | None = None
        self._load_faces()

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

    def face_pixel_center(self, face: dict, frame_shape: tuple) -> tuple[int, int]:
        """Return face center as (u, v) pixel coordinates."""
        cx, cy = face["center"]
        h, w = frame_shape[:2]
        return (int(cx * w), int(cy * h))

    def register_face(self, name: str, embedding: np.ndarray):
        self.known_faces[name] = embedding / np.linalg.norm(embedding)
        self._save_faces()
        log.info("Registered face: %s", name)

    def identify(self, embedding: np.ndarray, threshold: float = 0.3) -> str | None:
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

    def start_tracking(self, robot_mini, movement_manager, stop_event: threading.Event | None = None):
        """Start face detection in a background thread.

        Detects faces as fast as possible and feeds target poses to the
        MovementManager, which handles smoothing and servo commands.
        """
        from reachy_mini.utils import create_head_pose

        if stop_event is None:
            stop_event = threading.Event()

        self._stop_event = stop_event
        self.open_camera()
        center_pose = create_head_pose(yaw=0, pitch=0, degrees=True)

        use_sdk_lookat = True

        def _pose_from_pixels(cx_norm, cy_norm):
            yaw = -(cx_norm - 0.5) * 2 * MAX_YAW
            pitch = (cy_norm - 0.5) * 2 * MAX_PITCH
            return create_head_pose(yaw=yaw, pitch=pitch, degrees=True)

        def _detection_loop():
            nonlocal use_sdk_lookat
            log.info("Face detection thread started")
            while not stop_event.is_set():
                frame = self.grab_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                faces = self.detect(frame)
                if faces:
                    face = faces[0]
                    if use_sdk_lookat:
                        try:
                            u, v = self.face_pixel_center(face, frame.shape)
                            raw_pose = robot_mini.look_at_image(u, v, perform_movement=False)
                        except RuntimeError:
                            log.info("look_at_image unavailable, using pixel-based tracking")
                            use_sdk_lookat = False
                            raw_pose = _pose_from_pixels(*face["center"])
                    else:
                        raw_pose = _pose_from_pixels(*face["center"])
                    scaled_pose = OVERSHOOT_SCALE * raw_pose + (1 - OVERSHOOT_SCALE) * center_pose
                    movement_manager.set_face_target(scaled_pose)
            log.info("Face detection thread stopped")

        self._detection_thread = threading.Thread(target=_detection_loop, name="face-detection", daemon=True)
        self._detection_thread.start()
        log.info("Face tracking started (detection → MovementManager)")

    def stop_tracking(self):
        if hasattr(self, "_stop_event") and self._stop_event:
            self._stop_event.set()
        if hasattr(self, "_detection_thread") and self._detection_thread:
            self._detection_thread.join(timeout=2.0)
        self.close_camera()
        log.info("Face tracking stopped")

    def _load_faces(self):
        if not FACES_PATH.exists():
            return
        try:
            data = json.loads(FACES_PATH.read_text())
            for name, emb_list in data.items():
                self.known_faces[name] = np.array(emb_list, dtype=np.float32)
            log.info("Loaded %d known faces from disk", len(self.known_faces))
        except Exception:
            log.exception("Failed to load known faces")

    def _save_faces(self):
        FACES_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {name: emb.tolist() for name, emb in self.known_faces.items()}
        FACES_PATH.write_text(json.dumps(data))
        log.info("Saved %d known faces to disk", len(self.known_faces))
