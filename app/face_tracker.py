import logging
import threading
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis

log = logging.getLogger(__name__)

EMA_ALPHA = 0.25
DECAY_ALPHA = 0.03
MOVEMENT_HZ = 60
OVERSHOOT_SCALE = 0.85
FACE_LOST_TIMEOUT = 0.5


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

    def face_pixel_center(self, face: dict, frame_shape: tuple) -> tuple[int, int]:
        """Return face center as (u, v) pixel coordinates."""
        cx, cy = face["center"]
        h, w = frame_shape[:2]
        return (int(cx * w), int(cy * h))

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

    def run_tracking_loop(self, robot_mini, stop_event: threading.Event | None = None):
        """Continuously track the largest face and move the robot head to follow it.

        Uses a decoupled two-thread architecture for smooth movement:
        - Detection thread: runs face detection as fast as it can and updates
          a shared target pose via look_at_image().
        - Movement thread: runs at MOVEMENT_HZ, applies EMA smoothing to the
          latest target pose, and sends updates to the robot servos.

        When no face is detected for FACE_LOST_TIMEOUT seconds, the head
        slowly decays back to center.
        Runs until stop_event is set. Call from a thread.
        """
        from reachy_mini.utils import create_head_pose

        if stop_event is None:
            stop_event = threading.Event()

        self.open_camera()
        center_pose = create_head_pose(yaw=0, pitch=0, degrees=True)

        # Shared state between detection and movement threads
        lock = threading.Lock()
        shared = {
            "target_pose": None,
            "last_detection_time": 0.0,
        }

        def _detection_loop():
            """Run face detection continuously, updating the shared target pose."""
            log.info("Detection thread started")
            while not stop_event.is_set():
                frame = self.grab_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                faces = self.detect(frame)
                if faces:
                    u, v = self.face_pixel_center(faces[0], frame.shape)
                    raw_pose = robot_mini.look_at_image(u, v, perform_movement=False)
                    # Scale toward center to reduce overshoot
                    scaled_pose = OVERSHOOT_SCALE * raw_pose + (1 - OVERSHOOT_SCALE) * center_pose
                    with lock:
                        shared["target_pose"] = scaled_pose
                        shared["last_detection_time"] = time.monotonic()
            log.info("Detection thread stopped")

        def _movement_loop():
            """Send smoothed pose updates to the robot at a steady rate."""
            log.info("Movement thread started at %d Hz", MOVEMENT_HZ)
            smooth_pose = center_pose.copy()
            interval = 1.0 / MOVEMENT_HZ

            while not stop_event.is_set():
                start = time.monotonic()

                with lock:
                    target_pose = shared["target_pose"]
                    last_det = shared["last_detection_time"]

                now = time.monotonic()
                if target_pose is not None and (now - last_det) < FACE_LOST_TIMEOUT:
                    # Face is being tracked — smooth toward target
                    smooth_pose = EMA_ALPHA * target_pose + (1 - EMA_ALPHA) * smooth_pose
                else:
                    # No face detected recently — decay toward center
                    smooth_pose = DECAY_ALPHA * center_pose + (1 - DECAY_ALPHA) * smooth_pose

                robot_mini.set_target(head=smooth_pose)

                elapsed = time.monotonic() - start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            log.info("Movement thread stopped")

        detection_thread = threading.Thread(target=_detection_loop, name="face-detection", daemon=True)
        movement_thread = threading.Thread(target=_movement_loop, name="face-movement", daemon=True)

        log.info("Face tracking started (detection + %d Hz movement)", MOVEMENT_HZ)
        detection_thread.start()
        movement_thread.start()

        try:
            # Block until stop is requested, then let threads wind down
            stop_event.wait()
            detection_thread.join(timeout=2.0)
            movement_thread.join(timeout=2.0)
        finally:
            self.close_camera()
            log.info("Face tracking stopped")
