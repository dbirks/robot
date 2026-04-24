import logging
import math
import queue
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

ACTIVATE_DBFS = -35
DEACTIVATE_DBFS = -45
SMOOTHING = 0.3
DECAY_RATE = 3.0
SILENCE_TIMEOUT = 0.3


class HeadWobbler:
    """Converts TTS audio amplitude into sinusoidal head movement offsets.

    Runs a worker thread that processes PCM audio chunks, computes RMS
    loudness, and produces 6-DOF offsets (x, y, z, roll, pitch, yaw)
    delivered to a MovementManager via callback.

    Uses a hysteretic VAD gate to avoid jitter during quiet passages.
    """

    def __init__(self, callback):
        self._callback = callback
        self._audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._smooth_amp = 0.0
        self._last_speech = 0.0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="head-wobbler", daemon=True)
        self._thread.start()
        log.info("HeadWobbler started")

    def stop(self):
        self._stop.set()
        self._audio_queue.put(None)
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("HeadWobbler stopped")

    def feed(self, audio_int16: np.ndarray):
        self._audio_queue.put(audio_int16)

    def reset(self):
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        self._active = False
        self._smooth_amp = 0.0
        self._callback((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def _run(self):
        while not self._stop.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                now = time.monotonic()
                if self._smooth_amp > 0.001 and (now - self._last_speech) > SILENCE_TIMEOUT:
                    dt = 0.1
                    self._smooth_amp *= math.exp(-DECAY_RATE * dt)
                    if self._smooth_amp < 0.001:
                        self._smooth_amp = 0.0
                        self._active = False
                        self._callback((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                    else:
                        self._emit_offsets(time.monotonic())
                continue

            if chunk is None:
                break

            audio_f32 = chunk.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio_f32**2))) + 1e-10
            dbfs = 20 * math.log10(rms)

            if not self._active and dbfs >= ACTIVATE_DBFS:
                self._active = True
            elif self._active and dbfs < DEACTIVATE_DBFS:
                self._active = False

            if self._active:
                self._smooth_amp = SMOOTHING * rms + (1 - SMOOTHING) * self._smooth_amp
                self._last_speech = time.monotonic()
            else:
                now = time.monotonic()
                if (now - self._last_speech) > SILENCE_TIMEOUT:
                    self._smooth_amp *= math.exp(-DECAY_RATE * (1 / 50))

            self._emit_offsets(time.monotonic())

        self._callback((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        log.info("HeadWobbler loop stopped")

    def _emit_offsets(self, t: float):
        amp = self._smooth_amp
        if amp < 0.001:
            self._callback((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            return

        z = amp * 0.008 * math.sin(t * 8.0)
        roll = amp * 0.15 * math.sin(t * 3.0)
        pitch = amp * 0.08 * math.sin(t * 5.0 + 0.5)
        yaw = amp * 0.05 * math.sin(t * 2.0)

        self._callback((0.0, 0.0, z, roll, pitch, yaw))
