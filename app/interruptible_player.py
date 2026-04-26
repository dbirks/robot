"""Non-blocking audio playback with barge-in detection.

Plays audio through an OutputStream callback while monitoring the mic
via VAD. If sustained speech is detected from the front hemisphere,
playback stops and the captured speech is returned for processing.
"""

import logging
import math
import threading

import numpy as np
import requests
import sounddevice as sd

log = logging.getLogger(__name__)

CHUNK_SAMPLES = 512
BARGE_IN_FRAMES = 15  # ~480ms of sustained speech to confirm interrupt
DOA_URL = "http://localhost:8000/api/state/doa"
DOA_FRONT_MIN = math.pi / 4
DOA_FRONT_MAX = 3 * math.pi / 4
WOBBLE_CHUNK = 1600


class InterruptiblePlayer:
    def __init__(self, vad_model, sample_rate=16000, vad_threshold=0.5, output_device=None, output_sr=None):
        self._vad = vad_model
        self._sr = sample_rate
        self._vad_thresh = vad_threshold
        self._out_device = output_device
        self._out_sr = output_sr

    def play(self, audio_int16: np.ndarray, sample_rate: int, wobbler=None) -> tuple[bool, np.ndarray | None]:
        """Play audio interruptibly. Returns (was_interrupted, captured_speech_f32)."""
        from .playback import _apply_boost, _resample

        audio = _apply_boost(audio_int16)
        play_sr = sample_rate
        if self._out_sr is not None and sample_rate != self._out_sr:
            audio = _resample(audio, sample_rate, self._out_sr)
            play_sr = self._out_sr

        if wobbler:
            for i in range(0, len(audio_int16), WOBBLE_CHUNK):
                wobbler.feed(audio_int16[i : i + WOBBLE_CHUNK])

        interrupted = threading.Event()
        done = threading.Event()
        pos = [0]

        def _callback(outdata, frames, _time_info, _status):
            start = pos[0]
            end = start + frames
            if interrupted.is_set() or start >= len(audio):
                outdata[:] = 0
                done.set()
                raise sd.CallbackStop
            chunk = audio[start:end]
            if len(chunk) < frames:
                outdata[: len(chunk), 0] = chunk
                outdata[len(chunk) :] = 0
                done.set()
                raise sd.CallbackStop
            outdata[:, 0] = chunk
            pos[0] = end

        stream = sd.OutputStream(
            samplerate=play_sr, channels=1, dtype="int16", device=self._out_device, callback=_callback
        )
        stream.start()

        barge_audio = self._monitor_mic(interrupted, done)

        stream.stop()
        stream.close()
        if wobbler:
            wobbler.reset()

        if interrupted.is_set() and barge_audio:
            captured = np.concatenate(barge_audio)
            log.info("Barge-in detected (%.1fs captured)", len(captured) * CHUNK_SAMPLES / self._sr)
            return True, captured
        return False, None

    def _monitor_mic(self, interrupted: threading.Event, done: threading.Event) -> list[np.ndarray]:
        """Monitor mic while playback runs. Returns captured speech chunks on barge-in."""
        import torch

        consecutive = 0
        barge_audio: list[np.ndarray] = []
        silence_after_barge = 0
        capturing = False

        with sd.InputStream(samplerate=self._sr, channels=1, dtype="int16", blocksize=CHUNK_SAMPLES) as mic:
            while not done.is_set():
                try:
                    data, _ = mic.read(CHUNK_SAMPLES)
                except Exception:
                    break
                audio_f32 = data[:, 0].astype(np.float32) / 32768.0
                conf = self._vad(torch.from_numpy(audio_f32), self._sr).item()

                if conf >= self._vad_thresh:
                    consecutive += 1
                    silence_after_barge = 0
                    if consecutive >= BARGE_IN_FRAMES and not capturing:
                        if self._is_from_front():
                            interrupted.set()
                            capturing = True
                    if capturing:
                        barge_audio.append(audio_f32)
                else:
                    if not capturing and consecutive > 0 and consecutive < BARGE_IN_FRAMES:
                        consecutive = 0
                    if capturing:
                        barge_audio.append(audio_f32)
                        silence_after_barge += 1
                        if silence_after_barge >= 25:
                            break

        # If interrupted, keep capturing until silence
        if interrupted.is_set() and not capturing:
            capturing = True
        if capturing:
            self._capture_remaining(barge_audio)

        return barge_audio

    def _capture_remaining(self, barge_audio: list[np.ndarray]):
        """Capture remaining speech after playback stops."""
        import torch

        silence_count = 0
        with sd.InputStream(samplerate=self._sr, channels=1, dtype="int16", blocksize=CHUNK_SAMPLES) as mic:
            while silence_count < 25:
                try:
                    data, _ = mic.read(CHUNK_SAMPLES)
                except Exception:
                    break
                audio_f32 = data[:, 0].astype(np.float32) / 32768.0
                conf = self._vad(torch.from_numpy(audio_f32), self._sr).item()
                barge_audio.append(audio_f32)
                if conf >= self._vad_thresh:
                    silence_count = 0
                else:
                    silence_count += 1

    def _is_from_front(self) -> bool:
        try:
            resp = requests.get(DOA_URL, timeout=0.1)
            if resp.status_code == 200:
                data = resp.json()
                if data and data.get("speech_detected"):
                    angle = data["angle"]
                    return DOA_FRONT_MIN <= angle <= DOA_FRONT_MAX
        except requests.RequestException:
            pass
        return True  # default to allowing interrupt if DOA unavailable
