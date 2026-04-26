import logging
import os
import threading
import time

import numpy as np
import sounddevice as sd

from .config import Config

log = logging.getLogger(__name__)

CHUNK_SAMPLES = 512
SILENCE_CHUNKS_THRESHOLD = 25  # ~800ms at 16kHz/512 samples


def _get_input_device() -> int | None:
    env = os.getenv("AUDIO_INPUT_DEVICE", "")
    if env:
        return int(env)
    for i, d in enumerate(sd.query_devices()):
        if "Reachy Mini" in d["name"] and d["max_input_channels"] > 0:
            return i
    return None


class AudioRecorder:
    """VAD-gated microphone recorder.

    Listens continuously via sounddevice. When silero-vad detects speech,
    buffers audio until speech ends, then returns the complete utterance.
    """

    def __init__(self, config: Config):
        self.sample_rate = config.sample_rate
        self.vad_threshold = config.vad_threshold
        silence_ms = config.vad_min_silence_ms
        self.silence_limit = max(1, int(silence_ms / (CHUNK_SAMPLES / config.sample_rate * 1000)))
        self._input_device = _get_input_device()
        if self._input_device is not None:
            log.info("Audio input: device %s", self._input_device)
        else:
            log.info("Audio input: system default")

        self._vad_model = None
        self._load_vad()

    def _load_vad(self):
        import torch

        self._vad_model, _utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
        )
        log.info("Silero VAD loaded")

    def record_utterance(self) -> np.ndarray | None:
        """Block until a complete speech utterance is captured. Returns float32 audio at self.sample_rate."""
        import torch

        buffer: list[np.ndarray] = []
        is_speaking = False
        silence_count = 0

        log.debug("Listening...")

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16", blocksize=CHUNK_SAMPLES, device=self._input_device) as stream:
            while True:
                data, _overflowed = stream.read(CHUNK_SAMPLES)
                audio_int16 = data[:, 0]
                audio_f32 = audio_int16.astype(np.float32) / 32768.0

                confidence = self._vad_model(torch.from_numpy(audio_f32), self.sample_rate).item()

                if confidence >= self.vad_threshold:
                    if not is_speaking:
                        log.debug("Speech started")
                        is_speaking = True
                        silence_count = 0
                    buffer.append(audio_f32)
                    silence_count = 0
                elif is_speaking:
                    buffer.append(audio_f32)
                    silence_count += 1
                    if silence_count >= self.silence_limit:
                        log.debug("Speech ended (%.1fs)", len(buffer) * CHUNK_SAMPLES / self.sample_rate)
                        break

        if not buffer:
            return None

        return np.concatenate(buffer)
