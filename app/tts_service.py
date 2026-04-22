import io
import logging
import time
import wave
from typing import Iterator

import numpy as np
from piper.voice import PiperVoice

from .config import Config

log = logging.getLogger(__name__)


class TTSService:
    def __init__(self, config: Config):
        log.info("Loading Piper voice: %s", config.piper_voice_path)
        self.voice = PiperVoice.load(config.piper_voice_path)
        self.sample_rate: int = self.voice.config.sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        start = time.monotonic()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self.voice.synthesize(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        log.info("TTS %.2fs for %d chars", time.monotonic() - start, len(text))
        return audio

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        for chunk in self.voice.synthesize_stream_raw(text):
            yield np.frombuffer(chunk, dtype=np.int16)
