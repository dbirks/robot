import logging
import time

import numpy as np
from faster_whisper import WhisperModel

from .config import Config

log = logging.getLogger(__name__)


class STTService:
    def __init__(self, config: Config):
        log.info(
            "Loading Whisper model: %s on %s (%s)",
            config.whisper_model,
            config.whisper_device,
            config.whisper_compute_type,
        )
        self.model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        start = time.monotonic()
        segments, _info = self.model.transcribe(audio, language="en", vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed = time.monotonic() - start
        log.info("STT %.2fs: %r", elapsed, text)
        return text
