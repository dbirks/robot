import logging
import os
import time

import numpy as np

from .config import Config

log = logging.getLogger(__name__)

STT_ENGINE = os.getenv("STT_ENGINE", "parakeet")


class STTService:
    def __init__(self, config: Config):
        if STT_ENGINE == "parakeet":
            self._init_parakeet()
        else:
            self._init_whisper(config)

    def _init_parakeet(self):
        import onnx_asr

        log.info("Loading Parakeet TDT 0.6B v3 (CPU, ONNX)...")
        self._model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
        self._engine = "parakeet"
        log.info("Parakeet ready")

    def _init_whisper(self, config: Config):
        from faster_whisper import WhisperModel

        log.info(
            "Loading Whisper model: %s on %s (%s)",
            config.whisper_model,
            config.whisper_device,
            config.whisper_compute_type,
        )
        self._model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )
        self._engine = "whisper"

    def transcribe(self, audio: np.ndarray) -> str:
        start = time.monotonic()
        if self._engine == "parakeet":
            text = self._transcribe_parakeet(audio)
        else:
            text = self._transcribe_whisper(audio)
        elapsed = time.monotonic() - start
        log.info("STT %.2fs: %r", elapsed, text)
        return text

    def _transcribe_parakeet(self, audio: np.ndarray) -> str:
        return self._model.recognize(audio, sample_rate=16000)

    def _transcribe_whisper(self, audio: np.ndarray) -> str:
        segments, _info = self._model.transcribe(audio, language="en", vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()
