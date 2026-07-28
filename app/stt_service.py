import logging
import time

import numpy as np

from .config import Config

log = logging.getLogger(__name__)


class STTService:
    """Speech-to-text via Parakeet TDT 0.6B v3 (ONNX, CPU).

    Runs on CPU deliberately: the GTX 1070 has 8GB and llama.cpp needs all of it.
    """

    def __init__(self, config: Config):
        import onnx_asr

        log.info("Loading Parakeet TDT 0.6B v3 (CPU, ONNX)...")
        self._model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
        log.info("Parakeet ready")

    def transcribe(self, audio: np.ndarray) -> str:
        start = time.monotonic()
        text = self._model.recognize(audio, sample_rate=16000)
        log.info("STT %.2fs: %r", time.monotonic() - start, text)
        return text
