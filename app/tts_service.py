import io
import logging
import os
import time
import wave

import numpy as np
import requests

from .config import Config

log = logging.getLogger(__name__)

TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")
TTS_SERVER_URL = os.getenv("TTS_SERVER_URL", "http://127.0.0.1:5100")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "Ryan")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "bm_george")
KOKORO_LANG = os.getenv("KOKORO_LANG", "b")


class TTSService:
    def __init__(self, config: Config):
        if TTS_ENGINE == "qwen3":
            self._init_qwen3()
        elif TTS_ENGINE == "kokoro":
            self._init_kokoro()
        else:
            self._init_piper(config)

    def _init_qwen3(self):
        log.info("Using Qwen3-TTS server at %s (speaker=%s)", TTS_SERVER_URL, TTS_SPEAKER)
        self._engine = "qwen3"
        self.sample_rate = 24000

    def _init_kokoro(self):
        from kokoro import KPipeline

        log.info("Loading Kokoro TTS (voice=%s, lang=%s)", KOKORO_VOICE, KOKORO_LANG)
        self._pipeline = KPipeline(lang_code=KOKORO_LANG, device="cpu")
        self._voice = KOKORO_VOICE
        self._engine = "kokoro"
        self.sample_rate = 24000

    def _init_piper(self, config: Config):
        from piper.voice import PiperVoice

        log.info("Loading Piper voice: %s", config.piper_voice_path)
        self._piper = PiperVoice.load(config.piper_voice_path)
        self._engine = "piper"
        self.sample_rate = self._piper.config.sample_rate

    def synthesize(self, text: str, instruct: str = "") -> np.ndarray:
        start = time.monotonic()
        if self._engine == "qwen3":
            audio = self._synthesize_qwen3(text, instruct)
        elif self._engine == "kokoro":
            audio = self._synthesize_kokoro(text)
        else:
            audio = self._synthesize_piper(text)
        log.info("TTS %.2fs for %d chars", time.monotonic() - start, len(text))
        return audio

    def _synthesize_qwen3(self, text: str, instruct: str = "") -> np.ndarray:
        resp = requests.post(
            f"{TTS_SERVER_URL}/synthesize",
            json={"text": text, "speaker": TTS_SPEAKER, "instruct": instruct, "language": "English"},
            timeout=30,
        )
        if resp.status_code != 200:
            log.error("TTS server error: %s", resp.text)
            return np.array([], dtype=np.int16)
        buf = io.BytesIO(resp.content)
        with wave.open(buf) as wf:
            return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    def _synthesize_kokoro(self, text: str) -> np.ndarray:
        chunks = []
        for _graphemes, _phonemes, audio in self._pipeline(text, voice=self._voice):
            chunks.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
        if not chunks:
            return np.array([], dtype=np.int16)
        audio_f = np.concatenate(chunks)
        return np.clip(audio_f * 32767, -32768, 32767).astype(np.int16)

    def _synthesize_piper(self, text: str) -> np.ndarray:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._piper.synthesize_wav(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
