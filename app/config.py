import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # LLM
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen3.5-4b"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "not-needed"))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "256")))

    # STT
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "small.en"))
    whisper_device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cuda"))
    whisper_compute_type: str = field(default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "float16"))

    # TTS
    piper_voice_path: str = field(
        default_factory=lambda: os.getenv("PIPER_VOICE_PATH", "models/piper/en_GB-northern_english_male-medium.onnx")
    )

    # VAD
    vad_threshold: float = field(default_factory=lambda: float(os.getenv("VAD_THRESHOLD", "0.5")))
    vad_min_silence_ms: int = field(default_factory=lambda: int(os.getenv("VAD_MIN_SILENCE_MS", "800")))

    # Robot
    reachy_host: str = field(default_factory=lambda: os.getenv("REACHY_HOST", "localhost"))

    # Audio
    sample_rate: int = 16000
    audio_channels: int = 1

    # Paths
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "models")))
    logs_dir: Path = field(default_factory=lambda: Path(os.getenv("LOGS_DIR", "logs")))
