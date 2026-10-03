"""Configuration for the realtime shell. Deliberately env-only and minimal;
attention tuning lives in versioned profiles, model identity in docs/pins.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_INSTRUCTIONS = (
    "You are Reachy, a small desk robot. Reply in one or two short spoken "
    "sentences. Never use markdown or lists. Be concise, warm, understated, "
    "and slightly charming."
)


@dataclass
class ShellConfig:
    # Realtime service (pinned huggingface/speech-to-speech, ADR 0001)
    s2s_url: str = field(default_factory=lambda: os.getenv("S2S_URL", "ws://127.0.0.1:8765/v1/realtime"))
    instructions: str = field(default_factory=lambda: os.getenv("REACHY_INSTRUCTIONS", DEFAULT_INSTRUCTIONS))

    # Attention
    attention_profile: str = field(default_factory=lambda: os.getenv("REACHY_PROFILE", "quiet"))

    # Audio (s2s PIPELINE_SAMPLE_RATE, both directions)
    sample_rate: int = 16000
    block_size: int = 512  # 32 ms at 16k
    channels: int = 1
    preroll_seconds: float = 1.0
    mic_device: str | None = field(default_factory=lambda: os.getenv("REACHY_MIC") or None)
    speaker_device: str | None = field(default_factory=lambda: os.getenv("REACHY_SPEAKER") or None)

    # Paths
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    kws_model_dir: str = field(default_factory=lambda: os.getenv("KWS_MODEL_DIR", "models/kws"))

    @property
    def journal_path(self) -> Path:
        return self.data_dir / "events.db"
