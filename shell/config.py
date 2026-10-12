"""Configuration for the realtime shell. Deliberately env-only and minimal;
attention tuning lives in versioned profiles, model identity in docs/pins.yaml.

Every value below resolves from code defaults + the process environment only.
The deployment-environment layer (docs/environments.md) sits on top of this in
shell/env.py: `envs/<env>.yaml` fills in what the process environment leaves
alone, so nothing here changes for anyone who only has a `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# docs/environments.md: REACHY_ENV picks the room. Always read from the real
# process environment - an env file can never select itself.
REACHY_ENV_VAR = "REACHY_ENV"
DEFAULT_ENV = "home"

DEFAULT_INSTRUCTIONS = (
    "You are Reachy, a small desk robot. Reply in one or two short spoken "
    "sentences. Never use markdown or lists. Be concise, warm, understated, "
    "and slightly charming."
)

# Appended to every persona. shell/llm_proxy.py stamps user messages with the
# time they were said; without this the model can't use (or might read) them.
TIME_NOTE = (
    " Each user message begins with the date and time it was said, like"
    " [Sunday, October 11, 2026, 5:38 PM]; the newest one is today's date and the current"
    " time - use it, never guess the date. Never read the stamps aloud. Answer only the newest message; if a"
    " lot of time has passed since the previous one, treat it as a fresh"
    " conversation and do not bring back old, unanswered topics."
)

# When to call tools. Granite 4.1 3B (2026-10-10/11) both over-calls (a bare
# play_emotion for "how's it going?", remember for "bye") and, when told to
# "speak first", narrates actions instead of calling them. Measured with this
# note and s2s's wrapper stripped (shell/llm_proxy.py): 35/38 on the tool bench.
# The narration ask (2026-10-11, user: "follow along with its progress
# audibly") also made Qwen actually call describe_scene/remember where it had
# been answering from imagination.
TOOLS_NOTE = (
    " Speak plain words: no emoji or action text like *laughs*."
    " Use a tool only when the user asks you to do something (move, nod, dance,"
    " play, change volume, sleep), asks what you see, asks for the time,"
    " weather, news or facts you are unsure of, or asks you to remember, recall"
    " or forget something. Also call remember on your own, without being asked,"
    " when the user tells you something lasting about themselves or their"
    " family: a name, who someone is, an age, where they live, a pet."
    " Greetings, small talk, jokes, feelings and goodbyes"
    " need no tool: just answer. Narrate tool use out loud so the user can"
    " follow along: when you call a tool, also say a few words about what you"
    " are doing, like 'Let me take a look.' or 'Turning left.' After each tool"
    " result, say one short sentence about what happened or what you found."
)

# Appended to every persona, in every deployment environment.
SHELL_NOTES = TOOLS_NOTE + TIME_NOTE


def persona_instructions(persona: str) -> str:
    """The one place a persona becomes system instructions.

    SHELL_NOTES is behaviour contract, not persona, so it appends to every persona
    in every environment (docs/environments.md schema v1: "TOOLS_NOTE/TIME_NOTE
    still append"). Name mis-hearings (Richie/Ricci) are fixed in
    shell/llm_proxy.py, not here: prompting the 4B model about them made it keep
    bringing it up.
    """
    return persona.rstrip() + SHELL_NOTES


def _env_name() -> str:
    return os.getenv(REACHY_ENV_VAR, "").strip() or DEFAULT_ENV


def _memory_namespace() -> str:
    # Empty = "follow the environment name", resolved in __post_init__ so that
    # ShellConfig(env="lab") namespaces as "lab" too. A new environment can then
    # never inherit another one's facts by omission (docs/environments.md rule 1).
    return os.getenv("REACHY_MEMORY_NAMESPACE", "").strip()


def _csv(name: str) -> list[str]:
    return [part.strip() for part in os.getenv(name, "").split(",") if part.strip()]


@dataclass
class ShellConfig:
    # Resolved deployment environment (docs/environments.md); stamped on every
    # journal event. Only shell/env.py writes it, from REACHY_ENV.
    env: str = field(default_factory=_env_name)

    # Realtime service (pinned huggingface/speech-to-speech, ADR 0001)
    s2s_url: str = field(default_factory=lambda: os.getenv("S2S_URL", "ws://127.0.0.1:8765/v1/realtime"))
    instructions: str = field(
        default_factory=lambda: persona_instructions(os.getenv("REACHY_INSTRUCTIONS", DEFAULT_INSTRUCTIONS))
    )

    # Tool subset for this environment (shell/tools.apply_tool_policy): empty
    # include = everything the shell knows about, exclude always wins.
    tools_include: list[str] = field(default_factory=lambda: _csv("REACHY_TOOLS_INCLUDE"))
    tools_exclude: list[str] = field(default_factory=lambda: _csv("REACHY_TOOLS_EXCLUDE"))

    # Which memories.<ns>.json this env reads and writes (shell/env, A5).
    # "" = follow `env`; __post_init__ resolves it.
    memory_namespace: str = field(default_factory=_memory_namespace)

    # Direct llama.cpp endpoint for side tasks (conversation compaction).
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "local"))
    compact_after_s: float = field(default_factory=lambda: float(os.getenv("REACHY_COMPACT_AFTER_S", "180")))
    reset_after_s: float = field(default_factory=lambda: float(os.getenv("REACHY_RESET_AFTER_S", "300")))

    # A user turn still open after this many seconds is cut by feeding the
    # service a moment of silence - the only endpoint it accepts from a
    # client (input_audio_buffer.commit is a no-op upstream). Bounds how much
    # background speech can pile into one turn.
    max_turn_s: float = field(default_factory=lambda: float(os.getenv("REACHY_MAX_TURN_S", "15")))

    # XVF3800 tuning (shell/audio/xmos_tuning.py): AGC gain ceiling (firmware
    # boots at 64x and pumps the TV up whenever the room goes quiet; 0 turns
    # AGC off) and steering both fixed beams at whoever said the wake word.
    xmos_agc_max_gain: float = field(default_factory=lambda: float(os.getenv("REACHY_XMOS_AGC_MAX_GAIN", "10")))
    xmos_beam_focus: bool = field(default_factory=lambda: os.getenv("REACHY_XMOS_BEAM_FOCUS", "1") != "0")

    # Attention
    attention_profile: str = field(default_factory=lambda: os.getenv("REACHY_PROFILE", "quiet"))

    # Audio (s2s PIPELINE_SAMPLE_RATE, both directions)
    sample_rate: int = 16000
    block_size: int = 512  # 32 ms at 16k
    channels: int = 1
    preroll_seconds: float = 1.0
    mic_device: str | None = field(default_factory=lambda: os.getenv("REACHY_MIC") or None)
    speaker_device: str | None = field(default_factory=lambda: os.getenv("REACHY_SPEAKER") or None)

    # Motion (shell/motion): one control loop; 50 Hz matches the daemon's
    # own control loop. Lower it if CPU is tight (ADR 0006).
    motion_hz: float = field(default_factory=lambda: float(os.getenv("REACHY_MOTION_HZ", "50")))

    # Paths
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    # Prerendered acknowledgements: <ack_dir>/wake/*.wav and <ack_dir>/think/*.wav,
    # recorded in the active voice (e.g. sounds/acks/obama).
    ack_dir: Path = field(default_factory=lambda: Path(os.getenv("REACHY_ACK_DIR", "sounds/acks")))
    kws_model_dir: str = field(default_factory=lambda: os.getenv("KWS_MODEL_DIR", "models/kws"))

    def __post_init__(self) -> None:
        if not self.memory_namespace:
            self.memory_namespace = self.env

    @property
    def journal_path(self) -> Path:
        return self.data_dir / "events.db"

    @property
    def memories_path(self) -> Path:
        """data/memories.<namespace>.json - one store per environment (A5)."""
        return self.data_dir / f"memories.{self.memory_namespace}.json"
