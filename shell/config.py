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
#
# What a stamp is FOR, settled 2026-10-12 against the booth rehearsal. Until now
# this note claimed the newest stamp is "the current time" while TOOLS_NOTE tells
# him to call a tool when asked for the time - so the same system prompt both said
# "the stamp is the clock" and "go check". Qwen 4B resolved the contradiction the
# worst way (`scripts/rehearse_booth.py`, 23:41 run): no `get_time` call, and the
# words "The clock indicates it is still late... I have checked the time, and it
# remains as it was." A bluff, because this note also forbids reading the stamp
# aloud, so the only time he was allowed to speak was the one he had not verified.
#
# The stamp is not a clock, and it cannot be one:
#   - shell/llm_proxy.py `Timestamper` keys the stamp on the message TEXT and reuses
#     the first-seen time for every re-send, so an identical line asked again after a
#     session reset (50 of them in data/events.db) is stamped with the OLD time. In
#     the journal, "hey richie." is re-used verbatim across resets 24.8 and 41.5 min
#     apart (2026-10-11 14:19 -> 15:00), which is the parent's 22:00/22:40 case: he
#     is not "picking the newest stamp", he is being handed a stale one.
#   - The stamp is also only ever when the guest SPOKE, not when he answers (a
#     tool-carrying turn ran 31.7 s behind its transcript, 2026-10-12 22:36:32).
# So the stamp keeps the two jobs it is good at - which day it is, and how long ago
# the last message was (2026-10-05: it answered a 40-minute-old dangling question)
# - and the spoken time comes from get_time, the one source that reads the clock at
# the moment he answers. Same local clock either way: `time.localtime` here vs
# `datetime.now().astimezone()` in app/robot_tools.py:get_time, whose "bare UTC ISO
# gave tomorrow's date every evening after 20:00 EDT" bug is why that tool spells
# the local zone out - and why the note below says to repeat its fields verbatim
# instead of letting a 4B re-derive a date from an offset it cannot see.
TIME_NOTE = (
    " Each user message begins with the date and time it was said, like"
    " [Sunday, October 11, 2026, 5:38 PM]; the newest one is how you know today's date and"
    " how long ago the guest spoke, so never guess the date. A stamp is not the time now: it"
    " can be minutes or hours old, so never state a time you only read off a stamp. When"
    " anyone asks the time, call get_time and say the date and time it returns exactly as it"
    " returns them - they are already the local clock, read just now. Never read the stamps"
    " aloud. Answer only the newest message; if a lot of time has passed since the previous"
    " one, treat it as a fresh conversation and do not bring back old, unanswered topics."
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

# The unprompted half of that note, kept as its own string so a room that turns
# automatic fact-learning OFF can also stop TELLING him to do it. Until 2026-10-12
# `behaviour.learn_facts: false` only disabled the miner in shell/conversation.py,
# while this sentence - shared code default, both rooms - still instructed him to
# save lasting facts about "the user or their family: a name, who someone is, an
# age, where they live, a pet" without being asked. At a public booth that is a
# stranger saying "my daughter is five" and him writing it down because a prompt
# told him to. This must stay an exact substring of TOOLS_NOTE (tested), so the
# home room stays byte-identical when the note is left in place.
LEARN_FACTS_NOTE = (
    "Also call remember on your own, without being asked, when the user tells you "
    "something lasting about themselves or their family: a name, who someone is, an "
    "age, where they live, a pet. "
)


def instructions_without_auto_learn(text: str) -> str:
    """Drop the unprompted-remember sentence from a prompt, leaving the explicit
    request half ("...or asks you to remember, recall or forget something") alone -
    an attendee asking to be remembered is a different act from volunteering it."""
    return text.replace(LEARN_FACTS_NOTE, "")


# Appended to every persona, in every deployment environment.
SHELL_NOTES = TOOLS_NOTE + TIME_NOTE


def without_shell_notes(persona: str) -> str:
    """Undo persona_instructions(): drop a trailing SHELL_NOTES if one is there."""
    text = persona.rstrip()
    if text.endswith(SHELL_NOTES):
        return text[: -len(SHELL_NOTES)].rstrip()
    return text


def persona_instructions(persona: str) -> str:
    """The one place a persona becomes system instructions.

    SHELL_NOTES is behaviour contract, not persona, so it appends to every persona
    in every environment (docs/environments.md schema v1: "TOOLS_NOTE/TIME_NOTE
    still append"). Name mis-hearings (Richie/Ricci) are fixed in
    shell/llm_proxy.py, not here: prompting the 4B model about them made it keep
    bringing it up.

    Appending is idempotent on purpose: `--print-env` prints *resolved*
    instructions, so the documented verify-then-paste flow hands back a string that
    already carries the notes, and TOOLS_NOTE + TIME_NOTE must not land in the
    system prompt twice.
    """
    return without_shell_notes(persona) + SHELL_NOTES


def _env_name() -> str:
    return os.getenv(REACHY_ENV_VAR, "").strip() or DEFAULT_ENV


def _memory_namespace() -> str:
    # Empty = "follow the environment name", resolved in __post_init__ so that
    # ShellConfig(env="lab") namespaces as "lab" too. A new environment can then
    # never inherit another one's facts by omission (docs/environments.md rule 1).
    return os.getenv("REACHY_MEMORY_NAMESPACE", "").strip()


def _csv(name: str) -> list[str]:
    return [part.strip() for part in os.getenv(name, "").split(",") if part.strip()]


def _bool_env(name: str, default: str = "1") -> bool:
    """A boolean process variable that understands the same words as the env file.

    shell/env.py casts file booleans with `_flag` (0/false/no/off). Reading the
    process variable as `!= "0"` meant `REACHY_LEARN_FACTS=false` in `.env` or a
    drop-in resolved to **True** on top of `envs/hackday.yaml`'s `false` - the
    operator's privacy keystroke turned the opposite way, silently (2026-10-12
    review F-3, hard rule 5). Empty stays True, exactly as before the fix.
    """
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _env(name: str, default: str) -> str:
    """One rule for the whole config: present-but-empty means "no opinion".

    shell/env.py treats an empty `REACHY_X=` in .env (an uncommented template line,
    a systemd `Environment=REACHY_X=` drop-in) as the variable being unset, so the
    code defaults have to read it the same way: `float("")` used to crash the shell
    on a blank REACHY_MOTION_HZ, and a blank REACHY_INSTRUCTIONS used to leave
    SHELL_NOTES as the whole persona.
    """
    return os.environ.get(name, "").strip() or default


@dataclass
class ShellConfig:
    # Resolved deployment environment (docs/environments.md); stamped on every
    # journal event. Only shell/env.py writes it, from REACHY_ENV.
    env: str = field(default_factory=_env_name)

    # Realtime service (pinned huggingface/speech-to-speech, ADR 0001)
    s2s_url: str = field(default_factory=lambda: _env("S2S_URL", "ws://127.0.0.1:8765/v1/realtime"))
    instructions: str = field(
        default_factory=lambda: persona_instructions(_env("REACHY_INSTRUCTIONS", DEFAULT_INSTRUCTIONS))
    )

    # Tool subset for this environment (shell/tools.apply_tool_policy): empty
    # include = everything the shell knows about, exclude always wins.
    tools_include: list[str] = field(default_factory=lambda: _csv("REACHY_TOOLS_INCLUDE"))
    tools_exclude: list[str] = field(default_factory=lambda: _csv("REACHY_TOOLS_EXCLUDE"))

    # Which memories.<ns>.json this env reads and writes (shell/env, A5).
    # "" = follow `env`; __post_init__ resolves it.
    memory_namespace: str = field(default_factory=_memory_namespace)

    # Auto fact extraction (behaviour.learn_facts, shell/conversation.py): mine
    # the transcript for lasting personal facts on every compaction and write
    # them into this env's store unasked. Off in a room with one shared store:
    # `prompt_block` labels those facts "what you know about the user", so with
    # one memories.hackday.json attendee A's name gets said to attendee B as
    # B's own data (docs/environments.md hard rule 1). The explicit `remember`
    # tool is NOT gated by this - a guest who asks to be remembered is answered.
    learn_facts: bool = field(default_factory=lambda: _bool_env("REACHY_LEARN_FACTS"))

    # Lie down by himself after this many seconds of silence with nobody
    # engaged (behaviour.auto_sleep_s, shell/sleep.py + shell/__main__.py). 0 =
    # never, which is what BOTH environments resolve to today (David's call:
    # Reachy stays awake even in an empty room); the knob stays for operators.
    # Whatever it is, the wake word always wakes him (issue #36).
    auto_sleep_s: float = field(default_factory=lambda: float(_env("REACHY_AUTO_SLEEP_S", "0")))

    # Proactive greeter (docs/environments.md `behaviour.greeter`, plan B0+B3). Off
    # by default and off in home: an empty living room must not be spoken at. The
    # gate itself is shell/vision/greeter.py; these are the knobs only.
    greeter_enabled: bool = field(default_factory=lambda: _bool_env("REACHY_GREETER", "0"))
    # Log every decision (dwell, angles, bbox share, verdict) without changing any
    # behaviour — the bench-tuning path for the kps-derived angle gates.
    greeter_debug: bool = field(default_factory=lambda: _bool_env("REACHY_GREETER_DEBUG", "0"))
    # Hard ceiling on greetings per hour, on top of the per-presence and global
    # spacing: the room-rate limit ADR 0003 asks for when the room is full of people.
    greeter_max_per_hour: int = field(default_factory=lambda: int(_env("REACHY_GREETER_MAX_PER_HOUR", "6")))
    greeter_dwell_s: float = field(default_factory=lambda: float(_env("REACHY_GREETER_DWELL_S", "2.0")))
    greeter_min_spacing_s: float = field(default_factory=lambda: float(_env("REACHY_GREETER_MIN_SPACING_S", "120")))
    greeter_group_min_spacing_s: float = field(
        default_factory=lambda: float(_env("REACHY_GREETER_GROUP_MIN_SPACING_S", "300"))
    )
    # Head-centring targets for a qualified face, through MotionOwner. Off: the
    # interaction with the DOA wake turn and the speech sway is the one part of B0
    # that cannot be measured without the robot, so it ships as a hook (see __main__).
    greeter_face_centering: bool = field(default_factory=lambda: _bool_env("REACHY_GREETER_FACE_CENTERING", "0"))
    greeter_idle_fps: float = field(default_factory=lambda: float(_env("REACHY_GREETER_IDLE_FPS", "2")))
    greeter_engaged_fps: float = field(default_factory=lambda: float(_env("REACHY_GREETER_ENGAGED_FPS", "3")))

    # Direct llama.cpp endpoint for side tasks (conversation compaction).
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://127.0.0.1:8080/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "local"))
    compact_after_s: float = field(default_factory=lambda: float(_env("REACHY_COMPACT_AFTER_S", "180")))
    reset_after_s: float = field(default_factory=lambda: float(_env("REACHY_RESET_AFTER_S", "300")))

    # A user turn still open after this many seconds is cut by feeding the
    # service a moment of silence - the only endpoint it accepts from a
    # client (input_audio_buffer.commit is a no-op upstream). Bounds how much
    # background speech can pile into one turn.
    max_turn_s: float = field(default_factory=lambda: float(_env("REACHY_MAX_TURN_S", "15")))

    # XVF3800 tuning (shell/audio/xmos_tuning.py): AGC gain ceiling (firmware
    # boots at 64x and pumps the TV up whenever the room goes quiet; 0 turns
    # AGC off) and steering both fixed beams at whoever said the wake word.
    xmos_agc_max_gain: float = field(default_factory=lambda: float(_env("REACHY_XMOS_AGC_MAX_GAIN", "10")))
    xmos_beam_focus: bool = field(default_factory=lambda: _bool_env("REACHY_XMOS_BEAM_FOCUS"))

    # Attention
    attention_profile: str = field(default_factory=lambda: _env("REACHY_PROFILE", "quiet"))

    # Audio (s2s PIPELINE_SAMPLE_RATE, both directions)
    sample_rate: int = 16000
    block_size: int = 512  # 32 ms at 16k
    channels: int = 1
    preroll_seconds: float = 1.0
    mic_device: str | None = field(default_factory=lambda: _env("REACHY_MIC", "") or None)
    speaker_device: str | None = field(default_factory=lambda: _env("REACHY_SPEAKER", "") or None)

    # Motion (shell/motion): one control loop; 50 Hz matches the daemon's
    # own control loop. Lower it if CPU is tight (ADR 0006).
    motion_hz: float = field(default_factory=lambda: float(_env("REACHY_MOTION_HZ", "50")))

    # Paths
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))
    # Prerendered acknowledgements: <ack_dir>/wake/*.wav and <ack_dir>/think/*.wav,
    # recorded in the active voice (e.g. sounds/acks/obama).
    ack_dir: Path = field(default_factory=lambda: Path(_env("REACHY_ACK_DIR", "sounds/acks")))
    kws_model_dir: str = field(default_factory=lambda: _env("KWS_MODEL_DIR", "models/kws"))

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
