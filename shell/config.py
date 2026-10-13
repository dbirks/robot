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

# Appended to every persona. This is the whole story the model is told about time,
# and it is a denial: he has no clock. Nothing in what he receives carries a time -
# no per-message stamp since 2026-10-13, no date in the persona - so every question
# about the time or the date has exactly one answer path, which is the tool.
#
# Why the stamp went, settled the night before the booth (two independent reasons):
#   - David heard it live: the model started working the stamps into its own replies.
#     A prefix on a transcript line is an artefact of our transport, not something a
#     guest said, and `TIME_NOTE` had spent a clause ("Never read the stamps aloud.")
#     trying to keep it out of the spoken half. That clause went with the feature.
#   - issue #45: `shell/llm_proxy.py` cached one stamp per message TEXT inside a
#     long-lived process, so an identical line re-asked after a `realtime.session_reset`
#     arrived wearing the OLD time - "hey richie." re-used 24.8 and 41.5 minutes apart
#     (2026-10-11 14:19 -> 15:00). A clock that can be hours stale is worse than none,
#     and it was the reason "What time is it?" got answered without a tool call at all
#     (`scripts/rehearse_booth.py`, 2026-10-12 23:41: "The clock indicates it is still
#     late ... I have checked the time, and it remains as it was.").
# What is lost with it, and where it went:
#   - "which day is it" -> `get_time`, the same local clock read at answer time. Its
#     result spells the local date and zone out (`app/robot_tools.py`:431) because a
#     bare UTC ISO string made him give tomorrow's date every evening after 20:00 EDT,
#     so the note below sends him to those words verbatim rather than to arithmetic
#     over an offset a 4B cannot see.
#   - "how long ago the guest spoke" (2026-10-05: a dangling question answered 40
#     minutes late) -> the shell, which compacts after `compact_after_s` and starts a
#     fresh s2s session after `reset_after_s` of silence (shell/conversation.py). That
#     was always a timer's job; a stale cached string could not do it anyway.
TIME_NOTE = (
    " You have no clock and you cannot see one: nothing in what you are shown tells you"
    " what day or hour it is, and a time mentioned earlier in the conversation is already"
    " stale. So any question about the time, the date, the day of the week, or how late it"
    " is gets answered by calling get_time and saying the date and time it returns exactly"
    " as it returns them - they are already the local clock, read just now. Never guess,"
    " estimate or reason out a time or a date, and never repeat one as if it still held."
    " Answer only the newest message; do not bring back old, unanswered topics."
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
    " Greetings, small talk, jokes, feelings and goodbyes need no tool: just answer - except"
    " play_emotion, which belongs to exactly those moments. And narrating a tool call is not"
    " making one: nothing changes his body, his loudness or his memory until the tool is"
    " CALLED - 'I am waving', 'I have turned it down' and 'I remember that' are all empty"
    " sentences, and saying he cannot do something a tool does is just as wrong. Narrate tool use out loud so the user can"
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
#: How to address a stranger. Written after the live log at 13:30 on 2026-10-13, where he said
#: "I am at your service, sir" and then "The view is currently open, sir" in the next reply - the
#: first "sir" entered the session history and the model copied it, the same way one narrated lie
#: ("I've turned it down") turned into three. Nothing in our own prompt said "sir": this is the
#: model's RLHF politeness, and the register is wrong for Marvin anyway.
#:
#: The rule has to be stated, not implied, for two reasons measured tonight. (1) The shell cannot
#: fix his words after the fact - the service synthesises audio from the model's text, so there is
#: no speak-side filter to write; the prompt is the only lever. (2) An honorific presumes gender
#: about a stranger, which sits badly next to a rule set that already refuses to name anybody.
ADDRESS_NOTE = (
    ' Never address a guest as "sir", "ma\'am", "madam" or any honorific or title - you do'
    ' not know anyone\'s name, gender or preference, and "I am at your service" is not how you'
    " speak. Be courteous by being useful, and answer people directly."
)

SHELL_NOTES = TOOLS_NOTE + TIME_NOTE + ADDRESS_NOTE


# The notes' opening words. `persona_instructions()` appends the notes last, so this is
# the boundary. A constant with an assert below rather than a slice of TOOLS_NOTE, so a
# rewording that moves the opening has to be noticed here.
NOTES_SENTINEL = " Speak plain words:"
assert TOOLS_NOTE.startswith(NOTES_SENTINEL), "NOTES_SENTINEL must be how TOOLS_NOTE opens"


def without_shell_notes(persona: str) -> str:
    """Undo persona_instructions(): drop the behaviour notes wherever they sit.

    This used to compare against a trailing `SHELL_NOTES`, and that quietly broke the
    day the notes became room-dependent: in a `learn_facts: false` room
    `reconcile_learn_facts_prompt()` replaces the auto-learn sentence *inside*
    TOOLS_NOTE, so the rendered prompt no longer ends with the literal constant, and the
    documented verify-then-paste flow re-appended the entire contract a second time.
    `scripts/check_env_safety.sh` check 7 caught it on 2026-10-13 - in hackday
    specifically, which is the room we demo in.

    Cutting at the first occurrence of the notes' opening words is idempotent for every
    room variant, because the notes go last and always open the same way. A persona that
    quotes that sentence verbatim loses its quote; the only way to get one there is to
    paste `--print-env` output, which is exactly the case being defended.
    """
    cut = persona.find(NOTES_SENTINEL)
    return (persona if cut < 0 else persona[:cut]).rstrip()


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
    # The idle search sweep: when the room is empty and has been for `scan_settle_s`,
    # pan +-35 deg in ~10 deg steps at >= 1 s per step until a face appears. Off in
    # both rooms until it has been watched with one person in front of him — the
    # servos have had about a minute of powered time in their life, and a sweep that
    # oscillates is a booth failure no log line recovers. See `Greeter._scan`.
    greeter_scan: bool = field(default_factory=lambda: _bool_env("REACHY_GREETER_SCAN", "0"))
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
