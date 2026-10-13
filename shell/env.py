"""Deployment environments: `envs/<env>.yaml` layered under the process environment.

`REACHY_ENV` picks the room — `home` (default) or `hackday`; the design and the
work list are docs/environments.md. Environments are data, not branches: the file
carries the persona, the attention profile, the tool subset, the memory namespace
and the room's audio/motion tuning, and nothing else.

Precedence is **code defaults -> env file -> process environment variables**, so
an on-site tweak is one line in `.env` (or a `systemctl --user set-environment`)
rather than an edit to a tracked file under time pressure. Two consequences worth spelling out:

- shell/config.py calls `load_dotenv()` at import, so `.env` values *are* process
  environment values by the time this module runs. Someone who only has a `.env`
  keeps working exactly as before, and their `.env` outranks the env file — which
  is what "`.env` keeps working as an override" means (A2). Set a name in both
  places and the env-file line is dead.
- An explicit `null` in the file is "no opinion", not "clear the value": the code
  default (or the process env) stands. `audio.mic: null` says "whatever this box
  defaults to"; it does not fight `REACHY_MIC`.
- An *empty* process env var is "no opinion" too (`REACHY_INSTRUCTIONS=` in `.env`
  does not silence the file): the file's silence is honoured as no opinion, so the
  environment's silence has to be, or one stray `KEY=` drops the attendee-safe
  persona and the booth tool excludes without an error.

Resolution itself can fail closed: a non-`home` environment that lands on the
`home` memory namespace is a hard `EnvError` (`guard_memory_namespace`), because
that is family facts in a public room.

A missing `envs/<env>.yaml` is a hard startup error: booting in a public room with
a half-known persona is the failure this layer exists to prevent. Unknown keys are
ignored, so a file written by a newer `main` still boots an older shell.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import yaml

from .config import (
    DEFAULT_ENV,
    LEARN_FACTS_NOTE,
    REACHY_ENV_VAR,
    ShellConfig,
    instructions_without_auto_learn,
    persona_instructions,
)

# Repo root, not cwd: `python -m shell` runs from %h/dev/robot under systemd, but
# --print-env and the tests must resolve the same files from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
ENVS_DIR = REPO_ROOT / "envs"
PERSONAS_DIR = REPO_ROOT / "personas"

# `persona:` in an env file is one of two shapes (docs/environments.md, persona
# picker): a bare name (`persona: marvin`, which means personas/marvin.yaml) or the
# v1 mapping (`persona: {instructions: ..., ack_dir: ...}`) that both shipped rooms
# still carry. `persona: {name: marvin, instructions: ...}` is the migration shape -
# it names the persona for its voice and keeps the room's own words, and the inline
# instructions OUTRANK the persona file, so the A2 parity bar stayed byte-exact.
# REACHY_PERSONA (what scripts/apply_env.py stamps into deploy/generated/persona.env)
# outranks both: the picker moves the voice and the acks, never the words.
PERSONA_VAR = "REACHY_PERSONA"
_PERSONA_INLINE = ("instructions", "ack_dir")


class EnvError(RuntimeError):
    """The deployment environment could not be resolved (hard startup error)."""


class EnvFileMissing(EnvError):
    """`envs/<env>.yaml` is not there — refuse to start rather than guess a persona."""


class PersonaError(EnvError):
    """`personas/<name>.yaml` is missing or malformed — the same fail-closed bar as a
    missing env file: a robot that cannot say who it is being stays silent."""


def known_envs(envs_dir: Path | None = None) -> list[str]:
    directory = Path(envs_dir) if envs_dir is not None else ENVS_DIR
    try:
        return sorted(p.stem for p in directory.glob("*.yaml"))
    except OSError:
        return []


def resolve_env_name(env: str | None = None) -> str:
    """The env name comes from the real process environment (or an explicit
    argument); the file it selects can never change it."""
    return (env if env is not None else os.getenv(REACHY_ENV_VAR, "")).strip() or DEFAULT_ENV


def env_file_path(env: str, envs_dir: Path | None = None) -> Path:
    directory = Path(envs_dir) if envs_dir is not None else ENVS_DIR
    return directory / f"{env}.yaml"


def load_env_doc(env: str, envs_dir: Path | None = None) -> dict:
    """Read `envs/<env>.yaml`. Missing file / unparsable / not a mapping are hard errors."""
    name = resolve_env_name(env)
    path = env_file_path(name, envs_dir)
    if not path.is_file():
        raise EnvFileMissing(
            f"no deployment environment file at {path} (REACHY_ENV={name!r}); "
            f"available: {', '.join(known_envs(envs_dir)) or 'none'}"
        )
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise EnvError(f"deployment environment file {path} does not parse: {e}") from e
    if not isinstance(doc, dict):
        raise EnvError(f"deployment environment file {path} must be a mapping, got {type(doc).__name__}")
    return doc


_MISSING = object()


def _lookup(doc: dict, key: str) -> Any:
    """Fetch `persona.ack_dir` from the parsed file, or _MISSING if it isn't there."""
    node: Any = doc
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _text(value: Any) -> str:
    return str(value).strip()


def _instructions(value: Any) -> str:
    # persona_instructions() strips a trailing SHELL_NOTES first, so a value copied
    # out of `--print-env` (resolved instructions) does not double-append.
    return persona_instructions(str(value))


def _number(value: Any) -> float:
    return float(value)


def _int(value: Any) -> int:
    return int(float(value))


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _path(value: Any) -> Path:
    return Path(str(value).strip())


@dataclass(frozen=True)
class _Layer:
    """One env-file key and the ShellConfig field it fills."""

    key: str  # dotted path in envs/<env>.yaml
    field: str  # ShellConfig attribute
    env_var: str  # process env var that outranks the file ("" = nothing does)
    cast: Callable[[Any], Any] = _text


# The v1 schema (docs/environments.md) plus the A2 tuning values. Keys outside
# this table — `name:`, `behaviour.nobody_has_implemented_this`, anything a later
# phase adds — are ignored on purpose, which is what makes an older shell
# forward-compatible.
LAYERS: tuple[_Layer, ...] = (
    _Layer("attention_profile", "attention_profile", "REACHY_PROFILE"),
    _Layer("persona.instructions", "instructions", "REACHY_INSTRUCTIONS", _instructions),
    _Layer("persona.ack_dir", "ack_dir", "REACHY_ACK_DIR", _path),
    _Layer("tools.include", "tools_include", "REACHY_TOOLS_INCLUDE", _names),
    _Layer("tools.exclude", "tools_exclude", "REACHY_TOOLS_EXCLUDE", _names),
    _Layer("memory.namespace", "memory_namespace", "REACHY_MEMORY_NAMESPACE"),
    _Layer("behaviour.learn_facts", "learn_facts", "REACHY_LEARN_FACTS", _flag),
    _Layer("behaviour.greeter", "greeter_enabled", "REACHY_GREETER", _flag),
    _Layer("behaviour.greeter_max_per_hour", "greeter_max_per_hour", "REACHY_GREETER_MAX_PER_HOUR", _int),
    _Layer("behaviour.greeter_cooldown_s", "greeter_min_spacing_s", "REACHY_GREETER_MIN_SPACING_S", _number),
    _Layer(
        "behaviour.greeter_group_cooldown_s",
        "greeter_group_min_spacing_s",
        "REACHY_GREETER_GROUP_MIN_SPACING_S",
        _number,
    ),
    _Layer("behaviour.greeter_dwell_s", "greeter_dwell_s", "REACHY_GREETER_DWELL_S", _number),
    _Layer("behaviour.greeter_debug", "greeter_debug", "REACHY_GREETER_DEBUG", _flag),
    _Layer("behaviour.greeter_idle_fps", "greeter_idle_fps", "REACHY_GREETER_IDLE_FPS", _number),
    _Layer("behaviour.greeter_engaged_fps", "greeter_engaged_fps", "REACHY_GREETER_ENGAGED_FPS", _number),
    _Layer("behaviour.greeter_face_centering", "greeter_face_centering", "REACHY_GREETER_FACE_CENTERING", _flag),
    _Layer("behaviour.auto_sleep_s", "auto_sleep_s", "REACHY_AUTO_SLEEP_S", _number),
    _Layer("audio.mic", "mic_device", "REACHY_MIC"),
    _Layer("audio.speaker", "speaker_device", "REACHY_SPEAKER"),
    _Layer("audio.xmos_agc_max_gain", "xmos_agc_max_gain", "REACHY_XMOS_AGC_MAX_GAIN", _number),
    _Layer("audio.xmos_beam_focus", "xmos_beam_focus", "REACHY_XMOS_BEAM_FOCUS", _flag),
    _Layer("motion.hz", "motion_hz", "REACHY_MOTION_HZ", _number),
    _Layer("conversation.compact_after_s", "compact_after_s", "REACHY_COMPACT_AFTER_S", _number),
    _Layer("conversation.reset_after_s", "reset_after_s", "REACHY_RESET_AFTER_S", _number),
    _Layer("conversation.max_turn_s", "max_turn_s", "REACHY_MAX_TURN_S", _number),
)


def layer_env_file(cfg: ShellConfig, doc: dict) -> ShellConfig:
    """Layer the file underneath the process environment: a field whose variable
    is set non-empty in os.environ — `.env` included, see the module docstring —
    is left alone.

    Present-but-empty is "no opinion" here too, exactly as `null`/`""` is in the
    file. `REACHY_INSTRUCTIONS=` or `REACHY_TOOLS_EXCLUDE=` in `.env` (a commented
    template line, a `Environment=REACHY_X=` drop-in) used to delete the
    attendee-safe persona / the tool excludes with exit 0, which is the one place
    the layer weakened ADR 0003's fail-closed intent.
    """
    for layer in LAYERS:
        value = _lookup(doc, layer.key)
        # absent, or explicit "no opinion" (null / "") in the file
        if value is _MISSING or value is None or value == "":
            continue
        if layer.env_var and (os.environ.get(layer.env_var) or "").strip():
            continue  # the process env said something; the file stays quiet
        setattr(cfg, layer.field, layer.cast(value))
    return cfg


def reconcile_learn_facts_prompt(cfg: ShellConfig) -> ShellConfig:
    """A room that turns automatic fact-learning off must not also be *told* to do it.

    `behaviour.learn_facts` gates the miner (shell/conversation.py), but the shared
    `TOOLS_NOTE` in config.py carried a standing instruction to call `remember`
    unprompted about "the user or their family" - so `hackday` set the flag false and
    the prompt still said otherwise, and the `remember` tool is deliberately still
    available there. Prompt and policy are one decision now.

    Home is untouched: `learn_facts` defaults true, so its resolved instructions stay
    byte-identical (the note is an exact substring; tests/test_env.py pins that).
    """
    if not cfg.learn_facts and LEARN_FACTS_NOTE in cfg.instructions:
        cfg.instructions = instructions_without_auto_learn(cfg.instructions)
    return cfg


def guard_memory_namespace(cfg: ShellConfig) -> None:
    """Hard rule 1 (docs/environments.md): a room that is not `home` may never
    land on the home memory store.

    `memory.namespace` defaults to the env name, so this only fires when someone
    sets `REACHY_MEMORY_NAMESPACE=home` (or writes `namespace: home` into a
    non-home file) — e.g. "let me keep my memories at the demo". That is a
    family-facts leak into a public room, so it is a hard startup error rather
    than something to notice in `--print-env`. `migrate_legacy_memories` already
    refuses the same shape on the write side (shell/memories.py).
    """
    if cfg.env != DEFAULT_ENV and cfg.memory_namespace == DEFAULT_ENV:
        raise EnvError(
            f"deployment environment {cfg.env!r} resolves to the home memory namespace: "
            f"{cfg.memories_path} holds family facts and must not be loaded outside `home`. "
            f"Leave memory.namespace/REACHY_MEMORY_NAMESPACE unset (it defaults to {cfg.env!r}) "
            "or name a store of its own (docs/environments.md hard rule 1)."
        )


def known_personas(personas_dir: Path | None = None) -> list[str]:
    directory = Path(personas_dir) if personas_dir is not None else PERSONAS_DIR
    try:
        return sorted(p.stem for p in directory.glob("*.yaml"))
    except OSError:
        return []


def persona_file_path(name: str, personas_dir: Path | None = None) -> Path:
    directory = Path(personas_dir) if personas_dir is not None else PERSONAS_DIR
    return directory / f"{name}.yaml"


def load_persona_doc(name: str, personas_dir: Path | None = None) -> dict:
    """Read `personas/<name>.yaml`. Missing / unparsable / not-a-mapping are hard errors."""
    clean = str(name).strip()
    path = persona_file_path(clean, personas_dir)
    if not clean:
        raise PersonaError("persona name is empty")
    if not path.is_file():
        raise PersonaError(
            f"no persona file at {path} (persona={clean!r}); "
            f"available: {', '.join(known_personas(personas_dir)) or 'none'}"
        )
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise PersonaError(f"persona file {path} does not parse: {e}") from e
    if not isinstance(doc, dict):
        raise PersonaError(f"persona file {path} must be a mapping, got {type(doc).__name__}")
    doc.setdefault("name", clean)
    return doc


def persona_selection(doc: dict) -> tuple[str, str]:
    """The persona this env file asks for, and which line said so.

    `REACHY_PERSONA` wins: that is the picker, arriving through persona.env as a
    process variable, so it rides the existing precedence rule instead of a new one.
    ("", "") means nobody named one, which is exactly today's behaviour - inline-only
    rooms resolve with no persona file in sight.
    """
    stamped = (os.environ.get(PERSONA_VAR) or "").strip()
    if stamped:
        return stamped, PERSONA_VAR
    node = doc.get("persona") if isinstance(doc, dict) else None
    if isinstance(node, str) and node.strip():
        return node.strip(), "persona"
    if isinstance(node, dict):
        named = str(node.get("name") or "").strip()
        if named:
            return named, "persona.name"
    return "", ""


def merge_persona_doc(doc: dict, persona_doc: dict) -> dict:
    """Fill the persona keys from `personas/<name>.yaml` without ever overwriting what
    the room's own file says. The result is the shape LAYERS already reads, so the
    precedence table (`persona.instructions`, process env last) did not change."""
    merged = dict(doc)
    node = merged.get("persona")
    filled = {key: value for key, value in persona_doc.items() if key in _PERSONA_INLINE and value not in (None, "")}
    if isinstance(node, dict):
        inline = {key: value for key, value in node.items() if key in _PERSONA_INLINE and value not in (None, "")}
        filled.update(inline)  # the room's inline block outranks the persona file
    merged["persona"] = filled
    return merged


def persona_report(env: str | None = None, *, envs_dir: Path | None = None, personas_dir: Path | None = None) -> dict:
    """`python -m shell --print-personas`: the picker's data, resolved on this box.

    JSON-ready and hardware-free, like --print-env. The dashboard renders it through
    dashboard/persona_panel.py, which has no yaml of its own (its venv has none).
    """
    name = resolve_env_name(env)
    doc = load_env_doc(name, envs_dir)
    selected, source = persona_selection(doc)
    node: Any = doc.get("persona")
    if selected:
        merged = merge_persona_doc(doc, load_persona_doc(selected, personas_dir))
        node = merged.get("persona")
    directory = Path(personas_dir) if personas_dir is not None else PERSONAS_DIR
    personas = []
    for stem in known_personas(personas_dir):
        item = load_persona_doc(stem, personas_dir)
        raw_voice = item.get("voice")
        voice: dict[str, Any] = raw_voice if isinstance(raw_voice, dict) else {}
        personas.append(
            {
                "name": stem,
                "instructions": str(item.get("instructions") or "").strip(),
                "ack_dir": str(item.get("ack_dir") or ""),
                "voice": voice,
                "file": str(persona_file_path(stem, personas_dir).relative_to(REPO_ROOT)),
            }
        )
    inline_text = str(node.get("instructions") or "") if isinstance(node, dict) else ""
    return {
        "env": name,
        "persona": selected,
        "persona_source": source,
        # true while envs/<env>.yaml carries its own `instructions:`: the picker moves
        # the voice and the acks here, not the words (the A2 parity bar, see README).
        "inline_instructions": bool(inline_text.strip()),
        "personas": personas,
    }


def load_config(
    env: str | None = None, *, envs_dir: Path | None = None, personas_dir: Path | None = None
) -> ShellConfig:
    """Resolve the config for `env` (default: REACHY_ENV, then `home`).

    Raises EnvError — startup must not continue — when the file is missing or
    when the result would load another room's memories.
    """
    name = resolve_env_name(env)
    doc = load_env_doc(name, envs_dir)
    selected, _ = persona_selection(doc)
    if selected:
        # A named persona with no file is a hard error, never a silent fallback to
        # whoever else's prompt is loaded next to this voice.
        doc = merge_persona_doc(doc, load_persona_doc(selected, personas_dir))
    cfg = layer_env_file(ShellConfig(env=name), doc)
    guard_memory_namespace(cfg)
    reconcile_learn_facts_prompt(cfg)
    return cfg


def config_to_dict(cfg: ShellConfig) -> dict:
    """Resolved config as JSON-ready data: `python -m shell --print-env`."""
    out: dict[str, Any] = {f.name: getattr(cfg, f.name) for f in fields(ShellConfig)}
    out["journal_path"] = cfg.journal_path
    out["memories_path"] = cfg.memories_path
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in out.items()}
