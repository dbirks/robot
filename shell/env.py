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

from .config import DEFAULT_ENV, REACHY_ENV_VAR, ShellConfig, persona_instructions

# Repo root, not cwd: `python -m shell` runs from %h/dev/robot under systemd, but
# --print-env and the tests must resolve the same files from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
ENVS_DIR = REPO_ROOT / "envs"


class EnvError(RuntimeError):
    """The deployment environment could not be resolved (hard startup error)."""


class EnvFileMissing(EnvError):
    """`envs/<env>.yaml` is not there — refuse to start rather than guess a persona."""


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
# this table — `name:`, `behaviour.greeter`, anything a later phase adds — are
# ignored on purpose, which is what makes an older shell forward-compatible.
LAYERS: tuple[_Layer, ...] = (
    _Layer("attention_profile", "attention_profile", "REACHY_PROFILE"),
    _Layer("persona.instructions", "instructions", "REACHY_INSTRUCTIONS", _instructions),
    _Layer("persona.ack_dir", "ack_dir", "REACHY_ACK_DIR", _path),
    _Layer("tools.include", "tools_include", "REACHY_TOOLS_INCLUDE", _names),
    _Layer("tools.exclude", "tools_exclude", "REACHY_TOOLS_EXCLUDE", _names),
    _Layer("memory.namespace", "memory_namespace", "REACHY_MEMORY_NAMESPACE"),
    _Layer("behaviour.learn_facts", "learn_facts", "REACHY_LEARN_FACTS", _flag),
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


def load_config(env: str | None = None, *, envs_dir: Path | None = None) -> ShellConfig:
    """Resolve the config for `env` (default: REACHY_ENV, then `home`).

    Raises EnvError — startup must not continue — when the file is missing or
    when the result would load another room's memories.
    """
    name = resolve_env_name(env)
    doc = load_env_doc(name, envs_dir)
    cfg = layer_env_file(ShellConfig(env=name), doc)
    guard_memory_namespace(cfg)
    return cfg


def config_to_dict(cfg: ShellConfig) -> dict:
    """Resolved config as JSON-ready data: `python -m shell --print-env`."""
    out: dict[str, Any] = {f.name: getattr(cfg, f.name) for f in fields(ShellConfig)}
    out["journal_path"] = cfg.journal_path
    out["memories_path"] = cfg.memories_path
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in out.items()}
