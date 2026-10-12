"""Deployment environments (docs/environments.md Phase A): resolution, precedence,
namespacing, and the --print-env seam. Hardware-free: no audio, no robot, no network,
and every file touched is under tmp_path.

Today's values are spelled out here on purpose (TODAY_*): the parity bar is
"REACHY_ENV unset + today's .env reproduces today's resolved config", and `.env` is
gitignored, so a test that read it would not exist on another machine.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

import shell.__main__ as shell_main
from shell.config import SHELL_NOTES, ShellConfig, persona_instructions
from shell.env import ENVS_DIR, LAYERS, EnvFileMissing, config_to_dict, load_config, resolve_env_name
from shell.memories import MemoryStore, migrate_legacy_memories
from shell.tools import apply_tool_policy

# Transcribed from the live .env on 2026-10-12 (the Obama persona and its ack pool).
TODAY_PERSONA = (
    "You are Reachy, a small desk robot, and you talk in the style of President Barack Obama: calm, measured, "
    "warm, with thoughtful pauses, a touch of dry humor, and his signature phrases like 'Let me be clear', "
    "'Look,', 'Here is the thing', and 'folks'. Reply in one or two short spoken sentences. Never use markdown "
    "or lists. You are still Reachy the robot, not actually Obama."
)
TODAY_INSTRUCTIONS = persona_instructions(TODAY_PERSONA)
TODAY_ACK_DIR = Path("sounds/acks/obama")

# Every variable shell/env.py treats as an override when it is present. On a dev box
# with a .env, shell/config.py's load_dotenv() has already put these in os.environ,
# so tests that mean to exercise the file have to clear them first.
OVERRIDE_VARS = ["REACHY_ENV"] + [layer.env_var for layer in LAYERS if layer.env_var]


@pytest.fixture
def clean_env(monkeypatch):
    """No .env leftovers: only code defaults and the env file are in play."""
    for var in OVERRIDE_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def write_env(tmp_path: Path, name: str, body: str) -> Path:
    envs = tmp_path / "envs"
    envs.mkdir(exist_ok=True)
    (envs / f"{name}.yaml").write_text(body)
    return envs


# --------------------------------------------------------------------------
# A1 — precedence: code defaults < env file < process environment
# --------------------------------------------------------------------------


def test_file_beats_code_default(tmp_path, clean_env):
    envs = write_env(
        tmp_path,
        "lab",
        """
attention_profile: event
persona:
  instructions: You are a lab bench robot.
  ack_dir: sounds/acks/rickman
audio:
  xmos_agc_max_gain: 3
  xmos_beam_focus: false
motion:
  hz: 20
conversation:
  max_turn_s: 8
tools:
  exclude: [go_to_sleep, reset_conversation]
memory:
  namespace: lab
""",
    )
    cfg = load_config("lab", envs_dir=envs)
    assert cfg.env == "lab"
    assert cfg.attention_profile == "event"
    assert cfg.instructions == persona_instructions("You are a lab bench robot.")
    assert cfg.ack_dir == Path("sounds/acks/rickman")
    assert cfg.xmos_agc_max_gain == 3.0
    assert cfg.xmos_beam_focus is False
    assert cfg.motion_hz == 20.0
    assert cfg.max_turn_s == 8.0
    assert cfg.tools_exclude == ["go_to_sleep", "reset_conversation"]
    assert cfg.memory_namespace == "lab"
    # keys the file does not carry keep their code default
    assert cfg.compact_after_s == ShellConfig().compact_after_s


def test_process_env_beats_file(tmp_path, clean_env, monkeypatch):
    monkeypatch.setenv("REACHY_PROFILE", "quiet")
    monkeypatch.setenv("REACHY_INSTRUCTIONS", "Process wins.")
    monkeypatch.setenv("REACHY_XMOS_AGC_MAX_GAIN", "12")
    envs = write_env(
        tmp_path,
        "lab",
        """
attention_profile: event
persona:
  instructions: File loses.
audio:
  xmos_agc_max_gain: 3
""",
    )
    cfg = load_config("lab", envs_dir=envs)
    assert cfg.attention_profile == "quiet"
    assert cfg.instructions == persona_instructions("Process wins.")
    assert cfg.xmos_agc_max_gain == 12.0


def test_explicit_null_in_the_file_is_no_opinion(tmp_path, clean_env, monkeypatch):
    """`mic: null` means "whatever this box defaults to"; it does not fight REACHY_MIC."""
    monkeypatch.setenv("REACHY_MIC", "pipewire:USB Audio")
    envs = write_env(tmp_path, "lab", "audio:\n  mic: null\n  speaker: null\n")
    cfg = load_config("lab", envs_dir=envs)
    assert cfg.mic_device == "pipewire:USB Audio"
    assert cfg.speaker_device is None


def test_unknown_keys_are_ignored(tmp_path, clean_env):
    """Forward compatibility: a file written by a newer main must boot an older shell."""
    envs = write_env(
        tmp_path,
        "lab",
        """
name: lab
attention_profile: event
behaviour:
  greeter: true
  greeter_cooldown_s: 120
  nobody_has_implemented_this: yes
brand_new_section:
  - one
  - two
""",
    )
    cfg = load_config("lab", envs_dir=envs)
    assert cfg.attention_profile == "event"
    assert "greeter" not in vars(cfg) and "behaviour" not in vars(cfg)


def test_missing_env_file_is_a_hard_error(tmp_path, clean_env):
    with pytest.raises(EnvFileMissing) as err:
        load_config("ghost", envs_dir=tmp_path / "envs")
    assert "ghost" in str(err.value) and "REACHY_ENV" in str(err.value)


def test_env_file_cannot_select_itself(tmp_path, clean_env):
    """REACHY_ENV picks the room; `name:` inside the file is descriptive only."""
    envs = write_env(tmp_path, "lab", "name: somethingelse\nattention_profile: event\n")
    assert load_config("lab", envs_dir=envs).env == "lab"
    clean_env.setenv("REACHY_ENV", "lab")
    assert resolve_env_name() == "lab"
    assert load_config(envs_dir=envs).env == "lab"


def test_env_defaults_to_home(clean_env):
    assert resolve_env_name() == "home"
    assert ShellConfig().env == "home"


# --------------------------------------------------------------------------
# A2/A7 — home parity: the env layer must not change a single resolved value
# --------------------------------------------------------------------------


def test_home_yaml_carries_todays_persona_and_ack_pool(clean_env):
    """home.yaml alone = today's values, for a machine that has no .env at all."""
    cfg = load_config("home", envs_dir=ENVS_DIR)
    assert cfg.instructions == TODAY_INSTRUCTIONS
    assert cfg.ack_dir == TODAY_ACK_DIR


def test_todays_dotenv_still_wins_over_the_file(clean_env, monkeypatch):
    """A2: `.env` keeps working as an override — nothing about it changes."""
    monkeypatch.setenv("REACHY_INSTRUCTIONS", TODAY_PERSONA)
    monkeypatch.setenv("REACHY_ACK_DIR", str(TODAY_ACK_DIR))
    cfg = load_config("home", envs_dir=ENVS_DIR)
    assert cfg.instructions == TODAY_INSTRUCTIONS
    assert cfg.ack_dir == TODAY_ACK_DIR


def test_home_parity_is_the_whole_resolved_config(clean_env, monkeypatch):
    """The parity bar: REACHY_ENV unset + today's .env == the pre-environment shell."""
    monkeypatch.setenv("REACHY_INSTRUCTIONS", TODAY_PERSONA)
    monkeypatch.setenv("REACHY_ACK_DIR", str(TODAY_ACK_DIR))
    before = config_to_dict(ShellConfig())  # exactly how the shell resolved config until now
    after = config_to_dict(load_config("home", envs_dir=ENVS_DIR))
    assert after == before


def test_hackday_persona_carries_no_home_facts(clean_env):
    cfg = load_config("hackday", envs_dir=ENVS_DIR)
    lowered = cfg.instructions.lower()
    for private in ["david", "rachel", "barney", "forrest", "timmy", "mccordsville", "46055", "wife", "his son"]:
        assert private not in lowered
    assert "hack day" in lowered and "never mention anyone's family" in lowered
    assert cfg.attention_profile == "event"
    assert cfg.tools_exclude == ["go_to_sleep", "reset_conversation"]
    assert cfg.memory_namespace == "hackday"
    # the notes every persona gets, whatever the room (docs/environments.md schema v1)
    assert cfg.instructions.endswith(SHELL_NOTES)


# --------------------------------------------------------------------------
# A4 — --print-env
# --------------------------------------------------------------------------


def test_print_env_prints_and_leaves_no_trace(clean_env, capsys, monkeypatch):
    """--print-env must not open audio, connect to the robot, write the journal,
    or touch the network — and stdout must be exactly one JSON document."""

    def forbidden(what):
        def _boom(*_args, **_kwargs):
            raise AssertionError(f"--print-env touched {what}")

        return _boom

    import socket
    import sqlite3

    import sounddevice

    monkeypatch.setattr(sqlite3, "connect", forbidden("sqlite"))
    monkeypatch.setattr(socket, "socket", forbidden("the network"))
    monkeypatch.setattr(sounddevice, "InputStream", forbidden("a microphone"))
    monkeypatch.setattr(sounddevice, "OutputStream", forbidden("a speaker"))
    monkeypatch.setattr(shell_main.MicOwner, "open", forbidden("the mic owner"))
    monkeypatch.setattr(shell_main.SpeakerOwner, "open", forbidden("the speaker owner"))
    monkeypatch.setattr(shell_main.RobotLink, "connect", forbidden("the robot"))
    monkeypatch.setattr(shell_main.asyncio, "run", forbidden("the event loop"))

    shell_main.main(["--print-env"])

    captured = capsys.readouterr()
    assert captured.err == ""
    printed = json.loads(captured.out)  # raises if anything else leaked to stdout
    assert printed["env"] == "home"
    assert printed["instructions"] == TODAY_INSTRUCTIONS  # resolved: notes appended
    assert printed["ack_dir"] == str(TODAY_ACK_DIR)
    assert printed["memories_path"].endswith("memories.home.json")
    assert json.dumps(printed)  # JSON-ready, not merely repr-able


def test_print_env_fails_hard_on_a_missing_file(clean_env, capsys, monkeypatch):
    monkeypatch.setenv("REACHY_ENV", "no_such_room")
    with pytest.raises(SystemExit) as exit_info:
        shell_main.main(["--print-env"])
    assert exit_info.value.code == 2
    assert "no_such_room" in capsys.readouterr().err


# --------------------------------------------------------------------------
# A5 — memory namespaces + migration
# --------------------------------------------------------------------------


def test_hackday_cannot_see_home_memories(tmp_path, clean_env):
    home = replace(load_config("home", envs_dir=ENVS_DIR), data_dir=tmp_path)
    event = replace(load_config("hackday", envs_dir=ENVS_DIR), data_dir=tmp_path)
    assert home.memories_path == tmp_path / "memories.home.json"
    assert event.memories_path == tmp_path / "memories.hackday.json"

    MemoryStore(home.memories_path).add("The user lives in McCordsville, Indiana.")
    assert not event.memories_path.exists()  # the event store starts empty
    assert MemoryStore(event.memories_path).all() == []
    assert [m["text"] for m in MemoryStore(home.memories_path).all()] == ["The user lives in McCordsville, Indiana."]


def test_an_unknown_env_namespaces_by_its_own_name(clean_env):
    """A new environment can never inherit another one's facts by omission."""
    assert ShellConfig(env="lab").memory_namespace == "lab"
    assert ShellConfig(env="lab").memories_path.name == "memories.lab.json"


def test_migrate_legacy_memories_copies_and_never_deletes(tmp_path, clean_env):
    legacy = tmp_path / "memories.json"
    legacy.write_text(json.dumps([{"id": 1, "text": "David's son is named Barney.", "saved": "x", "source": "auto"}]))

    report = migrate_legacy_memories(tmp_path, "home")
    assert report["ok"] and report["action"] == "copied" and report["facts"] == 1
    assert legacy.exists(), "copy-not-delete: the legacy file must survive"
    assert [m["text"] for m in MemoryStore(tmp_path / "memories.home.json").all()] == ["David's son is named Barney."]

    # A second run must not overwrite the live namespaced store with stale facts.
    MemoryStore(tmp_path / "memories.home.json").add("The user has a dog.")
    assert migrate_legacy_memories(tmp_path, "home")["action"] == "skipped"
    assert len(MemoryStore(tmp_path / "memories.home.json").all()) == 2

    assert migrate_legacy_memories(tmp_path, "hackday")["ok"] is False
    assert not (tmp_path / "memories.hackday.json").exists()  # home facts never reach the event store
    assert migrate_legacy_memories(tmp_path / "nowhere", "home")["action"] == "nothing_to_migrate"


def test_migrate_legacy_memories_refuses_to_seed_another_env(tmp_path, clean_env):
    """Hard rule 1: the hack-day store starts empty, so the legacy file may only
    move to `home`. A wrong REACHY_ENV must not put family facts in a booth."""
    legacy = tmp_path / "memories.json"
    legacy.write_text(json.dumps([{"id": 1, "text": "The user lives in McCordsville.", "saved": "x"}]))
    report = migrate_legacy_memories(tmp_path, "hackday")
    assert report["ok"] is False and "hard rule 1" in report["error"]
    assert not (tmp_path / "memories.hackday.json").exists()
    assert legacy.exists()


def test_migrate_legacy_memories_reports_instead_of_raising(tmp_path, clean_env):
    (tmp_path / "memories.json").write_text("not json at all")
    report = migrate_legacy_memories(tmp_path, "home")
    assert report["ok"] and report["facts"] == 0  # copied verbatim, unreadable, warned about
    assert (tmp_path / "memories.home.json").read_text() == "not json at all"


# --------------------------------------------------------------------------
# A3 — the env tool subset is applied at assembly, not just documented
# --------------------------------------------------------------------------


def test_hackday_drops_the_excluded_tools(clean_env):
    cfg = load_config("hackday", envs_dir=ENVS_DIR)
    names = ("go_to_sleep", "reset_conversation", "nod", "remember")
    tools = [{"function": {"name": name}} for name in names]
    handlers = {name: (lambda **_kw: {"ok": True}) for name in names}

    kept_tools, kept_handlers = apply_tool_policy(tools, handlers, include=cfg.tools_include, exclude=cfg.tools_exclude)
    assert [t["function"]["name"] for t in kept_tools] == ["nod", "remember"]
    assert set(kept_handlers) == {"nod", "remember"}  # excluded handlers go too

    only_greeting, _ = apply_tool_policy(tools, handlers, include=["nod"])
    assert [t["function"]["name"] for t in only_greeting] == ["nod"]
    # a name the shell has never heard of is a no-op, not a startup failure
    unchanged, _ = apply_tool_policy(tools, handlers, exclude=["teleport"])
    assert len(unchanged) == len(names)


def test_home_ships_no_tool_policy(clean_env):
    cfg = load_config("home", envs_dir=ENVS_DIR)
    assert cfg.tools_include == [] and cfg.tools_exclude == []
    names = ("go_to_sleep", "reset_conversation", "nod")
    tools = [{"function": {"name": name}} for name in names]
    handlers = {name: (lambda **_kw: {"ok": True}) for name in names}
    kept_tools, kept_handlers = apply_tool_policy(tools, handlers, include=cfg.tools_include, exclude=cfg.tools_exclude)
    assert len(kept_tools) == len(names) and set(kept_handlers) == set(names)
