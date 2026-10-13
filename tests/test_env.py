"""Deployment environments (docs/environments.md Phase A): resolution, precedence,
namespacing, and the --print-env seam. Hardware-free: no audio, no robot, no network,
and every file touched is under tmp_path.

Today's values are spelled out here on purpose (TODAY_*): the parity bar is
"REACHY_ENV unset + today's .env reproduces today's resolved config", and `.env` is
gitignored, so a test that read it would not exist on another machine.
"""

import asyncio
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import shell.__main__ as shell_main
from shell.config import SHELL_NOTES, ShellConfig, instructions_without_auto_learn, persona_instructions
from shell.conversation import FACT_PROMPT, SUMMARY_PROMPT, ConversationMemory, grounded, parse_facts
from shell.env import (
    ENVS_DIR,
    LAYERS,
    REPO_ROOT,
    EnvError,
    EnvFileMissing,
    config_to_dict,
    load_config,
    resolve_env_name,
)
from shell.memories import MEMORY_TOOLS, MemoryStore, make_memory_handlers, migrate_legacy_memories
from shell.tools import apply_tool_policy
from tests.parity_baseline import HOME_PROMPT_BYTES

# Transcribed from the live .env on 2026-10-12 (the Obama persona and its ack pool).
TODAY_PERSONA = (
    "You are Reachy, a small desk robot, and you talk in the style of President Barack Obama: calm, measured, "
    "warm, with thoughtful pauses, a touch of dry humor, and his signature phrases like 'Let me be clear', "
    "'Look,', 'Here is the thing', and 'folks'. Reply in one or two short spoken sentences. Never use markdown "
    "or lists. You are still Reachy the robot, not actually Obama."
)
TODAY_INSTRUCTIONS = persona_instructions(TODAY_PERSONA)
TODAY_ACK_DIR = Path("sounds/acks/obama")


def expected_tail(cfg: ShellConfig) -> str:
    """The shared tail as THIS room receives it.

    `TOOLS_NOTE` lives inside `SHELL_NOTES`, and a room with
    `behaviour.learn_facts: false` gets that one sentence removed by
    `reconcile_learn_facts_prompt` (shell/env.py) - so asserting a bare
    `endswith(SHELL_NOTES)` would demand the very thing the booth must not say.
    """
    return SHELL_NOTES if cfg.learn_facts else instructions_without_auto_learn(SHELL_NOTES)


# Names/places/relations that are private to this desk (and the ones the reviews
# found compiled into prompts and tool schemas). Nothing below may appear in
# anything that leaves the box: system instructions, the compaction and fact
# extraction prompts, or a tool schema (tests/test_env.py scans all of them).
PRIVATE_WORDS = [
    "david",
    "rachel",
    "barney",
    "sarah",
    "forrest",
    "timmy",
    "mccordsville",
    "46055",
    "wife",
    "his son",
]

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


def shipped_tool_schemas() -> list[dict]:
    """Every tool schema the shell can put in front of the model.

    Phase A still bridges the legacy schemas (shell/robot_tools.build_handlers ->
    app.robot_tools.TOOLS) and swaps the memory trio for the shell-native one,
    exactly as build_handlers does. A hand-written list would let an upstream
    rename silently re-advertise an excluded tool.
    """
    legacy = pytest.importorskip("app.robot_tools").TOOLS
    replaced = {t["function"]["name"] for t in MEMORY_TOOLS}
    return [t for t in legacy if t.get("function", {}).get("name") not in replaced] + list(MEMORY_TOOLS)


def stub_handlers(names):
    return {name: (lambda **_kw: {"ok": True}) for name in names}


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


def test_an_empty_process_env_var_is_no_opinion(tmp_path, clean_env, monkeypatch):
    """P1 (2026-10-12): presence-only used to outrank the file, so a stray `KEY=`
    (an uncommented template line, an `Environment=REACHY_X=` drop-in) deleted the
    attendee-safe persona and re-advertised go_to_sleep in the booth - exit 0, no
    log line. The file's empty counts as "no opinion" (above); the environment's
    empty has to as well, or the layer fails open exactly where ADR 0003 says it
    must not.

    Both repros from the review, against the shipped hackday file.
    """
    monkeypatch.setenv("REACHY_ENV", "hackday")
    monkeypatch.setenv("REACHY_INSTRUCTIONS", "")  # the persona-killer repro
    monkeypatch.setenv("REACHY_TOOLS_EXCLUDE", "")  # the go_to_sleep repro
    monkeypatch.setenv("REACHY_XMOS_AGC_MAX_GAIN", "   ")  # whitespace is empty too
    monkeypatch.setenv("REACHY_MOTION_HZ", "")  # blank used to crash float("")
    monkeypatch.setenv("REACHY_PROFILE", "")  # blank used to resolve "" -> KeyError

    cfg = load_config("hackday", envs_dir=ENVS_DIR)
    assert "never mention anyone's family" in cfg.instructions  # the persona survived
    assert cfg.instructions.endswith(expected_tail(cfg)) and cfg.instructions.count("Speak plain words") == 1
    assert cfg.tools_exclude == [
        "go_to_sleep",
        "reset_conversation",
        "identify_face",
        "learn_face",
        "forget_face",
    ]
    assert cfg.xmos_agc_max_gain == 6.0  # the file's value, not the code default 10
    assert cfg.motion_hz == 50.0 and cfg.attention_profile == "event"  # blank, not crash

    # a value that says something still outranks the file (test_process_env_beats_file)
    monkeypatch.setenv("REACHY_TOOLS_EXCLUDE", "nod")
    assert load_config("hackday", envs_dir=ENVS_DIR).tools_exclude == ["nod"]


def test_unknown_keys_are_ignored(tmp_path, clean_env):
    """Forward compatibility: a file written by a newer main must boot an older shell."""
    envs = write_env(
        tmp_path,
        "lab",
        """
name: lab
attention_profile: event
behaviour:
  nobody_has_implemented_this: yes
brand_new_section:
  - one
  - two
""",
    )
    cfg = load_config("lab", envs_dir=envs)
    assert cfg.attention_profile == "event"
    assert "nobody_has_implemented_this" not in vars(cfg) and "behaviour" not in vars(cfg)


# B3 — the proactive greeter is a per-room behaviour flag (docs/environments.md)


def test_greeter_is_on_in_the_booth_and_off_at_home(clean_env):
    """A desk in a living room must not start talking to the room; a booth must.

    Both are the code default (off), so the hackday line is what turns it on and a
    dropped key would silently take the demo feature away, hence asserting True
    rather than "not False".
    """
    assert load_config("hackday", envs_dir=ENVS_DIR).greeter_enabled is True
    assert load_config("home", envs_dir=ENVS_DIR).greeter_enabled is False
    day = load_config("hackday", envs_dir=ENVS_DIR)
    assert day.greeter_dwell_s == 2.0  # David's call: about two seconds of looking
    assert day.greeter_min_spacing_s == 120.0 and day.greeter_group_min_spacing_s == 300.0
    assert day.greeter_max_per_hour == 6
    assert day.greeter_face_centering is False  # the deferred head hook (bench pass)


def test_greeter_flags_still_outrank_the_env_file(clean_env, monkeypatch):
    """The operator's on-site keystroke is the process environment, both ways."""
    monkeypatch.setenv("REACHY_GREETER", "0")  # kill the booth feature for a minute
    monkeypatch.setenv("REACHY_GREETER_DEBUG", "1")  # log every decision, change nothing
    monkeypatch.setenv("REACHY_GREETER_MAX_PER_HOUR", "2")
    cfg = load_config("hackday", envs_dir=ENVS_DIR)
    assert cfg.greeter_enabled is False
    assert cfg.greeter_debug is True
    assert cfg.greeter_max_per_hour == 2


def test_an_empty_greeter_variable_does_not_outrank_the_env_file(clean_env, monkeypatch):
    """`REACHY_GREETER=` in .env is a commented template line, not a switch."""
    monkeypatch.setenv("REACHY_GREETER", "")
    monkeypatch.setenv("REACHY_GREETER_MAX_PER_HOUR", "  ")
    cfg = load_config("hackday", envs_dir=ENVS_DIR)
    assert cfg.greeter_enabled is True
    assert cfg.greeter_max_per_hour == 6


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
    for private in PRIVATE_WORDS:
        assert private not in lowered
    assert "hack day" in lowered and "never mention anyone's family" in lowered
    assert cfg.attention_profile == "event"
    assert cfg.memory_namespace == "hackday"
    # the notes every persona gets, whatever the room (docs/environments.md schema v1)
    assert cfg.instructions.endswith(expected_tail(cfg))


# --------------------------------------------------------------------------
# Hard rule 1, the parts that are not memory files (2026-10-12 review)
# --------------------------------------------------------------------------


def test_no_face_tool_survives_the_hackday_schema(clean_env):
    """P0: faces are not namespaced until B1 - `app/face_tracker.py` hardcodes
    data/known_faces.json (David, Barney, Tim on this box), so in the booth
    identify_face reads the family gallery out loud and learn_face writes
    attendee biometrics into the home file, where the Phase C purge has nothing
    to delete. The exclusion is the whole mitigation, not the design: real
    per-env faces files are B1 (docs/environments.md).
    """
    cfg = load_config("hackday", envs_dir=ENVS_DIR)
    schemas = shipped_tool_schemas()
    known = {t["function"]["name"] for t in schemas}
    # the excludes bite real schemas: a typo or an upstream rename would silently
    # re-advertise the tool (apply_tool_policy ignores unknown names by design)
    assert {"go_to_sleep", "reset_conversation", "identify_face", "learn_face", "forget_face"} <= known
    assert set(cfg.tools_exclude) <= known

    kept_tools, kept_handlers = apply_tool_policy(
        schemas, stub_handlers(known), include=cfg.tools_include, exclude=cfg.tools_exclude
    )
    surviving = {t["function"]["name"] for t in kept_tools}
    assert not [name for name in surviving if name.endswith("_face")]
    # apply_tool_policy intersects handlers with the kept schemas, so the
    # shell-native vision handlers installed over the legacy ones go too
    assert set(kept_handlers) == surviving
    assert {"go_to_sleep", "reset_conversation"}.isdisjoint(surviving)
    # and what the booth does keep is still there
    assert {"remember", "recall", "play_emotion", "nod"} <= surviving


def test_home_still_ships_the_face_tools(clean_env):
    """The exclusion is a room policy, not a removal: home keeps what it has."""
    cfg = load_config("home", envs_dir=ENVS_DIR)
    known = {t["function"]["name"] for t in shipped_tool_schemas()}
    assert {"identify_face", "learn_face", "forget_face"} <= known
    kept_tools, _ = apply_tool_policy(shipped_tool_schemas(), stub_handlers(known), exclude=cfg.tools_exclude)
    assert {"identify_face", "learn_face", "forget_face"} <= {t["function"]["name"] for t in kept_tools}


def test_nothing_that_leaves_the_box_names_the_family(clean_env):
    """The privacy scan the suite was missing: cfg.instructions is only one of the
    strings sent in every environment. The fact-extraction prompt shipped
    "The user's wife is named Rachel." and the `remember` schema shipped "Sarah"
    - both were in the booth request every compaction until 2026-10-12.
    """
    haystacks = {
        "fact prompt": FACT_PROMPT,
        "summary prompt": SUMMARY_PROMPT,
        "tool schemas": json.dumps(shipped_tool_schemas()),
    }
    for env_name in ("home", "hackday"):
        haystacks[f"{env_name} instructions"] = load_config(env_name, envs_dir=ENVS_DIR).instructions
    for what, text in haystacks.items():
        lowered = text.lower()
        for private in PRIVATE_WORDS:
            assert private not in lowered, f"{what} carries the private word {private!r}"


def test_the_fact_prompt_example_is_still_a_grounded_fact():
    """Neutralising the prompt may not break the property the example was there
    for: it shows the shape of a fact `parse_facts` accepts and `grounded()`
    keeps (third person, starts 'The user', names someone actually said)."""
    example = re.search(r'e.g\. "([^"]+\.)"', FACT_PROMPT)
    assert example is not None, "the extraction prompt lost its example"
    fact = example.group(1)
    assert fact.lower().startswith("the user")
    assert parse_facts(json.dumps([fact])) == [fact]
    assert grounded([fact], fact) == [fact]


# --------------------------------------------------------------------------
# A4 — --print-env
# --------------------------------------------------------------------------


def test_print_env_round_trips_into_an_env_file(clean_env, capsys, tmp_path):
    """P2 (2026-10-12): --print-env prints *resolved* instructions, which already
    carry SHELL_NOTES. The documented verify-then-paste flow (deploy/README.md)
    would otherwise land TOOLS_NOTE + TIME_NOTE in twice; appending is idempotent
    instead, so what you read is what you get back.
    """
    shell_main.main(["--print-env"])
    printed = json.loads(capsys.readouterr().out)
    body = yaml.safe_dump({"persona": {"instructions": printed["instructions"]}})
    cfg = load_config("lab", envs_dir=write_env(tmp_path, "lab", body))
    assert cfg.instructions == printed["instructions"]
    assert cfg.instructions.count("Speak plain words") == 1
    assert cfg.instructions.endswith(SHELL_NOTES)


def test_print_env_prints_and_leaves_no_trace(clean_env, capsys, monkeypatch):
    """--print-env must not open audio, connect to the robot, write the journal,
    or touch the network — and stdout must be exactly one JSON document."""

    def forbidden(what):
        def _boom(*_args, **_kwargs):
            raise AssertionError(f"--print-env touched {what}")

        return _boom

    import socket
    import sqlite3

    # The microphone tripwire is best-effort: sounddevice is imported lazily inside
    # MicOwner/SpeakerOwner.open(), and this test already forbids those methods
    # directly, which is the guarantee that holds everywhere. Importing the library
    # just to assert we do not call it would mean the hardware-free CI env cannot run
    # the check at all - a dependency in a test for the absence of a dependency.
    try:
        import sounddevice
    except ModuleNotFoundError:
        sounddevice = None

    monkeypatch.setattr(sqlite3, "connect", forbidden("sqlite"))
    monkeypatch.setattr(socket, "socket", forbidden("the network"))
    if sounddevice is not None:
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


def test_a_foreign_env_may_not_land_on_the_home_store(tmp_path, clean_env):
    """P1 (2026-10-12): hard rule 1 is a resolution error, not a comment.

    `REACHY_MEMORY_NAMESPACE=home` is the natural "let me keep my memories at the
    demo" edit; it puts data/memories.home.json - family names, the home address -
    into the system prompt of a robot standing in a public room.
    migrate_legacy_memories already refuses this shape on the write side;
    resolution has to refuse it too, and before any audio device is opened.
    """
    clean_env.setenv("REACHY_ENV", "hackday")
    clean_env.setenv("REACHY_MEMORY_NAMESPACE", "home")
    with pytest.raises(EnvError) as err:
        load_config(envs_dir=ENVS_DIR)
    assert "home memory namespace" in str(err.value) and "hard rule 1" in str(err.value)

    # the same thing written into the file rather than the process env
    clean_env.delenv("REACHY_MEMORY_NAMESPACE")
    envs = write_env(tmp_path, "lab", "attention_profile: event\nmemory:\n  namespace: home\n")
    with pytest.raises(EnvError):
        load_config("lab", envs_dir=envs)

    # both shipped environments still resolve untouched
    clean_env.delenv("REACHY_ENV")
    assert load_config("home", envs_dir=ENVS_DIR).memory_namespace == "home"
    assert load_config("hackday", envs_dir=ENVS_DIR).memory_namespace == "hackday"
    # naming a store of its own is fine - the guard is about the home file only
    envs = write_env(tmp_path, "lab", "memory:\n  namespace: lab_booth\n")
    assert load_config("lab", envs_dir=envs).memories_path.name == "memories.lab_booth.json"


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


# --------------------------------------------------------------------------
# Hard rule 1, write side: no auto-mined fact crosses attendees
# --------------------------------------------------------------------------


def run_one_compaction(env_name, tmp_path, journal, monkeypatch, mined):
    """Assemble shell/__main__.fact_mining + ConversationMemory the way the shell
    does, then idle it through one compaction. Returns (cfg, store)."""

    def fake_extract(base_url, model, turns, known):
        async def go():
            mined.append(len(turns))
            return ["The user's sister is named Willow."]

        return go()

    monkeypatch.setattr(shell_main, "llm_extract_facts", fake_extract)
    cfg = replace(load_config(env_name, envs_dir=ENVS_DIR), data_dir=tmp_path)
    store = MemoryStore(cfg.memories_path)
    extract, save = shell_main.fact_mining(cfg, store)
    clock = [1_800_000_000.0]

    async def summarize(prior, turns):
        return "They talked."

    memory = ConversationMemory(
        journal,
        summarize,
        compact_after_s=180,
        clock=lambda: clock[0],
        extract_facts=extract,
        save_fact=save,
    )

    async def go():
        memory.add("user", "my sister is named Willow")
        clock[0] += 200  # idle past compact_after_s, as the watchdog sample does
        memory.tick(engaged=False)
        task = memory._task
        if task is not None:
            await task

    asyncio.run(go())
    return cfg, store


def test_hackday_does_not_mine_the_room_for_facts(tmp_path, clean_env, journal, monkeypatch):
    """P1 (2026-10-12): auto extraction writes through save_fact — past the router
    and the ADR 0003 attention gate — into one shared store, and `prompt_block`
    hands the last 30 facts to the next session labelled "what you know about the
    user". With one memories.hackday.json and a 5-minute session reset that says
    attendee A's business to attendee B as B's own data. `behaviour.learn_facts:
    false` stops the mining; the summary/compaction itself is untouched.
    """
    mined = []
    cfg, store = run_one_compaction("hackday", tmp_path, journal, monkeypatch, mined)
    assert cfg.learn_facts is False
    assert mined == []  # not even a request to llama.cpp
    assert store.all() == []
    assert journal.find("conversation.facts_saved") == []
    assert journal.find("conversation.compacted")  # compaction still happens


def test_home_still_learns_facts_on_its_own(tmp_path, clean_env, journal, monkeypatch):
    """A2 parity: the desk keeps doing what it did before, source tag included."""
    mined = []
    cfg, store = run_one_compaction("home", tmp_path, journal, monkeypatch, mined)
    assert cfg.learn_facts is True
    assert mined == [1]
    saved = store.all()
    assert [m["text"] for m in saved] == ["The user's sister is named Willow."]
    assert saved[0]["source"] == "auto"


def test_learn_facts_is_an_env_key(tmp_path, clean_env, monkeypatch):
    """Default true, off per env file, `REACHY_LEARN_FACTS=0` still outranks it."""
    envs = write_env(tmp_path, "lab", "behaviour:\n  learn_facts: false\n")
    assert load_config("home", envs_dir=ENVS_DIR).learn_facts is True
    assert load_config("lab", envs_dir=envs).learn_facts is False
    monkeypatch.setenv("REACHY_LEARN_FACTS", "1")
    assert load_config("lab", envs_dir=envs).learn_facts is True


def test_auto_sleep_s_is_an_env_key(tmp_path, clean_env, monkeypatch):
    """Issue #36: auto-sleep is OFF everywhere it ships - code default, home and
    hackday all resolve to 0, so he stays awake in an empty room (David's call).
    The knob still has to work, which is why a non-zero value is checked through
    both an env file and the override. Precedence is the same as every other
    layer, and an empty `REACHY_AUTO_SLEEP_S=` is no opinion - it must not zero
    a value the way a stray `REACHY_LEARN_FACTS=` once re-armed the fact miner."""
    assert ShellConfig().auto_sleep_s == 0.0
    assert load_config("home", envs_dir=ENVS_DIR).auto_sleep_s == 0.0
    assert load_config("hackday", envs_dir=ENVS_DIR).auto_sleep_s == 0.0
    envs = write_env(tmp_path, "lab", "behaviour:\n  auto_sleep_s: 45\n")
    assert load_config("lab", envs_dir=envs).auto_sleep_s == 45.0  # non-zero still resolves
    monkeypatch.setenv("REACHY_AUTO_SLEEP_S", "90")
    assert load_config("lab", envs_dir=envs).auto_sleep_s == 90.0
    assert load_config("hackday", envs_dir=ENVS_DIR).auto_sleep_s == 90.0  # override reaches the booth
    monkeypatch.setenv("REACHY_AUTO_SLEEP_S", "  ")
    assert load_config("lab", envs_dir=envs).auto_sleep_s == 45.0
    assert load_config("hackday", envs_dir=ENVS_DIR).auto_sleep_s == 0.0
    assert ShellConfig().auto_sleep_s == 0.0


def test_the_explicit_remember_tool_still_works_in_the_booth(tmp_path, clean_env):
    """A guest who asks to be remembered is a request, not mining: `remember`
    stays in the hack-day tool list and writes to the hack-day store."""
    cfg = replace(load_config("hackday", envs_dir=ENVS_DIR), data_dir=tmp_path)
    assert "remember" not in cfg.tools_exclude
    schemas = {t["function"]["name"] for t in shipped_tool_schemas()}
    kept, handlers = apply_tool_policy(
        shipped_tool_schemas(), stub_handlers(schemas), include=cfg.tools_include, exclude=cfg.tools_exclude
    )
    assert "remember" in {t["function"]["name"] for t in kept} and "remember" in handlers
    store = MemoryStore(cfg.memories_path)
    assert make_memory_handlers(store)["remember"](fact="The user's name is Ana.") == {
        "ok": True,
        "saved": "The user's name is Ana.",
    }
    assert [m["text"] for m in store.all()] == ["The user's name is Ana."]


# --------------------------------------------------------------------------
# P0-2 — the dashboard served the family to the LAN
# --------------------------------------------------------------------------


def test_the_dashboard_binds_loopback_unless_told_otherwise():
    """dashboard/ runs in its own venv, so this checks the source, not the import.

    No auth anywhere in that process: /faces serves data/known_faces.json (names +
    thumbnails) and / serves data/sessions.db. Binding 0.0.0.0 handed both to the
    LAN, and the hack-day move is exactly what puts the box on an office network.
    The kiosk reads 127.0.0.1 (deploy/kiosk), so loopback-by-default costs nothing.
    """
    src = (REPO_ROOT / "dashboard" / "server.py").read_text()
    assert 'host="0.0.0.0"' not in src
    assert re.search(r'DASHBOARD_HOST",\s*""\)\s*(\.strip\(\)\s*)?or\s*"127\.0\.0\.1"', src)
    assert re.search(r"uvicorn\.run\(\s*app,\s*host=host", src)


def test_booting_with_no_REACHY_ENV_says_so_at_WARN(clean_env, caplog, monkeypatch):
    """P2-7 (2026-10-12): a missing REACHY_ENV resolves to `home` and boots — at an
    event that is the family persona with the family memories, in public, and the
    only signal was one INFO line inside the startup noise. Fail-closed covers the
    typo, not the shrug, so the shrug has to be loud (the asyncio loop is stubbed:
    nothing here starts the shell for real).
    """
    started = []
    monkeypatch.setattr(shell_main.asyncio, "run", lambda *_a, **_k: started.append(True))

    with caplog.at_level("WARNING"):
        shell_main.main([])

    assert started and "REACHY_ENV was never set" in caplog.text and "home" in caplog.text

    caplog.clear()
    monkeypatch.setenv("REACHY_ENV", "hackday")
    with caplog.at_level("WARNING"):
        shell_main.main([])

    assert "was never set" not in caplog.text


# --------------------------------------------------------------------------
# P2-D (2026-10-12 review): every greeter knob is room-configurable
# --------------------------------------------------------------------------


def test_greeter_cadence_and_centering_come_from_the_env_file(tmp_path, clean_env):
    envs = write_env(
        tmp_path,
        "lab",
        """
attention_profile: event
behaviour:
  greeter: true
  greeter_idle_fps: 1.5
  greeter_engaged_fps: 2.5
  greeter_face_centering: true
""",
    )
    cfg = load_config("lab", envs_dir=envs)
    assert cfg.greeter_idle_fps == 1.5 and cfg.greeter_engaged_fps == 2.5
    assert cfg.greeter_face_centering is True


def test_greeter_cadence_variables_outrank_the_file_and_empty_means_unset(tmp_path, clean_env, monkeypatch):
    envs = write_env(
        tmp_path,
        "lab",
        """
attention_profile: event
behaviour:
  greeter_idle_fps: 1.5
  greeter_face_centering: true
""",
    )
    monkeypatch.setenv("REACHY_GREETER_ENGAGED_FPS", "4")
    assert load_config("lab", envs_dir=envs).greeter_engaged_fps == 4.0
    monkeypatch.setenv("REACHY_GREETER_IDLE_FPS", "")  # present-but-empty: no opinion
    monkeypatch.setenv("REACHY_GREETER_FACE_CENTERING", "")
    cfg = load_config("lab", envs_dir=envs)
    assert cfg.greeter_idle_fps == 1.5 and cfg.greeter_face_centering is True


def test_both_shipped_rooms_keep_the_greeter_defaults(clean_env):
    for name in ("home", "hackday"):
        cfg = load_config(name, envs_dir=ENVS_DIR)
        assert cfg.greeter_idle_fps == 2.0 and cfg.greeter_engaged_fps == 3.0
        assert cfg.greeter_face_centering is False  # gated off until benched (docs/pins.yaml)


def test_the_learn_facts_note_is_removable_without_touching_home():
    """`instructions_without_auto_learn` works by exact substring, so the day somebody
    rewords TOOLS_NOTE the booth room silently keeps being told to harvest facts.
    This fails the moment the two strings drift apart."""
    from shell.config import LEARN_FACTS_NOTE, TOOLS_NOTE, instructions_without_auto_learn

    assert LEARN_FACTS_NOTE in TOOLS_NOTE, (
        "the note and the shared tail drifted; the booth prompt still says to auto-remember"
    )
    stripped = instructions_without_auto_learn(TOOLS_NOTE)
    assert LEARN_FACTS_NOTE not in stripped
    assert "  " not in stripped, "removal leaves a double space in the prompt"
    # the explicit half survives: being asked to remember is not the same act
    assert "asks you to remember, recall or forget something" in stripped


def test_a_room_that_disables_fact_learning_is_not_also_told_to_do_it(clean_env):
    """hackday sets behaviour.learn_facts: false while `remember` stays in its tool
    list by design (a guest asking to be remembered is fine). Before 2026-10-12 the
    shared prompt tail still instructed him to save lasting facts about 'the user or
    their family' unprompted - at a booth, a stranger's child's age written down
    because a prompt said to. Prompt and policy are one decision now."""
    from shell.config import LEARN_FACTS_NOTE

    booth = load_config("hackday", envs_dir=ENVS_DIR)
    assert booth.learn_facts is False
    assert LEARN_FACTS_NOTE not in booth.instructions
    assert "remember, recall or forget" in booth.instructions

    home = load_config("home", envs_dir=ENVS_DIR)
    assert home.learn_facts is True
    assert LEARN_FACTS_NOTE in home.instructions
    assert len(home.instructions.encode()) == HOME_PROMPT_BYTES, (
        "home parity is the A2 bar; see tests/parity_baseline.py before moving it"
    )


def test_reconcile_is_a_noop_when_learning_is_on():
    from shell.env import reconcile_learn_facts_prompt

    cfg = ShellConfig(env="home")
    before = cfg.instructions
    assert reconcile_learn_facts_prompt(cfg).instructions == before
