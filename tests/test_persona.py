"""The persona picker (issue #35): one file = one voice + one manner, and the rules
that decide which half is actually in force.

Hardware-free: no audio device, no robot, no network. `deploy/generated/persona.env`
is only ever written under tmp_path, and no test runs systemctl — apply_env.py is
exercised with --check and with --out/--lock pointed at a temp directory.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

import shell.env as shell_env
from dashboard import persona_panel as pp
from shell.config import persona_instructions

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))


def persona_text(name: str) -> str:
    return str(pp_load_persona(name)["instructions"]).strip()


def pp_load_persona(name: str) -> dict:
    return yaml.safe_load((ROOT / "personas" / f"{name}.yaml").read_text())


@pytest.fixture
def clean_env(monkeypatch):
    """The live .env sets REACHY_INSTRUCTIONS/REACHY_ACK_DIR on this desk; importing
    shell.config has already pushed them into os.environ, so clear the ones that would
    make these tests measure the desk instead of the code (same trick as check_env_safety).
    """
    import shell.config  # noqa: F401  - load_dotenv() runs at import, on purpose

    for key in (
        "REACHY_ENV",
        "REACHY_PERSONA",
        "REACHY_INSTRUCTIONS",
        "REACHY_ACK_DIR",
        "REACHY_MEMORY_NAMESPACE",
    ):
        monkeypatch.delenv(key, raising=False)


def write_env(tmp_path: Path, name: str, doc: dict) -> Path:
    envs = tmp_path / "envs"
    envs.mkdir(exist_ok=True)
    (envs / f"{name}.yaml").write_text(yaml.safe_dump(doc))
    return envs


def apply_script():
    """scripts/ is not a package; load it the way an operator runs it."""
    spec = importlib.util.spec_from_file_location("apply_env", ROOT / "scripts" / "apply_env.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# the two shipped personas
# --------------------------------------------------------------------------


def test_two_personas_ship(clean_env):
    assert shell_env.known_personas() == ["marvin", "obama"]


def test_obama_is_byte_identical_to_the_persona_home_has_always_used(clean_env):
    """The A2 parity bar, measured on the resolved config with REACHY_ENV unset."""
    cfg = shell_env.load_config("home")
    assert len(cfg.instructions.encode()) == 1568
    assert cfg.instructions == persona_instructions(persona_text("obama"))
    assert cfg.ack_dir == Path("sounds/acks/obama")


def test_the_marvin_text_is_the_hackday_persona_of_today(clean_env):
    cfg = shell_env.load_config("hackday")
    assert cfg.instructions == persona_instructions(persona_text("marvin"))
    assert "never mention anyone's family" in cfg.instructions
    assert "hack day" in cfg.instructions.lower()
    # behaviour-only: naming no real person, so a public screen carries no likeness claim
    assert not any(word in persona_text("marvin").lower() for word in ("obama", "rickman", "broadbent"))


def test_marvin_borrows_the_obama_voice_until_a_model_exists(clean_env):
    assert pp.voice_of(pp_load_persona("marvin")) == pp.voice_of(pp_load_persona("obama"))


# --------------------------------------------------------------------------
# resolution and precedence (shell/env.py)
# --------------------------------------------------------------------------


def test_a_bare_persona_name_resolves_from_the_persona_file(tmp_path, clean_env):
    envs = write_env(tmp_path, "lab", {"persona": "marvin", "memory": {"namespace": "lab"}})
    cfg = shell_env.load_config("lab", envs_dir=envs, personas_dir=ROOT / "personas")
    assert cfg.instructions == persona_instructions(persona_text("marvin"))
    assert cfg.ack_dir == Path(pp_load_persona("marvin")["ack_dir"])


def test_inline_instructions_outrank_the_named_persona(tmp_path, clean_env):
    """The rule that keeps the shipped rooms byte-exact: the file's own words win."""
    envs = write_env(
        tmp_path,
        "lab",
        {"persona": {"name": "obama", "instructions": "Plain desk robot."}, "memory": {"namespace": "lab"}},
    )
    cfg = shell_env.load_config("lab", envs_dir=envs, personas_dir=ROOT / "personas")
    assert cfg.instructions == persona_instructions("Plain desk robot.")


def test_the_picker_outranks_the_rooms_persona_name(tmp_path, monkeypatch, clean_env):
    """REACHY_PERSONA arrives from deploy/generated/persona.env as a process variable,
    so it rides the existing precedence rule instead of needing a new one."""
    envs = write_env(tmp_path, "lab", {"persona": "obama", "memory": {"namespace": "lab"}})
    monkeypatch.setenv("REACHY_PERSONA", "marvin")
    cfg = shell_env.load_config("lab", envs_dir=envs, personas_dir=ROOT / "personas")
    assert cfg.instructions == persona_instructions(persona_text("marvin"))


def test_an_empty_persona_variable_is_no_opinion(tmp_path, monkeypatch, clean_env):
    envs = write_env(tmp_path, "lab", {"persona": "marvin", "memory": {"namespace": "lab"}})
    monkeypatch.setenv("REACHY_PERSONA", "  ")
    cfg = shell_env.load_config("lab", envs_dir=envs, personas_dir=ROOT / "personas")
    assert cfg.instructions == persona_instructions(persona_text("marvin"))


def test_naming_a_persona_that_has_no_file_is_a_hard_error(tmp_path, clean_env):
    envs = write_env(tmp_path, "lab", {"persona": "alan-rickman"})
    with pytest.raises(shell_env.PersonaError) as exc:
        shell_env.load_config("lab", envs_dir=envs, personas_dir=tmp_path / "personas")
    assert "no persona file" in str(exc.value)


def test_process_instructions_beat_the_persona_too(tmp_path, monkeypatch, clean_env):
    envs = write_env(tmp_path, "lab", {"persona": "marvin", "memory": {"namespace": "lab"}})
    monkeypatch.setenv("REACHY_INSTRUCTIONS", "Operator override.")
    cfg = shell_env.load_config("lab", envs_dir=envs, personas_dir=ROOT / "personas")
    assert cfg.instructions == persona_instructions("Operator override.")


def test_shell_config_still_has_no_persona_field(clean_env):
    """Parity is `config_to_dict(ShellConfig()) == config_to_dict(load_config('home'))`;
    a `persona:` attribute would have broken it, so the picker reports separately."""
    assert "persona" not in shell_env.config_to_dict(shell_env.ShellConfig())


def test_persona_report_names_the_room_and_the_inline_shadow(clean_env):
    report = shell_env.persona_report("home")
    assert report["persona"] == "obama" and report["persona_source"] == "persona.name"
    assert report["inline_instructions"] is True
    assert [p["name"] for p in report["personas"]] == ["marvin", "obama"]
    assert report["personas"][0]["voice"]["engine"] == "qwen3"


# --------------------------------------------------------------------------
# the pure panel module (no yaml, no fastapi)
# --------------------------------------------------------------------------


def test_persona_panel_imports_only_stdlib():
    tree = ast.parse((ROOT / "dashboard" / "persona_panel.py").read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    extra = mods - set(sys.stdlib_module_names)
    assert not extra, f"non-stdlib import in persona_panel: {sorted(extra)}"


def obama_voice() -> dict:
    return pp.persona_env_values(pp_load_persona("obama"))


def test_the_persona_writes_every_voice_key_so_a_stale_env_line_cannot_survive():
    values = obama_voice()
    assert values["S2S_TTS"] == "qwen3"
    assert values["S2S_QWEN3_TALKER_GGUF"].endswith("qwen-talker-1.7b-obama-Q8_0.gguf")
    assert values["S2S_QWEN3_SPEAKER"] == "laxmikant"
    assert values["S2S_QWEN3_CODEC_GGUF"] == ""  # run_s2s.sh globs it when empty
    assert values["PERSONA"] == "obama" and values["REACHY_PERSONA"] == "obama"
    assert values["REACHY_ACK_DIR"] == "sounds/acks/obama"


def test_a_kokoro_persona_clears_the_qwen3_keys_and_moves_the_speaker():
    values = pp.voice_of({"name": "plain", "voice": {"engine": "kokoro", "speaker": "bm_daniel", "model": "leftover"}})
    assert values["S2S_TTS"] == "kokoro"
    assert values["KOKORO_VOICE"] == "bm_daniel"
    assert values["S2S_QWEN3_MODEL"] == "" and values["S2S_QWEN3_TALKER_GGUF"] == ""


def test_no_persona_can_ever_stamp_the_system_prompt():
    """One global file read in both rooms: a stamped home persona is the leak."""
    for name in shell_env.known_personas():
        assert "REACHY_INSTRUCTIONS" not in pp.persona_env_values(pp_load_persona(name))


def test_render_and_parse_round_trip_quoted_and_plain_values():
    values = {**obama_voice(), "S2S_QWEN3_INSTRUCT": "dry, weary #1", "REACHY_ACK_DIR": "sounds/acks/my voice"}
    text = pp.render_persona_env(values)
    assert pp.parse_env_text(text) == values
    assert "\n#" in text  # the header says where the file came from


def test_restart_decision(clean_env):
    values = pp.persona_env_values(pp_load_persona("obama"))
    assert pp.restart_plan(values, values) == {"changed": [], "units": [], "needed": False}
    assert pp.restart_plan({}, values)["units"] == ["reachy-s2s", "reachy-shell"]
    voice_only = {**values, "S2S_QWEN3_SPEAKER": "Ryan"}
    plan = pp.restart_plan(values, voice_only)
    assert plan["changed"] == ["S2S_QWEN3_SPEAKER"] and plan["units"] == ["reachy-s2s", "reachy-shell"]
    acks_only = {**values, "REACHY_ACK_DIR": "sounds/acks/other"}
    assert pp.restart_plan(values, acks_only)["units"] == ["reachy-shell"]


def test_the_unmeasured_voice_is_said_out_loud_and_refused_only_by_strict():
    pins = (ROOT / "docs" / "pins.yaml").read_text()
    warnings = pp.pins_guard(pp_load_persona("obama"), pins)
    assert warnings, "the guard went quiet on a voice with no pins entry"
    assert any("pins.yaml" in line for line in warnings)
    measured = (
        pins + "\n  tts:\n    id: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\n    weights: qwen-talker-1.7b-obama-Q8_0.gguf\n"
    )
    assert pp.pins_guard(pp_load_persona("obama"), measured) == []
    assert pp.pins_guard({"name": "k", "voice": {"engine": "kokoro"}}, pins) == []


def test_a_voice_field_with_no_flag_is_not_dropped_silently():
    persona = {"name": "x", "voice": {"engine": "qwen3", "speed": 1.2}}
    assert pp.voice_warnings(persona) == [f"voice.speed={1.2!r} is not applied: {pp.UNSUPPORTED['speed']}"]


def test_the_inline_shadow_is_reported_from_the_rooms_file(clean_env):
    assert pp.inline_shadow({"persona": {"name": "obama", "instructions": "text"}}) is True
    assert pp.inline_shadow({"persona": {"name": "obama"}}) is False
    assert pp.inline_shadow({"persona": "obama"}) is False


def test_one_apply_at_a_time(tmp_path):
    lock = tmp_path / "persona.apply.lock"
    assert pp.apply_busy(lock)["busy"] is False
    assert pp.try_acquire_lock(lock, pid=4242, now=100.0) is True
    assert pp.try_acquire_lock(lock, pid=99, now=101.0) is False
    assert pp.apply_busy(lock, now=110.0) == {"busy": True, "age_s": 10.0, "stale": False}
    assert pp.apply_busy(lock, now=100.0 + pp._LOCK_STALE_S + 5)["busy"] is False
    pp.release_lock(lock)
    assert pp.try_acquire_lock(lock) is True


# --------------------------------------------------------------------------
# scripts/apply_env.py
# --------------------------------------------------------------------------


def test_check_writes_nothing(tmp_path, capsys, clean_env):
    out = tmp_path / "persona.env"
    rc = apply_script().main(["--check", "--persona", "obama", "--out", str(out), "--lock", str(tmp_path / "l")])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "S2S_QWEN3_TALKER_GGUF=" in printed and "would write" in printed
    assert not out.exists()


def test_apply_renders_atomically_and_keeps_the_previous_one(tmp_path, capsys, clean_env):
    script = apply_script()
    out, lock = tmp_path / "generated" / "persona.env", tmp_path / "l"
    assert script.main(["--persona", "obama", "--out", str(out), "--lock", str(lock)]) == 0
    first = pp.parse_env_text(out.read_text())
    assert first["PERSONA"] == "obama" and first["REACHY_PERSONA"] == "obama"
    assert "needs restart" in capsys.readouterr().out
    assert script.main(["--persona", "marvin", "--out", str(out), "--lock", str(lock)]) == 0
    second = pp.parse_env_text(out.read_text())
    assert second["PERSONA"] == "marvin"
    assert pp.parse_env_text(out.with_suffix(".env.prev").read_text()) == first
    assert not list(tmp_path.glob("*.tmp"))  # no temp file left behind


def test_applying_the_same_persona_twice_asks_for_no_restart(tmp_path, capsys, clean_env):
    script, out, lock = apply_script(), tmp_path / "persona.env", tmp_path / "l"
    script.main(["--persona", "obama", "--out", str(out), "--lock", str(lock)])
    capsys.readouterr()
    assert script.main(["--persona", "obama", "--out", str(out), "--lock", str(lock)]) == 0
    assert "unchanged" in capsys.readouterr().out


def test_a_second_concurrent_apply_is_refused_and_changes_nothing(tmp_path, capsys, clean_env):
    out, lock = tmp_path / "persona.env", tmp_path / "l"
    assert pp.try_acquire_lock(lock) is True  # another operator, or the dashboard's click
    rc = apply_script().main(["--persona", "marvin", "--out", str(out), "--lock", str(lock)])
    assert rc == 3
    assert "second concurrent apply" in capsys.readouterr().err
    assert not out.exists()


def test_the_lock_holder_is_told_not_to_reacquire(tmp_path, clean_env):
    out, lock = tmp_path / "persona.env", tmp_path / "l"
    assert pp.try_acquire_lock(lock) is True
    assert apply_script().main(["--persona", "obama", "--out", str(out), "--lock", str(lock), "--lock-held"]) == 0
    assert pp.parse_env_text(out.read_text())["PERSONA"] == "obama"


def test_strict_refuses_an_unmeasured_voice(tmp_path, capsys, clean_env):
    out = tmp_path / "persona.env"
    rc = apply_script().main(["--persona", "obama", "--strict", "--out", str(out), "--lock", str(tmp_path / "l")])
    captured = capsys.readouterr()
    assert rc == 4 and not out.exists()
    assert "refused by --strict" in captured.err
    assert "pins.yaml" in captured.err + captured.out


def test_the_room_is_warned_about_its_own_inline_block(tmp_path, capsys, clean_env):
    apply_script().main(
        ["--check", "--persona", "marvin", "--out", str(tmp_path / "p.env"), "--lock", str(tmp_path / "l")]
    )
    assert "NOT the words" in capsys.readouterr().err


def test_apply_env_never_names_the_retired_units_or_the_room(tmp_path, clean_env):
    src = (ROOT / "scripts" / "apply_env.py").read_text()
    assert "REACHY_ENV=" not in src and '"REACHY_ENV"' not in src
    for retired in ("reachy-agent", "qwen3-tts"):
        assert retired not in src


# --------------------------------------------------------------------------
# the wiring: units and routes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("unit", ["reachy-s2s", "reachy-shell"])
def test_persona_env_is_listed_after_env_in_both_units(unit):
    """The measured ordering (deploy/README.md): a later EnvironmentFile= wins per key,
    so the persona outranks a stray .env voice line rather than losing to it."""
    text = (ROOT / "deploy" / "systemd" / f"{unit}.service").read_text()
    assert text.index("EnvironmentFile=-%h/dev/robot/deploy/generated/persona.env") > text.index(
        "EnvironmentFile=-%h/dev/robot/.env"
    )


def test_the_persona_routes_exist_and_are_gated():
    src = (ROOT / "dashboard" / "server.py").read_text()
    assert '@app.post("/persona/apply"' in src and '@app.get("/persona"' in src
    body = src.split("async def persona_apply")[1].split("@app.")[0]
    assert "refuse_mutation(request)" in body, "the picker escaped the read-only gate"
    assert "try_acquire_lock" in body and "another apply is running" in body
    assert "apply_env.py" in body and "--lock-held" in body


def test_the_room_is_still_not_a_control_on_the_persona_page():
    text = (ROOT / "dashboard" / "templates" / "partials" / "persona.html").read_text()
    for control in ("<select", "<input"):
        assert control not in text
    assert "REACHY_ENV" in text and "never a control" in text
    page = (ROOT / "dashboard" / "templates" / "persona.html").read_text()
    assert "hx-post" not in page
    assert "readout, not a control" in text


def test_the_generated_file_is_not_tracked():
    import subprocess

    out = subprocess.run(
        ["git", "check-ignore", "-q", "deploy/generated/persona.env"], cwd=ROOT, capture_output=True
    ).returncode
    assert out == 0


def test_only_the_selected_persona_prints_its_words():
    """The panel is on a screen in a public room. Rendering every persona's full
    instructions would print the home persona - which names a real living person -
    beside the booth one, contradicting the rule in envs/hackday.yaml (#34)."""
    partial = (ROOT / "dashboard" / "templates" / "partials" / "persona.html").read_text()
    line = [ln for ln in partial.splitlines() if "p.instructions" in ln]
    assert line, "the persona partial stopped showing instructions at all"
    for ln in line:
        assert "selected_for_room" in ln, f"instructions render unconditionally: {ln.strip()[:80]}"


def test_the_apply_lock_outlives_the_restart_it_guards():
    """Two units x a 180 s subprocess timeout is the worst case an apply can occupy;
    a stale-lock window shorter than that lets a second apply start mid-restart."""
    from dashboard import persona_panel

    assert persona_panel._LOCK_STALE_S > 2 * 180, (
        f"_LOCK_STALE_S={persona_panel._LOCK_STALE_S} is shorter than the restart it is meant to serialise"
    )
