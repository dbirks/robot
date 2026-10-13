"""The booth's text/voice pairing is a decision, and it is guarded as one.

OWNER DECISION, NAMED AND DATED: David Birks, 2026-10-13, at the Hack Day booth - "make it
really Obama... remove that line that says deliberately naming no real person. We're doing
that on purpose... you update all three gates with my name." So `envs/hackday.yaml` now names
`persona.name: obama` and carries NO inline `persona.instructions:` block: the WORDS, the ack
pool and the VOICE all come from `personas/obama.yaml`, which is the documented mechanism
(`shell/env.py` `merge_persona_doc`: an inline block outranks the persona file, so a room that
keeps one silently decides its own words while another file decides its timbre).

What these tests guard after that change — two subsystems, one robot:
  - the room has not re-grown an inline shadow that outranks the persona file;
  - the ack pool still belongs to the voice the persona names;
  - the voice is still the one measured talker GGUF on this box (`personas/obama.yaml`),
    with the blank style line `scripts/run_s2s.sh` depends on;
  - the words name exactly ONE public figure and nobody private;
  - the room's own file still says out loud whose decision this is, so the next reader
    does not "fix" it back in front of a room.

Nothing about the timbre moved: `voice:` is byte-identical in `personas/obama.yaml` and
`personas/marvin.yaml` (no Marvin model exists), which tests/test_persona.py keeps true.

Hardware-free, and never reads or writes `~/.config`: `voice_routing_gaps()` is pointed at
a temp directory that pretends to be the systemd unit dir.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dashboard"))

import rehearse_booth as rb  # ty: ignore[unresolved-import]  # noqa: E402

import shell.env as shell_env
from dashboard import persona_panel as pp  # noqa: E402


def apply_script():
    spec = importlib.util.spec_from_file_location("apply_env", ROOT / "scripts" / "apply_env.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_env(monkeypatch):
    """Measure the files, not this desk: shell/config.py has already pushed `.env`
    into os.environ, and a live REACHY_PERSONA/REACHY_INSTRUCTIONS would outrank the
    room's own file (same fixture shape as tests/test_persona.py)."""
    import shell.config  # noqa: F401  - load_dotenv() runs at import, on purpose

    for key in ("REACHY_ENV", "REACHY_PERSONA", "REACHY_INSTRUCTIONS", "REACHY_ACK_DIR"):
        monkeypatch.delenv(key, raising=False)


def room_persona(env_name: str) -> tuple[str, dict]:
    """The persona the room's own file names, and the parsed room doc (no picker involved)."""
    doc = shell_env.load_env_doc(env_name)
    name, source = shell_env.persona_selection(doc)
    assert source == "persona.name", f"envs/{env_name}.yaml does not name a persona itself"
    return name, doc


def persona_text(name: str) -> str:
    return str(shell_env.load_persona_doc(name)["instructions"]).strip()


# --------------------------------------------------------------------- the pairing


def test_hackday_carries_no_inline_shadow_over_the_persona_that_supplies_its_voice(clean_env):
    """The deletion IS the mechanism (owner decision: David Birks, 2026-10-13). An inline
    block would make the room speak its own words while personas/obama.yaml went on
    deciding the voice and the acks - two sources, one robot, and the winner is invisible
    from the picker.
    """
    name, doc = room_persona("hackday")
    assert "instructions" not in doc["persona"], (
        f"envs/hackday.yaml re-grew an inline instructions block: it outranks personas/{name}.yaml "
        "(shell/env.py merge_persona_doc), so the words and the voice come from two files again"
    )
    persona = shell_env.load_persona_doc(name)
    cfg = shell_env.load_config("hackday")
    expected = shell_env.persona_instructions(str(persona["instructions"]).strip())
    if not cfg.learn_facts:  # the same reconcile tests/test_persona.py applies
        expected = shell_env.instructions_without_auto_learn(expected)
    assert cfg.instructions == expected
    assert str(doc["persona"]["ack_dir"]).strip() == str(persona["ack_dir"]).strip(), (
        "the ack pool is spoken audio: it must belong to the voice the persona names"
    )


def test_hackday_voice_is_the_measured_obama_talker_its_text_shares(clean_env):
    """One file supplies both halves tonight; a persona rename must not silently move the
    booth onto an unmeasured voice (AGENTS.md: nothing runs on this box unmeasured)."""
    name, _ = room_persona("hackday")
    assert name == "obama", "the room no longer names the persona its owner chose (David Birks, 2026-10-13)"
    voice = pp.voice_of(shell_env.load_persona_doc(name))
    assert voice == pp.voice_of(shell_env.load_persona_doc("obama"))
    assert voice["S2S_QWEN3_TALKER_GGUF"].endswith("qwen-talker-1.7b-obama-Q8_0.gguf")
    assert voice["S2S_QWEN3_INSTRUCT"] == "", "a blank style line degrades the output; see scripts/run_s2s.sh"


def test_the_room_names_where_its_words_come_from_and_under_whose_decision(clean_env):
    """A booth persona that names a real public figure is a named owner decision, not drift:
    the room's own file has to carry the attribution with it. The old "BORROWED VOICE" note
    said the opposite thing (that the words and the timbre came from two different files);
    it must not come back."""
    text = (ROOT / "envs" / "hackday.yaml").read_text()
    assert "David Birks" in text and "2026-10-13" in text, "the owner decision lost its attribution"
    assert "personas/obama.yaml" in text, "the room does not say where its words come from"
    assert "BORROWED VOICE" not in text, "the two-files note is back: one file supplies both halves now"


def test_the_words_hackday_speaks_name_the_owners_one_public_figure_and_no_private_person(clean_env):
    """David Birks, 2026-10-13: one public figure, named on purpose, and nothing private.

    `rb.REAL_PEOPLE` matches him as two word tokens (`barack`, `obama`). The attendee
    protections never depended on the persona's name, and the first-person line that says
    he is a robot is what keeps an Obama-voiced desk robot an impression on a recording.
    """
    name, _ = room_persona("hackday")
    words = str(shell_env.load_persona_doc(name)["instructions"])
    assert rb.named_people(words) == {"barack", "obama"}
    assert rb.named_people(shell_env.load_config("hackday").instructions) == {"barack", "obama"}
    for private in ("david", "rachel", "barney", "rickman"):
        assert private not in words.lower(), f"the booth persona names a private person: {private}"
    assert "You are still Reachy the robot, not actually Obama" in words


# --------------------------------------------------------------------- voice routing


def write_unit(dir_: Path, unit: str, body: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{unit}.service").write_text(body)


def test_a_unit_that_never_loads_the_render_is_said_out_loud(tmp_path):
    """The failure this box is living with: `wrote deploy/generated/persona.env` + exit 0,
    and no service on the machine will ever read that path."""
    units = tmp_path / "user"
    write_unit(units, "reachy-s2s", "[Service]\nEnvironmentFile=-%h/dev/robot/.env\n")
    write_unit(
        units,
        "reachy-shell",
        "[Service]\nEnvironmentFile=-%h/dev/robot/.env\nEnvironmentFile=-%h/dev/robot/deploy/generated/persona.env\n",
    )
    gaps = apply_script().voice_routing_gaps(Path("persona.env"), unit_dir=units)
    assert len(gaps) == 1 and "reachy-s2s" in gaps[0] and "NOT the talker" in gaps[0]


def test_the_measured_order_is_persona_env_last(tmp_path):
    units = tmp_path / "user"
    write_unit(
        units,
        "reachy-s2s",
        "[Service]\nEnvironmentFile=-%h/dev/robot/deploy/generated/persona.env\nEnvironmentFile=-%h/dev/robot/.env\n",
    )
    gaps = apply_script().voice_routing_gaps(Path("persona.env"), unit_dir=units)
    assert len(gaps) == 1 and "inverting the measured order" in gaps[0]


def test_an_uninstalled_box_is_quiet(tmp_path, clean_env):
    """CI has no ~/.config/systemd/user: a guard that cries there gets ignored here."""
    assert apply_script().voice_routing_gaps(Path("persona.env"), unit_dir=tmp_path / "nope") == []
    # and the shipped reference units pass their own check
    units = tmp_path / "user"
    for unit in apply_script().UNITS:
        write_unit(units, unit, (ROOT / "deploy" / "systemd" / f"{unit}.service").read_text())
    assert apply_script().voice_routing_gaps(Path("persona.env"), unit_dir=units) == []
