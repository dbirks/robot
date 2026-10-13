"""The volume policy is pure, so it is testable without a speaker (ADR 0006 rule:
hardware-free tests). What these pin down is the *vocabulary* the model will use -
a 4B model picking 'much louder' at a booth must land somewhere a human meant."""

from __future__ import annotations

from pathlib import Path

from shell import volume as V

ROOT = Path(__file__).resolve().parent.parent


def test_relative_words_move_from_the_current_level():
    assert V.resolve("louder", 1.0) == 1.1
    assert V.resolve("quieter", 1.0) == 0.9
    assert V.resolve("much louder", 1.5) == 1.75
    assert V.resolve("volume down", 0.5) == 0.4


def test_a_relative_word_still_works_when_the_mixer_is_unreadable():
    """`current=None` is the booth state when somebody plugged the card in muted.
    Refusing there was the old behaviour: it made 'louder' a robot that ignores you
    twice. Now it steps from the boot default."""
    assert V.resolve("louder", None) == round(V.DEFAULT_GAIN + V.STEP, 2)
    assert V.resolve("max", None) == V.MAX_GAIN


def test_named_levels_are_loud_enough_for_a_crowd():
    assert V.NAMED["normal"] == V.DEFAULT_GAIN
    assert V.NAMED["max"] == V.MAX_GAIN
    assert V.NAMED["medium"] == 0.7
    assert V.NAMED["quiet"] < V.NAMED["medium"] < V.NAMED["loud"] < V.NAMED["max"]
    # whisper must stay useful: a step of 0.0 would make 'whisper' and 'mute' the
    # same answer, and a person asking for quiet is not asking for silence
    assert V.NAMED["whisper"] > 0.0


def test_percentages_survive_the_way_people_say_them():
    assert V.resolve("150%", 0.5) == 1.5
    assert V.resolve("200%", 0.5) == 2.0
    assert V.resolve("200", 0.5) == 2.0  # nobody means two hundredths of a percent
    assert V.resolve("1.4", 0.5) == 1.4
    assert V.resolve("40", 0.5) == 0.4
    assert V.resolve("300%", 0.5) == V.MAX_GAIN  # out of range, clamped not refused


def test_unintelligible_levels_refuse_instead_of_guessing():
    """A made-up word must not become a volume change: the model will occasionally
    pass 'ambient' or 'normal-ish', and silently picking a number is worse."""
    for junk in ("", "   ", "normal-ish", "ambient", "abc"):
        assert V.resolve(junk, 1.0) is None, junk


def test_the_ceiling_is_the_ceiling_everywhere():
    assert V.clamp_gain(99.0) == V.MAX_GAIN
    assert V.clamp_gain(-5.0) == V.MIN_GAIN
    assert V.clamp_gain(float("nan")) == V.MIN_GAIN  # nan would leave the old level playing
    assert V.clamp_gain(1.5, ceiling=1.2) == 1.2  # a room may still be tuned quieter


def test_clipping_is_flagged_not_hidden():
    """Between 1.2 (measured distortion) and 2.0 (asked for, for a crowd) the tool
    is allowed to obey, but it must say so, so 'he sounds awful' has an answer."""
    assert V.clipping_risk(V.resolve("max", 1.0) or 0) is True
    assert V.clipping_risk(V.resolve("normal", None) or 0) is False
    assert V.CLIPPING_ABOVE < V.MAX_GAIN, "if the ceiling drops below the distortion point the flag is dead code"


def test_the_tool_path_delegates_instead_of_keeping_its_own_numbers():
    """app/robot_tools.py used to own a private level map with its own 1.2 cap, and
    the dashboard owned a second clamp. The policy is one module now; this fails if
    somebody reintroduces a literal table in the legacy bridge."""
    src = (ROOT / "app" / "robot_tools.py").read_text()
    assert "from shell import volume as V" in src, "set_volume stopped reading the shared policy"
    assert '"whisper": 0.2' not in src, "the retired private level table came back"
    assert "min(1.2, target)" not in src, "a second hardcoded ceiling came back"


# --- the 2026-10-13 voice-path fixes: reporting, and not claiming inability ---


def test_read_words_ask_for_the_level_and_are_not_change_words():
    assert V.is_read_request("current") and V.is_read_request("How loud are you?")
    assert V.is_read_request("  what is it  ") and V.is_read_request("status.")
    assert not V.is_read_request("quieter") and not V.is_read_request("louder")
    assert not V.is_read_request("150%") and not V.is_read_request("")


def test_a_read_word_cannot_shadow_a_level_that_changes_something():
    """ "level" is a question here and "loud" is a command there; if those sets ever intersect,
    "set the volume to max" could silently become a status report - the worst kind of robot that
    answers a request it was given."""
    change_words = set(V.STEPS) | set(V.NAMED)
    assert not (V.READ_WORDS & change_words), f"collision: {sorted(V.READ_WORDS & change_words)}"


def test_the_read_path_returns_before_anything_is_written():
    """Order is the contract: the read branch must sit above `set_volume_boost(target)`, or
    "how loud are you?" changes the room. Asserted structurally because a unit test of this
    handler needs the whole legacy bridge."""
    src = (ROOT / "app" / "robot_tools.py").read_text()
    read = src.index("if V.is_read_request(level):")
    write = src.index("set_volume_boost(target)")
    assert read < write, "the report path moved below the write"
    assert 'changed": False' in src[read:write], "the report stopped saying it changed nothing"


def _tool_schema(name: str) -> dict:
    """A tool schema, without importing the module that owns it.

    `app/robot_tools.py` pulls in cv2, which the bare CI environment does not have, so importing it
    would turn this file's schema guards into skips - and a skipped guard is how a description
    regresses back to "I cannot" with nothing going red. TOOLS is constant data, so the AST can
    read it directly."""
    import ast

    tree = ast.parse((ROOT / "app" / "robot_tools.py").read_text())
    for st in ast.walk(tree):
        # TOOLS is annotated (`TOOLS: list[dict] = [...]`), which is an AnnAssign, not an Assign -
        # checking only the latter made this guard report "not declared" while the list was right
        # there on the page.
        names = [getattr(x, "id", "") for x in getattr(st, "targets", []) or [getattr(st, "target", None)]]
        if isinstance(st, (ast.Assign, ast.AnnAssign)) and "TOOLS" in names:
            for tool in ast.literal_eval(st.value):
                fn = tool.get("function") or {}
                if fn.get("name") == name:
                    return fn
    raise AssertionError(f"{name} is not declared in a TOOLS list")


def test_the_schema_does_not_let_him_claim_he_cannot_be_louder():
    """Live 2026-10-13 13:14, three times: "turn the volume down" produced NO tool call and the
    sentence "I'm afraid I cannot adjust the volume as you've requested" - on a box where the tool
    exists, is enabled, and had worked a few seconds earlier. A 4B model that is not told it is
    allowed will decline politely, and a guest hears a broken robot either way."""
    fn = _tool_schema("set_volume")
    assert "never answer that he cannot" in fn["description"], "the authority clause was dropped"
    level = fn["parameters"]["properties"]["level"]["description"]
    assert "'current'" in level, "the read word must be reachable from the schema"
