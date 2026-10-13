"""The prompt/tool pair for "what time is it?" (2026-10-12 booth rehearsal).

Three things used to disagree in front of a crowd:

  * `shell/config.py` TIME_NOTE told the model the newest per-message stamp "is
    today's date and the current time - use it",
  * the same note ended with "Never read the stamps aloud",
  * TOOLS_NOTE said to call a tool when the guest "asks for the time".

Qwen 3.5 4B resolved that by doing neither half: no `get_time` call, and
"It is still late ... I have checked the time, and it remains as it was." (verbatim in
the 23:41 run of `scripts/rehearse_booth.py`) — a time asserted, never verified, and a
save claim nobody could audit.

The settled answer, proven in `tests/test_llm_proxy.py::
test_a_fresh_session_re_sends_a_repeated_line_with_its_old_stamp`: a stamp is a LOCAL
clock reading taken when the guest SPOKE and cached per message text for the life of
the proxy process, so it can be minutes-to-hours stale and is authoritative for the
DATE and the GAP only. The speakable time comes from `get_time`, which reads the same
local clock at the moment he answers.

CI-safe: strings and pure functions only. No model, no network, no clock waits.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shell.config import (
    DEFAULT_INSTRUCTIONS,
    SHELL_NOTES,
    TIME_NOTE,
    TOOLS_NOTE,
    instructions_without_auto_learn,
    persona_instructions,
)

REPO = Path(__file__).resolve().parent.parent

# The claim that started it: the stamp as "the current time". Re-adding it re-opens the
# bluff, so the exact wording is pinned, not paraphrased.
STAMP_IS_CLOCK_CLAIM = "and the current time - use it"

PROMPT_FILES = tuple(sorted((REPO / "envs").glob("*.yaml"))) + tuple(sorted((REPO / "personas").glob("*.yaml")))


def test_the_stamp_is_not_called_the_current_time_anymore():
    assert STAMP_IS_CLOCK_CLAIM not in TIME_NOTE
    assert "is not the time now" in TIME_NOTE


def test_a_spoken_time_is_routed_to_the_tool():
    """The half that makes the pair agree: the note names the tool, in the sentence
    where it used to name the stamp."""
    assert "never state a time you only read off a stamp" in TIME_NOTE
    assert "When anyone asks the time, call get_time" in TIME_NOTE


def test_the_tool_result_is_repeated_verbatim_not_re_derived():
    """The other edge of the recorded UTC bug (app/robot_tools.py:431): a model handed
    an instant re-derives a date, and every evening after 20:00 EDT it derived
    tomorrow's. get_time already answers in local `date`/`time` fields, so the note
    sends him to those words, not to arithmetic over an offset."""
    assert "say the date and time it returns exactly as it" in TIME_NOTE
    assert "already the local clock, read just now" in TIME_NOTE
    for iso_shape in ("UTC", "iso", "timezone", "convert"):
        assert iso_shape not in TIME_NOTE  # nothing here invites him to read an offset


def test_the_stamp_keeps_the_two_jobs_it_is_good_at():
    """Date and freshness — the 2026-10-05 and 2026-10-10/11 fixes. Dropping either to
    fix the time question would trade one recorded regression for another."""
    assert "today's date" in TIME_NOTE and "never guess the date" in TIME_NOTE
    assert "how long ago the guest spoke" in TIME_NOTE
    assert "treat it as a fresh conversation" in TIME_NOTE  # the dangling-question fix
    assert "Never read the stamps aloud." in TIME_NOTE  # unchanged: still not a script


def test_the_note_still_asks_for_no_time_guessing_from_the_model_either():
    """The contradiction had a second edge: 'use it' next to 'never read it aloud'. The
    replacement must not read as permission to speak the stamp, nor as permission to
    invent."""
    assert "is how you know today's date" in TIME_NOTE  # scoped to the date, not the minute
    assert "guess the time" not in TIME_NOTE  # the ban is the tool call, not a hedge


def test_tools_note_and_time_note_point_at_the_same_tool():
    """TOOLS_NOTE keeps "asks for the time" in its list, now unambiguously: the
    qualifier there is "you are unsure of", and TIME_NOTE is what tells him a stamp
    never makes him sure of the minute. Both notes name the same tool."""
    assert "asks for the time" in TOOLS_NOTE
    assert "get_time" in TIME_NOTE
    joined = persona_instructions(DEFAULT_INSTRUCTIONS)
    assert joined.count("get_time") == 1  # one instruction, said once
    assert joined.endswith(SHELL_NOTES)  # appending still works byte for byte


def test_a_booth_prompt_cannot_carry_a_baked_copy_of_the_old_claim():
    """The documented verify-then-paste flow (deploy/README.md) hands an operator the
    RESOLVED instructions, and an inline `instructions:` block in envs/<env>.yaml
    OUTRANKS the code default. A stale paste would keep the contradiction alive in the
    one room where it was measured, with every test here still green."""
    if not PROMPT_FILES:  # a hardware-free checkout with no env files: nothing to scan
        pytest.skip("no envs/ or personas/ files in this checkout")
    for path in PROMPT_FILES:
        text = path.read_text(encoding="utf-8")
        assert STAMP_IS_CLOCK_CLAIM not in text, f"{path.relative_to(REPO)} carries a stale prompt paste"
    env_file = REPO / ".env"
    if env_file.exists():  # gitignored, so this only bites on the deployment box
        assert STAMP_IS_CLOCK_CLAIM not in env_file.read_text(encoding="utf-8")


def test_the_booth_room_resolves_with_the_fixed_wording():
    """The rehearsal resolves the room through `python -m shell --print-env`; this is
    the same string it sends as the system prompt, minus the note the booth strips
    (shell/config.py LEARN_FACTS_NOTE). Import inside so a checkout without pyyaml
    fails this one test, not the module."""
    pytest.importorskip("yaml")
    from shell.env import load_config

    instructions = load_config("hackday").instructions
    assert instructions_without_auto_learn(SHELL_NOTES)[-40:] in instructions
    assert STAMP_IS_CLOCK_CLAIM not in instructions
    assert "When anyone asks the time, call get_time" in instructions
