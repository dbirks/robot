"""The prompt/tool pair for "what time is it?" (2026-10-12 booth rehearsal, closed
2026-10-13 by taking the clock out of the prompt).

What used to be wrong was a system prompt that disagreed with itself in front of a
crowd: TIME_NOTE called the per-message stamp "the current time" and ended with
"Never read the stamps aloud", while TOOLS_NOTE said to call a tool for the time.
Qwen 3.5 4B resolved that by doing neither half — no `get_time` call, and "It is still
late ... I have checked the time, and it remains as it was." (verbatim, 23:41 run of
`scripts/rehearse_booth.py`): a time asserted, never verified.

The settled answer is a denial rather than a rule. `shell/llm_proxy.py` stamps nothing
any more (the cache keyed on message TEXT could hand a re-sent line a stamp 24.8 / 41.5
minutes old — issue #45 — and the model was reading them aloud), so the note can simply
tell the truth: he has no clock, and a time or date question is a `get_time` call. The
tests below pin the denial, the single path, and the fact that no prompt in the repo
still claims a message carries a time.

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

# Wording that would re-open the contradiction, pinned exactly rather than paraphrased.
# Each one either claimed a stamp was a clock, or referred to a stamp that no longer
# exists anywhere in the stack.
STAMP_CLAIMS: tuple[str, ...] = (
    "Each user message begins",  # the old opening of TIME_NOTE
    "and the current time - use it",  # the claim that started the bluff
    "read off a stamp",
    "Never read the stamps aloud",  # a rule about a feature that is gone
)

PROMPT_FILES = tuple(sorted((REPO / "envs").glob("*.yaml"))) + tuple(sorted((REPO / "personas").glob("*.yaml")))


def test_the_note_says_plainly_that_he_has_no_clock():
    assert "You have no clock" in TIME_NOTE
    assert "nothing in what you are shown tells you" in TIME_NOTE
    for claim in STAMP_CLAIMS:
        assert claim not in TIME_NOTE
        assert claim not in SHELL_NOTES


def test_the_note_never_names_the_removed_feature():
    """`get_time` is now the only time word in the prompt; a note that still talked
    about stamps would send him looking for something the proxy no longer sends."""
    assert "stamp" not in TIME_NOTE.lower()
    assert "message begins" not in TIME_NOTE.lower()


def test_every_time_or_date_question_is_routed_to_the_tool():
    """Not just "the time": with no stamp the model cannot see the DATE either, so the
    note has to cover date and day as well or he guesses those, which is the very
    failure the stamp was originally added for (2026-10-10/11: "October 14, 2023",
    weather for "May 2025")."""
    for asks in ("the time", "the date", "the day of the week", "how late it"):
        assert asks in TIME_NOTE, asks
    assert "calling get_time" in TIME_NOTE
    assert "When anyone asks the time, call get_time" not in TIME_NOTE  # the stamp-era sentence


def test_the_tool_result_is_repeated_verbatim_not_re_derived():
    """The other edge of the recorded UTC bug (app/robot_tools.py:431): a model handed
    an instant re-derives a date, and every evening after 20:00 EDT it derived
    tomorrow's. get_time already answers in local `date`/`time` fields, so the note
    sends him to those words, not to arithmetic over an offset."""
    assert "saying the date and time it returns exactly as it returns them" in TIME_NOTE
    assert "calling get_time" in TIME_NOTE  # and saying it, not re-doing it
    assert "already the local clock, read just now" in TIME_NOTE
    for iso_shape in ("UTC", "iso", "timezone", "convert", "offset"):
        assert iso_shape not in TIME_NOTE  # nothing here invites him to read an offset


def test_guessing_is_closed_off_from_both_ends():
    """The 4B may not invent a time, and may not recycle one from earlier in the
    conversation - the shape #45 produced when a stale stamp made "still late" sound
    defensible."""
    assert "Never guess" in TIME_NOTE and "or a date" in TIME_NOTE
    assert "already" in TIME_NOTE and "stale" in TIME_NOTE  # an earlier time is expired
    assert "how long ago" not in TIME_NOTE  # no stamp, so no gap he could measure


def test_the_newest_message_is_still_the_one_he_answers():
    """The dangling-question half (2026-10-05, a 40-minute-old question answered late)
    survives as an instruction; the MEASURING half of it moved to the shell, which
    compacts after `compact_after_s` and resets the session after `reset_after_s` of
    silence (shell/conversation.py)."""
    assert "Answer only the newest message" in TIME_NOTE
    assert "do not bring back old, unanswered topics" in TIME_NOTE


def test_tools_note_and_time_note_point_at_the_same_tool():
    """TOOLS_NOTE keeps "asks for the time" in its list, now unambiguously: TIME_NOTE
    is what tells him nothing in the transcript makes him sure of the minute. Both
    notes name the same tool, once."""
    assert "asks for the time" in TOOLS_NOTE
    assert "get_time" in TIME_NOTE
    joined = persona_instructions(DEFAULT_INSTRUCTIONS)
    assert joined.count("get_time") == 1  # one instruction, said once
    assert joined.endswith(SHELL_NOTES)  # appending still works byte for byte


def test_the_tool_itself_is_worded_as_the_only_path():
    """With no stamp there is nothing to compete with, so the description can say it:
    one way to learn a time, and it is this call. Schema text only - the handler's
    local rendering is pinned where it lives."""
    tools = pytest.importorskip("app.robot_tools").TOOLS
    got = {t["function"]["name"]: t["function"]["description"] for t in tools}
    desc = got["get_time"]
    assert "ONLY way to learn" in desc
    assert "no clock" in desc
    assert "the date" in desc and "day of the week" in desc
    assert "stamp" not in desc.lower()  # no rule left pointing at the removed feature


def test_no_prompt_in_the_repo_carries_a_baked_copy_of_the_stamp_claims():
    """The documented verify-then-paste flow (deploy/README.md) hands an operator the
    RESOLVED instructions, and an inline `instructions:` block in envs/<env>.yaml
    OUTRANKS the code default. A stale paste would keep the contradiction alive in the
    one room where it was measured, with every test here still green."""
    if not PROMPT_FILES:  # a hardware-free checkout with no env files: nothing to scan
        pytest.skip("no envs/ or personas/ files in this checkout")
    for path in PROMPT_FILES:
        text = path.read_text(encoding="utf-8")
        for claim in STAMP_CLAIMS:
            assert claim not in text, f"{path.relative_to(REPO)} carries a stale prompt paste"
    env_file = REPO / ".env"
    if env_file.exists():  # gitignored, so this only bites on the deployment box
        assert not any(claim in env_file.read_text(encoding="utf-8") for claim in STAMP_CLAIMS)


def test_the_booth_room_resolves_with_the_fixed_wording():
    """The rehearsal resolves the room through `python -m shell --print-env`; this is
    the same string it sends as the system prompt, minus the note the booth strips
    (shell/config.py LEARN_FACTS_NOTE). Import inside so a checkout without pyyaml
    fails this one test, not the module."""
    pytest.importorskip("yaml")
    from shell.env import load_config

    instructions = load_config("hackday").instructions
    assert instructions_without_auto_learn(SHELL_NOTES)[-40:] in instructions
    assert "You have no clock" in instructions
    assert "calling get_time" in instructions
    assert "stamp" not in instructions.lower()  # the room cannot be told about a stamp
