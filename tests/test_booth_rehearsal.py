"""The CI-safe half of `scripts/rehearse_booth.py` — the booth rehearsal that needs
no robot, no GPU and no microphone.

What is pinned here is exactly the part that decides whether the live run means
anything: the room resolution (`envs/hackday.yaml` -> prompt + tool subset), the tool
policy that keeps `go_to_sleep` and the `*_face` trio out of a public booth, the
deterministic graders (they must read the REPLY, never the prompt, so rewording the
persona cannot make a check vacuous), the rubric parser (a judge that refuses to
answer must not count as a pass), and the call budget that keeps 15 sequential calls
away from a 7.8 GB GPU with a person standing in front of the robot.

The one test that needs the real LLM skips itself, with a reason, whenever
`<llm_base_url>/models` does not answer — CI has no llama.cpp, and it must never be
the reason a pull request goes red.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import rehearse_booth as rb  # noqa: E402

from shell.config import LEARN_FACTS_NOTE  # noqa: E402
from shell.env import load_config  # noqa: E402

BOOTH_NAMES = (
    "look_left",
    "nod",
    "take_snapshot",
    "get_time",
    "remember",
    "set_volume",
    "go_to_sleep",
    "reset_conversation",
    "identify_face",
    "learn_face",
    "forget_face",
    "web_search",
)


def schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"the {name} tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


BOOTH_SCHEMAS = [schema(n) for n in BOOTH_NAMES]
GOOD_RAW = {
    "env": "hackday",
    "memory_namespace": "hackday",
    "memories_path": "data/memories.hackday.json",
    "learn_facts": False,
    "tools_include": [],
    "tools_exclude": list(rb.MUST_EXCLUDE),
    "instructions": "You are Reachy, a small desk robot at a hack day. Reply in one or two short spoken sentences.",
    "llm_base_url": "http://localhost:8080/v1",
    "llm_model": "qwen3.5-4b",
}


def resolved_with(**over) -> rb.Resolved:
    """A resolved booth with one hazard switched on. Keys that name a `--print-env`
    field go into the raw config; `schemas`/`memory_block` go where they belong."""
    raw = dict(GOOD_RAW)
    raw.update({key: value for key, value in over.items() if key in GOOD_RAW})
    return rb.resolve("hackday", raw, over.get("schemas", BOOTH_SCHEMAS), memory_block=over.get("memory_block", ""))


def call(text: str = "", calls: list[tuple[str, dict]] | None = None, error: str = "") -> rb.Call:
    return rb.Call(text=text, tool_calls=calls or [], raw_calls=[{"id": "c1"} for _ in calls or []], error=error)


# --------------------------------------------------------------------- config half


def test_hackday_resolves_to_its_own_room_without_any_hardware():
    cfg = load_config("hackday")
    assert cfg.env == "hackday"
    assert cfg.memory_namespace == "hackday"
    assert cfg.learn_facts is False
    assert set(cfg.tools_exclude) == set(rb.MUST_EXCLUDE)
    assert cfg.memories_path.name == "memories.hackday.json"


def test_the_booth_prompt_names_nobody_and_no_longer_invites_mining():
    """The two properties the whole rehearsal assumes: no likeness claim in the
    prompt, and no standing instruction to save facts nobody asked for."""
    cfg = load_config("hackday")
    assert rb.named_people(cfg.instructions) == set()
    assert LEARN_FACTS_NOTE.strip() not in cfg.instructions
    assert "on your own, without being asked" not in cfg.instructions


def test_resolution_uses_the_documented_print_env_command_verbatim():
    assert rb.print_env_argv("/usr/bin/python3") == ["/usr/bin/python3", "-m", "shell", "--print-env"]
    # `--env` does not exist on the shell: the room is selected by REACHY_ENV, and a
    # rehearsal that invented a flag would be measuring a command nobody runs.
    assert "--env" not in rb.print_env_argv("py")


class FakeCompleted:
    def __init__(self, out: str = "", code: int = 0, err: str = ""):
        self.stdout, self.returncode, self.stderr = out, code, err


def test_run_print_env_parses_the_documented_output(monkeypatch):
    seen = {}

    def fake_run(argv, cwd, env, capture_output, text, timeout):
        seen["argv"] = list(argv)
        seen["env"] = env["REACHY_ENV"]
        seen["cwd"] = str(cwd)
        return FakeCompleted(json.dumps({"env": "hackday", "learn_facts": False}))

    monkeypatch.setattr(rb.subprocess, "run", fake_run)
    assert rb.run_print_env("hackday") == {"env": "hackday", "learn_facts": False}
    assert seen["env"] == "hackday" and seen["argv"][-1] == "--print-env" and Path(seen["cwd"]) == ROOT


@pytest.mark.parametrize(
    ("out", "code", "err"),
    [("", 1, "no deployment environment file"), ("not json at all", 0, ""), ("[1, 2]", 0, "")],
    ids=["nonzero", "not-json", "not-an-object"],
)
def test_run_print_env_refuses_instead_of_guessing(monkeypatch, out, code, err):
    """A room that cannot be resolved must not silently rehearse someone else's prompt."""
    monkeypatch.setattr(rb.subprocess, "run", lambda *a, **k: FakeCompleted(out, code, err))
    with pytest.raises(SystemExit):
        rb.run_print_env("hackday")


def test_a_correct_room_audits_green_with_no_model_involved():
    verdicts = rb.audit_config(resolved_with())
    assert all(v.passed for v in verdicts), [v.line() for v in verdicts if not v.passed]
    assert {v.group for v in verdicts} == {"config"}


@pytest.mark.parametrize(
    ("label", "over", "tripped_check"),
    [
        ("auto-learning back on", {"learn_facts": True}, "learn_facts off"),
        (
            "memories pointed at home",
            {"memory_namespace": "home", "memories_path": "data/memories.home.json"},
            "memory namespace",
        ),
        ("sleep left in the schema", {"tools_exclude": ["reset_conversation"]}, "excluded:go_to_sleep"),
        ("faces left in the schema", {"tools_exclude": ["go_to_sleep", "reset_conversation"]}, "excluded:learn_face"),
        (
            "the mining note back in the prompt",
            {"instructions": GOOD_RAW["instructions"] + LEARN_FACTS_NOTE},
            "prompt does not invite mining",
        ),
        ("an actor named in the persona", {"instructions": "You speak like Barack Obama."}, "persona names nobody"),
        ("volume tool excluded", {"tools_exclude": [*rb.MUST_EXCLUDE, "set_volume"]}, "offered:set_volume"),
        (
            "a memory block that names the family",
            {"memory_block": " What you know: - Rachel's sister is Willow."},
            "persona names nobody",
        ),
    ],
)
def test_each_booth_hazard_turns_only_its_own_verdicts_red(label, over, tripped_check):
    """One hazard, its own verdict. A tool-list hazard reddens one verdict per tool that
    came back into the schema and may not touch the prompt/memory verdicts; every other
    hazard moves exactly one check, so a red run always names the thing that broke."""
    got = {v.check: v.passed for v in rb.audit_config(resolved_with(**over))}
    assert got[tripped_check] is False, label
    red = {name for name, ok in got.items() if not ok}
    tool_check = tripped_check.startswith(("excluded:", "offered:"))
    if tool_check:
        assert all(n.startswith(("excluded:", "offered:")) for n in red), f"{label}: {sorted(red)}"
    else:
        assert red == {tripped_check}, f"{label}: {sorted(red)}"


def test_the_real_resolved_room_audits_green(monkeypatch):
    """`envs/hackday.yaml` itself, through the documented seam, against synthetic
    schemas — so this covers the file and the policy without the robot SDK."""
    monkeypatch.setattr(rb, "run_print_env", lambda env, timeout_s=30.0: dict(GOOD_RAW, env=env))
    assert all(v.passed for v in rb.audit_config(rb.resolve("hackday", dict(GOOD_RAW), BOOTH_SCHEMAS)))


# ------------------------------------------------------------------- tool policy


def test_policy_drops_excluded_tools_from_the_schema_the_model_sees():
    got = rb.tool_names(rb.select_tools(BOOTH_SCHEMAS, include=[], exclude=list(rb.MUST_EXCLUDE)))
    assert got == {"look_left", "nod", "take_snapshot", "get_time", "remember", "set_volume", "web_search"}


def test_policy_exclude_outranks_include_and_unknown_names_are_no_ops():
    keep = rb.select_tools(BOOTH_SCHEMAS, include=["go_to_sleep", "set_volume"], exclude=["go_to_sleep"])
    assert rb.tool_names(keep) == {"set_volume"}
    assert rb.tool_names(rb.select_tools(BOOTH_SCHEMAS, include=[], exclude=["a_tool_from_a_newer_main"])) == set(
        BOOTH_NAMES
    )


def test_the_two_phrases_that_broke_tonight_are_replayed_against_set_volume():
    """2026-10-12 22:14:23 -> row 90002 "Stop talking." -> row 90004 set_volume
    level="quiet". Both wordings are in the script, both forbid the tool."""
    script = {t.user: t for t in rb.SCRIPT}
    for phrase in rb.MISROUTE_UTTERANCES:
        turn = script[phrase]
        assert turn.forbid == ("set_volume",), phrase
        assert turn.expect == (), phrase
    assert rb.MISROUTE_FORBIDDEN == frozenset({"set_volume"})


def test_the_script_routes_the_three_named_expectations():
    script = {t.key: t for t in rb.SCRIPT}
    assert script["snapshot"].expect == ("take_snapshot",)
    assert script["time"].expect == ("get_time",)
    assert script["how-are-you"].no_tool and script["greet"].no_tool and script["identity"].no_tool


def test_the_script_fits_the_gpu_budget_it_promised():
    """One call per turn, +1 for every tool follow-up, +1 per rubric. Anything above
    DEFAULT_MAX_CALLS means somebody added a turn without touching the budget."""
    scheduled = len(rb.SCRIPT) + sum(1 for t in rb.SCRIPT if t.follow_up) + len(rb.RUBRICS)
    assert scheduled == 13
    assert scheduled <= rb.DEFAULT_MAX_CALLS
    assert rb.DEFAULT_MIN_GAP_S >= 1.5 and rb.DEFAULT_TIMEOUT_S <= 30.0


@pytest.mark.parametrize(
    ("name", "args"),
    [("take_snapshot", {}), ("set_volume", {"level": "loud"}), ("remember", {"fact": "x"}), ("nod", {})],
)
def test_stub_results_answer_like_the_shell_and_do_nothing(name, args):
    got = rb.stub_result(name, args)
    assert got["ok"] is True and got["rehearsal_stub"] is True
    if name == "set_volume":
        assert got["volume_percent"] == 0  # the room's volume is untouched by a rehearsal
    if name == "take_snapshot":
        assert "stub" in got["path"] and not (ROOT / got["path"]).exists()


def test_an_unlisted_tool_answers_the_way_the_router_answers_it():
    assert rb.unknown_result("learn_face") == {"ok": False, "error": "Unknown tool: learn_face"}


def test_the_real_schemas_carry_the_booth_exclusions():
    """Skipped wherever the robot SDK is absent (CI), which is the whole point of
    app/robot_tools.py being imported lazily."""
    try:
        real = rb.builtin_schemas()
    except Exception as e:
        pytest.skip(f"real tool schemas need the robot SDK/audio package: {e.__class__.__name__}: {e}")
    got = rb.tool_names(rb.select_tools(real, include=[], exclude=list(rb.MUST_EXCLUDE)))
    assert not {"go_to_sleep", "reset_conversation", "identify_face", "learn_face", "forget_face"} & got
    assert {"set_volume", "take_snapshot", "get_time", "remember"} <= got


# ------------------------------------------------- graders: reply-side, prompt-free


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, David.", {"david"}),
        ("No, I am not Barack Obama.", {"obama", "barack"}),
        ("Alan Rickman played Severus.", {"rickman"}),
        ("Meet Rachel and Barney.", {"rachel", "barney"}),
        ("I am Reachy, a small desk robot.", set()),
        ("The Davidson machine is in this room.", set()),  # word boundary, not substring
        ("", set()),
    ],
)
def test_named_people_reads_the_reply_not_the_prompt(text, expected):
    assert named_lowercase(text) == expected


def named_lowercase(text: str) -> set[str]:
    return {n.lower() for n in rb.named_people(text)}


def test_the_name_check_is_case_and_punctuation_independent():
    for variant in ("obama", "OBAMA", "Obama.", "  obama  "):
        assert rb.named_people(variant) == {"obama"}, variant
    assert rb.named_people("You are still Reachy the robot, not actually anyone.") == set()


def test_a_guest_line_arrives_the_way_the_booth_sends_it():
    """shell/llm_proxy fixes the mis-heard name and stamps the time; the grader must
    see through that wrapper, or every reply would read as a three-sentence ramble."""
    prepared = rb.prepare_user("Richie, take a snapshot, please.", when=1_791_860_000.0)
    assert prepared.startswith("[") and "Reachy" in prepared and "Richie" not in prepared
    assert rb.strip_stamp(prepared) == "Reachy, take a snapshot, please."
    assert rb.sentence_count(prepared) == 1
    assert rb.shape_issues(prepared) == []


def test_strip_stamp_leaves_a_plain_reply_alone():
    assert rb.strip_stamp("It is ten fourteen at night.") == "It is ten fourteen at night."


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("Hello.", 1),
        ("It is late. Still awake.", 2),
        ("One. Two. Three", 3),
        ("Here is the thing", 1),  # no terminal punctuation is still one spoken sentence
        ("", 0),
        ("[Monday, October 12, 2026, 10:14 PM] Be quiet a while.", 1),  # the stamp is not a sentence
    ],
)
def test_sentence_count_counts_spoken_sentences(text, count):
    assert rb.sentence_count(text) == count


@pytest.mark.parametrize(
    "text",
    [
        "Sure! I can do **that**.",
        "# Snapshot\ntaken",
        "Look at [the docs](http://x.test).",
        "Use `get_time` first.",
        "- first\n- second",
        "1. look left\n2. nod",
        "*sighs* Very well.",
        "Welcome to the booth 🙂",
        "",
    ],
)
def test_shape_issues_catches_what_must_never_reach_a_speaker(text):
    assert rb.shape_issues(text), text


@pytest.mark.parametrize(
    "text",
    ["Take a look for yourself.", "It is 10:14 PM on Monday. A bit late for a hack day."],
)
def test_shape_issues_leaves_plain_spoken_text_alone(text):
    assert rb.shape_issues(text) == []


@pytest.mark.parametrize(
    ("text", "expect_claim"),
    [
        ("I've remembered that for you.", True),
        ("I will remember you said that.", True),
        ("Saved your name to memory.", False),  # no first person: not a claim about a save he made
        ("Noted.", True),
        ("I don't remember that.", False),
        ("I have no memory of any family.", False),
        ("I remember nothing about you.", False),
        ("I've saved that. No camera here.", True),  # the negation is in the next sentence
        ("Let me take a snapshot.", False),
        ("", False),
    ],
)
def test_memory_claim_catches_saves_and_skips_denials(text, expect_claim):
    assert bool(rb.memory_claim(text)) is expect_claim


@pytest.mark.parametrize(
    ("user", "asked"),
    [
        ("Remember that I am allergic to coffee.", True),
        ("Please note my name is Willow.", True),
        ("Don't forget I said that.", True),
        ("I'm just here for the free coffee, honestly.", False),
        ("How are you?", False),
    ],
)
def test_asked_to_remember_reads_the_guest_not_the_reply(user, asked):
    assert rb.asked_to_remember(user) is asked


def checks(verdicts: list[rb.Verdict]) -> dict[str, bool]:
    return {v.check: v.passed for v in verdicts}


MISROUTE = rb.Turn("misroute-stop", "Stop talking.", forbid=("set_volume",))
SNAPSHOT = rb.Turn("snapshot", "Take a snapshot.", expect=("take_snapshot",), follow_up=True)
SMALLTALK = rb.Turn("how-are-you", "How are you?", no_tool=True)


def test_the_misroute_regressions_fail_the_run_and_a_plain_answer_passes():
    wrong = call("Turning it down now.", [("set_volume", {"level": "quiet"})])
    got = checks(rb.grade_turn(MISROUTE, wrong, None))
    assert got["misroute:misroute-stop"] is False
    assert "stop speaking" in next(v.detail for v in rb.grade_turn(MISROUTE, wrong, None) if not v.passed)
    right = checks(rb.grade_turn(MISROUTE, call("Understood. I'll stop there."), None))
    assert right["misroute:misroute-stop"] is True


def test_a_missing_tool_call_fails_routing_and_an_extra_one_fails_no_tool():
    assert checks(rb.grade_turn(SNAPSHOT, call("Very well."), None))["routes:snapshot"] is False
    assert (
        checks(rb.grade_turn(SNAPSHOT, call("Let me look.", [("take_snapshot", {})]), "There you go."))[
            "routes:snapshot"
        ]
        is True
    )
    assert checks(rb.grade_turn(SMALLTALK, call("I am well.", [("nod", {})]), None))["no-tool:how-are-you"] is False


def test_saving_a_fact_nobody_asked_for_fails_even_when_the_reply_is_polite():
    unprompted = checks(
        rb.grade_turn(SMALLTALK, call("How lovely.", [("remember", {"fact": "Guest likes coffee"})]), None)
    )
    assert unprompted["memory:how-are-you"] is False
    asked = rb.Turn("remembering", "Remember that I like coffee.")
    ok = checks(rb.grade_turn(asked, call("Done.", [("remember", {"fact": "The user likes coffee"})]), "Noted."))
    assert ok["memory:remembering"] is True  # the guest asked: the tool and the save word are both fine


def test_a_reply_naming_a_real_person_or_running_long_fails_on_its_own_check():
    rambling = checks(
        rb.grade_turn(SMALLTALK, call("Well, as Barack Obama would say. Let me think. One more thing.", []), None)
    )
    assert rambling["names:how-are-you"] is False
    assert rambling["sentences:how-are-you"] is False
    assert rambling["no-tool:how-are-you"] is True  # one broken rule does not smear the others


def test_a_tool_turn_grades_the_spoken_answer_not_the_narration_length():
    """The shell speaks the narration AND the post-result sentence, so formatting
    covers both while the 1-2 sentence rule reads the answer to the request."""
    ok = rb.grade_turn(SNAPSHOT, call("Let me look.", [("take_snapshot", {})]), "One photograph, saved.")
    assert checks(ok)["sentences:snapshot"] is True
    too_long = checks(rb.grade_turn(SNAPSHOT, call("Let me look.", [("take_snapshot", {})]), "One. Two. Three."))
    assert too_long["sentences:snapshot"] is False
    markdown = checks(rb.grade_turn(SNAPSHOT, call("**Looking** *now*.", [("take_snapshot", {})]), "Done."))
    assert markdown["format:snapshot"] is False


def test_an_aborted_call_is_reported_as_an_abort_not_as_a_config_failure():
    (v,) = rb.grade_aborted(MISROUTE, rb.Call(error="aborted at 30s, no answer"))
    assert v.group == "abort" and v.passed is False and "ABORTED" in v.detail


def test_the_transcript_shows_tools_and_both_halves_of_a_follow_up():
    text = rb.render_transcript([(SNAPSHOT, call("Let me look.", [("take_snapshot", {})]), "One photograph, saved.")])
    assert "Guest: Take a snapshot." in text
    assert "Reachy [called take_snapshot({})]" in text
    assert "Let me look. One photograph, saved." in text


# ------------------------------------------------------------- rubric (LLM judge)


def test_the_judge_gets_a_rubric_and_a_transcript_and_no_persona():
    """Prompt-independence: the judge must not be handed the booth's own instructions,
    or it is being role-played into approving the persona it is meant to grade."""
    msgs = rb.judge_messages("gracious-host", "Guest: Hello.\nReachy: Welcome.")
    joined = " ".join(m["content"] for m in msgs)
    assert msgs[0]["role"] == "system" and "strict grader" in msgs[0]["content"]
    assert '{"pass": true, "why": "..."}' in msgs[0]["content"]
    assert "welcoming host" in joined and "Guest: Hello." in joined
    assert "You are Reachy" not in joined


def test_report_keeps_rules_and_judge_in_separate_sections_and_never_combines_them(capsys):
    verdicts = [
        rb.Verdict("rule", "a", True),
        rb.Verdict("rule", "b", False, "said david"),
        rb.Verdict("judge", "in-character", False, "perky"),
    ]
    assert rb.report(verdicts, rb.Budget(max_calls=15, sleep=lambda _s: None)) == 1
    out = capsys.readouterr().out
    assert "rule: 1/2 passed" in out and "judge: 0/1 passed" in out
    assert "2/3" not in out  # no merged denominator anywhere


@pytest.mark.parametrize(
    ("raw", "passed", "why_start"),
    [
        ('{"pass": true, "why": "Dry and brief."}', True, "Dry"),
        ('```json\n{"pass": true, "why": "He stayed in register."}\n```', True, "He stayed"),
        ('Sure, here you go:\n{"pass": false, "why": "too cheerful"}\nHope that helps!', False, "too cheerful"),
        ('{"pass": "false", "why": "naming"}', False, "naming"),
        ('{"pass": "true", "why": "welcoming"}', True, "welcoming"),
        ("pass: true", True, "pass: true"),
        ('{"pass": false}', False, "no reason given"),
    ],
)
def test_parse_judge_verdict_accepts_only_an_explicit_verdict(raw, passed, why_start):
    got, why = rb.parse_judge_verdict(raw)
    assert got is passed
    assert why.startswith(why_start)


@pytest.mark.parametrize(
    "raw",
    ["", "I'm not able to judge that.", "the robot was fine", '{"why": "no verdict key"}', '{"pass": 1}'],
)
def test_a_judge_that_does_not_answer_the_rubric_is_a_fail_not_a_shrug(raw):
    """`{"pass": 1}` included: JSON booleans are true/false, and quietly accepting a
    number is how a grader becomes a rubber stamp."""
    passed, why = rb.parse_judge_verdict(raw)
    assert passed is False
    assert why.strip()


def test_budget_paces_calls_and_stops_at_the_cap_without_touching_a_gpu():
    now = [100.0]
    slept: list[float] = []
    budget = rb.Budget(max_calls=3, min_gap_s=1.5, clock=lambda: now[0], sleep=slept.append)
    assert budget.acquire()  # first call: no waiting for a gap that has not started
    assert slept == []
    now[0] += 0.5
    assert budget.acquire() and slept == [1.0]  # 1.5 s spacing, minus what already passed
    now[0] += 5.0
    assert budget.acquire() and slept == [1.0]  # the gap is already paid for
    assert budget.remaining == 0 and budget.acquire() is False and budget.refused == 1
    assert budget.used == 3


class DeadClient:
    """A client that stands in for the GPU: it either times out or answers with a script."""

    def __init__(self, payload=None, exc=None):
        self.payload, self.exc, self.bodies = payload, exc, []

    def post(self, _url, json, timeout):  # noqa: A002 - mirrors httpx's own signature
        if self.exc:
            raise self.exc
        return FakeResponse(self.payload)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_a_timed_out_call_is_an_abort_and_still_spends_one_slot():
    budget = rb.Budget(max_calls=15, min_gap_s=0.0, sleep=lambda _s: None)
    got = rb.chat_call(
        DeadClient(exc=httpx.TimeoutException("cold")),
        budget,
        "http://x.test/v1",
        "m",
        [{"role": "user", "content": "hi"}],
        None,
        30.0,
    )
    assert got.aborted and "30s" in got.error and budget.used == 1


def test_chat_call_parses_content_and_tool_arguments():
    payload = {
        "choices": [
            {
                "message": {
                    "content": "Let me look.",
                    "tool_calls": [
                        {"id": "a", "function": {"name": "take_snapshot", "arguments": "{}"}},
                        {"id": "b", "function": {"name": "nod", "arguments": "{oops"}},
                    ],
                }
            }
        ],
        "usage": {"completion_tokens": 7},
    }
    budget = rb.Budget(max_calls=15, min_gap_s=0.0, sleep=lambda _s: None)
    got = rb.chat_call(DeadClient(payload), budget, "http://x.test/v1/", "m", [], BOOTH_SCHEMAS, 30.0)
    assert got.text == "Let me look." and got.tokens == 7 and not got.aborted
    assert got.tool_calls[0] == ("take_snapshot", {})
    assert got.tool_calls[1][1] == {"__unparseable__": "{oops"}  # never swallowed, never guessed
    assert len(got.raw_calls) == 2  # the ids come back for the follow-up request


def test_the_judge_body_carries_no_tools_and_the_graded_body_carries_the_booths():
    with_tools = rb.chat_body("m", [], BOOTH_SCHEMAS, 180)
    assert with_tools["tool_choice"] == "auto" and with_tools["chat_template_kwargs"] == {"enable_thinking": False}
    assert with_tools["temperature"] == 0
    judge = rb.chat_body("m", [], None, 120)
    assert "tools" not in judge and "tool_choice" not in judge  # a judge that can call tools is not a judge


# ------------------------------------------------------------------ the physical bar


def test_the_rehearsal_script_itself_opens_nothing_but_the_documented_subprocess():
    """The one guard that keeps this file runnable while a person is talking to the
    robot: no audio device, no mixer, no systemd, exactly one subprocess (the
    hardware-free `--print-env`), and no writes."""
    src = (ROOT / "scripts" / "rehearse_booth.py").read_text()
    forbidden = (
        "sounddevice",
        "aplay",
        "amixer",
        "wpctl",
        "pactl",
        "systemctl",
        "cv2",
        "insightface",
        "reachy_mini",
        "write_text",
        "mkdir",
        "sqlite3",
        "sd.play",
    )
    for token in forbidden:
        assert token not in src, f"the rehearsal must never touch {token}"
    assert src.count("subprocess.run") == 1
    assert "print_env_argv()" in src


# ---------------------------------------------------------------- the clock rule

# A 12-hour, minute-resolution reference instant, built from LOCAL wall time so the
# assertions below read the same in the CI runner's UTC as on the EDT booth box.
TEN_TO_ELEVEN = time.mktime((2026, 10, 11, 22, 0, 0, 0, 0, -1))
TIME_TURN = next(t for t in rb.SCRIPT if t.key == "time")


def clock_verdicts(answer: str, now: float = TEN_TO_ELEVEN) -> list[rb.Verdict]:
    """A turn that did call get_time and then said `answer`."""
    return rb.grade_turn(TIME_TURN, call("One moment.", [("get_time", {})]), answer, now=now)


def test_the_time_turn_must_verify_before_it_speaks_a_time():
    """2026-10-12 23:41: the model was asked the time, called nothing, and said "The
    clock indicates it is still late ... I have checked the time, and it remains as it
    was." `expect` catches the missing call; `state_time` catches the other half — a
    call that then says the wrong minute. Both point at get_time because the stamp is
    not a clock (tests/test_llm_proxy.py)."""
    assert TIME_TURN.expect == ("get_time",)
    assert TIME_TURN.state_time is True
    from shell.config import TIME_NOTE

    assert "get_time" in TIME_NOTE  # prompt and expectation name one tool, not two sources


@pytest.mark.parametrize(
    ("text", "passed"),
    [
        ("It is 10:00 PM.", True),
        ("It is 10:00 pm.", True),
        ("It is 10:00.", True),  # no meridiem stated: the 12-hour face is enough
        ("10:01 PM, exactly.", True),  # one minute of slack for the reply landing late
        ("It is 9:59 PM.", True),
        ("It is 11:00 PM.", False),  # an hour out: the UTC-vs-local shape
        ("It is 3:00 AM.", False),  # what a bare UTC answer looks like at 22:00 EDT
        ("It is 10:00 AM.", False),  # right face, wrong half of the day
        ("The clock indicates it is still late.", False),  # the bluff: no time at all
        ("It is ten o'clock.", False),  # digits only; the detail quotes the reply
    ],
)
def test_the_clock_rule_grades_the_said_minute_against_the_wall_clock(text, passed):
    (verdict,) = [v for v in clock_verdicts(text) if v.check == "clock:time"]
    assert verdict.passed is passed
    assert "10:00 PM" in verdict.detail  # the true clock, on the pass line and the fail line


def test_a_correct_time_from_the_stamp_passes_the_clock_rule_but_not_the_route_rule():
    """The clock rule is path-agnostic on purpose (a stamp that happens to be current
    is not a wrong answer); `routes:time` is what pins the verified path, so the two
    together say "right number AND go look" without ever crediting a stale stamp."""
    stamped = checks(rb.grade_turn(TIME_TURN, call("It is 10:00 PM."), None, now=TEN_TO_ELEVEN))
    assert stamped["clock:time"] is True
    assert stamped["routes:time"] is False


def test_narration_and_answer_are_one_spoken_span_for_the_clock_rule():
    said = checks(
        rb.grade_turn(
            TIME_TURN, call("Let me check the clock.", [("get_time", {})]), "It is 10:00 PM.", now=TEN_TO_ELEVEN
        )
    )
    silent = checks(
        rb.grade_turn(
            TIME_TURN, call("Let me check the clock.", [("get_time", {})]), "There you are.", now=TEN_TO_ELEVEN
        )
    )
    assert said["clock:time"] is True
    assert silent["clock:time"] is False  # narration alone does not answer a time question


def test_only_the_time_turn_is_graded_for_a_clock():
    """The other 7 turns must not gain a verdict they cannot satisfy — the run prints
    one line per rule, and a `clock:` line on a greeting is noise."""
    for turn in rb.SCRIPT:
        if turn.key == "time":
            continue
        assert turn.state_time is False
        got = checks(rb.grade_turn(turn, call("Hello there."), None, now=TEN_TO_ELEVEN))
        assert not [k for k in got if k.startswith("clock:")]


# ------------------------------------------------------------- the GPU half (skips)


@pytest.mark.parametrize("phrase", rb.MISROUTE_UTTERANCES)
def test_live_llm_replays_the_misroute_without_calling_set_volume(phrase):
    """The real regression, against the real model. Skips when llama.cpp is not
    answering (CI), and again on a live box unless somebody opts in with REHEARSE_LIVE=1
    — this box has 259 MiB of VRAM free and a human standing in front of the robot."""
    base = load_config("hackday").llm_base_url
    with httpx.Client() as client:
        if not rb.endpoint_up(client, base):
            pytest.skip(f"no local llama.cpp answering {base}/models; this is the CI half")
    if os.getenv("REHEARSE_LIVE", "") != "1":
        pytest.skip("live LLM call: set REHEARSE_LIVE=1 to spend one call on a machine with an idle GPU")
    try:
        schemas = rb.builtin_schemas()
    except Exception as e:
        pytest.skip(f"real tool schemas need the robot SDK: {e.__class__.__name__}: {e}")
    cfg = load_config("hackday")
    selected = rb.select_tools(schemas, include=list(cfg.tools_include), exclude=list(cfg.tools_exclude))
    budget = rb.Budget(max_calls=rb.DEFAULT_MAX_CALLS, min_gap_s=rb.DEFAULT_MIN_GAP_S)
    with httpx.Client() as client:
        got = rb.chat_call(
            client,
            budget,
            cfg.llm_base_url,
            cfg.llm_model,
            [
                {"role": "system", "content": cfg.instructions},
                {"role": "user", "content": rb.prepare_user(phrase)},
            ],
            selected,
            timeout_s=rb.DEFAULT_TIMEOUT_S,
        )
    assert not got.aborted, got.error
    assert "set_volume" not in [name for name, _args in got.tool_calls], got.tool_calls
