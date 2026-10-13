"""What the proxy between s2s and llama.cpp may do to a request - and what it may not.

Two rewrites, and nothing else: the mis-heard name in user text, and s2s's
voice-channel wrapper on the system prompt. Everything else passes through byte for
byte, which is what keeps llama.cpp's prompt-prefix cache hitting; the shape tests
below are the guard on that claim.

The per-message time stamp is gone (2026-10-13). It had two independent failures, both
about the prefix itself: the model started working the stamps into its own spoken
replies, and the cache behind them keyed on the message TEXT for the life of the proxy
process, so a repeated line arrived wearing an OLD time (issue #45, measured 24.8 and
41.5 minutes stale across session resets). `
test_a_repeated_line_arrives_with_no_time_at_all` is the regression that keeps #45 from
coming back as a "fixed" stamp.
"""

import json
import re

from shell.llm_proxy import fix_name, rewrite_body


def test_fix_name_variants():
    assert fix_name("Hey Richie, are you there?") == "Hey Reachy, are you there?"
    assert fix_name("Ricci what's up") == "Reachy what's up"
    assert fix_name("hey rachi") == "hey Reachy"
    assert fix_name("I'm rich, enriching") == "I'm rich, enriching"  # word-bounded


def test_rewrites_only_user_text_in_responses_input():
    body = {
        "instructions": "You are Reachy.",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hey Richie"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Richie here"}]},
            {"role": "user", "content": "Ritchie, look left"},
            {"type": "function_call_output", "output": "Richie"},
        ],
    }
    out = rewrite_body(body)["input"]
    assert out[0]["content"][0]["text"] == "Hey Reachy"
    assert out[1]["content"][0]["text"] == "Richie here"
    assert out[2]["content"] == "Reachy, look left"
    assert out[3]["output"] == "Richie"


def test_chat_completions_shape_and_string_input():
    assert rewrite_body({"input": "Ricci?"})["input"] == "Reachy?"
    body = rewrite_body({"messages": [{"role": "user", "content": "Richie!"}]})
    assert body["messages"][0]["content"] == "Reachy!"


def test_s2s_voice_wrapper_is_stripped_to_our_instructions():
    wrapped = (
        "You are in a spoken conversation. The user speaks and hears you.\n"
        "The session prompt defines persona...\n\n"
        "Session Prompt:\nYou are Reachy. Use tools.\n\n"
        "## Voice Rules\n- For expression/background tools, speak first.\n- If unsure, just speak."
    )
    body = rewrite_body(
        {
            "input": [
                {"type": "message", "role": "system", "content": [{"type": "input_text", "text": wrapped}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            ],
            "instructions": wrapped,
        }
    )
    assert body["input"][0]["content"][0]["text"] == "You are Reachy. Use tools."
    assert body["instructions"] == "You are Reachy. Use tools."
    other = rewrite_body({"messages": [{"role": "system", "content": "Summarize this."}]})
    assert other["messages"][0]["content"] == "Summarize this."  # not the s2s wrapper: untouched


# ---------------------------------------------------------------- no time on a message


MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
A_TIME_SHAPE = re.compile(rf"\d{{1,2}}:\d{{2}}|\b(?:AM|PM|UTC|GMT)\b|\b(?:{MONTHS})\b", re.IGNORECASE)


def test_the_stamp_and_everything_it_needed_are_gone():
    """Nothing named `stamp`/`Timestamper`/`format_stamp` survives on the live path.

    Half of #45's fix is that there is no cache to go stale: the proxy keeps no state
    per message at all, so a repeat cannot arrive wearing an old reading.
    """
    import shell.llm_proxy as proxy

    for name in ("Timestamper", "format_stamp", "STAMP_RE", "_stamper"):
        assert not hasattr(proxy, name), f"{name} is back: the proxy would stamp again"
    with open(proxy.__file__, encoding="utf-8") as f:
        src = f.read()
    below_docstring = src.split('"""', 2)[2].lower()
    assert "stamp" not in below_docstring, "code below the docstring still mentions stamping"


def test_a_repeated_line_arrives_with_no_time_at_all():
    """The behaviour #45 left behind.

    s2s re-sends the whole history on every request, and a fresh `realtime.session_reset`
    starts at occurrence 0 again, which is what used to make the failure: the old
    `Timestamper` handed a re-sent line the time recorded for its TEXT months-of-process
    earlier, so the newest stamp in a request was not necessarily recent (journal:
    "hey richie." re-asked verbatim 24.8 and 41.5 minutes apart across resets,
    2026-10-11 14:19 -> 15:00). What remains, and what must stay true, is that the proxy
    is a pure function of the request: the same text re-sent in a later session comes
    back identical and carries no clock reading of any kind, stale or fresh.
    """
    first = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "What time is it?"}]}]}
    again = json.loads(json.dumps(first))  # a fresh session holding the same one line

    got = rewrite_body(first)["input"][0]["content"][0]["text"]
    later = rewrite_body(again)["input"][0]["content"][0]["text"]

    assert got == "What time is it?"  # not stamped, not paraphrased, not quoted
    assert later == got  # and re-sending it changes nothing
    assert not A_TIME_SHAPE.search(got), got
    assert got.startswith("W") and "[" not in got  # no prefix at all, so nothing to read aloud


def test_no_message_in_the_history_can_carry_a_stale_reading():
    """The multi-turn case: history grows, old turns re-send, and no line picks up a
    time from anywhere. This is the property `TIME_NOTE` now relies on - the model has
    no clock to be lied to about, so every time question has to reach the tool."""
    body = {
        "instructions": "You are Reachy.",
        "input": [
            {"role": "user", "content": "Hey Richie, are you awake?"},
            {"role": "assistant", "content": "Barely."},
            {"role": "user", "content": [{"type": "input_text", "text": "What time is it?"}]},
            {"type": "function_call_output", "output": '{"date": "Monday, October 12, 2026", "time": "11:09 PM"}'},
        ],
    }
    out = rewrite_body(body)
    assert [i["role"] for i in out["input"][:3]] == ["user", "assistant", "user"]
    assert out["input"][0]["content"] == "Hey Reachy, are you awake?"
    assert out["input"][2]["content"][0]["text"] == "What time is it?"
    # the tool result is the one place a time legitimately appears, and it is his to say
    assert out["input"][3]["output"] == body["input"][3]["output"]


# ------------------------------------------------------------------ byte stability


def captured_body() -> dict:
    """A request in the shape s2s actually sends (the keys `data/llm_last_request.json`
    on the booth box carries, kept inline so CI never has to read a gitignored dump)."""
    return {
        "input": [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "You are Reachy, a small desk robot."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Er Richie, can you go to sleep now?"}],
            },
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Certainly."}]},
            {"type": "function_call", "name": "go_to_sleep", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": '{"ok": true}'},
        ],
        "model": "qwen3.5-4b",
        "reasoning": {"effort": "low"},
        "stream": True,
        "tools": [{"type": "function", "function": {"name": "go_to_sleep", "description": "d", "parameters": {}}}],
    }


def test_the_only_change_is_the_name_in_user_text():
    """The contract the proxy is allowed to keep: same keys in the same order, same
    values everywhere except user text. A stamp was a violation of exactly this."""
    raw = captured_body()
    out = rewrite_body(json.loads(json.dumps(raw)))

    assert list(out) == list(raw)  # no key added, dropped, or reordered
    assert out["model"] == raw["model"] and out["stream"] == raw["stream"] and out["tools"] == raw["tools"]
    assert out["input"][0] == raw["input"][0]  # system text: unwrapped only, and it is not wrapped
    assert out["input"][2:] == raw["input"][2:]  # assistant, tool call, tool result: untouched
    assert out["input"][1]["content"][0]["text"] == "Er Reachy, can you go to sleep now?"
    # every other user turn re-sends byte for byte, which is what the prefix cache needs
    assert json.dumps(out["input"][1:]) == json.dumps(rewrite_body(json.loads(json.dumps(out)))["input"][1:])


def test_a_request_body_is_stable_across_re_sends():
    """Rewriting an already-rewritten request is a no-op: the prompt prefix for old
    turns cannot drift between calls, which is the property the stamp used to be
    carefully (and expensively) engineered to provide."""
    once = rewrite_body(captured_body())
    twice = rewrite_body(json.loads(json.dumps(once)))
    assert json.dumps(once, sort_keys=False) == json.dumps(twice, sort_keys=False)


def test_unwrap_system_leaves_other_system_prompts_alone():
    from shell.llm_proxy import unwrap_system

    assert unwrap_system("Summarize this.") == "Summarize this."
