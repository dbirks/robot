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


def test_timestamps_are_stable_across_resent_history():
    from shell.llm_proxy import Timestamper

    clock = [1_800_000_000.0]
    st = Timestamper(clock=lambda: clock[0])

    def req(*texts):
        body = {"input": [{"role": "user", "content": [{"type": "input_text", "text": t}]} for t in texts]}
        return [i["content"][0]["text"] for i in rewrite_body(body, st)["input"]]

    first = req("How bad is the economy?")
    clock[0] += 2 * 3600  # two hours later
    second = req("How bad is the economy?", "What do you think?")
    assert second[0] == first[0]  # old turn keeps its original stamp
    assert second[0].split("]")[0] != second[1].split("]")[0]  # new turn is stamped later
    assert second[1].endswith("] What do you think?")


def test_repeated_text_gets_its_own_time():
    from shell.llm_proxy import Timestamper

    clock = [1_800_000_000.0]
    st = Timestamper(clock=lambda: clock[0])
    a = st.stamp("Okay.", 0)
    clock[0] += 3600
    b = st.stamp("Okay.", 1)
    assert a != b and st.stamp("Okay.", 0) == a


def test_stamp_carries_the_full_date():
    import re
    import time

    from shell.llm_proxy import STAMP_RE, Timestamper

    t = time.mktime((2026, 10, 11, 11, 40, 0, 0, 0, -1))
    s = Timestamper(clock=lambda: t).stamp("what year is it?", 0)
    assert s == "[Sunday, October 11, 2026, 11:40 AM] what year is it?"
    assert STAMP_RE.match(s) and STAMP_RE.match("[Mon 17:38] old-style")
    assert STAMP_RE.match("[Sun 11 Oct 2026 11:40] previous-style")
    assert not STAMP_RE.match("[laughs] hello")
    assert Timestamper().stamp(s, 0) == s  # never double-stamped
    assert re.search(r"2026", s)


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


# ------------------------------------------------------------------ stamp authority
# The booth rehearsal (scripts/rehearse_booth.py, 2026-10-12) turned "What time is
# it?" into a disagreement between shell/config.py TIME_NOTE ("the newest [stamp] is
# ... the current time") and the get_time tool. These two tests are the evidence that
# settled it: the stamp is a LOCAL clock reading, which is what makes it good for the
# date, and it is cached per message TEXT, which is what disqualifies it as a clock.


def test_the_stamp_is_local_wall_time_and_never_an_iso_instant():
    """The recorded bug (app/robot_tools.py get_time): "a bare UTC ISO string made the
    model give tomorrow's date every evening after 20:00 EDT". The stamp must not
    repeat it — local, spelled out, no ISO/Z/offset for the model to reinterpret.
    """
    import time

    from shell.llm_proxy import format_stamp

    evening = time.mktime((2026, 10, 11, 21, 30, 0, 0, 0, -1))  # 9:30 PM local
    stamp = format_stamp(evening)
    lt = time.localtime(evening)
    assert f"{lt.tm_year}" in stamp and time.strftime("%A", lt) in stamp and time.strftime("%B", lt) in stamp
    assert " 9:30 PM]" in stamp  # 12-hour, unpadded hour, exactly what the note quotes
    for utc_shape in ("T", "Z", "+00:00", "UTC", "GMT"):
        assert utc_shape not in stamp
    if time.timezone:  # any zone but UTC: the same instant must read local, not Zulu
        gt = time.gmtime(evening)
        assert time.strftime("%A %B %Y", lt) != time.strftime("%A %B %Y", gt) or gt.tm_hour == lt.tm_hour
        assert f", {(gt.tm_hour % 12) or 12}:{gt.tm_min:02d} " not in stamp


def test_a_fresh_session_re_sends_a_repeated_line_with_its_old_stamp():
    """The stamp is per message TEXT and survives every session reset, so the newest
    stamp in a request is NOT guaranteed to be recent.

    `Timestamper` keys on text and hands occurrence k the k-th recorded time; the proxy
    is its own long-lived process (deploy/systemd/reachy-llm-proxy.service) and
    `ConversationMemory.on_wake` starts a FRESH s2s session (50 `realtime.session_reset`
    rows in data/events.db), so the new session's first "What time is it?" is
    occurrence 0 again — and comes back wearing the time of the old one. The journal
    shows the identical-line precondition over and over: "hey richie." re-asked 24.8 and
    41.5 minutes apart across resets (2026-10-11 14:19 -> 15:00).

    This is pinned as a KNOWN LIMIT, not an aspiration: it is why shell/config.py
    TIME_NOTE may tell him the date and the gap, but must never tell him a stamp is
    "the current time" he is allowed to say out loud.
    """
    import time

    from shell.llm_proxy import Timestamper

    clock = [time.mktime((2026, 10, 11, 22, 0, 0, 0, 0, -1))]  # 10:00 PM local
    st = Timestamper(clock=lambda: clock[0])

    def request(*texts: str) -> list[str]:
        body = {"input": [{"role": "user", "content": t} for t in texts]}
        return [m["content"] for m in rewrite_body(body, st)["input"]]

    asked = request("What time is it?")
    clock[0] += 40 * 60  # 10:40 PM: the room went quiet, the session reset
    after_reset = request("What time is it?")  # a FRESH session holds one occurrence
    assert after_reset == asked  # same stamp, 40 minutes stale
    assert " 10:00 PM]" in after_reset[0] and "10:40" not in after_reset[0]
    # ... while inside one session the history grows, and the second ask is stamped fresh
    in_session = request("What time is it?", "What time is it?")
    assert " 10:40 PM]" in in_session[1]
