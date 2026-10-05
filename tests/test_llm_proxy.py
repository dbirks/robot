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
    assert second[0][:11] != second[1][:11]  # new turn is stamped later
    assert second[1].endswith("] What do you think?")


def test_repeated_text_gets_its_own_time():
    from shell.llm_proxy import Timestamper

    clock = [1_800_000_000.0]
    st = Timestamper(clock=lambda: clock[0])
    a = st.stamp("Okay.", 0)
    clock[0] += 3600
    b = st.stamp("Okay.", 1)
    assert a != b and st.stamp("Okay.", 0) == a
