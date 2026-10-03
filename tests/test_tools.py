from shell.tools import ToolRouter


def make_router(journal, handlers, **kw):
    return ToolRouter([], handlers, journal, **kw)


def test_unknown_tool_never_raises(journal):
    r = make_router(journal, {})
    out = r.execute("teleport", {})
    assert out["ok"] is False and "Unknown tool" in out["error"]


def test_handler_exception_becomes_error_dict(journal):
    def angry(**kw):
        raise ValueError("boom")

    out = make_router(journal, {"angry": angry}).execute("angry", {})
    assert out["ok"] is False and "boom" in out["error"]
    assert "tool.failed" in journal.types()


def test_dedupe_identical_calls_in_one_response(journal):
    calls = [
        ("look_left", {}),
        ("look_left", {}),
        ("look_left", {}),
        ("look_right", {}),
        ("nod", {"n": 1}),
        ("nod", {"n": 2}),
    ]
    assert ToolRouter.dedupe(calls) == [("look_left", {}), ("look_right", {}), ("nod", {"n": 1}), ("nod", {"n": 2})]


def test_fire_and_forget_skips_llm_followup(journal):
    r = make_router(journal, {"nod": lambda **k: {"ok": True}, "web_search": lambda **k: {"ok": True, "results": []}})
    assert r.needs_followup("nod", {"ok": True}) is False
    assert r.needs_followup("web_search", {"ok": True}) is True


def test_denied_tool_not_narrated(journal):
    r = make_router(journal, {})
    assert r.needs_followup("set_volume", {"ok": False, "denied": "x"}) is False


def test_state_changing_denied_low_confidence_is_journaled(journal):
    r = make_router(
        journal, {"go_to_sleep": lambda **k: {"ok": True}}, confidence_fn=lambda n: 0.2, confidence_threshold=0.6
    )
    out = r.execute("go_to_sleep", {})
    assert out["ok"] is False and out["denied"] == "attention_confidence_too_low"
    assert "attention.tool_denied" in journal.types()
    # the deny hint must stop the model retrying in a loop
    assert "Do not retry" in out["hint"]
