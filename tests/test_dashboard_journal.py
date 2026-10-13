"""Journal read path for the attendee dashboard (issue #38).

Stdlib + shell.journal only — dashboard/journal_read.py must stay importable
under `uvx --with numpy --with pyyaml --with pytest pytest tests/`, where
fastapi and sse_starlette are not installed.
"""

from __future__ import annotations

import ast
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from dashboard import journal_read as jr
from shell.journal import Journal


def seed(tmp_path: Path, rows: list[tuple[str, dict]], env: str | None = None) -> Path:
    """Write with the real shell helper (same Journal the shell uses), then close."""
    db = tmp_path / "events.db"
    j = Journal(db, env=env)
    for type_, payload in rows:
        j.write(type_, **payload)
    j.closed()
    return db


# --- the guard that keeps CI green ------------------------------------------------


def test_journal_read_imports_only_stdlib():
    src = Path(jr.__file__).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            root = node.names[0].name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
        else:
            continue
        assert root in sys.stdlib_module_names, f"{root} is not stdlib"


# --- collapse_finals --------------------------------------------------------------


def test_partial_run_collapses_into_the_final(tmp_path):
    db = seed(
        tmp_path,
        [
            (jr.TURN_TRANSCRIPT_PARTIAL, {"text": "hey"}),
            (jr.TURN_TRANSCRIPT_PARTIAL, {"text": "hey reachy"}),
            (jr.TURN_TRANSCRIPT_FINAL, {"text": "hey reachy"}),
            (jr.RESPONSE_TRANSCRIPT_FINAL, {"text": "yes?"}),
        ],
    )
    with jr.open_ro(db) as conn:
        kept = jr.latest_events(conn, limit=50)
    assert [e["type"] for e in jr.collapse_finals(kept)] == [
        jr.TURN_TRANSCRIPT_FINAL,
        jr.RESPONSE_TRANSCRIPT_FINAL,
    ]


def test_duplicate_and_blank_finals_drop_unrelated_partials_stay(tmp_path):
    db = seed(
        tmp_path,
        [
            (jr.TURN_TRANSCRIPT_PARTIAL, {"text": "completely"}),
            (jr.TURN_TRANSCRIPT_FINAL, {"text": "open the drawer"}),
            (jr.TURN_TRANSCRIPT_FINAL, {"text": "  open the   drawer  "}),
            (jr.TURN_TRANSCRIPT_FINAL, {"text": ""}),
        ],
    )
    with jr.open_ro(db) as conn:
        kept = jr.latest_events(conn, limit=50)
    collapsed = jr.collapse_finals(kept)
    assert [e["type"] for e in collapsed] == [
        jr.TURN_TRANSCRIPT_PARTIAL,
        jr.TURN_TRANSCRIPT_FINAL,
    ]
    assert len([e for e in collapsed if e["type"] == jr.TURN_TRANSCRIPT_FINAL]) == 1


def test_unfinished_partial_run_is_kept(tmp_path):
    db = seed(tmp_path, [(jr.TURN_TRANSCRIPT_PARTIAL, {"text": "trailing"})])
    with jr.open_ro(db) as conn:
        kept = jr.latest_events(conn, limit=50)
    assert jr.collapse_finals(kept) == kept


# --- turn assembly ---------------------------------------------------------------


def test_turns_pair_user_and_assistant_with_tools(tmp_path):
    db = seed(
        tmp_path,
        [
            (jr.TURN_TRANSCRIPT_PARTIAL, {"text": "how warm"}),
            (jr.TURN_TRANSCRIPT_FINAL, {"text": "how warm is it"}),
            (jr.TOOL_STARTED, {"tool": "read_temp", "args": {}}),
            (jr.TOOL_COMPLETED, {"tool": "read_temp", "result": {"ok": True, "temp": "61C"}}),
            (jr.RESPONSE_TRANSCRIPT_FINAL, {"text": "Sixty one.", "response_id": "r1"}),
            (jr.TURN_TRANSCRIPT_FINAL, {"text": "thanks"}),
            (jr.RESPONSE_TRANSCRIPT_FINAL, {"text": "", "response_id": "r2"}),
        ],
    )
    with jr.open_ro(db) as conn:
        turns = jr.turns_from_events(jr.latest_events(conn, limit=100))
    assert [(t["role"], t["content"]) for t in turns] == [
        ("user", "how warm is it"),
        ("assistant", "Sixty one."),
        ("user", "thanks"),
    ]
    assert turns[0]["timestamp"] <= turns[1]["timestamp"] <= turns[2]["timestamp"]
    calls = turns[1]["tool_calls"]
    assert calls[0]["function"]["name"] == "read_temp"
    assert '"ok": true' in calls[0]["_result"]
    assert set(turns[0]) == {"id", "timestamp", "role", "content", "tool_calls"}


def test_tool_turn_without_spoken_text_still_renders(tmp_path):
    db = seed(
        tmp_path,
        [
            (jr.TURN_TRANSCRIPT_FINAL, {"text": "nod"}),
            (jr.TOOL_STARTED, {"tool": "nod", "args": {}}),
            (jr.TOOL_FAILED, {"tool": "nod", "error": "Robot not connected"}),
        ],
    )
    with jr.open_ro(db) as conn:
        turns = jr.turns_from_events(jr.latest_events(conn, limit=100))
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[1]["content"] == ""
    assert turns[1]["tool_calls"][0]["_result"] == "Robot not connected"


def test_assistant_transcript_row_round_trips(tmp_path):
    """The proof issue #38 asks for: the helper the shell writes with lands his
    reply in events.db and it comes back out as an assistant turn."""
    db = seed(tmp_path, [(jr.RESPONSE_TRANSCRIPT_FINAL, {"text": "Hello from the journal."})])
    with jr.open_ro(db) as conn:
        rows = jr.latest_events(conn, limit=10, type_=jr.RESPONSE_TRANSCRIPT_FINAL)
        turns = jr.turns_from_events(jr.latest_events(conn, limit=10))
    assert rows[0]["payload"]["text"] == "Hello from the journal."
    assert turns == [
        {
            "id": 1,
            "timestamp": rows[0]["wall"],
            "role": "assistant",
            "content": "Hello from the journal.",
            "tool_calls": None,
        }
    ]


# --- env badge ------------------------------------------------------------------


def test_env_badge_from_the_stamp(tmp_path):
    db = seed(
        tmp_path,
        [(jr.TURN_TRANSCRIPT_FINAL, {"text": "hi"}), (jr.RESPONSE_TRANSCRIPT_FINAL, {"text": "yo"})],
        env="hackday",
    )
    with jr.open_ro(db) as conn:
        assert jr.env_badge(jr.latest_events(conn, limit=50)) == "hackday"


def test_env_badge_none_when_unstamped(tmp_path):
    db = seed(tmp_path, [(jr.TURN_TRANSCRIPT_FINAL, {"text": "hi"})])
    with jr.open_ro(db) as conn:
        assert jr.env_badge(jr.latest_events(conn, limit=50)) is None


# --- liveness ------------------------------------------------------------------


def test_liveness_reports_last_wall_and_health(tmp_path):
    db = seed(
        tmp_path,
        [
            (jr.AUDIO_HEALTH, {"device": "xvf3800", "dropped": 3}),
            (jr.RESPONSE_TRANSCRIPT_FINAL, {"text": "hi"}),
        ],
    )
    with jr.open_ro(db) as conn:
        live = jr.liveness(conn, now=time.time() + 1.0)
    assert live["alive"] is True and live["age_s"] <= 2.0
    assert live["audio_health"]["device"] == "xvf3800"
    assert live["audio_health"]["dropped"] == 3
    assert live["last_wall"] == pytest.approx(time.time(), abs=5.0)


def test_liveness_goes_stale(tmp_path):
    db = seed(tmp_path, [(jr.TURN_TRANSCRIPT_FINAL, {"text": "hi"})])
    with jr.open_ro(db) as conn:
        assert jr.liveness(conn, now=time.time() + 4 * jr.STALE_AFTER_S)["alive"] is False
        assert jr.liveness(conn, now=time.time() + 4 * jr.STALE_AFTER_S)["audio_health"] is None


def test_liveness_empty_journal(tmp_path):
    db = seed(tmp_path, [])
    with jr.open_ro(db) as conn:
        live = jr.liveness(conn)
    assert live == {"last_wall": None, "age_s": 0.0, "alive": False, "audio_health": None}


# --- events_after / latest_events ----------------------------------------------


def test_events_after_is_an_ascending_cursor(tmp_path):
    db = seed(
        tmp_path,
        [(jr.TURN_TRANSCRIPT_FINAL, {"text": str(i)}) for i in range(5)],
    )
    with jr.open_ro(db) as conn:
        first = jr.events_after(conn, 0, limit=2)
        rest = jr.events_after(conn, first[-1]["id"], limit=10)
        past = jr.events_after(conn, 9999)
    assert [e["payload"]["text"] for e in first] == ["0", "1"]
    assert [e["payload"]["text"] for e in rest] == ["2", "3", "4"]
    assert past == []  # cursor at the end of the journal


def test_events_after_filters_types(tmp_path):
    db = seed(
        tmp_path,
        [
            (jr.TURN_TRANSCRIPT_FINAL, {"text": "hi"}),
            (jr.AUDIO_HEALTH, {"device": "default"}),
        ],
    )
    with jr.open_ro(db) as conn:
        only = jr.events_after(conn, 0, types=(jr.AUDIO_HEALTH,))
    assert [e["type"] for e in only] == [jr.AUDIO_HEALTH]


# --- the read-only decision table ---------------------------------------------


@pytest.mark.parametrize(
    ("method", "allow", "rejected"),
    [
        ("GET", (), False),
        ("get", (), False),
        ("HEAD", (), False),
        ("OPTIONS", (), False),
        ("POST", (), True),
        ("post", (), True),
        ("PUT", (), True),
        ("PATCH", (), True),
        ("DELETE", (), True),
        ("", (), True),
        ("POST", ("/services/",), False),
        ("POST", ("/faces/",), True),
    ],
)
def test_readonly_reject_table(method, allow, rejected):
    verdict = jr.readonly_reject(method, "/services/reachy-s2s/restart", allow=allow)
    if rejected:
        assert verdict is not None
        assert verdict["status"] == 405
        assert verdict["body"]["method"] == method.upper()
    else:
        assert verdict is None


# --- open_ro is actually read-only --------------------------------------------


def test_open_ro_rejects_writes(tmp_path):
    db = seed(tmp_path, [(jr.TURN_TRANSCRIPT_FINAL, {"text": "hi"})])
    with jr.open_ro(db) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO events(mono, wall, type, payload) VALUES(0,0,'x','{}')")


def test_open_ro_missing_file_raises(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        jr.open_ro(tmp_path / "nope.db")


# --- traces (issue #39) -----------------------------------------------------------
#
# `events.mono` is CLOCK_MONOTONIC for the *writing process* (Journal._t0), so the
# only honest way to build a turn is inside one process's run of that clock. These
# tests hand-write mono values (and, below, really restart the Journal twice) because
# a mono regression cannot be produced by one long-lived writer.


def ev(id_: int, mono: float, type_: str, **payload) -> dict:
    return {"id": id_, "mono": mono, "wall": 1_791_800_000.0 + mono, "type": type_, "payload": payload}


ROOT = Path(__file__).resolve().parent.parent


def test_two_journal_writes_are_two_segments(tmp_path):
    """The regression is the restart: seed two synthetic processes in one file."""
    db = tmp_path / "events.db"
    first = Journal(db)
    first.write(jr.TURN_TRANSCRIPT_FINAL, text="hello there")
    first.closed()
    second = Journal(db)  # a new process, so a new clock: mono starts near zero again
    second.write(jr.RESPONSE_CREATED, response_id="resp_2")
    second.write(jr.RESPONSE_COMPLETED, response_id="resp_2", status="completed")
    second.closed()

    with jr.open_ro(db) as conn:
        events = jr.latest_events(conn, limit=50, types=jr.TRACE_TYPES)
        segments = jr.split_segments(events)
        traces = jr.build_traces(events)

    assert len(segments) == 2, [e["mono"] for e in events]
    assert len(traces) == 1
    turn = traces[0]
    assert turn["segment"] == 0
    # The response rows belong to the *next* process. Attributing them here would be
    # the arithmetic that produced a 4183 s p50, so this turn stays incomplete.
    assert turn["incomplete"] is True
    assert turn["total_ms"] is None
    assert turn["durations_ms"]["llm"] is None
    assert turn["outcome"] is None


def test_spans_attribute_to_their_own_turn():
    events = [
        ev(1, 10.0, jr.AUDIO_SPEECH_STARTED, cut_playback=False),
        ev(2, 11.5, jr.AUDIO_SPEECH_STOPPED),
        ev(3, 12.0, jr.TURN_TRANSCRIPT_FINAL, text="what time is it"),
        ev(4, 12.4, jr.RESPONSE_CREATED, response_id="resp_a"),
        ev(5, 12.9, jr.RESPONSE_FIRST_AUDIO, response_id="resp_a"),
        ev(6, 14.0, jr.RESPONSE_COMPLETED, response_id="resp_a", status="completed"),
        ev(7, 20.0, jr.AUDIO_SPEECH_STARTED, cut_playback=False),
        ev(8, 21.0, jr.AUDIO_SPEECH_STOPPED),
        ev(9, 21.5, jr.TURN_TRANSCRIPT_FINAL, text="remember my face"),
        ev(10, 21.6, jr.RESPONSE_CREATED, response_id="resp_b"),
        ev(11, 21.7, jr.TOOL_STARTED, tool="learn_face", args={"name": "Sam"}),
        ev(12, 23.2, jr.TOOL_COMPLETED, tool="learn_face", result={"ok": True}),
        ev(13, 24.0, jr.RESPONSE_COMPLETED, response_id="resp_b", status="completed"),
    ]
    a, b = jr.build_traces(events)

    assert [s["kind"] for s in a["spans"] if s["kind"] == "tool"] == []
    assert a["durations_ms"] == {"vad": 1500, "endpoint": 500, "llm": 400, "first_audio": 500, "total": 4000}
    assert a["incomplete"] is False and a["outcome"] == "done"

    tools = [s for s in b["spans"] if s["kind"] == "tool"]
    assert len(tools) == 1 and tools[0]["label"] == "tool:learn_face"
    assert tools[0]["dur_ms"] == 1500 and tools[0]["t0_ms"] == 1700  # from the start of the utterance
    assert "learn_face" not in tools[0]["detail"] and "ok" in tools[0]["detail"]
    assert b["durations_ms"]["total"] == 4000


def test_denied_tool_and_barge_in_are_recorded_where_they_happened():
    events = [
        ev(1, 0.0, jr.TURN_TRANSCRIPT_FINAL, text="remember that"),
        ev(2, 0.5, jr.RESPONSE_CREATED, response_id="resp_c"),
        ev(3, 0.6, jr.ATTENTION_TOOL_DENIED, tool="remember", confidence=0.0, threshold=0.5),
        ev(4, 1.0, jr.AUDIO_SPEECH_STARTED, cut_playback=True),
        ev(5, 2.0, jr.RESPONSE_CANCELLED, response_id="resp_c", reason="turn_detected"),
    ]
    (turn,) = jr.build_traces(events)
    denied = [s for s in turn["spans"] if s["kind"] == "denied"]
    assert len(denied) == 1
    assert denied[0]["label"] == "denied:remember" and denied[0]["dur_ms"] == 0
    assert denied[0]["t0_ms"] == 600
    assert turn["outcome"] == "cancelled" and turn["cancel_reason"] == "turn_detected"
    assert turn["interrupts"] == 1
    assert turn["durations_ms"]["total"] == 2000
    assert turn["incomplete"] is False


def test_turn_with_no_audio_rows_still_reports_what_it_can():
    """No vad rows: the endpoint span has nothing to measure, and says so."""
    (turn,) = jr.build_traces([ev(1, 5.0, jr.TURN_TRANSCRIPT_FINAL, text="hi")])
    endpoint = next(s for s in turn["spans"] if s["kind"] == "stt")
    assert endpoint["dur_ms"] == 0 and endpoint["t0_ms"] == 0
    assert "no vad rows" in endpoint["detail"]
    assert turn["incomplete"] is True and turn["durations_ms"]["total"] is None


def test_first_audio_only_binds_its_own_response():
    events = [
        ev(1, 0.0, jr.TURN_TRANSCRIPT_FINAL, text="go on"),
        ev(2, 1.0, jr.RESPONSE_CREATED, response_id="resp_mine"),
        ev(3, 1.2, jr.RESPONSE_FIRST_AUDIO, response_id="resp_other"),
        ev(4, 2.0, jr.RESPONSE_FIRST_AUDIO, response_id="resp_mine"),
        ev(5, 3.0, jr.RESPONSE_COMPLETED, response_id="resp_mine", status="completed"),
    ]
    (turn,) = jr.build_traces(events)
    assert turn["durations_ms"]["first_audio"] == 1000


def test_limit_keeps_the_newest_turns_in_order():
    events = [ev(i, float(i), jr.TURN_TRANSCRIPT_FINAL, text=f"t{i}") for i in range(1, 6)]
    traces = jr.build_traces(events, limit=2)
    assert [t["user_text"] for t in traces] == ["t4", "t5"]


def test_trace_window_reports_how_much_it_saw(tmp_path):
    db = seed(tmp_path, [(jr.TURN_TRANSCRIPT_FINAL, {"text": "one"})], env="hackday")
    with jr.open_ro(db) as conn:
        panel = jr.trace_window(conn, window=100, limit=5)
        assert panel["turn"] is None  # no deep link asked for one
        anchor = panel["traces"][0]["id"]
        found = jr.trace_window(conn, window=100, limit=5, turn_id=anchor)
        missing = jr.trace_window(conn, window=100, limit=5, turn_id=anchor + 500)
    assert panel["turns_total"] == 1 and panel["segments"] == 1 and panel["env"] == "hackday"
    assert panel["traces"][0]["user_text"] == "one"  # newest first
    assert panel["traces"][0]["incomplete"] is True
    assert found["turn"]["id"] == anchor
    assert missing["turn"] is None


# --- the guard again, for the code and the templates this issue added --------------


def test_traces_do_not_need_fastapi():
    """`build_traces` must survive the minimal CI env: no web imports in the reader."""
    tree = ast.parse(Path(jr.__file__).read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {n.name.split(".")[0] for n in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not roots & {"fastapi", "starlette", "sse_starlette", "jinja2", "uvicorn"}
    assert roots <= set(sys.stdlib_module_names), sorted(roots - set(sys.stdlib_module_names))


def test_trace_templates_reach_off_box_nothing():
    """Mirrors check_env_safety 9: the booth has no network, so no CDN either."""
    templates = sorted((ROOT / "dashboard" / "templates").rglob("*.html"))
    assert len(templates) >= 12, f"the glob is wrong, found {len(templates)}"
    for path in templates:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "http-equiv" in line:
                continue
            assert "http://" not in line and "https://" not in line and '="//' not in line, f"{path}:{lineno}"


def test_trace_page_renders_without_a_network_and_says_what_is_missing(tmp_path):
    """Renders the real templates: a Jinja typo on this page is a blank kiosk tomorrow.

    jinja2 arrives with fastapi, so the minimal CI env has neither and skips; the
    reader functions above are what CI is guarding.
    """
    pytest.importorskip("jinja2")
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(ROOT / "dashboard" / "templates")))
    env.filters["format_ts"] = lambda ts: time.strftime("%H:%M:%S", time.localtime(ts))
    empty = {"traces": [], "turns_total": 0, "segments": 1, "env": None, "traced_rows": 0, "turn": None}
    html = env.get_template("partials/traces.html").render(panel=empty)
    assert "The journal records no completed turns in this segment" in html
    assert "env unstamped" in html

    events = [
        ev(1, 0.0, jr.AUDIO_SPEECH_STARTED),
        ev(2, 1.0, jr.AUDIO_SPEECH_STOPPED),
        ev(3, 1.5, jr.TURN_TRANSCRIPT_FINAL, text="hello", env="hackday"),
        ev(4, 2.0, jr.RESPONSE_CREATED, response_id="r"),
        ev(5, 2.5, jr.RESPONSE_FIRST_AUDIO, response_id="r"),
        ev(6, 3.0, jr.RESPONSE_CANCELLED, response_id="r", reason="turn_detected"),
    ]
    (turn,) = jr.build_traces(events)
    panel = {"traces": [turn], "turns_total": 1, "segments": 1, "env": "hackday", "traced_rows": 6, "turn": turn}
    html = env.get_template("traces.html").render(panel=panel)
    assert "cancelled: turn_detected" in html and "1000 ms" in html and 'href="/traces/3"' in html
    assert "hackday" in html  # the env badge comes off the stamped row
    html = env.get_template("trace.html").render(panel=panel, turn=turn, unknown_id=3)
    assert "first audio" in html and "cancelled" in html and "1500" in html
    html = env.get_template("trace.html").render(panel=panel, turn=None, unknown_id=99)
    assert "is not in the traced window" in html
