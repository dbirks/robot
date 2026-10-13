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
