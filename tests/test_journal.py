from pathlib import Path

from shell import journal as J
from shell.journal import Journal


def test_roundtrip_and_query(tmp_path: Path):
    j = Journal(tmp_path / "e.db")
    j.write(J.TURN_TRANSCRIPT_FINAL, text="hello there")
    j.write(J.ATTENTION_DECISION, action="ignore", evidence={"score": 0.1})
    rows = j.recent(10)
    assert [r["type"] for r in rows] == [J.TURN_TRANSCRIPT_FINAL, J.ATTENTION_DECISION]
    assert rows[0]["payload"]["text"] == "hello there"
    only = j.recent(10, type_=J.ATTENTION_DECISION)
    assert len(only) == 1 and only[0]["payload"]["action"] == "ignore"
    j.closed()


def test_write_never_raises_on_bad_payload(tmp_path: Path):
    j = Journal(tmp_path / "e.db")

    class Unserializable:
        def __str__(self):
            return "ok"

    j.write(J.TOOL_STARTED, tool="x", args={"weird": Unserializable()})
    assert j.recent(1)[0]["payload"]["args"]["weird"] == "ok"
    j.closed()


def test_retention_bounds_rows(tmp_path: Path):
    j = Journal(tmp_path / "e.db", max_rows=50)
    for i in range(700):  # prune triggers every 512 writes
        j.write(J.SYSTEM_RESOURCE_SAMPLE, i=i)
    rows = j.recent(10_000)
    assert len(rows) <= 50 + 512  # prune is amortized over 512 writes
    assert rows[-1]["payload"]["i"] == 699
    j.closed()


def test_writes_visible_to_other_connections_immediately(tmp_path):
    import sqlite3

    from shell.journal import Journal

    j = Journal(tmp_path / "events.db")
    j.write("audio.health", rms=1.0)
    other = sqlite3.connect(str(tmp_path / "events.db"))
    assert other.execute("SELECT type FROM events").fetchall() == [("audio.health",)]
