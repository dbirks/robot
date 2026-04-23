import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DB_PATH = Path("data/sessions.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_call_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(content, content=turns, content_rowid=id);
CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


class SessionStore:
    def __init__(self, session_id: str | None = None):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.session_id = session_id or f"s_{int(time.time())}"
        log.info("Session store ready: %s", self.session_id)

    def log_turn(
        self, role: str, content: str | None = None, tool_calls: list | None = None, tool_call_id: str | None = None
    ):
        tc_json = json.dumps(tool_calls) if tool_calls else None
        self.db.execute(
            "INSERT INTO turns (session_id, timestamp, role, content, tool_calls, tool_call_id) VALUES (?, ?, ?, ?, ?, ?)",
            (self.session_id, time.time(), role, content, tc_json, tool_call_id),
        )
        self.db.commit()

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT t.session_id, t.timestamp, t.role, t.content FROM turns t JOIN turns_fts f ON t.id = f.rowid WHERE turns_fts MATCH ? ORDER BY t.timestamp DESC LIMIT ?",
            (query, limit),
        ).fetchall()
        return [{"session_id": r[0], "timestamp": r[1], "role": r[2], "content": r[3]} for r in rows]

    def get_session_history(self, session_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sid = session_id or self.session_id
        rows = self.db.execute(
            "SELECT timestamp, role, content, tool_calls FROM turns WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (sid, limit),
        ).fetchall()
        return [{"timestamp": r[0], "role": r[1], "content": r[2], "tool_calls": r[3]} for r in reversed(rows)]

    def close(self):
        self.db.close()
