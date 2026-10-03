"""Typed event journal (ADR 0007).

SQLite WAL, typed events with monotonic timestamps, retention-bounded.
Writing an event must never raise and never block meaningfully; on failure
the journal degrades to dropped events plus a stderr line, because losing a
telemetry row must never take down a conversation.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

# Event type vocabulary (superset grows, never renames - ADR 0007).
AUDIO_SPEECH_STARTED = "audio.speech_started"
AUDIO_SPEECH_STOPPED = "audio.speech_stopped"
AUDIO_DEVICE_ERROR = "audio.device_error"
AUDIO_HEALTH = "audio.health"

KWS_DETECTED = "kws.detected"

ATTENTION_DECISION = "attention.decision"
ATTENTION_LEASE_STARTED = "attention.lease_started"
ATTENTION_LEASE_RENEWED = "attention.lease_renewed"
ATTENTION_LEASE_EXPIRED = "attention.lease_expired"
ATTENTION_LEASE_TRANSFERRED = "attention.lease_transferred"
ATTENTION_PROFILE_CHANGED = "attention.profile_changed"
ATTENTION_TOOL_DENIED = "attention.tool_denied"

PARTICIPANT_RESOLVED = "participant.resolved"
PARTICIPANT_OBSERVATION = "participant.observation"

TURN_TRANSCRIPT_PARTIAL = "turn.transcript_partial"
TURN_TRANSCRIPT_FINAL = "turn.transcript_final"

RESPONSE_CREATED = "response.created"
RESPONSE_FIRST_AUDIO = "response.first_audio"
RESPONSE_CANCELLED = "response.cancelled"
RESPONSE_COMPLETED = "response.completed"

TOOL_STARTED = "tool.started"
TOOL_COMPLETED = "tool.completed"
TOOL_FAILED = "tool.failed"
TOOL_DENIED = "tool.denied"

TTS_CHUNK_QUEUED = "tts.chunk_queued"
TTS_FLUSHED = "tts.flushed"

MODEL_LOADED = "model.loaded"
SYSTEM_RESOURCE_SAMPLE = "system.resource_sample"


class Journal:
    def __init__(
        self,
        path: Path,
        max_rows: int = 200_000,
        max_age_s: float = 30 * 24 * 3600.0,
    ) -> None:
        self.path = Path(path)
        self.max_rows = max_rows
        self.max_age_s = max_age_s
        self.dropped = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            " id INTEGER PRIMARY KEY,"
            " mono REAL NOT NULL,"
            " wall REAL NOT NULL,"
            " type TEXT NOT NULL,"
            " payload TEXT NOT NULL)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS ix_events_type ON events(type)")
        self._db.commit()
        self._t0 = time.monotonic()
        self._n_since_prune = 0

    def write(self, type_: str, **payload) -> None:
        try:
            self._db.execute(
                "INSERT INTO events(mono, wall, type, payload) VALUES(?,?,?,?)",
                (
                    time.monotonic() - self._t0,
                    time.time(),
                    type_,
                    json.dumps(payload, default=str),
                ),
            )
            self._n_since_prune += 1
            if self._n_since_prune >= 512:
                self._n_since_prune = 0
                self._prune()
        except Exception as e:  # telemetry must never break conversation
            self.dropped += 1
            print(f"journal: dropped {type_}: {e}", file=sys.stderr)

    def _prune(self) -> None:
        try:
            self._db.execute(
                "DELETE FROM events WHERE id < (SELECT MAX(id) FROM events) - ?",
                (self.max_rows,),
            )
            self._db.execute("DELETE FROM events WHERE wall < ?", (time.time() - self.max_age_s,))
            self._db.commit()
        except Exception:
            pass

    def recent(self, limit: int = 200, type_: str | None = None) -> list[dict]:
        q = "SELECT mono, type, payload FROM events"
        args: tuple = ()
        if type_:
            q += " WHERE type=?"
            args = (type_,)
        q += " ORDER BY id DESC LIMIT ?"
        rows = self._db.execute(q, args + (limit,)).fetchall()
        return [{"mono": m, "type": t, "payload": json.loads(p)} for m, t, p in reversed(rows)]

    def closed(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass
