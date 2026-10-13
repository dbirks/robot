"""Read-only views over the live event journal (`data/events.db`).

Stdlib only, on purpose: CI runs `uvx --with numpy --with pyyaml --with pytest
pytest tests/`, so nothing a test imports may pull fastapi or sse_starlette.
`dashboard/server.py` imports this module, never the other way around.

The journal is append-only and owned by `shell/journal.py` (ADR 0007); this file
opens it in SQLite read-only mode and turns typed rows into the turn dicts the
conversation template already renders. It never writes, creates or prunes.
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.parse
from pathlib import Path
from typing import Any

# Event types we read, spelled once in shell/journal.py (imported for the
# constants only — that module is stdlib-only too). Names repeated here as
# literals so the dashboard can be vendored without the shell package.
TURN_TRANSCRIPT_PARTIAL = "turn.transcript_partial"
TURN_TRANSCRIPT_FINAL = "turn.transcript_final"
RESPONSE_TRANSCRIPT_FINAL = "response.transcript_final"
RESPONSE_CANCELLED = "response.cancelled"
TOOL_STARTED = "tool.started"
TOOL_COMPLETED = "tool.completed"
TOOL_FAILED = "tool.failed"
AUDIO_HEALTH = "audio.health"

# The rows a conversation is made of. `audio.health`, `robot.connection` and the
# rest of the watchdog traffic are 4 orders of magnitude more frequent than these,
# so a window that does not filter is almost always pure health (issue #38).
CONVERSATION_TYPES = (
    TURN_TRANSCRIPT_PARTIAL,
    TURN_TRANSCRIPT_FINAL,
    RESPONSE_TRANSCRIPT_FINAL,
    RESPONSE_CANCELLED,
    TOOL_STARTED,
    TOOL_COMPLETED,
    TOOL_FAILED,
)

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# The watchdog posts audio.health every 10 s (shell/__main__.py watchdog_sample),
# so three missed samples means the shell is gone, not merely mid-turn.
STALE_AFTER_S = 30.0

_EVENT_COLS = "id, mono, wall, type, payload"


def open_ro(path: Path | str) -> sqlite3.Connection:
    """Open the journal read-only. Raises if the file does not exist — that is
    a real answer for a booth dashboard, not something to paper over with an
    empty conversation."""
    p = Path(path)
    uri = "file:" + urllib.parse.quote(str(p.resolve() if p.exists() else p), safe="/~") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    try:
        payload = json.loads(row["payload"])
    except (ValueError, TypeError):
        payload = {}
    return {
        "id": row["id"],
        "mono": row["mono"],
        "wall": row["wall"],
        "type": row["type"],
        "payload": payload if isinstance(payload, dict) else {},
    }


def events_after(
    conn: sqlite3.Connection,
    after_id: int = 0,
    limit: int = 500,
    types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Ascending rows with id > after_id — the SSE cursor."""
    q = f"SELECT {_EVENT_COLS} FROM events WHERE id > ?"
    args: list[Any] = [after_id]
    if types:
        q += f" AND type IN ({','.join('?' * len(types))})"
        args.extend(types)
    q += " ORDER BY id ASC LIMIT ?"
    args.append(max(1, int(limit)))
    return [_row_to_event(r) for r in conn.execute(q, args)]


def latest_events(
    conn: sqlite3.Connection,
    limit: int = 200,
    type_: str | None = None,
    types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Newest `limit` rows, returned oldest-first (the order turn assembly wants).

    `types` is how a conversation window avoids drowning in watchdog samples;
    `type_` stays for the single-type callers (liveness).
    """
    wanted = tuple(types) if types else ((type_,) if type_ else ())
    q = f"SELECT {_EVENT_COLS} FROM events"
    args: list[Any] = []
    if wanted:
        q += f" WHERE type IN ({','.join('?' * len(wanted))})"
        args.extend(wanted)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(max(1, int(limit)))
    return [_row_to_event(r) for r in reversed(conn.execute(q, args).fetchall())]


def _norm(text: object) -> str:
    return " ".join(str(text or "").split()).lower()


def collapse_finals(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop transcript rows that a later row superseded.

    `turn.transcript_partial` is provisional: a partial whose text is a prefix of
    the next partial/final in the same run is dropped, and once the run's final
    arrives every surviving partial in it goes too (a partial that did NOT lead to
    that final is kept rather than silently eaten). Identical consecutive finals —
    the same utterance delivered twice — collapse to the first. Everything else
    passes through untouched, in order.
    """
    out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []  # partials of the run in flight
    last_final = ""
    for ev in events:
        t = ev["type"]
        if t == TURN_TRANSCRIPT_PARTIAL:
            text = _norm(ev["payload"].get("text"))
            if pending and text.startswith(_norm(pending[-1]["payload"].get("text"))):
                pending[-1] = ev  # a longer partial supersedes the shorter one
            else:
                for p in pending:
                    out.append(p)
                pending = [ev]
            continue

        if t == TURN_TRANSCRIPT_FINAL:
            text = _norm(ev["payload"].get("text"))
            for p in pending:
                if not text.startswith(_norm(p["payload"].get("text"))):
                    out.append(p)  # unrelated partial: keep it, don't hide data
            pending = []
            if not text or text == last_final:
                continue  # blank, or a duplicate of the final we already kept
            last_final = text
            out.append(ev)
            continue

        for p in pending:
            out.append(p)
        pending = []
        out.append(ev)

    out.extend(pending)  # run never finalized: show what he heard so far
    return out


def turns_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Journal rows -> the turn dicts partials/conversation.html renders.

    Keys match what the legacy `sessions.db` reader produced (id, timestamp, role,
    content, tool_calls) so the template and the SSE payload shape do not change.
    `timestamp` is the wall clock. Tool rows merge into the assistant turn that
    called them, with each call carrying `_result` like the legacy merge did.
    """
    turns: list[dict[str, Any]] = []
    next_id = 1

    def push(role: str, wall: float) -> dict[str, Any]:
        turn = {
            "id": next_id,
            "timestamp": wall,
            "role": role,
            "content": "",
            "tool_calls": None,
        }
        turns.append(turn)
        return turn

    def assistant(wall: float) -> dict[str, Any]:
        if turns and turns[-1]["role"] == "assistant":
            return turns[-1]
        return push("assistant", wall)

    for ev in collapse_finals(events):
        t, wall, payload = ev["type"], ev["wall"], ev["payload"]
        if t == TURN_TRANSCRIPT_FINAL:
            push("user", wall)["content"] = str(payload.get("text") or "")
        elif t == RESPONSE_TRANSCRIPT_FINAL:
            text = str(payload.get("text") or "").strip()
            if not text:
                continue  # barge-in cut it; nothing to render
            turn = assistant(wall)
            if turn["content"]:
                turn = push("assistant", wall)
            turn["content"] = text
        elif t in (TOOL_STARTED, TOOL_COMPLETED, TOOL_FAILED):
            turn = assistant(wall)
            name = str(payload.get("tool") or "unknown")
            if t == TOOL_STARTED:
                call = {
                    "id": f"journal-{ev['id']}",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(payload.get("args") or {}, default=str),
                    },
                    "_result": None,
                }
                turn["tool_calls"] = (turn["tool_calls"] or []) + [call]
                continue
            result = payload.get("error") if t == TOOL_FAILED else payload.get("result")
            for call in reversed(turn["tool_calls"] or []):
                if call["function"]["name"] == name and call["_result"] is None:
                    call["_result"] = result if isinstance(result, str) else json.dumps(result, default=str)
                    break
        elif t == RESPONSE_CANCELLED:
            continue  # timing rows are for the latency strip, not the bubbles

    return [
        t for t in turns if (t["content"].strip() or t["tool_calls"]) and not (t["role"] == "user" and not t["content"])
    ]


def env_badge(events: list[dict[str, Any]]) -> str | None:
    """Deployment room from the `env` stamp on the newest stamped payload
    (docs/environments.md A4). None until the journal has a stamped row."""
    for ev in reversed(events):
        env = ev["payload"].get("env")
        if isinstance(env, str) and env:
            return env
    return None


def liveness(conn: sqlite3.Connection, now: float | None = None) -> dict[str, Any]:
    """Is the shell alive, and how does its audio chain look right now?

    `last_wall`/`age_s` come from the newest row of any type; `audio_health` is the
    newest `audio.health` sample (mic device, drops, levels).
    """
    head = conn.execute("SELECT MAX(wall) AS wall FROM events").fetchone()
    last_wall = head["wall"] if head else None
    health = latest_events(conn, limit=1, type_=AUDIO_HEALTH)
    if last_wall is None:
        return {"last_wall": None, "age_s": 0.0, "alive": False, "audio_health": None}
    now = time.time() if now is None else now
    return {
        "last_wall": last_wall,
        "age_s": max(0.0, now - last_wall),
        "alive": now - last_wall <= STALE_AFTER_S,
        "audio_health": health[0]["payload"] if health else None,
    }


def readonly_reject(method: str, path: str, allow: tuple[str, ...] = ()) -> dict[str, Any] | None:
    """Pure read-only gate: None means allow, else the response to return.

    Safe methods always pass. A mutating method passes only when `path` starts with
    one of `allow` — the caller's explicit list of endpoints proven not to touch the
    journal. With the default empty list the dashboard refuses every write, so a
    booth visitor cannot make the source of truth diverge from what is on screen.
    """
    m = str(method or "").upper()
    if m in SAFE_METHODS:
        return None
    p = str(path or "")
    if any(p.startswith(prefix) for prefix in allow):
        return None
    return {"status": 405, "body": {"error": "journal source is read-only", "method": m, "path": p}}
