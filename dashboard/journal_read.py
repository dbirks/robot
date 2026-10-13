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
import re
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
KWS_DETECTED = "kws.detected"
RESPONSE_TRANSCRIPT_FINAL = "response.transcript_final"
RESPONSE_CANCELLED = "response.cancelled"
TOOL_STARTED = "tool.started"
TOOL_COMPLETED = "tool.completed"
TOOL_FAILED = "tool.failed"
AUDIO_HEALTH = "audio.health"
AUDIO_SPEECH_STARTED = "audio.speech_started"
AUDIO_SPEECH_STOPPED = "audio.speech_stopped"
RESPONSE_CREATED = "response.created"
RESPONSE_FIRST_AUDIO = "response.first_audio"
RESPONSE_COMPLETED = "response.completed"
ATTENTION_TOOL_DENIED = "attention.tool_denied"

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
    # The wake word is not a bubble. It is read because the display correction in
    # `display_transcript` only fires when the spotter actually heard the name - see
    # that docstring for why that gate is the whole design.
    KWS_DETECTED,
)

# The rows a trace is built from (issue #39). Everything here is written by the
# shell with the same `mono` clock, which is what makes the segment split below
# possible; `turn.transcript_partial` and the lease traffic are excluded because
# they multiply the window without adding a span.
TRACE_TYPES = (
    AUDIO_SPEECH_STARTED,
    AUDIO_SPEECH_STOPPED,
    TURN_TRANSCRIPT_FINAL,
    RESPONSE_CREATED,
    RESPONSE_FIRST_AUDIO,
    RESPONSE_COMPLETED,
    RESPONSE_CANCELLED,
    RESPONSE_TRANSCRIPT_FINAL,
    TOOL_STARTED,
    TOOL_COMPLETED,
    TOOL_FAILED,
    ATTENTION_TOOL_DENIED,
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
    env: str | None = None,
) -> list[dict[str, Any]]:
    """Ascending rows with id > after_id — the SSE cursor.

    `env` matters most here: the SSE stream is what a live screen actually shows, and
    an unfiltered cursor would push another room's conversation onto the booth display
    the moment somebody spoke (ADR 0003, issue #38).
    """
    q = f"SELECT {_EVENT_COLS} FROM events WHERE id > ?"
    args: list[Any] = [after_id]
    if types:
        q += f" AND type IN ({','.join('?' * len(types))})"
        args.extend(types)
    if env:
        q += " AND json_extract(payload, '$.env') = ?"
        args.append(env)
    q += " ORDER BY id ASC LIMIT ?"
    args.append(max(1, int(limit)))
    return [_row_to_event(r) for r in conn.execute(q, args)]


def latest_events(
    conn: sqlite3.Connection,
    limit: int = 200,
    type_: str | None = None,
    types: tuple[str, ...] | None = None,
    env: str | None = None,
) -> list[dict[str, Any]]:
    """Newest `limit` rows, returned oldest-first (the order turn assembly wants).

    `types` is how a conversation window avoids drowning in watchdog samples;
    `type_` stays for the single-type callers (liveness).

    `env` is the room filter (docs/environments.md, ADR 0003). The journal has always
    been ONE file, so the newest 100 conversation rows were the newest 100 rows of
    whatever room had last been talking - which meant the booth screen would open on
    the family's home history until an attendee said something. Only stamped rows can
    match, so pre-Phase-A rows are excluded from every room view: ambiguous belongs to
    nobody. `env=None` means unfiltered and is only for the badge probe that decides
    which room to filter to.
    """
    wanted = tuple(types) if types else ((type_,) if type_ else ())
    clauses: list[str] = []
    args: list[Any] = []
    if wanted:
        clauses.append(f"type IN ({','.join('?' * len(wanted))})")
        args.extend(wanted)
    if env:
        clauses.append("json_extract(payload, '$.env') = ?")
        args.append(env)
    q = f"SELECT {_EVENT_COLS} FROM events"
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(max(1, int(limit)))
    return [_row_to_event(r) for r in reversed(conn.execute(q, args).fetchall())]


def current_room(conn: sqlite3.Connection, probe: int = 20) -> str | None:
    """The room the shell is currently writing as: the env stamp on the newest
    stamped row. Callers filter every view by this, so a screen can never scroll into
    another room's conversations.

    None means the journal has no stamped row yet - a fresh machine, or history from
    before the stamp existed. That renders as an empty room rather than "show
    everything", because "show everything" is exactly the leak.
    """
    return env_badge(latest_events(conn, limit=max(1, int(probe))))


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


# What the STT actually returns for "hey Reachy" on this box, taken from tonight's
# journal rows rather than a guessed homophone list: `Richie`, `Ricci`, and "Hey
# Reachy" collapsing into the single word `Hotel`. Extend this by looking, not by
# imagining - the wake word is spotter-matched in its own rows:
#
#   sqlite3 -readonly data/events.db "select wall,payload from events where
#     type='kws.detected' or type='turn.transcript_final' order by id"
#
# and reading what follows each kws.detected. The name is BPE-out-of-vocabulary for
# Parakeet TDT v3 and that backend has no hotword/context knob at all
# (`nano_parakeet.transcribe(audio, timestamps)` is the whole signature, and the
# pinned service exposes no such flag for Parakeet), so this is a caption
# readability fix at the display layer - it does not make him hear better.
WAKE_WORD = "Reachy"
WAKE_ALIASES = {"richie", "ricci", "hotel"}
# The spotter fires on the same breath as the transcript it belongs to; anything
# further apart is a different utterance and none of our business.
KWS_WINDOW_S = 2.5


def wake_word_heard(kws_walls: list[float], wall: float) -> bool:
    return any(abs(kw - wall) <= KWS_WINDOW_S for kw in kws_walls)


def display_transcript(text: str, *, wake_word_heard_this_turn: bool) -> str:
    """The user's words as the caption should read them, for a screen in a public room.

    Three guards, because rewriting what somebody said is exactly the kind of thing
    an audit record must never do (ADR 0007): the keyword spotter must have fired
    within `KWS_WINDOW_S` of this transcript (proof the wake word was said, not that
    the ASR guessed), the alias must sit in the first few words ("was it quiet in
    Rome" has no business being touched, and neither does a request about a hotel),
    and only that one token changes. The journal keeps the verbatim text; the LLM is
    not touched either - `RealtimeClient` is the only thing that injects into that
    conversation, and a caption is not a reason to edit what a person asked.
    """
    if not text or not wake_word_heard_this_turn:
        return text
    words = text.split()
    for i, word in enumerate(words[:3]):
        core = re.sub(r"[^a-z']", "", word.lower())
        if core in WAKE_ALIASES:
            tail = re.match(r"[A-Za-z']*([^A-Za-z']*)", word)
            words[i] = WAKE_WORD + (tail.group(1) if tail else "")
            return " ".join(words)
    return text


def turns_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Journal rows -> the turn dicts partials/conversation.html renders.

    Keys match what the legacy `sessions.db` reader produced (id, timestamp, role,
    content, tool_calls) so the template and the SSE payload shape do not change.
    `timestamp` is the wall clock. Tool rows merge into the assistant turn that
    called them, with each call carrying `_result` like the legacy merge did.
    """
    turns: list[dict[str, Any]] = []
    next_id = 1
    kws_walls = [float(ev["wall"]) for ev in events if ev["type"] == KWS_DETECTED]

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
            said = str(payload.get("text") or "")
            push("user", wall)["content"] = display_transcript(
                said, wake_word_heard_this_turn=wake_word_heard(kws_walls, wall)
            )
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


def split_segments(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Cut the stream wherever `mono` goes backwards — i.e. a new process.

    `events.mono` is CLOCK_MONOTONIC relative to the *writing process* (see
    `Journal._t0` in shell/journal.py), so it is only comparable inside one run of
    `python -m shell`, and the shell restarts a dozen times a day. Across a restart
    the values are unrelated: a nearest-neighbour join over the whole table happily
    pairs a transcript from Tuesday with a response from Wednesday and reports a
    4000-second turn. A regression is therefore a segment boundary, and no duration
    is ever computed across one.

    The split is deliberately conservative in the safe direction: extra boundaries
    lose durations, they cannot invent them. There is no pid or boot row in the
    journal, so a regression is the only signal available — a restart whose very
    first traced event lands before the previous run's last one is indistinguishable
    from a long-lived process, and that shows up as one implausibly long turn, which
    is why `total_ms` is only reported for turns with a terminal row.
    """
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last: float | None = None
    for ev in events:
        mono = ev.get("mono")
        numeric = isinstance(mono, (int, float)) and not isinstance(mono, bool)
        if current and numeric and last is not None and mono < last:  # type: ignore[operator]
            segments.append(current)
            current = []
        current.append(ev)
        if numeric:
            last = float(mono)  # type: ignore[arg-type]
    if current:
        segments.append(current)
    return segments


def _ms(a: float | None, b: float | None) -> int | None:
    """b - a in milliseconds, clamped at zero (rows from the audio callback thread
    can land one commit out of order), None if either side is missing."""
    if a is None or b is None:
        return None
    return max(0, round((b - a) * 1000))


def _span(label: str, kind: str, t0_ms: int | None, dur_ms: int | None, detail: str = "") -> dict[str, Any]:
    return {"label": label, "kind": kind, "t0_ms": t0_ms, "dur_ms": dur_ms, "detail": detail}


def _mono(ev: dict[str, Any]) -> float | None:
    m = ev.get("mono")
    return float(m) if isinstance(m, (int, float)) and not isinstance(m, bool) else None


def _tool_verdict(result: Any) -> str:
    """One word about a tool result. Not the payload: `recall` returns saved memories
    and this page faces a room, so the trace says a lookup happened, not what it said."""
    if isinstance(result, dict):
        return "ok" if result.get("ok") else "not ok"
    return "done" if result else "empty"


def _turn_env(events: list[dict[str, Any]]) -> str | None:
    for ev in events:
        env = ev.get("payload", {}).get("env")
        if isinstance(env, str) and env:
            return env
    return None


def build_traces(events: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    """Journal rows -> one dict per conversation turn, oldest first.

    A *trace* is one turn of conversation: everything from the moment the shell
    started listening to the utterance that produced this reply, through the
    terminal `response.completed` / `response.cancelled` row. A *span* is one step
    inside it, drawn from the two events that bracket that step:

      vad           audio.speech_started -> audio.speech_stopped
      endpoint      audio.speech_stopped -> turn.transcript_final   (STT + endpointing)
      llm           turn.transcript_final -> response.created       (queue + prefill)
      first audio   response.created     -> response.first_audio   (time to first audio)
      done          response.created     -> response.completed / response.cancelled
      tool:<name>   tool.started         -> tool.completed / tool.failed
      denied        attention.tool_denied (marker, zero-length: it never runs)

    Every duration is a difference of two `mono` values *inside one segment* (see
    `split_segments`); `t0_ms` is the offset from the turn's own start so the page
    can lay spans out on one axis. Turns with no terminal row are flagged
    `incomplete` rather than given a made-up length — and note what no arithmetic
    can recover here: there is no playback-end event (`tts.flushed` reports a queue
    flush with `had_audio`, not the last sample out of the speaker), so
    time-to-silence is not measurable today and no span claims to be it.

    `limit` keeps the newest N turns, still in chronological order.
    """
    traces: list[dict[str, Any]] = []
    for index, segment in enumerate(split_segments(events)):
        traces.extend(_traces_in_segment(segment, index))
    if limit is not None and limit > 0:
        traces = traces[-int(limit) :]
    return traces


def _traces_in_segment(segment: list[dict[str, Any]], segment_index: int) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    open_tools: dict[str, list[Any]] = {}
    run_start: float | None = None
    run_stop: float | None = None

    for ev in segment:
        t = ev["type"]
        mono = _mono(ev)
        payload = ev.get("payload") or {}

        if t == TURN_TRANSCRIPT_FINAL:
            cur = _begin_turn(ev, mono, run_start, run_stop, segment_index, len(traces) + 1)
            open_tools = {}
            run_start = run_stop = None
            traces.append(cur)
            continue
        if t == AUDIO_SPEECH_STARTED:
            run_start, run_stop = mono, None
            if cur is not None and payload.get("cut_playback"):
                cur["interrupts"] += 1  # he was still talking: a real barge-in
            continue
        if t == AUDIO_SPEECH_STOPPED:
            run_stop = mono
            continue
        if cur is None:
            continue  # rows before this segment's first transcript have no turn to hang on

        cur["events"].append(ev)
        if t == RESPONSE_CREATED:
            if cur["created_mono"] is None:
                cur["created_mono"] = mono
                cur["response_id"] = str(payload.get("response_id") or "")
            cur["n_created"] += 1
        elif t == RESPONSE_FIRST_AUDIO and cur["first_audio_mono"] is None:
            if not cur["response_id"] or str(payload.get("response_id") or "") in ("", cur["response_id"]):
                cur["first_audio_mono"] = mono
        elif t == RESPONSE_COMPLETED and cur["terminal_mono"] is None:
            cur["terminal_mono"], cur["outcome"] = mono, "done"
            cur["status"] = str(payload.get("status") or "")
        elif t == RESPONSE_CANCELLED and cur["terminal_mono"] is None:
            cur["terminal_mono"], cur["outcome"] = mono, "cancelled"
            cur["cancel_reason"] = str(payload.get("reason") or "")
        elif t == RESPONSE_TRANSCRIPT_FINAL and not cur["assistant_text"]:
            cur["assistant_text"] = str(payload.get("text") or "")
        elif t == TOOL_STARTED:
            name = str(payload.get("tool") or "unknown")
            args = json.dumps(payload.get("args") or {}, default=str)[:160]
            open_tools.setdefault(name, []).append((mono, args))
        elif t in (TOOL_COMPLETED, TOOL_FAILED):
            name = str(payload.get("tool") or "unknown")
            pending = open_tools.get(name) or []
            started, args = pending.pop(0) if pending else (None, "")
            note = payload.get("error") if t == TOOL_FAILED else payload.get("result")
            verdict = (str(note)[:160] if t == TOOL_FAILED else _tool_verdict(note))[:160]
            cur["tool_spans"].append(
                (name, started, mono, "failed" if t == TOOL_FAILED else "ok", f"{args} -> {verdict}")
            )
        elif t == ATTENTION_TOOL_DENIED:
            cur["denied"].append(
                (mono, str(payload.get("tool") or ""), payload.get("confidence"), payload.get("threshold"))
            )

    for trace in traces:
        _finish_turn(trace)
    return traces


def _begin_turn(
    ev: dict[str, Any],
    mono: float | None,
    run_start: float | None,
    run_stop: float | None,
    segment_index: int,
    seq: int,
) -> dict[str, Any]:
    base = run_start if run_start is not None else (run_stop if run_stop is not None else mono)
    return {
        "id": int(ev.get("id") or 0),
        "seq": seq,
        "segment": segment_index,
        "env": None,
        "start_wall": ev.get("wall"),
        "base_mono": base,
        "transcript_mono": mono,
        "user_text": str((ev.get("payload") or {}).get("text") or ""),
        "assistant_text": "",
        "response_id": "",
        "vad_start_mono": run_start,
        "vad_stop_mono": run_stop,
        "created_mono": None,
        "first_audio_mono": None,
        "terminal_mono": None,
        "outcome": None,
        "status": "",
        "cancel_reason": "",
        "n_created": 0,
        "interrupts": 0,
        "tool_spans": [],
        "denied": [],
        "events": [ev],
    }


def _finish_turn(cur: dict[str, Any]) -> dict[str, Any]:
    base = cur["base_mono"]
    transcript = cur["transcript_mono"]
    spans: list[dict[str, Any]] = []
    if cur["vad_start_mono"] is not None and cur["vad_stop_mono"] is not None:
        spans.append(
            _span("vad", "vad", _ms(base, cur["vad_start_mono"]), _ms(cur["vad_start_mono"], cur["vad_stop_mono"]))
        )
        endpoint_t0 = _ms(base, cur["vad_stop_mono"])
    else:
        endpoint_t0 = _ms(base, transcript)
    spans.append(
        _span(
            "endpoint + stt",
            "stt",
            endpoint_t0,
            _ms(cur["vad_stop_mono"] if cur["vad_stop_mono"] is not None else base, transcript),
            "no vad rows in this segment" if cur["vad_start_mono"] is None else "",
        )
    )
    spans.append(_span("llm", "llm", _ms(base, transcript), _ms(transcript, cur["created_mono"])))
    spans.append(
        _span("first audio", "audio", _ms(base, cur["created_mono"]), _ms(cur["created_mono"], cur["first_audio_mono"]))
    )
    for name, started, ended, status, detail in cur["tool_spans"]:
        spans.append(_span(f"tool:{name}", "tool", _ms(base, started), _ms(started, ended), f"{status} · {detail}"))
    for mono_at, tool, conf, thresh in cur["denied"]:
        spans.append(_span(f"denied:{tool}", "denied", _ms(base, mono_at), 0, f"confidence {conf} < {thresh}"))
    terminal_label = "cancelled" if cur["outcome"] == "cancelled" else "done"
    spans.append(
        _span(
            terminal_label,
            "cancelled" if cur["outcome"] == "cancelled" else "done",
            _ms(base, cur["created_mono"] if cur["created_mono"] is not None else transcript),
            _ms(cur["created_mono"], cur["terminal_mono"]),
            cur["cancel_reason"] or cur["status"],
        )
    )

    ends = [s["t0_ms"] + s["dur_ms"] for s in spans if s["t0_ms"] is not None and s["dur_ms"] is not None]
    cur["spans"] = spans
    cur["env"] = cur["env"] or _turn_env(cur["events"])
    cur["durations_ms"] = {
        "vad": next((s["dur_ms"] for s in spans if s["kind"] == "vad"), None),
        "endpoint": next((s["dur_ms"] for s in spans if s["kind"] == "stt"), None),
        "llm": next((s["dur_ms"] for s in spans if s["kind"] == "llm"), None),
        "first_audio": next((s["dur_ms"] for s in spans if s["kind"] == "audio"), None),
        "total": _ms(base, cur["terminal_mono"]),
    }
    cur["total_ms"] = cur["durations_ms"]["total"]
    cur["span_ms"] = max(ends) if ends else 0
    cur["incomplete"] = cur["terminal_mono"] is None
    cur["wall_end"] = max((e.get("wall") or 0) for e in cur["events"]) or cur["start_wall"]
    for key in (
        "vad_start_mono",
        "vad_stop_mono",
        "created_mono",
        "first_audio_mono",
        "terminal_mono",
        "base_mono",
        "events",
    ):
        cur.pop(key, None)
    return cur


def trace_window(
    conn: sqlite3.Connection,
    window: int = 2000,
    limit: int = 12,
    turn_id: int | None = None,
    env: str | None = None,
) -> dict[str, Any]:
    """The newest turns as traces, plus what the reader had to give up on.

    `window` traced rows are enough to see the last few processes: the trace types
    are ~5k of the journal's ~90k rows, and the segment split needs the regression
    itself to be inside the window to be seen (which it is, because a restart writes
    its low `mono` values before it writes the next transcript).
    """
    events = latest_events(conn, limit=window, types=TRACE_TYPES, env=env)
    traces = build_traces(events)
    newest = traces[-limit:] if limit > 0 else traces
    return {
        "traces": list(reversed(newest)),
        "turns_total": len(traces),
        "segments": len(split_segments(events)),
        "env": env_badge(events),
        "traced_rows": len(events),
        "turn": next((t for t in traces if turn_id is not None and t["id"] == int(turn_id)), None),
    }


def liveness(conn: sqlite3.Connection, now: float | None = None, env: str | None = None) -> dict[str, Any]:
    """Is the shell alive, and how does its audio chain look right now?

    `last_wall`/`age_s` come from the newest row of any type; `audio_health` is the
    newest `audio.health` sample (mic device, drops, levels). With `env` set, both are
    this room's rows: a home process writing health every 10 s must not keep a
    "SHELL ALIVE" badge up for a booth screen whose own shell is stopped.
    """
    if env:
        head = conn.execute(
            "SELECT MAX(wall) AS wall FROM events WHERE json_extract(payload, '$.env') = ?", (env,)
        ).fetchone()
    else:
        head = conn.execute("SELECT MAX(wall) AS wall FROM events").fetchone()
    last_wall = head["wall"] if head else None
    health = latest_events(conn, limit=1, type_=AUDIO_HEALTH, env=env)
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
