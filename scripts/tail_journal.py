#!/usr/bin/env python3
"""Read the last N minutes of the journal as a timeline, with the interruption
question answered in one line.

Built for two moments: the bench (is he hearing himself? did my interrupt land?)
and the booth, when something looked wrong thirty seconds ago and the only record
is the database. Stdlib only, read-only on the DB (`mode=ro`), and it never opens
an audio device - safe to run while the shell is live.

The primary barge-in signal is the one the payload already carries:
`audio.speech_started {"cut_playback": true}` means the shell cut audible playback
at that moment (shell/realtime/client.py -> speaker.cancel_current()). That is the
only barge-in count that does not depend on row order. `tts.flushed {"had_audio":
true}` and `response.cancelled` stay separate counts: a flush with nothing queued
and a cancel that arrived after the reply already finished are not audio cuts.

The secondary signal, speech_started_MID_reply, answers the narrower question "was
he heard while he was talking" (window = response.created/first_audio -> the first
completed/cancelled). It needs the causal tolerance described on `is_mid_reply()`:
without it a real interrupt is judged against an already-closed window and the
number prints 0 while the journal clearly shows barge-ins.

The advisory only blames a closed mic gate when there were replies AND zero
mid-reply speech AND zero cut_playback barge-ins; with cuts on the board the gate
is demonstrably open and saying otherwise is just wrong on a live booth screen.

    scripts/tail_journal.py                 # last 5 minutes
    scripts/tail_journal.py --minutes 30
    scripts/tail_journal.py --grep volume   # only rows mentioning volume
    scripts/tail_journal.py --json          # machine-readable

Timestamps: `wall` is a unix epoch and `mono` is seconds since the writing process
started (never subtract across a restart). Hand-written SQL needs
`datetime(wall,'unixepoch','localtime')` - without `unixepoch` SQLite reads the number as
a Julian day and answers 1999-12-31, which is a wrong query, not a broken CLI. Everything
here is computed in Python from the raw epoch, so it cannot make that mistake.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO / "data" / "events.db"
NOISE = ("audio.health", "attention.lease_renewed")

# The shell writes response.cancelled, tts.flushed and audio.speech_started from one
# interrupt within milliseconds of each other. Generous enough to cover a slow CPU,
# tight enough that two unrelated events seconds apart never pair up.
CAUSAL_PAIRING_S = 0.150


def load(db_path: Path, minutes: float, grep: str = "") -> list[tuple[float, str, str]]:
    if not db_path.is_file():
        raise SystemExit(f"no journal at {db_path} (run the shell once, or pass --db)")
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = db.execute(
        "select wall, type, payload from events where wall > ? order by id", (time.time() - minutes * 60,)
    ).fetchall()
    db.close()
    if grep:
        needle = grep.lower()
        rows = [r for r in rows if needle in (r[1] + (r[2] or "")).lower()]
    return [(float(w), t, p or "") for w, t, p in rows]


def reply_windows(rows: list[tuple[float, str, str]]) -> list[tuple[float, float, str]]:
    """(start, end, end_reason) per reply. An unclosed window runs to `now`, so a
    reply in progress still counts as 'he is talking right now'.

    first_audio also opens a window when the created row is missing (journal started
    mid-reply, or a retention trim cut the head off the window).
    """
    wins: list[tuple[float, float, str]] = []
    open_at: float | None = None
    for wall, t, p in rows:
        if t == "response.created":
            if open_at is not None:  # a second response before the first closed (tool follow-up)
                wins.append((open_at, wall, "superseded"))
            open_at = wall
        elif t == "response.first_audio" and open_at is None:
            open_at = wall
        elif t in ("response.completed", "response.cancelled") and open_at is not None:
            wins.append((open_at, wall, t.split(".")[1]))
            open_at = None
    if open_at is not None:
        wins.append((open_at, time.time(), "in-progress"))
    return wins


def payload_dict(p: str) -> dict:
    """The payload column as a dict; a truncated or non-JSON row just reads as empty."""
    try:
        parsed = json.loads(p)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def is_mid_reply(wall: float, wins: list[tuple[float, float, str]]) -> bool:
    """Did this speech_started land while a reply was live?

    ORDERING TRAP — do not "simplify" this back to a plain containment test. An
    interrupt closes its own window before it is judged: the shell writes
    response.cancelled, then tts.flushed, then audio.speech_started from the same
    callback, milliseconds apart, in that row order. Measured on tonight's journal
    (events id 89997 then 89999): cancelled at ...662.8480358, speech_started at
    ...662.848265. The speech lands ~0.2 ms AFTER `end`, outside every window, so a
    containment-only test printed speech_started_MID_reply=0 over a conversation
    with three real barge-ins in it.

    Pairing a speech_started with a window-closing cancel within CAUSAL_PAIRING_S on
    either side of the close is causal, not coincidence: the cancel is only ever
    written by the interrupt path (shell/realtime/client.py handle_event).
    """
    return any(a <= wall <= b or (reason == "cancelled" and abs(wall - b) <= CAUSAL_PAIRING_S) for a, b, reason in wins)


def summarize(rows: list[tuple[float, str, str]], wins: list[tuple[float, float, str]]) -> dict:
    barge_in = 0
    mid_reply = 0
    cut_audio = 0
    cancelled = 0
    for wall, t, p in rows:
        if t == "audio.speech_started":
            if payload_dict(p).get("cut_playback") is True:
                barge_in += 1  # the shell cut audible playback: this is a barge-in
            if is_mid_reply(wall, wins):
                mid_reply += 1
        elif t == "tts.flushed" and payload_dict(p).get("had_audio") is True:
            cut_audio += 1
        elif t == "response.cancelled":
            cancelled += 1
    return {
        "replies": len(wins),
        "barge_in_cut_playback": barge_in,
        "speech_started_mid_reply": mid_reply,
        "flushed_with_audio_queued": cut_audio,
        "responses_cancelled": cancelled,
    }


def advisory_lines(stats: dict) -> list[str]:
    """What the counts entitle us to say, one line each.

    The gating suspect (issue #40) may only be named when there were replies and
    neither interruption signal fired. A cut_playback barge-in is proof the mic was
    open during a reply — blaming the gate next to it is a lie on a public screen.
    """
    lines: list[str] = []
    if stats["barge_in_cut_playback"]:
        lines.append(
            f"  → barge-in works: {stats['barge_in_cut_playback']} speech_started cut_playback=true, "
            "audible playback was cut."
        )
    if stats["replies"] and not stats["barge_in_cut_playback"] and stats["speech_started_mid_reply"] == 0:
        lines.append(
            "  → no one was heard while he was talking. Either nobody interrupted, or the mic is\n"
            "    gated shut during a reply (issue #40). Talk over his next answer to tell which."
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--grep", default="", help="only rows whose type or payload contains this")
    ap.add_argument("--all", action="store_true", help="include audio.health and lease_renewed noise")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    rows = load(Path(args.db), args.minutes, args.grep)
    if not args.all:
        rows = [r for r in rows if r[1] not in NOISE]
    wins = reply_windows(rows)
    stats = summarize(rows, wins)

    if args.json:
        print(json.dumps({"stats": stats, "rows": [{"wall": w, "type": t, "payload": p} for w, t, p in rows]}))
        return 0
    if not rows:
        print(f"nothing in the last {args.minutes:g} minutes" + (f" matching {args.grep!r}" if args.grep else ""))
        print(
            "(no speech, no lease, no tools - if you were talking to him, the mic gate or the wake word is the suspect)"
        )
        return 1

    t0 = rows[0][0]
    print(f"journal, last {args.minutes:g} min ({datetime.fromtimestamp(t0):%H:%M:%S} → now)\n")
    prev = None
    for wall, t, p in rows:
        if prev is not None and wall - prev > 15:
            print(f"   …{wall - prev:4.0f}s quiet…")
        prev = wall
        detail = p[:74].replace("\n", " ")
        print(f"{wall - t0:6.1f}s  {t:26} {detail}")

    print(
        f"\n{len(rows)} rows | replies={stats['replies']}"
        f" | barge_in(audio cut)={stats['barge_in_cut_playback']}"
        f" | speech_started_MID_reply={stats['speech_started_mid_reply']}"
    )
    for line in advisory_lines(stats):
        print(line)
    print(
        f"  flushes that cut audible audio={stats['flushed_with_audio_queued']} "
        f"| responses cancelled={stats['responses_cancelled']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
