#!/usr/bin/env python3
"""Read the last N minutes of the journal as a timeline, with the interruption
question answered in one line.

Built for two moments: the bench (is he hearing himself? did my interrupt land?)
and the booth, when something looked wrong thirty seconds ago and the only record
is the database. Stdlib only, read-only on the DB (`mode=ro`), and it never opens
an audio device - safe to run while the shell is live.

The barge-in summary counts speech_started events that fall INSIDE a reply window
(response.created -> completed/cancelled), because that is the whole question: if
he cannot hear you while he is talking, the interrupt never happens and the journal
shows a clean conversation. It also separates "flushed while audio was queued"
(had_audio: true) from a flush that cut nothing, so a cancel that arrived after the
reply already finished is not counted as a barge-in.

    scripts/tail_journal.py                 # last 5 minutes
    scripts/tail_journal.py --minutes 30
    scripts/tail_journal.py --grep volume   # only rows mentioning volume
    scripts/tail_journal.py --json          # machine-readable

NOTE on this box: the sqlite3 CLI's datetime() returns 1999-12-31 for 2026 epochs
(see docs/booth-card.md). Everything here is computed in Python from the raw epoch.
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
    reply in progress still counts as 'he is talking right now'."""
    wins: list[tuple[float, float, str]] = []
    open_at: float | None = None
    for wall, t, p in rows:
        if t == "response.created":
            if open_at is not None:  # a second response before the first closed (tool follow-up)
                wins.append((open_at, wall, "superseded"))
            open_at = wall
        elif t in ("response.completed", "response.cancelled") and open_at is not None:
            wins.append((open_at, wall, t.split(".")[1]))
            open_at = None
    if open_at is not None:
        wins.append((open_at, time.time(), "in-progress"))
    return wins


def summarize(rows: list[tuple[float, str, str]], wins: list[tuple[float, float, str]]) -> dict:
    inside: list[float] = []
    cut_audio = 0
    cancelled = 0
    for wall, t, p in rows:
        if t == "audio.speech_started" and any(a <= wall <= b for a, b, _ in wins):
            inside.append(wall)
        if t == "tts.flushed" and '"had_audio": true' in p.replace("'had_audio': True", '"had_audio": true'):
            cut_audio += 1
        if t == "response.cancelled":
            cancelled += 1
    return {
        "replies": len(wins),
        "speech_started_mid_reply": len(inside),
        "flushed_with_audio_queued": cut_audio,
        "responses_cancelled": cancelled,
    }


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
        f"\n{len(rows)} rows | replies={stats['replies']} | speech_started_MID_reply={stats['speech_started_mid_reply']}"
    )
    if stats["replies"] and stats["speech_started_mid_reply"] == 0:
        print(
            "  → no one was heard while he was talking. Either nobody interrupted, or the mic is\n"
            "    gated shut during a reply (issue #40). Talk over his next answer to tell which."
        )
    print(
        f"  flushes that cut audible audio={stats['flushed_with_audio_queued']} "
        f"| responses cancelled={stats['responses_cancelled']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
