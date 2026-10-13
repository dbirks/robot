"""The barge-in metric in scripts/tail_journal.py, pinned with synthetic rows.

Hardware-free and DB-free: every test hands the module (wall, type, payload) tuples
in the shape `load()` returns them. This is the metric that reads a live booth, so
the ordering trap that made it print 0 during three real barge-ins is a test, not a
comment (see the live journal rows quoted inside is_mid_reply).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# scripts/ is not a package: this insert is what makes the import below real at
# runtime, and the only reason ty cannot resolve it (hence its ignore, same line).
sys.path.insert(0, str(ROOT / "scripts"))

import tail_journal as tj  # ty: ignore[unresolved-import]  # noqa: E402

E = 1_791_857_600.0  # an arbitrary epoch; only deltas matter


def ev(wall: float, type_: str, **payload: object) -> tuple[float, str, str]:
    """One journal row, payload rendered the way shell/journal.py stores it."""
    return (E + wall, type_, json.dumps({"env": "home", **payload}))


def stats_for(rows: list[tuple[float, str, str]]) -> dict:
    return tj.summarize(rows, tj.reply_windows(rows))


def test_a_cancel_milliseconds_before_speech_started_still_counts_as_mid_reply():
    """The trap: the interrupt closes its own window before it is judged.

    response.cancelled is journalled ahead of the audio.speech_started that caused
    it, so containment alone (start <= wall <= end) says the reply was over.
    """
    rows = [
        ev(0.0, "response.created"),
        ev(1.0, "response.first_audio"),
        ev(2.0, "response.cancelled"),
        ev(2.005, "audio.speech_started", cut_playback=True),
    ]
    wins = tj.reply_windows(rows)
    wall = rows[3][0]
    assert wins == [(E + 0.0, E + 2.0, "cancelled")]
    assert not any(a <= wall <= b for a, b, _ in wins), "the containment test this replaces"
    assert tj.is_mid_reply(wall, wins) is True
    stats = stats_for(rows)
    assert stats["speech_started_mid_reply"] == 1
    assert stats["barge_in_cut_playback"] == 1
    assert stats["responses_cancelled"] == 1


def test_a_cut_playback_speech_started_is_counted_as_a_barge_in():
    """The payload is the primary signal: the shell cut audible playback, period.

    The third barge-in of the evening looked like this: the reply had already
    completed upstream while its audio was still queued, so there is no window and
    no response.cancelled — only cut_playback and had_audio carry it.
    """
    rows = [
        ev(0.0, "response.created"),
        ev(1.0, "response.first_audio"),
        ev(2.0, "response.completed"),
        ev(3.9, "tts.flushed", had_audio=True),
        ev(3.9, "audio.speech_started", cut_playback=True),
    ]
    stats = stats_for(rows)
    assert stats["barge_in_cut_playback"] == 1
    assert stats["flushed_with_audio_queued"] == 1
    assert stats["responses_cancelled"] == 0
    assert stats["replies"] == 1


def test_speech_during_silence_is_not_mid_reply():
    """Nobody was interrupted: quiet room, then one completed reply, then talk."""
    rows = [
        ev(0.0, "audio.speech_started", cut_playback=False),
        ev(2.0, "response.created"),
        ev(3.0, "response.first_audio"),
        ev(6.0, "response.completed"),
        ev(30.0, "audio.speech_started", cut_playback=False),
        ev(31.0, "audio.speech_started", cut_playback=False),
    ]
    stats = stats_for(rows)
    assert stats["barge_in_cut_playback"] == 0
    assert stats["speech_started_mid_reply"] == 0
    assert stats["flushed_with_audio_queued"] == 0


def test_the_advisory_stops_blaming_the_gate_when_audio_was_cut():
    """The issue #40 suspect is allowed only with replies and no interruption signal."""
    quiet = {"barge_in_cut_playback": 0, "speech_started_mid_reply": 0, "replies": 4}
    assert len(tj.advisory_lines(quiet)) == 1 and "issue #40" in tj.advisory_lines(quiet)[0]
    nobody_home = {"barge_in_cut_playback": 0, "speech_started_mid_reply": 0, "replies": 0}
    assert tj.advisory_lines(nobody_home) == []
    cut = {"barge_in_cut_playback": 3, "speech_started_mid_reply": 0, "replies": 4}
    lines = tj.advisory_lines(cut)
    assert len(lines) == 1 and "barge-in works: 3" in lines[0]


def test_an_unclosed_reply_still_counts_as_talking():
    """A reply in flight has no completed/cancelled row yet; he is talking right now."""
    now = time.time()
    rows = [
        (now - 10.0, "response.created", '{"env": "home"}'),
        (now - 9.0, "response.first_audio", '{"env": "home"}'),
        (now - 0.5, "audio.speech_started", '{"env": "home", "cut_playback": false}'),
    ]
    stats = stats_for(rows)
    assert stats["replies"] == 1
    assert stats["speech_started_mid_reply"] == 1
    assert tj.reply_windows(rows)[0][2] == "in-progress"
