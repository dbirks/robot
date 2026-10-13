#!/usr/bin/env python
"""Bench the echo path by asking the question the complaint actually poses:
does he hear himself? (issue #40, ADR 0005)

    scripts/bench_aec.py                      route + SELF-HEARING + chip read-back + summary
    scripts/bench_aec.py --self-hear          the self-hearing question alone: journal only, no
                                              USB handle at all (safe beside a live session)
    scripts/bench_aec.py --minutes 180        widen the journal window
    scripts/bench_aec.py --chip-during-reply 60
    scripts/bench_aec.py --self-check         pure logic only, no device, no DB (the CI path)
    scripts/bench_aec.py --echo-rms           the old tone/RMS step (opens the mic)
    scripts/bench_aec.py --person             + talk over him (opens the mic)

SELF-HEARING is the verdict, and it is measured at the layer the complaint lives
at: the shell plays a known single-word clip (`reaction.ack_played` in the
journal) and we ask whether `turn.transcript_final` came back with that word in
it. The acks were played BY the shell, so `speaker.playing` was true
(shell/audio/speaker.py:153) and the mic gate was genuinely open
(shell/__main__.py:361 forwards real PCM only while `lease.active() or
speaker.playing`). A bench that played its own clips could never satisfy that
condition, and ADR 0002 forbids it opening a second stream beside a live shell
anyway - so this script opens no capture or playback stream unless `--echo-rms`
or `--person` is asked for, and those still refuse while reachy-shell is active.

Why the old verdict is gone: the tone step compared mic RMS during a 4 s tone to
mic RMS in silence. Twice in one night, on identical input, it printed x2.7 and
then x0.5. That is the XVF3800's own AGC/DNS reacting to a loud far-end; the
ratio cannot tell "echo survived" from "the front end moved the gain", so it
answers nothing. The step is DEMOTED, not deleted - the instability is itself a
finding - and it is labelled below the line as not a cancellation measurement.

Two traps worth naming before anyone reaches for a tool:
  * `amixer -c N` takes a card INDEX and this box moved the Reachy card from
    card 1 to card 3 across a power cycle. Everything here identifies audio by
    NAME (wpctl); a card number in the output is a bug.
  * `AEC_AECCONVERGED` read at idle means nothing. It has to be sampled while the
    far-end is active, which is what --chip-during-reply does: it polls the chip
    only inside the window where the journal says he is audible
    (response.first_audio -> response.completed/cancelled).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO / "data" / "events.db"
SR = 16000
BLOCK = 512

ACK_EVENT = "reaction.ack_played"
TRANSCRIPT_EVENT = "turn.transcript_final"
SPEECH_STARTED = "audio.speech_started"
LEASE_OPEN = ("attention.lease_started", "attention.lease_renewed", "attention.lease_transferred")
LEASE_CLOSE = "attention.lease_expired"

# The idle read-back table. This script never writes a chip parameter.
PARAMS = (
    "VERSION",
    "AEC_AECCONVERGED",
    "AEC_AECPATHCHANGE",
    "AEC_NUM_FARENDS",
    "AEC_RT60",
    "AUDIO_MGR_SYS_DELAY",
    "AUDIO_MGR_REF_GAIN",
    "AUDIO_MGR_MIC_GAIN",
    "AUDIO_MGR_OP_L",
    "AUDIO_MGR_OP_R",
    "AUDIO_MGR_SELECTED_CHANNELS",
    "PP_ECHOONOFF",
    "PP_DTSENSITIVE",
    "PP_MIN_NS",
    "PP_AGCMAXGAIN",
    "AEC_FIXEDBEAMSONOFF",
    "DOA_VALUE_RADIANS",
)
# The subset that answers "is the reference there AND working while he is busy".
REPLY_PARAMS = (
    "AEC_AECCONVERGED",
    "AEC_NUM_FARENDS",
    "AEC_AECPATHCHANGE",
    "PP_ECHOONOFF",
    "PP_DTSENSITIVE",
    "PP_MIN_NS",
)

# The clip corpus is scripts/make_acks.py output: <word>-<i><take>.wav. Parakeet's
# spelling of a 300 ms grunt is not stable, so the accepted spellings are written
# down rather than left to a fuzzy matcher.
CLIP_WORDS: dict[str, tuple[str, ...]] = {
    "hmm": ("hmm", "hmmm", "hm", "hum", "mhm", "mhmm", "hem", "mm"),
    "yes": ("yes", "yess", "jes"),
    "yeah": ("yeah", "yeahh", "yeh"),
    "right": ("right", "rite", "ryte"),
    "okay": ("okay", "ok", "okey", "ohkay", "kay"),
    "well": ("well", "weel", "wel"),
}
# A self-heard grunt is one word. Three tokens is generous; past that it is a
# person talking, and counting it as self-hearing would invent an echo.
MAX_ECHO_TOKENS = 3
MATCH_WINDOW_S = 2.5
# Below this many judgeable acks, "nobody heard him" is not an answer yet.
MIN_ACKS_FOR_A_NO = 5

Row = tuple[float, str, str]  # (wall, type, payload json) - the tail_journal.load() shape


# --------------------------------------------------------------- journal reading


def rows_from_db(db: Path, minutes: float, now: float | None = None) -> list[Row]:
    """The last `minutes` of the journal, read-only, newest window first by id.

    Only `wall` windows anything here: `mono` is CLOCK_MONOTONIC relative to the
    writing process (shell/journal.py:_t0) and never lines up across a restart.
    """
    if not Path(db).is_file():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cutoff = (now if now is not None else time.time()) - minutes * 60.0
        got = con.execute(
            "select wall, type, payload from events where wall > ? order by id",
            (cutoff,),
        ).fetchall()
    finally:
        con.close()
    return [(float(w), str(t), p or "") for w, t, p in got]


def payload_dict(p: str) -> dict:
    """The payload column as a dict; a truncated or non-JSON row reads as empty."""
    try:
        parsed = json.loads(p)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ------------------------------------------------------------- pure decision bits


def clip_word(sound: str) -> str | None:
    """`hmm-70.wav` -> `hmm`; a clip whose word we do not know cannot be judged."""
    stem = Path(str(sound)).name.lower().split(".")[0]
    prefix = stem.rsplit("-", 1)[0] if "-" in stem else stem
    return prefix if prefix in CLIP_WORDS else None


def tokens(text: str) -> list[str]:
    return re.sub(r"[^a-z ]", " ", str(text).lower()).split()


def within_one(a: str, b: str) -> bool:
    """Edit distance <= 1, dependency-free."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    long, short = (a, b) if len(a) > len(b) else (b, a)
    return any(long[:i] + long[i + 1 :] == short for i in range(len(long)))


def resembles(text: str, word: str | None, max_tokens: int = MAX_ECHO_TOKENS) -> bool:
    """Would this transcript plausibly BE the clip, coming back through the mic?"""
    if not word:
        return False
    toks = tokens(text)
    if not toks or len(toks) > max_tokens:
        return False
    want = set(CLIP_WORDS.get(word, ())) | {word}
    for tok in toks:
        if tok in want:
            return True
        if len(tok) >= 4 and len(word) >= 4 and any(within_one(tok, w) for w in want):
            return True
    return False


def last_wall(rows: list[Row], now: float | None = None) -> float:
    """The far edge of an unclosed interval.

    max(), not rows[-1][0]: the DB hands rows back in id order, but a caller that
    builds rows by hand (every test, and --self-check) does not, and an unclosed
    interval clipped to the last row in the list read as a lease that ended before
    the ack it was supposed to cover.
    """
    if now is not None:
        return float(now)
    return max([w for w, _t, _p in rows], default=0.0)


def lease_intervals(rows: list[Row], now: float | None = None) -> list[tuple[float, float]]:
    """Attention-lease coverage. A lease that never expired runs to `now`.

    Only the OPEN/CLOSE event names matter (renewals widen the interval because
    the interval still ends at the expiry that follows them).
    """
    out: list[list[float]] = []
    open_at: float | None = None
    for wall, t, _p in rows:
        if t in LEASE_OPEN:
            open_at = wall if open_at is None else open_at
        elif t == LEASE_CLOSE and open_at is not None:
            out.append([open_at, wall])
            open_at = None
    if open_at is not None:
        out.append([open_at, float(max(last_wall(rows, now), open_at))])
    return [(a, b) for a, b in out]


def lease_live(intervals: list[tuple[float, float]], wall: float) -> bool:
    return any(a <= wall <= b for a, b in intervals)


def reply_windows(rows: list[Row], now: float | None = None) -> list[tuple[float, float]]:
    """Windows where he is AUDIBLE: response.first_audio until completed/cancelled.

    first_audio, not response.created: between creation and the first audio block
    the speaker is silent, and an idle chip read in there would be sold as a
    far-end-active read. A window with no close runs to `now`.
    """
    out: list[list[float]] = []
    open_at: float | None = None
    for wall, t, _p in rows:
        if t == "response.first_audio":
            if open_at is not None:  # a reply that never closed (restart, crash)
                out.append([open_at, wall])
            open_at = wall
        elif t in ("response.completed", "response.cancelled") and open_at is not None:
            out.append([open_at, wall])
            open_at = None
    if open_at is not None:
        out.append([open_at, float(max(last_wall(rows, now), open_at))])
    return [(a, b) for a, b in out]


def is_audible(windows: list[tuple[float, float]], wall: float) -> bool:
    return any(a <= wall <= b for a, b in windows)


def gate_state(rows: list[Row]) -> dict:
    """Is the mic gate open during a reply? Only one row type proves it.

    `audio.speech_started {"cut_playback": true}` is written when the shell cut
    audible playback (shell/realtime/client.py), so it is proof the capture was
    live mid-reply. None means unproven, never "closed": the richer mid-reply
    metric lives in scripts/tail_journal.py and is not re-derived weaker here.
    """
    cuts = [w for w, t, p in rows if t == SPEECH_STARTED and payload_dict(p).get("cut_playback") is True]
    replies = sum(1 for _w, t, _p in rows if t == "response.created")
    return {
        "barge_in_cut_playback": len(cuts),
        "replies": replies,
        "gate_open_during_reply": True if cuts else None,
    }


def self_hearing(rows: list[Row], *, match_window_s: float = MATCH_WINDOW_S) -> dict:
    """Per ack clip: did the pipeline transcribe what he just played at himself?

    The pairing trap: the ack fires AFTER the transcript that caused it (wake word
    -> reaction.ack_played, shell/reactions.py), so the transcript that explains an
    ack is always at a NEGATIVE delta and must never be read as self-hearing. The
    service also re-emits the same turn a second later, so a post-ack transcript
    whose text already appeared just before the ack is the same human speech twice,
    not his own voice coming back.
    """
    acks = [(w, str(payload_dict(p).get("sound") or "")) for w, t, p in rows if t == ACK_EVENT]
    trs = [(w, str(payload_dict(p).get("text") or "")) for w, t, p in rows if t == TRANSCRIPT_EVENT]
    leases = lease_intervals(rows)
    clips: list[dict] = []
    for wall, sound in acks:
        word = clip_word(sound)
        before = {tuple(tokens(t)) for tw, t in trs if wall - 3.0 <= tw < wall}
        after = [(tw, t) for tw, t in trs if 0.0 < tw - wall <= match_window_s]
        fresh = [(tw, t) for tw, t in after if tuple(tokens(t)) not in before]
        hits = [(tw, t) for tw, t in fresh if resembles(t, word)]
        shown = hits or fresh
        clips.append(
            {
                "clip": sound,
                "word": word,
                "heard_himself": bool(hits),
                "transcript": shown[0][1] if shown else "",
                "transcript_delta_s": round(shown[0][0] - wall, 2) if shown else None,
                "lease_live": lease_live(leases, wall),
                "wall": wall,
            }
        )
    judged = [c for c in clips if c["word"] is not None]
    heard = [c for c in judged if c["heard_himself"]]
    return {
        "acks": len(clips),
        "judged": len(judged),
        "heard": len(heard),
        "state": self_hearing_state(len(judged), len(heard)),
        "clips": clips,
        "heard_clips": [f"{c['clip']} {c['transcript']!r}" for c in heard],
    }


def self_hearing_state(judged: int, heard: int) -> str:
    """`never-exercised` and `inconclusive` are states, never passes."""
    if judged == 0:
        return "never-exercised"
    if heard:
        return "self-hearing-observed"
    if judged < MIN_ACKS_FOR_A_NO:
        return "inconclusive"
    return "no-self-transcription-observed"


# ------------------------------------------------------------- chip read-back


def first_number(values: Any) -> float | None:
    """Numeric readback of a parameter: the first coercible element, or None.

    A raw read() of an int32 comes back as [status, b0, b1, b2, b3] - bytes, not a
    number, and its first element is a status code - so this is only ever fed
    read_values() output, which decodes to a tuple of numbers. Anything else
    (including a string, which is what a `char` parameter reads as) is None rather
    than a fabricated measurement.
    """
    if values is None or isinstance(values, (bool, str)):
        return None
    if isinstance(values, (int, float)):
        return float(values)
    if isinstance(values, (tuple, list)):
        for v in values:
            if isinstance(v, (bool, str)):
                continue
            if isinstance(v, (int, float)):
                return float(v)
    return None


def read_numbers(dev: Any, names: tuple[str, ...] = REPLY_PARAMS) -> dict[str, float | None]:
    """Parameters as numbers through an open handle. Never raises; a failed read
    is a None plus a named error key, so a missing readback cannot look like a 0."""
    out: dict[str, float | None] = {}
    if dev is None:
        return {name: None for name in names}
    for name in names:
        try:
            out[name] = first_number(dev.read_values(name))
        except Exception as e:  # noqa: BLE001
            out[name] = None
            out[f"{name}_error"] = type(e).__name__
    return out


def summarize_chip(idle: dict, during: list[dict]) -> dict:
    """Idle values next to the values sampled with a live far-end, plus verdicts."""
    merged: dict[str, dict] = {}
    for name in REPLY_PARAMS:
        merged[name] = {"idle": idle.get(name), "during": [d.get(name) for d in during if d.get(name) is not None]}
    if not during:
        return {
            "samples": 0,
            "reference_present": None,
            "params": merged,
            "verdict": [
                "  no mid-reply samples: he was never audible in the window, so the chip was never\n"
                "  read with a live far-end. That is a MISSING measurement, not a 'no'."
            ],
        }
    n = len(during)
    conv = [v for v in merged["AEC_AECCONVERGED"]["during"] if v]
    far = [v for v in merged["AEC_NUM_FARENDS"]["during"] if v]
    echo_on = [v for v in merged["PP_ECHOONOFF"]["during"] if v]
    path = [v for v in merged["AEC_AECPATHCHANGE"]["during"] if v]
    lines = [f"  {n} samples taken while the journal says he was audible:"]
    for name in REPLY_PARAMS:
        got = merged[name]["during"]
        pretty = ", ".join(f"{v:g}" for v in got) if got else "no numeric readback"
        idl = merged[name]["idle"]
        lines.append(f"    {name:22s} idle={'?' if idl is None else format(idl, 'g')}  during=[{pretty}]")
    if not far:
        lines.append(
            "  => AEC_NUM_FARENDS is 0 in every mid-reply sample: while he is actually playing,\n"
            "     the chip counts no far-end reference at all. No cancellation on this route, and\n"
            "     option A would make him interrupt himself."
        )
        ref: bool | None = False
    else:
        ref = True
        lines.append(f"  => far-end reference present in {len(far)}/{n} mid-reply samples.")
        if not echo_on:
            lines.append("     PP_ECHOONOFF read 0 the whole time: cancellation is switched OFF in firmware.")
        elif len(conv) == n:
            lines.append(f"     AEC_AECCONVERGED reported convergence in all {n} mid-reply samples.")
        elif conv:
            lines.append(
                f"     AEC_AECCONVERGED came up in {len(conv)}/{n} samples: it converges while playing\n"
                "     but does not hold it - read PP_MIN_NS and AUDIO_MGR_SYS_DELAY next."
            )
        else:
            lines.append(
                "     AEC never reported convergence during playback. An idle AECCONVERGED=1 does\n"
                "     not rescue this: it was never true while the far-end was live."
            )
    if path:
        lines.append(f"  AEC_AECPATHCHANGE nonzero in {len(path)}/{n} samples: the echo path is moving.")
    return {"samples": n, "reference_present": ref, "params": merged, "verdict": lines}


# ---------------------------------------------------------------- the summary


def summary_lines(sh: dict, gate: dict, chip: dict) -> list[str]:
    """The two decisions the barge-in work needs, in the order they gate each other."""
    out: list[str] = []
    if gate["gate_open_during_reply"]:
        out.append(
            f"1. GATE OPEN DURING A REPLY: YES - {gate['barge_in_cut_playback']} "
            "audio.speech_started(cut_playback=true) over\n"
            f"   {gate['replies']} replies in this window. He was heard while he was talking, so the\n"
            "   barge-in complaint is not a mic gate that is shut on this route."
        )
    elif gate["replies"]:
        out.append(
            f"1. GATE OPEN DURING A REPLY: UNPROVEN - {gate['replies']} replies, zero cut_playback\n"
            "   barge-ins. Either nobody interrupted him or the gate really is shut (issue #40);\n"
            "   talk over his next answer to tell those apart. Do not decide from this line alone."
        )
    else:
        out.append("1. GATE OPEN DURING A REPLY: no replies in this window, so nothing to say.")

    state = sh["state"]
    if state == "self-hearing-observed":
        out.append(
            f"2. DOES HIS OWN VOICE REACH STT: YES - {sh['heard']}/{sh['judged']} ack clips came back as their\n"
            f"   own word ({'; '.join(sh['heard_clips'][:3])}). He hears himself through the pipeline, so\n"
            "   opening the gate for the whole reply window (option A) will make him interrupt\n"
            "   himself unless something cancels the echo first."
        )
    elif state == "no-self-transcription-observed":
        out.append(
            f"2. DOES HIS OWN VOICE REACH STT: NO SELF-TRANSCRIPTION OBSERVED behind {sh['judged']} clips he\n"
            "   played himself (0 came back as their own word). On this evidence his own voice does\n"
            "   not reach the transcriber, so a widened gate is not automatically self-interrupting.\n"
            '   That is not "the AEC works" - see LIMITS.'
        )
    elif state == "never-exercised":
        out.append(
            "2. DOES HIS OWN VOICE REACH STT: NEVER EXERCISED. No reaction.ack_played row in the\n"
            "   window, so he never played anything at himself and this run measured nothing. Say the\n"
            "   wake word alone a few times (that queues an ack through his own speaker), then\n"
            "   re-run. THIS IS NOT A PASS."
        )
    else:
        out.append(
            f"2. DOES HIS OWN VOICE REACH STT: INCONCLUSIVE - only {sh['judged']} judgeable acks, fewer than\n"
            f"   {MIN_ACKS_FOR_A_NO}. None came back as their own word, but that is not yet a result. "
            "Keep waking him."
        )
    idle_acks = sum(1 for c in sh["clips"] if not c["lease_live"])
    if sh["acks"] and idle_acks:
        out.append(
            f"   ({idle_acks} of {sh['acks']} acks had no attention lease live at that moment: only the\n"
            "   clip's own playback was forwarded then. That is still the window that matters for\n"
            "   self-hearing, but an echo TAIL after the clip could not have been heard at all.)"
        )

    if chip.get("samples"):
        out.append("3. CHIP, SAMPLED WHILE THE FAR-END WAS LIVE:")
        out.extend(chip["verdict"])
    elif "error" in chip:
        out.append(f"3. CHIP: {chip['error']} - a missing readback is a result, not a crash.")
    elif chip.get("skipped"):
        out.append(
            "3. CHIP: not read this run (--self-hear asks only the self-hearing question).\n"
            "   --chip-during-reply SECONDS is what reads AECCONVERGED with a live far-end."
        )
    else:
        out.append(
            "3. CHIP: read at idle only. --chip-during-reply SECONDS is what catches AECCONVERGED\n"
            "   with a live far-end; an idle 1 says nothing about playback."
        )
    out.append(
        "LIMITS. An ack is 0.2-0.8 s of one known word at ack level, so a 'no' above is weak\n"
        "  evidence about his own RUNNING speech, which is longer, hotter and a different\n"
        "  spectrum: the honest claim is 'no self-transcription observed behind N acks', never\n"
        "  'the AEC works'. Nothing here is ERLE, and the tone/RMS step is AGC/DNS-confounded by\n"
        "  construction (x2.7 then x0.5 on identical input). ADR 0005 still wants a measured\n"
        "  echo-return-loss number before the cancellation counts as characterized."
    )
    return out


# ------------------------------------------------- device-touching code, kept late


def shell_running() -> bool:
    if shutil.which("systemctl") is None:
        return False
    r = subprocess.run(["systemctl", "--user", "is-active", "reachy-shell"], capture_output=True, text=True)
    return r.stdout.strip() == "active"


def default_devices() -> tuple[str, str]:
    out = subprocess.run(["wpctl", "status"], capture_output=True, text=True).stdout

    def one(section: str) -> str:
        lines = out.split(f"{section}\n", 1)[-1].splitlines() if section in out else []
        for ln in lines:
            # wpctl draws the selected item as `* 53.  Name  [vol: ...]` behind a box
            # character - that is why this returned "(none)" on a machine whose route
            # was correct, and a bench that misreads the route tells you the opposite
            # of the truth about interruptions.
            s = ln.replace("\u2502", "").replace("|", "").strip()
            if s.startswith("*"):
                return s.lstrip("* ").strip()
        return "(none)"

    return one("Sinks:"), one("Sources:")


def open_chip() -> tuple[Any, str | None]:
    """Open the XVF control interface: (handle, None) or (None, why_not).

    Vendor control transfers only - the same thing shell/audio/xmos_tuning.py does
    while the shell is live - so this is allowed beside a running shell. It is
    still one handle at a time, and it is always closed. `init_respeaker_usb()`
    returning None is a result: fail soft, never a traceback.
    """
    try:
        from reachy_mini.media.audio_control_utils import init_respeaker_usb
    except Exception as e:  # noqa: BLE001
        return None, f"cannot import audio_control_utils: {e!r}"
    try:
        dev = init_respeaker_usb()
    except Exception as e:  # noqa: BLE001
        return None, f"XVF control interface unavailable ({type(e).__name__}: {e})"
    if dev is None:
        return None, "no XVF3800 on USB (robot off, or the daemon holds the control interface)"
    return dev, None


def read_table(dev: Any, names: tuple[str, ...] = PARAMS) -> dict:
    """The idle read-back table, raw values as the SDK returns them. One handle,
    never a raise: a parameter this firmware does not have is a row that says so."""
    out: dict = {}
    for p in names:
        try:
            out[p] = dev.read(p)
        except Exception as e:  # noqa: BLE001
            out[p] = f"read failed: {type(e).__name__}"
    return out


def close_chip(dev: Any) -> None:
    try:
        dev.close()
    except Exception:  # noqa: BLE001
        pass


def chip_during_reply(seconds: float, db: Path, interval: float = 0.4) -> tuple[dict, list[dict]]:
    """Sample the chip only while the journal says he is audible, plus one idle read.

    The journal is the clock: it is the only thing that can say the far-end was
    active, and reading it costs nothing. Nobody replying in the window leaves
    `during` empty, which summarize_chip reports as a missing measurement.
    """
    dev, err = open_chip()
    if dev is None:
        return {"error": err}, []
    idle: dict = {}
    during: list[dict] = []
    try:
        deadline = time.time() + seconds
        while time.time() < deadline:
            wall = time.time()
            windows = reply_windows(rows_from_db(db, 2.0, now=wall), now=wall)
            got = read_numbers(dev)
            if is_audible(windows, wall):
                during.append({**got, "_wall": wall})
            elif not idle:
                idle = dict(got)
            time.sleep(interval)
    finally:
        close_chip(dev)
    return idle, during


# ------------------------------------------- the demoted capture step (opens the mic)


class Capture:
    def __init__(self) -> None:
        self.bufs: list[np.ndarray] = []

    def cb(self, ind, frames, t, s) -> None:  # noqa: ANN001, ARG002
        self.bufs.append(ind.copy())

    def drain(self) -> np.ndarray:
        if not self.bufs:
            return np.zeros((0, 2), dtype=np.float32)
        out = np.concatenate(self.bufs, axis=0)
        self.bufs.clear()
        return out


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.35 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def read_wav(path: Path) -> np.ndarray:
    """Mono float32 at SR from any wav, with stdlib `wave`.

    sd.read does not exist (that is soundfile) and a bench that cannot load its own
    corpus skips the measurement while still printing a clean run.
    """
    import wave

    with wave.open(str(path), "rb") as r:
        rate, n = r.getframerate(), r.getnframes()
        pcm = np.frombuffer(r.readframes(n), dtype="<i2").reshape(-1, r.getnchannels())
    x = pcm.mean(axis=1).astype(np.float32) if pcm.shape[1] > 1 else pcm[:, 0].astype(np.float32)
    if rate != SR and len(x) > 1:
        idx = np.linspace(0.0, len(x) - 1.0, max(1, int(round(len(x) * SR / rate))))
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return x / 32768.0


def speech_files(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    return sorted(p.rglob("*.wav")) if p.is_dir() else []


def _save(base: Path, name: str, data: np.ndarray) -> None:
    """Write what the mic actually got, with stdlib wave (sd.write does not exist)."""
    try:
        import wave

        out = base if base.suffix == ".wav" else base.parent / f"bench-aec-{name}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
        with wave.open(str(out), "wb") as w:
            w.setnchannels(pcm.shape[1])
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        print(f"  saved {out} (listen: this is what the mic actually got)")
    except Exception as e:  # noqa: BLE001
        print(f"  (could not save {name}: {e})")


def echo_rms_step(args: argparse.Namespace, stats: dict) -> str:
    """The original steps 1 and 2, demoted. Returns "ok", "refused" or "skipped".

    Kept because (a) the captured wavs are the only look at what the mic actually
    received and (b) the number's instability is the finding. It is NOT a
    cancellation measurement and nothing below it may be quoted as one.
    """
    if shell_running() and not args.force:
        print("\n== step 3: tone/RMS capture REFUSED (opens the mic) ==")
        print("reachy-shell is ACTIVE: it owns the mic and the speaker (ADR 0002). Two capture")
        print("owners produce numbers that look fine and mean nothing.")
        print("  systemctl --user stop reachy-shell   # then: bench_aec.py --echo-rms")
        print("  systemctl --user start reachy-shell  # and put him back afterwards")
        print("The self-hearing result above needs none of this and is already answered.")
        return "refused"

    import sounddevice as sd  # lazy: the CI path must import this module with no PortAudio

    print("\n== step 3: tone/RMS capture - NOT a cancellation measurement ==")
    print(
        "  Mic RMS over a loud tone is a FRONT-END-GAIN measurement: PP_AGCMAXGAIN and the\n"
        "  noise suppressor both react to the far-end. Measured x2.7 then x0.5 on identical\n"
        "  input the same night. Read the wavs, not the ratio; the verdict is step 1.\n"
    )
    cap = Capture()
    # sounddevice has no CallbackInputStream - the callback constructor is
    # InputStream. Two channels if the device exposes them, one if not; the channel
    # count is printed because "which column is his voice in" is part of the question.
    try:
        stream = sd.InputStream(samplerate=SR, blocksize=BLOCK, channels=2, callback=cap.cb)
    except ValueError:
        stream = sd.InputStream(samplerate=SR, blocksize=BLOCK, channels=1, callback=cap.cb)
    stats["capture_channels"] = stream.channels
    with stream:
        time.sleep(2.0)
        base = cap.drain()
        base_r = [rms(base[:, i : i + 1]) for i in range(base.shape[1])]
        print(f"  capture channels        {stream.channels}")
        print(f"  room silence rms        L={base_r[0]:.5f}" + (f"  R={base_r[1]:.5f}" if len(base_r) > 1 else ""))
        sd.play(tone(args.tone_seconds), SR)
        time.sleep(args.tone_seconds + 1.0)
        echo = cap.drain()
        echo_r = [rms(echo[:, i : i + 1]) for i in range(echo.shape[1])]
        print(f"  while playing a tone    L={echo_r[0]:.5f}" + (f"  R={echo_r[1]:.5f}" if len(echo_r) > 1 else ""))
        sd.stop()
        rise = echo_r[0] / base_r[0] if base_r[0] > 0 else float("inf")
        print(f"  RMS rise over silence   x{rise:.1f}  (confounded: AGC/DNS moved this, not the AEC)")
        stats["silence_rms"] = base_r
        stats["echo_rms"] = echo_r
        stats["echo_over_silence"] = rise
        stats["echo_rms_step"] = "AGC/DNS-confounded; not a cancellation measurement"
        _save(Path(args.out or "/tmp/bench-aec"), "tone", echo)

        if args.person:
            print("\n== step 3b: a person over him (a LEVEL comparison, not a cancellation test) ==")
            files = speech_files(Path(args.speech_file))
            if not files:
                print(f"  no wav under {args.speech_file}; skipping - play anything at demo volume by hand")
                return "ok"
            clip = np.concatenate([read_wav(f) for f in files[:3]])
            clip = np.resize(clip, int(SR * 15)).astype(np.float32)
            cap.drain()
            sd.play(clip.reshape(-1, 1), SR)
            print("  playing his voice for 15 s - speak a full sentence at second 3 and second 10")
            time.sleep(15.0)
            over = cap.drain()
            sd.stop()
            over_r = rms(over[:, 0]) if over.shape[1] else 0.0
            print(f"  capture while he speaks  L={over_r:.5f}")
            stats["over_playback_rms"] = over_r
            _save(Path(args.out or "/tmp/bench-aec"), "over_playback", over)
            print("  now say the SAME sentence with him silent:")
            if sys.stdin.isatty():
                input("  press enter to record 5 s of capture...")
            else:
                # A non-tty run must never sit on input(): a scripted bench run that
                # hangs is a run that reports nothing at all.
                print("  no tty: counting down 5 s instead of waiting on enter")
                time.sleep(1.0)
            cap.drain()
            time.sleep(5.0)
            quiet = cap.drain()
            quiet_r = rms(quiet[:, 0]) if quiet.shape[1] else 0.0
            ratio = over_r / quiet_r if quiet_r > 0 else 0.0
            print(f"  interrupter level while he plays / when silent: {ratio:.2f}")
            _save(Path(args.out or "/tmp/bench-aec"), "quiet_reference", quiet)
            stats["person_ratio"] = ratio
            print(
                "  The old rule here (>=0.5 means policy, <=0.1 means chip suppression) is\n"
                "  RETRACTED: the same AGC/DNS that moved the tone ratio moves this one, so the\n"
                "  ratio cannot tell double-talk survival from a gain step. It is a level probe.\n"
                "  The decision comes from step 1 (does his own voice reach STT) and from whether\n"
                "  a person's words actually arrive as transcripts - which the journal answers."
            )
    return "ok"


# ------------------------------------------------------------------ self check


def _self_check() -> int:
    """Import + pure decision logic, WITHOUT touching an audio device, a DB, the
    chip or a subprocess. This is what CI is allowed to run."""
    e = 1_791_857_600.0

    def row(off: float, type_: str, **payload: object) -> Row:
        return (e + off, type_, json.dumps({"env": "home", **payload}))

    checks: list[tuple[str, bool]] = [
        (
            "the device import stays inside the step that needs it",
            "sd" not in globals() and "sounddevice" not in globals(),
        ),
        ("numpy imported", "numpy" in sys.modules),
        ("clip_word", clip_word("hmm-70.wav") == "hmm" and clip_word("unknown-1.wav") is None),
        ("resembles a grunt", resembles("Hm.", "hmm") and resembles("Mhm.", "hmm")),
        ("rejects a person", not resembles("Hey Richie what day is it", "hmm")),
        ("rejects another word", not resembles("Yes.", "hmm") and not resembles("Right.", "hmm")),
        ("fuzzy on long words", resembles("Rite.", "right") and not resembles("Write.", "hmm")),
        ("empty transcript never matches", not resembles("", "yes")),
    ]
    rows = [
        row(0.0, TRANSCRIPT_EVENT, text="Hey Richie."),
        row(0.02, ACK_EVENT, sound="yes-60.wav"),
        row(1.4, TRANSCRIPT_EVENT, text="Yes."),
        row(0.0, "attention.lease_started"),
    ]
    sh = self_hearing(rows)
    checks.append(("heard himself", sh["heard"] == 1 and sh["state"] == "self-hearing-observed"))
    checks.append(
        (
            "the wake phrase that caused the ack is at a negative delta and never counts",
            sh["clips"][0]["transcript"] == "Yes." and (sh["clips"][0]["transcript_delta_s"] or 0) > 0,
        )
    )
    dup = [
        row(0.0, ACK_EVENT, sound="yes-60.wav"),
        row(-0.03, TRANSCRIPT_EVENT, text="Hey Richie."),
        row(0.9, TRANSCRIPT_EVENT, text="Hey Richie."),
    ]
    checks.append(("the same turn re-emitted is not self-hearing", self_hearing(dup)["heard"] == 0))
    checks.append(("never exercised is loud", self_hearing([])["state"] == "never-exercised"))
    few = [row(i * 10.0, ACK_EVENT, sound="hmm-70.wav") for i in range(3)]
    checks.append(("a small no is inconclusive", self_hearing(few)["state"] == "inconclusive"))
    many = [row(i * 10.0, ACK_EVENT, sound="hmm-70.wav") for i in range(MIN_ACKS_FOR_A_NO)]
    checks.append(("enough acks to say no", self_hearing(many)["state"] == "no-self-transcription-observed"))
    checks.append(("lease coverage", lease_live(lease_intervals(rows), e + 0.02)))
    win = reply_windows([row(5.0, "response.first_audio"), row(8.0, "response.completed")])
    checks.append(("reply window", bool(win) and is_audible(win, e + 6.0) and not is_audible(win, e + 9.0)))
    checks.append(
        (
            "barge-in proof",
            gate_state([row(1.0, SPEECH_STARTED, cut_playback=True), row(2.0, "response.created")])[
                "gate_open_during_reply"
            ]
            is True,
        )
    )
    checks.append(
        (
            "no barge-in is unproven, not closed",
            gate_state([row(2.0, "response.created")])["gate_open_during_reply"] is None,
        )
    )
    checks.append(
        (
            "first_number decodes",
            first_number((2.0, 3.0)) == 2.0 and first_number(None) is None and first_number("x") is None,
        )
    )

    class FakeDev:
        def read_values(self, name: str):
            return {"AEC_AECCONVERGED": (1.0,), "AEC_NUM_FARENDS": (1.0,), "PP_ECHOONOFF": (1.0,)}.get(name)

        def close(self) -> None:
            pass

    checks.append(
        ("read_numbers via a fake handle", read_numbers(FakeDev(), ("AEC_AECCONVERGED",))["AEC_AECCONVERGED"] == 1.0)
    )
    idle = {"AEC_AECCONVERGED": 0.0, "AEC_NUM_FARENDS": 0.0, "PP_ECHOONOFF": 1.0, "PP_MIN_NS": 0.8}
    during = [{"AEC_AECCONVERGED": 1.0, "AEC_NUM_FARENDS": 1.0, "PP_ECHOONOFF": 1.0, "PP_MIN_NS": 0.8}] * 3
    chip = summarize_chip(idle, during)
    checks.append(("chip reference found mid-reply", chip["reference_present"] is True and chip["samples"] == 3))
    checks.append(("idle-only reads are not a mid-reply result", summarize_chip(idle, [])["samples"] == 0))
    checks.append(("an idle-only chip has no reference verdict", summarize_chip(idle, [])["reference_present"] is None))
    lines = summary_lines(sh, gate_state(rows), chip)
    checks.append(("summary states both decisions", sum(("GATE OPEN" in x or "REACH STT" in x) for x in lines) == 2))
    checks.append(("summary states the limits", lines[-1].startswith("LIMITS")))
    never = summary_lines(self_hearing([]), gate_state(rows), chip)
    checks.append(("never-exercised says it is not a pass", any("NOT A PASS" in x for x in never)))
    skipped = {"skipped": True, "samples": 0, "reference_present": None, "params": {}, "verdict": []}
    checks.append(
        (
            "--self-hear wording for an unread chip",
            any("not read this run" in x for x in summary_lines(sh, gate_state(rows), skipped)),
        )
    )

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print(
        f"self-check: {len(checks) - len(failed)}/{len(checks)} pure checks passed (no audio device, no journal, no chip)"
    )
    return 1 if failed else 0


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure whether Reachy hears himself (issue #40).")
    ap.add_argument(
        "--self-hear",
        action="store_true",
        help="the self-hearing question alone: journal only, and skip the idle chip read-back",
    )
    ap.add_argument("--minutes", type=float, default=60.0, help="journal window for the self-hearing pass")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="the event journal, opened read-only")
    ap.add_argument(
        "--match-window", type=float, default=MATCH_WINDOW_S, help="seconds after an ack in which a transcript counts"
    )
    ap.add_argument(
        "--chip-during-reply",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="sample the XVF only while the journal says he is audible",
    )
    ap.add_argument("--echo-rms", action="store_true", help="the demoted tone/RMS step (opens the mic)")
    ap.add_argument("--person", action="store_true", help="with the capture step: also talk over him")
    ap.add_argument("--tone-seconds", type=float, default=4.0)
    ap.add_argument(
        "--speech-file", default="sounds/acks/obama", help="wav file or dir of wavs to play as 'his' speech"
    )
    ap.add_argument("--out", default="", help="write the numbers as JSON here")
    ap.add_argument(
        "--force",
        action="store_true",
        help="run the capture step anyway with reachy-shell active (numbers are then meaningless)",
    )
    ap.add_argument(
        "--self-check", action="store_true", help="import + pure logic only: no device, no DB, no chip (the CI path)"
    )
    args = ap.parse_args(argv)

    if args.self_check:
        print("== self-check: pure decision logic, no audio device opened ==")
        return _self_check()

    print("== step 0: audio route (read-only) ==")
    sink, source = default_devices()
    print(f"  sink:   {sink}")
    print(f"  source: {source}")
    route_ok = "Reachy Mini" in sink and "Reachy Mini" in source
    if not route_ok:
        print("  NO. Playback is not landing on the Reachy card, so the chip gets no far-end")
        print("  reference on this route and every plan above changes. Fix the defaults first;")
        print("  at the office this is the single most likely cause of 'he can't be interrupted'.")
    print("  Name, never index: `amixer -c N` is a card NUMBER and this box moved the Reachy")
    print("  card from card 1 to card 3 across a power cycle, so an index is wrong by accident.")
    stats: dict = {"route": {"sink": sink, "source": source, "both_reachy": route_ok}}

    print(f"\n== step 1: SELF-HEARING (journal only - this script opened no audio device) ==")
    rows = rows_from_db(Path(args.db), args.minutes)
    if not rows:
        print(f"  nothing in {args.db} in the last {args.minutes:g} min")
    sh = self_hearing(rows, match_window_s=args.match_window)
    gate = gate_state(rows)
    print(
        f"  {sh['acks']} acks played by the shell, {sh['judged']} judgeable, {sh['heard']} heard back -> {sh['state']}"
    )
    for c in sh["clips"][-12:]:
        tag = "HEARD-HIMSELF" if c["heard_himself"] else ("other-speech " if c["transcript"] else "silent       ")
        delta = "   --  " if c["transcript_delta_s"] is None else f"+{c['transcript_delta_s']:5.2f}s "
        print(
            f"    {c['clip']:14s} word={str(c['word']):6s} {tag} lease={'live' if c['lease_live'] else 'idle'} "
            f"{delta}{c['transcript']!r}"
        )
    print("  (a transcript BEFORE the ack is the wake phrase that caused it, never self-hearing)")
    stats["self_hearing"] = sh
    stats["gate"] = gate

    print("\n== step 2: chip read-back (idle) ==")
    if args.self_hear and args.chip_during_reply <= 0:
        print("  skipped by --self-hear: the idle read-back answers nothing about playback, and")
        print("  this asks no USB handle at all. Add --chip-during-reply to sample a live far-end.")
        chip: dict = {"skipped": True, "samples": 0, "reference_present": None, "params": {}, "verdict": []}
        stats["chip_read"] = "skipped by --self-hear"
    else:
        dev, err = open_chip()
        if dev is None:
            print(f"  {err}")
            print("  Not a crash and not a verdict: the AEC question stays open without this.")
            chip = {
                "error": err or "chip unavailable",
                "samples": 0,
                "reference_present": None,
                "params": {},
                "verdict": [],
            }
            stats["chip_error"] = chip["error"]
        else:
            try:
                table = read_table(dev)
                idle = read_numbers(dev)
            finally:
                close_chip(dev)
            stats["params"] = {k: repr(v) for k, v in table.items()}
            for k in REPLY_PARAMS + ("VERSION", "AEC_RT60", "AUDIO_MGR_SYS_DELAY", "AEC_FIXEDBEAMSONOFF"):
                print(f"  {k:24s} {table.get(k)}")
            chip = summarize_chip(idle, [])
            print("  Every value above was read while he was IDLE. AECCONVERGED here says nothing")
            print("  about playback; --chip-during-reply is the only way to read it with a live far-end.")

    if args.chip_during_reply > 0:
        print(f"\n== step 2b: chip sampled while the journal says he is audible ({args.chip_during_reply:g} s) ==")
        idle2, during = chip_during_reply(args.chip_during_reply, Path(args.db))
        chip = summarize_chip(idle2, during)
        for line in chip["verdict"]:
            print(line)
        stats["chip_during_reply"] = {
            "idle": idle2,
            "during": during,
            "samples": chip["samples"],
            "reference_present": chip["reference_present"],
        }

    if args.echo_rms or args.person:
        if echo_rms_step(args, stats) == "refused":
            print("\n== summary skipped: the capture step refused, nothing new was measured ==")
            return 2

    print("\n== summary ==")
    lines = summary_lines(sh, gate, chip)
    for line in lines:
        print(line)
    stats["summary"] = lines

    if args.out:
        Path(args.out).write_text(json.dumps(stats, indent=2, default=str))
        print(f"\nwrote {args.out}")
    # A run that measured nothing exits non-zero: an empty self-hearing section is
    # not a pass, and a script run from CI or a checklist must be able to tell.
    return 1 if sh["state"] == "never-exercised" else 0


if __name__ == "__main__":
    sys.exit(main())
