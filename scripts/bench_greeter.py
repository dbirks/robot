#!/usr/bin/env python3
"""Turn the eye-contact greeter's first live night into numbers, not impressions.

Tomorrow a person stands in front of the robot, and the only record will be the
journal. This reads it and answers the questions that actually decide whether B3 is
ready: how many hellos fired, what the dwell distribution really was, every spacing
violation, whether the per-hour cap bound, how long the camera poll ran and what it
would cost (ADR 0006 on a 4-core box that is doing speech end to end), and the one
that must never be true — whether he greeted someone in the middle of a conversation
the attention lease owned.

Read-only by construction: `mode=ro` on the SQLite file, stdlib only, no device, no
model, no camera, no network. Safe to run while reachy-shell is live.

    scripts/bench_greeter.py --minutes 90              # judge the historical journal
    scripts/bench_greeter.py --watch                   # one line per decision, live
    scripts/bench_greeter.py --watch --max-seconds 300 # ... bounded
    scripts/bench_greeter.py --minutes 0               # whole journal

The limits are never literals here. They come from the resolved room config, the same
JSON the shell boots with:

    REACHY_ENV=hackday python -m shell --print-env

which is the only honest source — `behaviour.greeter_cooldown_s` in envs/hackday.yaml
lands on `greeter_min_spacing_s` (shell/env.py:191), any of them can be overridden by
REACHY_GREETER_* in the process env, and the fallback defaults are used only when that
subprocess is unreachable (the report then says so on its own line).

What the greeter journals (shell/vision/greeter.py, shell/journal.py:77-80):
`face.present` once when a track first qualifies, `greeter.fired`
(kind/track/dwell_s/azimuth_deg/area_share) on every hello, `greeter.suppressed`
(reason/track/dwell_s) at most once per 10 s, and `camera.disabled` when the one frame
owner handed the device back.

Two gaps this bench refuses to paper over, both printed in the report:

  * EFFECTIVE FPS AND CAMERA UPTIME ARE NOT JOURNALLED. The greeter journals
    decisions, not frames; `camera.disabled` is the only camera row. So the fps line is
    the *projected* duty cycle (configured cadence x the ms/frame measured in
    docs/pins.yaml `face_detection`, whose own text says per-sample `cap.read()` and
    the UVC warm-up are UNMEASURED PENDING BENCH), and the run spans below are lower
    bounds on the poll, not measurements of it. The real number needs a frame counter.
  * `mono` is CLOCK_MONOTONIC relative to the writing process (shell/journal.py:132),
    so it is only compared inside one run. Runs are split where `mono` goes backwards;
    wall-clock is what the timeline and the spacing/hour windows use.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO / "data" / "events.db"

FIRED = "greeter.fired"
SUPPRESSED = "greeter.suppressed"
FACE = "face.present"
CAMERA = "camera.disabled"
LEASE_STARTED = "attention.lease_started"
LEASE_RENEWED = "attention.lease_renewed"
LEASE_EXPIRED = "attention.lease_expired"
GREETER_TYPES = (FIRED, SUPPRESSED, FACE, CAMERA)
# attention.lease_transferred is deliberately absent: acquire() writes it and then
# lease_started in the same call (shell/attention/lease.py:55-62), so the started row
# already carries the instant the new holder took the lease.
LEASE_TYPES = (LEASE_STARTED, LEASE_RENEWED, LEASE_EXPIRED)
WATCH_TYPES = GREETER_TYPES + LEASE_TYPES

HOUR_S = 3600.0
CORES = 4  # i7-6700K; ADR 0006: "the machine is 4 cores. Everything below is measured on it"
# docs/pins.yaml `face_detection`, measured on this box 2026-10-12 (detection-only,
# 320 px, 2 intra-op threads). Used only when the file cannot be read.
PINS_MS_FALLBACK = (2.87, 3.7)

# Reasons that reach the journal, and reasons that exist only in the debug log.
# `Greeter.sample()` returns `disabled` / `asleep` / `no-camera` / `nobody-looking`
# before the verdict is ever computed, so those four never produce a row: an empty
# room and a dead camera are indistinguishable in the database. Saying so is part of
# the bench, not a footnote.
JOURNALED_REASONS = ("speaking", "hourly-cap", "presence-greeted", "spacing", "conversation", "group-spacing")
LOG_ONLY_REASONS = ("disabled", "asleep", "no-camera", "nobody-looking")
WHY = {
    "speaking": "he is mid-sentence: never talks over his own voice",
    "hourly-cap": "the per-hour ceiling is spent",
    "presence-greeted": "this presence already got its hello",
    "spacing": "another greeting is still inside the min-spacing window",
    "conversation": "lease held and fewer than two un-greeted faces in frame",
    "group-spacing": "the in-conversation arrival notice is still inside its window",
    "delivery-failed": "the cue never reached the realtime client",
}


@dataclass(frozen=True)
class Event:
    """One journal row with the payload decoded. The tests hand-build these."""

    id: int
    mono: float
    wall: float
    type: str
    payload: dict = field(default_factory=dict)

    @property
    def env(self) -> str:
        return str(self.payload.get("env") or "")

    def num(self, key: str) -> float | None:
        try:
            return float(self.payload.get(key))
        except (TypeError, ValueError):
            return None

    def text(self, key: str, default: str = "?") -> str:
        value = self.payload.get(key)
        return default if value is None else str(value)

    def hm(self) -> str:
        return datetime.fromtimestamp(self.wall).strftime("%H:%M:%S")


@dataclass(frozen=True)
class Limits:
    """The gate as this room resolved it — the only thing the report judges against."""

    enabled: bool
    dwell_s: float
    min_spacing_s: float
    group_min_spacing_s: float
    max_per_hour: int
    idle_fps: float
    engaged_fps: float
    face_centering: bool
    debug: bool
    source: str

    @property
    def fallback(self) -> bool:
        return self.source.startswith("fallback")


# shell/config.py:194-211, for when --print-env cannot be run at all.
FALLBACK_LIMITS = Limits(
    enabled=False,
    dwell_s=2.0,
    min_spacing_s=120.0,
    group_min_spacing_s=300.0,
    max_per_hour=6,
    idle_fps=2.0,
    engaged_fps=3.0,
    face_centering=False,
    debug=False,
    source="fallback: shell/config.py defaults (could not run --print-env)",
)

_COERCE = {
    "greeter_enabled": ("enabled", bool),
    "greeter_dwell_s": ("dwell_s", float),
    "greeter_min_spacing_s": ("min_spacing_s", float),
    "greeter_group_min_spacing_s": ("group_min_spacing_s", float),
    "greeter_max_per_hour": ("max_per_hour", int),
    "greeter_idle_fps": ("idle_fps", float),
    "greeter_engaged_fps": ("engaged_fps", float),
    "greeter_face_centering": ("face_centering", bool),
    "greeter_debug": ("debug", bool),
}


def resolve_limits(env: str, run=subprocess.run) -> Limits:
    """Ask the shell which room this is instead of restating its config.

    `--print-env` resolves env file + process env + defaults exactly as boot does, and
    opens nothing (shell/__main__.py:510-516). Tried under this interpreter first, then
    under `uv run`, so the bench works both from `uv run scripts/bench_greeter.py` and
    from a bare shell.
    """
    environ = {**os.environ, "REACHY_ENV": env}
    candidates = (
        [sys.executable, "-m", "shell", "--print-env"],
        ["uv", "run", "python", "-m", "shell", "--print-env"],
    )
    for cmd in candidates:
        try:
            proc = run(cmd, cwd=REPO, env=environ, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0 or not proc.stdout.strip():
            continue
        try:
            printed = json.loads(proc.stdout)
        except json.JSONDecodeError:
            continue
        if not all(key in printed for key in _COERCE):
            continue  # an older/newer config surface: refuse to guess the limits
        resolved = {name: cast(printed[key]) for key, (name, cast) in _COERCE.items()}
        resolved["source"] = f"REACHY_ENV={env} python -m shell --print-env"
        return Limits(**resolved)
    return FALLBACK_LIMITS


def pins_ms_per_frame(pins_path: Path = REPO / "docs" / "pins.yaml") -> tuple[float, float, str]:
    """(p50, p95) detect-only ms/frame, read out of docs/pins.yaml so the projection
    cannot drift from the measurement. A regex over one block rather than a YAML parse:
    this script stays stdlib-only (CI runs it without pyyaml)."""
    try:
        text = pins_path.read_text()
    except OSError:
        return (*PINS_MS_FALLBACK, "docs/pins.yaml unreadable (fallback constants)")
    body = text.split("face_detection:", 1)[-1].split("face_detection_alt:", 1)[0]
    p50 = re.search(r"p50 ([\d.]+) ms", body)
    p95 = re.search(r"p95 ([\d.]+) ms", body)
    if not (p50 and p95):
        return (*PINS_MS_FALLBACK, "docs/pins.yaml face_detection: no p50/p95 ms line (fallback constants)")
    return float(p50.group(1)), float(p95.group(1)), f"docs/pins.yaml face_detection ({pins_path})"


# ---- reading the journal (read-only, always) ----


def _payload(raw: object) -> dict:
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise SystemExit(f"no journal at {db_path} (run the shell once, or pass --db)")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _rows(db_path: Path, types: Sequence[str], after_id: int = 0, since_wall: float | None = None) -> list[Event]:
    placeholders = ",".join("?" * len(types))
    sql = f"select id, mono, wall, type, payload from events where type in ({placeholders})"
    args: list = list(types)
    if after_id:
        sql += " and id > ?"
        args.append(after_id)
    if since_wall is not None:
        sql += " and wall > ?"
        args.append(since_wall)
    sql += " order by id"
    db = _connect(db_path)
    try:
        rows = db.execute(sql, args).fetchall()
    finally:
        db.close()
    return [Event(int(i), float(m), float(w), t, _payload(p)) for i, m, w, t, p in rows]


def load(db_path: Path, minutes: float | None = None, types: Sequence[str] = WATCH_TYPES) -> list[Event]:
    """Rows of `types` in id order, optionally only the last `minutes` (None = all)."""
    since = None if not minutes or minutes <= 0 else time.time() - minutes * 60
    return _rows(db_path, types, since_wall=since)


def last_row_id(db_path: Path) -> int:
    """Where to start watching: the newest row written so far (history is not reprinted)."""
    db = _connect(db_path)
    try:
        row = db.execute("select max(id) from events").fetchone()
    finally:
        db.close()
    return int(row[0] or 0)


def journal_totals(db_path: Path) -> dict:
    """Whole-DB counts, because "has the greeter EVER run" is not a windowed question.
    ~89k of the older rows carry no env stamp, so stamping is not usable as a filter."""
    placeholders = ",".join("?" * len(GREETER_TYPES))
    db = _connect(db_path)
    try:
        total = db.execute("select count(*) from events").fetchone()[0]
        greeter = db.execute(f"select count(*) from events where type in ({placeholders})", GREETER_TYPES).fetchone()[0]
        unstamped = db.execute(
            f"select count(*) from events where type in ({placeholders}) and payload not like '%\"env\"%'",
            GREETER_TYPES,
        ).fetchone()[0]
    finally:
        db.close()
    return {"rows_total": int(total), "greeter_rows_ever": int(greeter), "greeter_rows_unstamped": int(unstamped)}


# ---- the attention lease, rebuilt from its rows ----


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    holder: str
    tight: bool  # False: the lease_started row aged out, so `start` is only a lower bound

    def covers(self, mono: float) -> bool:
        return self.start <= mono < self.end

    def label(self) -> str:
        return "" if self.tight else " (start inferred: lease rows trimmed by retention)"


class LeaseWindows:
    """`lease.active()` as the greeter saw it, rebuilt from the lease rows.

    The suppressor is `clock() < _expires` (shell/__main__.py:313 ->
    shell/attention/lease.py:38-40) and expiry is lazy: `attention.lease_expired` is
    written by whatever calls `expire_if_due()` next (lease.py:112-118), possibly long
    after the lease actually lapsed. So a window ends at the deadline the payload names
    (`lease_started.base_s`, `lease_renewed.remaining_s`), never at the expired row —
    ending it there would condemn every greeting in the quiet gap after a conversation,
    which is precisely the false alarm this bench must not produce. `mono` is the unit,
    because the greeter's own `clock` is the same monotonic clock in the same process.
    """

    def __init__(self) -> None:
        self.intervals: list[Interval] = []
        self._open: Interval | None = None
        self._last_mono = float("-inf")

    def feed(self, ev: Event) -> None:
        if ev.mono < self._last_mono:  # a new process: mono is not comparable across it
            self.reset()
        self._last_mono = ev.mono
        if ev.type == LEASE_STARTED:
            self._close(ev.mono)
            self._open = Interval(ev.mono, ev.mono + (ev.num("base_s") or 0.0), ev.text("holder"), True)
        elif ev.type == LEASE_RENEWED:
            remaining = ev.num("remaining_s") or 0.0
            if self._open is None:
                self._open = Interval(ev.mono, ev.mono + remaining, ev.text("holder"), False)
            else:
                self._open = Interval(
                    self._open.start,
                    max(self._open.end, ev.mono + remaining),
                    self._open.holder,
                    self._open.tight,
                )
        elif ev.type == LEASE_EXPIRED:
            self._close(ev.mono)

    def _close(self, now: float) -> None:
        if self._open is None:
            return
        start, holder, tight = self._open.start, self._open.holder, self._open.tight
        self.intervals.append(Interval(start, max(start, min(self._open.end, now)), holder, tight))
        self._open = None

    def finalize(self) -> None:
        """Commit a grant that was still open when the rows stop, keeping its deadline.

        Not clamped to the last row: the greeter can only have fired before the process
        died, so a deadline past the last row covers no extra greeting, while clamping
        here would cut the window short exactly often enough to miss the one break this
        bench exists to catch (a hello a few seconds after the last renewal row).
        """
        if self._open is None:
            return
        self.intervals.append(self._open)
        self._open = None

    def reset(self) -> None:
        self.intervals = []
        self._open = None
        self._last_mono = float("-inf")

    def covering(self, mono: float) -> Interval | None:
        return next((i for i in self.intervals if i.covers(mono)), None)

    def live_at(self, mono: float) -> Interval | None:
        """As `covering()`, but also honours a grant still open when the poll read it.
        The deadline the payload names still bounds it: a lease that lapsed without
        writing its expired row is not a conversation."""
        if self._open is not None and self._open.covers(mono):
            return self._open
        return self.covering(mono)


# ---- the analysis (pure: no DB, no clock, no camera) ----


@dataclass(frozen=True)
class Fire:
    ev: Event
    run: int
    kind: str
    dwell_s: float | None
    lease: Interval | None


@dataclass(frozen=True)
class Gap:
    gap_s: float
    earlier: Event
    later: Event
    window: str  # "min_spacing" | "group_min_spacing"
    limit_s: float
    across_restart: bool


@dataclass(frozen=True)
class RunWindow:
    index: int
    first: Event
    last: Event
    greeter_rows: int
    faces: int
    camera_disabled: int
    lease_windows: int
    lease_seconds: float
    span_s: float


@dataclass
class Report:
    limits: Limits
    minutes: float | None
    rows_total: int
    greeter_rows_ever: int
    greeter_rows_unstamped: int
    window_rows: int
    runs: list[RunWindow]
    fires: list[Fire]
    dwell_fired: list[float]
    dwell_near_miss: list[float]
    suppressed: Counter
    spacing: list[Gap]
    group_spacing: list[Gap]
    across_restart: list[Gap]
    cap_suppressed: int
    busiest_hour: int
    cap_exceeded: list[Fire]
    solo_in_lease: list[Fire]
    group_in_lease: list[Fire]
    dwell_below_gate: list[Fire]
    envs: Counter

    @property
    def never_exercised(self) -> bool:
        """No greeter-family row has ever been written. This is not a pass."""
        return self.greeter_rows_ever == 0


def split_runs(events: Iterable[Event]) -> list[list[Event]]:
    """Group rows by writing process. `mono` restarts near zero when a new shell boots
    (shell/journal.py:132), so mono going backwards is the boundary."""
    runs: list[list[Event]] = []
    previous = float("-inf")
    for ev in events:
        if ev.mono < previous or not runs:
            runs.append([])
        runs[-1].append(ev)
        previous = ev.mono
    return runs


def stats(values: Sequence[float]) -> dict[str, float]:
    """n/min/p50/p95/max — the dwell question is "how long does a real person actually
    hold eye contact", so the distribution is the answer, not an average."""
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def pct(q: float) -> float:
        return ordered[min(n - 1, int(round(q * (n - 1))))]

    return {"n": float(n), "min": ordered[0], "p50": pct(0.50), "p95": pct(0.95), "max": ordered[-1]}


def analyze(
    events: Sequence[Event], limits: Limits, *, minutes: float | None = None, totals: dict | None = None
) -> Report:
    """Judge journal rows against the resolved limits. `events` must be in id order and
    may mix greeter and lease rows; anything else is ignored."""
    in_window = [ev for ev in events if ev.type in WATCH_TYPES]
    if totals is None:
        totals = {
            "rows_total": 0,
            "greeter_rows_ever": sum(1 for ev in in_window if ev.type in GREETER_TYPES),
            "greeter_rows_unstamped": sum(1 for ev in in_window if ev.type in GREETER_TYPES and not ev.env),
        }
    fires: list[Fire] = []
    near_miss: list[float] = []
    suppressed: Counter = Counter()
    envs: Counter = Counter()
    runs: list[RunWindow] = []

    for index, run in enumerate(split_runs(in_window)):
        windows = LeaseWindows()
        for ev in run:  # pass 1: the lease timeline is closed before any greeting is judged
            windows.feed(ev)
        windows.finalize()
        rows = faces = camera = 0
        for ev in run:
            if ev.type not in GREETER_TYPES:
                continue
            rows += 1
            envs[ev.env or "(no env stamp)"] += 1
            if ev.type == FACE:
                faces += 1
            elif ev.type == CAMERA:
                camera += 1
            elif ev.type == SUPPRESSED:
                suppressed[ev.text("reason")] += 1
                dwell = ev.num("dwell_s")
                if dwell is not None:
                    near_miss.append(dwell)
            elif ev.type == FIRED:
                fires.append(
                    Fire(
                        ev=ev,
                        run=index,
                        kind=ev.text("kind"),
                        dwell_s=ev.num("dwell_s"),
                        lease=windows.covering(ev.mono),
                    )
                )
        runs.append(
            RunWindow(
                index=index,
                first=run[0],
                last=run[-1],
                greeter_rows=rows,
                faces=faces,
                camera_disabled=camera,
                lease_windows=len(windows.intervals),
                lease_seconds=round(sum(i.end - i.start for i in windows.intervals), 1),
                span_s=run[-1].mono - run[0].mono,
            )
        )

    spacing, group_spacing, across = find_gaps(fires, limits)
    busiest, exceeded = hour_load(fires, limits)
    return Report(
        limits=limits,
        minutes=minutes,
        rows_total=int(totals.get("rows_total", 0)),
        greeter_rows_ever=int(totals.get("greeter_rows_ever", 0)),
        greeter_rows_unstamped=int(totals.get("greeter_rows_unstamped", 0)),
        window_rows=len(in_window),
        runs=runs,
        fires=fires,
        dwell_fired=[f.dwell_s for f in fires if f.dwell_s is not None],
        dwell_near_miss=near_miss,
        suppressed=suppressed,
        spacing=spacing,
        group_spacing=group_spacing,
        across_restart=across,
        cap_suppressed=suppressed.get("hourly-cap", 0),
        busiest_hour=busiest,
        cap_exceeded=exceeded,
        solo_in_lease=[f for f in fires if f.lease is not None and f.kind == "solo"],
        group_in_lease=[f for f in fires if f.lease is not None and f.kind != "solo"],
        dwell_below_gate=[f for f in fires if f.dwell_s is not None and f.dwell_s < limits.dwell_s],
        envs=envs,
    )


def find_gaps(fires: Sequence[Fire], limits: Limits) -> tuple[list[Gap], list[Gap], list[Gap]]:
    """Every spacing violation in the room, plus the one class that is not a violation.

    Two windows, both from `_verdict`: the global one (`now - _last_any < min_spacing_s`)
    covers every fire of either kind, and group notices additionally owe
    `group_min_spacing_s` to the previous GROUP notice (`_last_group`, greeter.py:213).
    Wall time is the unit — it is comparable across a restart and a 2-minute window does
    not care about clock discipline at that scale.

    `_last_any` is an instance attribute (greeter.py:127), so the first greeting after a
    restart cannot respect a window the dead process was holding. Those pairs are
    reported as `across_restart`, not as failures.
    """
    bad: list[Gap] = []
    group_bad: list[Gap] = []
    restart: list[Gap] = []
    ordered = sorted(fires, key=lambda f: f.ev.wall)
    pairs = [
        (before, after, limits.min_spacing_s, "min_spacing")
        for before, after in zip(ordered, ordered[1:], strict=False)
    ]
    groups = [f for f in ordered if f.kind != "solo"]
    pairs += [
        (before, after, limits.group_min_spacing_s, "group_min_spacing")
        for before, after in zip(groups, groups[1:], strict=False)
    ]
    for before, after, limit_s, window in pairs:
        gap = after.ev.wall - before.ev.wall
        if gap >= limit_s:
            continue
        item = Gap(
            gap_s=gap,
            earlier=before.ev,
            later=after.ev,
            window=window,
            limit_s=limit_s,
            across_restart=before.run != after.run,
        )
        if item.across_restart:
            restart.append(item)
        elif window == "group_min_spacing":
            group_bad.append(item)
        else:
            bad.append(item)
    return bad, group_bad, restart


def hour_load(fires: Sequence[Fire], limits: Limits) -> tuple[int, list[Fire]]:
    """(busiest sliding 60-min window, fires that landed in an already-full hour).

    The cap prunes its deque at 3600 s and compares `len >= max_per_hour`
    (greeter.py:206-210), so the same sliding window is the right test — per process,
    on `mono`, because `_fired` dies with the process: two runs of six fires inside one
    wall-clock hour are two hours of a full room, not a broken cap.

    Anything over the ceiling is a gate bug, not a crowded room: `_fired` only grows
    inside `_fire`, so a fire above the cap means the cap was not consulted for it.
    """
    busiest = 0
    exceeded: list[Fire] = []
    by_run: dict[int, list[Fire]] = {}
    for fire in sorted(fires, key=lambda f: f.ev.mono):
        by_run.setdefault(fire.run, []).append(fire)
    for run_fires in by_run.values():
        marks = [f.ev.mono for f in run_fires]
        for fire in run_fires:
            inside = sum(1 for m in marks if fire.ev.mono - HOUR_S < m <= fire.ev.mono)
            busiest = max(busiest, inside)
            if inside > limits.max_per_hour:
                exceeded.append(fire)
    return busiest, exceeded


# ---- the report ----


def _fmt(stats_: dict[str, float]) -> str:
    if not stats_:
        return "n=0 (nothing to measure)"
    return (
        f"n={stats_['n']:.0f} min={stats_['min']:.2f}s p50={stats_['p50']:.2f}s "
        f"p95={stats_['p95']:.2f}s max={stats_['max']:.2f}s"
    )


def cpu_projection(limits: Limits, ms_p50: float, ms_p95: float) -> str:
    """Projected poll cost at the configured cadence. NOT a measurement — see the
    docstring: the journal carries no frame counter.

    ms of thread time per second of wall time is one tenth of a percent of a core, so
    2 FPS x 2.87 ms is 5.7 ms/s = 0.57% of one core. That is the arithmetic
    docs/pins.yaml `face_detection` records ("~5.7 ms/s and ~8.9 ms/s of thread time,
    ~0.6-0.9% of one core"), and this function reproducing it is the check that the
    projection and that measurement still agree.
    """
    idle = limits.idle_fps * ms_p50
    engaged = limits.engaged_fps * ms_p95
    return (
        f"{limits.idle_fps:g} fps idle / {limits.engaged_fps:g} fps engaged x {ms_p50:.2f}-{ms_p95:.2f} ms/frame "
        f"= {idle:.1f}-{engaged:.1f} ms/s = {idle / 10:.2f}-{engaged / 10:.2f}% of one core "
        f"({idle / 10 / CORES:.2f}-{engaged / 10 / CORES:.2f}% of the {CORES}-core box); inference only — "
        "cap.read() and the UVC warm-up are UNMEASURED PENDING BENCH (docs/pins.yaml)"
    )


def format_report(r: Report, ms: tuple[float, float, str]) -> list[str]:
    limits = r.limits
    out: list[str] = []
    a = out.append
    a(f"greeter bench - limits from {limits.source}")
    a(
        f"  greeter_enabled={limits.enabled} dwell>={limits.dwell_s:g}s spacing>={limits.min_spacing_s:g}s "
        f"group_spacing>={limits.group_min_spacing_s:g}s cap<={limits.max_per_hour}/h "
        f"fps={limits.idle_fps:g}/{limits.engaged_fps:g} "
        f"face_centering={'on' if limits.face_centering else 'off'} debug={'on' if limits.debug else 'off'}"
    )
    window = f"last {r.minutes:g} min" if r.minutes and r.minutes > 0 else "whole journal"
    a(
        f"  journal rows total={r.rows_total}; window={window} rows={r.window_rows}; greeter-family rows ever={r.greeter_rows_ever}"
    )
    if r.never_exercised:
        out.extend(never_exercised_lines(r, limits))
        return out
    if limits.fallback:
        a("  !! LIMITS ARE FALLBACK DEFAULTS: what this room booted with may differ from every verdict below")
    if r.greeter_rows_unstamped:
        a(f"  !! {r.greeter_rows_unstamped} greeter rows predate env stamping: they cannot be attributed to a room")
    if r.envs:
        a("  rows per room: " + ", ".join(f"{k}={v}" for k, v in sorted(r.envs.items())))

    a("")
    a(f"GREETINGS FIRED: {len(r.fires)}")
    a(f"  solo={sum(1 for f in r.fires if f.kind == 'solo')} group={sum(1 for f in r.fires if f.kind != 'solo')}")
    a(f"  dwell at fire: {_fmt(stats(r.dwell_fired))}")
    a(f"  dwell at suppression (near-misses): {_fmt(stats(r.dwell_near_miss))}")
    a("  suppressed by reason: " + (", ".join(f"{k}={v}" for k, v in sorted(r.suppressed.items())) or "none"))
    for i, fire in enumerate(r.fires):
        gap = "" if i == 0 else f" gap={fire.ev.wall - r.fires[i - 1].ev.wall:.1f}s"
        lease = " LEASED" if fire.lease is not None else ""
        a(f"  {fire.ev.hm()} {fire.kind:5} track={fire.ev.text('track')} dwell={fire.dwell_s}s{gap}{lease}")

    a("")
    a("CAMERA / CPU BUDGET (ADR 0006: 4 cores doing speech end to end)")
    if not r.runs:
        a("  no rows at all: nothing to bound")
    for run in r.runs:
        a(
            f"  run {run.index}: {run.first.hm()} -> {run.last.hm()} = {run.span_s / 60.0:.1f} min of journalled "
            f"evidence (a LOWER BOUND on the poll: no frame is journalled); greeter rows={run.greeter_rows}, "
            f"face.present={run.faces}, camera.disabled={run.camera_disabled}, lease windows={run.lease_windows} "
            f"({run.lease_seconds:.0f}s held)"
        )
        if run.camera_disabled:
            a(f"    !! the frame owner handed the device back {run.camera_disabled}x: inference-free after each")
    a("  effective fps: NOT MEASURABLE FROM THE JOURNAL (decisions are journalled, frames are not)")
    a(f"  projected cost: {cpu_projection(limits, ms[0], ms[1])}")
    a(f"    [cost constants from {ms[2]}]")

    a("")
    a("GATES")
    _gate_lines(out, r)
    a("")
    a("LEASE INTERLOCK (ADR 0003: a face buys a hello, never a window into the room's speech)")
    if r.solo_in_lease:
        for fire in r.solo_in_lease:
            lease = fire.lease
            a(
                f"  NEVER-MUST-HAPPEN: solo greeting at {fire.ev.hm()} while holder '{lease.holder}' held the "
                f"lease (mono {lease.start:.1f}..{lease.end:.1f}{lease.label()})"
            )
    else:
        a("  no solo greeting fired inside a live lease.")
    if r.group_in_lease:
        a(
            f"  {len(r.group_in_lease)} group notice(s) fired during a lease: sanctioned by design "
            "(_verdict's in-conversation arrival notice, envs/hackday.yaml behaviour.greeter), judged by the "
            f"{limits.group_min_spacing_s:g}s group window above - not counted as a failure."
        )
    a("")
    failures = hard_failures(r)
    a("VERDICT: " + ("FAIL" if failures else "HELD"))
    for line in failures:
        a(f"  - {line}")
    if not failures:
        a(
            f"  {len(r.fires)} greeting(s) over {len(r.runs)} run(s)"
            + (", and the gate never fired." if not r.fires else ".")
        )
    return out


def _gate_lines(out: list[str], r: Report) -> None:
    a = out.append
    limits = r.limits
    a(f"  spacing violations (< {limits.min_spacing_s:g}s between any two fires, one process): {len(r.spacing)}")
    for gap in r.spacing:
        a(
            f"    FAIL {gap.earlier.hm()} ({gap.earlier.text('kind')}) -> {gap.later.hm()} "
            f"({gap.later.text('kind')}) = {gap.gap_s:.1f}s, under the {gap.limit_s:g}s {gap.window} window"
        )
    a(f"  group-spacing violations (< {limits.group_min_spacing_s:g}s between group notices): {len(r.group_spacing)}")
    for gap in r.group_spacing:
        a(f"    FAIL {gap.earlier.hm()} -> {gap.later.hm()} = {gap.gap_s:.1f}s between two group notices")
    a(f"  fire pairs straddling a restart (expected: _last_any is per process): {len(r.across_restart)}")
    for gap in r.across_restart:
        a(f"    note {gap.earlier.hm()} -> {gap.later.hm()} = {gap.gap_s:.1f}s across a process boundary")
    a(
        f"  hourly cap {limits.max_per_hour}/h: {r.cap_suppressed} fire(s) suppressed by it, busiest 60-min "
        f"window inside one process = {r.busiest_hour} fire(s)"
    )
    if r.cap_exceeded:
        a(f"    FAIL {len(r.cap_exceeded)} fire(s) landed in an already-full hour: the cap did not hold")
        for fire in r.cap_exceeded:
            a(f"    FAIL {fire.ev.hm()} {fire.kind} track={fire.ev.text('track')}")
    a(f"  dwell below the configured gate ({limits.dwell_s:g}s): {len(r.dwell_below_gate)}")
    for fire in r.dwell_below_gate:
        a(f"    FAIL {fire.ev.hm()} fired after {fire.dwell_s:.2f}s")


def hard_failures(r: Report) -> list[str]:
    """What makes tomorrow's run a failure rather than an unflattering measurement."""
    out: list[str] = []
    if r.solo_in_lease:
        out.append(f"{len(r.solo_in_lease)} solo greeting(s) fired while the attention lease was held")
    if r.spacing:
        out.append(f"{len(r.spacing)} min-spacing violation(s) inside one process")
    if r.group_spacing:
        out.append(f"{len(r.group_spacing)} group-spacing violation(s)")
    if r.cap_exceeded:
        out.append(f"hourly cap exceeded on {len(r.cap_exceeded)} fire(s) (busiest window {r.busiest_hour})")
    if r.dwell_below_gate:
        out.append(f"{len(r.dwell_below_gate)} fire(s) below the {r.limits.dwell_s:g}s dwell gate")
    return out


def never_exercised_lines(r: Report, limits: Limits) -> list[str]:
    """The loud answer: there is nothing to bench. Every gate is untested."""
    unstamped = "" if not r.greeter_rows_unstamped else f" ({r.greeter_rows_unstamped} of them predate env stamping)"
    return [
        "",
        "=" * 78,
        f"NEVER EXERCISED - 0 greeter-family rows in the whole journal{unstamped}.",
        f"The greeter has not run once since the journal began ({r.rows_total} events).",
        "This is NOT 'all good'. Dwell, min spacing, group spacing, the hourly cap, the",
        "lease interlock and the camera duty cycle are all unmeasured against a real",
        "person, and the only box they will ever be measured on is this one.",
        "=" * 78,
        "What to watch for tomorrow:",
        "  greeter.fired       kind=solo|group, track, dwell_s, azimuth_deg, area_share.",
        f"                      dwell_s must be >= {limits.dwell_s:g}s; two fires under {limits.min_spacing_s:g}s apart (or two",
        f"                      group notices under {limits.group_min_spacing_s:g}s) is a rate-limiter bug; a solo fire",
        "                      inside a live attention lease is the never-must-happen.",
        "  face.present        one row per newly qualified face (track, azimuth_deg, area_share).",
        "                      None while someone is clearly looking at him = detection or the",
        "                      gaze gate (shell/vision/gaze.py), not the rate limiter.",
        "  greeter.suppressed  reason=why a dwelled face got nothing, at most once per 10 s",
        "                      (greeter._suppress): " + ", ".join(JOURNALED_REASONS) + ",",
        "                      plus delivery-failed (the cue never reached the client).",
        "  camera.disabled     the one frame owner gave /dev/videoN back to the daemon",
        "                      (shell/camera.py:336). After this the poll runs inference-free.",
        "  NOT IN THE JOURNAL, only in the shell log: " + ", ".join(LOG_ONLY_REASONS) + ".",
        "                      sample() returns those before the verdict, so an empty room and a",
        "                      dead camera are indistinguishable in the DB. That is what",
        "                      REACHY_GREETER_DEBUG=1 is for (it logs every track's margins and",
        "                      every suppression: journalctl -u reachy-shell -f).",
        "  REACHY_GREETER_FACE_CENTERING=1  the head-centring hook (greeter._center). Off in both",
        "                      rooms today, and it is the one part of the greeter the journal",
        "                      cannot judge at all - watch the head, not the DB.",
        "  Rooms: greeter and faces are OFF at home (envs/home.yaml:42 greeter: false) and",
        f"  ON in hackday - this run resolved greeter_enabled={limits.enabled} for the room named above.",
        "  Check the room before the first guest, then watch it live:",
        "    REACHY_ENV=hackday python -m shell --print-env | grep greeter",
        "    scripts/bench_greeter.py --watch",
    ]


# ---- watch mode: the gate, live, one line per decision ----


def watch_line(ev: Event, limits: Limits, lease: Interval | None) -> str:
    p = ev.payload
    if ev.type == FIRED:
        kind = ev.text("kind")
        dwell = ev.num("dwell_s")
        line = (
            f"{ev.hm()}  FIRED      {kind:5} track={ev.text('track')} dwell={ev.text('dwell_s')}s "
            f"bearing={ev.text('azimuth_deg')}deg share={ev.text('area_share')}"
        )
        if lease is not None and kind == "solo":
            line += f"\n            !! NEVER-MUST-HAPPEN: holder '{lease.holder}' had the lease -> he spoke over a conversation"
        elif lease is not None:
            line += "   (group notice during a lease: allowed by design)"
        elif dwell is not None and dwell < limits.dwell_s:
            line += f"   !! below the {limits.dwell_s:g}s dwell gate"
        return line
    if ev.type == SUPPRESSED:
        reason = ev.text("reason")
        return (
            f"{ev.hm()}  suppressed   {reason:18} track={ev.text('track')} dwell={ev.text('dwell_s')}s"
            f"  [{WHY.get(reason, reason)}]"
        )
    if ev.type == FACE:
        return (
            f"{ev.hm()}  face         track={ev.text('track')} bearing={ev.text('azimuth_deg')}deg "
            f"share={ev.text('area_share')}  [qualified: the dwell clock is running]"
        )
    return f"{ev.hm()}  CAMERA OFF   {p.get('error') or p}  [the greeter is inference-free from here]"


def watch(
    db_path: Path, limits: Limits, max_seconds: float = 0.0, interval: float = 0.5, clock=time.monotonic
) -> list[Event]:
    """Print one line per greeter decision as the journal grows.

    A fresh `mode=ro` connection per poll is exactly right: the writer commits every
    row on its own (shell/journal.py:_write) and WAL lets readers see them live.
    Returns every row it printed a line for, so the caller can run the same analysis as
    `--minutes` on the watch window.
    """
    read: list[Event] = []
    windows = LeaseWindows()
    last_id = last_row_id(db_path)
    started = clock()
    print(f"watching {db_path} (id>{last_id}) for greeter decisions; Ctrl-C or --max-seconds to stop")
    while True:
        for ev in _rows(db_path, WATCH_TYPES, after_id=last_id):
            last_id = ev.id
            read.append(ev)
            if ev.type in LEASE_TYPES:
                windows.feed(ev)  # feeds the interlock check; no line of its own
                continue
            print(watch_line(ev, limits, windows.live_at(ev.mono)), flush=True)
        if max_seconds and clock() - started >= max_seconds:
            return read
        time.sleep(interval)


# ---- entry point ----


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--minutes", type=float, default=90.0, help="historical window to judge (0 = whole journal)")
    ap.add_argument("--watch", action="store_true", help="print one line per gate decision as it happens")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="with --watch: stop after this long (0 = forever)")
    ap.add_argument("--interval", type=float, default=0.5, help="with --watch: poll interval, seconds")
    ap.add_argument("--env", default="hackday", help="which room's resolved limits to judge against")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="journal file (read-only, never written)")
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    limits = resolve_limits(args.env)
    if args.watch:
        try:
            events = watch(db_path, limits, args.max_seconds, args.interval)
        except KeyboardInterrupt:
            print("\n(stopped)")
            events = []
        observed = sum(1 for ev in events if ev.type in GREETER_TYPES)
        report = analyze(events, limits, minutes=None, totals=journal_totals(db_path))
        print(f"\nwatch window: {observed} greeter-family row(s) of {len(events)} read (lease rows included).")
        for line in format_report(report, pins_ms_per_frame()):
            print(line)
        if observed == 0 and not report.never_exercised:
            print(
                "NOTHING OBSERVED in this watch window. If someone was looking at him, the frame owner or the\n"
                "gaze gate is the suspect, and the reasons for that (no-camera, nobody-looking, asleep, disabled)\n"
                "are log-only: run journalctl -u reachy-shell -f with REACHY_GREETER_DEBUG=1."
            )
        return 1 if (report.never_exercised or observed == 0) else (2 if hard_failures(report) else 0)

    events = load(db_path, minutes=args.minutes)
    report = analyze(events, limits, minutes=args.minutes, totals=journal_totals(db_path))
    for line in format_report(report, pins_ms_per_frame()):
        print(line)
    return 1 if report.never_exercised else (2 if hard_failures(report) else 0)


if __name__ == "__main__":
    sys.exit(main())
