"""The greeter bench's arithmetic, pinned with synthetic rows — no DB, no camera.

Tomorrow's test is a person standing in front of the robot, and scripts/bench_greeter.py
is what turns that into numbers. Every gate it reports on is one of these rows: two
hellos too close together, a seventh hello inside an hour, a hello fired while the
attention lease belonged to somebody mid-conversation. Those three are the ways the
greeter can misbehave in a room, so each is a test here rather than a paragraph in a
docstring — and the fourth (an empty journal) is the one the bench will actually print
tonight, which must read as "never exercised" and never as "all good".

Events are handed to `analyze()` in id order, the way the journal returns them, with
hand-picked `mono`/`wall` pairs: `mono` is the per-process clock the lease windows are
rebuilt on (shell/journal.py:132), `wall` is what the spacing and hour windows use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# scripts/ is not a package: this insert is what makes the import below real at
# runtime, and the only reason ty cannot resolve it (hence its ignore, same line).
sys.path.insert(0, str(ROOT / "scripts"))

import bench_greeter as bg  # ty: ignore[unresolved-import]  # noqa: E402

E = 1_791_857_600.0  # an arbitrary epoch; only deltas matter
LIMITS = bg.Limits(
    enabled=True,
    dwell_s=2.0,
    min_spacing_s=120.0,
    group_min_spacing_s=300.0,
    max_per_hour=6,
    idle_fps=2.0,
    engaged_fps=3.0,
    face_centering=False,
    debug=False,
    source="test: hand-built limits",
)

_id = 0


def ev(mono: float, type_: str, wall: float | None = None, **payload: object) -> bg.Event:
    """One journalled row. `wall` defaults to `mono` on an arbitrary epoch, which is what
    one process looks like; a restart is `mono` going backwards while `wall` keeps
    running, so the two are passed independently."""
    global _id
    _id += 1
    return bg.Event(
        id=_id, mono=mono, wall=E + (mono if wall is None else wall), type=type_, payload={"env": "hackday", **payload}
    )


def face(mono: float, track: int = 1, wall: float | None = None) -> bg.Event:
    return ev(mono, bg.FACE, wall=wall, track=track, azimuth_deg=3.1, area_share=0.04)


def fire(mono: float, kind: str = "solo", dwell: float = 2.4, track: int = 1, wall: float | None = None) -> bg.Event:
    return ev(mono, bg.FIRED, wall=wall, kind=kind, track=track, dwell_s=dwell, azimuth_deg=2.0, area_share=0.03)


def lease(start: float, base_s: float = 30.0, holder: str = "p-1") -> bg.Event:
    return ev(start, bg.LEASE_STARTED, holder=holder, evidence="kws:reachy", base_s=base_s)


def report_of(*events: bg.Event) -> bg.Report:
    return bg.analyze(list(events), LIMITS)


def text_of(report: bg.Report) -> str:
    return "\n".join(bg.format_report(report, bg.pins_ms_per_frame()))


# ---- the empty journal: the case that prints tonight ----


def test_an_empty_journal_reports_never_exercised_rather_than_all_good():
    """Zero greeter rows is the opposite of a pass: nothing was measured."""
    report = report_of()
    assert report.greeter_rows_ever == 0
    assert report.never_exercised is True
    text = text_of(report)
    assert "NEVER EXERCISED" in text
    assert "NOT 'all good'" in text
    assert "VERDICT: HELD" not in text, "an untested greeter printed a clean bill of health"
    assert bg.hard_failures(report) == []  # nothing to fail — and nothing to believe either


def test_the_never_exercised_report_names_the_events_and_the_env_vars_to_watch():
    """Tomorrow's operator reads this block at the robot, so it has to be actionable."""
    text = text_of(report_of())
    for needle in (
        "greeter.fired",
        "greeter.suppressed",
        "face.present",
        "camera.disabled",
        "REACHY_GREETER_DEBUG=1",
        "REACHY_GREETER_FACE_CENTERING=1",
        "greeter: false",  # home has it off...
        "ON in hackday",  # ...and the booth has it on
    ):
        assert needle in text, f"missing from the watch-for list: {needle}"
    # the four reasons that never reach the journal: without this line an empty DB next
    # to a dead camera looks identical to an empty DB next to an empty room
    assert "nobody-looking" in text and "no-camera" in text


def test_rows_that_are_not_greeter_rows_never_count_as_exercise():
    report = report_of(ev(10.0, "audio.health"), ev(11.0, "response.created"), ev(12.0, "turn.transcript_final"))
    assert report.never_exercised is True
    assert report.window_rows == 0


# ---- spacing ----


def test_two_greetings_closer_than_min_spacing_is_a_spacing_violation():
    report = report_of(fire(100.0, track=1), fire(130.0, track=2))
    assert len(report.spacing) == 1
    gap = report.spacing[0]
    assert gap.gap_s == 30.0 and gap.window == "min_spacing" and gap.limit_s == LIMITS.min_spacing_s
    assert gap.across_restart is False
    assert any("min-spacing violation" in line for line in bg.hard_failures(report))
    text = text_of(report)
    assert "FAIL" in text and "30.0s" in text and "NEVER EXERCISED" not in text


def test_greetings_outside_the_window_are_clean():
    report = report_of(fire(100.0), fire(300.0), fire(500.0))
    assert report.spacing == [] and report.group_spacing == [] and report.across_restart == []
    assert bg.hard_failures(report) == []
    assert "VERDICT: HELD" in text_of(report)


def test_a_pair_that_straddles_a_restart_is_not_reported_as_a_violation():
    """`_last_any` is an instance attribute (greeter.py:127): a restarted shell has no
    memory of the window the dead process was holding, so this pair is expected."""
    first = fire(500.0)
    second = fire(10.0, wall=510.0)  # mono restarts near zero: a new writing process
    assert second.mono < first.mono and second.wall > first.wall
    report = report_of(first, second)
    assert len(report.runs) == 2
    assert report.spacing == [], "a restart-straddling pair was blamed on the rate limiter"
    assert len(report.across_restart) == 1
    assert bg.hard_failures(report) == []
    assert "across a process boundary" in text_of(report)


def test_two_group_notices_closer_than_the_group_window_are_a_violation():
    """The group notice is the rarer event (300 s, not 120 s), and it owes its own
    window to the previous group notice: `now - _last_group < group_min_spacing_s`.
    Both halves matter here: 150 s clears the global window, so only the group window
    can be what fails."""
    report = report_of(fire(100.0, kind="group", track=1), fire(250.0, kind="group", track=2))
    assert report.spacing == [], "150s is inside the global window"
    assert len(report.group_spacing) == 1
    assert report.group_spacing[0].window == "group_min_spacing"
    assert report.group_spacing[0].gap_s == 150.0
    assert any("group-spacing" in line for line in bg.hard_failures(report))


# ---- the per-hour ceiling ----


def test_a_full_hour_of_greetings_is_reported_as_the_cap_binding():
    """Six fires, all legal, then a suppression that names the cap: the ceiling worked."""
    events = [face(0.0)]
    events += [fire(m, track=i) for i, m in enumerate((0.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0), start=1)]
    events.append(ev(3100.0, bg.SUPPRESSED, reason="hourly-cap", track=7, dwell_s=2.6))
    report = report_of(*events)
    assert len(report.fires) == 6
    assert report.busiest_hour == 6
    assert report.cap_exceeded == []
    assert report.cap_suppressed == 1
    assert "the cap did not hold" not in text_of(report)
    assert report.suppressed["hourly-cap"] == 1


def test_a_seventh_greeting_inside_the_hour_is_a_cap_failure():
    events = [fire(m, track=i) for i, m in enumerate(range(0, 3501, 500), start=1)]
    report = report_of(*events)
    assert report.busiest_hour == 8
    assert [f.ev.mono for f in report.cap_exceeded] == [3000.0, 3500.0]
    assert any("hourly cap exceeded" in line for line in bg.hard_failures(report))
    assert "the cap did not hold" in text_of(report)


def test_the_cap_is_counted_per_process_not_per_wall_clock_hour():
    """Two runs of six inside one hour is a busy evening, not a broken counter: the
    `_fired` deque died with each process."""
    first = [fire(m, track=i) for i, m in enumerate(range(0, 3001, 600), start=1)]
    second = [fire(m, wall=3600 + m, track=i) for i, m in enumerate(range(0, 3001, 600), start=9)]
    report = report_of(*first, *second)
    assert len(report.runs) == 2
    assert report.busiest_hour == 6
    assert report.cap_exceeded == []
    assert report.spacing == [] and report.across_restart == []
    assert bg.hard_failures(report) == []


def test_near_miss_dwell_values_are_reported_separately_from_fired_ones():
    report = report_of(
        fire(100.0, dwell=2.4),
        ev(200.0, bg.SUPPRESSED, reason="spacing", track=2, dwell_s=9.5),
        ev(400.0, bg.SUPPRESSED, reason="speaking", track=3, dwell_s=5.0),
    )
    assert report.dwell_fired == [2.4]
    assert sorted(report.dwell_near_miss) == [5.0, 9.5]
    assert report.suppressed["spacing"] == 1 and report.suppressed["speaking"] == 1


def test_a_fire_below_the_dwell_gate_is_a_failure():
    """dwell_s is what the greeter measured when it let itself fire; under the
    configured gate the gate is not what fired."""
    report = report_of(fire(100.0, dwell=0.4))
    assert len(report.dwell_below_gate) == 1
    assert any("dwell gate" in line for line in bg.hard_failures(report))


# ---- the lease interlock ----


def test_a_greeting_while_a_lease_is_active_must_be_flagged():
    report = report_of(lease(100.0, base_s=30.0), fire(110.0))
    assert len(report.solo_in_lease) == 1
    assert report.solo_in_lease[0].lease is not None
    assert report.solo_in_lease[0].lease.holder == "p-1"
    failures = bg.hard_failures(report)
    assert any("attention lease" in line for line in failures)
    assert "NEVER-MUST-HAPPEN" in text_of(report)


def test_a_greeting_after_a_renewal_still_counts_as_in_conversation():
    report = report_of(
        lease(100.0, base_s=30.0),
        ev(120.0, bg.LEASE_RENEWED, holder="p-1", evidence="speech_started", remaining_s=20.0),  # live to 140
        fire(130.0),
    )
    assert len(report.solo_in_lease) == 1


def test_a_greeting_after_the_lease_deadline_is_not_flagged_even_when_the_expired_row_is_late():
    """The trap this pins: `attention.lease_expired` is written lazily by whatever calls
    `expire_if_due()` next (lease.py:112-118), long after the lease lapsed. Judging the
    greeting against that row instead of the payload's deadline condemns a hello in an
    empty room."""
    report = report_of(
        lease(100.0, base_s=30.0),  # deadline at 130
        ev(500.0, bg.LEASE_EXPIRED, holder="p-1", held_s=400.0),  # noticed only at 500
        fire(300.0),  # he was well out of the conversation by then
    )
    assert report.solo_in_lease == []
    assert bg.hard_failures(report) == []
    assert report.runs[0].lease_windows == 1
    assert report.runs[0].lease_seconds == 30.0


def test_a_renewed_row_with_no_started_row_still_shields_the_room():
    """Retention can trim the `lease_started` row and leave the renewal. The window is
    then narrower than reality, and the report says the start was inferred."""
    report = report_of(
        ev(120.0, bg.LEASE_RENEWED, holder="p-9", evidence="speech_started", remaining_s=20.0),
        fire(130.0),
    )
    assert len(report.solo_in_lease) == 1
    assert report.solo_in_lease[0].lease.tight is False
    assert "start inferred" in text_of(report)


def test_a_group_notice_during_a_lease_is_the_designed_exception_not_a_failure():
    """`_verdict` returns `fire:group` precisely while a lease is held (the arrival
    notice, envs/hackday.yaml behaviour.greeter). Counting it as a violation would make
    the bench cry wolf on the one behaviour that is supposed to happen."""
    report = report_of(lease(100.0, base_s=30.0), fire(110.0, kind="group"))
    assert report.solo_in_lease == []
    assert len(report.group_in_lease) == 1
    assert bg.hard_failures(report) == []
    text = text_of(report)
    assert "NEVER-MUST-HAPPEN" not in text
    assert "allowed by design" in text or "sanctioned by design" in text


def test_a_greeting_after_the_lease_expired_by_the_last_row_is_not_flagged():
    report = report_of(lease(100.0, base_s=30.0), fire(200.0))
    assert report.solo_in_lease == []  # the process ended with the grant: mono 200 is past 130


# ---- limits come from the resolved room, never from literals ----


class _Proc:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def test_limits_are_read_from_print_env_rather_than_restated():
    printed = {
        "greeter_enabled": True,
        "greeter_dwell_s": 1.5,
        "greeter_min_spacing_s": 7.0,
        "greeter_group_min_spacing_s": 9.0,
        "greeter_max_per_hour": 2,
        "greeter_idle_fps": 1.0,
        "greeter_engaged_fps": 4.0,
        "greeter_face_centering": True,
        "greeter_debug": True,
        "instructions": "nothing here matters",
    }
    limits = bg.resolve_limits("hackday", run=lambda *a, **k: _Proc(json.dumps(printed)))
    assert limits.dwell_s == 1.5
    assert limits.min_spacing_s == 7.0
    assert limits.group_min_spacing_s == 9.0
    assert limits.max_per_hour == 2
    assert (limits.idle_fps, limits.engaged_fps) == (1.0, 4.0)
    assert limits.enabled and limits.face_centering and limits.debug
    assert limits.fallback is False
    assert "print-env" in limits.source


def test_unreachable_print_env_falls_back_and_the_report_says_so():
    def boom(*a, **k):
        raise OSError("no interpreter")

    limits = bg.resolve_limits("hackday", run=boom)
    assert limits.fallback is True
    report = bg.analyze([fire(100.0), fire(500.0)], limits)
    assert "FALLBACK DEFAULTS" in text_of(report)


def test_print_env_output_that_no_longer_names_the_greeter_keys_is_not_guessed_at():
    limits = bg.resolve_limits("hackday", run=lambda *a, **k: _Proc(json.dumps({"env": "hackday"})))
    assert limits.fallback is True


# ---- what the report says about the camera and the CPU ----


def test_effective_fps_is_reported_as_unmeasurable_and_the_cost_as_a_projection():
    """The honest gap: no frame is journalled, so the only fps number this box can give
    is the projected duty cycle from docs/pins.yaml, never a measurement."""
    report = report_of(fire(100.0), face(101.0))
    text = text_of(report)
    assert "NOT MEASURABLE FROM THE JOURNAL" in text
    assert "ms/s" in text and "% of one core" in text
    assert "UNMEASURED PENDING BENCH" in text
    assert "LOWER BOUND" in text


def test_the_projection_uses_the_configured_cadence_and_the_measured_ms_per_frame():
    """The number the CPU-budget line carries must be fps x the pins measurement, in the
    units ADR 0006 is quoted in: ms/s, one tenth of that as a share of one core, and the
    4-core share on top of it. All four are recomputed here from the same parsed inputs
    rather than written as literals, so re-measuring pins.yaml cannot stale this."""
    ms_p50, ms_p95, _ = bg.pins_ms_per_frame()
    assert ms_p50 > 0 and ms_p95 >= ms_p50
    line = bg.cpu_projection(LIMITS, ms_p50, ms_p95)
    idle, engaged = LIMITS.idle_fps * ms_p50, LIMITS.engaged_fps * ms_p95
    assert f"{idle:.1f}-{engaged:.1f} ms/s" in line
    assert f"{idle / 10:.2f}-{engaged / 10:.2f}% of one core" in line
    assert f"{idle / 10 / bg.CORES:.2f}-{engaged / 10 / bg.CORES:.2f}% of the {bg.CORES}-core box" in line


def test_a_camera_disabled_run_is_called_out():
    report = report_of(fire(100.0), ev(200.0, bg.CAMERA, error="daemon holds the device"))
    assert report.runs[0].camera_disabled == 1
    assert "handed the device back" in text_of(report)


# ---- watch mode: one line per decision, for a human standing in front of him ----


def test_watch_mode_prints_one_line_per_greeting_with_the_gate_verdict():
    line = bg.watch_line(fire(100.0), LIMITS, None)
    assert line.splitlines()[0].count("FIRED") == 1
    assert "solo" in line and "dwell=2.4s" in line and "bearing=2.0deg" in line


def test_watch_mode_flags_a_greeting_fired_over_a_lease_inline():
    line = bg.watch_line(fire(100.0), LIMITS, bg.Interval(90.0, 130.0, "p-1", True))
    assert "NEVER-MUST-HAPPEN" in line
    assert "p-1" in line


def test_watch_mode_names_the_reason_for_a_suppression():
    line = bg.watch_line(ev(100.0, bg.SUPPRESSED, reason="spacing", track=4, dwell_s=3.2), LIMITS, None)
    assert "suppressed" in line and "spacing" in line
    assert "min-spacing window" in line


def test_watch_mode_reports_a_near_miss_face_as_the_dwell_clock_running():
    line = bg.watch_line(face(100.0), LIMITS, None)
    assert "face" in line and "dwell clock" in line


def test_watch_mode_does_not_let_a_group_notice_look_like_a_lease_break():
    line = bg.watch_line(fire(100.0, kind="group"), LIMITS, bg.Interval(90.0, 130.0, "p-1", True))
    assert "NEVER-MUST-HAPPEN" not in line
    assert "allowed by design" in line


def test_watch_mode_warns_when_a_fire_undercuts_the_dwell_gate():
    line = bg.watch_line(fire(100.0, dwell=0.5), LIMITS, None)
    assert "below the 2s dwell gate" in line


# ---- the lease windows themselves ----


def test_lease_windows_merge_a_renewal_into_the_open_grant():
    windows = bg.LeaseWindows()
    for event in (lease(100.0, base_s=30.0), ev(120.0, bg.LEASE_RENEWED, holder="p-1", remaining_s=20.0)):
        windows.feed(event)
    windows.finalize()
    assert len(windows.intervals) == 1
    assert (windows.intervals[0].start, windows.intervals[0].end) == (100.0, 140.0)
    assert windows.covering(139.9) is not None and windows.covering(140.0) is None


def test_a_grant_still_open_when_the_rows_stop_keeps_its_deadline():
    """Clamping the tail to the last row would cut the window short and let a greeting
    that fired seconds after the last renewal slip past unflagged."""
    windows = bg.LeaseWindows()
    windows.feed(lease(100.0, base_s=30.0))
    windows.finalize()
    assert windows.covering(129.0) is not None


def test_a_second_lease_in_the_same_run_opens_a_second_window():
    windows = bg.LeaseWindows()
    for event in (lease(100.0), ev(131.0, bg.LEASE_EXPIRED, holder="p-1"), lease(200.0, base_s=10.0)):
        windows.feed(event)
    windows.finalize()
    assert len(windows.intervals) == 2
    assert windows.covering(205.0) is not None and windows.covering(150.0) is None


def test_split_runs_breaks_where_mono_restarts():
    runs = bg.split_runs([ev(0.0, bg.FACE), ev(50.0, bg.FACE), ev(1.0, bg.FACE), ev(9.0, bg.FACE)])
    assert [[e.mono for e in r] for r in runs] == [[0.0, 50.0], [1.0, 9.0]]


def test_live_at_answers_for_a_grant_that_has_not_closed_yet():
    """The watch path asks this the instant a greeting lands, with no expired row in
    sight - the lease is still being held."""
    windows = bg.LeaseWindows()
    windows.feed(lease(100.0, base_s=30.0))
    assert windows.covering(110.0) is None  # nothing closed yet: covering() cannot see it
    open_grant = windows.live_at(110.0)
    assert open_grant is not None and open_grant.holder == "p-1"
    assert windows.live_at(131.0) is None  # past the deadline the payload named


def test_a_greeting_in_the_last_rows_of_a_run_is_still_judged_against_the_open_lease():
    """Tomorrow's failure looks exactly like this: no lease_expired row at all, because
    the hello that broke the rule is the last thing in the journal."""
    report = report_of(lease(100.0, base_s=30.0), fire(110.0))
    assert len(report.solo_in_lease) == 1
    assert any("solo greeting(s) fired while the attention lease was held" in line for line in bg.hard_failures(report))


# ---- rows that predate env stamping ----


def test_a_greeter_row_with_no_env_stamp_is_attributed_instead_of_dropped():
    """~89k rows of this journal carry no `env` (shell/journal.py only stamps it when the
    room names itself), and a bench that quietly filters them would report a booth that
    never happened."""
    untracked = bg.Event(id=99, mono=100.0, wall=E + 100.0, type=bg.FIRED, payload={"kind": "solo", "dwell_s": 2.2})
    report = bg.analyze([untracked], LIMITS)
    assert report.greeter_rows_unstamped == 1  # derived from the rows themselves, not from a stamp filter
    assert report.greeter_rows_ever == 1, "an unstamped greeting was treated as a greeter that never ran"
    assert report.envs["(no env stamp)"] == 1
    assert "cannot be attributed to a room" in text_of(report)
