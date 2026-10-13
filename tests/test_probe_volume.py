"""The pure parts of scripts/probe_volume.py, runnable in bare CI.

Nothing here touches a device: no wpctl, no pw-dump, no amixer, no GPU. The probe
exists to answer a measurement question about the booth box, but its arithmetic - the
cubic-volume conversion, the dB maths, the clamp, the requested-vs-actual table - is
what an operator reasons with afterwards, so that arithmetic is pinned here and the
device calls stay in the script.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# The idiom the other scripts' tests use (test_bench_aec, test_bench_greeter,
# test_booth_rehearsal): a path insert plus the targeted suppression, so the import resolves at
# runtime and ty does not gain a diagnostic for a file that is a script, not a package.
import probe_volume as P  # noqa: E402  # ty: ignore[unresolved-import]

from shell import volume as V  # noqa: E402

# --- the volume scale -----------------------------------------------------------------


def test_wpctl_volume_is_cubic_not_amplitude():
    """`wpctl get-volume` printed 1.10 on the booth box while pw-dump reported
    channelVolumes 1.331022 - the cube, to within the daemon's own rounding (1.1**3 =
    1.331000, 0.0001 dB away). Anyone doing 20*log10(1.10) understates every step by 3x."""
    assert math.isclose(P.amplitude_of(1.1), 1.331022, abs_tol=1e-4)
    assert P.db(1.0) == 0.0
    assert P.db(P.amplitude_of(2.0)) == P.printed_db(2.0)
    assert math.isclose(P.printed_db(2.0), 18.062, abs_tol=1e-2)
    assert math.isclose(P.printed_db(0.5), -18.062, abs_tol=1e-2)
    assert P.printed_db(1.0) == 0.0
    assert P.printed_db(0.0) == -math.inf


def test_db_of_amplitude_keeps_the_usual_reference_points():
    assert math.isclose(P.db(2.0), 6.0206, abs_tol=1e-3)
    assert math.isclose(P.db(0.5), -6.0206, abs_tol=1e-3)
    assert P.db(0.0) == -math.inf
    assert P.db(-1.0) == -math.inf  # silence and below are one loudness, and no domain error


# --- full-scale clipping --------------------------------------------------------------


def test_below_the_clip_point_the_model_is_just_the_printed_dB():
    before, after = 0.4, 0.5
    assert math.isclose(P.audible_step_db(before, after), P.printed_db(after) - P.printed_db(before), abs_tol=1e-9)


def test_a_step_at_the_top_of_the_advertised_range_is_below_the_just_noticeable_difference():
    """The complaint, as a number: +0.10 at 1.5 -> 1.6 asks for 1.68 dB and delivers
    under 1 dB, while the same word down at 0.5 is unmistakable."""
    top = P.audible_step_db(1.5, 1.6)
    bottom = P.audible_step_db(0.5, 0.6)
    assert top < 1.0, top
    assert bottom > 4.0, bottom
    assert P.audible_step_db(1.9, 2.0) < top


def test_audible_gain_never_goes_up_faster_than_asked_for_gain():
    for before, after in ((1.0, 1.1), (1.2, 1.3), (1.5, 1.6), (1.8, 2.0), (1.9, 2.0)):
        assert P.audible_step_db(before, after) <= P.printed_db(after) - P.printed_db(before) + 1e-9


def test_clipped_rms_saturates_at_full_scale():
    """Hard clipping cannot raise RMS forever: past a point more gain only widens the
    flat tops. A thousand times the amplitude is worth 0.06 dB."""
    big, huge = P.clipped_rms(100.0), P.clipped_rms(1000.0)
    assert big <= 1.0 and huge <= 1.0
    assert abs(20 * math.log10(huge / big)) < 0.15
    # the clip level is the digital full scale of the sink format, so it must scale through
    assert math.isclose(P.clipped_rms(2.0, clip=0.5), 0.5 * P.clipped_rms(4.0), rel_tol=1e-9)
    assert P.clipped_rms(0.0) == 0.0


# --- the clamp ------------------------------------------------------------------------


def test_step_rows_use_the_real_policy_and_mark_the_rows_the_clamp_ate():
    # The ceiling itself is the interesting start: it is the only place a press can be
    # legal, requested, and worth exactly nothing.
    rows = {r["word"]: r for r in P.step_rows(starts=(V.MAX_GAIN,))}
    louder = rows["louder"]
    assert louder["target"] == V.MAX_GAIN  # clamp_gain(MAX * ratio) -> MAX
    assert louder["clamped"] is True
    assert louder["db_delta"] == 0.0 and louder["db_audible"] == 0.0  # ok=True and worth nothing
    mid = P.step_rows(starts=(1.0,))
    assert [r["clamped"] for r in mid] == [False, False]
    assert mid[0]["target"] == pytest.approx(1.0 * V.STEP_RATIO, rel=1e-6)
    assert mid[1]["target"] == pytest.approx(1.0 * V.BIG_STEP_RATIO, rel=1e-6)


def test_a_step_from_just_under_the_ceiling_is_legal_and_silently_almost_nothing():
    """One press from the top: the command is legal, returns ok=True, and buys a fraction of a
    decibel. A booth that hears nothing and gets no error learns "volume is broken". This is the
    row that justified dropping the ceiling from 2.0 to 1.5 - the old version ran it at 1.95
    against a 2.0 ceiling, so the shape of the finding is unchanged, only the arithmetic.
    """
    near_top = round(V.MAX_GAIN - 0.05, 2)  # genuinely one press from the top
    target = V.resolve("louder", near_top)
    assert target == pytest.approx(V.MAX_GAIN, rel=1e-6)
    assert P.audible_step_db(near_top, target) < 1.0, (
        "a press at the top must still be shown for what it is worth - this is the "
        "number the operator was hearing as 'it stopped changing'"
    )


# --- the alleged ceiling ----------------------------------------------------------------


def test_landing_models_the_alleged_ceiling_without_touching_the_policy():
    assert P.landing(1.6) == 1.6  # no cap: the daemon stores what it is told
    assert P.landing(1.6, cap=P.ALLEGED_CAP) == 1.5
    assert P.landing(2.0, cap=P.ALLEGED_CAP) == 1.5
    assert P.landing(1.234) == 1.23  # the write path formats %.2f, so 1.6 is 1.60 already


def test_a_capped_range_would_show_up_as_repeated_identical_readings():
    """The shape hypothesis 1's evidence would have, so the table cannot hide it."""
    requested = (1.5, 1.6, 1.8, 2.0)
    assert {P.landing(g, cap=P.ALLEGED_CAP) for g in requested} == {1.5}
    assert [P.landing(g) for g in requested] == [1.5, 1.6, 1.8, 2.0]


# --- table formatting -----------------------------------------------------------------


def test_probe_table_marks_unmeasured_rows_instead_of_inventing_numbers():
    rows = [{"requested": g, "simulated": P.landing(g), "measured": None} for g in (1.0, 1.6, 2.0)]
    out = P.format_probe_table(rows)
    lines = out.splitlines()
    assert len(lines) == 2 + len(rows)  # header + rule + one row each
    assert lines[0].startswith("requested")
    assert all("not measured" in ln for ln in lines[2:])
    assert "n/a" in out


def test_probe_table_calls_a_capped_write_capped():
    out = P.format_probe_table([{"requested": 1.6, "simulated": 1.5, "measured": 1.5}])
    assert "CAPPED at 1.50" in out and "fiction" in out


def test_probe_table_reports_an_exact_and_a_short_landing_differently():
    assert "exact" in P.format_probe_table([{"requested": 1.3, "simulated": 1.3, "measured": 1.3}])
    assert "short by 0.10" in P.format_probe_table([{"requested": 1.8, "simulated": 1.8, "measured": 1.7}])


def test_step_table_renders_every_row_and_the_clamp_marker():
    out = P.format_step_table(P.step_rows(starts=(0.5, 2.0)))
    lines = out.splitlines()
    assert len(lines) == 2 + 4 + 2  # header + rule + 2 starts x 2 words + the two legend lines
    assert "dB audible" in lines[0]
    assert out.count("*") >= 3  # one marker per clamped row, plus the legend


def test_restore_line_only_says_restored_when_the_readback_matches():
    good = P.format_restore({"before": 1.1, "after_restore": 1.1, "before_state": {"channel_volumes": [1.331]}})
    assert "RESTORED (verified)" in good and "1.331" in good
    assert "CHECK BY HAND" in P.format_restore({"before": 1.1, "after_restore": 1.0, "before_state": {}})
    assert "CHECK BY HAND" in P.format_restore({"before": None, "after_restore": 1.0, "before_state": {}})


# --- safety: the probe must stay a probe -----------------------------------------------


def test_the_probe_never_opens_a_stream_or_writes_anything_it_was_not_asked_to():
    """The booth rule: one mic owner and one speaker owner (ADR 0002), and a second SDK
    client hangs this box. The probe may read the control surfaces and, under --write,
    set-volume - nothing else. This is the pin, because an audio import would not fail
    loudly here; it would fail at the demo, on whatever already held the device."""
    tree = ast.parse((ROOT / "scripts" / "probe_volume.py").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not imported & {"sounddevice", "pyaudio", "reachy_mini", "app.playback", "shell.audio"}

    # argv tokens are string constants; the probe must never name a mixer or route write.
    tokens = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {"amixer", "sget", "scontrols", "set-volume"} <= tokens
    assert tokens & {"set-mute", "set-profile", "set-route", "set-default", "sset", "set"} == set()
