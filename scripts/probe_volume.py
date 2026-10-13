#!/usr/bin/env python3
"""Why do repeated loudness steps stop sounding like steps? Numbers, not opinion.

Read-only by default: it reports the sink that is actually being written, the card's
hardware mixer ceiling, the dB cost of each policy step, and a requested-vs-actual
table simulated against the cap PipeWire is accused of. `--write` performs the real
writes through the *production* argv (`app/playback.py:43`) and always restores the
level it found, printing before/after.

Safety by construction: this file only shells out to `wpctl get-volume`/`status`,
`pw-dump`, `amixer scontrols|sget`, and (under `--write`) `wpctl set-volume`. It never
imports sounddevice, never opens a stream, never connects to the robot, and never
calls `amixer set`, `set-mute`, `set-profile` or `set-default`.

Run at the booth:
    uv run python scripts/probe_volume.py            # nothing moves
    uv run python scripts/probe_volume.py --write    # brief, self-restoring
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shell import volume as V  # noqa: E402  pure numbers; this import opens no device

# The alias the production write path uses (app/playback.py:43, dashboard/server.py:178).
# Not @DEFAULT_AUDIO_SINK@: measuring an alias the code does not write would answer a
# question nobody asked.
SINK = "@DEFAULT_SINK@"
DEFAULT_STARTS = (0.5, 1.0, 1.3, 1.5, 2.0)
# The gains the complaint is about: the top of the advertised range.
DEFAULT_PROBE = (1.0, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0)
# The ceiling people expect from PulseAudio-era clients when `wpctl set-volume` runs
# without -l/--limit. Hypothesis 1 written as arithmetic so it can be falsified.
ALLEGED_CAP = 1.5
RESTORE_TOLERANCE = 0.005  # the write path formats with %.2f, so 0.5% is the rounding floor

# `wpctl get-volume 1.10` is not 1.1x amplitude. PipeWire's UI volume is the cubic
# ("POLY") scale, measured here on this box: the sink printed 1.10 while its pw-dump
# channelVolumes read 1.331022 = 1.1**3. Every dB number in this report therefore goes
# through amplitude_of(), because 20*log10(1.10) would understate the step by 3x.
AMPLITUDE_EXPONENT = 3

# Speech crest factor (peak over RMS) of what actually goes out of this speaker:
# mean of `sounds/acks/obama/**/*.wav`, 11 files, peak -1.94 dBFS / RMS -15.34 dBFS.
# Re-derive it with stdlib wave+array if the voice changes; the number decides where
# full-scale clipping starts to eat the steps.
MEASURED_CREST_DB = 13.4


# --- pure math (unit-tested in CI with no device, no wpctl, no GPU) -------------------


def db(gain: float) -> float:
    """Loudness scale of a linear *amplitude* gain: dB = 20*log10(g). Silence is -inf."""
    g = float(gain)
    return -math.inf if g <= 0.0 else 20.0 * math.log10(g)


def amplitude_of(printed: float) -> float:
    """The multiplier PipeWire applies for a `wpctl` volume of `printed`."""
    p = float(printed)
    return p if p <= 0.0 else p**AMPLITUDE_EXPONENT


def printed_db(printed: float) -> float:
    """dB of a volume as `wpctl get-volume` prints it. 1.0 -> 0, 2.0 -> +18.06."""
    return db(amplitude_of(printed))


def clipped_rms(sigma: float, clip: float = 1.0) -> float:
    """RMS of N(0, sigma) hard-clipped to +/-clip; the S16LE full-scale limit.

    Exact closed form for a symmetric hard clip, checked against numeric integration
    to 6 decimals. `clip` stays a parameter because the limit is digital full scale,
    not a property of the speaker.
    """
    s = float(sigma)
    if s <= 0.0:
        return 0.0
    t = clip / (s * math.sqrt(2.0))
    phi = math.exp(-t * t) / math.sqrt(2.0 * math.pi)
    e2 = s * s * math.erf(t) - 2.0 * clip * s * phi + clip * clip * math.erfc(t)
    return math.sqrt(max(e2, 0.0))


def audible_step_db(before: float, after: float, crest_db: float = MEASURED_CREST_DB) -> float:
    """dB an actual speech-like signal really gains when the volume moves before -> after.

    Below the clip point this equals printed_db(after) - printed_db(before); above it the
    extra gain is spent turning peaks into flat tops instead of turning up the room, which
    is the effect the booth keeps hearing. Peak-normalised speech, RMS = 1/crest.
    """
    sigma = 10.0 ** (-float(crest_db) / 20.0)
    r0 = clipped_rms(sigma * amplitude_of(before))
    r1 = clipped_rms(sigma * amplitude_of(after))
    return 0.0 if r0 <= 0.0 or r1 <= 0.0 else 20.0 * math.log10(r1 / r0)


def landing(requested: float, cap: float | None = None) -> float:
    """What a request for `requested` ends up as under an optional ceiling.

    `cap=None` models "PipeWire stores what it is told"; `cap=1.5` models the
    PulseAudio app limit. Rounding to 2 dp is not decoration: the write path formats
    the number with `%.2f` (app/playback.py:43), so 1.6 is already 1.60 before it
    reaches the daemon.
    """
    value = round(float(requested), 2)
    return value if cap is None else round(min(value, float(cap)), 2)


def step_rows(
    starts: tuple[float, ...] = DEFAULT_STARTS, words: tuple[str, ...] = ("louder", "much louder")
) -> list[dict]:
    """One row per (starting gain, step word): where the policy lands and what it costs in dB.

    The target comes from `V.resolve`, i.e. the real policy including `clamp_gain`, so
    a row sitting on the ceiling shows up as a 0.00 dB step — the clamp eating a whole
    command is part of the answer, and it is marked rather than hidden.
    """
    rows = []
    for start in starts:
        for word in words:
            target = V.resolve(word, start)
            # The steps are RATIOS since 2026-10-13 (+2.95 dB per press, so the word means
            # the same thing at 0.6 and at 1.4). The old additive `start + STEPS[word]` here
            # silently reported every row as clamped once that changed - which is the same
            # trap as comparing floats for equality: clamp_gain rounds to the 2 decimals
            # wpctl can take, so "clamped" has to mean *not merely rounded*.
            raw_target = float(start) * V.STEPS[word]
            rows.append(
                {
                    "start": round(float(start), 2),
                    "word": word,
                    "target": target,
                    "clamped": target != round(raw_target, 2),
                    "ratio": V.STEPS[word],
                    "db_start": printed_db(start),
                    "db_delta": printed_db(target) - printed_db(start),
                    "db_audible": audible_step_db(start, target),
                }
            )
    return rows


def format_step_table(rows: list[dict]) -> str:
    head = (
        f"{'start':>6}  {'word':>12}  {'ratio':>6}  {'target':>6}  {'dB(start)':>10}"
        f"  {'dB asked for':>13}  {'dB audible':>12}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        mark = " *" if r["clamped"] else ""
        lines.append(
            f"{r['start']:>6.2f}  {r['word']:>12}  {r['ratio']:>6.2f}  {r['target']:>6.2f}"
            f"  {r['db_start']:>10.2f}  {r['db_delta']:>+12.2f}dB  {r['db_audible']:>+11.2f}dB{mark}"
        )
    lines.append("  * = clamp_gain ate the whole step: the command succeeds and nothing changes")
    lines.append(
        "  'dB audible' models the sink's S16LE full-scale clip at a %.1f dB crest factor; ~1 dB is"
        " the just-noticeable difference" % MEASURED_CREST_DB
    )
    return "\n".join(lines)


def landing_note(requested: float, measured: float | None) -> str:
    if measured is None:
        return "not measured (rerun with --write)"
    if abs(measured - requested) <= RESTORE_TOLERANCE:
        return "exact"
    if abs(measured - ALLEGED_CAP) <= RESTORE_TOLERANCE:
        return "CAPPED at 1.50 - the top of the advertised range is fiction"
    return f"short by {requested - measured:.2f}"


def format_probe_table(rows: list[dict]) -> str:
    """requested -> what the alleged cap predicts -> what the sink really reported."""
    head = f"{'requested':>9}  {'cap 1.5':>8}  {'measured':>9}  note"
    lines = [head, "-" * len(head)]
    for r in rows:
        measured = r.get("measured")
        shown = "      n/a" if measured is None else f"{measured:>9.2f}"
        sim = r.get("simulated")
        sim_shown = "      n/a" if sim is None else f"{sim:>8.2f}"
        lines.append(f"{r['requested']:>9.2f}  {sim_shown}  {shown}  {landing_note(r['requested'], measured)}")
    return "\n".join(lines)


# --- device reads (subprocess only) ---------------------------------------------------


def _run(argv: list[str]) -> tuple[int, str]:
    """Run a control command. stderr is dropped: the RTKit warnings are this box's noise."""
    p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    return p.returncode, p.stdout.strip()


def _wpctl(*args: str) -> str:
    rc, out = _run(["wpctl", *args])
    return out if rc == 0 else ""


def read_volume(target: str = SINK) -> float | None:
    """`wpctl get-volume` as a float, parsed the way app/robot_tools.py:282 parses it."""
    line = next((ln for ln in _wpctl("get-volume", target).splitlines() if ln.startswith("Volume:")), "")
    parts = line.split()
    return float(parts[1]) if len(parts) > 1 else None


def sink_identity() -> dict:
    """Which node @DEFAULT_SINK@ is, every other sink, and the configured default (hypothesis 3)."""
    status = _wpctl("status")
    out: dict = {"default_line": "", "default_id": None, "configured": "", "sinks": []}
    in_sinks = False
    for ln in status.splitlines():
        s = ln.strip(" │├└─")
        if s.startswith("Sinks:"):
            in_sinks = True
            continue
        if s.startswith(("Sources:", "Filters:", "Streams:", "Settings")):
            in_sinks = False
        if in_sinks and re.match(r"^(\*\s*)?\d+\.", s):
            out["sinks"].append(s)
            m = re.match(r"^\*\s+(\d+)\.", s)
            if m:
                out["default_line"], out["default_id"] = s, int(m.group(1))
        elif "Audio/Sink" in s:
            out["configured"] = s
    return out


def node_dumps() -> list[dict]:
    rc, out = _run(["pw-dump"])
    if rc != 0 or not out:
        return []
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return []


def _param_value(params: list, key: str):
    """Pull one value out of the flat [key, value, key, value...] PipeWire param list."""
    flat = list(params)
    return flat[flat.index(key) + 1] if key in flat else None


def resolve_node_id(target: str) -> int | None:
    if target.isdigit():
        return int(target)
    return sink_identity().get("default_id")


def node_state(target: str = SINK) -> dict:
    """The live gain in the units PipeWire multiplies by, plus the negotiated format.

    `wpctl get-volume` prints the *cubic* volume - 1.10 means 1.1^3 = 1.331x amplitude -
    so any dB claim about the printed number has to be checked against channelVolumes.
    """
    want = resolve_node_id(target)
    for info in node_dumps():
        if info.get("id") != want:
            continue
        props = (info.get("info", {}) or {}).get("props", {}) or {}
        params = (info.get("info", {}) or {}).get("params", {}) or {}
        fmt = (params.get("Format") or [{}])[0]
        vol = next((p for p in params.get("Props", []) if isinstance(p, dict) and "channelVolumes" in p), {})
        mix = vol.get("params", [])
        return {
            "node_id": info.get("id"),
            "description": props.get("node.description", ""),
            "format": fmt.get("format"),
            "rate": fmt.get("rate"),
            "channel_map": vol.get("channelMap"),
            "channel_volumes": vol.get("channelVolumes"),
            "soft_volumes": vol.get("softVolumes"),
            "mixer_max_volume": _param_value(mix, "channelmix.max-volume"),
            "dither": _param_value(mix, "dither.method"),
            "mute": vol.get("mute"),
            "alsa_headroom": props.get("api.alsa.headroom"),
        }
    return {}


def reachy_card_index() -> int | None:
    """Find the card by name. A hardcoded `amixer -c 3` is exactly what broke reachy-audio-init."""
    try:
        lines = Path("/proc/asound/cards").read_text().splitlines()
    except OSError:
        return None
    for ln in lines:
        if "Reachy Mini Audio" in ln:
            m = re.match(r"^\s*(\d+)\s", ln)
            if m:
                return int(m.group(1))
    return None


def hardware_controls(card: int) -> str:
    """Every control on the card and its value, read with `amixer sget` only."""
    rc, names = _run(["amixer", "-c", str(card), "scontrols"])
    if rc != 0:
        return f"amixer -c {card} scontrols failed"
    blocks = []
    for m in re.finditer(r"control '([^']+)',(\d+)", names):
        name, index = m.group(1), m.group(2)
        _rc, out = _run(["amixer", "-c", str(card), "sget", f"{name},{index}"])
        blocks.append(out or f"(no output for {name},{index})")
    return "\n\n".join(blocks)


def write_probe(target: str, levels: tuple[float, ...]) -> tuple[list[dict], dict]:
    """Ask for `levels` through the production argv, then put the room back.

    The level and the channel state are recorded FIRST and restored in `finally`: the
    booth's loudness is not ours to leave changed.
    """
    before = read_volume(target)
    before_state = node_state(target)
    rows: list[dict] = []
    try:
        for level in levels:
            requested = V.clamp_gain(level)
            _run(["wpctl", "set-volume", target, f"{requested:.2f}"])
            rows.append({"requested": requested, "simulated": landing(requested, ALLEGED_CAP)})
            rows[-1]["measured"] = read_volume(target)
    finally:
        if before is not None:
            _run(["wpctl", "set-volume", target, f"{before:.2f}"])
    return rows, {"before": before, "after_restore": read_volume(target), "before_state": before_state}


def format_restore(restore: dict) -> str:
    before, after = restore.get("before"), restore.get("after_restore")
    ok = before is not None and after is not None and abs(before - after) <= RESTORE_TOLERANCE
    state = restore.get("before_state") or {}
    return (
        f"  restore: before={before} after={after} -> {'RESTORED (verified)' if ok else 'CHECK BY HAND'}\n"
        f"  channel volumes before the probe: {state.get('channel_volumes')}"
    )


# --- report -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probe the loudness path: sink, hardware ceiling, dB steps.")
    ap.add_argument("--write", action="store_true", help="do the real writes (restores the level after)")
    ap.add_argument("--target", default=SINK, help=f"wpctl node to probe (default {SINK})")
    ap.add_argument("--levels", default="", help="comma-separated gains for the write test")
    ap.add_argument("--starts", default="", help="comma-separated starting gains for the dB table")
    args = ap.parse_args(argv)

    levels = tuple(float(x) for x in args.levels.split(",")) if args.levels else DEFAULT_PROBE
    starts = tuple(float(x) for x in args.starts.split(",")) if args.starts else DEFAULT_STARTS

    ident = sink_identity()
    print("== hypothesis 3: which device is being written ==")
    print(f"  {args.target} -> {ident['default_line'] or '(no default sink)'}")
    print(f"  configured default: {ident['configured'] or '(none)'}")
    for s in ident["sinks"]:
        print(f"  sink: {s}")
    print(f"  read back: {read_volume(args.target)}")

    print("\n== node state (the gain PipeWire multiplies by) ==")
    state = node_state(args.target)
    for key in (
        "node_id",
        "description",
        "format",
        "rate",
        "channel_map",
        "channel_volumes",
        "soft_volumes",
        "mixer_max_volume",
        "dither",
        "mute",
        "alsa_headroom",
    ):
        print(f"  {key}: {state.get(key)}")
    if state.get("channel_volumes") and read_volume(args.target) is not None:
        printed = read_volume(args.target)
        amp = float(state["channel_volumes"][0])
        print(
            f"  printed {printed:.2f} => amplitude x{amp:.4f} ({db(amp):+.2f} dB); "
            f"20*log10(printed) would say {db(printed):+.2f} dB and be wrong"
        )
        if amp * 10 ** (-MEASURED_CREST_DB / 20.0) > 1.0:
            print("  the speech RMS is already past digital full scale at this reading: peaks are clipping")

    print("\n== hardware controls (Reachy card, amixer get only) ==")
    card = reachy_card_index()
    print(f"  card index for 'Reachy Mini Audio': {card}")
    if card is not None:
        print(hardware_controls(card))

    print("\n== hypothesis 2: what each policy step is worth in dB ==")
    print(format_step_table(step_rows(starts)))
    lo, hi = min(starts), max(starts)
    print(
        f"  the whole advertised range {lo:.2f} -> {hi:.2f}: {printed_db(hi) - printed_db(lo):+.2f} dB asked for,"
        f" {audible_step_db(lo, hi):+.2f} dB audible"
    )

    print("\n== hypothesis 1: requested vs actual ==")
    if args.write:
        rows, restore = write_probe(args.target, levels)
        print(format_probe_table(rows))
        print(format_restore(restore))
    else:
        rows = [{"requested": V.clamp_gain(g), "simulated": landing(g), "measured": None} for g in levels]
        print(format_probe_table(rows))
        print("  (measured column is empty without --write; the 1.50 column is the alleged ceiling)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
