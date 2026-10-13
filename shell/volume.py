"""One volume policy, read by the tool path AND the dashboard.

Why this file exists: on 2026-10-12 the ceiling went 2.0 -> 1.2 in
`app/robot_tools.py`, and an hour later the dashboard slider got its own separate
1.2 clamp. Two copies of a number that describes a physical room is how you get a
web page that can outrank the tool the model trusts, and an operator who cannot
tell which one the robot is obeying. The room moved again the same night: with a
crowd around it, 0.5 was too quiet to hear at all, so David asked for headroom to
200% and a start around 100-120%. The numbers live here; enforcement lives in
`app/robot_tools.py` (`set_volume`) and `dashboard/env_panel.py` (the slider).

What "200%" means physically: the Reachy card's hardware gain is already at max
(ADR 0005), so this is PipeWire sink gain on top of a Qwen3-TTS stream that is
already hotter than the retired Kokoro path. Past about 1.2 the desk speaker
measured as distortion territory. That is why going up there is allowed but
flagged (`clipping_risk`) instead of being silently normal.
"""

from __future__ import annotations

MIN_GAIN = 0.0

# Measured on this box 2026-10-13 (scripts/probe_volume.py, live node), not estimated.
# Three facts decided these numbers:
#
#   1. `wpctl`'s number is CUBIC volume: printed 1.10 is 1.10^3 = 1.331x amplitude, i.e.
#      +2.48 dB, and the dB arithmetic for it is 60*log10, not 20*log10. Every dB figure
#      quoted about this control before today was understated threefold.
#   2. The ack audio peaks at -1.94 dBFS (11 real assets) and the TTS stream is the same
#      neighbourhood; the card's hardware PCM is ALREADY at its maximum 60/60, 0.00 dB
#      (ADR 0005); the graph is S16LE with no dither. So the clean gain ceiling is
#      10^(1.94/60) = 1.077. Above that, more "volume" is not more loudness - it is digital
#      clipping, and there is no hardware headroom left to spend.
#   3. With the old ceiling of 2.0, clamp_gain() returned 2.0 for every request above it:
#      ok: true, +0.00 dB, no change whatsoever. That is precisely what the operator
#      reported hearing ("at a certain point it doesn't change much at all") - the tool had
#      been reporting success for a control that could not move. 2.0 was also a promise of
#      +18 dB (8x amplitude) on a stream with no headroom, i.e. distortion sold as loudness.
#
# 1.5 is where the last press is still worth something (+0.88 dB from 1.45); past it the curve
# is flat, the sound is only worse, and clamp_gain would keep answering ok:true for no change.
MAX_GAIN = 1.5
DEFAULT_GAIN = 1.1  # +2.5 dB: a deliberate hair over the clean ceiling. A crowded room
# beats a clean desk recording, and this is the point where that trade is still worth
# making rather than surrendering.
#
# Steps are RATIOS now, so "louder" means the same thing at 0.6 and at 1.4. On a cubic
# scale, absolute steps shrink as you climb: +0.10 was worth +4.75 dB at 0.50 and +0.85 dB
# at 1.50 - the same word, six times less response, which is what "it stopped changing"
# sounds like from across a booth.
STEP_RATIO = 1.12  # +2.95 dB per "louder" (60*log10)
BIG_STEP_RATIO = 1.25  # +5.58 dB per "much louder"
CLIPPING_ABOVE = 1.08  # 10^(1.94/60): the cube volume at which a full-scale ack first clips

# Absolute levels the model can name. Retuned into the new range: "medium" is no
# longer 0.5, because at a desk with people around, 0.5 is not medium any more.
NAMED: dict[str, float] = {
    "mute": 0.0,
    "silence": 0.0,
    "whisper": 0.15,
    "quiet": 0.35,
    "low": 0.35,
    "medium": 0.7,
    "normal": 1.1,
    "default": 1.1,
    "loud": 1.4,
    "high": 1.4,
    "max": MAX_GAIN,
    "maximum": MAX_GAIN,
    "full": MAX_GAIN,
}

#: Multipliers, not increments: >1 raises, <1 lowers. The table still owns the words so the
#: arithmetic lives in one place; STEP/BIG_STEP keep their names as the reciprocal ("quieter")
#: side of the same two ratios.
STEP = 1.0 / STEP_RATIO
BIG_STEP = 1.0 / BIG_STEP_RATIO
STEPS: dict[str, float] = {
    "louder": STEP_RATIO,
    "up": STEP_RATIO,
    "raise": STEP_RATIO,
    "volume up": STEP_RATIO,
    "much louder": BIG_STEP_RATIO,
    "way louder": BIG_STEP_RATIO,
    "quieter": STEP,
    "down": STEP,
    "softer": STEP,
    "lower": STEP,
    "volume down": STEP,
    "much quieter": BIG_STEP,
    "way quieter": BIG_STEP,
}


def clamp_gain(gain: float, ceiling: float = MAX_GAIN) -> float:
    """Bound a requested gain to [MIN_GAIN, ceiling]; NaN becomes silence.

    NaN is not hypothetical: `float("nan")` arrives from a malformed form field,
    and `wpctl set-volume nan` fails in a way that leaves the *previous* level
    playing while the page reports a new one.
    """
    try:
        value = float(gain)
    except (TypeError, ValueError):
        return MIN_GAIN
    if value != value:  # NaN
        return MIN_GAIN
    return round(max(MIN_GAIN, min(float(ceiling), value)), 2)


def parse_level(text: str) -> float | None:
    """A percentage or bare number ('150%', '1.4', '140') as a gain, else None."""
    key = str(text).strip().lower().rstrip("% ")
    if not key:
        return None
    try:
        value = float(key)
    except ValueError:
        return None
    # "150" plainly means 150%; "1.4" means 1.4x. The split point is what a person
    # would say out loud, not a mathematical ambiguity: nobody asks for "1.4 percent".
    return value / 100.0 if "%" in str(text).lower() or value > 3.0 else value


def resolve(level: str, current: float | None) -> float | None:
    """What gain to request for the word the model passed, or None if unintelligible.

    A relative word with no readable current level starts from DEFAULT_GAIN rather
    than erroring: at a booth the sink can be muted by whoever plugged it in, and
    "louder" that refuses because it could not read the mixer is a robot that
    ignores you twice.
    """
    key = str(level).strip().lower()
    if key in STEPS:
        base = DEFAULT_GAIN if current is None else float(current)
        # Ratio, not increment: one press of "louder" buys the same decibels at the top of
        # the dial as at the bottom, which is what the word promises.
        return clamp_gain(base * STEPS[key])
    if key in NAMED:
        return clamp_gain(NAMED[key])
    parsed = parse_level(key)
    return None if parsed is None else clamp_gain(parsed)


#: Words that ask what the loudness IS rather than what it should be. Live evidence 2026-10-13:
#: asked "what's your volume level?" with no way to read it back, the model confabulated ("my
#: speaker only has the va..."), and later claimed outright that it could not adjust the volume at
#: all while never calling the tool. A question that has no path to an answer gets invented, and at
#: a booth an invented answer about the robot's own state is indistinguishable from a real one.
READ_WORDS = frozenset({"current", "now", "level", "status", "report", "what is it", "how loud", "how loud are you"})


def is_read_request(level: str) -> bool:
    """True when the word asks for the present level, so nothing must be written."""
    return str(level).strip().lower().rstrip("?!. ") in READ_WORDS


def clipping_risk(gain: float) -> bool:
    return float(gain) > CLIPPING_ABOVE
