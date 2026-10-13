"""Dance presets: the Reachy Mini Dances Library, read offline, played through the one owner.

The pack is real, and it is the vendor's own sibling of the emotions library we already ship:
`GET /api/datasets/pollen-robotics/reachy-mini-dances-library` answers **HTTP 200**, HEAD
`3564295e72d41c1271f46bf5540a3fcad9d5f669`, `license: apache-2.0`, tag
`reachy_mini_community_moves`, 19 `<dance>.json` files at the repo root (420,505 B). The SDK has
known about it all along - `reachy_mini/motion/recorded_move.py` lists it in `DEFAULT_DATASETS`
and the daemon dashboard has a "Dances" option - we just never read it. It is NOT the same content
as the `dance` entry in `EMOTE_FILES` (that one is `dance1.json` from the emotions library).

Everything about the format matches the emotions library: `{"description", "time",
"set_target_data": [{head: 4x4, antennas: [2], body_yaw}]}`, ~50 Hz. So this module does not
reimplement the keyframe math - it calls `emotes.parse_emote` with its own ceiling and its own
file map. A second parser would be a second way to be wrong.

**Measured envelope, before this ever touched a servo** (all 19 files, from
`pose6_from_matrix` over the real matrices): |roll| <= 25 deg, |pitch| <= 20 deg, |yaw| <= 45 deg,
antennas <= 1.047 rad, and **body_yaw is 0.0 in every file** - the base never turns, only the
head and ears move. Raw lengths are 1.82-5.00 s, so the whole library is shorter than one
`curious1.json` (11.8 s) from the pack already playing at the booth. The owner clamps gaze yaw at
`MAX_YAW_DEG = 60`, which 45 sits inside.

**Offline, or not at all** - same rule and same reason as `emotes.py`: `local_files_only=True`,
and `scripts/fetch_dances.py` is the travel step. The s2s unit sets `HF_HUB_OFFLINE=1`.

**"I don't know that one" is not "I can't read my files."** `UnknownDance` and `DanceUnavailable`
are separate exceptions for the same reason they are in `emotes.py`: a cache miss reported as a
bad word is a retry storm.

**No preemption, and no new gaze arbitration.** `Animation(priority=0)`, played without
`preempt=True`. This is load-bearing for tonight's safety question, so it is worth being exact
about what `MotionOwner.tick()` does with it: the animation pose is *absolute* (`relative=False`)
and composes as `pose = base + w * (apose - base)` with `w` ramping in over `blend_in` and out
over `blend_out`. The greeter's idle search sweep and face lock-on only ever move `base` (the
gaze spring); while a dance plays, `w` drives to 1.0 and the gaze commands are simply overridden,
then spring back on `blend_out`. `_update_aversion` and the idle-breathing layer are both gated on
`not animating` for exactly this case. So a dance does not fight the lock-on stack and the
greeter does not need to be told to stand down - the owner already arbitrates it, in one
direction, with a ramp at both ends. What it does cost is eye contact for up to
`REACHY_DANCE_MAX_S`, which is the same trade `emotes.py` already made deliberately.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import emotes
from .owner import Animation

log = logging.getLogger("shell.dances")

REPO_ID = "pollen-robotics/reachy-mini-dances-library"
REPO_TYPE = "dataset"

#: friendly name -> library file. Every key is the file's own stem; the vendor names the files
#: for the move, so there is no translation layer to drift. The keys of this map are the tool's
#: whole vocabulary and tests/test_dances.py fails if schema, map and cache disagree.
DANCE_FILES: dict[str, str] = {
    "chicken_peck": "chicken_peck.json",
    "chin_lead": "chin_lead.json",
    "dizzy_spin": "dizzy_spin.json",
    "grid_snap": "grid_snap.json",
    "groovy_sway_and_roll": "groovy_sway_and_roll.json",
    "head_tilt_roll": "head_tilt_roll.json",
    "interwoven_spirals": "interwoven_spirals.json",
    "jackson_square": "jackson_square.json",
    "neck_recoil": "neck_recoil.json",
    "pendulum_swing": "pendulum_swing.json",
    "polyrhythm_combo": "polyrhythm_combo.json",
    "sharp_side_tilt": "sharp_side_tilt.json",
    "side_glance_flick": "side_glance_flick.json",
    "side_peekaboo": "side_peekaboo.json",
    "side_to_side_sway": "side_to_side_sway.json",
    "simple_nod": "simple_nod.json",
    "stumble_and_recover": "stumble_and_recover.json",
    "uh_huh_tilt": "uh_huh_tilt.json",
    "yeah_nod": "yeah_nod.json",
}
DANCE_NAMES: tuple[str, ...] = tuple(DANCE_FILES)

#: How long one dance may hold the animation queue, in wall-clock seconds. Nothing in the pack
#: needs it today - the longest clip is 5.00 s (`jackson_square`, `side_peekaboo`) and the
#: shortest 1.82 s - so this is a ceiling for the pack, not a trim for it. It exists because the
#: dataset is upstream-owned and community-tagged: if Pollen adds an 18 s routine, the default
#: behaviour must be "play it faster", not "park the head in a spin for 18 s while a guest waits
#: for a nod". Same scaling rule as `REACHY_EMOTE_MAX_S`: durations shrink, frames do not.
#: REACHY_DANCE_MAX_S=0 means "play them at their written length".
DANCE_MAX_S_DEFAULT = 6.0
DANCE_MAX_S_ENV = "REACHY_DANCE_MAX_S"

#: A dance is a performance, so a guest will ask for the same one again to show someone else.
#: One journal row per request would flush the ring the way the scan detents were going to
#: (shell/vision/greeter.py `SCAN_JOURNAL_S`); repeats inside this window are dropped, distinct
#: names are not. Failures are never rate-limited - see the handler.
DANCE_JOURNAL_S = 10.0


def dance_max_s(raw: str | None = None) -> float:
    import os

    if raw is None:
        raw = os.getenv(DANCE_MAX_S_ENV, "")
    raw = (raw or "").strip()
    if not raw:
        return DANCE_MAX_S_DEFAULT
    try:  # nonsense is the shipped default, same convention as emote_max_s
        return max(float(raw), 0.0)
    except ValueError:
        return DANCE_MAX_S_DEFAULT


class UnknownDance(Exception):
    """Not in our vocabulary. Nothing to do with the library being readable."""

    def __init__(self, name: str):
        super().__init__(f"no such dance: {name!r}")
        self.name = name


class DanceUnavailable(Exception):
    """Named correctly; the data is not readable offline. Retrying cannot help."""

    def __init__(self, name: str, detail: str = ""):
        super().__init__(f"dance {name!r} is not available locally {detail}".strip())
        self.name = name


def _path_for(name: str) -> Path:
    """Resolve one library file from the local cache only - never the network."""
    if name not in DANCE_FILES:
        raise UnknownDance(name)
    try:
        # Import INSIDE the try, for the emotes.py reason: a box without huggingface_hub has an
        # unreadable library, which is DanceUnavailable, not a ModuleNotFoundError in a handler.
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(REPO_ID, DANCE_FILES[name], repo_type=REPO_TYPE, local_files_only=True)
    except UnknownDance:
        raise
    except Exception as e:  # LocalEntryNotFoundError, ImportError, and friends
        raise DanceUnavailable(name, f"({type(e).__name__})") from e
    return Path(local)


def parse_dance(name: str, payload: dict, *, max_s: float | None = None) -> Animation:
    """Frames -> an Animation, on the emote parser's keyframe math with the dance ceiling.

    The `EmoteUnavailable` translation is not decoration. `parse_emote` raises it for a file with
    no keyframes and no head poses - a malformed *dance* file - and the dance handler only catches
    `DanceUnavailable`. Without the re-raise, that case escapes to `@_guard`, the guest hears an
    unhelpful sentence, the journal gets an unclassifiable row, and the message calls the routine
    an emote. Vocabulary rejection stays `UnknownDance`, raised earlier, in `_path_for`.
    """
    try:
        anim = emotes.parse_emote(name, payload, max_s=dance_max_s() if max_s is None else max_s)
    except emotes.EmoteUnavailable as e:
        raise DanceUnavailable(name, f"({type(e).__name__})") from e
    anim.name = f"dance:{name}"
    return anim


def load(name: str) -> Animation:
    """Read and parse one dance from the local cache. Raises UnknownDance / DanceUnavailable."""
    path = _path_for(name)
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise DanceUnavailable(name, f"({type(e).__name__})") from e
    return parse_dance(name, payload)


def duration_s(name: str) -> float:
    return round(sum(kf.duration for kf in load(name).keyframes), 2)


#: Model-facing guidance. Same discipline as EMOTE_WHEN: a boundary, a prohibition, and the
#: vocabulary - the 4B model does not need an essay in every request. The named exclusion of
#: `play_emotion` is deliberate: `dance` already exists as an emotion word, and `simple_nod` /
#: `yeah_nod` / `uh_huh_tilt` here overlap the `nod` tool, so without a boundary the two tools
#: compete and the routing measurement (docs/emote-routing.md) stops meaning anything.
DANCE_WHEN = (
    "Perform one of his stock dance routines - a short, whole-head and antenna sequence from the "
    "dances library: chicken_peck, chin_lead, dizzy_spin, grid_snap, groovy_sway_and_roll, "
    "head_tilt_roll, interwoven_spirals, jackson_square, neck_recoil, pendulum_swing, "
    "polyrhythm_combo, sharp_side_tilt, side_glance_flick, side_peekaboo, side_to_side_sway, "
    "simple_nod, stumble_and_recover, uh_huh_tilt, yeah_nod. Use it when a guest asks him to "
    "dance, or to celebrate something they just asked for; for an ordinary greeting or a bit of "
    "agreement use play_emotion or nod instead, not a routine. This is a MOVEMENT, not a claim "
    "about feeling festive, and calling this tool is the ONLY way his body performs: saying "
    '"I shall dance" moves nothing. At most once per turn, and do not narrate it.'
)

#: The tool schema. Adjacent to DANCE_FILES for the emotes.py reason: `enum` must be that map's
#: keys exactly.
DANCE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "dance",
        "description": DANCE_WHEN,
        "parameters": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "enum": list(DANCE_NAMES),
                    "description": "Which dance to perform",
                }
            },
            "required": ["move"],
        },
    },
}
DANCE_TOOLS: list[dict] = [DANCE_TOOL]


def status() -> dict:
    """What a preflight needs: can every vocabulary word actually be played right now?"""
    out: dict = {"repo": REPO_ID, "cached": [], "missing": [], "revision": None, "bytes": 0}
    for name in DANCE_NAMES:
        try:
            path = _path_for(name)
        except (UnknownDance, DanceUnavailable) as e:
            out["missing"].append(f"{name}: {type(e).__name__}")
            continue
        try:
            out["bytes"] += path.stat().st_size  # Path.stat follows the cache symlink
        except OSError:
            pass
        out["cached"].append(name)
        if out["revision"] is None:
            # .../<hub>/datasets--<org>--<repo>/snapshots/<revision>/<file>
            parts = Path(path).parts
            out["revision"] = parts[-2] if len(parts) >= 2 else None
    return out
