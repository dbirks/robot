"""Emote presets: the Reachy Mini emotion library, read offline, played through the one owner.

Issue #34's gap. The robot could nod, shake, dance and sleep, and could not express anything
else, because `play_emotion` lived on the legacy bridge (see the comment that used to sit in
`shell/robot_tools.py.build_handlers`) and loaded its keyframes through `app/emotion_loader.py`
- a module that depends on the legacy `MovementManager` keyframe type and on
`app/movement_manager.py`, all of which issue #22 deletes.

Three things this module does differently, each learned the hard way tonight:

**Offline, or not at all.** `hf_hub_download(..., local_files_only=True)`. The library lives in
the default `HF_HOME` cache (116 MB, 250 files); the shell unit sets no `HF_HOME`, so it
resolves. The legacy loader omitted `local_files_only`, which means a cold cache turns an emote
into a network round trip inside a 12 s tool timeout at a venue with filtered egress - the same
shape as the snapshot pile-in, where a slow tool and a retry message produced a 54-second wait.

**"I don't know that one" must not mean "I can't read my files".** `UnknownEmote` and
`EmoteUnavailable` are different exceptions, and the handler says different things for them. The
legacy handler caught every failure as `Unknown emotion: <name>` plus the full available list, so
a cache miss told the model that `sad` does not exist - and a model that is told a tool call was
wrong tries again, which is how a broken library becomes a stuttering robot.

**No preemption.** `Animation(priority=0)` and `play(anim)` without `preempt=True`: an emote
queues behind the live gaze instead of overriding it. The legacy call passed `preempt=True,
priority=1`, so a "cheerful" would have yanked the head away from whoever he was looking at, mid
-sentence. A gesture that costs eye contact is worse than no gesture.

The library's own files are dense (898 frames for `sad1.json`), so they are subsampled to
~15 keyframes - the same target the legacy loader used - but each keyframe's duration comes from
the file's real `time` axis, so a 3.6-second gesture takes 3.6 seconds rather than 15 x the
0.3 s default.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .kinematics import pose6_from_matrix
from .owner import Animation, Keyframe

log = logging.getLogger("shell.emotes")

REPO_ID = "pollen-robotics/reachy-mini-emotions-library"
REPO_TYPE = "dataset"

#: friendly name -> library file. The keys of this dict are the tool's whole vocabulary, and
#: tests/test_emotes.py fails if the schema, this map and the cache disagree.
EMOTE_FILES: dict[str, str] = {
    "hello": "welcoming1.json",
    "goodbye": "go_away1.json",
    "cheerful": "cheerful1.json",
    "curious": "curious1.json",
    "surprised": "surprised1.json",
    "confused": "confused1.json",
    "laughing": "laughing1.json",
    "sad": "sad1.json",
    "attentive": "attentive1.json",
    "dance": "dance1.json",
}
EMOTE_NAMES: tuple[str, ...] = tuple(EMOTE_FILES)

#: The library's files carry 100-2000 frames; the owner interpolates between keyframes at
#: CONTROL_HZ, so ~15 is plenty and keeps one emote from occupying the queue for a minute.
SUBSAMPLE_TO = 15

#: How long one gesture may hold the animation queue, in wall-clock seconds. The library's
#: `curious1.json` is 12.6 s and `sad1.json` 9.7 s; the owner plays queued animations in order,
#: so an uncapped emote parks his head in a feeling for longer than a booth turn lasts. Measured
#: against the snapshot pile-in (issue tonight): a tool whose work outlives the guest's patience
#: gets retried, and retries queue behind the original. Over the ceiling the keyframe durations
#: are scaled rather than the frames dropped - a quicker gesture still has a shape; a truncated
#: one stops mid-shrug. REACHY_EMOTE_MAX_S=0 means "play them at their written length".
EMOTE_MAX_S_DEFAULT = 7.0
EMOTE_MAX_S_ENV = "REACHY_EMOTE_MAX_S"


def emote_max_s(raw: str | None = None) -> float:
    import os

    if raw is None:
        raw = os.getenv(EMOTE_MAX_S_ENV, "")
    raw = (raw or "").strip()
    if not raw:
        return EMOTE_MAX_S_DEFAULT
    try:  # nonsense is the shipped default, same convention as the camera warm-up deadline
        return max(float(raw), 0.0)
    except ValueError:
        return EMOTE_MAX_S_DEFAULT


#: Model-facing guidance. Kept short on purpose: this text is in every tool schema in every
#: request, and tonight's tool-description audit found the 4B model does not need an essay - it
#: needs a boundary ("at most once") and a prohibition ("do not narrate it").
EMOTE_WHEN = (
    "Perform one of his stock physical gestures - a whole-head and ear movement, the same kind "
    "of thing as nodding: hello, goodbye, cheerful, curious, surprised, confused, laughing, sad, "
    "attentive, dance. This is a MOVEMENT, not a claim about what he feels, so a modest or dry "
    "persona can still perform it: declining because he cannot truly feel something is wrong, the "
    "gesture is a greeting, not a confession. Calling this tool is the ONLY way his body moves - "
    'saying "I am waving" or "I shall look curious" moves nothing, so never describe a '
    "gesture instead of performing it. Use it when the guest's own words call for it (a greeting, "
    "good news, a joke, a farewell) or whenever they ask directly, at most once in a turn, and let "
    "the movement speak rather than announcing it."
)


class UnknownEmote(Exception):
    """Not in our vocabulary. Nothing to do with the library being readable."""

    def __init__(self, name: str):
        super().__init__(f"no such emote: {name!r}")
        self.name = name


class EmoteUnavailable(Exception):
    """Named correctly; the data is not readable offline. Retrying cannot help."""

    def __init__(self, name: str, detail: str = ""):
        super().__init__(f"emote {name!r} is not available locally {detail}".strip())
        self.name = name


def _subsample(n: int, target: int = SUBSAMPLE_TO) -> list[int]:
    """Evenly spaced indices, always including first and last."""
    if n <= 0:
        return []
    if n <= target:
        return list(range(n))
    step = (n - 1) / (target - 1)
    seen: list[int] = []
    for i in range(target):
        idx = round(i * step)
        if not seen or idx != seen[-1]:
            seen.append(idx)
    return seen


def _path_for(name: str) -> Path:
    """Resolve one library file from the local cache only - never the network."""
    if name not in EMOTE_FILES:
        raise UnknownEmote(name)
    try:
        # Import INSIDE the try. Where huggingface_hub is not installed the library really is
        # unreadable on this box, which is EmoteUnavailable - not a ModuleNotFoundError escaping
        # into a tool handler. The @_guard in shell/robot_tools.py would still catch it, but the
        # guest would hear an unhelpful sentence and the journal would get an unclassifiable one.
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(REPO_ID, EMOTE_FILES[name], repo_type=REPO_TYPE, local_files_only=True)
    except UnknownEmote:
        raise
    except Exception as e:  # LocalEntryNotFoundError, ImportError, and friends
        raise EmoteUnavailable(name, f"({type(e).__name__})") from e
    return Path(local)


def parse_emote(name: str, payload: dict, *, fallback_seconds: float = 2.4) -> Animation:
    """Frames -> an Animation in our own keyframe format (no legacy `AnimationKeyframe`).

    `payload` is the parsed JSON: {"time": [...], "set_target_data": [{head: 4x4, antennas: [2],
    body_yaw: float}]}. Splitting the file's real duration across the subsampled keyframes is
    what keeps a slow emote slow and a fast one fast.
    """
    frames = payload.get("set_target_data") or []
    times = payload.get("time") or []
    indices = _subsample(len(frames))
    if not indices:
        raise EmoteUnavailable(name, "(the library file has no keyframes)")

    total = float(times[-1]) if len(times) >= 2 and float(times[-1]) > 0 else fallback_seconds
    ceiling = emote_max_s()
    if ceiling and total > ceiling:
        total = ceiling  # scaled below, so the gesture keeps its shape and loses its drag
    keyframes: list[Keyframe] = []
    for idx in indices:
        frame = frames[idx]
        pose = frame.get("head")
        if pose is None:
            continue
        arr = pose6_from_matrix(np.asarray(pose, dtype=np.float64)) if _is_matrix(pose) else pose
        antennas = frame.get("antennas")
        keyframes.append(
            Keyframe(
                pose=arr,
                antennas=tuple(antennas) if antennas is not None and len(antennas) == 2 else None,
                body_yaw=frame.get("body_yaw"),
                # Keyframe.duration is the time to REACH this keyframe (see owner._step_animation),
                # so dividing by the count - not count-minus-one - is what makes the sum of the
                # keyframes equal the gesture's real length instead of overshooting it by a step.
                duration=max(total / len(indices), 0.05),
            )
        )
    if not keyframes:
        raise EmoteUnavailable(name, "(no head poses in the library file)")
    return Animation(name=f"emote:{name}", keyframes=keyframes, priority=0, blend_in=0.25, blend_out=0.35)


def _is_matrix(pose) -> bool:
    return hasattr(pose, "__len__") and len(pose) == 4 and hasattr(pose[0], "__len__") and len(pose[0]) == 4


def load(name: str) -> Animation:
    """Read and parse one emote from the local cache. Raises UnknownEmote / EmoteUnavailable."""
    path = _path_for(name)
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise EmoteUnavailable(name, f"({type(e).__name__})") from e
    anim = parse_emote(name, payload)
    anim.name = f"emote:{name}"
    return anim


def duration_s(name: str) -> float:
    return round(sum(kf.duration for kf in load(name).keyframes), 2)


#: The tool schema. Lives here, adjacent to EMOTE_FILES, because `enum` must be that map's keys
#: exactly - a vocabulary the model invents is a rejected call, and a vocabulary the library
#: dropped is a silent failure. tests/test_emotes.py asserts schema == map == parseable.
EMOTE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "play_emotion",
        "description": EMOTE_WHEN,
        "parameters": {
            "type": "object",
            "properties": {
                "emotion": {
                    "type": "string",
                    "enum": list(EMOTE_NAMES),
                    "description": "Which gesture to play",
                }
            },
            "required": ["emotion"],
        },
    },
}
EMOTE_TOOLS: list[dict] = [EMOTE_TOOL]


def status() -> dict:
    """What a preflight needs: can every vocabulary word actually be played right now?"""
    out: dict = {"repo": REPO_ID, "cached": [], "missing": [], "revision": None, "bytes": 0}
    for name in EMOTE_NAMES:
        try:
            path = _path_for(name)
        except (UnknownEmote, EmoteUnavailable) as e:
            out["missing"].append(f"{name}: {type(e).__name__}")
            continue
        try:
            out["bytes"] += path.stat().st_size  # Path.stat follows the cache symlink; `stat -c%s` does not
        except OSError:
            pass
        out["cached"].append(name)
        if out["revision"] is None:
            # .../<hub>/datasets--<org>--<repo>/snapshots/<revision>/<file>
            parts = Path(path).parts
            out["revision"] = parts[-2] if len(parts) >= 2 else None
    return out
