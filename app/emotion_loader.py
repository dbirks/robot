"""Load emotion animations from the pollen-robotics/reachy-mini-emotions-library HuggingFace dataset.

Subsamples the dense keyframe data (~100 Hz) down to ~10-20 keyframes per emotion
and converts them into AnimationKeyframe lists for use with MovementManager.
"""

import json
import logging
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from .movement_manager import AnimationKeyframe

log = logging.getLogger(__name__)

REPO_ID = "pollen-robotics/reachy-mini-emotions-library"
REPO_TYPE = "dataset"

# Map friendly names to dataset filenames
EMOTION_FILES: dict[str, str] = {
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

# Cache of loaded emotion keyframes
_emotion_cache: dict[str, list[AnimationKeyframe]] = {}


def _subsample_indices(n: int, target: int = 15) -> list[int]:
    """Pick ~target evenly-spaced indices from 0..n-1, always including first and last."""
    if n <= target:
        return list(range(n))
    step = (n - 1) / (target - 1)
    indices = [round(i * step) for i in range(target)]
    # Deduplicate while preserving order
    seen = set()
    result = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


def _parse_emotion_file(path: str | Path) -> list[AnimationKeyframe]:
    """Parse a single emotion JSON file into a list of AnimationKeyframe."""
    with open(path) as f:
        data = json.load(f)

    times = data["time"]
    frames = data["set_target_data"]
    n = len(frames)

    if n == 0:
        return []

    # Subsample to ~15 keyframes
    indices = _subsample_indices(n, target=15)

    keyframes = []
    for i, idx in enumerate(indices):
        frame = frames[idx]
        pose = np.array(frame["head"], dtype=np.float64)
        antennas = frame.get("antennas")
        body_yaw = frame.get("body_yaw", 0.0)

        # Calculate duration: time delta from previous sampled frame
        if i == 0:
            duration = times[idx] if idx > 0 else 0.1
        else:
            prev_idx = indices[i - 1]
            duration = times[idx] - times[prev_idx]

        # Ensure minimum duration
        duration = max(duration, 0.01)

        keyframes.append(
            AnimationKeyframe(
                pose=pose,
                antennas=list(antennas) if antennas is not None else None,
                body_yaw=float(body_yaw) if body_yaw else None,
                duration=duration,
            )
        )

    return keyframes


def load_emotion(name: str) -> list[AnimationKeyframe] | None:
    """Load a named emotion, downloading from HuggingFace if needed.

    Returns a list of AnimationKeyframe, or None if the emotion is not found.
    Results are cached after first load.
    """
    if name in _emotion_cache:
        return _emotion_cache[name]

    filename = EMOTION_FILES.get(name)
    if filename is None:
        log.warning("Unknown emotion: %s (available: %s)", name, ", ".join(EMOTION_FILES.keys()))
        return None

    try:
        local_path = hf_hub_download(REPO_ID, filename, repo_type=REPO_TYPE)
        keyframes = _parse_emotion_file(local_path)
        _emotion_cache[name] = keyframes
        log.info("Loaded emotion '%s' from %s (%d keyframes)", name, filename, len(keyframes))
        return keyframes
    except Exception:
        log.exception("Failed to load emotion '%s'", name)
        return None


def preload_emotions():
    """Pre-download and parse all configured emotions. Call at startup."""
    for name in EMOTION_FILES:
        load_emotion(name)


def available_emotions() -> list[str]:
    """Return the list of available emotion names."""
    return list(EMOTION_FILES.keys())
