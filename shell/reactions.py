"""Wake reaction: turn FIRST, acknowledge lazily, shut up if they keep talking.

EPIC order of importance: the sub-100ms-class physical reaction beats a
clever spoken acknowledgement. On a wake phrase:
  1. head-turn toward DOA fires immediately (callback, not the LLM)
  2. a prerendered acknowledgement from a small rotating pool is scheduled
     after 250-400ms and bypasses TTS entirely
  3. if the person continues speaking before it fires, it is cancelled
  4. no repeated grunts while already attending to the same participant
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


class WakeReaction:
    def __init__(
        self,
        speaker,
        journal,
        *,
        sound_dir: str | Path = "sounds/acks",
        delay_range_s: tuple[float, float] = (0.25, 0.40),
        rate: int = 16000,
        on_wake=None,  # callable(doa_deg: float|None) - immediate head-turn
        rng: random.Random | None = None,
        loop=None,
    ) -> None:
        self.speaker = speaker
        self.journal = journal
        self.delay_range_s = delay_range_s
        self.rate = rate
        self.on_wake = on_wake
        self._rng = rng or random.Random()
        self._loop = loop
        self._timer = None
        self._pool = sorted(Path(sound_dir).glob("*.wav"))
        self._last_pick: Path | None = None
        self._cache: dict[Path, np.ndarray] = {}

    def wake(self, doa_deg: float | None = None, attending_same: bool = False) -> None:
        if self.on_wake:
            self.on_wake(doa_deg)
        if attending_same or not self._pool:
            return
        delay = self._rng.uniform(*self.delay_range_s)
        self.cancel()
        if self._loop is not None:
            self._timer = self._loop.call_later(delay, self._fire)
        else:
            import threading

            self._timer = threading.Timer(delay, self._fire)
            self._timer.start()

    def cancel(self) -> None:
        """Person kept talking - suppress the acknowledgement."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _fire(self) -> None:
        self._timer = None
        pick = self._pick()
        samples = self._load(pick)
        gen = self.speaker.generation
        if self.speaker.enqueue(gen, "ack", samples):
            self.journal.write("reaction.ack_played", sound=pick.name)

    def _pick(self) -> Path:
        pool = [p for p in self._pool if p != self._last_pick] or self._pool
        self._last_pick = self._rng.choice(pool)
        return self._last_pick

    def _load(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="int16")
            if data.ndim > 1:
                data = data[:, 0]
            if sr != self.rate:
                import numpy as np

                idx = (np.arange(len(data)) * self.rate / sr).astype(int)
                idx = idx[idx < len(data)]
                data = data[idx]
            self._cache[path] = np.ascontiguousarray(data)
        return self._cache[path]
