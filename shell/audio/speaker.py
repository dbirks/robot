"""The single speaker owner / output mixer (ADR 0002, invariant 4).

One OutputStream for the process lifetime. Every audible thing - realtime
TTS audio, prerendered acknowledgements, earcons, tool sounds - is enqueued
as a generation-tagged chunk on a named channel with a priority. On
interruption the current generation is cancelled: queued chunks for it are
dropped immediately and any late chunks still arriving for it are refused,
so stale speech can never become audible.

Priorities (highest wins each callback slot):
    sound > ack > tts
so an earcon or "hm?" lands over, not after, queued TTS.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..journal import JournalLike

PRIORITY = {"tts": 0, "ack": 1, "sound": 2}


@dataclass
class _Chunk:
    generation: int
    channel: str
    samples: list  # list[np.int16 array] assembled view; kept flat per chunk
    pos: int = 0


@dataclass
class SpeakerOwner:
    rate: int = 16000
    block: int = 512
    device: str | int | None = None
    journal: JournalLike | None = None
    _stream: Any = field(default=None, repr=False)
    _q: deque = field(default_factory=deque, repr=False)
    _cancelled: set = field(default_factory=set, repr=False)
    _generation: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    playing: bool = False
    level: float = 0.0

    # ---- generation control (owned by the realtime client) ----

    @property
    def generation(self) -> int:
        return self._generation

    def begin_generation(self) -> int:
        with self._lock:
            self._generation += 1
            # Keep cancelled generations refused; bound the set, never resurrect it.
            if len(self._cancelled) > 8:
                self._cancelled = {g for g in self._cancelled if g > self._generation - 8}
            return self._generation

    def cancel_current(self) -> bool:
        """Drop everything queued for the in-flight generation and refuse any
        late chunks for it. Returns True if audible audio was cut off."""
        with self._lock:
            self._cancelled.add(self._generation)
            had_tts = any(c.channel == "tts" for c in self._q)
            self._q = deque(c for c in self._q if c.channel != "tts")
        if self.journal:
            self.journal.write("tts.flushed", generation=self._generation, had_audio=had_tts)
        return had_tts

    # ---- enqueue / playback ----

    def enqueue(self, generation: int, channel: str, samples) -> bool:
        """Returns False if the generation was cancelled (late chunk refused)."""
        with self._lock:
            if generation in self._cancelled:
                return False
            self._q.append(_Chunk(generation=generation, channel=channel, samples=[samples]))
        return True

    def pending(self) -> int:
        with self._lock:
            return len(self._q)

    # ---- lifecycle ----

    def open(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.rate,
            channels=1,
            dtype="int16",
            blocksize=self.block,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ---- internal ----

    def _next_chunk_locked(self):
        best_i, best_prio = None, -1
        for i, c in enumerate(self._q):
            p = PRIORITY.get(c.channel, 0)
            if p > best_prio:
                best_i, best_prio = i, p
        if best_i is None:
            return None
        return self._q[best_i] if self._q[best_i].pos == 0 else self._q[best_i]

    def _callback(self, outdata, frames, time_info, status) -> None:
        import numpy as np

        outdata[:] = 0
        written = 0
        with self._lock:
            while written < frames:
                chunk = self._next_chunk_locked()
                if chunk is None:
                    break
                if chunk.pos == 0:
                    # pop it out of the queue; we own it now
                    self._q.remove(chunk)
                arr = chunk.samples[0]
                take = min(frames - written, len(arr) - chunk.pos)
                outdata[written : written + take, 0] = arr[chunk.pos : chunk.pos + take]
                written += take
                chunk.pos += take
                if chunk.pos < len(arr):
                    # put the remainder back at the front position
                    self._q.appendleft(chunk)
                    break
            self.playing = bool(self._q) or written > 0
        if written:
            self.level = float(np.sqrt(np.mean((outdata[:written, 0].astype(np.float32) / 32768.0) ** 2)))
