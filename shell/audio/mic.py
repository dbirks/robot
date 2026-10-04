"""The single microphone owner (ADR 0002).

Exactly one InputStream is ever opened, at open(), for the process lifetime.
Everything else - the realtime client, KWS, ambient STT, the watchdog,
experiments - subscribes to the fan-out. Nothing else may touch a capture
device; a mute is a first-class state here, never a flag file.

The sounddevice import happens inside open() so this module is importable
and unit-testable on machines without audio hardware.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

Subscriber = Callable[[bytes], None]  # callback(pcm_s16le_mono), device-thread


class MicOwner:
    def __init__(
        self,
        device: str | int | None,
        rate: int = 16000,
        block: int = 512,
        preroll_seconds: float = 1.0,
        journal=None,
    ) -> None:
        self.device = device
        self.rate = rate
        self.block = block
        self.journal = journal
        self._subs: list[Subscriber] = []
        self._ring: deque[bytes] = deque(maxlen=max(1, int(preroll_seconds * rate * 2 / block)))
        self._lock = threading.Lock()
        self._stream = None
        self.muted = False
        self.blocks_total = 0
        self.blocks_dropped = 0
        self.rms_recent = 0.0
        self.rms_peak = 0.0
        self._last_active_wall = time.monotonic()
        self._last_sound = time.monotonic()  # last block with any nonzero sample
        self._np = __import__("numpy")  # lazy-ish; cheap module

    # ---- subscription API (the ONLY way to get mic audio) ----

    def subscribe(self, cb: Subscriber) -> None:
        with self._lock:
            self._subs.append(cb)

    def unsubscribe(self, cb: Subscriber) -> None:
        with self._lock:
            if cb in self._subs:
                self._subs.remove(cb)

    def preroll(self, seconds: float | None = None) -> bytes:
        """Recently-seen audio for interruption preservation (invariant 4)."""
        with self._lock:
            blocks = list(self._ring)
        data = b"".join(blocks)
        if seconds is not None:
            data = data[-int(seconds * self.rate * 2) :]
        return data

    # ---- lifecycle ----

    def open(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
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

    # ---- health signals for the watchdog (watchdog never opens a stream) ----

    def health(self) -> dict:
        import subprocess  # stdlib; wpctl is the PipeWire control surface

        h = {
            "muted": self.muted,
            "rms_recent": round(self.rms_recent, 5),
            "rms_peak": round(self.rms_peak, 5),
            "blocks_total": self.blocks_total,
            "blocks_dropped": self.blocks_dropped,
            "seconds_since_last_block": round(time.monotonic() - self._last_active_wall, 2),
            # A live room is never exactly digital zero; sustained exact zero
            # while blocks still flow is the XVF3800 firmware stall (ADR 0005).
            "seconds_since_sound": round(time.monotonic() - self._last_sound, 2),
        }
        # Distinguish an explicit mute from dead firmware (ADR 0005). Best-effort:
        # if we cannot ask PipeWire, report unknown rather than guessing.
        try:
            # There is no `wpctl get-mute`; get-volume prints "Volume: 1.00"
            # plus " [MUTED]" when muted. No "Volume:" line => unknown.
            out = subprocess.run(
                ["wpctl", "get-volume", "@DEFAULT_SOURCE@"],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            line = next((ln for ln in out.stdout.splitlines() if ln.startswith("Volume:")), None)
            h["pipewire_mute"] = None if line is None else "[MUTED]" in line
        except Exception:
            h["pipewire_mute"] = None
        return h

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        if self.journal:
            self.journal.write("audio.mute_changed", muted=muted)

    # ---- internal ----

    def _callback(self, indata, frames, time_info, status) -> None:
        self.blocks_total += 1
        self._last_active_wall = time.monotonic()
        mono = indata[:, 0]
        pcm = bytes(mono.tobytes())
        rms = float(self._np.sqrt(self._np.mean((mono.astype(self._np.float32) / 32768.0) ** 2)))
        self.rms_recent = rms
        if rms > 0.0:
            self._last_sound = self._last_active_wall
        self.rms_peak = max(self.rms_peak, rms)
        if not self.muted:
            with self._lock:
                self._ring.append(pcm)
                subs = list(self._subs)
            for cb in subs:
                try:
                    cb(pcm)
                except Exception as e:  # a bad subscriber must not kill capture
                    self.blocks_dropped += 1
                    if self.journal:
                        self.journal.write("audio.subscriber_error", error=repr(e))
