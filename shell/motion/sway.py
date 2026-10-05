"""Speech sway: Reachy's head moves with his own voice.

Driven by the PCM the speaker owner actually writes to the device (its
`tap`), analysed with the SDK's SwayRollRT (the tuned conversation-app
speech tapper). LeanSwayRollRT reuses its maths and constants verbatim but
keeps only the last analysis frame: upstream appends every sample to a 10 s
deque via .tolist() and islices from its start each hop, measured here at
2.1% of a core while talking vs 0.04% for this (ADR 0006); outputs are
identical (tests/test_sway.py).
Two rules from the legacy bugs:

- the audio callback only appends a copy to a bounded deque; all DSP runs
  on the motion thread (no work in the real-time audio path)
- we do NOT enable the daemon's own wobbling (ReachyMini.enable_wobbling):
  it would wobble a second time on top of this

Offsets come out in the MotionOwner 6-vector convention
[x, y, z (m), roll, pitch, yaw (rad)] and decay smoothly to zero when the
audio stops, so a cut-off reply never leaves the head mid-sway.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

from .kinematics import exp_alpha

SILENCE_AFTER_S = 0.15  # no audio this long: playback ended or was cut
DECAY_TAU = 0.15


class SpeechSway:
    def __init__(self, rate: int = 16000, max_chunks: int = 64) -> None:
        self.rate = rate
        self._q: deque = deque(maxlen=max_chunks)  # ~2 s of 32 ms blocks; drops oldest
        self._lock = threading.Lock()
        self._rt = None
        try:  # build now: a first-speech import would stall the motion loop
            self._rt = make_lean_sway(rate)
        except Exception:
            pass
        self._offsets = np.zeros(6)
        self._last_audio: float | None = None
        self._last_step: float | None = None

    def feed(self, pcm) -> None:
        """Audio-callback side: copy and enqueue, nothing else."""
        chunk = np.array(pcm, dtype=np.int16, copy=True)
        with self._lock:
            self._q.append(chunk)

    def _tapper(self):
        if self._rt is None:
            self._rt = make_lean_sway(self.rate)
        return self._rt

    def step(self, now: float) -> np.ndarray:
        with self._lock:
            chunks = list(self._q)
            self._q.clear()
        dt = 0.02 if self._last_step is None else max(0.0, now - self._last_step)
        self._last_step = now
        if chunks:
            pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
            hops = self._tapper().feed(pcm)
            self._last_audio = now
            if hops:
                h = hops[-1]
                self._offsets = np.array(
                    [
                        h["x_mm"] / 1000.0,
                        h["y_mm"] / 1000.0,
                        h["z_mm"] / 1000.0,
                        h["roll_rad"],
                        h["pitch_rad"],
                        h["yaw_rad"],
                    ]
                )
            return self._offsets.copy()
        if self._last_audio is not None and now - self._last_audio > SILENCE_AFTER_S:
            self._offsets = self._offsets * (1.0 - exp_alpha(dt, DECAY_TAU))
            if np.all(np.abs(self._offsets) < 1e-5):
                self._offsets = np.zeros(6)
                self._last_audio = None
                if self._rt is not None:
                    self._rt.reset()
        return self._offsets.copy()


def make_lean_sway(rate: int = 16000):
    """SwayRollRT whose feed() keeps only the last frame (same outputs)."""
    import math

    from reachy_mini.motion import speech_tapper as st

    class LeanSwayRollRT(st.SwayRollRT):
        def reset(self) -> None:
            super().reset()
            self._tail = np.zeros(0, dtype=np.float32)

        def feed(self, pcm):  # mirrors SwayRollRT.feed, minus the 10 s history
            if pcm.size == 0:
                return []
            self.carry = np.concatenate([self.carry, pcm]) if self.carry.size else pcm
            tail = getattr(self, "_tail", np.zeros(0, dtype=np.float32))
            out = []
            while self.carry.size >= self.hop:
                hop = self.carry[: self.hop]
                self.carry = self.carry[self.hop :]
                tail = np.concatenate([tail, hop])[-self.frame :]
                if tail.size < self.frame:
                    self.t += st.HOP_MS / 1000.0
                    continue
                db = st._rms_dbfs(tail)
                if db >= st.VAD_DB_ON:
                    self.vad_above += 1
                    self.vad_below = 0
                    if not self.vad_on and self.vad_above >= st.ATTACK_FR:
                        self.vad_on = True
                elif db <= st.VAD_DB_OFF:
                    self.vad_below += 1
                    self.vad_above = 0
                    if self.vad_on and self.vad_below >= st.RELEASE_FR:
                        self.vad_on = False
                if self.vad_on:
                    self.sway_up = min(st.SWAY_ATTACK_FR, self.sway_up + 1)
                    self.sway_down = 0
                else:
                    self.sway_down = min(st.SWAY_RELEASE_FR, self.sway_down + 1)
                    self.sway_up = 0
                up = self.sway_up / st.SWAY_ATTACK_FR
                down = 1.0 - (self.sway_down / st.SWAY_RELEASE_FR)
                target = up if self.vad_on else down
                self.sway_env = min(1.0, max(0.0, self.sway_env + st.ENV_FOLLOW_GAIN * (target - self.sway_env)))
                k = st._loudness_gain(db) * st.SWAY_MASTER * self.sway_env
                self.t += st.HOP_MS / 1000.0
                w = 2 * math.pi * self.t
                out.append(
                    {
                        "pitch_rad": math.radians(st.SWAY_A_PITCH_DEG)
                        * k
                        * math.sin(w * st.SWAY_F_PITCH + self.phase_pitch),
                        "yaw_rad": math.radians(st.SWAY_A_YAW_DEG) * k * math.sin(w * st.SWAY_F_YAW + self.phase_yaw),
                        "roll_rad": math.radians(st.SWAY_A_ROLL_DEG)
                        * k
                        * math.sin(w * st.SWAY_F_ROLL + self.phase_roll),
                        "x_mm": st.SWAY_A_X_MM * k * math.sin(w * st.SWAY_F_X + self.phase_x),
                        "y_mm": st.SWAY_A_Y_MM * k * math.sin(w * st.SWAY_F_Y + self.phase_y),
                        "z_mm": st.SWAY_A_Z_MM * k * math.sin(w * st.SWAY_F_Z + self.phase_z),
                    }
                )
            self._tail = tail
            return out

    return LeanSwayRollRT(sample_rate=rate)
