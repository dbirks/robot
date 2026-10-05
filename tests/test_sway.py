"""Speech sway + conversation-state hooks, hardware-free."""

import asyncio
import math
import random

import numpy as np
import pytest

from shell.motion import MotionOwner, SpeechSway
from shell.motion.owner import LISTENING_MAX_S
from tests.conftest import pump_speaker


def voice(seconds, rate=16000, amp=0.3):
    t = np.arange(int(seconds * rate)) / rate
    # vowel-ish: 150 Hz fundamental with a 4 Hz syllable envelope
    x = amp * np.sin(2 * math.pi * 150 * t) * (0.6 + 0.4 * np.sin(2 * math.pi * 4 * t))
    return (x * 32767).astype(np.int16)


def test_sway_moves_with_voice_and_decays_after():
    s = SpeechSway()
    pcm = voice(2.0)
    now, peak = 0.0, 0.0
    for i in range(0, len(pcm), 512):  # 32 ms blocks, stepped at the same pace
        s.feed(pcm[i : i + 512])
        now += 0.032
        off = s.step(now)
        peak = max(peak, abs(off[5]), abs(off[4]))
    assert peak > math.radians(1.0)
    for _ in range(100):  # 2 s of silence
        now += 0.02
        off = s.step(now)
    assert np.all(off == 0.0)


def test_lean_tapper_matches_sdk_exactly():
    from reachy_mini.motion.speech_tapper import SwayRollRT

    from shell.motion.sway import make_lean_sway

    ref, lean = SwayRollRT(sample_rate=16000), make_lean_sway(16000)
    pcm = voice(3.0).astype(np.float32) / 32768.0
    a, b = [], []
    for i in range(0, len(pcm), 512):
        a += ref.feed(pcm[i : i + 512])
        b += lean.feed(pcm[i : i + 512])
    assert len(a) == len(b) > 50
    for x, y in zip(a, b):
        for k in x:
            assert y[k] == pytest.approx(x[k], abs=1e-6)


def test_sway_silent_input_stays_still():
    s = SpeechSway()
    for i in range(30):
        s.feed(np.zeros(512, dtype=np.int16))
        off = s.step(i * 0.032)
    assert np.allclose(off, 0.0, atol=1e-6)


def test_sway_feed_is_bounded():
    s = SpeechSway(max_chunks=4)
    for _ in range(100):
        s.feed(np.ones(512, dtype=np.int16))
    assert len(s._q) == 4


def test_speaker_tap_gets_exactly_what_was_played(speaker):
    got = []
    speaker.tap = lambda pcm: got.append(np.array(pcm))
    speaker.enqueue(speaker.generation, "tts", np.arange(700, dtype=np.int16))
    pump_speaker(speaker, n=3)
    played = np.concatenate(got)
    assert np.array_equal(played, np.arange(700, dtype=np.int16))


def test_speaker_survives_a_broken_tap(speaker):
    def bad(pcm):
        raise RuntimeError("motion bug")

    speaker.tap = bad
    speaker.enqueue(speaker.generation, "tts", np.ones(600, dtype=np.int16))
    pump_speaker(speaker, n=2)  # must not raise
    assert speaker.level > 0


def test_owner_applies_sway_offsets():
    class Stub:
        def step(self, now):
            return np.array([0, 0, 0, 0, 0, math.radians(4)])

    clock = [10.0]
    m = MotionOwner(None, clock=lambda: clock[0], rng=random.Random(0), sway=Stub())
    clock[0] += 0.02
    assert math.degrees(m.tick().pose[5]) == pytest.approx(4, abs=0.01)


def test_listening_state_times_out():
    clock = [10.0]
    m = MotionOwner(None, clock=lambda: clock[0], rng=random.Random(0))
    m.set_listening(True)
    clock[0] += LISTENING_MAX_S + 1
    m.tick()
    assert m._listening is False


def test_client_hooks_fire(journal):
    from shell.realtime.client import RealtimeClient

    calls = []

    class Spk:
        generation = 1

        def begin_generation(self):
            return 1

        def enqueue(self, *a):
            return True

        def cancel_current(self):
            return False

    c = RealtimeClient(
        "ws://x",
        instructions="",
        tools=[],
        tool_router=None,
        speaker=Spk(),
        journal=journal,
        on_speech_stopped=lambda: calls.append("stopped"),
        on_first_audio=lambda: calls.append("first_audio"),
    )
    pcm = np.zeros(10, dtype=np.int16).tobytes()
    import base64

    async def go():
        await c.handle_event({"type": "input_audio_buffer.speech_stopped"})
        await c.handle_event({"type": "response.created", "response": {"id": "r1"}})
        delta = {"type": "response.output_audio.delta", "response_id": "r1", "delta": base64.b64encode(pcm).decode()}
        await c.handle_event(delta)
        await c.handle_event(delta)

    asyncio.run(go())
    assert calls == ["stopped", "first_audio"]
