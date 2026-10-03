from __future__ import annotations

import numpy as np
import pytest

from shell.audio import MicOwner, SpeakerOwner


class RecJournal:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def write(self, type_, **payload):
        self.events.append((type_, payload))

    def types(self):
        return [t for t, _ in self.events]

    def find(self, type_):
        return [p for t, p in self.events if t == type_]


@pytest.fixture
def journal():
    return RecJournal()


@pytest.fixture
def speaker(journal):
    return SpeakerOwner(journal=journal)


@pytest.fixture
def mic(journal):
    return MicOwner(device=None, journal=journal)


def cb_frames(block=512):
    return np.zeros((block, 1), dtype=np.int16)


def pump_mic(mic, n=4, value=1000):
    data = np.full((mic.block, 1), value, dtype=np.int16)
    for _ in range(n):
        mic._callback(data, mic.block, None, None)


def pump_speaker(speaker, n=4):
    out = cb_frames()
    for _ in range(n):
        speaker._callback(out, speaker.block, None, None)
    return out
