import numpy as np
import soundfile as sf

from shell.reactions import WakeReaction


class FakeJournal:
    def __init__(self):
        self.events = []

    def write(self, type_, **payload):
        self.events.append(type_)


class FakeSpeaker:
    generation = 0

    def __init__(self):
        self.queued = []
        self.busy = 0

    def pending(self):
        return self.busy

    def enqueue(self, gen, channel, samples):
        self.queued.append(channel)
        return True


def make(tmp_path, **kw):
    sf.write(tmp_path / "hmm-0.wav", np.zeros(800, dtype=np.float32), 16000)
    sf.write(tmp_path / "hmm-1.wav", np.zeros(800, dtype=np.float32), 16000)
    clock = [100.0]
    r = WakeReaction(FakeSpeaker(), FakeJournal(), sound_dir=tmp_path, clock=lambda: clock[0], **kw)
    return r, clock


def test_filler_plays_as_ack(tmp_path):
    r, _ = make(tmp_path)
    r._fire()
    assert r.speaker.queued == ["ack"]
    assert r.journal.events == ["reaction.ack_played"]


def test_filler_rate_limited(tmp_path):
    r, clock = make(tmp_path, min_interval_s=4.0)
    r._fire()
    clock[0] += 1.0
    r._fire()  # second speculative revision of the same turn
    clock[0] += 4.0
    r._fire()
    assert r.speaker.queued == ["ack", "ack"]


def test_filler_skipped_when_speech_already_queued(tmp_path):
    r, _ = make(tmp_path)
    r.speaker.busy = 3
    r._fire()
    assert r.speaker.queued == []


def test_empty_pool_is_silent(tmp_path):
    r = WakeReaction(FakeSpeaker(), FakeJournal(), sound_dir=tmp_path / "missing")
    r.wake()
    assert r._timer is None
