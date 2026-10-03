import numpy as np
from conftest import pump_mic, pump_speaker


def test_playback_drains_queue(speaker):
    gen = speaker.begin_generation()
    tone = np.full(200, 1000, dtype=np.int16)
    assert speaker.enqueue(gen, "tts", tone)
    speaker.enqueue(gen, "tts", tone)
    assert speaker.pending() == 2
    pump_speaker(speaker)
    assert speaker.pending() == 0


def test_cancel_drops_queued_and_refuses_late_chunks(speaker):
    gen = speaker.begin_generation()
    tone = np.full(speaker.block, 500, dtype=np.int16)
    speaker.enqueue(gen, "tts", tone)
    speaker.enqueue(gen, "tts", tone)

    assert speaker.cancel_current() is True  # audible audio was cut
    assert speaker.pending() == 0
    # a stale TTS chunk still in flight for the cancelled generation:
    assert speaker.enqueue(gen, "tts", tone) is False
    assert speaker.pending() == 0
    assert "tts.flushed" in speaker.journal.types()


def test_new_generation_after_cancel_is_accepted(speaker):
    old = speaker.begin_generation()
    speaker.cancel_current()
    new = speaker.begin_generation()
    assert new != old
    assert speaker.enqueue(new, "tts", np.zeros(speaker.block, dtype=np.int16))


def test_sound_channel_preempts_tts(speaker):
    gen = speaker.begin_generation()
    long_tts = np.full(speaker.block * 3, 100, dtype=np.int16)
    beep = np.full(speaker.block, 9999, dtype=np.int16)
    speaker.enqueue(gen, "tts", long_tts)
    speaker.enqueue(gen, "sound", beep)

    # first block should be the sound channel, not the tts
    out = np.zeros((speaker.block, 1), dtype=np.int16)
    speaker._callback(out, speaker.block, None, None)
    assert np.all(out[:, 0] == 9999)
    assert speaker.pending() == 1  # tts remainder still queued


def test_mic_fanout_and_preroll(mic):
    got = []
    mic.subscribe(got.append)
    pump_mic(mic, n=3)
    assert len(got) == 3 and got[0]
    pre = mic.preroll(0.05)
    assert len(pre) > 0 and len(pre) <= int(0.05 * mic.rate * 2) + 2048


def test_mic_mute_stops_fanout_but_keeps_health(mic):
    got = []
    mic.subscribe(got.append)
    mic.set_muted(True)
    pump_mic(mic, n=2)
    assert got == []
    h = mic.health()
    assert h["muted"] is True
    assert h["blocks_total"] == 2  # capture itself never stops on mute


def test_bad_subscriber_does_not_kill_capture(mic):
    def boom(_pcm):
        raise RuntimeError("subscriber exploded")

    good = []
    mic.subscribe(boom)
    mic.subscribe(good.append)
    pump_mic(mic, n=2)
    assert len(good) == 2
    assert mic.blocks_dropped == 2
    assert "audio.subscriber_error" in mic.journal.types()


def test_chunk_longer_than_block_plays_once_and_never_spins(speaker):
    # Regression: a multi-block chunk used to be duplicated on resume, and an
    # exhausted duplicate made the callback loop forever holding the lock.
    gen = speaker.begin_generation()
    speech = np.arange(speaker.block * 3 + 100, dtype=np.int16)
    speaker.enqueue(gen, "tts", speech)
    out = pump_speaker(speaker, n=6)
    assert speaker.pending() == 0
    assert out is not None
    assert speaker.enqueue(gen, "tts", speech)  # lock is free, queue usable


def test_resumed_chunk_audio_is_contiguous(speaker):
    gen = speaker.begin_generation()
    speech = np.arange(speaker.block * 2 + 7, dtype=np.int16)
    speaker.enqueue(gen, "tts", speech)
    played = []
    for _ in range(4):
        buf = np.zeros((speaker.block, 1), dtype=np.int16)
        speaker._callback(buf, speaker.block, None, None)
        played.append(buf[:, 0].copy())
    got = np.concatenate(played)[: len(speech)]
    assert np.array_equal(got, speech)
