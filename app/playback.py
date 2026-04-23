import logging

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

WOBBLE_CHUNK_SAMPLES = 1600  # ~100ms at 16kHz


def play_audio(audio: np.ndarray, sample_rate: int):
    """Play int16 audio array through the default output device. Blocks until complete."""
    log.debug("Playing %d samples at %d Hz", len(audio), sample_rate)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def play_audio_with_wobble(audio: np.ndarray, sample_rate: int, wobbler):
    """Play int16 audio while feeding chunks to the HeadWobbler for speech motion."""
    log.debug("Playing %d samples at %d Hz (with wobble)", len(audio), sample_rate)

    for i in range(0, len(audio), WOBBLE_CHUNK_SAMPLES):
        chunk = audio[i : i + WOBBLE_CHUNK_SAMPLES]
        wobbler.feed(chunk)

    sd.play(audio, samplerate=sample_rate)
    sd.wait()
    wobbler.reset()
