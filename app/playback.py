import logging

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


def play_audio(audio: np.ndarray, sample_rate: int):
    """Play int16 audio array through the default output device. Blocks until complete."""
    log.debug("Playing %d samples at %d Hz", len(audio), sample_rate)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()
