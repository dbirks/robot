import logging
import os

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

WOBBLE_CHUNK_SAMPLES = 1600  # ~100ms at 16kHz


def _get_output_device() -> int | str | None:
    env = os.getenv("AUDIO_OUTPUT_DEVICE", "")
    if env:
        return int(env) if env.isdigit() else env
    for i, d in enumerate(sd.query_devices()):
        if "Reachy Mini" in d["name"] and d["max_output_channels"] > 0:
            return i
    return None


_output_device = _get_output_device()
_device_sr: int | None = None
if _output_device is not None:
    _device_sr = int(sd.query_devices(_output_device)["default_samplerate"])
    log.info("Audio output: device %s (%d Hz)", _output_device, _device_sr)


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    ratio = dst_rate / src_rate
    n_samples = int(len(audio) * ratio)
    indices = np.arange(n_samples) / ratio
    return np.interp(indices, np.arange(len(audio)), audio.astype(np.float64)).astype(np.int16)


VOLUME_BOOST = float(os.getenv("VOLUME_BOOST", "1.4"))


def set_volume_boost(boost: float):
    global VOLUME_BOOST
    VOLUME_BOOST = boost
    log.info("Volume boost set to %.1f", boost)


def _apply_boost(audio: np.ndarray) -> np.ndarray:
    if VOLUME_BOOST == 1.0:
        return audio
    boosted = np.clip(audio.astype(np.float64) * VOLUME_BOOST, -32768, 32767)
    return boosted.astype(np.int16)


def _play(audio: np.ndarray, sample_rate: int):
    audio = _apply_boost(audio)
    if _device_sr is not None and sample_rate != _device_sr:
        audio = _resample(audio, sample_rate, _device_sr)
        sample_rate = _device_sr
    sd.play(audio, samplerate=sample_rate, device=_output_device)
    sd.wait()


def play_audio(audio: np.ndarray, sample_rate: int):
    """Play int16 audio array through the output device. Blocks until complete."""
    log.debug("Playing %d samples at %d Hz", len(audio), sample_rate)
    _play(audio, sample_rate)


def play_audio_with_wobble(audio: np.ndarray, sample_rate: int, wobbler):
    """Play int16 audio while feeding chunks to the HeadWobbler for speech motion."""
    log.debug("Playing %d samples at %d Hz (with wobble)", len(audio), sample_rate)

    for i in range(0, len(audio), WOBBLE_CHUNK_SAMPLES):
        chunk = audio[i : i + WOBBLE_CHUNK_SAMPLES]
        wobbler.feed(chunk)

    _play(audio, sample_rate)
    wobbler.reset()
