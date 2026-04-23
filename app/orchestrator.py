import logging
import threading
import time

import numpy as np

from .agent_client import AgentClient
from .audio_io import AudioRecorder
from .head_wobbler import HeadWobbler
from .movement_manager import MovementManager
from .playback import play_audio_with_wobble
from .stt_service import STTService
from .tts_service import TTSService

log = logging.getLogger(__name__)

WOBBLE_CHUNK_SIZE = 1600  # ~100ms at 16kHz


def run_loop(
    recorder: AudioRecorder,
    stt: STTService,
    agent: AgentClient,
    tts: TTSService,
    movement: MovementManager | None = None,
    wobbler: HeadWobbler | None = None,
):
    """Main conversation loop: listen -> transcribe -> agent -> TTS -> play."""
    log.info("Conversation loop started. Speak to begin.")

    while True:
        try:
            if movement:
                movement.set_listening(True)

            audio = recorder.record_utterance()

            if movement:
                movement.set_listening(False)

            if audio is None or len(audio) < 1600:
                continue

            loop_start = time.monotonic()

            text = stt.transcribe(audio)
            if not text.strip():
                log.debug("Empty transcription, skipping")
                continue

            log.info("User: %s", text)

            if movement:
                movement.set_processing(True)

            response = agent.send(text)

            if movement:
                movement.set_processing(False)

            if not response.strip():
                log.debug("Empty agent response, skipping")
                continue

            log.info("Assistant: %s", response)

            tts_audio = tts.synthesize(response)

            if wobbler:
                play_audio_with_wobble(tts_audio, tts.sample_rate, wobbler)
            else:
                from .playback import play_audio

                play_audio(tts_audio, tts.sample_rate)

            log.info("Turn complete in %.2fs", time.monotonic() - loop_start)

        except KeyboardInterrupt:
            log.info("Interrupted, shutting down")
            break
        except Exception:
            log.exception("Error in conversation loop")
