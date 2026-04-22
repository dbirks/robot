import logging
import time

from .agent_client import AgentClient
from .audio_io import AudioRecorder
from .playback import play_audio
from .stt_service import STTService
from .tts_service import TTSService

log = logging.getLogger(__name__)


def run_loop(
    recorder: AudioRecorder,
    stt: STTService,
    agent: AgentClient,
    tts: TTSService,
):
    """Main conversation loop: listen -> transcribe -> agent -> TTS -> play."""
    log.info("Conversation loop started. Speak to begin.")

    while True:
        try:
            audio = recorder.record_utterance()
            if audio is None or len(audio) < 1600:
                continue

            loop_start = time.monotonic()

            text = stt.transcribe(audio)
            if not text.strip():
                log.debug("Empty transcription, skipping")
                continue

            log.info("User: %s", text)

            response = agent.send(text)
            if not response.strip():
                log.debug("Empty agent response, skipping")
                continue

            log.info("Assistant: %s", response)

            tts_audio = tts.synthesize(response)
            play_audio(tts_audio, tts.sample_rate)

            log.info("Turn complete in %.2fs", time.monotonic() - loop_start)

        except KeyboardInterrupt:
            log.info("Interrupted, shutting down")
            break
        except Exception:
            log.exception("Error in conversation loop")
