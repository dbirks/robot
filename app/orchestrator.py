import logging
import re
import threading
import time

import numpy as np
from reachy_mini.utils import create_head_pose

from .agent_client import AgentClient
from .audio_io import AudioRecorder
from .head_wobbler import HeadWobbler
from .movement_manager import MovementManager
from .playback import play_audio_with_wobble
from .stt_service import STTService
from .tts_service import TTSService

log = logging.getLogger(__name__)

WOBBLE_CHUNK_SIZE = 1600  # ~100ms at 16kHz

WAKE_PATTERNS = re.compile(
    r"wake\s*up|good\s*morning|time\s*to\s*(?:get\s*up|wake)|rise\s*and\s*shine|hey\s*robot|hello\s*robot|are\s*you\s*(?:there|awake)",
    re.IGNORECASE,
)

INIT_ANTENNAS = [-0.1745, 0.1745]


def run_loop(
    recorder: AudioRecorder,
    stt: STTService,
    agent: AgentClient,
    tts: TTSService,
    movement: MovementManager | None = None,
    wobbler: HeadWobbler | None = None,
    sleep_event: threading.Event | None = None,
    robot_mini=None,
    face_tracker=None,
):
    """Main conversation loop: listen -> transcribe -> agent -> TTS -> play."""
    log.info("Conversation loop started. Speak to begin.")

    while True:
        try:
            sleeping = sleep_event is not None and sleep_event.is_set()

            if not sleeping and movement:
                movement.set_listening(True)

            audio = recorder.record_utterance()

            if not sleeping and movement:
                movement.set_listening(False)

            if audio is None or len(audio) < 1600:
                continue

            loop_start = time.monotonic()

            text = stt.transcribe(audio)
            if not text.strip():
                log.debug("Empty transcription, skipping")
                continue

            if sleeping:
                if WAKE_PATTERNS.search(text):
                    log.info("Wake phrase detected: %s", text)
                    sleep_event.clear()
                    if robot_mini is not None:
                        robot_mini.goto_target(head=np.eye(4), antennas=INIT_ANTENNAS, duration=2)
                        time.sleep(0.5)
                    if movement:
                        movement.start()
                    if wobbler:
                        wobbler.start()
                    if face_tracker and robot_mini and movement:
                        face_tracker.start_tracking(robot_mini, movement, threading.Event())
                    greeting = tts.synthesize("Good morning! I'm awake.")
                    log.info("Assistant: Good morning! I'm awake.")
                    if wobbler:
                        play_audio_with_wobble(greeting, tts.sample_rate, wobbler)
                    else:
                        from .playback import play_audio
                        play_audio(greeting, tts.sample_rate)
                    continue
                else:
                    log.debug("Sleeping, ignoring: %s", text)
                    continue

            log.info("User: %s", text)

            if movement:
                movement.set_processing(True)

            response = agent.send(text)

            if movement:
                movement.set_processing(False)

            if sleep_event is not None and sleep_event.is_set():
                if face_tracker:
                    face_tracker.stop_tracking()
                if movement:
                    movement.stop()
                if wobbler:
                    wobbler.stop()
                log.info("Robot is now sleeping")
                continue

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
