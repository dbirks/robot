import logging
import threading
import time

import numpy as np
from reachy_mini.utils import create_head_pose

from .agent_client import AgentClient
from .audio_io import AudioRecorder
from .head_wobbler import HeadWobbler
from .movement_manager import MovementManager
from .playback import play_audio_with_wobble, play_sentence_with_wobble
from .stt_service import STTService
from .tts_service import TTSService
from .wake_detector import WakeDetector

log = logging.getLogger(__name__)

WOBBLE_CHUNK_SIZE = 1600  # ~100ms at 16kHz

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
    wake_detector: WakeDetector | None = None,
    doa_tracker=None,
):
    """Main conversation loop: listen -> transcribe -> agent -> TTS -> play."""
    log.info("Conversation loop started. Speak to begin.")

    while True:
        try:
            sleeping = sleep_event is not None and sleep_event.is_set()

            if not sleeping and movement:
                movement.set_listening(True)
            if not sleeping and doa_tracker:
                doa_tracker.set_locked(True)

            audio = recorder.record_utterance()

            if not sleeping and movement:
                movement.set_listening(False)
            if not sleeping and doa_tracker:
                doa_tracker.set_locked(False)

            if audio is None or len(audio) < 1600:
                continue

            loop_start = time.monotonic()

            text = stt.transcribe(audio)
            if not text.strip():
                log.debug("Empty transcription, skipping")
                continue

            if sleeping:
                should_wake = False
                if wake_detector:
                    should_wake = wake_detector.should_wake(text)
                else:
                    should_wake = any(w in text.lower() for w in ("wake up", "good morning", "hey robot"))

                if should_wake:
                    log.info("Waking up from: %s", text)
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
                else:
                    continue

                log.info("User: %s", text)

                if movement:
                    movement.set_processing(True)

                response = agent.send(text)

                if movement:
                    movement.set_processing(False)

                if not response.strip():
                    continue

                log.info("Assistant: %s", response)
                tts_audio = tts.synthesize(response)
                if wobbler:
                    play_audio_with_wobble(tts_audio, tts.sample_rate, wobbler)
                else:
                    from .playback import play_audio

                    play_audio(tts_audio, tts.sample_rate)
                continue

            log.info("User: %s", text)

            if movement:
                movement.set_processing(True)

            first_sentence = True
            full_response = []
            for sentence in agent.send_streaming(text):
                if first_sentence and movement:
                    movement.set_processing(False)
                    movement.set_speaking(True)
                    first_sentence = False

                full_response.append(sentence)

                tts_audio = tts.synthesize(sentence)
                if wobbler:
                    play_sentence_with_wobble(tts_audio, tts.sample_rate, wobbler)
                else:
                    from .playback import play_audio

                    play_audio(tts_audio, tts.sample_rate)

                if sleep_event is not None and sleep_event.is_set():
                    break

            if first_sentence and movement:
                movement.set_processing(False)
            if movement:
                movement.set_speaking(False)

            response_text = " ".join(full_response)
            if response_text:
                log.info("Assistant: %s", response_text)

            if sleep_event is not None and sleep_event.is_set():
                if face_tracker:
                    face_tracker.stop_tracking()
                if movement:
                    movement.stop()
                if wobbler:
                    wobbler.stop()
                log.info("Robot is now sleeping")
                continue

            log.info("Turn complete in %.2fs", time.monotonic() - loop_start)

        except KeyboardInterrupt:
            log.info("Interrupted, shutting down")
            break
        except Exception:
            log.exception("Error in conversation loop")
