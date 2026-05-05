import logging
import threading
import time

import numpy as np
from reachy_mini.utils import create_head_pose

from .agent_client import PROCESSING_SENTINEL, AgentClient
from .audio_io import AudioRecorder
from .head_wobbler import HeadWobbler
from .interruptible_player import InterruptiblePlayer
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
    player: InterruptiblePlayer | None = None,
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

            record_start = time.monotonic()
            audio = recorder.record_utterance()
            record_end = time.monotonic()

            if not sleeping and movement:
                movement.set_listening(False)
            if not sleeping and doa_tracker:
                doa_tracker.set_locked(False)

            if audio is None or len(audio) < 1600:
                continue

            # Discard stale audio — if recording took too long, the speech
            # likely happened during a previous response and isn't directed at us
            audio_duration = len(audio) / recorder.sample_rate
            if audio_duration > 30:
                log.info("Discarding stale audio (%.1fs — too long)", audio_duration)
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

            # Filter ambient speech — only respond when spoken to directly
            if wake_detector:
                intent = wake_detector.classify_utterance(text)
                if intent == "ignore":
                    log.debug("Ignoring ambient speech: %s", text)
                    continue
                elif intent == "check":
                    log.info("Uncertain speech, asking to clarify: %s", text)
                    check_audio = tts.synthesize("Sorry, were you talking to me?")
                    if wobbler:
                        play_sentence_with_wobble(check_audio, tts.sample_rate, wobbler)
                    else:
                        from .playback import play_audio
                        play_audio(check_audio, tts.sample_rate)
                    continue

            log.info("User: %s", text)

            if movement:
                movement.set_processing(True)

            first_sentence = True
            full_response = []
            barge_in_text = None
            for sentence in agent.send_streaming(text):
                if sentence == PROCESSING_SENTINEL:
                    if movement:
                        movement.set_speaking(False)
                        movement.set_processing(True)
                    continue

                if first_sentence and movement:
                    movement.set_processing(False)
                    movement.set_speaking(True)
                    first_sentence = False

                full_response.append(sentence)

                tts_audio = tts.synthesize(sentence)

                if player:
                    was_interrupted, barge_audio = player.play(tts_audio, tts.sample_rate, wobbler)
                    if was_interrupted and barge_audio is not None:
                        log.info("User interrupted mid-speech")
                        barge_in_text = stt.transcribe(barge_audio)
                        if barge_in_text and barge_in_text.strip():
                            log.info("Barge-in: %s", barge_in_text)
                        break
                elif wobbler:
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

            # Handle barge-in: process the interruption as a new user turn
            if barge_in_text and barge_in_text.strip():
                text = barge_in_text
                log.info("User (barge-in): %s", text)
                if movement:
                    movement.set_processing(True)
                # Continue with this text as the new user input
                # by not going back to record_utterance
                first_sentence = True
                full_response = []
                for sentence in agent.send_streaming(text):
                    if sentence == PROCESSING_SENTINEL:
                        if movement:
                            movement.set_speaking(False)
                            movement.set_processing(True)
                        continue
                    if first_sentence and movement:
                        movement.set_processing(False)
                        movement.set_speaking(True)
                        first_sentence = False
                    full_response.append(sentence)
                    tts_audio = tts.synthesize(sentence)
                    if player:
                        player.play(tts_audio, tts.sample_rate, wobbler)
                    elif wobbler:
                        play_sentence_with_wobble(tts_audio, tts.sample_rate, wobbler)
                    else:
                        from .playback import play_audio
                        play_audio(tts_audio, tts.sample_rate)
                if first_sentence and movement:
                    movement.set_processing(False)
                if movement:
                    movement.set_speaking(False)
                barge_response = " ".join(full_response)
                if barge_response:
                    log.info("Assistant (after barge-in): %s", barge_response)

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
