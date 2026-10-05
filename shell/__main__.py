"""Assembly of the realtime shell (Phases 1+3).

Process model: asyncio loop; the two audio owners open once; the mic
callback bridges PCM into the loop. Engaged vs ambient is decided by the
attention lease - the single gate for "does room audio reach the s2s
service?" (ADR 0003: ignored ambient speech never enters the LLM).

Robot tools: one real daemon connection (shell/robot.py); schemas and most
handlers are bridged from the legacy package during the migration window,
vision tools are shell-native (shell/robot_tools.py). Everything degrades gracefully with
no robot connected (EPIC "run without a robot").
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time

from .attention import (
    AttentionLease,
    AttentionManager,
    ParticipantResolver,
    load_profile,
)
from .audio import MicOwner, SpeakerOwner
from .audio.xmos_watchdog import XmosWatchdog, reboot_xmos
from .camera import CameraGrabber
from .config import ShellConfig
from .journal import Journal
from .motion import DoaTracker, MotionOwner, SpeechSway, make_doa_source
from .reactions import WakeReaction
from .realtime import RealtimeClient
from .robot import RobotLink
from .robot_tools import build_handlers, load_sdk_sound
from .tools import ToolRouter

log = logging.getLogger("shell")


async def amain() -> None:
    cfg = ShellConfig()
    journal = Journal(cfg.journal_path)
    profile = load_profile(cfg.attention_profile)
    lease = AttentionLease(profile, journal)
    resolver = ParticipantResolver(profile, journal)
    manager = AttentionManager(profile, lease, journal)

    mic = MicOwner(cfg.mic_device, cfg.sample_rate, cfg.block_size, cfg.preroll_seconds, journal)
    speaker = SpeakerOwner(rate=cfg.sample_rate, block=cfg.block_size, device=cfg.speaker_device, journal=journal)

    robot = RobotLink(journal)
    await asyncio.to_thread(robot.connect)  # robot absent -> tools say so, shell runs
    camera = CameraGrabber(robot.daemon_url)
    sway = SpeechSway(cfg.sample_rate)
    speaker.tap = sway.feed  # head sways with the samples actually played
    motion = MotionOwner(robot, journal, hz=cfg.motion_hz, is_speaking=lambda: speaker.playing, sway=sway)
    sounds: dict[str, object] = {}

    def play_sound(name: str) -> None:  # tool earcons go through the one speaker owner
        if name not in sounds:
            sounds[name] = load_sdk_sound(name, cfg.sample_rate)
        if sounds[name] is not None:
            speaker.enqueue(speaker.generation, "sound", sounds[name])

    tools, handlers = build_handlers(robot, camera, data_dir=cfg.data_dir, motion=motion, play_sound=play_sound)
    router = ToolRouter(
        tools,
        handlers,
        journal,
        confidence_fn=manager.tool_confidence,
        confidence_threshold=profile.tool_confidence,
    )

    # One DOA reader feeding a 2 s ring buffer; the wake turn reads it on the
    # KWS hit (mic thread) - the physical reaction never waits for the LLM.
    doa = DoaTracker(
        make_doa_source(daemon_url=robot.daemon_url),
        motion,
        journal,
        is_speaking=lambda: speaker.playing,
        lease_active=lease.active,
    )
    kws_hit_t = [0.0]

    loop = asyncio.get_event_loop()
    reaction = WakeReaction(
        speaker,
        journal,
        sound_dir=cfg.ack_dir / "wake",
        loop=loop,
        on_wake=lambda _deg: doa.wake_turn(kws_hit_t[0]),
    )
    filler = WakeReaction(
        speaker, journal, sound_dir=cfg.ack_dir / "think", delay_range_s=(0.0, 0.05), min_interval_s=4.0, loop=loop
    )

    def on_speech_started() -> None:
        lease.renew(None, "speech-start")
        reaction.cancel()
        filler.cancel()
        motion.set_thinking(False)  # barge-in: he is listening again
        motion.set_listening(True)  # antennas hold still while the user talks

    def on_speech_stopped() -> None:
        motion.set_listening(False)

    filled_for: list[str | None] = [None]  # lease holder that already got its "hmm"

    def on_transcript(text: str) -> None:
        lease.note_interaction()
        if text and (lease.active() or speaker.playing):
            motion.set_thinking(True)  # look away until his first audio
        # "hmm" only on the first turn after a wake word; mid-conversation it
        # is just noise before every answer.
        if text and lease.active() and filled_for[0] != lease.holder:
            filled_for[0] = lease.holder
            filler.wake()

    def on_response_done(status: str) -> None:
        motion.set_thinking(False)
        if status == "completed":
            lease.exchange()  # Reachy answered: the user gets a fresh window

    client = RealtimeClient(
        cfg.s2s_url,
        instructions=cfg.instructions,
        tools=tools,
        tool_router=router,
        speaker=speaker,
        journal=journal,
        on_speech_started=on_speech_started,
        on_transcript=on_transcript,
        on_response_done=on_response_done,
        on_speech_stopped=on_speech_stopped,
        on_first_audio=lambda: motion.set_thinking(False),
    )

    kws = None
    try:
        from .attention.kws import KeywordSpotter

        kws = KeywordSpotter(
            cfg.kws_model_dir,
            thresholds=profile.kws_thresholds,
            default_threshold=profile.kws_default_threshold,
        )
    except Exception as e:
        log.warning("KWS unavailable (%r); wake fast path disabled until Phase 3 setup", e)

    silence: dict[int, bytes] = {}

    was_speaking = [False]

    def on_pcm(pcm: bytes) -> None:
        # While Reachy is talking he is always engaged: the user must be able
        # to interrupt (barge-in), and replies routinely outlast a lease timed
        # from the user's last words (2026-10-04: 13 expiries 0.5-18 s into a
        # reply, zero barge-ins in 356 speech starts).
        speaking = speaker.playing
        if was_speaking[0] and not speaking:
            lease.exchange("reply-played")  # answer window starts when he stops
        was_speaking[0] = speaking
        if lease.active() or speaking:
            client.feed(pcm)  # engaged: s2s owns VAD/turn/STT
            return
        # Not engaged: send silence, not nothing. If the stream just stops,
        # s2s never sees a turn end and keeps it open until the NEXT wake -
        # then answers a conversation from minutes ago (2026-10-04: 13 min).
        client.feed(silence.setdefault(len(pcm), bytes(len(pcm))))
        if kws is None:
            return
        for hit in kws.process(pcm):
            kws_hit_t[0] = time.monotonic()
            decision = manager.on_keyword(None, hit.keyword)
            reaction.wake(doa_deg=None, attending_same=not decision.acquired)  # head turn first
            journal.write("kws.detected", keyword=hit.keyword, score=hit.score)
            client.feed(mic.preroll())  # preserve "Reachy, ..." as one utterance

    mic.subscribe(on_pcm)

    mic.open()
    speaker.open()
    if motion.start():  # no-op without a robot
        doa.start()
    log.info("shell up: profile=%s tools=%d lease=%s", profile.name, len(tools), lease.holder)

    rt = asyncio.create_task(client.run(), name="realtime")

    xmos = XmosWatchdog(journal, reboot_xmos)

    async def watchdog_sample():
        while True:
            await asyncio.sleep(10.0)
            if not speaker.playing:  # never drop attention mid-reply
                lease.expire_if_due()
            health = await asyncio.to_thread(mic.health)  # runs wpctl
            journal.write("audio.health", **health)
            await asyncio.to_thread(xmos.check, health)

    wd = asyncio.create_task(watchdog_sample(), name="health")
    try:
        await asyncio.gather(rt, wd)
    finally:
        mic.close()
        speaker.close()
        await asyncio.to_thread(doa.stop)
        await asyncio.to_thread(motion.stop)
        await asyncio.to_thread(robot.disconnect)
        journal.closed()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-18s %(levelname)-5s %(message)s",
    )
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
