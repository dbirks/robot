"""Assembly of the realtime shell (Phases 1+3).

Process model: asyncio loop; the two audio owners open once; the mic
callback bridges PCM into the loop. Engaged vs ambient is decided by the
attention lease - the single gate for "does room audio reach the s2s
service?" (ADR 0003: ignored ambient speech never enters the LLM).

Robot tools are bridged from the legacy package during the migration
window; Phase 2 moves them natively. Everything degrades gracefully with
no robot connected (EPIC "run without a robot").
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from .attention import (
    AttentionLease,
    AttentionManager,
    ParticipantResolver,
    load_profile,
)
from .audio import MicOwner, SpeakerOwner
from .audio.xmos_watchdog import XmosWatchdog, reboot_xmos
from .config import ShellConfig
from .conversation import ConversationMemory, llm_summarize
from .journal import Journal
from .reactions import WakeReaction
from .realtime import RealtimeClient
from .tools import ToolRouter

log = logging.getLogger("shell")


def load_robot_tools(journal):
    """Best-effort bridge to the legacy handlers; empty shell still runs."""
    try:
        from app.robot_state import RobotConnection
        from app.robot_tools import TOOLS, make_handlers
    except Exception as e:  # missing robot SDK, etc.
        log.warning("robot tools unavailable: %r", e)
        return [], {}, None

    class _DisconnectedRobot(RobotConnection):  # redefines __init__; no config needed
        def __init__(self) -> None:
            self.mini = None  # property 'connected' derives from this

    robot = _DisconnectedRobot()
    handlers = {}
    try:
        handlers = dict(make_handlers(robot))
    except Exception as e:
        log.warning("robot tool construction failed: %r", e)
    return TOOLS, handlers, robot


async def amain() -> None:
    cfg = ShellConfig()
    journal = Journal(cfg.journal_path)
    profile = load_profile(cfg.attention_profile)
    lease = AttentionLease(profile, journal)
    resolver = ParticipantResolver(profile, journal)
    manager = AttentionManager(profile, lease, journal)

    mic = MicOwner(cfg.mic_device, cfg.sample_rate, cfg.block_size, cfg.preroll_seconds, journal)
    speaker = SpeakerOwner(rate=cfg.sample_rate, block=cfg.block_size, device=cfg.speaker_device, journal=journal)

    tools, handlers, _robot = load_robot_tools(journal)
    router = ToolRouter(
        tools,
        handlers,
        journal,
        confidence_fn=manager.tool_confidence,
        confidence_threshold=profile.tool_confidence,
    )

    loop = asyncio.get_event_loop()
    reaction = WakeReaction(speaker, journal, sound_dir=cfg.ack_dir / "wake", loop=loop)
    filler = WakeReaction(
        speaker, journal, sound_dir=cfg.ack_dir / "think", delay_range_s=(0.0, 0.05), min_interval_s=4.0, loop=loop
    )

    def on_speech_started() -> None:
        lease.renew(None, "speech-start")
        reaction.cancel()
        filler.cancel()

    filled_for: list[str | None] = [None]  # lease holder that already got its "hmm"
    memory = ConversationMemory(
        journal,
        lambda prior, turns: llm_summarize(cfg.llm_base_url, cfg.llm_model, prior, turns),
        compact_after_s=cfg.compact_after_s,
        reset_after_s=cfg.reset_after_s,
    )

    def on_transcript(text: str) -> None:
        lease.note_interaction()
        memory.add("user", text)
        # "hmm" only on the first turn after a wake word; mid-conversation it
        # is just noise before every answer.
        if text and lease.active() and filled_for[0] != lease.holder:
            filled_for[0] = lease.holder
            filler.wake()

    def on_response_done(status: str) -> None:
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
        on_assistant_text=lambda text: memory.add("assistant", text),
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

    def start_conversation() -> None:
        # After a long quiet spell, start a fresh s2s session seeded with the
        # compacted summary instead of hours-old raw history.
        note = memory.on_wake()
        if note is not None:
            client.reset_session(note)

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
            decision = manager.on_keyword(None, hit.keyword)
            journal.write("kws.detected", keyword=hit.keyword, score=hit.score)
            if decision.acquired:
                loop.call_soon_threadsafe(start_conversation)
            client.feed(mic.preroll())  # preserve "Reachy, ..." as one utterance
            reaction.wake(doa_deg=None, attending_same=not decision.acquired)

    mic.subscribe(on_pcm)

    mic.open()
    speaker.open()
    log.info("shell up: profile=%s tools=%d lease=%s", profile.name, len(tools), lease.holder)

    rt = asyncio.create_task(client.run(), name="realtime")

    xmos = XmosWatchdog(journal, reboot_xmos)

    async def watchdog_sample():
        while True:
            await asyncio.sleep(10.0)
            if not speaker.playing:  # never drop attention mid-reply
                lease.expire_if_due()
            memory.tick(engaged=lease.active() or speaker.playing)
            health = await asyncio.to_thread(mic.health)  # runs wpctl
            journal.write("audio.health", **health)
            await asyncio.to_thread(xmos.check, health)

    wd = asyncio.create_task(watchdog_sample(), name="health")
    try:
        await asyncio.gather(rt, wd)
    finally:
        mic.close()
        speaker.close()
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
