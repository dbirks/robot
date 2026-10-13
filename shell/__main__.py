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

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable

from .attention import (
    AttentionLease,
    AttentionManager,
    ParticipantResolver,
    load_profile,
)
from .audio import MicOwner, SpeakerOwner
from .audio.xmos_tuning import XmosTuner
from .audio.xmos_watchdog import XmosWatchdog, reboot_xmos
from .camera import CameraGrabber, CameraStream
from .config import REACHY_ENV_VAR, ShellConfig
from .conversation import ConversationMemory, Turn, llm_extract_facts, llm_summarize
from .env import EnvError, config_to_dict, load_config, persona_report
from .intents import is_bare_wake, wants_silence, wants_sleep
from .journal import Journal
from .memories import MemoryStore, migrate_legacy_memories
from .motion import DoaTracker, MotionOwner, SpeechSway, make_doa_source
from .reactions import WakeReaction
from .realtime import RealtimeClient
from .realtime.client import ORIGIN_GREETER
from .robot import RobotLink
from .robot_tools import build_handlers, load_sdk_sound
from .sleep import SleepState, feeds_keyword_spotter, feeds_service, should_auto_sleep
from .tools import ToolRouter, apply_tool_policy
from .vision import FaceTracker, Greeter, greet_hook

log = logging.getLogger("shell")

# Longer than the s2s reopen grace (--unanswered_reopen_ms in run_s2s.sh), so
# the next sound starts a NEW turn instead of extending the cut one.
FORCED_END_SILENCE_S = 1.5
REPLY_WAIT_S = 30.0  # longest we keep attention waiting on one reply
WAKE_ACK_WAIT_S = 1.5  # quiet after a bare wake word before the "yes?"

FactExtractor = Callable[[list[Turn]], Awaitable[list[str]]]
FactSaver = Callable[[str], None]


def exchange_for_reply(lease: AttentionLease, origin: str, evidence: str = "reply-done") -> bool:
    """Did a REAL exchange just end? Only then does the holder get a fresh window.

    A reply that answered a greeter directive is the shell talking to the room:
    exchanging here would revive an expired holder (AttentionLease.exchange
    re-arms a lease that lapsed during the reply, and the reaper is skipped
    while the speaker plays), i.e. a walking face would hand some stale
    participant a fresh mic-open window. That is exactly the ambient-speech-
    into-LLM class ADR 0003 exists to stop, so greeter-originated replies
    exchange nothing; every other reply keeps the existing behaviour exactly.
    """
    if origin == ORIGIN_GREETER:
        return False
    return lease.exchange(evidence)


def fact_mining(cfg: ShellConfig, memories: MemoryStore) -> tuple[FactExtractor | None, FactSaver | None]:
    """(extract_facts, save_fact) for ConversationMemory, or (None, None).

    Auto extraction writes through `save_fact` with no attention gate and no
    ask, straight into this environment's store, which `prompt_block` then hands
    to the next session as "what you know about the user". In a booth with one
    shared store that says attendee A's business to attendee B, so
    `behaviour.learn_facts: false` (envs/hackday.yaml) turns the miner off; the
    explicit `remember` tool keeps working either way (docs/environments.md rule 1).
    """
    if not cfg.learn_facts:
        return None, None

    def save(fact: str) -> None:
        memories.add(fact, source="auto")

    return (
        lambda turns: llm_extract_facts(cfg.llm_base_url, cfg.llm_model, turns, [m["text"] for m in memories.all()]),
        save,
    )


async def amain(cfg: ShellConfig) -> None:
    journal = Journal(cfg.journal_path, env=cfg.env)  # env on every event (A4)
    sleep = SleepState(journal)  # the one asleep flag: head and microphone (issue #36)
    profile = load_profile(cfg.attention_profile)
    lease = AttentionLease(profile, journal, asleep=lambda: sleep.sleeping)
    resolver = ParticipantResolver(profile, journal)
    manager = AttentionManager(profile, lease, journal)

    mic = MicOwner(cfg.mic_device, cfg.sample_rate, cfg.block_size, cfg.preroll_seconds, journal)
    speaker = SpeakerOwner(rate=cfg.sample_rate, block=cfg.block_size, device=cfg.speaker_device, journal=journal)

    robot = RobotLink(journal)
    await asyncio.to_thread(robot.connect)  # robot absent -> tools say so, shell runs
    camera = CameraGrabber(robot.daemon_url)
    # One continuous frame owner when the greeter is on (ADR 0002): the poller and
    # the vision tools both read it, so exactly one thing ever opens the device.
    # It degrades to `available = False` with no camera and nobody else notices.
    camera_stream = CameraStream(
        camera,
        idle_fps=cfg.greeter_idle_fps,
        engaged_fps=cfg.greeter_engaged_fps,
        journal=journal,
        asleep=lambda: sleep.sleeping,  # napping: the poll is silent, not merely cheap
    )
    sway = SpeechSway(cfg.sample_rate)
    speaker.tap = sway.feed  # head sways with the samples actually played
    motion = MotionOwner(
        robot,
        journal,
        hz=cfg.motion_hz,
        is_speaking=lambda: speaker.playing,
        sway=sway,
        sleep_state=sleep,
    )
    sounds: dict[str, object] = {}

    def play_sound(name: str) -> None:  # tool earcons go through the one speaker owner
        if name not in sounds:
            sounds[name] = load_sdk_sound(name, cfg.sample_rate)
        if sounds[name] is not None:
            speaker.enqueue(speaker.generation, "sound", sounds[name])

    def wake_up(reason: str) -> bool:
        """End the nap: head and antennas up through the motion owner, the SDK
        chirp through the one speaker. False when he was already awake."""
        if not motion.wake(reason=reason):
            return False
        play_sound("wake_up.wav")
        return True

    memories = MemoryStore(cfg.memories_path)  # per-environment store (A5)
    extract_facts, save_fact = fact_mining(cfg, memories)  # off in a shared booth

    def tool_confidence(name: str) -> float:
        # Asleep: no state-changing tool runs. The lease cannot be acquired
        # while he sleeps, but one that was already live keeps its confidence
        # for its remaining seconds, so sleep has to zero it out (issue #36).
        return 0.0 if sleep.sleeping else manager.tool_confidence(name)

    tools, handlers = build_handlers(
        robot, camera_stream, data_dir=cfg.data_dir, motion=motion, play_sound=play_sound, memories=memories
    )
    tools, handlers = apply_tool_policy(tools, handlers, include=cfg.tools_include, exclude=cfg.tools_exclude)
    router = ToolRouter(
        tools,
        handlers,
        journal,
        confidence_fn=tool_confidence,
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
    xmos_tuner = XmosTuner(journal, agc_max_gain=cfg.xmos_agc_max_gain, beam_focus=cfg.xmos_beam_focus)
    await asyncio.to_thread(xmos_tuner.ensure)

    loop = asyncio.get_event_loop()
    # "yes?" only for a bare wake word: after WAKE_ACK_WAIT_S of quiet with
    # nothing else said, or as soon as the transcript is just the name.
    wake_t: list[float | None] = [None]  # last wake word still waiting for its first transcript
    speech_open = [False]
    reaction = WakeReaction(
        speaker,
        journal,
        sound_dir=cfg.ack_dir / "wake",
        loop=loop,
        delay_range_s=(WAKE_ACK_WAIT_S, WAKE_ACK_WAIT_S),
        gate=lambda: wake_t[0] is not None and not speech_open[0],
        on_wake=lambda _deg: doa.wake_turn(kws_hit_t[0]),
    )

    turn_t0: list[float | None] = [None]  # first speech-start of the open user turn
    mute_until = [0.0]  # forced endpoint: feed silence until then
    stopped_at: list[float | None] = [None]  # last speech-stop while no speech is open

    def on_speech_started() -> None:
        # A bare VAD start is weak evidence (the TV makes plenty): it may only
        # stretch the lease a little past the wake word / last reply.
        lease.renew(None, "speech-start", cap_s=profile.lease_unanswered_s)
        stopped_at[0] = None
        if turn_t0[0] is None:
            turn_t0[0] = time.monotonic()
        speech_open[0] = True
        motion.set_thinking(False)  # barge-in: he is listening again
        motion.set_listening(True)  # antennas hold still while the user talks

    def on_speech_stopped() -> None:
        stopped_at[0] = time.monotonic()
        speech_open[0] = False
        motion.set_listening(False)

    memory = ConversationMemory(
        journal,
        lambda prior, turns: llm_summarize(cfg.llm_base_url, cfg.llm_model, prior, turns),
        compact_after_s=cfg.compact_after_s,
        reset_after_s=cfg.reset_after_s,
        extract_facts=extract_facts,
        save_fact=save_fact,
    )

    awaiting_reply = [0.0]  # monotonic time of the last transcript not yet answered

    def on_transcript(text: str) -> None:
        turn_t0[0] = None
        if sleep.sleeping:
            # Nothing said after he went down reaches history, the LLM or a tool
            # call - including the in-flight transcript his sleep command cut
            # short (issue #36).
            return
        if text:
            awaiting_reply[0] = time.monotonic()
        if text and lease.active() and wants_sleep(text):
            # don't trust the LLM to act on it (it often just says goodnight)
            journal.write("intent.sleep", text=text)
            loop.run_in_executor(None, router.execute, "go_to_sleep", {})
        lease.note_interaction()
        if text and lease.active() and wants_silence(text) and client.last_turn_cut_playback:
            # "stop talking" is an order, not a turn. The interrupt itself already did
            # its work (speaker.cancel_current, before this transcript arrived); what
            # must not happen is the phrase reaching the model as user input, which
            # answers it - measured live 2026-10-12, four times, each time by calling
            # set_volume, because loudness was the nearest affordance a 4B had.
            # Home Assistant routes HassNevermind around its agent the same way.
            # No memory.add: a command he obeyed is not something he was told.
            journal.write("intent.stop", text=text)
            awaiting_reply[0] = 0.0  # we are deliberately not waiting for a reply
            reaction.cancel()  # no "hmm" filler over the silence just demanded
            wake_t[0] = None
            client.drop_next_response()
            return
        if wake_t[0] is not None and text:
            fresh = time.monotonic() - wake_t[0] < 10.0
            wake_t[0] = None
            if fresh and is_bare_wake(text):
                # Just "Reachy": the prerendered "yes?" is the answer; drop the
                # LLM's slower, wordier one.
                reaction.fire_now()
                client.drop_next_response()
                return
            reaction.cancel()
        memory.add("user", text)
        if text and (lease.active() or speaker.playing):
            motion.set_thinking(True)  # look away until his first audio

    def on_response_done(status: str) -> None:
        awaiting_reply[0] = 0.0
        motion.set_thinking(False)
        if status == "completed":
            # Reachy answered: the user gets a fresh window - unless the reply
            # only answered a greeter directive (ADR 0003, exchange_for_reply).
            exchange_for_reply(lease, client.response_origin)

    client = RealtimeClient(
        cfg.s2s_url,
        instructions=lambda: cfg.instructions + memories.prompt_block(),
        tools=tools,
        tool_router=router,
        speaker=speaker,
        journal=journal,
        on_speech_started=on_speech_started,
        on_transcript=on_transcript,
        on_response_done=on_response_done,
        on_assistant_text=lambda text: memory.add("assistant", text),
        on_speech_stopped=on_speech_stopped,
        on_first_audio=lambda: motion.set_thinking(False),
    )

    # B3: the proactive greeter. Only built when the environment turns it on, and
    # only streams if the camera is there to stream; InsightFace is loaded lazily by
    # the tracker on the first frame, so an absent camera never costs a model load.
    greeter = None
    if cfg.greeter_enabled:
        greeter = Greeter(
            source=camera_stream,
            tracker=FaceTracker(),
            journal=journal,
            on_greet=greet_hook(client, loop, journal=journal),
            motion=motion,
            face_centering=cfg.greeter_face_centering,
            # Fail-closed suppressors (ADR 0003): he is never proactive while he is
            # speaking, asleep, holding a conversation, or owed a reply.
            is_playing=lambda: speaker.playing or sleep.sleeping,
            asleep=lambda: sleep.sleeping,
            lease_active=lease.active,
            awaiting_reply=lambda: time.monotonic() - awaiting_reply[0] < REPLY_WAIT_S,
            dwell_s=cfg.greeter_dwell_s,
            min_spacing_s=cfg.greeter_min_spacing_s,
            group_min_spacing_s=cfg.greeter_group_min_spacing_s,
            max_per_hour=cfg.greeter_max_per_hour,
            idle_fps=cfg.greeter_idle_fps,
            engaged_fps=cfg.greeter_engaged_fps,
            debug=cfg.greeter_debug,
        )
        camera_stream.start()
        if not camera_stream.available:
            log.info("greeter waiting on a camera (daemon holds it, or there is none)")
        if not cfg.greeter_face_centering:
            log.info("greeter head-centring off (REACHY_GREETER_FACE_CENTERING=1 to try it)")

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
            # answer window starts when he stops - never for a greeter reply
            exchange_for_reply(lease, client.response_origin, "reply-played")
        was_speaking[0] = speaking
        asleep = sleep.sleeping  # one read per frame: head and mic cannot disagree
        now = time.monotonic()
        t0 = turn_t0[0]
        st = stopped_at[0]
        if t0 is not None and st is not None and now - st > FORCED_END_SILENCE_S:
            turn_t0[0] = t0 = None  # quiet past the reopen grace: that turn is over (or was discarded)
        if t0 is not None and now - t0 > cfg.max_turn_s:
            # One user turn has absorbed > max_turn_s of "speech": background
            # talk keeps reopening it (s2s reopens on any sound within its
            # grace window). A short silence is the only way to end it.
            turn_t0[0] = None
            mute_until[0] = now + FORCED_END_SILENCE_S
            journal.write("turn.forced_end", after_s=round(now - t0, 1))
        if feeds_service(sleeping=asleep, engaged=lease.active() or speaking, muted=now < mute_until[0]):
            client.feed(pcm)  # engaged: s2s owns VAD/turn/STT
            return
        # Not engaged (or asleep): send silence, not nothing. If the stream just stops,
        # s2s never sees a turn end and keeps it open until the NEXT wake -
        # then answers a conversation from minutes ago (2026-10-04: 13 min).
        client.feed(silence.setdefault(len(pcm), bytes(len(pcm))))
        # KWS keeps hearing the real frames however deaf the service is: the
        # wake word is the alarm clock, and this is the one path out of a nap
        # that does not need the room to be listened to (issue #36).
        if kws is None or not feeds_keyword_spotter(sleeping=asleep, engaged=lease.active() or speaking):
            return
        for hit in kws.process(pcm):
            kws_hit_t[0] = time.monotonic()
            if sleep.sleeping:
                wake_up(f"kws:{hit.keyword}")  # up, antennas up, chirp - then normal attention
            decision = manager.on_keyword(None, hit.keyword)
            reaction.wake(doa_deg=None, attending_same=not decision.acquired)  # head turn first
            journal.write("kws.detected", keyword=hit.keyword, score=hit.score)
            if decision.acquired:
                wake_t[0] = time.monotonic()
                theta, _ = doa.buffer.median_speech(time.monotonic(), 1.0)
                loop.call_soon_threadsafe(start_conversation)
                loop.call_soon_threadsafe(lambda th=theta: loop.run_in_executor(None, xmos_tuner.focus, th))
            client.feed(mic.preroll())  # preserve "Reachy, ..." as one utterance

    mic.subscribe(on_pcm)

    mic.open()
    speaker.open()
    if motion.start():  # no-op without a robot
        doa.start()
    if greeter is not None and greeter.start():
        log.info(
            "greeter on: dwell=%.1fs spacing=%.0fs cap=%d/h",
            cfg.greeter_dwell_s,
            cfg.greeter_min_spacing_s,
            cfg.greeter_max_per_hour,
        )
    log.info("shell up: env=%s profile=%s tools=%d lease=%s", cfg.env, profile.name, len(tools), lease.holder)

    rt = asyncio.create_task(client.run(), name="realtime")

    xmos = XmosWatchdog(journal, reboot_xmos)

    boot_t = time.monotonic()  # an idle booth that nobody ever talks to goes to sleep too

    async def attention_tick():
        # 1 Hz: the beam must widen again soon after attention ends, or the
        # next wake word from another direction is attenuated.
        while True:
            await asyncio.sleep(1.0)
            # Never drop attention mid-reply, nor while the LLM is still
            # answering: the holder must survive until lease.exchange().
            waiting = time.monotonic() - awaiting_reply[0] < REPLY_WAIT_S
            if not speaker.playing and not waiting:
                lease.expire_if_due()
            if xmos_tuner.focused is not None and not lease.active() and not speaker.playing:
                await asyncio.to_thread(xmos_tuner.release)
            idle_s = time.monotonic() - max(boot_t, lease.last_interaction, stopped_at[0] or 0.0, turn_t0[0] or 0.0)
            if should_auto_sleep(
                auto_sleep_s=cfg.auto_sleep_s,
                idle_s=idle_s,
                engaged=lease.active() or speaker.playing or speech_open[0],
                sleeping=sleep.sleeping,
            ):
                motion.sleep(reason="auto_sleep")  # journals sleep.enter (ADR 0007)

    async def watchdog_sample():
        while True:
            await asyncio.sleep(10.0)
            engaged = lease.active() or speaker.playing
            note = memory.tick(engaged=engaged)
            if note is not None:
                client.reset_session(note)  # quiet room: fresh session now, not at the wake word
            await asyncio.to_thread(xmos_tuner.ensure)
            health = await asyncio.to_thread(mic.health)  # runs wpctl
            journal.write("audio.health", **health)
            await asyncio.to_thread(xmos.check, health)

    wd = asyncio.create_task(watchdog_sample(), name="health")
    at = asyncio.create_task(attention_tick(), name="attention")
    try:
        await asyncio.gather(rt, wd, at)
    finally:
        await asyncio.to_thread(xmos_tuner.close)
        mic.close()
        speaker.close()
        await asyncio.to_thread(doa.stop)
        if greeter is not None:
            await asyncio.to_thread(greeter.stop)
        await asyncio.to_thread(camera_stream.stop)
        await asyncio.to_thread(motion.stop)
        await asyncio.to_thread(robot.disconnect)
        journal.closed()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shell",
        description="Reachy Mini realtime shell: audio owners, attention, tools (docs/environments.md).",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="print the resolved deployment environment as JSON and exit; "
        "opens no audio device, connects to no robot, touches no network",
    )
    parser.add_argument(
        "--print-personas",
        action="store_true",
        help="print the persona picker's resolved data as JSON and exit "
        "(personas/*.yaml + which one this room names); opens nothing, like --print-env",
    )
    parser.add_argument(
        "--migrate-memory",
        action="store_true",
        help="copy the pre-namespaces data/memories.json into this env's store and exit "
        "(stop reachy-shell first: the running shell still writes the old file)",
    )
    return parser


def _resolved_config() -> ShellConfig:
    """load_config() with an operator-readable failure. A missing env file is a
    hard startup error; it is not a traceback at 9am in an office."""
    try:
        return load_config()
    except EnvError as e:
        print(f"shell: {e}", file=sys.stderr)
        raise SystemExit(2) from e


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    # These two flags resolve config and leave. They run before logging is set up
    # and before anything is constructed: no audio device, no robot, no journal,
    # no network. --print-env has to work on a machine where the shell cannot.
    if args.print_env:
        print(json.dumps(config_to_dict(_resolved_config()), indent=2, sort_keys=True))
        return
    if args.print_personas:
        # The dashboard's persona panel reads this instead of importing shell.env:
        # reachy-dashboard runs in its own venv and has no pyyaml (env_panel.py).
        try:
            print(json.dumps(persona_report(), indent=2, sort_keys=True))
        except EnvError as e:
            print(f"shell: {e}", file=sys.stderr)
            raise SystemExit(2) from e
        return
    if args.migrate_memory:
        cfg = _resolved_config()
        report = migrate_legacy_memories(cfg.data_dir, cfg.memory_namespace)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["ok"] else 1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-18s %(levelname)-5s %(message)s",
    )
    cfg = _resolved_config()
    if not os.environ.get(REACHY_ENV_VAR, "").strip():
        # Nobody said which room this is, so it is the desk one - with family
        # facts and the whole tool list. At an event that is the leak, and the
        # only way to notice is to have asked (docs/environments.md hard rule 1).
        log.warning(
            "%s was never set: running deployment environment %r (the default). "
            "Set it in .env before this machine is in a public room; verify with --print-env",
            REACHY_ENV_VAR,
            cfg.env,
        )
    try:
        asyncio.run(amain(cfg))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
