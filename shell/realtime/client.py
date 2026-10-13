"""OpenAI-Realtime WebSocket client for the pinned HF speech-to-speech service.

Production promotion of spike/s2s_client.py; read
experiments/2026-07-28-realtime-spike/README.md first. Protocol quirks
verified there (also ADR 0001):

- no session.updated echo is ever sent: send and move on, never await it
- malformed sub-objects in session.update get one opaque "Unknown or
  invalid event": keep the payload minimal
- assistant transcript and function-call arguments arrive only as .done
  events; audio arrives as response.output_audio.delta
- a dead sender task means SILENCE, not an error: task death is logged
  loudly and never swallowed

Tool calls run as background tasks (2026-10-11): awaiting a 2 s web search
inside the receiver froze every other event - the "Let me check..." audio
already on the wire, and speech_started for barge-in. Results go back once
the server has finished the response that asked for them, batched into one
follow-up, and the follow-up is skipped if the user spoke in the meantime.

Barge-in invariant (ADR 0002): on input_audio_buffer.speech_started the
speaker owner drops the in-flight generation immediately; the server's
CancelScope guarantees the response itself dies. The interrupting
utterance is preserved because the mic stream never stops.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Callable

import numpy as np

from .. import journal as J
from ..tools import TOOL_TIMEOUT_S

log = logging.getLogger("shell.realtime")


# The budget now belongs to the router (shell/tools.py: `timeout_for`), which knows
# that a camera tool and a `get_time` are not the same call. This constant is the
# fallback for a router that has no opinion.

# What a timeout says. `asyncio.to_thread` cannot be cancelled, so on a timeout the
# work is STILL running and will complete - telling the model "took too long" made it
# retry, and each retry queued another capture behind the call that had already been
# reported as failed. Three "failed" snapshots, one real one, 54 seconds, and a robot
# apologising to a guest for a tool that had quietly succeeded.
TIMEOUT_MESSAGE = (
    "{name} is slower than this reply can wait, but it is still running and will finish on "
    "its own. Do not call it again: a second attempt queues behind the first and makes it "
    "slower. Say one short sentence that it is taking a moment."
)
BATCH_WAIT_S = 30.0  # longest we wait for response.done before sending results

# Who asked for a reply (ADR 0003 wiring). A greeter-prompted reply is the shell
# talking to the room, not the room talking to the shell: it must not exchange
# the attention lease. Anything else (VAD turn, KWS wake, real transcript) does.
ORIGIN_USER = "user"
ORIGIN_GREETER = "greeter"


class _ToolBatch:
    """Tool calls of one response, answered together after its response.done."""

    def __init__(self, epoch: int, origin: str = ORIGIN_USER) -> None:
        self.epoch = epoch  # user-speech counter when the calls started
        # WHO asked for this reply, carried across the tool round trip. Without it
        # the follow-up `response.create` below gets latched by the next
        # `response.created` as ORIGIN_USER, and a greeting that happened to call a
        # tool ends up exchanging the attention lease exactly like a real turn
        # (ADR 0003, found in the #36 review).
        self.origin = origin
        self.tasks: list[asyncio.Task] = []
        self.done = asyncio.Event()


class SlotBusy(RuntimeError):
    """s2s rejected the connection: its single pipeline slot is still held by
    the session we just closed (it drains through every handler first)."""


class RealtimeClient:
    def __init__(
        self,
        url: str,
        *,
        instructions: str | Callable[[], str],
        tools: list[dict],
        tool_router,
        speaker,
        journal,
        on_speech_started: Callable[[], None] | None = None,
        on_transcript: Callable[[str], None] | None = None,
        on_response_done: Callable[[str], None] | None = None,
        on_assistant_text: Callable[[str], None] | None = None,
        on_speech_stopped: Callable[[], None] | None = None,
        on_first_audio: Callable[[], None] | None = None,
        reconnect_s: float = 2.0,
        busy_retry_s: float = 0.25,
        connect=None,
    ) -> None:
        self.url = url
        self.instructions = instructions
        self.tools = [_realtime_tool(t) for t in tools]
        self.router = tool_router
        self.speaker = speaker
        self.journal = journal
        self.on_speech_started = on_speech_started
        self.on_transcript = on_transcript
        self.on_response_done = on_response_done
        self.on_assistant_text = on_assistant_text
        self._extra_instructions = ""  # e.g. summary of an earlier conversation
        self._reset: asyncio.Event | None = None
        self._carry: bytes | None = None
        self.on_speech_stopped = on_speech_stopped
        self.on_first_audio = on_first_audio
        self._turn_cut = False  # did the utterance now being transcribed cut his audio?
        self.reconnect_s = reconnect_s
        self.busy_retry_s = busy_retry_s
        self._connect = connect  # injectable for tests; defaults to websockets
        self._in: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._ws = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._resp_gen: dict[str, int] = {}  # response_id -> speaker generation
        self.connected = False
        self._cur: dict[str, float] = {}  # per-turn latency marks
        self._first_audio: set[str] = set()  # response ids already marked
        self._drop_until = 0.0  # loop time before which the next response is dropped
        # Who asked for the current (or next) response: see insert_user_transcript.
        # `response_origin` survives response.done until the NEXT response is
        # created, because the lease exchange on the played path fires when the
        # speaker falls silent, which is after the last response.done.
        self._pending_origin = ORIGIN_USER
        self._origin = ORIGIN_USER
        self._epoch = 0  # bumped on every user speech start (barge-in)
        self._tool_batches: dict[str, _ToolBatch] = {}

    def _instructions(self) -> str:
        # a callable is re-read per session (e.g. to include new memories).
        # isinstance rather than callable(): ty has no callable() narrowing, and
        # the attribute is a closed `str | Callable[[], str]` union anyway.
        if isinstance(self.instructions, str):
            return self.instructions
        return self.instructions()

    @property
    def response_origin(self) -> str:
        """Who asked for the reply that just finished (ORIGIN_GREETER | ORIGIN_USER).
        Read from on_response_done and from the speaker fall-silent edge."""
        return self._origin

    # ---- ingestion (thread-safe: called from the mic callback) ----

    def feed(self, pcm: bytes) -> None:
        """Queue mic audio for the service. Safe from any thread."""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._feed_nowait, pcm)
        except (RuntimeError, asyncio.QueueFull):
            pass  # closed loop or full buffer: drop, never block capture

    def _feed_nowait(self, pcm: bytes) -> None:
        try:
            self._in.put_nowait(pcm)
        except asyncio.QueueFull:
            try:
                self._in.get_nowait()  # drop OLDEST: fresher audio wins
            except asyncio.QueueEmpty:
                pass
            self._in.put_nowait(pcm)

    def insert_user_transcript(self, text: str, *, origin: str = ORIGIN_USER) -> None:
        """Inject an already-transcribed (accepted ambient) utterance without
        re-running STT, then request a response. Must be called on the loop.

        `origin` labels WHO asked for the reply this creates: it rides through to
        the `response_origin` the shell reads when the reply finishes, so a reply
        the greeter prompted can never renew the attention lease (ADR 0003 - a
        face is not a participant). Real dialogue keeps the default."""
        asyncio.get_event_loop().create_task(self._inject(text, origin))

    async def _inject(self, text: str, origin: str) -> None:
        if self._ws is None:
            return
        self._pending_origin = origin  # latched by the next response.created
        await self._ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
                }
            )
        )
        await self._ws.send(json.dumps({"type": "response.create"}))

    def drop_next_response(self, within_s: float = 5.0) -> None:
        """Silence and cancel the next response the server starts within
        `within_s` (e.g. its reply to a bare wake word we already answered
        with a prerendered "yes?"). Call on the loop."""
        self._drop_until = asyncio.get_event_loop().time() + within_s

    @property
    def last_turn_cut_playback(self) -> bool:
        """Did the utterance currently being transcribed cut audible playback?

        The interrupting words are also committed as a fresh user turn, so
        "stop talking" reaches the model and gets answered - by `set_volume`
        (measured 2026-10-12). The shell only suppresses that reply when the
        words genuinely arrived over his own voice; this is the one bit that
        distinguishes an interruption from a remark about the interruption.
        Read by `on_transcript`, cleared once the transcript is delivered.
        """
        return self._turn_cut

    # ---- lifecycle ----

    def reset_session(self, extra_instructions: str = "") -> None:
        """Drop the s2s conversation and reconnect with a fresh session whose
        instructions carry `extra_instructions`. Safe from any thread; queued
        mic audio is kept and flows into the new session."""
        if self._loop is None:
            return

        def _do() -> None:
            self._extra_instructions = extra_instructions
            self._pending_origin = ORIGIN_USER  # a fresh session starts with no owed reply
            self._origin = ORIGIN_USER
            if self._reset is not None:
                self._reset.set()

        try:
            on_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            _do()  # must land before audio queued right after this call
        else:
            self._loop.call_soon_threadsafe(_do)

    async def run(self) -> None:
        self._loop = asyncio.get_event_loop()
        self._reset = asyncio.Event()
        busy = 0
        while True:
            try:
                await self._session()
                busy = 0
            except asyncio.CancelledError:
                raise
            except SlotBusy:
                # Normal right after a reset. Mic audio waits in self._in
                # (~6 s) and goes to the session that is finally admitted.
                self.connected = False
                busy += 1
                if busy % 40 == 0:
                    log.warning("realtime: s2s slot still busy after %d tries", busy)
                await asyncio.sleep(self.busy_retry_s)
                continue
            except Exception as e:
                log.warning("realtime session error: %r; reconnecting in %.1fs", e, self.reconnect_s)
                self.connected = False
                await asyncio.sleep(self.reconnect_s)

    async def _session(self) -> None:
        connect = self._connect
        if connect is None:
            import websockets

            connect = lambda: websockets.connect(self.url, max_size=None)  # noqa: E731
        async with connect() as ws:
            # Admission first: a rejected socket still accepts writes, and the
            # wake preroll sent into it was lost. The server's first frame is
            # session.created, or an error when the slot is still taken.
            recv = getattr(ws, "recv", None) or ws.__aiter__().__anext__
            first = await asyncio.wait_for(recv(), timeout=10.0)
            try:
                ev = json.loads(first)
            except (TypeError, json.JSONDecodeError):
                ev = {}
            if ev.get("type") == "error":
                if (ev.get("error") or {}).get("type") == "session_limit_reached":
                    raise SlotBusy()
                await self.handle_event(ev)
            self._ws = ws
            self.connected = True
            log.info("realtime connected: %s", self.url)
            # Never await session.updated (it is never sent); send a minimal
            # session.update and move on.
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "instructions": self._instructions() + self._extra_instructions,
                            "tools": self.tools,
                        },
                    }
                )
            )
            sender = asyncio.create_task(self._sender(ws), name="rt-sender")
            receiver = asyncio.create_task(self._receiver(ws), name="rt-receiver")
            if self._reset is not None:
                self._reset.clear()
            resetter = asyncio.create_task(self._reset.wait() if self._reset else asyncio.Event().wait())
            try:
                done, _ = await asyncio.wait({sender, receiver, resetter}, return_when=asyncio.FIRST_COMPLETED)
                for t in done:  # a dead sender streams silence; make noise instead
                    if t is not resetter and not t.cancelled() and t.exception():
                        raise RuntimeError(f"{t.get_name()} died") from t.exception()
                if resetter in done:
                    log.info("realtime: starting a fresh session")
                    self.journal.write("realtime.session_reset")
                elif receiver in done:
                    raise RuntimeError("realtime connection closed")
            finally:
                sender.cancel()
                receiver.cancel()
                resetter.cancel()
                self._ws = None
                self.connected = False

    async def _sender(self, ws) -> None:
        if self._carry is not None:  # audio taken just as a reset began
            pcm, self._carry = self._carry, None
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}))
        while True:
            pcm = await self._in.get()
            if self._reset is not None and self._reset.is_set():
                self._carry = pcm  # belongs to the NEW session (e.g. the wake preroll)
                await asyncio.Event().wait()  # cancelled by the reset momentarily
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm).decode("ascii"),
                    }
                )
            )

    async def _receiver(self, ws) -> None:
        async for raw in ws:
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            await self.handle_event(ev)

    # ---- event handling (sync core, unit-testable with a fake ws) ----

    async def handle_event(self, ev: dict) -> None:
        t = ev.get("type", "")

        if t == "input_audio_buffer.speech_started":
            self._epoch += 1
            cut = self.speaker.cancel_current()
            # sticky within one turn: a person can start twice before the words land
            self._turn_cut = self._turn_cut or bool(cut)
            self.journal.write(J.AUDIO_SPEECH_STARTED, cut_playback=cut)
            if self.on_speech_started:
                self.on_speech_started()

        elif t == "input_audio_buffer.speech_stopped":
            self.journal.write(J.AUDIO_SPEECH_STOPPED)
            if self.on_speech_stopped:
                self.on_speech_stopped()

        elif t == "conversation.item.input_audio_transcription.completed":
            text = (ev.get("transcript") or "").strip()
            self.journal.write(J.TURN_TRANSCRIPT_FINAL, text=text)
            try:
                if self.on_transcript:
                    self.on_transcript(text)
            finally:
                self._turn_cut = False  # one turn's evidence, never the next one's

        elif t == "response.created":
            rid = (ev.get("response") or {}).get("id", "")
            self._origin, self._pending_origin = self._pending_origin, ORIGIN_USER
            self._resp_gen[rid] = self.speaker.begin_generation()
            self._first_audio.clear()  # one response in flight at a time
            self.journal.write(J.RESPONSE_CREATED, response_id=rid)
            if self._drop_until and asyncio.get_event_loop().time() < self._drop_until:
                self._drop_until = 0.0
                self.speaker.cancel_current()  # refuse its audio
                self.journal.write("response.dropped", response_id=rid)
                if self._ws is not None:
                    await self._ws.send(json.dumps({"type": "response.cancel"}))

        elif t == "response.output_audio.delta":
            rid = ev.get("response_id", "")
            gen = self._resp_gen.get(rid, self.speaker.generation)
            pcm = np.frombuffer(base64.b64decode(ev["delta"]), dtype=np.int16)
            if self.speaker.enqueue(gen, "tts", pcm) and rid not in self._first_audio:
                self._first_audio.add(rid)  # only mark the FIRST delta
                self.journal.write(J.RESPONSE_FIRST_AUDIO, response_id=rid)
                if self.on_first_audio:
                    self.on_first_audio()

        elif t == "response.output_audio_transcript.done":
            # Journal first: a listener that raises must not cost us the transcript (ADR 0007).
            self.journal.write(
                J.RESPONSE_TRANSCRIPT_FINAL,
                text=ev.get("transcript") or "",
                response_id=ev.get("response_id", ""),
            )
            if self.on_assistant_text:
                self.on_assistant_text(ev.get("transcript") or "")

        elif t == "response.function_call_arguments.done":
            self._start_tool(ev)

        elif t == "response.done":
            resp = ev.get("response") or {}
            batch = self._tool_batches.get(resp.get("id", ""))
            if batch is not None:
                batch.done.set()
            status = resp.get("status", "?")
            if status == "cancelled":
                self.journal.write(
                    J.RESPONSE_CANCELLED,
                    response_id=resp.get("id", ""),
                    reason=(resp.get("status_details") or {}).get("reason", ""),
                )
            else:
                self.journal.write(J.RESPONSE_COMPLETED, response_id=resp.get("id", ""), status=status)
            if self.on_response_done:
                self.on_response_done(status)

        elif t == "error":
            log.error("realtime server error: %s", ev.get("error"))
            self.journal.write("realtime.error", error=ev.get("error"))

    def _start_tool(self, ev: dict) -> None:
        rid = ev.get("response_id") or ""
        batch = self._tool_batches.get(rid)
        if batch is None:
            batch = self._tool_batches[rid] = _ToolBatch(self._epoch, self._origin)
            asyncio.get_event_loop().create_task(self._finish_batch(rid, batch), name=f"tools-{rid}")
        batch.tasks.append(asyncio.get_event_loop().create_task(self._run_tool(ev), name=f"tool-{ev.get('name')}"))
        if not rid:
            batch.done.set()  # no response id to wait for: answer when the tool is done

    async def _run_tool(self, ev: dict) -> tuple[str, dict, bool]:
        name = ev.get("name", "")
        try:
            args = json.loads(ev.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        try:
            fn = getattr(self.router, "timeout_for", None)
            timeout = float(fn(name)) if callable(fn) else TOOL_TIMEOUT_S
        except (TypeError, ValueError):  # a router that returns nonsense must not kill the tool
            timeout = TOOL_TIMEOUT_S
        try:
            result = await asyncio.wait_for(asyncio.to_thread(self.router.execute, name, args), timeout)
        except asyncio.TimeoutError:
            result = {"ok": False, "error": TIMEOUT_MESSAGE.format(name=name)}
            self.journal.write("tool.timeout", tool=name, seconds=timeout)
        return ev.get("call_id", ""), result, self.router.needs_followup(name, result)

    async def _finish_batch(self, rid: str, batch: _ToolBatch) -> None:
        try:
            await asyncio.wait_for(batch.done.wait(), BATCH_WAIT_S)
        except asyncio.TimeoutError:
            pass
        results = await asyncio.gather(*batch.tasks)
        self._tool_batches.pop(rid, None)
        if self._ws is None:
            return
        for call_id, result, _ in results:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(result, default=str),
                        },
                    }
                )
            )
        if not any(follow for _, _, follow in results):
            return
        if self._epoch != batch.epoch:
            # the user spoke while the tool ran: answer them, not the old question
            self.journal.write("tool.followup_skipped", reason="user spoke")
            return
        # Re-latch the origin this reply belongs to: the next response.created swaps
        # _pending_origin into _origin, and a default of ORIGIN_USER here would hand
        # a greeter-prompted follow-up to the lease as if a person had asked.
        self._pending_origin = batch.origin
        await self._ws.send(json.dumps({"type": "response.create"}))


def _realtime_tool(tool: dict) -> dict:
    """Realtime sessions take FLAT function tools ({type, name, description,
    parameters}); our TOOLS are Chat-Completions shaped ({type, function:
    {...}}). Sent nested, s2s wraps them again and llama.cpp 500s with
    "Failed to parse tools: key 'name' not found" on every turn."""
    fn = tool.get("function")
    if tool.get("type") == "function" and isinstance(fn, dict):
        return {"type": "function", **fn}
    return tool
