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

log = logging.getLogger("shell.realtime")


class RealtimeClient:
    def __init__(
        self,
        url: str,
        *,
        instructions: str,
        tools: list[dict],
        tool_router,
        speaker,
        journal,
        on_speech_started: Callable[[], None] | None = None,
        on_transcript: Callable[[str], None] | None = None,
        on_response_done: Callable[[str], None] | None = None,
        on_assistant_text: Callable[[str], None] | None = None,
        reconnect_s: float = 2.0,
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
        self.reconnect_s = reconnect_s
        self._connect = connect  # injectable for tests; defaults to websockets
        self._in: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._ws = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._resp_gen: dict[str, int] = {}  # response_id -> speaker generation
        self.connected = False
        self._cur: dict[str, float] = {}  # per-turn latency marks
        self._first_audio: set[str] = set()  # response ids already marked

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

    def insert_user_transcript(self, text: str) -> None:
        """Inject an already-transcribed (accepted ambient) utterance without
        re-running STT, then request a response. Must be called on the loop."""
        asyncio.get_event_loop().create_task(self._inject(text))

    async def _inject(self, text: str) -> None:
        if self._ws is None:
            return
        await self._ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
                }
            )
        )
        await self._ws.send(json.dumps({"type": "response.create"}))

    # ---- lifecycle ----

    def reset_session(self, extra_instructions: str = "") -> None:
        """Drop the s2s conversation and reconnect with a fresh session whose
        instructions carry `extra_instructions`. Safe from any thread; queued
        mic audio is kept and flows into the new session."""
        if self._loop is None:
            return

        def _do() -> None:
            self._extra_instructions = extra_instructions
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
        while True:
            try:
                await self._session()
            except asyncio.CancelledError:
                raise
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
            self._ws = ws
            self.connected = True
            log.info("realtime connected: %s", self.url)
            # The server's first frame is some flavor of session created; we do
            # not await a specific type because it varies and never guarantees
            # session.updated. Then send a minimal session.update.
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "instructions": self.instructions + self._extra_instructions,
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
            cut = self.speaker.cancel_current()
            self.journal.write(J.AUDIO_SPEECH_STARTED, cut_playback=cut)
            if self.on_speech_started:
                self.on_speech_started()

        elif t == "input_audio_buffer.speech_stopped":
            self.journal.write(J.AUDIO_SPEECH_STOPPED)

        elif t == "conversation.item.input_audio_transcription.completed":
            text = (ev.get("transcript") or "").strip()
            self.journal.write(J.TURN_TRANSCRIPT_FINAL, text=text)
            if self.on_transcript:
                self.on_transcript(text)

        elif t == "response.created":
            rid = (ev.get("response") or {}).get("id", "")
            self._resp_gen[rid] = self.speaker.begin_generation()
            self._first_audio.clear()  # one response in flight at a time
            self.journal.write(J.RESPONSE_CREATED, response_id=rid)

        elif t == "response.output_audio.delta":
            rid = ev.get("response_id", "")
            gen = self._resp_gen.get(rid, self.speaker.generation)
            pcm = np.frombuffer(base64.b64decode(ev["delta"]), dtype=np.int16)
            if self.speaker.enqueue(gen, "tts", pcm) and rid not in self._first_audio:
                self._first_audio.add(rid)  # only mark the FIRST delta
                self.journal.write(J.RESPONSE_FIRST_AUDIO, response_id=rid)

        elif t == "response.output_audio_transcript.done":
            if self.on_assistant_text:
                self.on_assistant_text(ev.get("transcript") or "")

        elif t == "response.function_call_arguments.done":
            await self._handle_function_call(ev)

        elif t == "response.done":
            resp = ev.get("response") or {}
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

    async def _handle_function_call(self, ev: dict) -> None:
        name = ev.get("name", "")
        call_id = ev.get("call_id", "")
        try:
            args = json.loads(ev.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        result = await asyncio.to_thread(self.router.execute, name, args)
        follow_up = self.router.needs_followup(name, result)
        if self._ws is None:
            return
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
        if follow_up:
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
