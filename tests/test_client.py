import asyncio
import base64
import json

import numpy as np

from shell.realtime import RealtimeClient


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))


def make_client(journal, speaker, router=None):
    return RealtimeClient(
        "ws://fake",
        instructions="test",
        tools=[],
        tool_router=router or object(),
        speaker=speaker,
        journal=journal,
    )


def test_speech_started_flushes_playback(speaker, journal):
    c = make_client(journal, speaker)
    fired = []
    c.on_speech_started = lambda: fired.append(1)
    gen = speaker.begin_generation()
    speaker.enqueue(gen, "tts", np.zeros(100, dtype=np.int16))

    asyncio.run(c.handle_event({"type": "input_audio_buffer.speech_started"}))
    assert speaker.pending() == 0
    assert fired == [1]
    assert journal.find("audio.speech_started")[0]["cut_playback"] is True


def test_barge_in_then_late_deltas_refused(speaker, journal):
    c = make_client(journal, speaker)
    asyncio.run(c.handle_event({"type": "response.created", "response": {"id": "r1"}}))
    delta = {
        "type": "response.output_audio.delta",
        "response_id": "r1",
        "delta": base64.b64encode(np.zeros(64, dtype=np.int16).tobytes()).decode(),
    }
    asyncio.run(c.handle_event(delta))
    assert journal.find("response.first_audio")

    asyncio.run(c.handle_event({"type": "input_audio_buffer.speech_started"}))
    before = speaker.pending()
    asyncio.run(c.handle_event(delta))  # stale chunk still in flight
    assert speaker.pending() == before


def test_transcript_routed_to_listeners(journal, speaker):
    seen = []
    c = make_client(
        journal,
        speaker,
    )
    c.on_transcript = seen.append
    asyncio.run(
        c.handle_event({"type": "conversation.item.input_audio_transcription.completed", "transcript": "  hey robot  "})
    )
    assert seen == ["hey robot"]
    assert journal.find("turn.transcript_final")[0]["text"] == "hey robot"


def test_function_call_output_and_followup(journal, speaker):
    class StubRouter:
        def execute(self, name, args):
            self.done = (name, args)
            return {"ok": True, "temp": "61C"}

        def needs_followup(self, name, result):
            return True

    r = StubRouter()
    c = make_client(journal, speaker, router=r)
    ws = FakeWS()
    c._ws = ws
    asyncio.run(
        c.handle_event(
            {
                "type": "response.function_call_arguments.done",
                "name": "get_robot_status",
                "call_id": "c1",
                "arguments": "{}",
            }
        )
    )
    assert r.done == ("get_robot_status", {})
    assert ws.sent[0]["item"]["type"] == "function_call_output"
    assert ws.sent[0]["item"]["call_id"] == "c1"
    assert ws.sent[1] == {"type": "response.create"}


def test_fire_and_forget_function_call_gets_no_followup(journal, speaker):
    class StubRouter:
        def execute(self, name, args):
            return {"ok": True}

        def needs_followup(self, name, result):
            return False

    c = make_client(journal, speaker, router=StubRouter())
    ws = FakeWS()
    c._ws = ws
    asyncio.run(
        c.handle_event(
            {"type": "response.function_call_arguments.done", "name": "nod", "call_id": "c2", "arguments": "{}"}
        )
    )
    assert len(ws.sent) == 1


def test_cancelled_response_journaled_with_reason(journal, speaker):
    c = make_client(journal, speaker)
    asyncio.run(
        c.handle_event(
            {
                "type": "response.done",
                "response": {"id": "r9", "status": "cancelled", "status_details": {"reason": "turn_detected"}},
            }
        )
    )
    ev = journal.find("response.cancelled")[0]
    assert ev["reason"] == "turn_detected"


def test_tools_sent_flat_for_realtime():
    from shell.realtime.client import _realtime_tool

    nested = {"type": "function", "function": {"name": "look_left", "description": "d", "parameters": {}}}
    assert _realtime_tool(nested) == {"type": "function", "name": "look_left", "description": "d", "parameters": {}}
    flat = {"type": "function", "name": "x", "parameters": {}}
    assert _realtime_tool(flat) == flat


def test_first_audio_journaled_once_per_response(journal, speaker):
    import asyncio
    import base64

    c = make_client(journal, speaker)
    delta = base64.b64encode(np.zeros(160, dtype=np.int16).tobytes()).decode()

    async def go():
        await c.handle_event({"type": "response.created", "response": {"id": "r1"}})
        for _ in range(5):
            await c.handle_event({"type": "response.output_audio.delta", "response_id": "r1", "delta": delta})

    asyncio.run(go())
    assert journal.types().count("response.first_audio") == 1


def test_assistant_transcript_routed(journal, speaker):
    got = []
    c = RealtimeClient(
        "ws://fake",
        instructions="t",
        tools=[],
        tool_router=object(),
        speaker=speaker,
        journal=journal,
        on_assistant_text=got.append,
    )
    asyncio.run(c.handle_event({"type": "response.output_audio_transcript.done", "transcript": "Hello there."}))
    assert got == ["Hello there."]


def test_reset_session_reconnects_with_extra_instructions_and_keeps_audio(journal, speaker):
    sessions = []

    class WS:
        def __init__(self):
            self.sent = []
            self.closed = asyncio.Event()

        async def send(self, raw):
            self.sent.append(json.loads(raw))

        def __aiter__(self):
            return self

        async def __anext__(self):
            await self.closed.wait()
            raise StopAsyncIteration

    class Conn:
        async def __aenter__(self):
            ws = WS()
            sessions.append(ws)
            return ws

        async def __aexit__(self, *a):
            sessions[-1].closed.set()

    c = RealtimeClient(
        "ws://fake",
        instructions="base",
        tools=[],
        tool_router=object(),
        speaker=speaker,
        journal=journal,
        connect=Conn,
    )

    async def go():
        task = asyncio.create_task(c.run())
        await asyncio.sleep(0.05)
        c.reset_session(" EXTRA")  # on-loop: applies before the next feed
        c.feed(b"\x01\x00" * 4)
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(go())
    assert len(sessions) == 2
    assert sessions[0].sent[0]["session"]["instructions"] == "base"
    assert sessions[1].sent[0]["session"]["instructions"] == "base EXTRA"
    appended = [m for m in sessions[1].sent if m["type"] == "input_audio_buffer.append"]
    assert appended, "audio fed at reset time must reach the NEW session"
    assert not [m for m in sessions[0].sent if m["type"] == "input_audio_buffer.append"]
