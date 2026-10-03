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
