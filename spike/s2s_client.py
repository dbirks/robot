"""Minimal OpenAI-Realtime WebSocket client for huggingface/speech-to-speech.

Spike for robot-dz9. Proves the realtime path end-to-end before we commit to
rewriting the voice loop.

The architectural point of this file: ONE persistent input stream and ONE
persistent output stream, opened at startup and never reopened. The current
app opens a new InputStream per turn, another inside InterruptiblePlayer to
watch for barge-in, and a third to capture the rest of the interruption --
plus the watchdog's sd.rec() every 10s. That churn is where speech gets
clipped and devices get contended. Here the audio devices are owned in one
place and everything else talks to queues.

Barge-in needs no local VAD at all: the server sends
input_audio_buffer.speech_started, and we drop the playback buffer on the
floor the moment it arrives. The server independently cancels the in-flight
response via its CancelScope, so no stale audio can arrive afterwards.

Usage:
    .venv/bin/python spike/s2s_client.py            # talk to it
    .venv/bin/python spike/s2s_client.py --seconds 120
"""

import argparse
import asyncio
import base64
import json
import sys
import time
from collections import deque

import numpy as np
import sounddevice as sd
import websockets

WS_URL = "ws://127.0.0.1:8765/v1/realtime"
RATE = 16000          # s2s PIPELINE_SAMPLE_RATE, both directions
BLOCK = 512           # 32ms at 16k
CHANNELS = 1

T0 = time.monotonic()


def ts() -> str:
    return f"{time.monotonic() - T0:7.2f}s"


class Metrics:
    """Per-turn latency, reported as percentiles at exit."""

    def __init__(self):
        self.turns: list[dict] = []
        self.cur: dict = {}
        self.barge_ins = 0
        self.cancelled = 0

    def mark(self, key: str):
        self.cur[key] = time.monotonic()

    def finish(self):
        if self.cur:
            self.turns.append(self.cur)
            self.cur = {}

    @staticmethod
    def _pct(xs, p):
        if not xs:
            return float("nan")
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    def report(self):
        def deltas(a, b):
            return [
                (t[b] - t[a]) * 1000
                for t in self.turns
                if a in t and b in t and t[b] >= t[a]
            ]

        print(f"\n{'=' * 62}\nMETRICS over {len(self.turns)} turns")
        print(f"barge-ins: {self.barge_ins}   cancelled responses: {self.cancelled}")
        rows = [
            ("speech_stopped -> transcript", "speech_stopped", "transcript"),
            ("transcript -> response.created", "transcript", "resp_created"),
            ("speech_stopped -> FIRST AUDIO", "speech_stopped", "first_audio"),
            ("speech_started -> playback flushed", "bargein", "flushed"),
        ]
        for label, a, b in rows:
            ds = deltas(a, b)
            if ds:
                print(
                    f"  {label:36s} n={len(ds):3d}  "
                    f"p50={self._pct(ds, 0.5):7.0f}ms  p95={self._pct(ds, 0.95):7.0f}ms"
                )
            else:
                print(f"  {label:36s} (no samples)")
        print("=" * 62)


class Audio:
    """The single owner of both audio devices."""

    def __init__(self, metrics: Metrics):
        self.metrics = metrics
        self.mic_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self.play_buf = deque()      # of np.int16 arrays
        self.playing = False
        self._loop = asyncio.get_event_loop()
        self.n_cb = 0
        self.n_sched = 0
        self.cb_err = None
        self._lock = __import__("threading").Lock()

        self.in_stream = sd.InputStream(
            samplerate=RATE, channels=CHANNELS, dtype="int16",
            blocksize=BLOCK, callback=self._on_mic,
        )
        self.out_stream = sd.OutputStream(
            samplerate=RATE, channels=CHANNELS, dtype="int16",
            blocksize=BLOCK, callback=self._on_speaker,
        )

    def start(self):
        self.in_stream.start()
        self.out_stream.start()

    def stop(self):
        for s in (self.in_stream, self.out_stream):
            try:
                s.stop()
                s.close()
            except Exception:
                pass

    def _on_mic(self, indata, frames, time_info, status):
        self.n_cb += 1
        # Track level so a dead/quiet mic is distinguishable from a VAD that
        # simply never fired. Silero's default threshold here is 0.6.
        r = float(np.sqrt(np.mean((indata[:, 0].astype(np.float32) / 32768.0) ** 2)))
        self.rms_recent = r
        self.rms_peak = max(getattr(self, "rms_peak", 0.0), r)
        try:
            self._loop.call_soon_threadsafe(
                self.mic_q.put_nowait, bytes(indata[:, 0].tobytes())
            )
            self.n_sched += 1
        except Exception as e:
            self.cb_err = repr(e)

    def _on_speaker(self, outdata, frames, time_info, status):
        need = frames
        pos = 0
        outdata[:] = 0
        with self._lock:
            while need > 0 and self.play_buf:
                chunk = self.play_buf[0]
                take = min(need, len(chunk))
                outdata[pos:pos + take, 0] = chunk[:take]
                if take == len(chunk):
                    self.play_buf.popleft()
                else:
                    self.play_buf[0] = chunk[take:]
                pos += take
                need -= take
            self.playing = bool(self.play_buf)

    def enqueue(self, pcm: np.ndarray):
        with self._lock:
            self.play_buf.append(pcm)
            self.playing = True

    def flush(self) -> bool:
        """Drop everything queued. Returns True if we actually cut off audio."""
        with self._lock:
            had = bool(self.play_buf)
            self.play_buf.clear()
            self.playing = False
        return had


async def sender(ws, audio: Audio):
    audio.n_sent = 0
    while True:
        pcm = await audio.mic_q.get()
        audio.n_sent += 1
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }))


async def receiver(ws, audio: Audio, m: Metrics):
    async for raw in ws:
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        t = ev.get("type", "")

        if t == "input_audio_buffer.speech_started":
            # Barge-in: kill queued audio immediately. The server cancels the
            # in-flight response on its side, so nothing stale follows.
            m.mark("bargein")
            cut = audio.flush()
            m.mark("flushed")
            if cut:
                m.barge_ins += 1
                print(f"{ts()}  >> BARGE-IN — playback cut")
            else:
                print(f"{ts()}  speech started")

        elif t == "input_audio_buffer.speech_stopped":
            m.mark("speech_stopped")
            print(f"{ts()}  speech stopped")

        elif t == "conversation.item.input_audio_transcription.completed":
            m.mark("transcript")
            print(f"{ts()}  YOU: {ev.get('transcript', '').strip()!r}")

        elif t == "response.created":
            m.mark("resp_created")

        elif t == "response.output_audio.delta":
            if "first_audio" not in m.cur:
                m.mark("first_audio")
                d = m.cur.get("first_audio", 0) - m.cur.get("speech_stopped", 0)
                print(f"{ts()}  first audio (+{d * 1000:.0f}ms after speech end)")
            pcm = np.frombuffer(base64.b64decode(ev["delta"]), dtype=np.int16)
            audio.enqueue(pcm)

        elif t == "response.output_audio_transcript.done":
            print(f"{ts()}  BOT: {ev.get('transcript', '').strip()!r}")

        elif t == "response.done":
            status = (ev.get("response") or {}).get("status", "?")
            if status == "cancelled":
                m.cancelled += 1
                reason = ((ev.get("response") or {}).get("status_details") or {}).get("reason", "")
                print(f"{ts()}  response CANCELLED ({reason})")
            m.finish()

        elif t == "error":
            print(f"{ts()}  ERROR: {ev.get('error')}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0, help="auto-exit after N seconds")
    ap.add_argument("--url", default=WS_URL)
    args = ap.parse_args()

    print(f"connecting to {args.url} ...")
    async with websockets.connect(args.url, max_size=None) as ws:
        first = json.loads(await ws.recv())
        print(f"{ts()}  {first.get('type')}")

        # Keep this minimal. The payload is validated against openai's real
        # RealtimeSessionCreateRequest model, and a malformed sub-object is
        # rejected wholesale as "Unknown or invalid event". We omit the audio
        # config entirely because the server default already matches RATE.
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": (
                    "You are Reachy, a small desk robot. Reply in one or two short "
                    "spoken sentences. Never use markdown or lists."
                ),
            },
        }))
        # NOTE: s2s never echoes session.updated -- do not await confirmation.

        m = Metrics()
        audio = Audio(m)
        audio.start()
        print(f"{ts()}  audio up ({RATE}Hz in+out, persistent). Speak. Ctrl-C to stop.\n")

        async def meter():
            while True:
                await asyncio.sleep(2.0)
                r = getattr(audio, "rms_recent", 0.0)
                pk = getattr(audio, "rms_peak", 0.0)
                bar = "#" * min(40, int(r * 600))
                print(f"{ts()}  mic |{bar:<40}| rms={r:.4f} peak={pk:.4f}")

        tasks = [asyncio.create_task(sender(ws, audio), name="sender"),
                 asyncio.create_task(receiver(ws, audio, m), name="receiver"),
                 asyncio.create_task(meter(), name="meter")]
        try:
            if args.seconds:
                done, _ = await asyncio.wait(tasks, timeout=args.seconds)
                # asyncio.wait never raises for a failed task -- a crashed
                # sender just goes quiet, which is exactly how the first run
                # of this spike silently streamed zero audio for 5 minutes.
                for t in done:
                    if not t.cancelled() and t.exception():
                        print(f"{ts()}  !! TASK {t.get_name()} DIED: {t.exception()!r}")
            else:
                await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            for t in tasks:
                t.cancel()
            audio.stop()
            print(f"DEBUG mic_cb={audio.n_cb} sched={audio.n_sched} "
                  f"sent={getattr(audio,'n_sent',0)} qsize={audio.mic_q.qsize()} "
                  f"cb_err={audio.cb_err}")
            m.report()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
