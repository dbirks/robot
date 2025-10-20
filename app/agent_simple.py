"""Simplified Reachy Mini Realtime API agent - just works."""

import asyncio
import os
import queue
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from agents import function_tool
from agents.realtime import RealtimeAgent, RealtimeRunner
from app.tools import Robot

# Robot instance
ROBOT = Robot()

# Audio config - OpenAI requires 24kHz PCM16
SAMPLE_RATE = 24000
CHANNELS = 1

# Audio queues
audio_in = queue.Queue()
audio_out = queue.Queue()


# --- Robot Tools ---

@function_tool
async def nod(times: int = 1) -> str:
    """Nod head up/down."""
    async with ROBOT:
        await ROBOT.nod(times)
    return "ok"

@function_tool
async def shake(times: int = 1) -> str:
    """Shake head left/right."""
    async with ROBOT:
        await ROBOT.shake(times)
    return "ok"

@function_tool
async def look_at(x_deg: float, y_deg: float) -> str:
    """Look at angles (yaw, pitch)."""
    async with ROBOT:
        await ROBOT.look_at(x_deg, y_deg)
    return "ok"

@function_tool
async def antenna_wiggle(seconds: int = 2) -> str:
    """Wiggle antennas."""
    async with ROBOT:
        await ROBOT.antenna_wiggle(seconds)
    return "ok"


# --- Audio Callbacks ---

def mic_callback(indata, frames, time, status):
    """Capture mic audio."""
    if status:
        print(f"[MIC ERROR] {status}")
    audio_in.put(indata.copy().tobytes())

def speaker_callback(outdata, frames, time, status):
    """Play audio to speaker."""
    if status:
        print(f"[SPEAKER ERROR] {status}")
    try:
        data = audio_out.get_nowait()
        audio_array = np.frombuffer(data, dtype=np.int16)
        if len(audio_array) < frames * CHANNELS:
            audio_array = np.pad(audio_array, (0, frames * CHANNELS - len(audio_array)))
        elif len(audio_array) > frames * CHANNELS:
            audio_array = audio_array[:frames * CHANNELS]
        outdata[:] = audio_array.reshape(-1, CHANNELS)
    except queue.Empty:
        outdata[:] = np.zeros((frames, CHANNELS), dtype=np.int16)


# --- Send Audio to OpenAI ---

async def send_audio_loop(session):
    """Send mic audio to OpenAI."""
    while True:
        try:
            audio_bytes = audio_in.get(timeout=0.1)
            await session.send_audio(audio_bytes)
        except queue.Empty:
            await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            break


# --- Main ---

async def main():
    """Run the agent."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return

    # Create agent
    agent = RealtimeAgent(
        name="Mini",
        instructions="You're a tiny desk robot. Use tools to emote. Be concise.",
        tools=[nod, shake, look_at, antenna_wiggle],
    )

    # Create runner with Realtime API model
    runner = RealtimeRunner(
        starting_agent=agent,
        config={
            "model_settings": {
                "model_name": "gpt-realtime-mini",  # Cost-efficient Realtime model
                "voice": "ash",
                "modalities": ["audio"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},  # Enable transcription
                "turn_detection": {"type": "server_vad", "threshold": 0.3},
            }
        },
    )

    # Check device sample rate support
    default_device = sd.query_devices(kind='input')
    print(f"🎤 Microphone: {default_device['name']}")
    print(f"   Default rate: {int(default_device['default_samplerate'])}Hz")
    print(f"   Using: {SAMPLE_RATE}Hz (required by OpenAI)")
    if int(default_device['default_samplerate']) != SAMPLE_RATE:
        print(f"   ⚠️  Device prefers {int(default_device['default_samplerate'])}Hz, forcing {SAMPLE_RATE}Hz")
    print()

    # Start audio streams
    mic_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.int16,
        callback=mic_callback
    )
    speaker_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.int16,
        callback=speaker_callback
    )

    mic_stream.start()
    speaker_stream.start()

    # Run session
    session = await runner.run()

    async with session:
        print("✓ Connected. Speak now!")
        print("📝 Transcriptions will appear below:")
        print()

        # Start audio sender
        sender = asyncio.create_task(send_audio_loop(session))

        try:
            async for event in session:
                # Log ALL events for debugging (except noisy ones)
                if event.type not in ["raw_model_event", "history_updated"]:
                    print(f"[EVENT] {event.type}")

                if event.type == "tool_start":
                    print(f"🤖 TOOL: {event.tool.name}")

                elif event.type == "tool_end":
                    print(f"✓ TOOL DONE: {event.tool.name}")

                elif event.type == "audio":
                    # Debug: what's actually in this event?
                    import base64

                    # The event itself might BE the audio data wrapper
                    # Try to access common attributes
                    audio_bytes = None

                    # Method 1: Check if event has audio attribute that's an object with data
                    if hasattr(event, 'audio'):
                        audio_obj = event.audio
                        if hasattr(audio_obj, 'data'):
                            audio_bytes = audio_obj.data
                        elif hasattr(audio_obj, 'delta'):
                            audio_bytes = audio_obj.delta

                    # Method 2: Direct attributes
                    if not audio_bytes and hasattr(event, 'data'):
                        audio_bytes = event.data

                    if not audio_bytes and hasattr(event, 'delta'):
                        audio_bytes = event.delta

                    # Decode base64 if needed
                    if audio_bytes and isinstance(audio_bytes, str):
                        try:
                            audio_bytes = base64.b64decode(audio_bytes)
                        except:
                            pass

                    # Queue if we got bytes
                    if audio_bytes and isinstance(audio_bytes, bytes):
                        audio_out.put(audio_bytes)
                        print(f"🔊 Playing audio ({len(audio_bytes)} bytes)")
                    else:
                        # Show what we actually have
                        attrs = {k: type(v).__name__ for k, v in event.__dict__.items() if not k.startswith('_')}
                        print(f"[DEBUG] Audio event attributes: {attrs}")

                elif event.type == "input_audio_buffer.speech_started":
                    print("🎤 Speech detected - listening...")

                elif event.type == "input_audio_buffer.speech_stopped":
                    print("🎤 Speech stopped - processing...")

                elif event.type == "conversation.item.input_audio_transcription.completed":
                    # YOUR SPEECH TRANSCRIBED
                    if hasattr(event, 'transcript'):
                        print(f"📝 YOU SAID: \"{event.transcript}\"")

                elif event.type == "response.audio_transcript.delta":
                    # BOT'S RESPONSE TEXT (streaming)
                    if hasattr(event, 'delta'):
                        print(event.delta, end='', flush=True)

                elif event.type == "response.audio_transcript.done":
                    # BOT FINISHED SPEAKING
                    print()  # Newline after bot's text

                elif event.type == "error":
                    print(f"❌ ERROR: {event.error}")

        except KeyboardInterrupt:
            print("\n👋 Bye!")
        finally:
            sender.cancel()
            mic_stream.stop()
            speaker_stream.stop()


if __name__ == "__main__":
    asyncio.run(main())
