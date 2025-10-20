"""Reachy Mini Realtime API agent with LIVE voice capture/playback."""

import asyncio
import os
import queue
import base64
import numpy as np
import sounddevice as sd
from scipy import signal

from dotenv import load_dotenv
from agents import function_tool
from agents.realtime import RealtimeAgent, RealtimeRunner
from app.tools import Robot

# Global robot context
ROBOT = Robot()

# Audio configuration for OpenAI Realtime API
API_SAMPLE_RATE = 24000  # Required by OpenAI
CHANNELS = 1
DTYPE = np.int16

# Device audio configuration (will be detected dynamically)
DEVICE_SAMPLE_RATE = None

# Queues for audio I/O
audio_input_queue = queue.Queue()
audio_output_queue = queue.Queue()


# --- Function Tools ---

@function_tool
async def nod(times: int = 1) -> str:
    """Nod head up/down a few times.

    Args:
        times: number of nods (1–5)
    """
    async with _robot_session():
        await ROBOT.nod(times=times)
    return "ok"


@function_tool
async def shake(times: int = 1) -> str:
    """Shake head left/right a few times.

    Args:
        times: number of shakes (1–5)
    """
    async with _robot_session():
        await ROBOT.shake(times=times)
    return "ok"


@function_tool
async def look_at(x_deg: float, y_deg: float) -> str:
    """Look at absolute angles (degrees).

    Args:
        x_deg: yaw (+right/−left), clamped to ±35
        y_deg: pitch (+up/−down), clamped to ±20
    """
    async with _robot_session():
        await ROBOT.look_at(x_deg, y_deg)
    return "ok"


@function_tool
async def antenna_wiggle(seconds: int = 2) -> str:
    """Wiggle antennas for N seconds.

    Args:
        seconds: 1–10
    """
    async with _robot_session():
        await ROBOT.antenna_wiggle(seconds=seconds)
    return "ok"


# --- Robot Session Helper ---

class _robot_session:
    """Async context manager to ensure Robot has active connection."""

    async def __aenter__(self):
        ROBOT.__enter__()
        return ROBOT

    async def __aexit__(self, exc_type, exc, tb):
        ROBOT.__exit__(exc_type, exc, tb)


# --- Audio I/O Handlers ---

_mic_chunks = 0
_audio_level_sum = 0

def audio_input_callback(indata, frames, time, status):
    """Callback for microphone input - called by sounddevice."""
    global _mic_chunks, _audio_level_sum, DEVICE_SAMPLE_RATE

    if status:
        print(f"[Audio Input Status]: {status}")

    # Validate global is set
    if DEVICE_SAMPLE_RATE is None:
        print("[ERROR] DEVICE_SAMPLE_RATE not set!")
        return

    # Convert to numpy array and validate
    audio_data = indata.copy().flatten()

    # Debug first chunk
    if _mic_chunks == 0:
        print(f"[DEBUG] First audio chunk:")
        print(f"  Shape: {indata.shape}")
        print(f"  Dtype: {indata.dtype}")
        print(f"  Min/Max: {np.min(indata)}/{np.max(indata)}")
        print(f"  First 5 samples: {audio_data[:5]}")

    # Calculate audio level (RMS) - convert to float first to avoid overflow
    audio_float = audio_data.astype(np.float64) / 32768.0  # Normalize to [-1, 1]
    audio_level = float(np.sqrt(np.mean(audio_float**2)))

    # Handle NaN
    if np.isnan(audio_level) or np.isinf(audio_level):
        audio_level = 0.0

    _audio_level_sum += audio_level

    # Resample from device rate to API rate if needed
    if DEVICE_SAMPLE_RATE != API_SAMPLE_RATE:
        # Calculate number of samples after resampling
        num_samples_out = int(len(audio_data) * API_SAMPLE_RATE / DEVICE_SAMPLE_RATE)
        # Resample using scipy (works on int16)
        audio_data_resampled = signal.resample(audio_data, num_samples_out)
        # Convert back to int16
        audio_data_resampled = np.clip(audio_data_resampled, -32768, 32767).astype(DTYPE)

        # Validate resampled data
        if _mic_chunks == 0:
            print(f"[DEBUG] After resampling:")
            print(f"  Original samples: {len(audio_data)}")
            print(f"  Resampled samples: {len(audio_data_resampled)}")
            print(f"  Min/Max: {np.min(audio_data_resampled)}/{np.max(audio_data_resampled)}")

        audio_data = audio_data_resampled

    # Queue the resampled audio
    audio_input_queue.put(audio_data.tobytes())

    _mic_chunks += 1
    if _mic_chunks % 100 == 0:
        avg_level = _audio_level_sum / 100
        print(f"[DEBUG] Mic: {_mic_chunks} chunks, avg level: {avg_level:.4f}, range: [{np.min(audio_data)}, {np.max(audio_data)}]")
        _audio_level_sum = 0


def audio_output_callback(outdata, frames, time, status):
    """Callback for speaker output - called by sounddevice."""
    if status:
        print(f"[Audio Output Status]: {status}")

    try:
        # Get audio from queue (non-blocking)
        data = audio_output_queue.get_nowait()

        # Ensure data is bytes
        if not isinstance(data, bytes):
            print(f"[ERROR] Expected bytes, got {type(data)}")
            outdata[:] = np.zeros((frames, CHANNELS), dtype=DTYPE)
            return

        # Convert bytes to numpy array
        audio_array = np.frombuffer(data, dtype=DTYPE)

        # Resample if needed (API rate → device rate)
        if DEVICE_SAMPLE_RATE != API_SAMPLE_RATE:
            num_samples_out = int(len(audio_array) * DEVICE_SAMPLE_RATE / API_SAMPLE_RATE)
            audio_array = signal.resample(audio_array, num_samples_out).astype(DTYPE)

        # Pad or trim to match frame size
        if len(audio_array) < frames:
            audio_array = np.pad(audio_array, (0, frames - len(audio_array)))
        elif len(audio_array) > frames:
            audio_array = audio_array[:frames]

        # Reshape for output
        outdata[:] = audio_array.reshape(-1, 1)
    except queue.Empty:
        # No audio available, output silence
        outdata[:] = np.zeros((frames, CHANNELS), dtype=DTYPE)


async def send_audio_task(session):
    """Background task to send mic audio to the Realtime API."""
    print("🎤 Audio input task started")
    chunk_count = 0
    try:
        while True:
            # Get audio from input queue
            try:
                audio_bytes = audio_input_queue.get(timeout=0.1)
                await session.send_audio(audio_bytes)
                chunk_count += 1
                if chunk_count % 100 == 0:  # Log every ~2 seconds of audio
                    print(f"[DEBUG] Sent {chunk_count} audio chunks to API")
            except queue.Empty:
                await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        print(f"🎤 Audio input task stopped (sent {chunk_count} chunks total)")


# --- Main Entry Point ---

async def main():
    """Run the Realtime agent with live voice I/O."""
    # Define realtime agent with motion tools
    agent = RealtimeAgent(
        name="Mini voice",
        instructions=(
            "You are the voice of a tiny desk robot named Reachy Mini. "
            "Use tools to emote and look around. Be friendly and concise."
        ),
        tools=[nod, shake, look_at, antenna_wiggle],
    )

    # Realtime runner config with improved VAD settings
    runner = RealtimeRunner(
        starting_agent=agent,
        config={
            "model_settings": {
                "model_name": "gpt-realtime",
                "voice": "ash",
                "modalities": ["audio"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.3,  # Lower threshold = more sensitive
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200,  # Faster response
                },
            }
        },
    )

    # Detect device sample rate
    global DEVICE_SAMPLE_RATE
    device_info = sd.query_devices(kind='input')
    DEVICE_SAMPLE_RATE = int(device_info['default_samplerate'])

    print("Starting Realtime session...")
    print("(Ensure reachy-mini-daemon --sim is running on localhost:8000)")
    print()
    print(f"🎤 Audio Configuration:")
    print(f"   Device sample rate: {DEVICE_SAMPLE_RATE} Hz")
    print(f"   API sample rate: {API_SAMPLE_RATE} Hz")
    if DEVICE_SAMPLE_RATE != API_SAMPLE_RATE:
        print(f"   ⚠️  Will resample {DEVICE_SAMPLE_RATE}Hz → {API_SAMPLE_RATE}Hz")
    print()

    # Start audio streams at device rate
    print("🔊 Starting audio I/O...")
    input_stream = sd.InputStream(
        samplerate=DEVICE_SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=audio_input_callback
    )
    output_stream = sd.OutputStream(
        samplerate=DEVICE_SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=audio_output_callback
    )

    input_stream.start()
    output_stream.start()

    session = await runner.run()

    async with session:
        print("✓ Realtime session started.")
        print("  Speak to interact with the robot (live voice mode).")
        print("  Press Ctrl+C to exit.")
        print()

        # Start background task to send audio
        audio_task = asyncio.create_task(send_audio_task(session))

        try:
            async for event in session:
                # Debug: log all events except noisy ones
                if event.type not in ["raw_model_event", "history_updated"]:
                    print(f"[EVENT] {event.type}")

                if event.type == "tool_start":
                    print(f"🤖 TOOL: {event.tool.name}")

                elif event.type == "tool_end":
                    print(f"✓ TOOL DONE: {event.tool.name}")

                elif event.type == "audio":
                    # Handle audio output event
                    # Try multiple attribute names based on SDK version
                    audio_bytes = None

                    # Method 1: Check for 'data' attribute
                    if hasattr(event, 'data') and event.data:
                        if isinstance(event.data, bytes):
                            audio_bytes = event.data
                        elif isinstance(event.data, str):
                            # Might be base64-encoded
                            try:
                                audio_bytes = base64.b64decode(event.data)
                            except Exception as e:
                                print(f"[ERROR] Failed to decode base64 data: {e}")

                    # Method 2: Check for 'delta' attribute (raw API)
                    if not audio_bytes and hasattr(event, 'delta') and event.delta:
                        if isinstance(event.delta, bytes):
                            audio_bytes = event.delta
                        elif isinstance(event.delta, str):
                            try:
                                audio_bytes = base64.b64decode(event.delta)
                            except Exception as e:
                                print(f"[ERROR] Failed to decode base64 delta: {e}")

                    # Method 3: Check for 'audio' attribute
                    if not audio_bytes and hasattr(event, 'audio') and event.audio:
                        if isinstance(event.audio, bytes):
                            audio_bytes = event.audio
                        elif isinstance(event.audio, str):
                            try:
                                audio_bytes = base64.b64decode(event.audio)
                            except Exception as e:
                                print(f"[ERROR] Failed to decode base64 audio: {e}")

                    if audio_bytes:
                        audio_output_queue.put(audio_bytes)
                        print(f"🔊 Queued audio ({len(audio_bytes)} bytes)")
                    else:
                        # Debug: show what attributes are available
                        attrs = [attr for attr in dir(event) if not attr.startswith('_')]
                        print(f"[DEBUG] Audio event has no recognizable data. Attributes: {attrs}")

                elif event.type == "input_audio_buffer.speech_started":
                    print("🎤 Speech detected - listening...")

                elif event.type == "input_audio_buffer.speech_stopped":
                    print("🎤 Speech stopped - processing...")

                elif event.type == "conversation.item.input_audio_transcription.completed":
                    if hasattr(event, 'transcript'):
                        print(f"📝 You: {event.transcript}")

                elif event.type == "response.audio_transcript.delta":
                    if hasattr(event, 'delta'):
                        print(f"{event.delta}", end="", flush=True)

                elif event.type == "response.audio_transcript.done":
                    print()  # Newline

                elif event.type == "error":
                    print(f"❌ ERROR: {event.error}")

        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            # Cleanup
            audio_task.cancel()
            input_stream.stop()
            output_stream.stop()
            input_stream.close()
            output_stream.close()


if __name__ == "__main__":
    # Load API key from .env
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Create a .env file with: OPENAI_API_KEY=sk-...")
        exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
