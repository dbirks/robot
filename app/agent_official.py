#!/usr/bin/env python
"""
Minimal Reachy Mini Realtime agent based on OpenAI official example.
Uses gpt-4o-realtime-preview with sounddevice for audio I/O.
"""

import asyncio
import os
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from agents.voice import (
    RealtimeVoicePipeline,
    StreamedAudioInput,
    VoicePipelineConfig,
)
from agents.tool import function_tool
from agents.voice.models.sdk_realtime import SDKRealtimeLLM
from app.tools import Robot

# Load environment
load_dotenv()

# Robot instance
ROBOT = Robot()

# Audio config
SAMPLE_RATE = 24000
CHUNK_DURATION = 0.1  # 100ms chunks


# --- Robot Tools ---

@function_tool
async def nod(times: int = 1) -> str:
    """Nod head up/down."""
    async with ROBOT:
        await ROBOT.nod(times)
    return "nodded"

@function_tool
async def shake(times: int = 1) -> str:
    """Shake head left/right."""
    async with ROBOT:
        await ROBOT.shake(times)
    return "shook head"

@function_tool
async def look_at(x_deg: float, y_deg: float) -> str:
    """Look at angles (yaw, pitch)."""
    async with ROBOT:
        await ROBOT.look_at(x_deg, y_deg)
    return f"looking at ({x_deg}, {y_deg})"

@function_tool
async def antenna_wiggle(seconds: int = 2) -> str:
    """Wiggle antennas."""
    async with ROBOT:
        await ROBOT.antenna_wiggle(seconds)
    return "wiggled antennas"


async def main():
    """Run the realtime voice assistant."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    print("🎤 Reachy Mini Realtime Agent")
    print("   Model: gpt-4o-realtime-preview")
    print("   Sample rate: 24kHz")
    print()
    print("✓ Speak now! (Ctrl+C to quit)")
    print()

    # Create audio input stream
    audio_input = StreamedAudioInput(sample_rate=SAMPLE_RATE)

    # Initialize model
    model = SDKRealtimeLLM(
        model_name="gpt-4o-realtime-preview",
        api_key=api_key,
    )

    # Configure pipeline
    config = VoicePipelineConfig(
        realtime_settings={
            "turn_detection": "server_vad",
            "assistant_voice": "ash",
            "system_message": "You're a tiny desk robot. Use tools to emote. Be concise.",
        }
    )

    # Create pipeline with tools
    pipeline = RealtimeVoicePipeline(
        llm=model,
        config=config,
        tools=[nod, shake, look_at, antenna_wiggle],
    )

    # Start pipeline
    stream = await pipeline.start(audio_input=audio_input)

    # Audio playback state
    output_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16,
    )
    output_stream.start()

    # Start mic input
    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"[MIC] {status}")
        audio_input.write(indata.copy())

    mic_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16,
        callback=mic_callback,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
    )
    mic_stream.start()

    try:
        # Process events
        async for event in stream:
            if event.type == "voice_stream_event_audio":
                # Play audio
                if event.data and len(event.data) > 0:
                    output_stream.write(event.data)

            elif event.type == "voice_stream_event_tool_call":
                print(f"🤖 TOOL: {event.tool_name}")

            elif event.type == "voice_stream_event_lifecycle":
                if "input_audio_transcription_completed" in str(event):
                    # User speech transcribed
                    pass

            elif event.type == "voice_stream_event_error":
                print(f"❌ ERROR: {event}")

    except KeyboardInterrupt:
        print("\n\n👋 Bye!")
    finally:
        mic_stream.stop()
        output_stream.stop()
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())
