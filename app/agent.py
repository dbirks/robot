"""Reachy Mini Realtime API agent with motion control."""

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from agents import function_tool
from agents.realtime import RealtimeAgent, RealtimeRunner
from app.tools import Robot

# Global robot context (connects to local daemon/sim)
# For wireless later: Robot(host="http://<pi-ip>:8000")
ROBOT = Robot()


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


# --- Main Entry Point ---

async def main():
    """Run the Realtime agent with voice control."""
    # 1) Define a realtime agent with motion tools
    agent = RealtimeAgent(
        name="Mini voice",
        instructions=(
            "You are the voice of a tiny desk robot named Reachy Mini. "
            "Use tools to emote and look around. Be friendly and concise."
        ),
        tools=[nod, shake, look_at, antenna_wiggle],
    )

    # 2) Realtime runner config: model & audio settings
    # TIP: Start with modalities=["text"] to test tool calls without audio,
    #      then switch to ["audio"] for full voice experience
    runner = RealtimeRunner(
        starting_agent=agent,
        config={
            "model_settings": {
                # Model options:
                # - "gpt-realtime" (canonical GA model)
                # - "gpt-4o-mini-realtime" (lower cost, check availability)
                "model_name": "gpt-realtime",
                "voice": "ash",  # Options: alloy, ash, ballad, coral, sage, verse
                "modalities": ["audio"],  # ["text"] for testing, ["audio"] for voice
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
            }
        },
    )

    # 3) Start the realtime session and process events
    print("Starting Realtime session...")
    print("(Ensure reachy-mini-daemon --sim is running on localhost:8000)")
    print()

    session = await runner.run()
    async with session:
        print("✓ Realtime session started.")
        print("  Speak to interact with the robot (voice mode enabled).")
        print("  Press Ctrl+C to exit.")
        print()

        try:
            async for event in session:
                # Log all event types for debugging
                print(f"[DEBUG] Event type: {event.type}")

                if event.type == "tool_start":
                    print(f"🤖 TOOL_START: {event.tool.name}")

                elif event.type == "tool_end":
                    print(f"✓ TOOL_END: {event.tool.name} -> {event.output}")

                elif event.type == "audio":
                    # Audio chunks for playback (implement audio output here)
                    print(f"🔊 Audio chunk received (length: {len(event.audio) if hasattr(event, 'audio') else 'unknown'})")

                elif event.type == "input_audio_buffer.speech_started":
                    print("🎤 Speech detected - listening...")

                elif event.type == "input_audio_buffer.speech_stopped":
                    print("🎤 Speech stopped - processing...")

                elif event.type == "conversation.item.input_audio_transcription.completed":
                    print(f"📝 Transcription: {event.transcript}")

                elif event.type == "response.text.delta":
                    print(f"💬 Text: {event.delta}", end="", flush=True)

                elif event.type == "response.text.done":
                    print()  # Newline after text completion

                elif event.type == "response.audio_transcript.delta":
                    print(f"🗣️ TTS: {event.delta}", end="", flush=True)

                elif event.type == "response.audio_transcript.done":
                    print()  # Newline after audio transcript

                elif event.type == "error":
                    print(f"❌ ERROR: {event.error}")

                elif event.type == "response_done":
                    print("✓ Response completed")
                    print()

        except KeyboardInterrupt:
            print("\n\nShutting down...")


if __name__ == "__main__":
    # Load API key from .env file
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Create a .env file with: OPENAI_API_KEY=sk-...")
        exit(1)

    asyncio.run(main())
