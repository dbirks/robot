# Agent Instructions

This file provides context for AI coding agents (Claude Code, Copilot, Cursor, etc.) working on this repo.

## What this project is

A fully local voice-driven agent for the [Reachy Mini](https://www.reachy-mini.org/) desk robot. No cloud APIs — all inference runs on a single machine with an NVIDIA GPU.

## Stack

| Layer | Component | Runs on |
|-------|-----------|---------|
| VAD | silero-vad | CPU |
| STT | faster-whisper (small.en) | GPU preferred |
| LLM | Qwen 3.5 4B GGUF via llama.cpp | GPU |
| TTS | Piper (en_GB-northern_english_male) | CPU |
| Agent | OpenAI-compatible client (designed to swap in [Hermes](https://github.com/NousResearch/hermes-agent)) | CPU |
| Robot | Reachy Mini SDK v1.6+ | USB/network to robot |

## Architecture

The system runs as separate processes:

1. **`reachy-mini-daemon`** — robot hardware interface (or `--sim` for simulator)
2. **`llama-server`** — LLM inference with OpenAI-compatible API (`--jinja` flag required for tool calling)
3. **`python -m app`** — voice agent (STT + VAD + agent + TTS + audio I/O)

The voice agent runs a synchronous conversation loop: listen → transcribe → agent reasoning + tool calls → TTS → play audio.

## Key design decisions

- **Tool calling uses OpenAI function-calling format.** Tool schemas are in `app/robot_tools.py` as a `TOOLS` list. Handlers are plain Python functions that return dicts. This format is compatible with both the direct OpenAI client and Hermes.
- **llama.cpp must run with `--jinja`** or tool calling silently fails.
- **The agent layer is intentionally thin.** `app/agent_client.py` is ~80 lines. It's designed to be replaced by Hermes once that integration is ready.
- **Robot tools never raise exceptions.** They catch errors and return `{"ok": False, "error": "..."}`.
- **No async in the main loop.** The conversation is inherently sequential (listen, think, speak). Threading is used only for audio capture.

## Workflow

- **Commit often.** Don't let large amounts of work accumulate uncommitted. Commit and push at natural breakpoints — after implementing a feature, fixing a bug, or completing a research-and-implement cycle.

## Working with this codebase

### Package management

Uses **uv** exclusively. Run `uv sync` to install deps, `uv run` to execute.

### Adding a new robot tool

1. Add the OpenAI function schema to `TOOLS` in `app/robot_tools.py`
2. Add a handler function inside `make_handlers()`
3. The handler must return a JSON-serializable dict and never raise

### Changing the LLM

Edit `.env` to point `LLM_BASE_URL` and `LLM_MODEL` at a different OpenAI-compatible server. The agent client doesn't care what's behind the endpoint.

### Running without a robot

The agent starts even if the Reachy Mini isn't connected — robot tools will return `{"ok": False, "error": "Robot not connected"}`. You can test the voice loop and LLM independently.

### Linting

```bash
uv run ruff check app/
uv run ruff format app/
```

## Target hardware

- Older i7 CPU
- NVIDIA GTX 1070 (8 GB VRAM)
- Arch Linux
- Reachy Mini (Lite or Wireless)

## VRAM budget

| Component | VRAM |
|-----------|------|
| faster-whisper small.en (float16) | ~1.5 GB |
| Qwen 3.5 4B Q4_K_M | ~3 GB |
| CUDA overhead | ~0.5 GB |
| **Total** | **~5 GB / 8 GB** |

## Future work

- **Hermes integration** — replace `agent_client.py` with Hermes for persistent memory, session management, and self-improvement
- **Attention model** — wake word or LLM-based filtering so the robot knows when it's being spoken to
- **Vision pipeline** — camera snapshots routed to a vision model for scene description
- **Face tracking** — follow people with head movement, local face embeddings for recognition
- **Emotion library** — re-integrate `reachy-mini-dances-library` for expressive gestures
