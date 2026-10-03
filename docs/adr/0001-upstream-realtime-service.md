# ADR 0001: Use huggingface/speech-to-speech as a pinned external realtime service

- Status: accepted
- Date: 2026-10-02
- Deciders: David (spec co-authored with a coding agent; upstream claims verified 2026-10-02)

## Context

The voice lifecycle bugs we keep hitting — stale speech after interruption, clipped
turn starts, device contention, eavesdropping that executes tools — all trace to our
cascade owning VAD, turn detection, STT, TTS, and cancellation itself, badly, on a
4-core machine. We re-verified upstream on 2026-10-02:

- `huggingface/speech-to-speech` v1.0.0 (2026-09-06), actively developed (13k+ stars,
  commits same-day), ships an OpenAI-Realtime-compatible WebSocket under
  `src/speech_to_speech/api/openai_realtime/`
- `pipeline/cancel_scope.py` — real generation-tagged cancellation: LLM and TTS
  threads check `is_stale(gen)` per token; audio chunks carry `cancel_generation`;
  on speech-start the server emits `response.done{status:cancelled}` and drains queues
- `VAD/smart_turn.py` turn endpointing; STT handlers incl. `parakeet_tdt`,
  `qwen3_asr`, `smart_progressive_streaming`; TTS handlers incl. `kokoro`,
  `qwen3_tts`, `pocket_tts`; LLM backend `responses_api_language_model.py`
  (generic — this is how llama.cpp is reached; there is no llama.cpp-specific backend)
- `LLM/compaction_prompt.py` exists upstream (relevant to ADR 0007 / Phase 5)

Alternative rejected: adopting `pollen-robotics/reachy_mini_conversation_app` — it was
gutted in v0.9/v0.10 (FastRTC removed PR #431, realtime backends deleted PR #444,
local VLM deleted PR #430), requires reachy-mini >= 1.9.0, its MovementManager owns a
100 Hz loop with no injection seam so ours cannot coexist, and its default profile
phones home via remote MCP tools. Also rejected: writing our own realtime service
(we'd rebuild the exact complexity we're trying to delete).

## Decision

Run HF speech-to-speech as a **separately installed, commit-pinned local service**
(its own venv, never vendored, no fork). This repo stays the robot shell: audio device
ownership, attention, movement, tools, identity, dashboard, journal, and a thin
Realtime protocol client. Transport is WebSocket on localhost; no WebRTC unless a
requirement appears.

## Consequences

- We inherit upstream protocol quirks; pin revisions and re-verify at upgrade time.
  Known quirks (verified by the 2026-07 spike against upstream):
  - no `session.updated` echo is ever sent — a client awaiting it hangs
  - `.done` events only for assistant transcript and tool arguments (no deltas)
  - `tool_choice` accepts strings only
  - no server-side AEC — we depend entirely on the XVF3800 (ADR 0005)
- "Streaming Parakeet" is progressive re-transcription of a growing window every
  500 ms — a real CPU multiplier on Skylake; must be measured, not assumed (ADR 0006).
- Do not copy the HF Reachy blog's llama.cpp flags (`-c 65536 --swa-full` does not fit
  8 GB); start at 8-16k context with quantized KV.
- The upstream demo client and `examples/realtime_web_search_tool.py` are the
  reference points when the docs lie; read their source.
