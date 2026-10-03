# Realtime transport spike (beads robot-dz9)

- Date: 2026-07-28 (spike), 2026-10-02 (record written)
- Machine: i7-6700K (4C/8T), GTX 1070 8GB sm_61, Arch Linux
- Code: `spike/s2s_client.py` @ commit b1de612 (~310 LOC, asyncio + websockets + sounddevice)
- Server: huggingface/speech-to-speech realtime mode, `--stt parakeet-tdt --tts kokoro
  --llm_backend responses-api` against local llama-server

## Hypothesis

One persistent mic stream + one persistent speaker stream + server-side VAD/turn/
cancellation works on this hardware and removes the need for local barge-in VAD,
before we rewrite anything.

## Results (verified)

- Handshake, `session.update`, full server pipeline end-to-end: feeding known
  synthesized speech into the socket, VAD -> Parakeet -> llama.cpp -> Kokoro returned
  the exact transcript ("Hello there, what is your name?") and a spoken reply.
- Live mic capture: 7498 frames sent, **zero errors**, persistent streams never reopened.
- Barge-in design confirmed structurally: server emits `input_audio_buffer.speech_started`,
  client flushes the playback deque; server-side CancelScope guarantees no stale audio
  follows. No local VAD needed for barge-in.

## Protocol notes learned the hard way

- Needs a dummy `OPENAI_API_KEY` even when fully local.
- A malformed sub-object inside `session.update` is rejected wholesale as
  "Unknown or invalid event" — no field-level error. Keep the payload minimal;
  server default 16 kHz already matches the pipeline rate.
- **No `session.updated` echo is ever sent.** A client that awaits confirmation hangs.
- `.done` events only for assistant transcript and tool-argument deltas.
- `tool_choice` accepts strings only.
- `asyncio.wait` never raises for failed tasks: a crashed sender streamed silence
  for 5 minutes silently. Task death must be reported loudly (fixed in the spike).
- Do NOT copy the HF Reachy blog llama.cpp line: `-c 65536 --swa-full` will not fit
  8 GB alongside anything. 8-16k with quantized KV.

## NOT yet measured (Phase 1 gate — requires the robot machine + human in the room)

- p50/p95 speech-end -> first audio (harness exists in spike Metrics class)
- interruption -> audible silence
- 50 consecutive barge-ins without stale speech
- CPU load with Silero + Parakeet progressive re-transcription + Kokoro; whether
  `--enable_realtime_transcription` is affordable at all on 4 Skylake cores

## Conclusion

Transport: **proven, proceed to production** (Phase 1). Load numbers outstanding —
carry the spike's Metrics harness into the shell so Phase 1 acceptance measures
rather than re-implements.
