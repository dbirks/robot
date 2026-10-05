# 2026-10-04 — LLM bakeoff: Qwen3.5-4B (current) vs Gemma 4 E4B

EPIC "LLM bakeoff" protocol, first pass. `toolbench.py`: Reachy's real 19-tool
schema + live persona (Obama-style prompt), 20 fixed prompts (13 should call
a specific tool, 7 should not), temperature 0.3, thinking off, llama.cpp
`91f8c9c5` on the GTX 1070. Gemma ran alone on :8090 (Qwen + TTS stopped).

| | Qwen3.5-4B UD-Q4_K_XL | Gemma 4 E4B Q4_0 (ggml-org) |
|---|---|---|
| tool test, 3 runs | 16 / 17 / 16 | 18 / 16 / 16 (+17 with PLE on CPU) |
| mean reply latency | 1.2-1.8 s | 0.45-0.54 s |
| decode | 29-30 tok/s | 43-44 tok/s |
| prefill (cold) | ~150 tok/s (985 tok = 6.6 s) | ~870 tok/s (1216 tok = 1.4 s) |
| VRAM, 32K ctx / 2 slots | ~4.4 GB (incl. 0.67 GB mmproj) | ~3.5 GB (no mmproj) |
| weights | 3.0 GB | 4.6 GB |

Failure modes differ:
- Qwen: denies its own body ("I can't physically turn my head") instead of
  calling look_right; describes the scene/faces without calling the camera
  tools (hallucination); once called go_to_sleep on "thanks, that's all".
- Gemma: sometimes emits the call as plain TEXT (`go_to_sleep()`,
  `look_center{}`) instead of a tool call - TTS would read it aloud; skips
  `remember`; hallucinates the scene. Same via /v1/responses (what s2s uses).

`-ot per_layer=CPU` (keep per-layer embeddings on CPU) changed nothing
measurable; llama.cpp already keeps VRAM low for E4B.

Verdict so far: accuracy is a tie (~16.5/20); Gemma is 2-3x faster end to end
and ~6x faster on cold prefill, with less VRAM. Blocker before switching: the
text-leaked tool calls. Next: newer llama.cpp Gemma 4 tool parser, or the
template's tool-call tokens, then re-run; KV q8_0/q8_0 per run_llama_server.sh
note (non-hybrid model) was used here.

## Round 2 — IBM Granite (official ibm-granite GGUFs), same harness

| model | tool test x2 | mean latency | decode | cold prefill | VRAM (32K ctx, 2 slots, LLM alone) |
|---|---|---|---|---|---|
| granite-4.1-3b Q8_0 (3.6 GB, no thinking mode) | 19 / 19 | 0.47-0.61 s | 44 tok/s | ~1040 tok/s | ~5.0 GB |
| granite-4.2-3b Q8_0 (thinking off) | 17 / 19 | 0.82-0.86 s | 43 tok/s | ~1040 tok/s | ~5.0 GB |
| granite-4.2-8b Q4_K_M | 19 / 19 | 1.17-1.41 s | 27 tok/s | ~440 tok/s | 7.9 GB (does not fit with TTS) |
| granite-4.0-micro Q8_0 | 15 / 15 | 0.48-0.54 s | 43 tok/s | ~1035 tok/s | ~5.0 GB |

- 4.1-3b: only miss was "Speak up" (answered in text / once emitted the bare word `louder`).
- 4.2-3b: refuses go_to_sleep on a plain "Go to sleep." (over-reads the tool's EXACTLY rule); missed nod/peekaboo once.
- 4.0-micro: over-calls tools on small talk (get_robot_status for "how are you", web_search for 2+2, go_to_sleep for "thanks, that's all").
- 4.2-8b: accurate but slow and does not fit beside the Obama TTS.

Leader: granite-4.1-3b. Before switching: VRAM with the live TTS (~5.0 GB here at 32K vs Qwen ~4.4 GB; Obama 1.7B TTS needs ~3.3 GB) - try 16K ctx or Q6_K, and listen-test persona quality.
