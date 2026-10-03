# ADR 0006: CPU and latency budgets — measured numbers this machine must honor

- Status: accepted
- Date: 2026-10-02
- Source: measurements preserved from beads robot-baa/b4w/q64/gsg and commit 28e0a33

## The machine is 4 cores. Everything below is measured on it, not assumed.

- **Thread starvation was the real "TTS is slow" bug.** llama-server with no
  `--threads` defaults to nproc=8 on a 4-core box and spin-waits, starving everything.
  Kokoro in isolation: 0.79 s for 31 chars; in production: 2.10 s. Forced contention:
  4 busy cores -> 6.5 s, 8 busy cores -> 11.3 s for the identical utterance. Any new
  service must have an explicit thread budget and it must be summed across all
  services, not per service.
- **Process-global thread env is a landmine.** `face_tracker.py` once set
  `OMP_NUM_THREADS=2` inside `__init__`, called after TTS construction — whether it
  reached torch was load-order dependent. Global thread config goes in `__main__`
  before imports; per-model caps go in the model's own session options.
- **GPU TTS frees the CPU**: Kokoro on-GPU (torch+cu126) is ~20x vs CPU
  (RTF 0.044 vs 0.79-2.10 s) and hands 4 cores back — see ADR 0004 for the wheel trap.
- **Turn-endpointing policy, not models, caused sluggishness.** Silero costs 0.127 ms
  per 32 ms chunk; the fixed 800 ms silence timer was 100% of the latency. Options
  measured: smart-turn-v3.2 ONNX is 8.68 MB, 29.4 ms @ 4 threads (run it at 2 to not
  fight TTS); threshold 0.5 is WRONG (LiveKit ships English-calibrated 0.36). BUT:
  HF s2s already implements speculative turns (soft-end at 64 ms, generate,
  discard/reopen within `speculative_reopen_ms`; hysteresis 384 ms start / 192 ms
  continue) — measure that FIRST, it may remove the need for a second model.
- **First-chunk latency tricks, measured:** Kokoro emits 243 ms leading + 510 ms
  trailing silence (trim the lead: free 243 ms per sentence). Clause-splitting the
  FIRST chunk only: TTFA 874 -> 605 ms. Synthesizing sentence N+1 while N plays: at
  RTF 0.33, reply length becomes irrelevant to perceived latency.
- **STT**: Parakeet TDT v3 -> v2 improves English WER (6.05 vs 6.32). Sub-second
  utterances are Parakeet's structural weakness (8x encoder subsampling — 'stop'
  became 'The Stob'); pad short segments with 200-300 ms silence.
- **Prompt caching dominates context architecture.** The stable system+tools prefix
  is expensive to prefill cold and cheap when cacheable; naive
  `messages[-20:]`-style trimming can cost multi-second re-prefills. Compaction must
  be infrequent and high-water-mark driven (measured, not message-count), must
  preserve the stable prefix byte-for-byte where possible, and must be benchmarked
  on-hardware before/after (see EPIC "llama.cpp and context budget").
- **Generation baseline**: Qwen 3.5 4B UD-Q4_K_XL, low-40s tok/s on this card with
  the Pascal-correct build (commits 9f6e673, 65e043d). MTP spec-decoding could give
  ~1.7x — see ADR 0004 for its caveats.

## Budget rule

Phase 1 acceptance requires measured p50/p95 of: speech-end -> first audible audio,
interruption -> silence, and CPU headroom with all subscribed consumers running.
No model joins the machine without a line in docs/pins.yaml with measured cost.
