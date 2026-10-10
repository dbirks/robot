#!/usr/bin/env bash
set -euo pipefail

# Defaults — override with env vars or edit below.
# Long flags throughout: this file is read far more often than it is typed.

# Default since 2026-10-10: IBM Granite 4.1 3B Q6_K (ibm-granite/granite-4.1-3b-GGUF),
# text-only. 20/20 on the Reachy toolbench (Qwen3.5-4B: 16-17/20), ~1040 tok/s
# cold prefill, 3.9 GB at 32K. Camera questions go to a separate small VLM
# (scripts/run_vision_server.sh). Previous model, still a two-line switch:
#   LLAMA_MODEL_PATH=models/gguf/mtp/Qwen3.5-4B-UD-Q4_K_XL.gguf
#   LLAMA_MMPROJ_PATH=models/gguf/mtp/mmproj-F16.gguf
# (UD-Q4_K_XL from unsloth/Qwen3.5-4B-MTP-GGUF: better KL-divergence than
# Q4_K_M; MTP heads baked in, unused - see --spec-type below.)
MODEL_PATH="${LLAMA_MODEL_PATH:-models/gguf/granite/granite-4.1-3b-Q6_K.gguf}"

# Vision projector for the LLM's image input (describe_scene / take_snapshot).
# F16, not the BF16 file we used to ship: the GTX 1070 (Pascal) has no BF16 in
# hardware and had to emulate it. F16 is native. Leave the file absent to run
# text-only.
MMPROJ_PATH="${LLAMA_MMPROJ_PATH:-}"   # vision projector, only for a VLM chat model

PORT="${LLAMA_PORT:-8080}"

# Loopback by default — the API key is "not-needed", so don't expose the model
# (or the robot tools behind it) to the LAN unless you opt in explicitly.
HOST="${LLAMA_HOST:-127.0.0.1}"

# --ctx-size is the TOTAL context, split across the --parallel slots:
# 32768/2 = 16K per slot. Ample for a voice turn, and it keeps us under the
# ~22-24K per-slot threshold where Pascal + quantized KV + flash-attn is
# reported to crash (llama.cpp issue #22032, closed as not-planned).
CTX="${LLAMA_CTX:-32768}"

# Two slots so an interruption isn't queued behind the in-flight request.
#
# MTP speculative decoding requires --parallel 1, and we measured it as NOT
# worth that trade on this card. Benchmarked 2026-07-28 on the 1070, same
# model/binary, 3 runs each:
#     baseline          43.0 / 43.0 / 42.6 tok/s   (mean 42.9)
#     --spec-type mtp   47.1 / 49.1 / 43.9 tok/s   (mean 46.7)
# ~+9%, at 0.48-0.58 draft acceptance. The 1.5-1.9x figures in llama.cpp
# PR #22673 are Ampere-and-newer; Pascal has no tensor cores, so verifying the
# draft costs nearly as much as generating. Giving up the barge-in slot for 9%
# is a bad trade. (It does at least RUN — issue #25713's pre-Ampere MTP crash
# did not reproduce here.) Re-evaluate if the GPU ever changes.
PARALLEL="${LLAMA_PARALLEL:-2}"

GPU_LAYERS="${LLAMA_GPU_LAYERS:-99}"

# llama.cpp defaults --threads to nproc, which is 8 on this 4-core/8-thread
# i7-6700K. It then spin-waits on all 8, starving Kokoro and Parakeet, which
# share the same cores. Measured effect on Kokoro for one 31-char utterance:
# 1 thread 2.33s, 2 threads 1.31s, 4 threads 0.93s -- and under 8 busy cores it
# degraded to 11.3s. Cap at the physical core count.
THREADS="${LLAMA_THREADS:-4}"

# Flash attention on Pascal is genuinely ambiguous — no MMA instructions, so it
# falls back to the vec kernel. Measured as a large win on some models and a
# ~50% regression on others (issue #19020). Left on because quantized KV
# requires it, but worth A/B-ing per model.
FLASH_ATTN="${LLAMA_FLASH_ATTN:-on}"

# K and V must be the SAME type on Pascal. Mixed q8_0/q4_0 (used until
# 2026-10-10) has no matching flash-attention vec kernel here and prefill
# collapses (llama-bench pp1024, GTX 1070, build 91f8c9c5):
#   Qwen3.5-4B  q8_0/q8_0 956 t/s  vs q8_0/q4_0 245 t/s
#   Granite 3B  q8_0/q8_0 1037 t/s vs q8_0/q4_0 53 t/s   (f16/f16 1061)
# That was the 6-8 s "LLM" share of slow replies. Decode is unaffected.
CACHE_TYPE_K="${LLAMA_CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${LLAMA_CACHE_TYPE_V:-q8_0}"

# Prompt-prefix caching is ON by default (--cache-ram defaults to 8192 MiB) and
# it matters enormously here: the system prompt plus 19 tool schemas is ~1570
# tokens, which costs ~6s to prefill cold and ~250ms once cached.
#
# --cache-reuse additionally recovers part of the cache when the prompt diverges
# mid-way rather than only at the end -- which is what happens every time the
# agent trims conversation history. Measured on the trim case:
#     cache-reuse 0    cached 1069   prefill 5663ms
#     cache-reuse 256  cached 1444   prefill 3779ms   <-- chosen
#     cache-reuse 64   cached 1444   prefill 3972ms
#     cache-reuse 16   cached 1444   prefill 4209ms
# Smaller chunks recover no more and cost overhead, so 256 it is. Note this only
# softens the trim penalty; the real fix was hysteresis in agent_client.py so
# trims are rare.
CACHE_REUSE="${LLAMA_CACHE_REUSE:-256}"

# Run the binary from the pinned build (scripts/build_llama.sh), never a bare
# PATH lookup: /usr/local/bin/llama-server was a stale April install that
# resolved the July build's shared libs and aborted on startup
# (GGML_ASSERT(params.n_gpu_layers < 0)) from 2026-09-01 until this fix.
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_DIR:-$HOME/dev/llama.cpp}/build-cuda/bin/llama-server}"

MMPROJ_ARGS=()
if [ -f "$MMPROJ_PATH" ]; then
    MMPROJ_ARGS=(--mmproj "$MMPROJ_PATH")
fi

exec "$LLAMA_SERVER_BIN" \
    --jinja \
    --model "$MODEL_PATH" \
    "${MMPROJ_ARGS[@]}" \
    --ctx-size "$CTX" \
    --parallel "$PARALLEL" \
    --n-gpu-layers "$GPU_LAYERS" \
    --threads "$THREADS" \
    --flash-attn "$FLASH_ATTN" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --cache-reuse "$CACHE_REUSE" \
    --host "$HOST" \
    --port "$PORT"
