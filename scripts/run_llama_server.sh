#!/usr/bin/env bash
set -euo pipefail

# Defaults — override with env vars or edit below.
# Long flags throughout: this file is read far more often than it is typed.

MODEL_PATH="${LLAMA_MODEL_PATH:-models/gguf/qwen3.5-4b-q4_k_m.gguf}"

# Vision projector for the LLM's image input (describe_scene / take_snapshot).
# NOTE: the shipped mmproj-BF16.gguf is BF16, which the GTX 1070 (Pascal) does
# not support in hardware — it costs ~675MB of VRAM in a dtype the card has to
# emulate. Set LLAMA_MMPROJ_PATH to an F16 projector, or leave the file absent
# to run text-only. Tracked in robot-yqu.
MMPROJ_PATH="${LLAMA_MMPROJ_PATH:-models/gguf/mmproj-BF16.gguf}"

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
# NOTE: MTP speculative decoding requires --parallel 1. Choosing MTP means
# reworking barge-in as request-cancellation instead. See robot-gsg.
PARALLEL="${LLAMA_PARALLEL:-2}"

GPU_LAYERS="${LLAMA_GPU_LAYERS:-99}"

# Flash attention on Pascal is genuinely ambiguous — no MMA instructions, so it
# falls back to the vec kernel. Measured as a large win on some models and a
# ~50% regression on others (issue #19020). Left on because quantized KV
# requires it, but worth A/B-ing per model.
FLASH_ATTN="${LLAMA_FLASH_ATTN:-on}"

# K at q8_0, V at q4_0. Verified lossless on Qwen3.5 specifically: only 8 of 32
# layers use full attention, and the linear/gated-delta layers absorb the
# quantization noise (BLEU 1.000 vs f16, llama.cpp issue #21385). Do NOT assume
# this carries to a non-hybrid model like Gemma 4 — re-test at q8_0/q8_0 first.
CACHE_TYPE_K="${LLAMA_CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${LLAMA_CACHE_TYPE_V:-q4_0}"

MMPROJ_ARGS=()
if [ -f "$MMPROJ_PATH" ]; then
    MMPROJ_ARGS=(--mmproj "$MMPROJ_PATH")
fi

exec llama-server \
    --jinja \
    --model "$MODEL_PATH" \
    "${MMPROJ_ARGS[@]}" \
    --ctx-size "$CTX" \
    --parallel "$PARALLEL" \
    --n-gpu-layers "$GPU_LAYERS" \
    --flash-attn "$FLASH_ATTN" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --host "$HOST" \
    --port "$PORT"
