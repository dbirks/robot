#!/usr/bin/env bash
set -euo pipefail

# Defaults — override with env vars or edit below.
# Long flags throughout: this file is read far more often than it is typed.

# UD-Q4_K_XL rather than plain Q4_K_M: better KL-divergence (0.410 vs 0.548)
# for +0.68GB. From unsloth/Qwen3.5-4B-MTP-GGUF, which bakes the MTP drafter
# heads into the GGUF itself — see the --spec-type note below for why we don't
# currently use them.
MODEL_PATH="${LLAMA_MODEL_PATH:-models/gguf/mtp/Qwen3.5-4B-UD-Q4_K_XL.gguf}"

# Vision projector for the LLM's image input (describe_scene / take_snapshot).
# F16, not the BF16 file we used to ship: the GTX 1070 (Pascal) has no BF16 in
# hardware and had to emulate it. F16 is native. Leave the file absent to run
# text-only.
MMPROJ_PATH="${LLAMA_MMPROJ_PATH:-models/gguf/mtp/mmproj-F16.gguf}"

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
    --threads "$THREADS" \
    --flash-attn "$FLASH_ATTN" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --host "$HOST" \
    --port "$PORT"
