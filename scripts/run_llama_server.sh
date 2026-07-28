#!/usr/bin/env bash
set -euo pipefail

# Defaults — override with env vars or edit below
MODEL_PATH="${LLAMA_MODEL_PATH:-models/gguf/qwen3.5-4b-q4_k_m.gguf}"
MMPROJ_PATH="${LLAMA_MMPROJ_PATH:-models/gguf/mmproj-BF16.gguf}"
PORT="${LLAMA_PORT:-8080}"
# Loopback by default — the API key is "not-needed", so don't expose the model
# (or the robot tools behind it) to the LAN unless you opt in explicitly.
HOST="${LLAMA_HOST:-127.0.0.1}"
# -c is the TOTAL context, split across the -np slots: 32768/2 = 16K per slot,
# which is ample for a voice turn and leaves VRAM headroom on the 8GB 1070.
CTX="${LLAMA_CTX:-32768}"
# Two slots so an interruption isn't queued behind the in-flight request.
PARALLEL="${LLAMA_PARALLEL:-2}"
GPU_LAYERS="${LLAMA_GPU_LAYERS:-99}"

MMPROJ_ARGS=()
if [ -f "$MMPROJ_PATH" ]; then
    MMPROJ_ARGS=(--mmproj "$MMPROJ_PATH")
fi

exec llama-server \
    --jinja -fa on \
    -m "$MODEL_PATH" \
    "${MMPROJ_ARGS[@]}" \
    -c "$CTX" \
    -np "$PARALLEL" \
    -ngl "$GPU_LAYERS" \
    --cache-type-k q8_0 --cache-type-v q4_0 \
    --host "$HOST" \
    --port "$PORT"
