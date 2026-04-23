#!/usr/bin/env bash
set -euo pipefail

# Defaults — override with env vars or edit below
MODEL_PATH="${LLAMA_MODEL_PATH:-models/gguf/qwen3.5-4b-q4_k_m.gguf}"
MMPROJ_PATH="${LLAMA_MMPROJ_PATH:-models/gguf/mmproj-BF16.gguf}"
PORT="${LLAMA_PORT:-8080}"
HOST="${LLAMA_HOST:-0.0.0.0}"
CTX="${LLAMA_CTX:-4096}"
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
    -ngl "$GPU_LAYERS" \
    --cache-type-k q8_0 --cache-type-v q4_0 \
    --host "$HOST" \
    --port "$PORT"
