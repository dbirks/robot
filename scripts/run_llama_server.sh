#!/usr/bin/env bash
set -euo pipefail

# Defaults — override with env vars or edit below
MODEL_PATH="${LLAMA_MODEL_PATH:-models/gguf/qwen3.5-4b-q4_k_m.gguf}"
PORT="${LLAMA_PORT:-8080}"
HOST="${LLAMA_HOST:-0.0.0.0}"
CTX="${LLAMA_CTX:-4096}"
GPU_LAYERS="${LLAMA_GPU_LAYERS:-99}"

exec llama-server \
    --jinja -fa on \
    -m "$MODEL_PATH" \
    -c "$CTX" \
    -ngl "$GPU_LAYERS" \
    --host "$HOST" \
    --port "$PORT"
