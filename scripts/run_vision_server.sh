#!/usr/bin/env bash
set -euo pipefail

# Small VLM for the camera tools (describe_scene) beside a text-only chat LLM.
# The shell reaches it via VISION_BASE_URL / VISION_MODEL (shell/robot_tools.py).
#
# CPU on purpose: the GPU holds the chat LLM + Qwen3-TTS (ADR 0004 budget), and
# a camera question is rare. Measured 2026-10-10 on the i7-6700K, 4 threads,
# one 448 px frame, beside the live stack (time per describe, cold):
#   LFM2-VL-450M Q8_0    1.9 s
#   Qwen3.5-0.8B Q4_K_M  2.5 s   <- default (same family the old in-LLM vision was)
#   LFM2-VL-1.6B Q4_0    5.3 s
# Idle cost is RAM only (~1 GB); the threads are busy only while describing.
MODEL_PATH="${VISION_MODEL_PATH:-models/gguf/vlm/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf}"
MMPROJ_PATH="${VISION_MMPROJ_PATH:-models/gguf/vlm/qwen3.5-0.8b/mmproj-F16.gguf}"
PORT="${VISION_PORT:-8082}"
THREADS="${VISION_THREADS:-4}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_DIR:-$HOME/dev/llama.cpp}/build-cuda/bin/llama-server}"

# CPU only: hide the GPU so this process does not even take a CUDA context (~130 MB).
export CUDA_VISIBLE_DEVICES=""

exec "$LLAMA_SERVER_BIN" \
    --jinja \
    --model "$MODEL_PATH" \
    --mmproj "$MMPROJ_PATH" \
    --n-gpu-layers 0 \
    --no-mmproj-offload \
    --threads "$THREADS" \
    --ctx-size 4096 \
    --parallel 1 \
    --host 127.0.0.1 \
    --port "$PORT"
