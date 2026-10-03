#!/usr/bin/env bash
# Launch the pinned s2s realtime service (ADR 0001). llama.cpp is reached
# generically via --llm_backend responses-api; there is no llama.cpp-specific
# backend. NEVER copy the HF blog's llama.cpp flags (-c 65536 --swa-full does
# not fit this 8GB Pascal card). Models are pinned in docs/pins.yaml.
set -euo pipefail

DIR="${S2S_DIR:-$HOME/.local/share/reachy/speech-to-speech}"
[ -x "$DIR/repo/.venv/bin/python" ] || { echo "run scripts/install_s2s.sh first" >&2; exit 1; }

# Fully-local still requires an API key string for the OpenAI client paths.
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8080/v1}"
# After assets are installed, run offline (EPIC locality requirement):
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_HOME="${HF_HOME:-$DIR/hf-cache}"

# Model bakeoffs (issues #24/#25) are flag swaps here - override, don't fork:
#   S2S_TTS=qwen3_tts scripts/run_s2s.sh   # candidate production voice (watch VRAM!)
#   S2S_STT=nemo_asr  scripts/run_s2s.sh   # streaming STT candidate
exec "$DIR/repo/.venv/bin/python" -m speech_to_speech.s2s_service \
  --mode realtime \
  --stt "${S2S_STT:-parakeet-tdt}" \
  --tts "${S2S_TTS:-kokoro}" \
  --llm_backend responses-api \
  --listenport "${S2S_PORT:-8765}"
# Phase 1 gate TODO (experiments/2026-07-28-realtime-spike): measure CPU with
# --enable_realtime_transcription on the 4-core box before enabling it; add
# --smart_turn only if the upstream speculative-turn behavior proves weak.
