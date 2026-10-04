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
# Entry point at v1.0.0 is the `speech-to-speech serve` console script
# (`--mode realtime` is deprecated; there is no s2s_service module).
# --responses_api_base_url is REQUIRED: upstream otherwise defaults to a hosted
# OpenAI model. Upstream's default reasoning effort "none" is honored by
# llama.cpp's /v1/responses and keeps Qwen3.5 from thinking (verified).
# --max_speech_ms: upstream default is infinite. With a TV talking, Smart Turn
# never sees a complete turn, so one turn grew to 74 s and never got answered
# (2026-10-03). 20 s forces a split; a real request is far shorter.
# Parakeet: v3 (upstream nano-parakeet hardcodes the v3 vocab; v2 fails to
# load with a decoder.embed size mismatch), fp32 on CPU - fp16 runs at 1/64 rate on GP104
# and fp32-on-GPU does not fit beside llama.cpp + Kokoro (ADR 0004).
exec "$DIR/repo/.venv/bin/speech-to-speech" serve \
  --host 127.0.0.1 \
  --port "${S2S_PORT:-8765}" \
  --stt "${S2S_STT:-parakeet-tdt}" \
  --parakeet_tdt_model_name "${S2S_PARAKEET_MODEL:-nvidia/parakeet-tdt-0.6b-v3}" \
  --parakeet_tdt_device "${S2S_PARAKEET_DEVICE:-cpu}" \
  --parakeet_tdt_compute_type float32 \
  --enable_live_transcription "${S2S_LIVE_TRANSCRIPTION:-False}" \
  --max_speech_ms "${S2S_MAX_SPEECH_MS:-20000}" \
  --tts "${S2S_TTS:-kokoro}" \
  --kokoro_device "${S2S_KOKORO_DEVICE:-cuda}" \
  --kokoro_voice "${KOKORO_VOICE:-bm_daniel}" \
  --llm_backend responses-api \
  --responses_api_base_url "$OPENAI_BASE_URL" \
  --responses_api_api_key "$OPENAI_API_KEY" \
  --responses_api_stream \
  --model_name "${LLM_MODEL:-qwen3.5-4b}"
# Phase 1 gate TODO (experiments/2026-07-28-realtime-spike): measure CPU of
# --enable_live_transcription on the 4-core box before turning it on (it
# re-transcribes the growing window every 500 ms); add --smart_turn only if
# the upstream speculative-turn behavior proves weak.
