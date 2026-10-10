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
# Via shell/llm_proxy.py (reachy-llm-proxy.service), which fixes Parakeet's
# spellings of "Reachy" in user text before llama.cpp (:8080) sees them.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8081/v1}"
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
# --max_speech_ms: upstream default is infinite. It only bounds ONE Silero
# segment (and discards it, despite the help text), not a turn - see below.
# Endpointing (2026-10-10, from reading VAD/vad_handler.py at v1.0.0): a turn
# ends only when no new speech starts within the reopen grace after a pause;
# any speech inside it REOPENS the same turn and Parakeet re-transcribes the
# whole growing buffer. The grace is max(speculative_reopen_ms,
# unanswered_reopen_ms=7000 default, smart_turn_max_wait_ms=2000), so a TV
# kept one turn open for ~2 min and the answer came a minute later
# (2026-10-09). 1200 ms keeps a mid-sentence breath inside the turn while
# background talk can't chain onto it. --thresh 0.7 (default 0.6) makes
# distant/quiet speech less likely to start or extend a turn. The shell also
# force-ends any turn longer than REACHY_MAX_TURN_S by feeding silence.
# Parakeet: v3 (upstream nano-parakeet hardcodes the v3 vocab; v2 fails to
# load with a decoder.embed size mismatch), fp32 on CPU - fp16 runs at 1/64 rate on GP104
# and fp32-on-GPU does not fit beside llama.cpp + Kokoro (ADR 0004).
TTS="${S2S_TTS:-kokoro}"
case "$TTS" in
  kokoro)
    VOICE="${KOKORO_VOICE:-bm_daniel}"
    # Kokoro voices are <lang><gender>_<name>; lang 'a' American, 'b' British.
    TTS_ARGS=(--kokoro_device "${S2S_KOKORO_DEVICE:-cuda}" --kokoro_voice "$VOICE"
              --kokoro_lang_code "${VOICE:0:1}") ;;
  qwen3)
    # Pascal: the PyPI qwentts wheel has no sm_61 kernels; use our pinned
    # build (scripts/build_qwentts.sh). Q8_0, never the BF16 default (no BF16
    # on Pascal). torch backend in fp32 does not fit beside llama.cpp.
    export QWENTTS_CPP_LIBRARY="${QWENTTS_CPP_LIBRARY:-$HOME/dev/qwentts.cpp/build-cuda61/libqwen.so}"
    [ -f "$QWENTTS_CPP_LIBRARY" ] || { echo "run scripts/build_qwentts.sh first" >&2; exit 1; }
    TTS_ARGS=(--qwen3_tts_backend ggml --qwen3_tts_ggml_quantization "${S2S_QWEN3_QUANT:-Q8_0}"
              --qwen3_tts_model_name "${S2S_QWEN3_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
              --qwen3_tts_speaker "${S2S_QWEN3_SPEAKER:-Ryan}" --qwen3_tts_language English)
    # Local fine-tunes converted with qwentts.cpp convert.py + quantize (e.g.
    # the Obama 1.7B voice, speaker "laxmikant"): set both GGUF paths and
    # point S2S_QWEN3_MODEL at the matching upstream id (1.7B-CustomVoice).
    if [ -n "${S2S_QWEN3_TALKER_GGUF:-}" ]; then
      CODEC="${S2S_QWEN3_CODEC_GGUF:-$(ls "$HF_HOME"/hub/models--Serveurperso--Qwen3-TTS-GGUF/snapshots/*/qwen-tokenizer-12hz-Q8_0.gguf | head -1)}"
      TTS_ARGS+=(--qwen3_tts_gguf_talker_path "$S2S_QWEN3_TALKER_GGUF" --qwen3_tts_gguf_codec_path "$CODEC")
    fi ;;
  *) TTS_ARGS=() ;;
esac

exec "$DIR/repo/.venv/bin/speech-to-speech" serve \
  --host 127.0.0.1 \
  --port "${S2S_PORT:-8765}" \
  --stt "${S2S_STT:-parakeet-tdt}" \
  --parakeet_tdt_model_name "${S2S_PARAKEET_MODEL:-nvidia/parakeet-tdt-0.6b-v3}" \
  --parakeet_tdt_device "${S2S_PARAKEET_DEVICE:-cpu}" \
  --parakeet_tdt_compute_type float32 \
  --enable_live_transcription "${S2S_LIVE_TRANSCRIPTION:-False}" \
  --max_speech_ms "${S2S_MAX_SPEECH_MS:-20000}" \
  --thresh "${S2S_VAD_THRESH:-0.7}" \
  --unanswered_reopen_ms "${S2S_REOPEN_MS:-1200}" \
  --smart_turn_max_wait_ms "${S2S_REOPEN_MS:-1200}" \
  --tts "$TTS" "${TTS_ARGS[@]}" \
  --llm_backend responses-api \
  --responses_api_base_url "$OPENAI_BASE_URL" \
  --responses_api_api_key "$OPENAI_API_KEY" \
  --responses_api_stream \
  --model_name "${LLM_MODEL:-qwen3.5-4b}"
# Phase 1 gate TODO (experiments/2026-07-28-realtime-spike): measure CPU of
# --enable_live_transcription on the 4-core box before turning it on (it
# re-transcribes the growing window every 500 ms). Smart Turn is ON by
# default upstream; it only picks the grace (complete -> 800 ms, incomplete
# -> smart_turn_max_wait_ms) and delays STT 600 ms on "incomplete".
