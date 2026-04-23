# Reachy Local Agent

Local voice-driven agent for [Reachy Mini](https://www.reachy-mini.org/). Runs entirely on-device — no cloud APIs. Uses faster-whisper for STT, llama.cpp for the LLM, and Piper for TTS.

## Architecture

```
Microphone
    |
    v
[silero-vad]  (CPU, always listening)
    |  speech detected
    v
[faster-whisper small.en]  (GPU, ~1.5 GB VRAM)
    |  transcribed text
    v
[Agent Client]  -->  [llama.cpp / Qwen 4B]  (GPU, ~3 GB VRAM)
    |                      |
    |                tool calls
    |                      v
    |              [Robot Tools]  -->  [Reachy Mini SDK]
    |                      |
    |              tool results
    |<---------------------'
    |  assistant text
    v
[Piper TTS]  (CPU)
    |  audio
    v
Speaker
```

All inference stays local on a single machine. Target hardware: older i7 + NVIDIA GTX 1070 (8 GB VRAM).

## Hardware Requirements

- NVIDIA GPU with 6+ GB VRAM (GTX 1070 or better)
- Microphone
- Speaker (or the Reachy Mini's built-in speaker)
- Reachy Mini (Lite or Wireless) — or run in simulator mode

## Prerequisites

Install these on the host machine (Arch Linux):

```bash
# NVIDIA drivers + CUDA
sudo pacman -S nvidia nvidia-utils cudnn
# CUDA toolkit — see "GPU Compatibility" section below for version choice
sudo pacman -S cuda

# Audio server + libraries
sudo pacman -S pipewire pipewire-pulse pipewire-alsa wireplumber portaudio
systemctl --user enable --now pipewire pipewire-pulse wireplumber

# GStreamer (for Reachy camera)
sudo pacman -S gstreamer gst-plugins-base gst-plugins-good

# llama.cpp (build from source)
# See: https://github.com/ggml-org/llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
CUDACXX=/opt/cuda/bin/nvcc cmake -B build -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/opt/cuda
cmake --build build --config Release -j$(nproc)
sudo cp build/bin/llama-server /usr/local/bin/

# uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Reachy Mini USB permissions (Lite version)

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout $USER
```

## Setup

### 1. Download models

```bash
mkdir -p models/gguf models/piper

# LLM — Qwen 3.5 4B Q4_K_M (recommended for GTX 1070)
curl -L -o models/gguf/qwen3.5-4b-q4_k_m.gguf \
  "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"

# TTS voice
wget -O models/piper/en_GB-northern_english_male-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx?download=true"
wget -O models/piper/en_GB-northern_english_male-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx.json?download=true"

# STT model downloads automatically on first run (faster-whisper handles this)
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — update LLAMA_MODEL_PATH in scripts/run_llama_server.sh
# if your GGUF filename differs from the default
```

### 3. Install Python dependencies

```bash
uv sync
```

## Running

You need three terminals (or use a process manager):

```bash
# Terminal 1: Reachy Mini daemon (or simulator)
reachy-mini-daemon          # real robot
reachy-mini-daemon --sim    # simulator

# Terminal 2: LLM server
# Edit LLAMA_MODEL_PATH if needed
LLAMA_MODEL_PATH=models/gguf/your-model.gguf ./scripts/run_llama_server.sh

# Terminal 3: Voice agent
./scripts/run_app.sh
```

The agent will start listening. Speak and it will transcribe, reason, execute tools, and respond via TTS.

## Project Structure

```
app/
    __init__.py
    main.py           # Entry point — initializes services and starts loop
    config.py          # Settings from .env
    orchestrator.py    # Main conversation loop
    audio_io.py        # VAD-gated microphone capture
    stt_service.py     # faster-whisper transcription
    tts_service.py     # Piper speech synthesis
    agent_client.py    # LLM client with tool calling
    robot_tools.py     # Tool schemas and handlers for Reachy
    robot_state.py     # Reachy Mini connection management
    playback.py        # Audio output
scripts/
    run_llama_server.sh
    run_app.sh
models/                # gitignored — download models here
    gguf/
    piper/
logs/                  # gitignored — runtime logs and snapshots
```

## Available Tools

The agent can call these tools during conversation:

| Tool | Description |
|------|-------------|
| `look_left` | Turn head left |
| `look_right` | Turn head right |
| `look_center` | Return head to center |
| `nod` | Nod yes |
| `shake_head` | Shake no |
| `take_snapshot` | Capture camera image |
| `get_robot_status` | Check connection status |
| `get_time` | Get current time |

### Adding tools

Add the OpenAI function schema to `TOOLS` in `robot_tools.py` and a handler function in `make_handlers()`. Handlers must return a JSON-serializable dict and never raise exceptions.

## GPU Compatibility

**CUDA 13 dropped support for Pascal GPUs** (GTX 1070 and older, compute capability < 7.5). If you have a Pascal card, you need CUDA 12.x:

```bash
# For Pascal GPUs (GTX 1070, etc.) — install CUDA 12.9 from AUR
yay -S cuda-12.9   # replaces the cuda package
```

The `ctranslate2` library (used by faster-whisper) also needs a matching CUDA version. If you see `libcublas.so.12 not found` errors, your CUDA toolkit version doesn't match. Use `WHISPER_DEVICE=cpu` with `WHISPER_COMPUTE_TYPE=int8` as a reliable fallback — it's fast enough for small.en.

## VRAM Budget (GTX 1070, 8 GB)

| Component | VRAM |
|-----------|------|
| faster-whisper small.en (int8, CUDA) | ~1 GB |
| Qwen 3.5 4B Q4_K_M via llama.cpp | ~3 GB |
| CUDA overhead | ~0.5 GB |
| **Total** | **~4.5 GB** |

To free more VRAM, set `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE=int8` in `.env` to run STT on CPU instead (~0 VRAM, still fast).

## llama.cpp Notes

The `--jinja` flag in `run_llama_server.sh` is **required** for tool calling to work. Without it, the server silently ignores the `tools` parameter.

Context is set to 4096 by default (`LLAMA_CTX`). This is conservative but keeps memory low. Increase if the model needs more context for complex conversations.

### Qwen 3.5 thinking mode

Qwen 3.5 enables "thinking" by default — the model spends tokens on internal reasoning before responding. This is wasteful for a voice agent (adds latency, consumes the token budget). The agent client disables it via `chat_template_kwargs: {"enable_thinking": false}`, passed through the OpenAI client's `extra_body` parameter. If you swap to a non-thinking model, this parameter is harmlessly ignored.

## Agent Framework

The current agent layer (`agent_client.py`) is a thin OpenAI-compatible client that handles tool-calling loops directly against the llama.cpp server. It uses the same tool schema format as [Hermes](https://github.com/NousResearch/hermes-agent), so swapping in Hermes later requires minimal changes — point Hermes at the same llama.cpp endpoint and register the same tool handlers.

## Milestone Roadmap

1. **Core loop** — STT + LLM + TTS working end-to-end locally
2. **Robot tools** — agent calls Reachy movement tools from conversation
3. **Session memory** — conversation context persists across turns (Hermes integration)
4. **Camera** — `take_snapshot` tool, optional vision pipeline
5. **Attention model** — wake word or LLM-based filtering for "am I being spoken to?"
6. **Self-improvement** — guarded tool/skill editing with human approval
