# Reachy Local Agent

<!-- status-banner -->
> **What is actually running today (2026-10-12).** The live stack is the realtime
> shell in [`shell/`](shell/) - pinned `huggingface/speech-to-speech` for VAD/STT/TTS,
> Qwen 3.5 4B through llama.cpp behind [`shell/llm_proxy.py`](shell/llm_proxy.py), and
> `python -m shell` owning the microphone, the speaker, attention and tools. Rooms are
> configuration: `REACHY_ENV` picks `envs/home.yaml` or `envs/hackday.yaml`
> ([design](docs/environments.md)). Tests: `uv run pytest tests/`.
> The synchronous turn-based loop in [`app/`](app/) is **legacy** and is being deleted at
> the Phase 2 cutover ([issue #22](https://github.com/dbirks/robot/issues/22)); the rest
> of this file still describes parts of it, so read `AGENTS.md` and
> [`docs/epic-realtime-rebuild.md`](docs/epic-realtime-rebuild.md) for the current truth.
> Dashboard: `http://localhost:3001`. Before you travel: `scripts/preflight_demo.sh`.

> **Under rebuild.** This repo is mid-way through the realtime rebuild:
> the canonical spec is [`docs/epic-realtime-rebuild.md`](docs/epic-realtime-rebuild.md),
> durable decisions live in [`docs/adr/`](docs/adr/), and the new stack is the
> [`shell/`](shell/) package (`python -m shell`) described in
> [`deploy/README.md`](deploy/README.md). The description below documents the
> legacy turn-based loop (`app/`), which stays functional until the Phase 2
> cutover (issue #22) once the new path passes its hardware gate.

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
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="uucp"
SUBSYSTEM=="usb", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="uucp"' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG uucp $USER
# Log out and back in for the group change to take effect
```

## Setup

### 1. Download models

```bash
mkdir -p models/gguf models/piper

# LLM — what the live unit actually loads (docs/pins.yaml models.llm, ADR 0004):
#   models/gguf/mtp/Qwen3.5-4B-UD-Q4_K_XL.gguf  (unsloth/Qwen3.5-4B-MTP-GGUF, UD-Q4_K_XL)
#   models/gguf/mtp/mmproj-F16.gguf             (F16 projector — never BF16 on Pascal)
# Both are already on this box; override the paths with LLAMA_MODEL_PATH /
# LLAMA_MMPROJ_PATH (scripts/run_llama_server.sh:16,22).
#
# The two downloads below are the SUPERSEDED pair this file used to document:
# qwen3.5-4b-q4_k_m.gguf and mmproj-BF16.gguf are still on disk (Apr 2026) and are not
# what runs. The BF16 projector in particular is the file ADR 0004 says never to use —
# GP104 has no BF16 in hardware and emulates it.
curl -L -o models/gguf/qwen3.5-4b-q4_k_m.gguf \
  "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"

# Vision projector (enables image understanding)
curl -L -o models/gguf/mmproj-BF16.gguf \
  "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-BF16.gguf"

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

Every variable this project reads — shell, service, dashboard, llama.cpp, kiosk — is
tabulated with its reader and its per-room default in [`docs/knobs.md`](docs/knobs.md).
`.env.example` is a template, not a list of live knobs: some of its lines (e.g.
`OPENAI_BASE_URL=http://localhost:8080/v1`) point at llama.cpp directly and would bypass the
name-fixing proxy that `scripts/run_s2s.sh:27` defaults to (proxy on :8081, llama.cpp on :8080).

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

## At a booth

`REACHY_ENV=hackday` is the room that runs in front of strangers. Everything below is newer
than the prose elsewhere in this file; the whole knob list is [`docs/knobs.md`](docs/knobs.md).

- **Personas**: `personas/obama.yaml`, `personas/marvin.yaml`, `personas/tars.yaml`. The
  dashboard persona page picks a persona, never a room (`dashboard/server.py`); the room
  comes only from `REACHY_ENV`.
- **Voice style**: `voice.instruct` in the persona file reaches the service as
  `--qwen3_tts_instruct`, appended only when `S2S_QWEN3_INSTRUCT` is non-empty — upstream's
  default is `None` and `""` is a value, not the absence of one (`scripts/run_s2s.sh`
  `build_tts_args`). Print the resolved argv with no service, GPU, mic or network:
  `scripts/run_s2s.sh --print-tts-args`.
- **Greeter**: eye-contact hello gated in `shell/vision/greeter.py`; off in `home`, on in
  `hackday` (`REACHY_GREETER`). Not yet proven on a live night.
- **Sleep is one state, not one pose** (`shell/sleep.py`): lying down closes the mic, the
  attention lease and the tool gate, not just the head.
- **Direct intents** (`shell/intents.py`): bare wake word, sleep and "stop talking" are
  matched on the transcript and executed by the shell — the 4B model narrates them instead
  of acting.
- **Volume** (`shell/volume.py`): 200% ceiling, ~110% at boot; above 120% it is allowed but
  flagged `clipping_risk`. Enforcement is `set_volume` in `app/robot_tools.py` and the
  slider in `dashboard/env_panel.py`.
- **Per-tool timeouts** (`shell/tools.py`): `CAMERA_TOOLS` get 35 s, every other tool 12 s.
- **No clock in the prompt.** `shell/llm_proxy.py` only rewrites mis-heard names; the
  per-message `[date, time]` prefix is gone. `TIME_NOTE` (`shell/config.py`) tells the model
  it has no clock and routes every time or date question to `get_time`.
- **Speaker at boot**: `scripts/reachy_audio_init.sh` sets the Reachy card's hardware
  playback gain by card name, never card index, and exits 0 when no card is present.
- **One room per screen**: journal reads are filtered by `env` (`dashboard/journal_read.py`),
  so the booth display cannot open on the home conversation.

### Operator tooling

- [`docs/knobs.md`](docs/knobs.md) — every env var actually read, its reader, and what each
  room resolves. A name not listed there does nothing.
- `scripts/preflight_demo.sh` — read-only check before travel; `DEMO_ENV=hackday` to check a room.
- `scripts/booth_recovery.sh` — diagnose and restart the unit chain (`--dry-run`, `--only <unit>`, `--yes`).
- `scripts/check_env_safety.sh` — 13 checks that no home data can reach the booth.
- `scripts/tail_journal.py` — the last N minutes of the journal as a timeline, barge-ins included.
- `scripts/smoke_greeter.py` — watch the greeter decide over real frames: no audio, no motion, no journal.
- `scripts/bench_greeter.py` — turn a live greeter night into numbers, read back from the journal.

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

The legacy agent can call these tools during conversation. **The table is stale** — it lists 8
of the 19 schemas in `app/robot_tools.py:24` `TOOLS` (no `describe_scene`, no `peekaboo`, no
`play_emotion`, no memory tools), and the realtime shell advertises 21 through
`shell/robot_tools.py:build_handlers` + `shell/memories.py:MEMORY_TOOLS`, minus whatever
`REACHY_TOOLS_EXCLUDE` / `envs/<env>.yaml` drops (16 in `hackday`). Read the live list with
`REACHY_ENV=<room> python -m shell --print-env` or the dashboard's environment panel, not from
this table:

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

The `ctranslate2` library (used by faster-whisper) also needs a matching CUDA version. If you see `libcublas.so.12 not found` errors, your CUDA toolkit version doesn't match.

> **No `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` knob exists.** Nothing in `app/`, `shell/`
> or `scripts/` ever read those two names — this whole section describes a faster-whisper STT
> that this repo does not run. The legacy loop transcribes with **Parakeet TDT 0.6B v3 on CPU
> via `onnx_asr`**, hard-wired at `app/stt_service.py:12-21`; the realtime stack's STT device is
> `S2S_PARAKEET_DEVICE` (`scripts/run_s2s.sh:124`). Do not set the whisper names expecting a CPU
> fallback — there is none to fall back to. Every knob that is real is in
> [`docs/knobs.md`](docs/knobs.md).

## VRAM Budget (GTX 1070, 8 GB)

> **Superseded by [`docs/pins.yaml`](docs/pins.yaml).** The live box is 4.47 GB for the LLM
> with the F16 projector and 7.84 GB total beside the Qwen3-TTS talker, with Parakeet on CPU
> (`models.llm` measured). The table below is the legacy loop's estimate and is kept only
> because it is the history of how the card got tight.

| Component | VRAM |
|-----------|------|
| faster-whisper small.en (int8, CUDA) | ~1 GB |
| Qwen 3.5 4B Q4_K_M via llama.cpp | ~3 GB |
| CUDA overhead | ~0.5 GB |
| **Total** | **~4.5 GB** |

To free more VRAM, run STT on CPU — which is where it already runs (`app/stt_service.py`, and
`S2S_PARAKEET_DEVICE=cpu` for the realtime stack). The `WHISPER_*` variables named in older
revisions of this file are not read by any code (see "GPU Compatibility" above).

## llama.cpp Notes

The `--jinja` flag in `run_llama_server.sh` is **required** for tool calling to work. Without it, the server silently ignores the `tools` parameter.

Context is set to **24576** by default (`LLAMA_CTX`, `scripts/run_llama_server.sh:36`: 2 slots
x 12K, KV q8_0/q8_0 — see `docs/pins.yaml` `models.llm`). The 4096 this file quoted for months
is not what the box runs.

### Qwen 3.5 thinking mode

Qwen 3.5 enables "thinking" by default — the model spends tokens on internal reasoning before responding. This is wasteful for a voice agent (adds latency, consumes the token budget). The agent client disables it via `chat_template_kwargs: {"enable_thinking": false}`, passed through the OpenAI client's `extra_body` parameter. If you swap to a non-thinking model, this parameter is harmlessly ignored.

## Agent Framework

The current agent layer (`agent_client.py`) is a thin OpenAI-compatible client that handles tool-calling loops directly against the llama.cpp server. It uses the same tool schema format as [Hermes](https://github.com/NousResearch/hermes-agent), so swapping in Hermes later requires minimal changes — point Hermes at the same llama.cpp endpoint and register the same tool handlers.

## Machine-Specific Setup

- [GTX 1070 + Arch Linux](docs/setup-gtx1070-arch.md) — CUDA 12 requirement, PipeWire, permissions, VRAM budget

## Milestone Roadmap

1. **Core loop** — STT + LLM + TTS working end-to-end locally
2. **Robot tools** — agent calls Reachy movement tools from conversation
3. **Session memory** — conversation context persists across turns (Hermes integration)
4. **Camera** — `take_snapshot` tool, optional vision pipeline
5. **Attention model** — wake word or LLM-based filtering for "am I being spoken to?"
6. **Self-improvement** — guarded tool/skill editing with human approval
