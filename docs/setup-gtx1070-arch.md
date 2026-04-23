# Setup: GTX 1070 + Arch Linux (dave-droid)

Machine-specific notes for running the voice agent on an older i7 with an NVIDIA GTX 1070 (8 GB VRAM) on Arch Linux.

## System packages

```bash
# GPU drivers
sudo pacman -S nvidia nvidia-utils cudnn

# CUDA toolkit — GTX 1070 is Pascal (compute_61), dropped in CUDA 13.
# Install CUDA 12.9 from AUR instead of the official cuda package.
yay -S cuda-12.9    # provides /opt/cuda, replaces cuda

# Audio
sudo pacman -S pipewire pipewire-pulse pipewire-alsa wireplumber portaudio alsa-utils
systemctl --user enable --now pipewire pipewire-pulse wireplumber

# GStreamer (for Reachy camera — optional)
sudo pacman -S gstreamer gst-plugins-base gst-plugins-good

# Build tools (for llama.cpp)
sudo pacman -S base-devel cmake

# Python
# uv should already be installed — if not:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## CUDA 12 vs 13

CUDA 13 removed support for compute capabilities below 7.5 (Turing). The GTX 1070 is compute_61 (Pascal). The `cuda-12.9` AUR package installs alongside `gcc14` (also from AUR, built from source — takes ~30 min).

This also affects `ctranslate2` (used by faster-whisper): it ships linked against CUDA 12, so `libcublas.so.12` must be present. With `cuda-12.9` installed this works. Without it, set `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE=int8` in `.env`.

## Building llama-server with CUDA

```bash
cd ~/dev/llama.cpp
CUDACXX=/opt/cuda/bin/nvcc cmake -B build-cuda \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDAToolkit_ROOT=/opt/cuda
cmake --build build-cuda --config Release -j$(nproc)
sudo cp build-cuda/bin/llama-server /usr/local/bin/
```

If you only have CUDA 13 (Turing+ GPUs), drop the `CUDAToolkit_ROOT` — the default path works.

## Reachy Mini permissions

Arch uses the `uucp` group for serial devices (not `dialout` like Debian/Ubuntu):

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="uucp"
SUBSYSTEM=="usb", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="uucp"' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG uucp $USER
sudo usermod -aG video $USER
# Log out and back in for group changes to take effect
```

## Model downloads

```bash
mkdir -p models/gguf models/piper

# LLM (2.55 GB)
curl -L -o models/gguf/qwen3.5-4b-q4_k_m.gguf \
  "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"

# Vision projector (644 MB) — enables image understanding
curl -L -o models/gguf/mmproj-BF16.gguf \
  "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-BF16.gguf"

# TTS voice
curl -L -o models/piper/en_GB-northern_english_male-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx?download=true"
curl -L -o models/piper/en_GB-northern_english_male-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx.json?download=true"

# STT model (faster-whisper small.en) downloads automatically on first run
```

## VRAM budget

| Component | VRAM |
|-----------|------|
| Qwen 3.5 4B Q4_K_M | ~3 GB |
| Vision projector (mmproj-BF16) | ~0.7 GB |
| faster-whisper small.en (int8, CUDA) | ~1 GB |
| CUDA overhead | ~0.5 GB |
| **Total** | **~5.2 GB / 8 GB** |

To save VRAM, run Whisper on CPU (`WHISPER_DEVICE=cpu` in `.env`) — frees ~1 GB with minimal speed impact for small.en.

## .env for this machine

```
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

Everything else can stay at defaults from `.env.example`.

## Known quirks

- The Reachy Mini camera (`/dev/video0`) needs `video` group membership. The image is blurry in low light — the Ricoh Theta module has a fixed-focus lens.
- `ReachyMini()` must be called with `media_backend='no_media'` unless the GStreamer WebRTC rust plugin is installed (it isn't in this setup).
- Qwen 3.5 enables "thinking" by default. The agent client disables it via `extra_body` — without this, the model burns the entire token budget on internal reasoning and never produces a spoken response.
