#!/usr/bin/env bash
#
# Build qwentts.cpp (the native Qwen3-TTS engine behind s2s's --tts qwen3 ggml
# backend) for this GTX 1070, at the exact revision the pinned
# qwentts-cpp-python binding requires.
#
# Why: the PyPI wheel ships CUDA kernels for sm_75/86/90/120 only - no sm_61
# cubin and no PTX low enough to JIT - so it cannot run on Pascal (ADR 0004).
# The binding loads libqwen from $QWENTTS_CPP_LIBRARY; run_s2s.sh points it
# here. Bump QWENTTS_REF only together with the s2s venv's binding
# (qwentts_cpp/_binding.py: QWENTTS_NATIVE_REVISION, which it ABI-checks).
#
set -euo pipefail

QWENTTS_REF="${QWENTTS_REF:-6fae92914045cd83364d2845ceaa0f7969727319}"
QWENTTS_DIR="${QWENTTS_DIR:-$HOME/dev/qwentts.cpp}"
BUILD_DIR="${BUILD_DIR:-$QWENTTS_DIR/build-cuda61}"
CUDA_ARCH="${CUDA_ARCH:-61}"
NVCC="${NVCC:-/opt/cuda/bin/nvcc}"   # CUDA 12.x: CUDA 13 dropped Pascal

[ -d "$QWENTTS_DIR/.git" ] || git clone https://github.com/ServeurpersoCom/qwentts.cpp "$QWENTTS_DIR"
cd "$QWENTTS_DIR"
git fetch --quiet origin
git checkout --quiet "$QWENTTS_REF"
git submodule update --init --depth 1
echo "==> qwentts.cpp $(git log -1 --format='%h %ad' --date=short) for sm_$CUDA_ARCH"

# GGML_CUDA_F16=OFF: GP104 runs fp16 at 1/64 rate (same rule as build_llama.sh).
cmake -B "$BUILD_DIR" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DGGML_CUDA_F16=OFF \
    -DQWEN_SHARED=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target qwen -j"${JOBS:-6}"

LIB="$(find "$BUILD_DIR" -name 'libqwen.so' | head -1)"
echo
echo "==> built: $LIB"
echo "    run_s2s.sh uses it via QWENTTS_CPP_LIBRARY when S2S_TTS=qwen3"
