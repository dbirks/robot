#!/usr/bin/env bash
#
# Build llama.cpp for this machine's GPU, at a pinned commit.
#
# Why this script exists: llama.cpp is not vendored or submoduled here, it just
# lives in a sibling checkout. Before this script there was no record of which
# commit the running binary came from, and the installed /usr/local/bin build
# turned out to be three months stale without anyone noticing. Pin the ref here
# and bump it deliberately.
#
# Usage:
#   scripts/build_llama.sh                 # build the pinned ref
#   LLAMA_REF=master scripts/build_llama.sh   # try latest (then update the pin)
#
set -euo pipefail

# ---- pinned version -------------------------------------------------------
# Bump this deliberately, and re-run the checks at the bottom afterwards.
LLAMA_REF="${LLAMA_REF:-91f8c9c5}"          # 2026-07-27
LLAMA_DIR="${LLAMA_DIR:-$HOME/dev/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_DIR/build-cuda}"

# ---- Pascal-specific build flags ------------------------------------------
# CMAKE_CUDA_ARCHITECTURES=61 : GTX 1070 is sm_61. Building only this arch keeps
#   compile time down and guarantees we get kernels the card can run. (CUDA 13
#   dropped Pascal entirely, so build against CUDA 12.x -- 12.9 here.)
# GGML_CUDA_F16=OFF : GP104 runs fp16 at 1/64 of fp32, unlike GP100. Any fp16
#   compute path is a catastrophe on this exact chip. Must stay off.
CUDA_ARCH="${CUDA_ARCH:-61}"

echo "==> llama.cpp $LLAMA_REF  (arch sm_$CUDA_ARCH, F16 compute off)"
cd "$LLAMA_DIR"

if [ -n "$(git status --porcelain)" ]; then
    echo "!! $LLAMA_DIR has local changes; refusing to check out $LLAMA_REF" >&2
    exit 1
fi

git fetch --quiet origin
git checkout --quiet "$LLAMA_REF"
echo "==> at $(git log -1 --format='%h %ad %s' --date=short)"

cmake -B "$BUILD_DIR" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DGGML_CUDA_F16=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"${JOBS:-6}"

BIN="$BUILD_DIR/bin/llama-server"
echo
echo "==> built: $BIN"
"$BIN" --version 2>&1 | tail -2

# ---- post-build checks ----------------------------------------------------
echo "==> speculative decoding types available:"
"$BIN" --help 2>&1 | grep -A1 -- "--spec-type" | head -2 | sed 's/^/    /'

cat <<'NOTE'

==> Reminders
    * run_llama_server.sh expects this build on PATH, e.g.
          PATH="$BUILD_DIR/bin:$PATH" ./scripts/run_llama_server.sh
      /usr/local/bin/llama-server is a separate, root-owned install (robot-wh0).
    * MTP (--spec-type draft-mtp) works on this card but measured only ~+9%
      (43.0 -> 46.7 tok/s) and requires --parallel 1, which costs the barge-in
      slot. Not enabled. See robot-gsg.
NOTE
