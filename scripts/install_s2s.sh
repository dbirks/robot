#!/usr/bin/env bash
# Install huggingface/speech-to-speech as a PINNED, SEPARATE service (ADR 0001).
# Never vendored into this repo, never sharing our venv. Verify the protocol
# quirks listed in docs/adr/0001 on every ref bump.
set -euo pipefail

REF="${S2S_REF:-$(sed -n 's/^\s*ref:\s*\(.*\)\s*#.*/\1/p' docs/pins.yaml | head -1)}"
: "${REF:?could not read services.speech-to-speech.ref from docs/pins.yaml}"
DIR="${S2S_DIR:-$HOME/.local/share/reachy/speech-to-speech}"

if [ -d "$DIR/repo/.git" ] && [ "$(git -C "$DIR/repo" describe --tags --always)" = "$REF" ]; then
  echo "speech-to-speech already installed at $REF ($DIR/repo)"
  exit 0
fi

mkdir -p "$DIR"
if [ ! -d "$DIR/repo/.git" ]; then
  git clone https://github.com/huggingface/speech-to-speech "$DIR/repo"
fi
git -C "$DIR/repo" fetch --tags origin
git -C "$DIR/repo" checkout "$REF"

command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
cd "$DIR/repo"
uv venv .venv
# On the Pascal box add --extra-index-url https://download.pytorch.org/whl/cu126
# and pin the matched torch/torchaudio pair (ADR 0004); re-check what upstream
# needs at the pinned ref (e.g. [realtime] extra) before running.
uv pip install -e .
echo
echo "installed speech-to-speech $REF in $DIR/repo/.venv"
echo "record the exact commit in docs/pins.yaml, then use scripts/run_s2s.sh"
