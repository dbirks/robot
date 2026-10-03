#!/usr/bin/env bash
# Install huggingface/speech-to-speech as a PINNED, SEPARATE service (ADR 0001).
# Never vendored into this repo, never sharing our venv. Verify the protocol
# quirks listed in docs/adr/0001 on every ref bump.
set -euo pipefail

REF="${S2S_REF:-$(sed -n 's/^\s*ref:\s*\([^[:space:]#]*\).*/\1/p' docs/pins.yaml | head -1)}"
: "${REF:?could not read services.speech-to-speech.ref from docs/pins.yaml}"
DIR="${S2S_DIR:-$HOME/.local/share/reachy/speech-to-speech}"

if [ -x "$DIR/repo/.venv/bin/python" ] && [ -d "$DIR/repo/.git" ] && [ "$(git -C "$DIR/repo" describe --tags --always)" = "$REF" ]; then
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
uv venv --python 3.12 .venv
# Pascal (sm_61): torch must come from the cu126 index as the matched
# torch/torchaudio pair (ADR 0004); cu128+ wheels cannot use the GTX 1070.
# kokoro extra = production TTS (docs/pins.yaml).
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match \
  -e ".[kokoro]" "torch==2.11.0+cu126" "torchaudio==2.11.0+cu126" \
  "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
# ^ Kokoro's misaki G2P spacy.load()s en_core_web_sm and cannot self-install
#   it into a uv venv (no pip), so the service dies at TTS setup without it.
.venv/bin/python -c 'import torch; print("torch", torch.__version__, torch.cuda.get_arch_list()); x = torch.ones(64, 64, device="cuda"); assert (x @ x).sum().item() == 64**3, "GPU smoke test failed (ADR 0004)"; print("cuda ok on", torch.cuda.get_device_name())'
echo
echo "installed speech-to-speech $REF in $DIR/repo/.venv"
echo "record the exact commit in docs/pins.yaml, then use scripts/run_s2s.sh"
