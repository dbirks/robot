#!/usr/bin/env bash
# Install the pinned open-vocabulary KWS model (ADR 0003 wake fast path) into
# models/kws and generate the BPE-tokenized keywords file sherpa-onnx needs.
# Pinned in docs/pins.yaml (models.kws). Re-run after changing KEYWORDS.
set -euo pipefail

NAME=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$NAME.tar.bz2"
SHA256=f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a
DIR="${KWS_MODEL_DIR:-models/kws}"
# One phrase per line; "@Label" after the phrase sets what the shell sees.
# "Reachy" is not an English word, so the BPE model needs two spellings to
# catch it (REACHY + REACHIE, measured on synthesized speech).
KEYWORDS=("REACHY @Reachy" "REACHIE @Reachy" "HEY REACHY @hey Reachy" "HEY REACHIE @hey Reachy" "ROBOT @robot" "HEY ROBOT @hey robot")

if [ ! -f "$DIR/tokens.txt" ]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -sSLf -o "$tmp/kws.tar.bz2" "$URL"
  echo "$SHA256  $tmp/kws.tar.bz2" | sha256sum -c -
  tar xjf "$tmp/kws.tar.bz2" -C "$tmp"
  mkdir -p "$(dirname "$DIR")"
  rm -rf "$DIR"
  mv "$tmp/$NAME" "$DIR"
fi

# text2token needs sentencepiece (and imports pypinyin unconditionally); borrow it for this one step only rather
# than making it a runtime dependency of the shell.
printf '%s\n' "${KEYWORDS[@]}" > "$DIR/reachy-keywords-raw.txt"
uv run --extra kws --with sentencepiece --with pypinyin python - "$DIR" <<'PY'
import sys
from pathlib import Path

import sherpa_onnx

d = Path(sys.argv[1])
phrases, labels = [], []
for line in (d / "reachy-keywords-raw.txt").read_text().splitlines():
    phrase, _, label = line.partition("@")
    phrases.append(phrase.strip())
    labels.append(label.strip())
toks = sherpa_onnx.text2token(phrases, tokens=str(d / "tokens.txt"), tokens_type="bpe", bpe_model=str(d / "bpe.model"))
out = [" ".join(t) + (f" @{label.replace(' ', '_')}" if label else "") for t, label in zip(toks, labels)]
(d / "reachy-keywords.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
PY
echo "KWS model ready in $DIR"
