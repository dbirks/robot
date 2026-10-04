"""Canonical Reachy voice, step 1: generate VoiceDesign candidates (EPIC
"Canonical Reachy voice: design once, then clone").

Each candidate is ONE generation (VoiceDesign invents a new voice per call),
so the same clip is both what you audition and the 10-15 s reference the
0.6B Base model will clone at runtime. Its transcript is REF_TEXT.

Run in the s2s venv with the Pascal qwentts build (scripts/build_qwentts.sh):
    QWENTTS_CPP_LIBRARY=~/dev/qwentts.cpp/build-cuda61/libqwen.so \\
    ~/.local/share/reachy/speech-to-speech/repo/.venv/bin/python design.py OUT_DIR
Stop reachy-s2s first: 1.7B VoiceDesign + llama.cpp + live TTS exceed 8 GB.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

BASE = (
    "British man, warm but understated, friendly and a little charming, "
    "contemporary southern-British/RP-leaning accent, natural conversational "
    "rhythm, not an announcer, not theatrical, moderate pitch, calm confidence."
)
VARIANTS = {
    "epic": BASE,
    "younger": BASE + " Late twenties, light and quick-witted.",
    "older": BASE + " Around fifty, a slightly lower, gentle voice.",
    "dry": BASE + " Dry, deadpan sense of humour, very relaxed delivery.",
    "bright": BASE + " Bright and upbeat, smiling as he speaks.",
    "soft": BASE + " Soft-spoken and close to the microphone, unhurried.",
    "london": "Young man from London with a friendly, modern London accent, "
    "casual and warm, natural conversational rhythm, not theatrical.",
    "northern": "Friendly man from the north of England, warm Yorkshire accent, "
    "down to earth and cheerful, natural conversational rhythm.",
}
TAKES = 2  # two samples per variant -> 16 candidates

REF_TEXT = (
    "Hello, I'm Reachy, your little desk robot. "
    "I can look around, keep you company, and help with small things. "
    "Honestly, I'm rather pleased to meet you. Shall we get started?"
)


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device="cuda", backend="ggml", quant="Q8_0", local_files_only=True
    )
    meta = []
    n = 0
    for take in range(TAKES):
        for name, instruct in VARIANTS.items():
            n += 1
            t0 = time.time()
            chunks, sr = [], 24000
            # Streaming decode: whole-clip decode OOMs on the 8 GB card.
            for audio, sr, _ in model.generate_voice_design_streaming(
                REF_TEXT, instruct=instruct, language="English", chunk_size=8, temperature=0.9
            ):
                chunks.append(np.asarray(audio, dtype=np.float32).squeeze())
            wav = np.concatenate(chunks)
            fname = f"{n:02d}-{name}-{take + 1}.wav"
            sf.write(out / fname, wav, sr)
            meta.append(
                {"id": n, "file": fname, "variant": name, "take": take + 1, "instruct": instruct,
                 "ref_text": REF_TEXT, "seconds": round(len(wav) / sr, 2), "gen_s": round(time.time() - t0, 2),
                 "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign Q8_0 (Serveurperso/Qwen3-TTS-GGUF)",
                 "temperature": 0.9}
            )
            print(f"[{n:02d}] {fname} {meta[-1]['seconds']}s audio in {meta[-1]['gen_s']}s", flush=True)
            (out / "candidates.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "candidates")
