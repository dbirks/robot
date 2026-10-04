"""Render prerendered acknowledgement clips in the live Qwen3 voice.

Single words ("Hmm.") come out as grunts from Qwen3-TTS, so each ack is
generated as the opening of a natural sentence and cut at the first pause,
keeping real sentence prosody. Every clip is read back with Parakeet and kept
only if it says what it should. Output: 16 kHz mono int16 WAVs (properly
resampled with soxr; the shell plays at 16 kHz).

Run in the s2s venv with reachy-s2s STOPPED (GPU memory):
  QWENTTS_CPP_LIBRARY=~/dev/qwentts.cpp/build-cuda61/libqwen.so HF_HUB_OFFLINE=1 \
  HF_HOME=~/.local/share/reachy/speech-to-speech/hf-cache \
  ~/.local/share/reachy/speech-to-speech/repo/.venv/bin/python scripts/make_acks.py \
      --talker ~/dev/qwentts.cpp/obama/models/qwen-talker-1.7b-obama-Q8_0.gguf \
      --speaker laxmikant --out sounds/acks/obama
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

# (pool, keep-phrase, full sentence). The clip is cut after keep-phrase.
ACKS = [
    ("think", "hmm", "Hmm, let me think about that for a second."),
    ("think", "hmm", "Hmm, that is a really good question."),
    ("think", "okay", "Okay, so here is the thing."),
    ("think", "well", "Well, let me see what I can do."),
    ("think", "right", "Right, okay, give me a moment."),
    ("think", "so", "So, let me think about this."),
    ("wake", "yes", "Yes? What can I do for you?"),
    ("wake", "hmm", "Hmm? Did somebody call me?"),
    ("wake", "yeah", "Yeah? I'm listening."),
]
TAKES = 3
OUT_RATE = 16000


def cut_at_first_pause(a: np.ndarray, sr: int, min_gap_s=0.10, thr_rel=0.05) -> np.ndarray | None:
    """Return audio from speech onset to the first >=min_gap_s pause."""
    hop = int(0.01 * sr)
    frames = np.abs(a[: len(a) // hop * hop]).reshape(-1, hop).max(axis=1)
    thr = max(0.01, frames.max() * thr_rel)
    voiced = frames > thr
    if not voiced.any():
        return None
    start = int(np.argmax(voiced))
    gap_need = int(min_gap_s / 0.01)
    run = 0
    for i in range(start + 5, len(voiced)):  # at least 50 ms of speech first
        run = run + 1 if not voiced[i] else 0
        if run >= gap_need:
            end = i - run + 1
            s = max(0, start * hop - int(0.02 * sr))
            e = min(len(a), end * hop + int(0.06 * sr))
            return a[s:e]
    return None


def finish(clip: np.ndarray, sr: int) -> np.ndarray:
    clip = soxr.resample(clip.astype(np.float32), sr, OUT_RATE, quality="HQ")
    clip = clip / max(1e-6, np.abs(clip).max()) * 0.8
    fade = int(0.008 * OUT_RATE)
    clip[:fade] *= np.linspace(0, 1, fade)
    clip[-fade:] *= np.linspace(1, 0, fade)
    return clip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--talker", required=True)
    ap.add_argument("--codec", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-s", type=float, default=1.1)
    args = ap.parse_args()

    from faster_qwen3_tts import FasterQwen3TTS
    from nano_parakeet import from_pretrained as parakeet

    codec = (
        args.codec
        or sorted(
            glob.glob(
                str(
                    Path.home()
                    / ".local/share/reachy/speech-to-speech/hf-cache/hub/models--Serveurperso--Qwen3-TTS-GGUF/snapshots/*/qwen-tokenizer-12hz-Q8_0.gguf"
                )
            )
        )[0]
    )
    tts = FasterQwen3TTS.from_pretrained(
        args.model,
        device="cuda",
        backend="ggml",
        gguf_talker_path=args.talker,
        gguf_codec_path=codec,
        local_files_only=True,
    )
    asr = parakeet(model_name="nvidia/parakeet-tdt-0.6b-v3", device="cpu")

    out = Path(args.out)
    for pool in ("think", "wake"):
        (out / pool).mkdir(parents=True, exist_ok=True)
        for old in (out / pool).glob("*.wav"):
            old.unlink()

    kept = 0
    for i, (pool, keep, sentence) in enumerate(ACKS):
        for take in range(TAKES):
            chunks, sr = [], 24000
            for a, sr, _ in tts.generate_custom_voice_streaming(
                sentence, speaker=args.speaker, language="English", chunk_size=8
            ):
                chunks.append(np.asarray(a, dtype=np.float32).squeeze())
            full = np.concatenate(chunks)
            # Parakeet is unreliable on half-second clips, so verify the FULL
            # sentence came out right, then cut its opening phrase.
            heard = str(asr.transcribe(soxr.resample(full, sr, OUT_RATE, quality="HQ")))
            want = re.sub(r"[^a-z ]", "", sentence.lower()).split()
            got = re.sub(r"[^a-z ]", "", heard.lower()).split()
            match = sum(w in got for w in want) / len(want)
            clip = cut_at_first_pause(full, sr)
            ok = match >= 0.8 and clip is not None and len(clip) / sr <= args.max_s
            dur = 0.0 if clip is None else len(clip) / sr
            print(
                f"{'KEEP' if ok else 'drop'} {pool} '{keep}' t{take} cut={dur:.2f}s match={match:.2f} heard={heard!r}"
            )
            if ok:
                clip = finish(clip, sr)
                sf.write(out / pool / f"{keep}-{i}{take}.wav", (clip * 32767).astype(np.int16), OUT_RATE)
                kept += 1
    print(f"kept {kept}")


if __name__ == "__main__":
    main()
