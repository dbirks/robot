# 2026-10-03 — Canonical Reachy voice: VoiceDesign candidates + first fine-tune trial

EPIC step "design once, then clone". `design.py` generated 16 candidates with
Qwen3-TTS 1.7B VoiceDesign (Q8_0, Pascal qwentts.cpp build `6fae929`),
8 description variants x 2 takes, temperature 0.9. Each clip is a single
generation of `REF_TEXT`, so it doubles as the 10-15 s clone reference for the
0.6B Base runtime. Outputs (gitignored): `models/voices/design-2026-10-03/`
(`candidates.json` records instruct/model/params per clip).

Measured on the GTX 1070 with llama-server resident: ~4.3 s to generate ~11 s
of audio (RTF ~0.39). Whole-clip codec decode OOMs on 8 GB; streaming decode
(chunk_size 8) is required.

Also tried: `kgptalkie/qwen3-tts-finetuned-baraq-obama` (1.7B Base SFT,
speaker `laxmikant`), converted with qwentts.cpp `convert.py` + `quantize`
to `~/dev/qwentts.cpp/obama/models/qwen-talker-1.7b-obama-Q8_0.gguf` (2.0 GB).
RTF ~0.38. Kept as a novelty voice.

Pending: user picks finalists -> fixed test corpus -> blind A/B -> freeze.
