"""Standalone TTS server using Qwen3-TTS with its own dependency environment.

Run with: cd tts_server && uv run server.py
Listens on port 5100. The main app calls POST /synthesize with JSON body.
"""

import io
import logging
import os
import time
import wave

import numpy as np
import torch
from flask import Flask, jsonify, request
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s")
log = logging.getLogger("tts-server")

MODEL_ID = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEVICE = os.getenv("TTS_DEVICE", "cpu")
SPEAKER = os.getenv("TTS_SPEAKER", "Ryan")
PORT = int(os.getenv("TTS_PORT", "5100"))

app = Flask(__name__)
model = None


def load_model():
    global model
    dtype = torch.float32 if DEVICE == "cpu" else torch.float16
    log.info("Loading %s on %s (%s)...", MODEL_ID, DEVICE, dtype)
    t = time.monotonic()
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map=DEVICE,
        dtype=dtype,
    )
    log.info("Model loaded in %.1fs", time.monotonic() - t)


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json()
    text = data.get("text", "")
    speaker = data.get("speaker", SPEAKER)
    instruct = data.get("instruct", "")
    language = data.get("language", "English")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        t = time.monotonic()
        kwargs = dict(text=text, language=language, speaker=speaker)
        if instruct:
            kwargs["instruct"] = instruct
        wavs, sr = model.generate_custom_voice(**kwargs)
        elapsed = time.monotonic() - t

        audio = wavs[0].cpu().numpy() if hasattr(wavs[0], "cpu") else np.asarray(wavs[0])
        if audio.dtype != np.int16:
            audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())

        log.info("TTS %.2fs for %d chars (%.1fs audio)", elapsed, len(text), len(audio) / sr)
        return buf.getvalue(), 200, {"Content-Type": "audio/wav"}
    except Exception as e:
        log.exception("Synthesis failed")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_ID, "device": DEVICE, "speaker": SPEAKER})


if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1", port=PORT, threaded=False)
