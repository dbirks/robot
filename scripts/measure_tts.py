#!/usr/bin/env python3
"""Measure a Qwen3-TTS talker on THIS card before it is allowed near the robot (ADR 0006).

Nothing here plays audio. It writes WAV files and prints a table: the speaker and microphone
have exactly one owner each (ADR 0002) and this script is not it, so an audition means files
you choose to open later.

Why it exists: every persona in personas/ currently points at the same fixed-speaker talker,
and the way that got noticed was nobody having measured an alternative. docs/pins.yaml
requires a revision plus a measured cost for every model, and until tonight there was no way
to produce the measurement, which is how "unmeasured but shipped" kept happening.

It deliberately drives the vendor CLI (`faster_qwen3_tts.cli`) rather than our own handler, so
what is measured is the engine, not a reimplementation of it by the person asking the question.

    uv run python scripts/measure_tts.py --device cpu --label obama --lines 2
    uv run python scripts/measure_tts.py --device cpu --config marvin-candidates.json
    REACHY_TTS_AUDITION=1 scripts/measure_tts.py --device cuda --force

Refuses to run on the GPU while the live service holds it unless --force, because two CUDA
contexts on an 8 GB Pascal card do not both survive - the failure is an OOM in the *live*
service, i.e. at the booth, not here.

Reported honestly: real-time factor and total synthesis time are measured. Time-to-first-audio
is NOT, because the vendor CLI writes the file after generating; ADR 0006 asks for TTFA, and
producing it means instrumenting `TTS/qwen3_tts_handler.py` inside the service. Until then the
table says "n/a" rather than inventing a number.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("S2S_DIR", Path.home() / ".local/share/reachy/speech-to-speech/repo"))
VENV_PY = REPO / ".venv/bin/python"
QWENTTS_LIB = os.environ.get("QWENTTS_CPP_LIBRARY", str(Path.home() / "dev/qwentts.cpp/build-cuda61/libqwen.so"))
MODELS = Path.home() / "dev/qwentts.cpp"
CUSTOMVOICE_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOICEDESIGN_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# Four lines that cover the ways the booth breaks him: a question, a long sentence, a
# date/number line (the format the clock tool returns), and the greeting he actually says.
LINES = [
    ("question", "What does this robot actually do?"),
    (
        "long",
        "The point is that everything you just heard was generated on this machine, with no network calls at all.",
    ),
    ("clock", "It is Monday, October 13, 2026, at 6:05 PM."),
    ("greeting", "Hi, I'm Reachy. Ask me something and I'll answer out loud."),
]

# A 1.7B Q8_0 talker asked for 1,765.04 MiB on this card - measured, in the load failure that
# motivated this constant. Rounded up, because the codec and activations want the remainder.
TALKER_VRAM_MIB = 2200


def gpu_free_mb() -> int | None:
    """Free GPU memory, or None if we cannot ask. Read-only; allocates nothing."""
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            .stdout.strip()
            .splitlines()
        )
        used, total = (int(value) for value in out[0].split(","))
        return total - used
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def wav_seconds(path: Path) -> float:
    """Audio duration from the RIFF header, no numpy and no soundfile, because this has to run
    in the bare CI environment too. Seconds are data bytes / BYTE RATE at offset 28; block align
    is at 32 and dividing by that instead (as the first draft did) reports a 24 kHz mono file as
    48000 seconds long and would have made every real-time factor here meaningless."""
    try:
        with path.open("rb") as fh:
            head = fh.read(44)
        if len(head) < 44 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return 0.0
        rate = int.from_bytes(head[24:28], "little")
        bytes_per_second = int.from_bytes(head[28:32], "little")
        if not bytes_per_second:  # malformed header: rebuild it from block align (32:34)
            block_align = int.from_bytes(head[32:34], "little")
            bytes_per_second = block_align * rate
        data = int.from_bytes(head[40:44], "little")
        return data / bytes_per_second if bytes_per_second else 0.0
    except OSError:
        return 0.0


def vram_gate(device: str, force: bool, free_mb: int | None, cuda_lib: bool = True) -> str | None:
    """Refusal message, or None if it is safe to load a talker. Pure, so the gate is testable
    without a GPU, a venv or a model - the decision, not the plumbing around it.

    Measured, not sniffed. The first version of this looked for the live service with
    `pgrep -fa speech_to_speech` and got both answers wrong: the server's argv[0] is plain
    `python`, so it found nothing and failed OPEN on a card with 239 MiB free; re-run with a
    different command line and it "passed" only by matching this script's own invocation.
    Free VRAM is the resource being contended for, so free VRAM is what gets checked, and an
    unreadable nvidia-smi fails closed rather than counting as room.
    """
    if force:
        return None
    if device != "cuda" and not cuda_lib:
        return None  # a genuinely CPU-built lib cannot contend for the card
    if free_mb is None:
        return (
            "refusing: --device cuda but nvidia-smi is unreadable, so there is no way to know "
            "the card has room. Pass --force only if you meant it."
        )
    if free_mb < TALKER_VRAM_MIB:
        return (
            f"refusing: {free_mb} MiB of GPU free and a talker needs ~{TALKER_VRAM_MIB} MiB. If "
            "the live service is holding the card, stopping reachy-s2s is an operator action; "
            "otherwise measure on another box. --force will OOM whichever process loads second."
        )
    return None


def run_one(
    py: Path,
    talker: Path,
    tokenizer: Path,
    model_id: str,
    line: str,
    out: Path,
    speaker: str | None,
    instruct: str | None,
    device: str,
    quant: str,
    timeout_s: int,
) -> tuple[bool, float, str]:
    """One vendor-CLI synthesis. Returns (ok, wall_seconds, detail)."""
    cmd = [
        str(py),
        "-m",
        "faster_qwen3_tts.cli",
        "--backend",
        "ggml",
        "--quant",
        quant,
        "--device",
        device,
        "--gguf-model",
        str(talker),
        "--qwentts-lib",
        QWENTTS_LIB,
    ]
    if tokenizer is not None:
        cmd += ["--gguf-codec", str(tokenizer)]
    if instruct:
        cmd += ["design", "--text", line, "--output", str(out), "--model", model_id, "--instruct", instruct]
    else:
        cmd += ["custom", "--text", line, "--output", str(out), "--model", model_id]
        if speaker:
            cmd += ["--speaker", speaker]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, time.monotonic() - t0, f"timeout after {timeout_s}s"
    wall = time.monotonic() - t0
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, wall, (tail[-1] if tail else f"exit {proc.returncode}")
    return True, wall, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", help="JSON: [{label, talker, speaker, instruct, model_id}, ...]")
    ap.add_argument(
        "--label", action="append", default=[], help="built-in candidate(s): obama, customvoice, voicedesign"
    )
    ap.add_argument("--lines", type=int, default=len(LINES), help="how many of the fixed lines to synth (1-4)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--quant", default=os.environ.get("S2S_QWEN3_QUANT", "Q8_0"))
    ap.add_argument("--out", default="/tmp/tts-audition")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--force", action="store_true", help="run on cuda even with the live service up")
    ap.add_argument("--list-speakers", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not VENV_PY.exists():
        print(f"pinned s2s venv not found at {VENV_PY}; set S2S_DIR", file=sys.stderr)
        return 2
    # Discovered here, not at import time: a model file that finishes downloading while this
    # process is starting must still be visible to it. An import-time glob made the very first
    # run of this script refuse a tokenizer that landed three seconds earlier.
    found = sorted(MODELS.glob("**/qwen-tokenizer-12hz-Q8_0.gguf"))
    tokenizer_path = os.environ.get("S2S_QWEN3_CODEC_GGUF", "") or (str(found[0]) if found else "")
    tokenizer = Path(tokenizer_path) if tokenizer_path else None
    if tokenizer is None:
        print("no qwen-tokenizer-12hz GGUF found; the ggml backend needs it (see docs/pins.yaml)", file=sys.stderr)
        return 2

    obama = sorted(MODELS.glob("**/qwen-talker-1.7b-obama-Q8_0.gguf"))
    custom = sorted(MODELS.glob("**/qwen-talker-1.7b-customvoice-Q8_0.gguf"))
    design = sorted(MODELS.glob("**/qwen-talker-1.7b-voicedesign-Q8_0.gguf"))
    builtin = {
        "obama": [{"label": "obama-current", "talker": str(obama[0]), "speaker": "laxmikant"}] if obama else [],
        "customvoice": [{"label": "customvoice-aiden", "talker": str(custom[0]), "speaker": "Aiden"}] if custom else [],
        "voicedesign": [
            {
                "label": "voicedesign-dry",
                "talker": str(design[0]),
                "model_id": VOICEDESIGN_ID,
                "instruct": "A calm, dry, flat male voice, unhurried, monotone, no warmth, slight resonance.",
            }
        ]
        if design
        else [],
    }
    configs = (
        json.loads(Path(args.config).read_text())
        if args.config
        else [c for name in args.label or ["obama"] for c in builtin.get(name, [])]
    )
    if not configs:
        print("nothing to measure: no candidate resolved to a file that exists", file=sys.stderr)
        return 2
    for cfg in configs:
        path = Path(cfg["talker"]).expanduser()
        if not path.is_file():
            print(f"refusing: talker not found for {cfg.get('label')}: {path}", file=sys.stderr)
            return 2
        if path.stat().st_size < 1_000_000:
            print(
                f"refusing: {path} is {path.stat().st_size} bytes - a git-lfs pointer stub, "
                "not a model (HF_HUB_OFFLINE=1 will not fetch it at runtime either)",
                file=sys.stderr,
            )
            return 2
    cpu_is_a_lie = "cuda" in QWENTTS_LIB.lower()
    if args.device == "cpu" and cpu_is_a_lie:
        print(
            f"note: QWENTTS_CPP_LIBRARY={QWENTTS_LIB} is a CUDA build, so ggml picks CUDA0 "
            "whatever --device says (observed: '[Load] Talker backend: CUDA0'). There is no CPU "
            "fallback on this box unless scripts/buildcpu.sh has been run and the variable "
            "points at it, so the VRAM gate applies to --device cpu too.",
            file=sys.stderr,
        )
    free = gpu_free_mb() if (args.device == "cuda" or cpu_is_a_lie) else None
    refusal = vram_gate(args.device, args.force, free, cuda_lib=cpu_is_a_lie)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.list_speakers:
        talker = Path(configs[0]["talker"]).expanduser()
        proc = subprocess.run(
            [
                str(VENV_PY),
                "-m",
                "faster_qwen3_tts.cli",
                "--backend",
                "ggml",
                "--quant",
                args.quant,
                "--device",
                args.device,
                "--gguf-model",
                str(talker),
                "--qwentts-lib",
                QWENTTS_LIB,
                "--gguf-codec",
                str(tokenizer),
                "custom",
                "--text",
                "x",
                "--output",
                "/tmp/_sp.wav",
                "--model",
                CUSTOMVOICE_ID,
                "--list-speakers",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        print(proc.stdout.strip() or proc.stderr.strip())
        return 0 if proc.returncode == 0 else 1

    rows = []
    for cfg in configs:
        talker = Path(cfg["talker"]).expanduser()
        lines = LINES[: max(1, min(args.lines, len(LINES)))]
        per = []
        for name, text in lines:
            out = out_dir / f"{cfg['label']}__{name}.wav"
            ok, wall, detail = run_one(
                VENV_PY,
                talker,
                tokenizer,
                cfg.get("model_id", CUSTOMVOICE_ID),
                text,
                out,
                cfg.get("speaker"),
                cfg.get("instruct"),
                args.device,
                args.quant,
                args.timeout,
            )
            secs = wav_seconds(out) if ok else 0.0
            per.append(
                {
                    "line": name,
                    "ok": ok,
                    "wall_s": round(wall, 2),
                    "audio_s": round(secs, 2),
                    "rtf": round(wall / secs, 2) if ok and secs else None,
                    "ttfa_s": None,
                    "detail": detail,
                    "wav": str(out) if ok else None,
                }
            )
            print(
                f"  {cfg['label']:22} {name:9} {'ok ' if ok else 'FAIL'} "
                f"wall={wall:6.1f}s audio={secs:5.1f}s rtf={(wall / secs if secs else 0):5.2f} {detail}",
                flush=True,
            )
        good = [p for p in per if p["ok"] and p["audio_s"]]
        rows.append(
            {
                "label": cfg["label"],
                "talker": str(talker),
                "talker_bytes": talker.stat().st_size,
                "speaker": cfg.get("speaker"),
                "instruct": cfg.get("instruct"),
                "device": args.device,
                "quant": args.quant,
                "gpu_used_mb_start": nvidia_smi_used_mb(),
                "runs": per,
                "mean_rtf": round(sum(p["wall_s"] for p in good) / sum(p["audio_s"] for p in good), 2)
                if good
                else None,
                "ok": bool(good),
            }
        )
    print(json.dumps(rows, indent=2) if args.json else "")
    print(f"{'label':24} {'mean RTF':>9}  {'VRAM MB':>8}  files")
    for r in rows:
        vram = r.get("gpu_used_mb_start")
        print(
            f"{r['label']:24} {str(r['mean_rtf']):>9}  {str(vram if r['device'] == 'cuda' else '-'):>8}  "
            f"{sum(1 for p in r['runs'] if p['ok'])}/{len(r['runs'])} in {out_dir}"
        )
    print("\nTime-to-first-audio is n/a above: the vendor CLI writes the WAV only after generating,")
    print("so TTFA needs instrumenting inside TTS/qwen3_tts_handler.py. It is not inferred from RTF.")
    if not rows:
        # An empty candidate list used to reach `all([])` here and report success while having
        # synthesised nothing. A measurement script that can pass by doing nothing is worse than
        # one that fails, because the whole point of ADR 0006 is that the gate is trusted.
        print("refusing: no candidate produced a measurement", file=sys.stderr)
        return 2
    # `rows and` below is belt-and-braces, not coverage: mutation-testing showed the branch
    # above is what fires for an empty config file, so nothing reachable can leave `rows` empty
    # here. It stays because `all([])` is True and this is the last line standing between "the
    # booth got an unmeasured voice" and a green exit code.
    return 0 if rows and all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
