#!/usr/bin/env python
"""Bench the XVF3800 echo path in about 10 minutes (issue #40, ADR 0005).

The question this answers, and nothing else: when Reachy is playing sound, is a
person still audible in the capture, and is the chip's echo reference actually
fed on THIS route? The answer decides whether barge-in is a threshold problem
(policy) or an architecture problem, and ADR 0005 forbids guessing.

    .venv/bin/python scripts/bench_aec.py                 # steps 0-1 + params
    .venv/bin/python scripts/bench_aec.py --person        # adds step 2 (you talk)
    .venv/bin/python scripts/bench_aec.py --out data/bench-aec.json

ADR 0002 says one mic owner and one speaker owner, so this REFUSES to run while
reachy-shell is up: two capture owners produce numbers that look fine and mean
nothing. Stop the shell first, and put the robot back after.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from array import array
from pathlib import Path

import numpy as np
import sounddevice as sd

SR = 16000
BLOCK = 512
PARAMS = (
    "VERSION",
    "AEC_AECCONVERGED",
    "AEC_AECPATHCHANGE",
    "AEC_NUM_FARENDS",
    "AEC_RT60",
    "AUDIO_MGR_SYS_DELAY",
    "AUDIO_MGR_REF_GAIN",
    "AUDIO_MGR_MIC_GAIN",
    "AUDIO_MGR_OP_L",
    "AUDIO_MGR_OP_R",
    "AUDIO_MGR_SELECTED_CHANNELS",
    "PP_ECHOONOFF",
    "PP_DTSENSITIVE",
    "PP_MIN_NS",
    "PP_AGCMAXGAIN",
    "AEC_FIXEDBEAMSONOFF",
    "DOA_VALUE_RADIANS",
)
# ADR 0005's own note: mono playback is the correct shape for this chip (the
# reference is channel 0), so a stereo capture here is diagnostic, not a bug fix.


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def shell_running() -> bool:
    if shutil.which("systemctl") is None:
        return False
    r = subprocess.run(["systemctl", "--user", "is-active", "reachy-shell"], capture_output=True, text=True)
    return r.stdout.strip() == "active"


def default_devices() -> tuple[str, str]:
    out = subprocess.run(["wpctl", "status"], capture_output=True, text=True).stdout
    def one(section: str) -> str:
        lines = out.split(f"{section}\n", 1)[-1].splitlines() if section in out else []
        for ln in lines:
            if ln.strip().startswith("*"):
                return ln.strip().lstrip("* ").strip()
        return "(none)"
    return one("Sinks:"), one("Sources:")


def read_params() -> dict:
    try:
        from reachy_mini.media.audio_control_utils import init_respeaker_usb
    except Exception as e:  # noqa: BLE001
        return {"error": f"cannot import audio_control_utils: {e!r}"}
    dev = init_respeaker_usb()
    if dev is None:
        return {"error": "no XVF3800 on USB (robot off, or the daemon holds the control interface)"}
    out: dict = {}
    try:
        for p in PARAMS:
            try:
                out[p] = dev.read(p)
            except Exception as e:  # noqa: BLE001
                out[p] = f"read failed: {type(e).__name__}"
    finally:
        dev.close()
    return out


class Capture:
    def __init__(self) -> None:
        self.bufs: list[np.ndarray] = []

    def cb(self, ind, frames, t, s) -> None:  # noqa: ANN001, ARG002
        self.bufs.append(ind.copy())

    def drain(self) -> np.ndarray:
        if not self.bufs:
            return np.zeros((0, 2), dtype=np.float32)
        out = np.concatenate(self.bufs, axis=0)
        self.bufs.clear()
        return out


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.35 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", action="store_true", help="include step 2 (you speak over the robot)")
    ap.add_argument("--speech-file", default="sounds/acks/obama", help="wav file or dir of wavs to play as 'his' speech")
    ap.add_argument("--tone-seconds", type=float, default=4.0)
    ap.add_argument("--out", default="", help="write the numbers as JSON here")
    ap.add_argument("--force", action="store_true", help="run anyway with reachy-shell active (numbers are then meaningless)")
    args = ap.parse_args()

    print("== step 0: audio route ==")
    sink, source = default_devices()
    print(f"  sink:   {sink}")
    print(f"  source: {source}")
    route_ok = "Reachy Mini" in sink and "Reachy Mini" in source
    if not route_ok:
        print("  NO. Playback is not landing on the Reachy card, so the chip gets no far-end")
        print("  reference on this route. Fix the default devices before reading anything else;")
        print("  at the office this is the single most likely cause of 'he can't be interrupted'.")
    if shell_running() and not args.force:
        print("\nreachy-shell is ACTIVE: it owns the mic and speaker (ADR 0002).")
        print("Stop it first:  systemctl --user stop reachy-shell")
        print("(--force exists, but two capture owners invalidate exactly what we are measuring.)")
        return 2

    print("\n== step 1: does echo survive? ==")
    cap = Capture()
    stats: dict = {"route": {"sink": sink, "source": source, "both_reachy": route_ok}}
    with sd.CallbackInputStream(samplerate=SR, blocksize=BLOCK, channels=2, callback=cap.cb):
        time.sleep(2.0)
        base = cap.drain()
        base_r = [rms(base[:, i : i + 1]) for i in range(base.shape[1])]
        print(f"  room silence rms        L={base_r[0]:.5f}" + (f"  R={base_r[1]:.5f}" if len(base_r) > 1 else ""))
        sd.play(tone(args.tone_seconds), SR)
        time.sleep(args.tone_seconds + 1.0)
        echo = cap.drain()
        echo_r = [rms(echo[:, i : i + 1]) for i in range(echo.shape[1])]
        print(f"  while playing a tone    L={echo_r[0]:.5f}" + (f"  R={echo_r[1]:.5f}" if len(echo_r) > 1 else ""))
        sd.stop()
        gain = echo_r[0] / base_r[0] if base_r[0] > 0 else float("inf")
        print(f"  echo rise over silence  x{gain:.1f}   (want ~1.0: the mic should barely notice)")
        stats["silence_rms"] = base_r
        stats["echo_rms"] = echo_r
        stats["echo_over_silence"] = gain

        if args.person:
            print("\n== step 2: can a person be heard over him? (talk at ~1 m, straight in) ==")
            files = _speech_files(Path(args.speech_file))
            if not files:
                print(f"  no wav under {args.speech_file}; skipping - play anything at demo volume by hand")
            else:
                clip = np.concatenate([np.asarray(sd.read(f)[0]).reshape(-1, 1)[:, 0] for f in files[:3]])
                clip = np.resize(clip, int(SR * 15)).astype(np.float32)
                cap.drain()
                sd.play(clip.reshape(-1, 1), SR)
                print("  playing his voice for 15 s - speak a full sentence at second 3 and second 10")
                time.sleep(15.0)
                over = cap.drain()
                sd.stop()
                over_r = [rms(over[:, i : i + 1]) for i in range(over.shape[1])]
                print(f"  capture while he speaks  L={over_r[0]:.5f}")
                stats["over_playback_rms"] = over_r
                _save(Path(args.out or "/tmp/bench-aec") , "over_playback", over)
                print("  now say the SAME sentence with him silent:")
                input("  press enter to record 5 s of silence-free capture...")
                cap.drain()
                time.sleep(5.0)
                quiet = cap.drain()
                quiet_r = [rms(quiet[:, i : i + 1]) for i in range(quiet.shape[1])]
                ratio = over_r[0] / quiet_r[0] if quiet_r[0] > 0 else 0.0
                print(f"  interrupter level while he plays / when silent: {ratio:.2f}")
                _save(Path(args.out or "/tmp/bench-aec"), "quiet_reference", quiet)
                stats["person_ratio"] = ratio
                print("  VERDICT RULE (from the plan):")
                if ratio >= 0.5:
                    print("    >= ~0.5  -> the acoustic path is fine. The complaint is policy:")
                    print("               --thresh 0.7, min_speech_ms 384, the AGC ceiling, beam focus.")
                    print("               Widening our mic gate during replies (option A) is the fix.")
                elif ratio and ratio <= 0.1:
                    print("    <= ~0.1  -> chip-side suppression (double-talk, PP_ECHOONOFF,")
                    print("               PP_DTSENSITIVE, PP_MIN_NS, AEC_FIXEDBEAMSONOFF) is eating the")
                    print("               interrupter. Option B: tune the chip, do NOT widen the gate.")
                else:
                    print("    in between -> ambiguous; read PP_DTSENSITIVE and AEC_FIXEDBEAMSONOFF,")
                    print("               flip the beam one and repeat step 2.")

    print("\n== chip read-back ==")
    params = read_params()
    stats["params"] = {k: repr(v) for k, v in params.items()}
    if "error" in params:
        print(f"  {params['error']}")
        print("  Without this the AEC question stays open: the RMS numbers above show")
        print("  echo surviving, but not whether the chip thinks it is cancelling.")
    else:
        for k in ("VERSION", "AEC_AECCONVERGED", "AEC_NUM_FARENDS", "AEC_RT60", "AUDIO_MGR_SYS_DELAY",
                  "PP_ECHOONOFF", "PP_DTSENSITIVE", "PP_MIN_NS", "AEC_FIXEDBEAMSONOFF"):
            print(f"  {k:24s} {params.get(k)}")
        conv = str(params.get("AEC_AECCONVERGED"))
        far = str(params.get("AEC_NUM_FARENDS"))
        if conv.startswith("1") and not far.startswith("0"):
            print("  => reference is being counted and the AEC reports convergence: ADR 0005 closes YES")
            print("     for this route, subject to the RMS numbers above.")
        elif far.startswith("0") or conv.startswith("0"):
            print("  => NO usable reference on this route. Every software barge-in plan must carry")
            print("     its own cancellation, and option A alone would make him interrupt himself.")

    if args.out:
        Path(args.out).write_text(json.dumps(stats, indent=2, default=str))
        print(f"\nwrote {args.out}")
    print("\nRestore the robot:  systemctl --user start reachy-shell")
    return 0


def _speech_files(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    return sorted(p.rglob("*.wav")) if p.is_dir() else []


def _save(base: Path, name: str, data: np.ndarray) -> None:
    try:
        import wave

        out = base if base.suffix == ".wav" else base.parent / f"bench-aec-{name}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
        with wave.open(str(out), "wb") as w:
            w.setnchannels(pcm.shape[1]); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        print(f"  saved {out} (listen: this is what the mic actually got)")
    except Exception as e:  # noqa: BLE001
        print(f"  (could not save {name}: {e})")


if __name__ == "__main__":
    sys.exit(main())
