"""Pure parts of the talker measurement gate: no model, no GPU, no vendor venv.

scripts/measure_tts.py exists because ADR 0006 says an unmeasured model does not run on the
robot, and until tonight there was no way to produce a measurement - which is how one cloned
voice ended up referenced by all three persona files. These tests cover the half that can run
anywhere, including the bare CI environment.

Deliberately no autouse fixtures and no fake `subprocess.run` shared across cases: the gate
reads exactly one measurement (`gpu_free_mb`), so tests patch that one function. An earlier
version patched `subprocess.run` for the whole module, which meant the tests were asserting
whatever the fixture happened to shadow rather than the decision under test.
"""

import importlib.util
import json
import struct
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("measure_tts", ROOT / "scripts" / "measure_tts.py")
mt = importlib.util.module_from_spec(spec)
sys.modules["measure_tts"] = mt
spec.loader.exec_module(mt)


def make_wav(path: Path, seconds: float = 2.0, rate: int = 24000) -> Path:
    """A real PCM file written by the stdlib, because a hand-packed header would only prove
    that I packed it the same way twice."""
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(bytes(frames * 2))
    return path


def write_config(tmp_path: Path, configs: list) -> Path:
    cfg = tmp_path / "candidates.json"
    cfg.write_text(json.dumps(configs))
    return cfg


def real_talker(tmp_path: Path, name: str = "talker.gguf") -> Path:
    talker = tmp_path / name
    talker.write_bytes(b"x" * 5_000_000)
    return talker


def quiet_main(monkeypatch, tmp_path: Path, argv: list[str], free_mb: int | None = 8192) -> int:
    """Run main() with the card and the CPU-lie note neutralised, so a unit test never depends
    on whether the booth's service happens to be holding the GPU right now."""
    (tmp_path / "qwen-tokenizer-12hz-Q8_0.gguf").write_bytes(b"y" * 5_000_000)
    monkeypatch.setattr(mt, "MODELS", tmp_path)
    monkeypatch.setattr(mt, "VENV_PY", Path(sys.executable))
    monkeypatch.setattr(mt, "QWENTTS_LIB", "/opt/cpu-build/libqwen.so")
    monkeypatch.setattr(mt, "gpu_free_mb", lambda: free_mb)
    return mt.main(argv)


# --- duration -------------------------------------------------------------------


def test_duration_comes_from_the_header_not_the_file_size(tmp_path):
    assert mt.wav_seconds(make_wav(tmp_path / "two.wav", 2.0)) == 2.0
    assert mt.wav_seconds(make_wav(tmp_path / "half.wav", 0.5, rate=16000)) == 0.5


def test_seconds_divide_by_byte_rate_not_block_align(tmp_path):
    """The first draft divided by block align (offset 32) instead of byte rate (offset 28),
    which reported a 24 kHz mono file as 48000 seconds long. Every real-time factor in the
    table would have been wrong by 48000x, and ADR 0006 is worth nothing if the measurement
    it guards is invented. Pinned here with an explicit 96000 bytes / 48000 B/s = 2.0 s."""
    path = tmp_path / "manual.wav"
    data = 96000
    path.write_bytes(
        struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data,
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            24000,
            48000,  # byte rate
            2,  # block align - the wrong divisor
            16,
            b"data",
            data,
        )
        + bytes(data)
    )
    assert mt.wav_seconds(path) == 2.0


def test_a_non_wav_file_measures_zero_rather_than_raising(tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"PK\x03\x04" + bytes(64))
    assert mt.wav_seconds(junk) == 0.0
    assert mt.wav_seconds(tmp_path / "missing.wav") == 0.0


# --- the gate, asserted directly -------------------------------------------------


def test_the_vram_gate_decides_both_directions():
    """A gate that only ever refuses and a gate that only ever passes are equally useless."""
    assert mt.vram_gate("cuda", False, 7292) is None  # a free card proceeds
    assert "2200 MiB" in mt.vram_gate("cuda", False, 239)  # this card, right now
    assert "unreadable" in mt.vram_gate("cuda", False, None)  # fails closed
    assert mt.vram_gate("cuda", True, 239) is None  # --force means what it says
    assert mt.vram_gate("cpu", False, 239, cuda_lib=False) is None  # a genuine CPU build
    assert "2200 MiB" in mt.vram_gate("cpu", False, 239, cuda_lib=True)  # ours: not CPU


def test_a_stub_gguf_is_refused_as_a_stub(tmp_path, monkeypatch, capsys):
    """76-byte git-lfs pointer stubs satisfy a filename and Path.exists(); only the byte count
    says the model is absent. With HF_HUB_OFFLINE=1 nothing fetches it at runtime either."""
    stub = tmp_path / "qwen-talker-1.7b-customvoice-Q8_0.gguf"
    stub.write_bytes(b"x" * 76)
    code = quiet_main(
        monkeypatch,
        tmp_path,
        ["--device", "cpu", "--config", str(write_config(tmp_path, [{"label": "s", "talker": str(stub)}]))],
    )
    assert code == 2
    assert "git-lfs pointer" in capsys.readouterr().err


def test_a_missing_talker_refuses(tmp_path, monkeypatch, capsys):
    code = quiet_main(
        monkeypatch,
        tmp_path,
        [
            "--device",
            "cpu",
            "--config",
            str(write_config(tmp_path, [{"label": "g", "talker": str(tmp_path / "nope.gguf")}])),
        ],
    )
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_a_full_card_refuses_before_any_synthesis(tmp_path, monkeypatch, capsys):
    talker = real_talker(tmp_path)
    code = quiet_main(
        monkeypatch,
        tmp_path,
        [
            "--device",
            "cuda",
            "--config",
            str(write_config(tmp_path, [{"label": "x", "talker": str(talker)}])),
            "--lines",
            "1",
        ],
        free_mb=239,
    )
    assert code == 2
    assert "239 MiB" in capsys.readouterr().err


def test_an_unreadable_card_refuses_rather_than_assuming_room(tmp_path, monkeypatch, capsys):
    """Fails closed: 'cannot measure' must never count as 'there is space'."""
    talker = real_talker(tmp_path)
    code = quiet_main(
        monkeypatch,
        tmp_path,
        [
            "--device",
            "cuda",
            "--config",
            str(write_config(tmp_path, [{"label": "x", "talker": str(talker)}])),
            "--lines",
            "1",
        ],
        free_mb=None,
    )
    assert code == 2
    assert "unreadable" in capsys.readouterr().err


def test_no_candidates_never_reports_success(tmp_path, monkeypatch, capsys):
    """`all([])` is True, so an empty candidate list used to exit 0 having synthesised nothing.

    Reachable path: an empty config file trips the `not configs` check two lines before the
    `rows and all(...)` return, so this test covers the check that fires, not the second one.
    Asserted as such - mutation showed the inline `rows and` cannot be reached from here."""
    code = quiet_main(monkeypatch, tmp_path, ["--device", "cpu", "--config", str(write_config(tmp_path, []))])
    assert code == 2
    assert "no candidate" in capsys.readouterr().err


def test_time_to_first_audio_is_declared_unmeasured_not_inferred():
    """RTF is measured; TTFA is not, because the vendor CLI writes the WAV only after it has
    generated everything. ADR 0006 asks for TTFA, so the script says n/a and names the real
    fix (instrumenting the handler) instead of back-computing a number from RTF."""
    src = (ROOT / "scripts" / "measure_tts.py").read_text()
    assert '"ttfa_s": None' in src
    assert "n/a" in src
    assert "qwen3_tts_handler" in src, "the script must name what would actually produce TTFA"
