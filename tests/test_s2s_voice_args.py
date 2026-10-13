"""The voice argv of scripts/run_s2s.sh, proven without launching the service.

`personas/<name>.yaml voice.instruct` reaches the pinned s2s service as
`S2S_QWEN3_INSTRUCT` (stamped into deploy/generated/persona.env), and the point of
that chain is that a persona switch can change the *style* of the voice. Until now
nothing on the launch side read the variable at all, so the field was decoration on
a public picker - exactly the failure class docs/knobs.md exists to catch. These
tests are the gate on both directions of the rule:

  * set   -> the flag arrives, commas and spaces intact, as ONE argument;
  * empty -> the flag is ABSENT, not `--qwen3_tts_instruct ""`. Upstream's default
    is None and a VoiceDesign model takes its entire style from the field, so an
    empty string is a value that erases the voice, not "no opinion" (a blank
    instruct was found to degrade output earlier in the rebuild). persona.env
    always writes the key, empty included, so emptiness - not presence - decides.

`--print-tts-args` is the pure seam inside the script (`build_tts_args`). It runs
ahead of the install guard and the exec, so this file needs no service, no model,
no GPU, no microphone and no network - and it can never be the test that boots the
booth's TTS out from under a live demo.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_S2S = ROOT / "scripts" / "run_s2s.sh"

INSTRUCT_FLAG = "--qwen3_tts_instruct"


def tts_args(tmp_path: Path, **overrides: str) -> list[str]:
    """The argv `build_tts_args` produces for this environment, one item per line.

    A minimal environment on purpose: the desk's own shell may carry S2S_* lines,
    and a gate that measures the desk instead of the script is the bug
    scripts/check_env_safety.sh had to fix in one of its own prelude edits.
    """
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "HF_HOME": str(tmp_path / "hf")}
    env.update({key: str(value) for key, value in overrides.items()})
    proc = subprocess.run(
        ["bash", str(RUN_S2S), "--print-tts-args"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.strip()[:400]
    return [line for line in proc.stdout.splitlines() if line != ""]


def value_after(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args else None


def test_a_set_instruct_is_passed_through_as_one_argument(tmp_path):
    style = "dry, clipped, flat, a touch of iron"
    args = tts_args(tmp_path, S2S_TTS="qwen3", S2S_QWEN3_INSTRUCT=style)
    assert INSTRUCT_FLAG in args, f"the dead knob is back: {args}"
    assert value_after(args, INSTRUCT_FLAG) == style  # not word-split on the commas


def test_an_empty_instruct_leaves_the_flag_absent(tmp_path):
    """Empty is what every shipped persona carries tonight; it must mean 'absent'."""
    for over in ({"S2S_TTS": "qwen3"}, {"S2S_TTS": "qwen3", "S2S_QWEN3_INSTRUCT": ""}):
        args = tts_args(tmp_path, **over)
        assert INSTRUCT_FLAG not in args, f'`{INSTRUCT_FLAG} ""` stamps a blank style: {args}'
        assert value_after(args, "--qwen3_tts_speaker") == "Ryan"  # the rest still goes out


def test_a_local_talker_and_a_style_line_coexist(tmp_path):
    """Tonight's persona path: a fine-tuned GGUF plus a style line, both in one argv."""
    args = tts_args(
        tmp_path,
        S2S_TTS="qwen3",
        S2S_QWEN3_TALKER_GGUF="/x/qwen-talker-1.7b-Q8_0.gguf",
        S2S_QWEN3_CODEC_GGUF="/x/qwen-tokenizer-12hz-Q8_0.gguf",
        S2S_QWEN3_INSTRUCT="flat and literal",
    )
    assert value_after(args, "--qwen3_tts_gguf_talker_path") == "/x/qwen-talker-1.7b-Q8_0.gguf"
    assert value_after(args, INSTRUCT_FLAG) == "flat and literal"


def test_the_kokoro_path_carries_no_qwen3_flags(tmp_path):
    args = tts_args(tmp_path, S2S_TTS="kokoro", S2S_QWEN3_INSTRUCT="dry, clipped", KOKORO_VOICE="af_maple")
    assert not any(str(a).startswith("--qwen3_tts") for a in args), args
    assert value_after(args, "--kokoro_voice") == "af_maple"
    assert value_after(args, "--kokoro_lang_code") == "a"


def test_an_unpinned_engine_prints_no_voice_args_at_all(tmp_path):
    # S2S_TTS=<anything else> = upstream default engine: no flags, and no crash.
    assert tts_args(tmp_path, S2S_TTS="pocket", S2S_QWEN3_INSTRUCT="dry") == []


def test_the_seam_does_not_need_the_pinned_service_installed(tmp_path):
    """S2S_DIR points at an empty dir. A gate that could only run on the booth box
    would skip itself here and tell nobody, which is how the dead knob survived."""
    args = tts_args(tmp_path, S2S_DIR=str(tmp_path / "no-service"), S2S_TTS="qwen3")
    assert value_after(args, "--qwen3_tts_backend") == "ggml"
