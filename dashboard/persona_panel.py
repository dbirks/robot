"""Persona picker: the pure decisions, with no yaml, no fastapi and no shell.

One persona is one thing — how Reachy SOUNDS (the Qwen3-TTS voice the pinned s2s
service loads) and how he BEHAVES (the spoken manner in the system prompt) — because
David's ask was "Obama or Alan Rickman", not "one of two prompts and, separately, one
of three .env lines". Before this, those two halves lived in different places
(`envs/*.yaml` prose vs three `S2S_QWEN3_*` lines in `.env`), which is exactly the
drift docs/environments.md was written to kill.

Every function here takes data that is already parsed (a persona doc, a rendered
`KEY=value` body, a `--print-personas` payload). Nothing in this module reads YAML:
the dashboard venv has no pyyaml, CI runs these tests with only numpy + pyyaml +
pytest, and neither may pull fastapi or the shell. Parsing happens once, on the side
that has a parser — `shell/env.py` for the repo files, `json` for the panel payload.

The one asymmetry worth reading twice: `persona.env` sets the voice, the ack pool and
`REACHY_PERSONA`, and never `REACHY_INSTRUCTIONS`. It is one global file read by both
rooms, so a stamped persona prompt would put the home persona — family assistant, full
tool list — into a public booth the moment someone forgot to move the room back
(docs/environments.md hard rule 1). The room keeps its own words; the picker moves the
sound. `inline_shadow()` says out loud when an env file's inline `instructions:` is
still outranking the named persona, which is the case in both shipped rooms tonight.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = REPO_ROOT / "deploy" / "generated"
PERSONA_ENV = GENERATED_DIR / "persona.env"
APPLY_LOCK = GENERATED_DIR / "persona.apply.lock"

# `voice:` field in personas/<name>.yaml -> the variable scripts/run_s2s.sh reads.
# Order is the order apply_env.py prints them.
VOICE_KEYS: tuple[tuple[str, str], ...] = (
    ("engine", "S2S_TTS"),
    ("model", "S2S_QWEN3_MODEL"),
    ("talker_gguf", "S2S_QWEN3_TALKER_GGUF"),
    ("codec_gguf", "S2S_QWEN3_CODEC_GGUF"),
    ("speaker", "S2S_QWEN3_SPEAKER"),
    ("instruct", "S2S_QWEN3_INSTRUCT"),
    ("speed", "S2S_QWEN3_SPEED"),
)
# Voice variables that only exist for the qwen3 engine; a kokoro persona must not
# leave a talker GGUF behind for the next boot (empty clears the stray .env line).
QWEN3_ONLY = (
    "S2S_QWEN3_MODEL",
    "S2S_QWEN3_TALKER_GGUF",
    "S2S_QWEN3_CODEC_GGUF",
    "S2S_QWEN3_SPEAKER",
    "S2S_QWEN3_INSTRUCT",
)
# What reachy-shell needs from the same file (config.py reads the process env).
PERSONA_VAR = "REACHY_PERSONA"
ACK_VAR = "REACHY_ACK_DIR"
NAME_STAMP = "PERSONA"

# A field with no flag to hang it on at the pinned upstream ref (docs/pins.yaml
# services.speech-to-speech). Dropping it silently would be the drift this whole
# layer exists to kill, so apply_env.py says it out loud instead.
UNSUPPORTED: dict[str, str] = {
    "speed": "no --qwen3_tts_speed flag exists at s2s v1.0.0 (voice_prompt has `instruct`, not speed)",
}

# Measured on this box 2026-10-12 (deploy/README.md): a persona change is not free, and
# the panel must say so before the operator clicks.
S2S_RESTART_SECONDS = "20-46"

_LOCK_STALE_S = 420.0  # apply + BOTH restarts (180 s timeout each) + margin; a stuck lock is cleared by hand
PERSONAS_DIR = REPO_ROOT / "personas"


# ---------------------------------------------------------------------------
# persona doc -> the env file the services read
# ---------------------------------------------------------------------------


def voice_of(persona: dict[str, Any]) -> dict[str, str]:
    """The `voice:` block as {env var: value}, engine-aware, always complete.

    A key the persona does not name is written as `KEY=` rather than omitted: this
    file is listed AFTER `.env` in both units (measured 2026-10-12: a later
    `EnvironmentFile=` beats an earlier one AND `Environment=`, and an empty value
    still counts as set), so an omitted line would let a stale `.env` talker GGUF
    survive a switch to kokoro.
    """
    voice = persona.get("voice")
    voice = voice if isinstance(voice, dict) else {}
    engine = str(voice.get("engine") or "").strip() or "kokoro"
    out: dict[str, str] = {"S2S_TTS": engine}
    if engine == "kokoro":
        # Kokoro voices are <lang><gender>_<name> and come from KOKORO_VOICE; the
        # qwen3 keys below still get written, empty, to clear a stale .env line.
        out["KOKORO_VOICE"] = str(voice.get("speaker") or "").strip()
    for field, var in VOICE_KEYS:
        if var == "S2S_TTS":
            continue
        value = str(voice.get(field) or "").strip()
        out[var] = value if engine == "qwen3" or var not in QWEN3_ONLY else ""
    return out


def persona_env_values(persona: dict[str, Any]) -> dict[str, str]:
    """Everything apply_env.py stamps: the voice, plus what reachy-shell needs.

    `REACHY_INSTRUCTIONS` is deliberately absent — see the module docstring.
    """
    name = str(persona.get("name") or "").strip()
    values = dict(voice_of(persona))
    values[NAME_STAMP] = name
    values[PERSONA_VAR] = name
    values[ACK_VAR] = str(persona.get("ack_dir") or "").strip()
    return values


def quote(value: str) -> str:
    """systemd `EnvironmentFile=` quoting: one line per value, no shell expansion.

    Paths with spaces and an instruct prompt with commas both have to survive, and an
    unquoted value with a `#` would silently truncate the line.
    """
    text = str(value)
    if text == "":
        return ""
    if all(ch.isalnum() or ch in "_-./:=" for ch in text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_persona_env(values: dict[str, str]) -> str:
    """The file body, with a header that tells the next operator where it came from."""
    lines = [
        "# GENERATED by scripts/apply_env.py — do not edit, it is overwritten on the next apply.",
        "# The persona (voice + acks): personas/<name>.yaml, named by envs/<env>.yaml.",
        f"# Listed after EnvironmentFile=-.env in reachy-s2s.service and reachy-shell.service,",
        "# so it wins over stray .env lines (ordering measured 2026-10-12, deploy/README.md).",
        "# It never sets REACHY_INSTRUCTIONS: the room's words come from envs/<env>.yaml.",
    ]
    lines += [f"{key}={quote(value)}" for key, value in values.items()]
    return "\n".join(lines) + "\n"


def parse_env_text(text: str) -> dict[str, str]:
    """Read a `KEY=value` body with the stdlib (the dashboard has no python-dotenv)."""
    out: dict[str, str] = {}
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def read_persona_env(path: Path | str = PERSONA_ENV) -> dict[str, str]:
    """What is in force on disk right now ({} when it has never been applied)."""
    try:
        return parse_env_text(Path(path).read_text())
    except OSError:
        return {}


def active_persona(current: dict[str, str]) -> str:
    """The persona whose voice is loaded in the running service."""
    return str(current.get(NAME_STAMP) or current.get(PERSONA_VAR) or "").strip()


# ---------------------------------------------------------------------------
# restart decision
# ---------------------------------------------------------------------------

S2S_VARS = frozenset({"S2S_TTS", "KOKORO_VOICE"} | set(QWEN3_ONLY))
SHELL_VARS = frozenset({PERSONA_VAR, ACK_VAR})


def restart_plan(current: dict[str, str], values: dict[str, str]) -> dict[str, Any]:
    """Pure: does applying this persona need a restart, and of which units?

    The s2s service reads the TTS flags once at startup, so a voice change is only
    live after `systemctl --user restart reachy-s2s` — measured 20-46 s of dead
    air, which is why the panel asks before the click rather than after it.
    """
    changed = sorted(key for key in set(current) | set(values) if current.get(key, "") != values.get(key, ""))
    units: list[str] = []
    if any(key in S2S_VARS for key in changed):
        units.append("reachy-s2s")
    if any(key in SHELL_VARS for key in changed) or not current:
        units.append("reachy-shell")
    if "reachy-s2s" in units and "reachy-shell" not in units:
        units.append("reachy-shell")  # the shell re-stamps its ack pool on restart
    return {"changed": changed, "units": units, "needed": bool(units)}


# ---------------------------------------------------------------------------
# the pins guard (AGENTS.md: nothing runs on this box unmeasured)
# ---------------------------------------------------------------------------


def pins_guard(persona: dict[str, Any], pins_text: str) -> list[str]:
    """Warnings for a voice with no measured entry in docs/pins.yaml.

    The check is textual on purpose: pins.yaml is a hand-written document about
    hardware, not a registry, and the one thing that must not happen is a persona
    reaching an unmeasured GGUF on an 8 GB Pascal card with nobody told. An
    unmentioned model id or talker file name is therefore a warning, and
    `--strict` makes it an error.
    """
    voice = persona.get("voice")
    voice = voice if isinstance(voice, dict) else {}
    engine = str(voice.get("engine") or "").strip()
    if not engine or engine == "kokoro":
        return []  # kokoro is pinned as models.tts_production
    want = [
        ("model", str(voice.get("model") or "").strip()),
        ("talker GGUF", Path(str(voice.get("talker_gguf") or "")).name),
    ]
    haystack = pins_text.lower()
    missing = [f"{label} {value!r}" for label, value in want if value and value.lower() not in haystack]
    return [f"{item} has no entry in docs/pins.yaml (AGENTS.md: no unmeasured model on this box)" for item in missing]


def voice_warnings(persona: dict[str, Any]) -> list[str]:
    """Voice fields the pinned service has no flag for — said out loud, never dropped."""
    voice = persona.get("voice")
    voice = voice if isinstance(voice, dict) else {}
    return [
        f"voice.{field}={voice[field]!r} is not applied: {why}"
        for field, why in UNSUPPORTED.items()
        if str(voice.get(field) or "").strip()
    ]


def inline_shadow(env_doc: dict[str, Any]) -> bool:
    """True when the room's inline `persona.instructions` still outranks the persona file.

    Both shipped rooms do this tonight (A2 parity bar), which means the picker changes
    the voice and the acks but not the words there. The panel says it instead of
    letting the operator believe otherwise.
    """
    persona = env_doc.get("persona") if isinstance(env_doc, dict) else None
    return isinstance(persona, dict) and bool(str(persona.get("instructions") or "").strip())


# ---------------------------------------------------------------------------
# one apply at a time
# ---------------------------------------------------------------------------


def lock_is_stale(age_s: float, stale_s: float = _LOCK_STALE_S) -> bool:
    """A lock older than a full apply + two restarts is a crash, not a busy operator."""
    return age_s > stale_s


def try_acquire_lock(path: Path | str = APPLY_LOCK, *, pid: int | None = None, now: float | None = None) -> bool:
    """O_EXCL is the lock: two `apply` processes cannot both think they won, and the
    dashboard's refuse check and apply_env.py's own check are the same file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except OSError:
        return False
    with os.fdopen(fd, "w") as handle:
        handle.write(f"{pid if pid is not None else os.getpid()} {now if now is not None else time.time()}\n")
    return True


def lock_age(path: Path | str = APPLY_LOCK, now: float | None = None) -> float | None:
    """None = no lock on disk."""
    try:
        fields = Path(path).read_text().split()
        started = float(fields[1]) if len(fields) > 1 else 0.0
    except (OSError, ValueError, IndexError):
        return None
    return max(0.0, (now if now is not None else time.time()) - started)


def apply_busy(
    path: Path | str = APPLY_LOCK, *, now: float | None = None, stale_s: float = _LOCK_STALE_S
) -> dict[str, Any]:
    """The busy decision, shared by the dashboard route and apply_env.py."""
    age = lock_age(path, now)
    if age is None:
        return {"busy": False, "age_s": None, "stale": False}
    stale = lock_is_stale(age, stale_s)
    return {"busy": not stale, "age_s": round(age, 1), "stale": stale}


def release_lock(path: Path | str = APPLY_LOCK) -> None:
    try:
        os.unlink(Path(path))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------

PERSONS_TTL_S = 20.0
PERSONS_TIMEOUT_S = 15.0


def interpreter(repo_root: Path | str = REPO_ROOT) -> str:
    """The interpreter `python -m shell` runs under, not this process's: the dashboard
    has its own venv with neither pyyaml nor numpy (dashboard/env_panel.py says the
    same thing for the same reason), so `sys.executable` would answer with a
    ModuleNotFoundError and the picker would list no personas."""
    venv = Path(repo_root) / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _run_print_personas(repo_root: Path) -> dict[str, Any]:
    """`python -m shell --print-personas`: resolves the personas with the repo venv and
    leaves. Hardware-free by construction — no device, no robot, no journal, no network."""
    proc = subprocess.run(
        [interpreter(repo_root), "-m", "shell", "--print-personas"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=PERSONS_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "print-personas failed").strip()[:400])
    return dict(json.loads(proc.stdout))


class PersonaResolver:
    """Cached `--print-personas` reader — one subprocess per TTL window, never per poll."""

    def __init__(
        self,
        repo_root: Path | str = REPO_ROOT,
        *,
        ttl: float = PERSONS_TTL_S,
        runner: Callable[[Path], dict[str, Any]] = _run_print_personas,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.ttl = ttl
        self.runner = runner
        self.clock = clock
        self._report: dict[str, Any] | None = None
        self._error: str | None = None
        self._at = -1.0

    def resolve(self, force: bool = False) -> dict[str, Any]:
        now = self.clock()
        if force or self._at < 0 or now - self._at >= self.ttl:
            self._at = now
            try:
                self._report, self._error = self.runner(self.repo_root), None
            except Exception as e:  # noqa: BLE001 - a panel must not take the page down
                self._report, self._error = None, f"{type(e).__name__}: {e}"
        return {"report": self._report, "error": self._error}


_RESOLVER = PersonaResolver()


def build_panel(
    resolver: PersonaResolver | None = None,
    *,
    current: dict[str, str] | None = None,
    pins_text: str | None = None,
    busy: dict[str, Any] | None = None,
    lock_path: Path | str = APPLY_LOCK,
    now: float | None = None,
) -> dict[str, Any]:
    """Everything the persona page renders, as plain JSON-ready data."""
    state = (resolver or _RESOLVER).resolve()
    report = state["report"] or {}
    in_force = read_persona_env() if current is None else dict(current)
    selected = str(report.get("persona") or "")
    active = active_persona(in_force)
    personas = report.get("personas") if isinstance(report.get("personas"), list) else []
    rows: list[dict[str, Any]] = []
    for persona in personas:
        if not isinstance(persona, dict):
            continue
        name = str(persona.get("name") or "")
        values = persona_env_values(persona)
        plan = restart_plan(in_force, values)
        voice = persona.get("voice") if isinstance(persona.get("voice"), dict) else {}
        rows.append(
            {
                "name": name,
                "selected_for_room": name == selected,
                "voice_in_force": name == active,
                "engine": str(voice.get("engine") or ""),
                "model": str(voice.get("model") or ""),
                "speaker": str(voice.get("speaker") or ""),
                "talker": Path(str(voice.get("talker_gguf") or "")).name,
                "ack_dir": str(persona.get("ack_dir") or ""),
                "instructions": str(persona.get("instructions") or "").strip(),
                "would_restart": plan["units"],
                "would_change": plan["changed"],
                "pins_warnings": pins_guard(persona, pins_text or ""),
                "voice_warnings": voice_warnings(persona),
            }
        )
    rows.sort(key=lambda row: row["name"])
    chosen = next((row for row in rows if row["name"] == selected), None)
    return {
        "available": bool(report),
        "error": state["error"],
        "env": str(report.get("env") or ""),
        "persona_selected": selected,
        "persona_active": active,
        "persona_source": str(report.get("persona_source") or ""),
        # the env file's inline instructions still win in both shipped rooms tonight
        "inline_shadow": bool(report.get("inline_instructions")),
        "personas": rows,
        "chosen": chosen,
        "in_force": in_force,
        "restart_seconds": S2S_RESTART_SECONDS,
        "busy": busy if busy is not None else apply_busy(lock_path, now=now),
        "persona_env_path": str(PERSONA_ENV.relative_to(REPO_ROOT)),
    }
