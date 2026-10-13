#!/usr/bin/env python3
"""Render the active persona's voice into deploy/generated/persona.env.

    uv run python scripts/apply_env.py                      # apply the room's persona
    uv run python scripts/apply_env.py --persona marvin     # apply a named one
    uv run python scripts/apply_env.py --check              # print what would change
    uv run python scripts/apply_env.py --persona obama --strict --restart
                                                            # the dashboard's Apply path

One persona is one thing: `personas/<name>.yaml` holds the spoken manner (which
shell/env.py resolves into the system prompt for the room that names it), the ack pool,
and the TTS voice. The voice half cannot be a config reload — the pinned s2s service
reads its `--qwen3_tts_*` flags once at startup — so this script writes the file both
services load and the units must be restarted (measured 20-46 s of dead air; that is
why the panel asks first).

Why a generated file at all: `.env` is one flat pile of overrides that outranks
`envs/<env>.yaml`, and the voice lived there while the persona lived here — the drift
docs/environments.md exists to kill. `persona.env` is listed in
reachy-s2s.service and reachy-shell.service AFTER `EnvironmentFile=-…/.env`, so the
persona wins over a stray `.env` voice line. That ordering was measured on this box
2026-10-12 with a throwaway user unit (deploy/README.md), not assumed: systemd applies
`EnvironmentFile=` after `Environment=`, and a later file beats an earlier one, key by
key, while keys only the earlier file sets survive.

What this script also checks: that the units on this box will read what it writes.
`deploy/README.md:8` copies `deploy/systemd/*.service` into `~/.config/systemd/user/` by
hand and nothing re-runs that copy, so an installed unit can predate the `persona.env`
line entirely - the render then changes the room's acks but never its talker, and the
panel still reports a successful apply. `voice_routing_gaps()` says so out loud.

What this script never writes: `REACHY_INSTRUCTIONS`. It is one global file read in
both rooms, so a stamped prompt would put the home persona — family assistant, full
tool list — onto the booth the moment someone forgot to move the room back
(docs/environments.md hard rule 1). The room keeps its words; the picker moves the
voice and the acknowledgements, and `--check` says out loud when the room's inline
`persona.instructions` is still outranking the named persona.

Exit codes: 0 applied (or nothing to do) · 2 the persona is broken · 3 another apply is
running · 4 --strict and the voice is unmeasured in docs/pins.yaml.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dashboard"))

import persona_panel  # noqa: E402  (stdlib-only pure core, shared with the dashboard)

from shell import env as shell_env  # noqa: E402

PINS_PATH = REPO_ROOT / "docs" / "pins.yaml"

# The directory systemd actually reads from (deploy/README.md:8, a manual `cp`). Read-only
# here: this script never writes or rewrites a unit, it only reports what it found.
UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
UNITS = ("reachy-s2s", "reachy-shell")
# Matched on the path tail, because the units spell the home directory as %h.
PERSONA_ENV_TOKEN = "dev/robot/deploy/generated/persona.env"
ENV_FILE_TOKEN = "dev/robot/.env"


def voice_routing_gaps(out: Path, unit_dir: Path | None = None) -> list[str]:
    """Why a rendered `persona.env` may never reach the process that reads a voice flag.

    `shell/env.py` LAYERS has no `persona.voice.*` key, so this file is the only route from
    a persona to the Qwen3 talker, speaker and style line. A unit that does not list it -
    or lists it before `.env`, which inverts the measured precedence - silently reduces the
    picker to an ack picker, which is the one user-visible promise the panel makes.
    """
    directory = UNIT_DIR if unit_dir is None else unit_dir
    gaps: list[str] = []
    for unit in UNITS:
        path = directory / f"{unit}.service"
        if not path.is_file():
            continue  # not installed on this box (CI, a fresh machine): nothing to claim
        try:
            text = path.read_text()
        except OSError:
            continue
        if PERSONA_ENV_TOKEN not in text:
            gaps.append(
                f"{path} never loads {out}, so the voice it renders reaches no service: "
                f"applying a persona here changes acks only, NOT the talker. Re-copy the "
                f"reference unit (cp deploy/systemd/{unit}.service {path}; systemctl --user "
                "daemon-reload), then restart - deploy/README.md:8."
            )
        elif ENV_FILE_TOKEN in text and text.index(PERSONA_ENV_TOKEN) < text.index(ENV_FILE_TOKEN):
            gaps.append(
                f"{path} loads persona.env before .env, inverting the measured order: a stray "
                f"S2S_QWEN3_* line in .env outranks this persona (deploy/systemd/{unit}.service "
                "has it the other way round)."
            )
    return gaps


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], prog="scripts/apply_env.py")
    parser.add_argument("--env", help="deployment environment (default: REACHY_ENV, then home)")
    parser.add_argument("--persona", help="personas/<name>.yaml (default: the one this room names)")
    parser.add_argument("--out", type=Path, default=persona_panel.PERSONA_ENV, help="file to render")
    parser.add_argument("--lock", type=Path, default=persona_panel.APPLY_LOCK, help="one-apply-at-a-time lock")
    parser.add_argument("--check", action="store_true", help="print the render and the plan, change nothing")
    parser.add_argument("--restart", action="store_true", help="restart reachy-s2s, then reachy-shell")
    parser.add_argument(
        "--strict", action="store_true", help="refuse (exit 4) when the voice has no docs/pins.yaml entry"
    )
    parser.add_argument(
        "--lock-held",
        action="store_true",
        help="the caller already created --lock (the dashboard does this before launching "
        "the child, so a second click cannot slip into the launch window); this process releases it",
    )
    return parser


def warn(text: str) -> None:
    print(f"WARN  {text}", file=sys.stderr)


def write_atomically(path: Path, body: str) -> Path | None:
    """Temp file + rename, keeping the previous render as `<name>.prev` to roll back to."""
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_text() if path.is_file() else None
    backup: Path | None = None
    if previous is not None and previous != body:
        backup = path.with_suffix(path.suffix + ".prev")
        temp = backup.with_suffix(".tmp")
        temp.write_text(previous)
        os.replace(temp, backup)
    temp = path.with_suffix(".tmp")
    temp.write_text(body)
    os.replace(temp, path)
    return backup


def restart(units: list[str]) -> int:
    """reachy-s2s first, then reachy-shell: the shell reconnects to whatever is up."""
    rc = 0
    for unit in units:
        started = time.monotonic()
        print(f"restart {unit} ...", flush=True)
        proc = subprocess.run(["systemctl", "--user", "restart", unit], capture_output=True, text=True, timeout=180)
        took = time.monotonic() - started
        if proc.returncode != 0:
            rc = rc or proc.returncode
            warn(f"{unit} restart failed (exit {proc.returncode}): {(proc.stderr or proc.stdout).strip()[:300]}")
        else:
            print(f"ok    {unit} restarted in {took:.1f} s")
    return rc


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out: Path = args.out
    try:
        report = shell_env.persona_report(args.env)
    except shell_env.EnvError as e:
        print(f"apply_env: {e}", file=sys.stderr)
        return 2

    name = (args.persona or report["persona"] or "").strip()
    if not name:
        print(
            f"apply_env: environment {report['env']!r} names no persona "
            f"(add `persona: <name>` to envs/{report['env']}.yaml; known: "
            f"{', '.join(shell_env.known_personas()) or 'none'})",
            file=sys.stderr,
        )
        return 2
    try:
        persona = shell_env.load_persona_doc(name)
    except shell_env.PersonaError as e:
        print(f"apply_env: {e}", file=sys.stderr)
        return 2

    values = persona_panel.persona_env_values(persona)
    current = persona_panel.read_persona_env(out)
    plan = persona_panel.restart_plan(current, values)
    pins_text = PINS_PATH.read_text() if PINS_PATH.is_file() else ""
    problems = persona_panel.pins_guard(persona, pins_text) + persona_panel.voice_warnings(persona)

    print(f"persona {name} (room {report['env']}, selected by {report['persona_source'] or 'nothing'}) -> {out}")
    for key, value in values.items():
        mark = " " if current.get(key, "") == value else "*"
        print(f"  {mark} {key}={value or ''}")
    if report["inline_instructions"]:
        warn(
            f"envs/{report['env']}.yaml still carries an inline persona.instructions: it "
            f"outranks personas/{name}.yaml, so the picker changes the voice and the acks, "
            "NOT the words. Delete that block to hand the words to the persona file."
        )
    for gap in voice_routing_gaps(out):
        warn(gap)
    for item in problems:
        warn(item)
    if problems and args.strict:
        print("apply_env: refused by --strict (measure the voice into docs/pins.yaml first)", file=sys.stderr)
        return 4
    if not plan["needed"]:
        print("unchanged: persona.env already matches; no restart needed")
        if args.restart:
            return 0
    else:
        print(
            f"needs restart: {', '.join(plan['units'])} (s2s reloads its voice at startup; measured {persona_panel.S2S_RESTART_SECONDS} s)"
        )

    if args.check:
        print("--- would write ---")
        print(persona_panel.render_persona_env(values), end="")
        return 0

    if not args.lock_held and not persona_panel.try_acquire_lock(args.lock):
        busy = persona_panel.apply_busy(args.lock)
        print(
            f"apply_env: another apply is already running ({busy['age_s']} s); "
            f"refusing a second concurrent apply. If that is stale, remove {args.lock}.",
            file=sys.stderr,
        )
        return 3
    try:
        backup = write_atomically(out, persona_panel.render_persona_env(values))
        print(f"wrote  {out}" + (f" (previous render kept as {backup})" if backup else ""))
        rc = 0
        if args.restart and plan["needed"]:
            rc = restart(plan["units"])
        elif plan["needed"]:
            print(f"not restarting (no --restart): systemctl --user restart {' then '.join(plan['units'])}")
        return rc
    finally:
        persona_panel.release_lock(args.lock)


if __name__ == "__main__":
    raise SystemExit(main())
