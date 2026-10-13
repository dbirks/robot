"""Pure glue for the booth panel: what room is this, and what may the dashboard change?

Stdlib only, like `dashboard/journal_read.py` (CI imports the tests without
fastapi). Nothing here opens an audio device, connects to a robot, or writes a
file; the resolved environment comes from the documented hardware-free path
(`python -m shell --print-env`), and the tool/attention facts are read out of the
source with `ast` rather than by importing the shell.

The panel exists because the booth operator needs to see which room the box
resolved to before he says anything to it (docs/environments.md hard rule 1): a
`REACHY_ENV` typo boots the home persona with the family memories attached, and
until this panel the only way to notice was to listen for it.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent

# The units that exist and matter (issue #38). `reachy-agent` and `qwen3-tts` are
# deliberately absent: the legacy loop is being deleted, and a visitor who clicks
# restart on a dead legacy row starts a *second* mic/speaker owner, which silences
# the real shell (ADR 0002). Nothing controllable here may own audio by surprise.
SERVICE_UNITS: tuple[str, ...] = (
    "reachy-mini-daemon",
    "llama-server",
    "reachy-llm-proxy",
    "reachy-s2s",
    "reachy-shell",
    "reachy-dashboard",
)

# Interactive by default (David's call, 2026-10-12 evening): the booth audience is meant
# to drive the robot, so the controls work. The gate stays in the code because arming it
# is one line — `REACHY_DASHBOARD_READONLY=1` — and this process has no auth, so a room
# that ever serves this page beyond the kiosk wants it on. Same shape as DASHBOARD_HOST.
READONLY_VAR = "REACHY_DASHBOARD_READONLY"
TRUTHY = {"1", "true", "yes", "on"}

# `--print-env` resolves config from disk; that is milliseconds of work but it is
# still a subprocess, so the panel caches it instead of paying per request.
ENV_TTL_S = 20.0
ENV_TIMEOUT_S = 15.0
MASK = "[masked-memory]"

_TOOLS_SOURCE = "app/robot_tools.py"
_TOOLS_LIST = "TOOLS"
_MEMORY_SOURCE = "shell/memories.py"
_MEMORY_LIST = "MEMORY_TOOLS"
_POLICY_SOURCE = "shell/tools.py"
_POLICY_SET = "STATE_CHANGING"
_PROFILES_SOURCE = "shell/attention/profiles.py"


def readonly_enabled(env: dict[str, str] | None = None) -> bool:
    """Read-only is OFF unless the operator explicitly arms it, in this process only.

    Only an unambiguous truthy value arms it: unset, blank and typos leave the
    dashboard interactive, because a control that silently freezes on a kiosk is a
    worse failure than one that visibly refuses — the refusal is a banner, a stuck
    slider is a shrug and a demo that stops mid-sentence.
    """
    source = os.environ if env is None else env
    raw = str(source.get(READONLY_VAR, "")).strip().lower()
    return raw in TRUTHY


def _assign_node(tree: ast.Module, name: str) -> ast.expr | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.value
    return None


def _parse(repo_root: Path, rel: str) -> ast.Module | None:
    try:
        return ast.parse((repo_root / rel).read_text())
    except (OSError, SyntaxError, ValueError):
        return None


def _schema_names(repo_root: Path, rel: str, var: str) -> list[str]:
    """Function names of a module-level list of OpenAI tool schemas, read with
    `ast` so listing the tools never imports the robot, the audio stack or `.env`."""
    tree = _parse(repo_root, rel)
    value = _assign_node(tree, var) if tree is not None else None
    names: list[str] = []
    if not isinstance(value, ast.List):
        return names
    for item in value.elts:
        try:
            node = ast.literal_eval(item)
        except (ValueError, TypeError, SyntaxError):
            node = None
        if isinstance(node, dict):
            name = str(node.get("function", {}).get("name", ""))
            if name:
                names.append(name)
    return names


def _set_members(repo_root: Path, rel: str, var: str) -> list[str]:
    tree = _parse(repo_root, rel)
    value = _assign_node(tree, var) if tree is not None else None
    if not isinstance(value, (ast.Set, ast.List, ast.Tuple)):
        return []
    return sorted({str(elt.value) for elt in value.elts if isinstance(elt, ast.Constant)})


def _profile_thresholds(repo_root: Path) -> dict[str, float]:
    """`AttentionProfile.tool_confidence` per profile name, from the PROFILES literal."""
    tree = _parse(repo_root, _PROFILES_SOURCE)
    value = _assign_node(tree, "PROFILES") if tree is not None else None
    out: dict[str, float] = {}
    if not isinstance(value, ast.Dict):
        return out
    for key, val in zip(value.keys, value.values):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        if not isinstance(val, ast.Call):
            continue
        for kw in val.keywords:
            if kw.arg == "tool_confidence" and isinstance(kw.value, ast.Constant):
                try:
                    out[key.value] = float(kw.value.value)
                except (TypeError, ValueError):
                    pass
    return out


def memory_facts(memories_path: Path | str | None) -> list[str]:
    """The stored fact strings, for masking only. Never rendered by the panel."""
    if not memories_path:
        return []
    try:
        data = json.loads(Path(memories_path).read_text())
    except (OSError, ValueError):
        return []
    rows = data if isinstance(data, list) else data.get("memories", []) if isinstance(data, dict) else []
    return [str(m.get("text")) for m in rows if isinstance(m, dict) and str(m.get("text") or "").strip()]


def mask_memory_text(text: str, facts: list[str]) -> tuple[str, int]:
    """Redact any memory content out of a string before it reaches the browser.

    The persona from `--print-env` should carry no facts, but an env file can
    interpolate them (and tomorrow's persona edit might), and a fact shown in a
    public room is exactly the leak this panel is supposed to catch. Longest
    first, so "Barney McCordsville" cannot survive as "McCordsville".
    """
    out = str(text or "")
    hits = 0
    for fact in sorted({f for f in facts if f.strip()}, key=len, reverse=True):
        for needle in {fact, fact.rstrip(".!?").strip()} - {""}:
            if needle.lower() in out.lower():
                hits += 1
                out = _replace_ci(out, needle)
    return out, hits


def _replace_ci(text: str, needle: str) -> str:
    out, start = [], 0
    low, hay = needle.lower(), text.lower()
    while True:
        at = hay.find(low, start)
        if at < 0:
            out.append(text[start:])
            return "".join(out)
        out.append(text[start:at])
        out.append(MASK)
        start = at + len(needle)


def interpreter(repo_root: Path | str = REPO_ROOT) -> str:
    """The interpreter `python -m shell` actually runs under (the shell unit's
    `.venv/bin/python`), not this process's: the dashboard has its own venv with
    neither numpy nor pyyaml, so resolving the room with `sys.executable` fails
    with a ModuleNotFoundError and the panel would report an empty environment."""
    venv = Path(repo_root) / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _run_print_env(repo_root: Path) -> dict[str, Any]:
    """`python -m shell --print-env`: resolves config and leaves. It is the
    documented hardware-free path (no device, no robot, no journal, no network)."""
    proc = subprocess.run(
        [interpreter(repo_root), "-m", "shell", "--print-env"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=ENV_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "print-env failed").strip()[:400])
    return dict(json.loads(proc.stdout))


class EnvResolver:
    """Cached `--print-env` reader. One subprocess per TTL window, never per request."""

    def __init__(
        self,
        repo_root: Path | str = REPO_ROOT,
        *,
        ttl: float = ENV_TTL_S,
        runner: Callable[[Path], dict[str, Any]] = _run_print_env,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.ttl = ttl
        self.runner = runner
        self.clock = clock
        self._config: dict[str, Any] | None = None
        self._error: str | None = None
        self._at = -1.0
        self._age = 0.0

    def resolve(self, force: bool = False) -> dict[str, Any]:
        now = self.clock()
        if force or self._age <= 0 or now - self._at >= self.ttl:
            self._at = now
            try:
                self._config, self._error = self.runner(self.repo_root), None
            except Exception as e:  # noqa: BLE001 - a panel must not take the page down
                self._config, self._error = None, f"{type(e).__name__}: {e}"
            self._age = now
        return {
            "config": self._config,
            "error": self._error,
            "age_s": max(0.0, now - self._at),
            "ttl_s": self.ttl,
        }


_RESOLVER = EnvResolver()


def build_panel(
    resolver: EnvResolver | None = None,
    repo_root: Path | str = REPO_ROOT,
    *,
    read_facts: Callable[[Path | str | None], list[str]] = memory_facts,
) -> dict[str, Any]:
    """Everything the environment panel renders, as plain JSON-ready data."""
    root = Path(repo_root)
    state = (resolver or _RESOLVER).resolve()
    cfg = state["config"] or {}
    facts = read_facts(cfg.get("memories_path"))
    persona, masked = mask_memory_text(str(cfg.get("instructions") or ""), facts)
    profile = str(cfg.get("attention_profile") or "")
    include = [str(n) for n in cfg.get("tools_include") or []]
    exclude = {str(n) for n in cfg.get("tools_exclude") or []}
    resolved = bool(state["config"])
    gated = set(_set_members(root, _POLICY_SOURCE, _POLICY_SET))
    tools = []
    for source, names in (
        (_TOOLS_SOURCE, _schema_names(root, _TOOLS_SOURCE, _TOOLS_LIST)),
        (_MEMORY_SOURCE, _schema_names(root, _MEMORY_SOURCE, _MEMORY_LIST)),
    ):
        for name in names:
            # Same rule as shell/tools.apply_tool_policy: empty include keeps
            # everything, exclude always wins. With no resolved config nothing is
            # known, so nothing is claimed — an empty cell beats a green tick.
            offered = resolved and (not include or name in include) and name not in exclude
            tools.append(
                {
                    "name": name,
                    "source": source,
                    "offered": offered,
                    "excluded": resolved and not offered,
                    "state_changing": name in gated,
                    "attention_gated": name in gated,
                }
            )
    tools.sort(key=lambda t: t["name"])
    return {
        "available": bool(state["config"]),
        "error": state["error"],
        "age_s": round(state["age_s"], 1),
        "env": cfg.get("env") or "",
        "attention_profile": profile,
        "tool_confidence": _profile_thresholds(root).get(profile),
        "persona": persona,
        "persona_masked": masked,
        "tools": tools,
        "tools_offered": sum(1 for t in tools if t["offered"]),
        "tools_excluded": sum(1 for t in tools if t["excluded"]),
        "learn_facts": cfg.get("learn_facts"),
        "memory_namespace": cfg.get("memory_namespace") or "",
        "memories_path": cfg.get("memories_path") or "",
        "memory_count": len(facts),
        "journal_path": cfg.get("journal_path") or "",
        "s2s_url": cfg.get("s2s_url") or "",
        "llm_base_url": cfg.get("llm_base_url") or "",
        "services": list(SERVICE_UNITS),
    }
