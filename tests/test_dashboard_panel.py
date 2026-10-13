"""The booth panel: what the dashboard says about the room, and what it may change.

Stdlib + the two dashboard modules only (`dashboard/server.py` needs fastapi, which
the minimal CI env does not have), so the server-side wiring is checked by reading
its AST instead of importing it — which is also the stricter test: it fails if a
mutating route is added without the gate, rather than passing because nobody
happened to call it.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import pytest

from dashboard import env_panel as ep
from dashboard import journal_read as jr
from shell.journal import Journal

ROOT = Path(__file__).resolve().parent.parent
SERVER_SRC = ROOT / "dashboard" / "server.py"


def test_imports_only_stdlib():
    import sys

    tree = ast.parse((ROOT / "dashboard" / "env_panel.py").read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    extra = mods - set(sys.stdlib_module_names)
    assert not extra, f"non-stdlib import in env_panel: {sorted(extra)}"


# ---------------------------------------------------------------------------
# The controllable list
# ---------------------------------------------------------------------------


def test_service_units_are_the_live_ones_and_no_legacy_audio_owner():
    assert ep.SERVICE_UNITS == (
        "reachy-mini-daemon",
        "llama-server",
        "reachy-llm-proxy",
        "reachy-s2s",
        "reachy-shell",
        "reachy-dashboard",
    )
    # A restart on a dead legacy row starts a second mic/speaker owner (ADR 0002),
    # which is how the real shell goes silent mid-demo.
    assert "reachy-agent" not in ep.SERVICE_UNITS
    assert not [u for u in ep.SERVICE_UNITS if "tts" in u.lower()]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),  # unset: the booth is interactive (David's call, 2026-10-12)
        ("", False),
        ("maybe", False),  # a typo must not freeze a kiosk's controls
        ("0", False),
        ("false", False),
        ("1", True),  # armed explicitly
        ("true", True),
        ("YES", True),
        (" on ", True),
    ],
)
def test_readonly_is_off_until_the_operator_arms_it(raw, expected):
    env = {} if raw is None else {ep.READONLY_VAR: raw}
    assert ep.readonly_enabled(env) is expected


# ---------------------------------------------------------------------------
# Tool facts read with ast, never by importing the robot
# ---------------------------------------------------------------------------


def robot_tools() -> list[str]:
    return ep._schema_names(ROOT, ep._TOOLS_SOURCE, ep._TOOLS_LIST)


def memory_tools() -> list[str]:
    return ep._schema_names(ROOT, ep._MEMORY_SOURCE, ep._MEMORY_LIST)


def test_tool_names_are_read_out_of_the_source_without_importing_it():
    robot = robot_tools()
    assert {"nod", "look_left", "set_volume", "go_to_sleep"} <= set(robot), robot
    assert {"remember", "recall", "forget_memory"} <= set(memory_tools())
    # The AST reader must not silently shrink: shell/tools.py's gate list has to
    # resolve against names that actually exist upstream.
    gated = set(ep._set_members(ROOT, ep._POLICY_SOURCE, ep._POLICY_SET))
    assert gated, "STATE_CHANGING no longer parses as a literal set"
    assert gated <= set(robot) | set(memory_tools()), sorted(gated - set(robot) - set(memory_tools()))


def test_profile_thresholds_parse():
    thresholds = ep._profile_thresholds(ROOT)
    assert thresholds and all(isinstance(v, float) for v in thresholds.values())


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def test_mask_memory_text_redacts_case_insensitively_and_longest_first():
    facts = ["Barney McCordsville", "Barney"]
    text = "Talk to Barney McCordsville about barney's dog."
    masked, hits = ep.mask_memory_text(text, facts)
    assert "Barney" not in masked and "barney" not in masked.lower()
    assert hits >= 2 and ep.MASK in masked


def test_mask_memory_text_leaves_clean_text_alone():
    masked, hits = ep.mask_memory_text("Speak plain words.", ["some unrelated fact"])
    assert masked == "Speak plain words." and hits == 0


def test_memory_facts_survives_missing_and_odd_files(tmp_path: Path):
    assert ep.memory_facts(None) == []
    assert ep.memory_facts(tmp_path / "nope.json") == []
    (tmp_path / "odd.json").write_text("not json")
    assert ep.memory_facts(tmp_path / "odd.json") == []
    (tmp_path / "ok.json").write_text(json.dumps([{"text": "loves trains"}, {"nope": 1}]))
    assert ep.memory_facts(tmp_path / "ok.json") == ["loves trains"]


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_print_env_is_resolved_once_per_ttl_not_per_request():
    calls = []

    def runner(root: Path) -> dict:
        calls.append(root)
        return {"env": "hackday"}

    clock = FakeClock()
    res = ep.EnvResolver(runner=runner, clock=clock, ttl=20.0)
    first = res.resolve()
    assert first["config"] == {"env": "hackday"} and first["error"] is None
    clock.t += 5
    res.resolve()
    res.resolve()
    assert len(calls) == 1, "the panel pays a subprocess per request"
    clock.t += 30
    res.resolve()
    assert len(calls) == 2 and first["age_s"] == 0.0


def test_a_failing_print_env_is_reported_not_hidden():
    def runner(_root: Path) -> dict:
        raise RuntimeError("envs/lab.yaml is missing")

    res = ep.EnvResolver(runner=runner, clock=FakeClock())
    state = res.resolve()
    assert state["config"] is None and "lab.yaml" in state["error"]


def test_a_forced_resolve_bypasses_the_cache():
    calls = []

    def runner(_root: Path) -> dict:
        calls.append(1)
        return {"env": "home"}

    res = ep.EnvResolver(runner=runner, clock=FakeClock())
    res.resolve()
    res.resolve(force=True)
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# The panel body
# ---------------------------------------------------------------------------


def fake_config(tmp_path: Path, **over: object) -> dict:
    cfg = {
        "env": "hackday",
        "attention_profile": "event",
        "instructions": "You are Reachy. Never mention anyone's family.",
        "tools_include": [],
        "tools_exclude": ["go_to_sleep"],
        "learn_facts": False,
        "memory_namespace": "hackday",
        "memories_path": str(tmp_path / "memories.hackday.json"),
        "journal_path": str(tmp_path / "events.db"),
        "s2s_url": "ws://127.0.0.1:8765/v1/realone",
        "llm_base_url": "http://localhost:8080/v1",
    }
    cfg.update(over)
    return cfg


def build(tmp_path: Path, cfg: dict) -> dict:
    clock = FakeClock()
    res = ep.EnvResolver(runner=lambda _root: cfg, clock=clock)
    return ep.build_panel(res, repo_root=ROOT)


def test_panel_reports_the_resolved_room_and_the_tool_subset(tmp_path: Path):
    panel = build(tmp_path, fake_config(tmp_path))
    assert panel["available"] is True and panel["error"] is None
    assert panel["env"] == "hackday" and panel["attention_profile"] == "event"
    assert panel["tool_confidence"] == pytest.approx(ep._profile_thresholds(ROOT)["event"])
    flags = {t["name"]: t for t in panel["tools"]}
    assert flags["go_to_sleep"]["excluded"] is True
    assert flags["go_to_sleep"]["state_changing"] is True and flags["go_to_sleep"]["attention_gated"] is True
    assert flags["nod"]["excluded"] is False and flags["nod"]["state_changing"] is False
    assert flags["remember"]["state_changing"] is True
    assert panel["tools_offered"] + panel["tools_excluded"] == len(panel["tools"])


def test_panel_honours_an_include_list_as_a_allowlist(tmp_path: Path):
    panel = build(tmp_path, fake_config(tmp_path, tools_include=["nod", "remember"], tools_exclude=["remember"]))
    flags = {t["name"]: t for t in panel["tools"]}
    assert flags["nod"]["offered"] is True
    assert flags["remember"]["excluded"] is True, "exclude must always win (shell/tools.py policy)"
    assert flags["look_left"]["excluded"] is True, "a non-empty include is an allowlist"


def test_the_panel_never_prints_a_fact_even_when_the_persona_carries_it(tmp_path: Path):
    store = tmp_path / "memories.hackday.json"
    store.write_text(json.dumps([{"text": "Forrest loves trains"}]))
    cfg = fake_config(tmp_path, instructions="Your friend Forrest loves trains. Say hi.")
    out = json.dumps(build(tmp_path, cfg))
    assert "Forrest loves trains" not in out
    assert ep.MASK in out


def test_an_unresolved_env_renders_an_error_not_an_empty_room(tmp_path: Path):
    res = ep.EnvResolver(runner=lambda _root: (_ for _ in ()).throw(RuntimeError("boom")), clock=FakeClock())
    panel = ep.build_panel(res, repo_root=ROOT, read_facts=lambda _p: [])
    assert panel["available"] is False and "boom" in panel["error"]
    assert panel["env"] == "" and panel["tools_offered"] == 0


def test_build_panel_uses_the_injected_reader_for_facts(tmp_path: Path):
    seen = []

    def reader(path):
        seen.append(path)
        return []

    res = ep.EnvResolver(runner=lambda _root: fake_config(tmp_path), clock=FakeClock())
    ep.build_panel(res, repo_root=ROOT, read_facts=reader)
    assert seen == [str(tmp_path / "memories.hackday.json")]


def test_the_panel_asks_the_shells_own_interpreter_not_the_dashboards():
    # dashboard/.venv has neither numpy nor pyyaml; `sys.executable` there cannot
    # import shell.config, so the panel would report an empty environment.
    assert env_interpreter() == str(ROOT / ".venv" / "bin" / "python")


def env_interpreter() -> str:
    return ep.interpreter(ROOT)


def test_the_real_resolved_room_renders(tmp_path: Path):
    """Slow-ish (one subprocess) but hardware-free: this is the proof that tomorrow
    the panel shows the room instead of an error box."""
    panel = ep.build_panel(ep.EnvResolver(), repo_root=ROOT)
    assert panel["available"] is True, panel["error"]
    assert panel["env"] in {"home", "hackday"}
    assert panel["persona"] and len(panel["tools"]) > 10
    facts = ep.memory_facts(ROOT / panel["memories_path"]) if panel["memories_path"] else []
    for fact in facts:
        assert fact not in panel["persona"], "a fact reached the persona unmasked"


# ---------------------------------------------------------------------------
# The live window must not drown in watchdog rows
# ---------------------------------------------------------------------------


def seed(tmp_path: Path, rows: list[tuple[str, dict]]) -> Path:
    db = tmp_path / "events.db"
    j = Journal(db, env="hackday")
    for type_, payload in rows:
        j.write(type_, **payload)
    time.sleep(0.01)
    return db


def test_a_health_flood_still_leaves_a_readable_conversation(tmp_path: Path):
    rows = [("audio.health", {"mic": "x"}) for _ in range(500)]
    rows += [
        (jr.TURN_TRANSCRIPT_FINAL, {"text": "hello"}),
        (jr.RESPONSE_TRANSCRIPT_FINAL, {"text": "hi there", "response_id": "r1"}),
    ]
    db = seed(tmp_path, rows)
    conn = jr.open_ro(db)
    window = jr.latest_events(conn, limit=50, types=jr.CONVERSATION_TYPES)
    assert [e["type"] for e in window] == [jr.TURN_TRANSCRIPT_FINAL, jr.RESPONSE_TRANSCRIPT_FINAL]
    assert [(t["role"], t["content"]) for t in jr.turns_from_events(window)] == [
        ("user", "hello"),
        ("assistant", "hi there"),
    ]
    conn.close()


# ---------------------------------------------------------------------------
# Server wiring, checked statically because importing server.py needs fastapi
# ---------------------------------------------------------------------------


def server_tree() -> ast.Module:
    return ast.parse(SERVER_SRC.read_text())


def mutating_handlers() -> list[ast.AsyncFunctionDef | ast.FunctionDef]:
    out: list[ast.AsyncFunctionDef | ast.FunctionDef] = []
    for node in server_tree().body:
        if not isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else None
            attr = getattr(call.func, "attr", "") if call is not None else ""
            if attr in {"post", "put", "patch", "delete"}:
                out.append(node)
    return out


def test_every_mutating_route_is_gated_or_retired():
    handlers = mutating_handlers()
    names = sorted(n.name for n in handlers)
    assert len(handlers) >= 5, f"expected the mutating routes, found {names}"
    unguarded = [
        n.name
        for n in handlers
        if "refuse_mutation" not in ast.unparse(n)
        and "retired_agent" not in ast.unparse(n)
        and "faces_hidden" not in ast.unparse(n)
    ]
    assert not unguarded, f"mutating routes with neither a gate nor a retirement: {unguarded}"
    # The two retired controls are the legacy-audio-owner and family-data ones:
    # they answer 404 at every privilege level, not only when the gate is armed.
    bodies = {n.name: ast.unparse(n) for n in handlers}
    assert "retired_agent()" in bodies["agent_toggle"]
    assert "faces_hidden()" in bodies["delete_face"]


def test_the_dashboard_never_starts_a_second_audio_owner():
    src = SERVER_SRC.read_text()
    # reachy-agent runs the legacy loop (its own mic + speaker) and qwen3-tts is the
    # legacy voice unit. No systemctl call in this process may name either one, so no
    # route, at any privilege level, can boot a second audio owner (ADR 0002).
    assert _systemctl_units() == set(), _systemctl_units()
    assert "apply-restart" not in src, "the apply-and-restart route is back"
    assert "SERVICES = list(env_panel.SERVICE_UNITS)" in src, "the controllable list was re-hardcoded"


def _systemctl_units() -> set[str]:
    """Unit names spelled as literals in a mutating `systemctl --user <verb> <unit>`."""
    units = set()
    for node in ast.walk(server_tree()):
        if not isinstance(node, ast.List):
            continue
        strs = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if strs[:2] != ["systemctl", "--user"] or strs[2:3] not in (["restart"], ["stop"], ["start"]):
            continue
        last = node.elts[-1]
        if isinstance(last, ast.Constant) and isinstance(last.value, str):
            units.add(last.value)
    return units


def test_faces_are_hidden_at_every_privilege_level():
    src = SERVER_SRC.read_text()
    assert "FACES_VISIBLE = False" in src
    bodies = {n.name: ast.unparse(n) for n in server_tree().body if isinstance(n, ast.AsyncFunctionDef)}
    for route in ("faces_page", "partial_faces", "face_thumbnail", "delete_face"):
        assert "faces_hidden()" in bodies[route], route
    # and nothing links to them
    nav = (ROOT / "dashboard" / "templates" / "base.html").read_text()
    assert "/faces" not in nav and "faces.html" not in nav
    assert "agent_button" not in nav, "the retired sleep/wake button is back in the nav"


def test_the_environment_is_a_readout_not_a_control():
    """home vs hackday changes via .env / the process environment, never this page."""
    src = SERVER_SRC.read_text()
    assert 'write_env_setting("REACHY_ENV"' not in src
    assert '"REACHY_ENV"' not in src.split("def refuse_mutation")[0], "the dashboard resolved the room itself"
    for rel in ("dashboard/templates/environment.html", "dashboard/templates/partials/environment.html"):
        text = (ROOT / rel).read_text()
        for control in ("<form", "<select", "<input", "<button", "hx-post", "hx-confirm"):
            assert control not in text, f"{rel} offers {control}"
    # the resolved room is shown, from the cached resolve
    assert "panel.env" in (ROOT / "dashboard/templates/partials/environment.html").read_text()


def test_conversation_routes_read_the_live_journal():
    src = SERVER_SRC.read_text()
    tree = server_tree()
    bodies = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.AsyncFunctionDef)}
    for route in ("conversation_page", "partial_conversation"):
        assert "live_turns()" in bodies[route], route
    # The legacy home archive is never rendered on the conversation page, in any
    # mode - read-only or not. It holds 3552 turns of family talk that the shell
    # never writes, so the first cut (render the archive when read-only is off,
    # let the SSE stream overwrite it seconds later) put private conversation on a
    # public screen and swapped the page's data source mid-demo (#38 review P1).
    for route in ("conversation_page", "partial_conversation"):
        assert "get_recent_turns()" not in bodies[route], route
    assert "DB_PATH = ROBOT_DIR / " in src and "sessions.db" in src, "the legacy reader is gone, not gated"


def test_sse_streams_from_the_journal_not_the_archive():
    bodies = {n.name: ast.unparse(n) for n in server_tree().body if isinstance(n, ast.AsyncFunctionDef)}
    gen = bodies["sse_conversation"]
    assert "journal_read.open_ro" in gen and "journal_read.events_after" in gen
    assert "sessions.db" not in gen and "FROM turns" not in gen


def test_the_read_only_gate_is_armable_but_not_engaged():
    src = SERVER_SRC.read_text()
    assert "env_panel.readonly_enabled()" in src, "the server stopped asking the pure function"
    assert "READONLY_ALLOW: tuple[str, ...] = ()" in src
    assert "if not READONLY:" in src, "refuse_mutation no longer short-circuits when the gate is off"
    assert "journal_read.readonly_reject(" in src, "the gate stopped using the pure decision function"


def test_a_refusal_is_returned_as_an_html_405():
    src = SERVER_SRC.read_text()
    assert "status_code=verdict[" in src and "readonly_reject(" in src


def test_the_kiosk_cannot_outrank_the_room_ceiling():
    """The page clamps to the same ceiling the tool path uses, read from one module.
    Two copies of this number is how a visitor with a slider got louder than the
    operator decided (2026-10-12)."""
    from dashboard.env_panel import VOLUME_MAX_DEFAULT, clamp_volume
    from shell.volume import MAX_GAIN

    assert clamp_volume(9.0) == VOLUME_MAX_DEFAULT
    assert clamp_volume(-1) == 0.0
    assert clamp_volume(0.85) == 0.85
    assert clamp_volume(float("nan")) == 0.0
    assert VOLUME_MAX_DEFAULT == MAX_GAIN, "the page and the tool must read one ceiling, not two copies"
    assert MAX_GAIN == 2.0, "headroom to 200% was measured as the crowd requirement; above it the card clips"


def test_settings_shows_the_voice_that_is_actually_loaded():
    from dashboard.env_panel import voice_in_force

    dotenv = {
        "S2S_TTS": "qwen3",
        "S2S_QWEN3_TALKER_GGUF": "/x/qwen-talker-1.7b-obama-Q8_0.gguf",
        "S2S_QWEN3_SPEAKER": "laxmikant",
        "TTS_ENGINE": "kokoro",
        "KOKORO_VOICE": "bm_daniel",
    }
    view = voice_in_force(dotenv, {})
    assert view["engine"] == "qwen3" and "obama" in view["voice"], view
    assert "TTS_ENGINE" in view["legacy"], "the legacy key must be named, not silently dropped"
    # the persona file outranks .env, exactly as systemd orders the EnvironmentFiles
    forced = voice_in_force(
        dotenv, {"S2S_QWEN3_TALKER_GGUF": "/x/kokoro-is-not-this.bin", "S2S_TTS": "kokoro", "KOKORO_VOICE": "bm_george"}
    )
    assert forced["engine"] == "kokoro" and forced["voice"] == "bm_george" and forced["source"] == "persona.env"
