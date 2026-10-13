"""Reachy Mini Dashboard — FastAPI + HTMX + Tailwind."""

import asyncio
import json
import logging
import os
import sqlite3
import subprocess
from pathlib import Path

from dotenv import dotenv_values
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

# `dashboard/server.py` runs two ways: `uvicorn dashboard.server:app` from the repo
# root (tests, ad-hoc) and `python server.py` with WorkingDirectory=dashboard (the
# systemd unit). The relative import is the first one; the flat one is the unit.
try:
    from . import env_panel, journal_read
except ImportError:  # pragma: no cover - systemd path
    import env_panel  # type: ignore[no-redef]
    import journal_read  # type: ignore[no-redef]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dashboard")

ROBOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROBOT_DIR / ".env"
DB_PATH = ROBOT_DIR / "data" / "sessions.db"
JOURNAL_FALLBACK = ROBOT_DIR / "data" / "events.db"
FACES_PATH = ROBOT_DIR / "data" / "known_faces.json"

# The controllable list is the units that exist and matter. `reachy-agent` and
# `qwen3-tts` are gone on purpose (issue #38): restarting a dead legacy row from a
# visitor's click would start a second mic/speaker owner and silence the real
# shell (ADR 0002). See dashboard/env_panel.py for the reasoning.
SERVICES = list(env_panel.SERVICE_UNITS)

app = FastAPI(title="Reachy Mini Dashboard")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
# Vendored assets (Tailwind play script, htmx, the SSE extension, both mono
# faces). This box has filtered egress and the booth has no guarantee of any, so
# a page that pulls from a CDN is a page that renders unstyled in front of a room.
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# ---------------------------------------------------------------------------
# Read-only gate (issue #38) — armed, not engaged
# ---------------------------------------------------------------------------

READONLY = env_panel.readonly_enabled()
# Nothing is allowed to write *while the gate is armed*: `allow=()` is the explicit
# statement that no endpoint has been proven safe to mutate from an unattended room.
# Default state is interactive (David's call for the booth), so this is currently a
# no-op — but the check stays in every handler rather than in a template, because
# hiding a button would leave `curl -X POST` working.
READONLY_ALLOW: tuple[str, ...] = ()

# Faces stay hidden at every privilege level (issue #38): `data/known_faces.json` and
# its thumbnails are family names and photos, and the interactive booth has hands.
# Nothing about them renders, and the routes answer 404 rather than an empty list,
# so "hidden" cannot be read as "no faces yet". Restore with FACES_VISIBLE = True at
# home; the template and helpers are kept for that.
FACES_VISIBLE = False


def refuse_mutation(request: Request) -> Response | None:
    """Return the refusal for a mutating request, or None to let it through.

    405 with an HTML body the page shows in a banner (`htmx:responseError` in
    base.html), so an armed gate refuses visibly instead of a control that appears
    to do nothing.
    """
    if not READONLY:
        return None
    verdict = journal_read.readonly_reject(request.method, request.url.path, allow=READONLY_ALLOW)
    if verdict is None:
        return None
    log.info("read-only refusal: %s %s", request.method, request.url.path)
    return HTMLResponse(
        render_refusal(request, request.url.path),
        status_code=verdict["status"],
        headers={"HX-Rewrite": "true"},
    )


def render_refusal(request: Request, path: str) -> str:
    return templates.get_template("partials/refused.html").render(
        request=request, path=path, readonly=READONLY, var=env_panel.READONLY_VAR
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_service_status(name: str) -> dict:
    """Get systemd user service status."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", "--property=ActiveState,SubState", name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        props = dict(line.split("=", 1) for line in result.stdout.strip().splitlines() if "=" in line)
        return {
            "name": name,
            "active": props.get("ActiveState", "unknown"),
            "sub": props.get("SubState", "unknown"),
            "running": props.get("ActiveState") == "active",
        }
    except Exception as e:
        return {"name": name, "active": "error", "sub": str(e), "running": False}


def get_all_services() -> list[dict]:
    return [get_service_status(s) for s in SERVICES]


def get_gpu_stats() -> dict | None:
    """Query nvidia-smi for GPU stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 5:
            return None
        mem_used, mem_total = int(parts[1]), int(parts[2])
        return {
            "name": parts[0],
            "mem_used": mem_used,
            "mem_total": mem_total,
            "mem_pct": round(mem_used / mem_total * 100, 1) if mem_total else 0,
            "util": int(parts[3]),
            "temp": int(parts[4]),
        }
    except Exception:
        return None


def get_pipewire_volume() -> str:
    """Read current pipewire sink volume."""
    try:
        result = subprocess.run(
            ["wpctl", "get-volume", "@DEFAULT_SINK@"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for part in result.stdout.strip().split():
            try:
                return str(float(part))
            except ValueError:
                continue
    except Exception:
        pass
    return "1.0"


def set_pipewire_volume(level: float):
    """Set pipewire default sink volume immediately."""
    subprocess.run(
        ["wpctl", "set-volume", "@DEFAULT_SINK@", f"{level:.2f}"],
        capture_output=True,
        timeout=5,
    )


def get_env_settings() -> dict:
    """Read settings from .env file."""
    vals = dotenv_values(ENV_PATH)
    return {
        "VOLUME_BOOST": get_pipewire_volume(),
        "TTS_ENGINE": vals.get("TTS_ENGINE", "kokoro"),
        "KOKORO_VOICE": vals.get("KOKORO_VOICE", "bm_daniel"),
        "WHISPER_MODEL": vals.get("WHISPER_MODEL", "small.en"),
    }


def write_env_setting(key: str, value: str):
    """Update a single key in the .env file, preserving other contents."""
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    found = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(new_lines) + "\n")


def journal_path() -> Path:
    """Where the live journal is, per the resolved env — not a hardcoded guess."""
    cfg = env_panel._RESOLVER.resolve()["config"] or {}
    rel = str(cfg.get("journal_path") or "").strip()
    if not rel:
        return JOURNAL_FALLBACK
    path = Path(rel)
    return path if path.is_absolute() else ROBOT_DIR / path


def live_turns(limit: int = 400) -> tuple[list[dict], str | None, dict]:
    """The conversation as the shell sees it: rows from the live event journal.

    The journal is the only conversation source, in every mode. `data/sessions.db`
    is the legacy agent's store - hours of home talk, last written 2026-07-28, and
    the shell never writes it - so rendering it first and letting the SSE stream
    overwrite it seconds later would both show family conversation on a public
    screen and swap the page's data source mid-demo (issue #38 review P1).
    A missing journal is reported as such, not as an empty chat.
    """
    path = journal_path()
    if not path.exists():
        return [], None, {"journal": str(path), "journal_error": "journal not found", "liveness": {}}
    try:
        conn = journal_read.open_ro(path)
    except Exception as e:  # noqa: BLE001 - the page must still render the failure
        log.error("cannot open %s: %s", path, e)
        return [], None, {"journal": str(path), "journal_error": str(e), "liveness": {}}
    try:
        events = journal_read.latest_events(conn, limit=limit, types=journal_read.CONVERSATION_TYPES)
        turns = journal_read.turns_from_events(events)[-100:]
        badge = journal_read.env_badge(journal_read.latest_events(conn, limit=20))
        return (
            turns,
            badge,
            {
                "journal": str(path),
                "journal_error": None,
                "liveness": journal_read.liveness(conn),
            },
        )
    finally:
        conn.close()


def get_recent_turns(limit: int = 100) -> tuple[list[dict], str | None]:
    """Get the most recent session's turns from SQLite (read-only).

    Tool-result turns (role="tool") are merged into the preceding assistant
    turn's tool_calls so the template can render call + result as one block.
    """
    if not DB_PATH.exists():
        return [], None
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Find most recent session
        row = conn.execute("SELECT session_id FROM turns ORDER BY timestamp DESC LIMIT 1").fetchone()
        if not row:
            conn.close()
            return [], None
        session_id = row["session_id"]
        rows = conn.execute(
            "SELECT id, session_id, timestamp, role, content, tool_calls, tool_call_id "
            "FROM turns WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        raw_turns = []
        for r in rows:
            tc = None
            if r["tool_calls"]:
                try:
                    tc = json.loads(r["tool_calls"])
                except json.JSONDecodeError:
                    tc = r["tool_calls"]
            raw_turns.append(
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "timestamp": r["timestamp"],
                    "role": r["role"],
                    "content": r["content"],
                    "tool_calls": tc,
                    "tool_call_id": r["tool_call_id"],
                }
            )
        conn.close()

        # Merge tool-result turns into their parent assistant turn
        turns = _merge_tool_results(raw_turns)
        return turns, session_id
    except Exception as e:
        log.error("Error reading sessions.db: %s", e)
        return [], None


def _merge_tool_results(raw_turns: list[dict]) -> list[dict]:
    """Attach tool-result content to the matching tool_call entry and drop standalone tool turns."""
    # Build a map: tool_call_id -> result content
    result_map: dict[str, str] = {}
    for t in raw_turns:
        if t["role"] == "tool" and t["tool_call_id"]:
            result_map[t["tool_call_id"]] = t["content"] or ""

    merged: list[dict] = []
    for t in raw_turns:
        if t["role"] == "tool":
            # Skip — already merged into the assistant turn
            continue
        if t["role"] == "assistant" and t["tool_calls"] and isinstance(t["tool_calls"], list):
            # Annotate each tool_call with its result
            for tc in t["tool_calls"]:
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    tc["_result"] = result_map.get(tc_id, None)
        merged.append(t)
    return merged


THUMB_DIR = ROBOT_DIR / "data" / "face_thumbnails"


def get_known_faces() -> list[dict]:
    """Get list of known faces with thumbnail info."""
    if not FACES_PATH.exists():
        return []
    try:
        data = json.loads(FACES_PATH.read_text())
        faces = []
        for name in sorted(data.keys()):
            has_thumb = (THUMB_DIR / f"{name}.jpg").exists()
            faces.append({"name": name, "has_thumbnail": has_thumb})
        return faces
    except Exception:
        return []


def delete_known_face(name: str) -> bool:
    """Delete a face from the known faces JSON."""
    if not FACES_PATH.exists():
        return False
    try:
        data = json.loads(FACES_PATH.read_text())
        if name in data:
            del data[name]
            FACES_PATH.write_text(json.dumps(data))
            return True
        return False
    except Exception:
        return False


def format_timestamp(ts: float) -> str:
    """Format a Unix timestamp to a human-readable string."""
    import datetime

    return datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def pretty_json(value: str) -> str:
    """Pretty-print a JSON string, or return as-is if not valid JSON."""
    try:
        parsed = json.loads(value)
        return json.dumps(parsed, indent=2)
    except (json.JSONDecodeError, TypeError):
        return value


def _html_escape(s: str) -> str:
    """Minimal HTML escaping."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def format_tool_result(value: str) -> str:
    """Format tool result JSON as HTML key-value pairs."""
    try:
        data = json.loads(value)
        if isinstance(data, dict):
            rows = []
            for k, v in data.items():
                key_html = f'<span class="font-bold text-black/60 min-w-[80px]">{_html_escape(str(k))}</span>'
                if k == "ok":
                    if v:
                        v_html = '<span class="text-green-700 font-bold">✓</span>'
                    else:
                        v_html = '<span class="text-red-700 font-bold">✗</span>'
                elif isinstance(v, bool):
                    color = "text-green-700" if v else "text-red-700"
                    v_html = f'<span class="{color}">{_html_escape(str(v).lower())}</span>'
                elif isinstance(v, (dict, list)):
                    v_str = json.dumps(v, indent=2)
                    v_html = f'<pre class="text-xs whitespace-pre-wrap break-all">{_html_escape(v_str)}</pre>'
                else:
                    v_html = f"<span>{_html_escape(str(v))}</span>"
                rows.append(f'<div class="flex gap-2">{key_html}{v_html}</div>')
            return "\n".join(rows)
    except (json.JSONDecodeError, TypeError):
        pass
    return _html_escape(value)


# Register template filters
templates.env.filters["format_ts"] = format_timestamp
templates.env.filters["pretty_json"] = pretty_json
templates.env.filters["format_result"] = format_tool_result


def render(request: Request, template: str, ctx: dict | None = None, **kwargs) -> HTMLResponse:
    """Shorthand for TemplateResponse with the new Starlette API."""
    context = ctx or {}
    context.update(kwargs)
    context.setdefault("readonly", READONLY)
    return templates.TemplateResponse(request, template, context=context)


# ---------------------------------------------------------------------------
# Routes — Full pages
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    """Redirect root to conversation page."""
    return RedirectResponse(url="/conversation")


@app.get("/conversation", response_class=HTMLResponse)
async def conversation_page(request: Request):
    turns, env, state = live_turns()
    return render(
        request,
        "conversation.html",
        turns=turns,
        session_id=state.get("session_id"),
        live_env=env,
        live=state,
        readonly=READONLY,
        active_page="conversation",
    )


@app.get("/environment", response_class=HTMLResponse)
async def environment_page(request: Request):
    return render(
        request,
        "environment.html",
        panel=env_panel.build_panel(),
        readonly=READONLY,
        active_page="environment",
    )


@app.get("/services", response_class=HTMLResponse)
async def services_page(request: Request):
    return render(
        request,
        "services.html",
        services=get_all_services(),
        gpu=get_gpu_stats(),
        active_page="services",
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return render(
        request,
        "settings.html",
        settings=get_env_settings(),
        active_page="settings",
    )


@app.get("/faces", response_class=HTMLResponse)
async def faces_page(request: Request):
    if not FACES_VISIBLE:
        return faces_hidden()
    return render(
        request,
        "faces.html",
        faces=get_known_faces(),
        active_page="faces",
    )


def faces_hidden() -> HTMLResponse:
    """404, not an empty list: hidden and "nobody has been seen yet" are different answers."""
    body = (
        "<div style='font-family:monospace;padding:1.5rem'>"
        "<b>FACES ARE HIDDEN IN THIS DEPLOYMENT</b>"
        "<div>known_faces.json and its thumbnails are personal data, so this panel does not "
        "serve them at any privilege level. Set FACES_VISIBLE = True in dashboard/server.py on "
        "the home machine.</div></div>"
    )
    return HTMLResponse(body, status_code=404)


@app.get("/face_thumbnail/{name}")
async def face_thumbnail(name: str):
    """Serve a face thumbnail image."""
    from fastapi.responses import FileResponse

    if not FACES_VISIBLE:
        return faces_hidden()
    path = THUMB_DIR / f"{name}.jpg"
    if path.exists():
        return FileResponse(path, media_type="image/jpeg")
    # Return a 1x1 transparent pixel as fallback
    from fastapi.responses import Response

    return Response(content=b"", status_code=404)


# ---------------------------------------------------------------------------
# Routes — HTMX partials
# ---------------------------------------------------------------------------


@app.get("/partials/services", response_class=HTMLResponse)
async def partial_services(request: Request):
    return render(request, "partials/services.html", services=get_all_services())


@app.post("/services/{name}/{action}", response_class=HTMLResponse)
async def service_action(request: Request, name: str, action: str):
    refused = refuse_mutation(request)
    if refused is not None:
        return refused
    if name not in SERVICES or action not in ("restart", "stop"):
        return HTMLResponse("Invalid", status_code=400)
    try:
        subprocess.run(["systemctl", "--user", action, name], capture_output=True, text=True, timeout=10)
    except Exception as e:
        log.error("Service %s %s failed: %s", action, name, e)
    # Small delay to let systemd state settle
    await asyncio.sleep(0.5)
    return render(request, "partials/services.html", services=get_all_services())


@app.get("/partials/gpu", response_class=HTMLResponse)
async def partial_gpu(request: Request):
    return render(request, "partials/gpu.html", gpu=get_gpu_stats())


@app.get("/partials/environment", response_class=HTMLResponse)
async def partial_environment(request: Request):
    return render(request, "partials/environment.html", panel=env_panel.build_panel(), readonly=READONLY)


@app.get("/partials/conversation", response_class=HTMLResponse)
async def partial_conversation(request: Request):
    turns, env, state = live_turns()
    return render(
        request,
        "partials/conversation.html",
        turns=turns,
        session_id=state.get("session_id"),
        live_env=env,
        live=state,
        readonly=READONLY,
    )


def state_blank() -> dict:
    return {"journal": None, "journal_error": None, "liveness": {}}


@app.get("/partials/settings", response_class=HTMLResponse)
async def partial_settings(request: Request):
    return render(request, "partials/settings.html", settings=get_env_settings())


@app.post("/settings/volume", response_class=HTMLResponse)
async def set_volume(request: Request, volume_boost: str = Form(...)):
    """Set pipewire volume immediately without restarting anything."""
    refused = refuse_mutation(request)
    if refused is not None:
        return refused
    try:
        level = float(volume_boost)
        level = max(0.0, min(5.0, level))
        set_pipewire_volume(level)
        write_env_setting("VOLUME_BOOST", volume_boost)
    except (ValueError, subprocess.SubprocessError) as e:
        log.error("Failed to set volume: %s", e)
    return HTMLResponse(volume_boost)


@app.post("/settings/save", response_class=HTMLResponse)
async def save_settings(
    request: Request,
    volume_boost: str = Form(...),
    tts_engine: str = Form(...),
    kokoro_voice: str = Form(...),
):
    refused = refuse_mutation(request)
    if refused is not None:
        return refused
    # Apply volume to pipewire immediately
    try:
        set_pipewire_volume(float(volume_boost))
    except (ValueError, subprocess.SubprocessError):
        pass
    write_env_setting("VOLUME_BOOST", volume_boost)
    write_env_setting("TTS_ENGINE", tts_engine)
    write_env_setting("KOKORO_VOICE", kokoro_voice)
    # There is no "apply & restart" route any more (issue #38): the only unit it ever
    # restarted was the retired reachy-agent, and starting it takes the audio devices
    # away from the live shell (ADR 0002). Volume is applied here and now; an
    # engine/voice change needs an operator to restart reachy-shell from Services.
    return render(
        request,
        "partials/settings.html",
        settings=get_env_settings(),
        saved=True,
        needs_restart=True,
    )


@app.get("/partials/faces", response_class=HTMLResponse)
async def partial_faces(request: Request):
    if not FACES_VISIBLE:
        return faces_hidden()
    return render(request, "partials/faces.html", faces=get_known_faces())


@app.delete("/faces/{name}", response_class=HTMLResponse)
async def delete_face(request: Request, name: str):
    if not FACES_VISIBLE:
        return faces_hidden()
    refused = refuse_mutation(request)
    if refused is not None:
        return refused
    deleted = delete_known_face(name)
    return render(request, "partials/faces.html", faces=get_known_faces(), deleted=name if deleted else None)


# ---------------------------------------------------------------------------
# Routes — Mic mute control
# ---------------------------------------------------------------------------


def _get_mic_muted() -> bool:
    """Check if the default source (mic) is muted via WirePlumber."""
    try:
        result = subprocess.run(
            ["wpctl", "get-volume", "@DEFAULT_SOURCE@"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "[MUTED]" in result.stdout
    except Exception:
        return False


@app.get("/api/mic_status")
async def mic_status():
    return {"muted": _get_mic_muted()}


@app.post("/api/mic_toggle", response_class=HTMLResponse)
async def mic_toggle(request: Request):
    refused = refuse_mutation(request)
    if refused is not None:
        return refused
    try:
        subprocess.run(
            ["wpctl", "set-mute", "@DEFAULT_SOURCE@", "toggle"],
            capture_output=True,
            timeout=5,
        )
    except Exception as e:
        log.error("Mic toggle failed: %s", e)
    await asyncio.sleep(0.1)
    return render(request, "partials/mic_button.html", mic_muted=_get_mic_muted())


@app.get("/partials/mic_button", response_class=HTMLResponse)
async def partial_mic_button(request: Request):
    return render(request, "partials/mic_button.html", mic_muted=_get_mic_muted())


# ---------------------------------------------------------------------------
# Routes — retired agent sleep/wake control (issue #38)
# ---------------------------------------------------------------------------


def retired_agent() -> HTMLResponse:
    """The sleep/wake control is retired, and it is not a read-only question.

    `reachy-agent` runs the legacy loop, which opens its own mic and speaker. With
    the new shell live, starting it from a kiosk button does not wake the robot — it
    takes the audio devices from the process that is actually talking (ADR 0002),
    which is the failure this list was supposed to be free of. Sleep is the shell's
    own business (`auto_sleep_s`, shell/sleep.py).
    """
    return HTMLResponse(
        "<div style='font-family:monospace;padding:1.5rem'>"
        "<b>SLEEP/WAKE CONTROL IS RETIRED</b><div>reachy-agent is the legacy loop; starting it "
        "opens a second audio owner and silences the running shell. Sleep is automatic "
        "(auto_sleep_s). Restart the real service from the Services panel as the operator.</div></div>",
        status_code=404,
    )


@app.post("/api/agent_toggle", response_class=HTMLResponse)
async def agent_toggle(request: Request):
    return retired_agent()


@app.get("/partials/agent_button", response_class=HTMLResponse)
async def partial_agent_button(request: Request):
    # Nothing polls it any more (base.html dropped the button); stay empty rather
    # than render a control that answers 404 when pressed.
    return HTMLResponse("")


# ---------------------------------------------------------------------------
# SSE — Live conversation stream
# ---------------------------------------------------------------------------


@app.get("/sse/conversation")
async def sse_conversation(request: Request):
    """Push the live conversation over Server-Sent Events.

    The payload is the rendered partial, not one turn at a time. The journal's
    transcript rows collapse into turns (`journal_read.collapse_finals`) and an
    assistant bubble can gain a tool block after it first appeared, so a stream of
    individual turns would either double-render or lose the merge. Re-rendering the
    window is one read-only query over a few hundred rows and cannot drift from the
    journal, which is the property that matters next to a room full of people.
    """

    async def event_generator():
        last_id = 0
        path = journal_path()
        if path.exists():
            try:
                conn = journal_read.open_ro(path)
                head = conn.execute("SELECT MAX(id) FROM events").fetchone()
                conn.close()
                last_id = int(head["id"] or 0) if head else 0
            except Exception as e:  # noqa: BLE001 - start from the top rather than lie
                log.error("SSE cursor init failed: %s", e)

        while True:
            if await request.is_disconnected():
                break
            path = journal_path()
            if path.exists():
                try:
                    conn = journal_read.open_ro(path)
                except Exception as e:  # noqa: BLE001
                    log.error("SSE open %s failed: %s", path, e)
                    conn = None
                if conn is not None:
                    try:
                        new = journal_read.events_after(conn, last_id, limit=500, types=journal_read.CONVERSATION_TYPES)
                        if new:
                            last_id = new[-1]["id"]
                            turns = journal_read.turns_from_events(
                                journal_read.latest_events(conn, limit=400, types=journal_read.CONVERSATION_TYPES)
                            )[-100:]
                            badge = journal_read.env_badge(journal_read.latest_events(conn, limit=20))
                            html = templates.get_template("partials/conversation_body.html").render(
                                request=request, turns=turns, live_env=badge, readonly=READONLY
                            )
                            yield {"event": "refresh", "data": html}
                    except Exception as e:  # noqa: BLE001
                        log.error("SSE journal error: %s", e)
                    finally:
                        conn.close()
            await asyncio.sleep(1.0)

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Loopback unless the operator says otherwise (2026-10-12). There is no auth
    # in this process: /faces serves data/known_faces.json (family names + their
    # thumbnails) and / renders data/sessions.db (hours of home conversation), so
    # binding 0.0.0.0 handed both to the whole LAN - including the hack-day booth.
    # The kiosk on this box reads http://127.0.0.1:3001, so nothing here changes
    # for it; opening the page to the room is a deliberate DASHBOARD_HOST=... and
    # a decision about what is in those files (deploy/README.md, "Changing rooms").
    host = os.environ.get("DASHBOARD_HOST", "").strip() or "127.0.0.1"
    uvicorn.run(app, host=host, port=3001, log_level="info")
