"""Reachy Mini Dashboard — FastAPI + HTMX + Tailwind."""

import asyncio
import json
import logging
import sqlite3
import subprocess
from pathlib import Path

from dotenv import dotenv_values
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dashboard")

ROBOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROBOT_DIR / ".env"
DB_PATH = ROBOT_DIR / "data" / "sessions.db"
FACES_PATH = ROBOT_DIR / "data" / "known_faces.json"

SERVICES = ["reachy-mini-daemon", "llama-server", "reachy-agent", "qwen3-tts"]

app = FastAPI(title="Reachy Mini Dashboard")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


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


# Register template filters
templates.env.filters["format_ts"] = format_timestamp
templates.env.filters["pretty_json"] = pretty_json


def render(request: Request, template: str, ctx: dict | None = None, **kwargs) -> HTMLResponse:
    """Shorthand for TemplateResponse with the new Starlette API."""
    context = ctx or {}
    context.update(kwargs)
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
    turns, session_id = get_recent_turns()
    return render(
        request,
        "conversation.html",
        turns=turns,
        session_id=session_id,
        active_page="conversation",
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
    return render(
        request,
        "faces.html",
        faces=get_known_faces(),
        active_page="faces",
    )


@app.get("/face_thumbnail/{name}")
async def face_thumbnail(name: str):
    """Serve a face thumbnail image."""
    from fastapi.responses import FileResponse

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


@app.get("/partials/conversation", response_class=HTMLResponse)
async def partial_conversation(request: Request):
    turns, session_id = get_recent_turns()
    return render(request, "partials/conversation.html", turns=turns, session_id=session_id)


@app.get("/partials/settings", response_class=HTMLResponse)
async def partial_settings(request: Request):
    return render(request, "partials/settings.html", settings=get_env_settings())


@app.post("/settings/volume", response_class=HTMLResponse)
async def set_volume(request: Request, volume_boost: str = Form(...)):
    """Set pipewire volume immediately without restarting anything."""
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
    whisper_model: str = Form(...),
):
    # Apply volume to pipewire immediately
    try:
        set_pipewire_volume(float(volume_boost))
    except (ValueError, subprocess.SubprocessError):
        pass
    write_env_setting("VOLUME_BOOST", volume_boost)
    write_env_setting("TTS_ENGINE", tts_engine)
    write_env_setting("KOKORO_VOICE", kokoro_voice)
    write_env_setting("WHISPER_MODEL", whisper_model)
    return render(request, "partials/settings.html", settings=get_env_settings(), saved=True)


@app.post("/settings/apply-restart", response_class=HTMLResponse)
async def apply_restart(
    request: Request,
    volume_boost: str = Form(...),
    tts_engine: str = Form(...),
    kokoro_voice: str = Form(...),
    whisper_model: str = Form(...),
):
    # Apply volume to pipewire immediately
    try:
        set_pipewire_volume(float(volume_boost))
    except (ValueError, subprocess.SubprocessError):
        pass
    write_env_setting("VOLUME_BOOST", volume_boost)
    write_env_setting("TTS_ENGINE", tts_engine)
    write_env_setting("KOKORO_VOICE", kokoro_voice)
    write_env_setting("WHISPER_MODEL", whisper_model)
    try:
        subprocess.run(
            ["systemctl", "--user", "restart", "reachy-agent"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as e:
        log.error("Restart reachy-agent failed: %s", e)
    await asyncio.sleep(1)
    return render(request, "partials/settings.html", settings=get_env_settings(), restarted=True)


@app.get("/partials/faces", response_class=HTMLResponse)
async def partial_faces(request: Request):
    return render(request, "partials/faces.html", faces=get_known_faces())


@app.delete("/faces/{name}", response_class=HTMLResponse)
async def delete_face(request: Request, name: str):
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


@app.post("/api/mic_toggle")
async def mic_toggle():
    try:
        subprocess.run(
            ["wpctl", "set-mute", "@DEFAULT_SOURCE@", "toggle"],
            capture_output=True,
            timeout=5,
        )
    except Exception as e:
        log.error("Mic toggle failed: %s", e)
    await asyncio.sleep(0.1)
    return {"muted": _get_mic_muted()}


@app.get("/partials/mic_button", response_class=HTMLResponse)
async def partial_mic_button(request: Request):
    return render(request, "partials/mic_button.html", mic_muted=_get_mic_muted())


# ---------------------------------------------------------------------------
# Routes — Agent sleep/wake control
# ---------------------------------------------------------------------------


def _get_agent_running() -> bool:
    """Check if the reachy-agent service is active."""
    status = get_service_status("reachy-agent")
    return status["running"]


@app.post("/api/agent_toggle")
async def agent_toggle():
    running = _get_agent_running()
    action = "stop" if running else "start"
    try:
        subprocess.run(
            ["systemctl", "--user", action, "reachy-agent"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as e:
        log.error("Agent %s failed: %s", action, e)
    await asyncio.sleep(0.5)
    return {"running": _get_agent_running()}


@app.get("/partials/agent_button", response_class=HTMLResponse)
async def partial_agent_button(request: Request):
    return render(request, "partials/agent_button.html", agent_running=_get_agent_running())


# ---------------------------------------------------------------------------
# SSE — Live conversation stream
# ---------------------------------------------------------------------------


@app.get("/sse/conversation")
async def sse_conversation(request: Request):
    """Push new conversation turns via Server-Sent Events."""

    async def event_generator():
        last_id = 0
        # Get initial max id
        if DB_PATH.exists():
            try:
                conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
                row = conn.execute("SELECT MAX(id) FROM turns").fetchone()
                if row and row[0]:
                    last_id = row[0]
                conn.close()
            except Exception:
                pass

        while True:
            if await request.is_disconnected():
                break
            if DB_PATH.exists():
                try:
                    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT id, session_id, timestamp, role, content, tool_calls, tool_call_id "
                        "FROM turns WHERE id > ? ORDER BY id ASC LIMIT 20",
                        (last_id,),
                    ).fetchall()
                    for r in rows:
                        last_id = r["id"]
                        tc = None
                        if r["tool_calls"]:
                            try:
                                tc = json.loads(r["tool_calls"])
                            except json.JSONDecodeError:
                                tc = r["tool_calls"]
                        turn = {
                            "id": r["id"],
                            "role": r["role"],
                            "content": r["content"],
                            "tool_calls": tc,
                            "tool_call_id": r["tool_call_id"],
                            "timestamp": r["timestamp"],
                        }
                        yield {"event": "new_turn", "data": json.dumps(turn)}
                    conn.close()
                except Exception as e:
                    log.error("SSE DB error: %s", e)
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3001, log_level="info")
