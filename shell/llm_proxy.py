"""Transcript-fixing proxy between s2s and llama.cpp.

Parakeet has never heard the name "Reachy" and writes it as Richie, Ricci,
Rachi, ... The LLM then either calls the user that name or keeps correcting
them ("I'm Reachy, not Richie"), and no system-prompt wording fixed that
reliably on the 4B model (2026-10-04: 6/12 replies still leaked it). s2s has
no transcript hook and we don't fork it (ADR 0001), so its LLM traffic goes
through here: user text gets the name rewritten, everything else - including
streamed responses - passes through byte for byte.

    s2s --responses_api_base_url http://127.0.0.1:8081/v1 -> here -> llama :8080

Each user message used to arrive wearing the local time it was first seen
("[Sunday, October 11, 2026, 5:38 PM] ..."). That prefix is gone (2026-10-13), for two
independent reasons, both about this one string:
  - the model started reading the stamps out inside its own replies, and a stamp is a
    transcript artefact nobody asked to hear; and
  - the cache behind it keyed on the message TEXT inside this long-lived process
    (deploy/systemd/reachy-llm-proxy.service), so a line repeated after an s2s session
    reset came back wearing the OLD time - 24.8 and 41.5 minutes stale in the journal
    (issue #45). A stamp that can be hours old is not a clock.
What the stamp was really for - stopping the 4B model guessing a date ("[Sun 11:40]"
bought "October 14, 2023" and a weather search for "May 2025", 2026-10-10/11) - is now
handled by the one source that reads the clock at the moment he answers: shell/config.py
TIME_NOTE says he has no clock and routes every time and date question to `get_time`.
The other job the stamp carried, "how long ago the guest spoke" (2026-10-05: a dangling
question answered 40 minutes later), was never the model's to measure from a cached
string; the shell starts a fresh session after `reset_after_s` of silence
(shell/conversation.py).

Both rewrites here are deterministic per message, so llama.cpp's prompt-prefix cache
still hits.
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from .intents import NAME_RE  # mis-hearings of "Reachy"

log = logging.getLogger("shell.llm_proxy")

UPSTREAM = os.getenv("LLM_PROXY_UPSTREAM", "http://127.0.0.1:8080")
# Debug aid: the last rewritten request body is written here (what the LLM saw).
DUMP_PATH = os.getenv("LLM_PROXY_DUMP", "data/llm_last_request.json")


def fix_name(text: str) -> str:
    return NAME_RE.sub("Reachy", text)


# s2s v1.0.0 wraps the session instructions in its own "voice channel" system
# prompt (LLM/voice_prompt.py): a lead paragraph, "Session Prompt:", ours, then
# "## Voice Rules" ("For expression/background tools, speak first", "If unsure
# whether a tool is needed, just speak", ...). Granite 4.1 3B then answers
# "Sure, here's my best nod." instead of calling nod and refuses go_to_sleep:
# 24/38 on the tool bench with the wrapper, 35/38 with only our instructions
# (2026-10-11). Our instructions already carry the voice rules we want.
VOICE_WRAP_RE = re.compile(
    r"\AYou are in a spoken conversation\..*?\nSession Prompt:\n(?P<session>.*?)\n+## Voice Rules\n.*\Z",
    re.DOTALL,
)


def unwrap_system(text: str) -> str:
    m = VOICE_WRAP_RE.match(text)
    return m.group("session").strip() if m else text


def rewrite_body(body: dict) -> dict:
    """Fix the name in user-authored text of a Responses or Chat request, and strip
    s2s's voice-channel wrapper from the system prompt.

    Nothing else in a request is touched - no time prefix, no added or reordered key -
    so the prompt bytes the model sees stay byte-stable for llama.cpp's prefix cache
    (proved against a captured live request in tests/test_llm_proxy.py).
    """
    items = body.get("input")
    if isinstance(body.get("instructions"), str):
        body["instructions"] = unwrap_system(body["instructions"])
    if isinstance(items, str):
        body["input"] = fix_name(items)
    elif isinstance(items, list):
        for item in items:
            _unwrap_message(item)
            _fix_message(item)
    for msg in body.get("messages") or []:  # chat-completions shape
        _unwrap_message(msg)
        _fix_message(msg)
    return body


def _unwrap_message(msg) -> None:
    if not isinstance(msg, dict) or msg.get("role") not in ("system", "developer"):
        return
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = unwrap_system(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part["text"] = unwrap_system(part["text"])


def _fix_message(msg) -> None:
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = fix_name(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part["text"] = fix_name(part["text"])


app = FastAPI()
_client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(300.0, connect=5.0))
_HOP = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    raw = await request.body()
    if raw and request.headers.get("content-type", "").startswith("application/json"):
        try:
            raw = json.dumps(rewrite_body(json.loads(raw))).encode()
            if DUMP_PATH:
                with open(DUMP_PATH, "wb") as f:
                    f.write(raw)
        except (ValueError, AttributeError):
            pass  # not our shape: forward untouched
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
    upstream = await _client.send(
        _client.build_request(request.method, "/" + path, content=raw, headers=headers, params=request.query_params),
        stream=True,
    )
    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP | {"content-encoding"}}
    if "text/event-stream" in upstream.headers.get("content-type", ""):

        async def relay():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(relay(), status_code=upstream.status_code, headers=out_headers)
    body = await upstream.aread()
    await upstream.aclose()
    return Response(body, status_code=upstream.status_code, headers=out_headers)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("LLM_PROXY_PORT", "8081")), log_level="warning")


if __name__ == "__main__":
    main()
