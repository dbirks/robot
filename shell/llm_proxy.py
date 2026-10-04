"""Transcript-fixing proxy between s2s and llama.cpp.

Parakeet has never heard the name "Reachy" and writes it as Richie, Ricci,
Rachi, ... The LLM then either calls the user that name or keeps correcting
them ("I'm Reachy, not Richie"), and no system-prompt wording fixed that
reliably on the 4B model (2026-10-04: 6/12 replies still leaked it). s2s has
no transcript hook and we don't fork it (ADR 0001), so its LLM traffic goes
through here: user text gets the name rewritten, everything else - including
streamed responses - passes through byte for byte.

    s2s --responses_api_base_url http://127.0.0.1:8081/v1 -> here -> llama :8080

Rewrites are deterministic, so llama.cpp's prompt-prefix cache still hits.
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

log = logging.getLogger("shell.llm_proxy")

UPSTREAM = os.getenv("LLM_PROXY_UPSTREAM", "http://127.0.0.1:8080")

# Mis-hearings of "Reachy" observed in the journal, plus near spellings.
NAME_RE = re.compile(
    r"\b(?:Richie|Richy|Ritchie|Ritchy|Richey|Ritchey|Ricci|Riche|Rachi|Rachie|Reechy|Reachie|Reachey|Reechie)\b",
    re.IGNORECASE,
)


def fix_name(text: str) -> str:
    return NAME_RE.sub("Reachy", text)


def rewrite_body(body: dict) -> dict:
    """Fix the name in user-authored text of a Responses or Chat request."""
    items = body.get("input")
    if isinstance(items, str):
        body["input"] = fix_name(items)
    elif isinstance(items, list):
        for item in items:
            _fix_message(item)
    for msg in body.get("messages") or []:  # chat-completions shape
        _fix_message(msg)
    return body


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
