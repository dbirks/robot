"""Transcript-fixing proxy between s2s and llama.cpp.

Parakeet has never heard the name "Reachy" and writes it as Richie, Ricci,
Rachi, ... The LLM then either calls the user that name or keeps correcting
them ("I'm Reachy, not Richie"), and no system-prompt wording fixed that
reliably on the 4B model (2026-10-04: 6/12 replies still leaked it). s2s has
no transcript hook and we don't fork it (ADR 0001), so its LLM traffic goes
through here: user text gets the name rewritten, everything else - including
streamed responses - passes through byte for byte.

    s2s --responses_api_base_url http://127.0.0.1:8081/v1 -> here -> llama :8080

It also stamps each user message with the local time it was first seen
("[Mon 17:38] ..."), so the model can tell a fresh question from one asked
hours ago (2026-10-05: it answered a 40-minute-old dangling question).

Both rewrites are deterministic per message, so llama.cpp's prompt-prefix
cache still hits.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import OrderedDict

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


STAMP_RE = re.compile(r"^\[[A-Z][a-z]{2} \d{2}:\d{2}\] ")


class Timestamper:
    """Remembers when each user message was first seen. s2s re-sends the whole
    history every request, so the k-th occurrence of a given text keeps its
    original time and the rendered prompt never changes for old turns."""

    def __init__(self, clock=time.time, max_texts: int = 2000) -> None:
        self.clock = clock
        self.max_texts = max_texts
        self._seen: OrderedDict[str, list[float]] = OrderedDict()

    def stamp(self, text: str, occurrence: int) -> str:
        if STAMP_RE.match(text):
            return text
        times = self._seen.setdefault(text, [])
        self._seen.move_to_end(text)
        while len(times) <= occurrence:
            times.append(self.clock())
        while len(self._seen) > self.max_texts:
            self._seen.popitem(last=False)
        return time.strftime("[%a %H:%M] ", time.localtime(times[occurrence])) + text


def rewrite_body(body: dict, stamper: Timestamper | None = None) -> dict:
    """Fix the name in user-authored text of a Responses or Chat request, and
    (with a stamper) prefix each user message with when it was said."""
    items = body.get("input")
    counts: dict[str, int] = {}
    if isinstance(items, str):
        body["input"] = fix_name(items)
    elif isinstance(items, list):
        for item in items:
            _fix_message(item, stamper, counts)
    for msg in body.get("messages") or []:  # chat-completions shape
        _fix_message(msg, stamper, counts)
    return body


def _fix_message(msg, stamper: Timestamper | None = None, counts: dict | None = None) -> None:
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = _fix_text(content, stamper, counts)
    elif isinstance(content, list):
        stamped = False
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part["text"] = _fix_text(part["text"], None if stamped else stamper, counts)
                stamped = True


def _fix_text(text: str, stamper: Timestamper | None, counts: dict | None) -> str:
    text = fix_name(text)
    if stamper is None or counts is None:
        return text
    n = counts.get(text, 0)
    counts[text] = n + 1
    return stamper.stamp(text, n)


app = FastAPI()
_stamper = Timestamper()
_client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(300.0, connect=5.0))
_HOP = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    raw = await request.body()
    if raw and request.headers.get("content-type", "").startswith("application/json"):
        try:
            raw = json.dumps(rewrite_body(json.loads(raw), _stamper)).encode()
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
