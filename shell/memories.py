"""Long-term memories the user asks Reachy to keep ("remember that ...").

A small JSON file (data/memories.json), human-readable and hand-editable.
Three tools: remember / recall / forget_memory. Recent memories are also put
in the session instructions (`prompt_block`) so Reachy knows them without a
tool call; `recall` covers anything said since the session started.

Matching is deliberately simple word overlap: a desk robot holds tens of
facts, not thousands, and with few memories recall just returns all of them
(best matches first) and lets the LLM pick.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("shell.memories")

RETURN_ALL_BELOW = 25
PROMPT_MAX = 20

_STOP = set(
    "a an the and or but of to in on at for with about is are was were be been am i me my mine you your "
    "we our it its this that these those what who whom which when where why how do does did can could "
    "should would will please remember recall forget know tell say said s t don".split()
)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP and len(w) > 1}


class MemoryStore:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock
        self._lock = threading.Lock()
        self._items: list[dict] = self._load()

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text())
            return [m for m in data if isinstance(m, dict) and m.get("text")]
        except FileNotFoundError:
            return []
        except Exception as e:  # corrupt file: keep it, start empty
            log.warning("memories unreadable (%r); not overwriting until a save", e)
            return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._items, indent=1, ensure_ascii=False))
        tmp.replace(self.path)

    def add(self, text: str) -> dict:
        text = " ".join(text.split())
        with self._lock:
            for m in self._items:
                if m["text"].lower() == text.lower():
                    return m  # already known
            mid = max((m["id"] for m in self._items), default=0) + 1
            m = {"id": mid, "text": text, "saved": time.strftime("%Y-%m-%d %H:%M", time.localtime(self.clock()))}
            self._items.append(m)
            self._save()
            return m

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._items)

    def search(self, query: str = "", limit: int = 10) -> list[dict]:
        items = self.all()
        q = _words(query)
        scored = [(len(q & _words(m["text"])), i, m) for i, m in enumerate(items)]
        if not q or len(items) <= RETURN_ALL_BELOW:
            # few memories: hand the LLM everything, best matches (then newest) first
            scored.sort(key=lambda s: (-s[0], -s[1]))
            return [m for _, _, m in scored][: max(limit, RETURN_ALL_BELOW)]
        hits = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], -s[1]))
        return [m for _, _, m in hits[:limit]]

    def forget(self, query: str) -> list[dict]:
        """Delete the memories that best match `query` (all ties)."""
        q = _words(query)
        if not q:
            return []
        with self._lock:
            scored = [(len(q & _words(m["text"])), m) for m in self._items]
            best = max((s for s, _ in scored), default=0)
            if best == 0:
                return []
            gone = [m for s, m in scored if s == best]
            self._items = [m for m in self._items if m not in gone]
            self._save()
            return gone

    def prompt_block(self) -> str:
        items = self.all()[-PROMPT_MAX:]
        if not items:
            return ""
        facts = " ".join(f"- {m['text']}" for m in items)
        return f" Things the user asked you to remember (use them naturally, don't recite them): {facts}"


MEMORY_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a fact to long-term memory when the user asks you to remember something "
            "(a name, a preference, a date, where something is). Write one short sentence about the user in "
            "the third person, never 'my' or 'you', e.g. 'The user's wife is named Sarah.' or "
            "'The user's favorite color is green.'",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "The fact, as one short sentence"}},
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Look up long-term memories. Use when the user asks what you remember, or asks "
            "about something they told you before (names, preferences, plans, where they put things).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for; empty returns everything"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": "Delete a saved memory when the user asks you to forget something.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Words identifying the memory"}},
                "required": ["query"],
            },
        },
    },
]


def make_memory_handlers(store: MemoryStore) -> dict[str, Callable[..., dict]]:
    def remember(fact: str = "", **_kw: Any) -> dict:
        if not fact.strip():
            return {"ok": False, "error": "Nothing to remember"}
        m = store.add(fact)
        return {"ok": True, "saved": m["text"]}

    def recall(query: str = "", **_kw: Any) -> dict:
        found = store.search(query)
        if not found:
            return {"ok": True, "memories": [], "note": "Nothing saved about that."}
        return {"ok": True, "memories": [f"{m['text']} (saved {m['saved']})" for m in found]}

    def forget_memory(query: str = "", **_kw: Any) -> dict:
        gone = store.forget(query)
        if not gone:
            return {"ok": False, "error": "No saved memory matches that"}
        return {"ok": True, "forgotten": [m["text"] for m in gone]}

    return {"remember": remember, "recall": recall, "forget_memory": forget_memory}
