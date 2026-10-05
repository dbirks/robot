"""Conversation lifetime: background compaction + fresh sessions after a gap.

s2s keeps the last N exchanges forever with no sense of time, so a question
left dangling by a failed reply got answered 40 minutes later (2026-10-05).
This keeps our own record of the dialogue, and once the room has been quiet
for `compact_after_s` it summarizes it in the background (LLM, idle time).
On the next wake word after `reset_after_s` of silence the shell starts a
FRESH s2s session seeded with that summary instead of the raw history. If the
summary is still being written when someone wakes Reachy, the existing
session is simply kept (no waiting, nothing lost).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

log = logging.getLogger("shell.conversation")

Turn = tuple[float, str, str]  # (wall time, "user"|"assistant", text)


@dataclass
class _Summary:
    text: str
    covers: int  # number of turns folded in
    ended_at: float  # wall time of the last turn summarized


class ConversationMemory:
    def __init__(
        self,
        journal,
        summarize: Callable[[str | None, list[Turn]], Awaitable[str]],
        *,
        compact_after_s: float = 180.0,
        reset_after_s: float = 600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.journal = journal
        self.summarize = summarize
        self.compact_after_s = compact_after_s
        self.reset_after_s = reset_after_s
        self.clock = clock
        self.turns: list[Turn] = []
        self.summary: _Summary | None = None
        self._task: asyncio.Task | None = None

    def add(self, role: str, text: str) -> None:
        text = text.strip()
        if text:
            self.turns.append((self.clock(), role, text))

    @property
    def idle_s(self) -> float:
        return self.clock() - self.turns[-1][0] if self.turns else float("inf")

    @property
    def compacting(self) -> bool:
        return self._task is not None and not self._task.done()

    def tick(self, engaged: bool) -> None:
        """Call periodically on the event loop. Starts compaction when idle."""
        if engaged or self.compacting or not self.turns:
            return
        if self.summary is not None and self.summary.covers >= len(self.turns):
            return
        if self.idle_s < self.compact_after_s:
            return
        self._task = asyncio.get_event_loop().create_task(self._compact(), name="compaction")

    async def _compact(self) -> None:
        turns = list(self.turns)
        prior = self.summary.text if self.summary else None
        new = turns[self.summary.covers :] if self.summary else turns
        t0 = time.monotonic()
        try:
            text = (await self.summarize(prior, new)).strip()
        except Exception as e:  # never break the conversation over a summary
            log.warning("compaction failed: %r", e)
            self.journal.write("conversation.compaction_failed", error=repr(e))
            return
        if text:
            self.summary = _Summary(text=text, covers=len(turns), ended_at=turns[-1][0])
            self.journal.write(
                "conversation.compacted", turns=len(turns), chars=len(text), seconds=round(time.monotonic() - t0, 2)
            )

    def on_wake(self) -> str | None:
        """New attention lease. Returns extra instructions for a FRESH session,
        or None to keep the current one."""
        if not self.turns or self.idle_s < self.reset_after_s:
            return None
        if self.compacting or self.summary is None or self.summary.covers < len(self.turns):
            self.journal.write("conversation.reset_skipped", reason="summary not ready", idle_s=round(self.idle_s))
            return None
        s = self.summary
        self.turns = []
        self.summary = _Summary(text=s.text, covers=0, ended_at=s.ended_at)
        self.journal.write("conversation.reset", idle_s=round(self.clock() - s.ended_at))
        when = time.strftime("%a %H:%M", time.localtime(s.ended_at))
        return (
            f"\n\nEarlier conversation (it ended {when}, so treat what follows as a new conversation): "
            f"{s.text} Only bring any of this up if the user does."
        )


def render_turns(turns: list[Turn]) -> str:
    return "\n".join(
        f"[{time.strftime('%a %H:%M', time.localtime(t))}] {'User' if r == 'user' else 'Reachy'}: {x}"
        for t, r, x in turns
    )


async def llm_summarize(base_url: str, model: str, prior: str | None, turns: list[Turn]) -> str:
    """Summarize via the local llama.cpp chat endpoint (thinking off)."""
    import httpx

    prompt = (
        "Summarize this conversation between a user and Reachy, a small desk robot, in at most 2 short "
        "sentences of plain prose. Keep only lasting facts: people's names and things they said about "
        "themselves or their preferences. Do NOT list topics, unanswered questions, failed replies or "
        "promises to follow up - that conversation is over."
    )
    body = (f"Summary of what came before:\n{prior}\n\n" if prior else "") + "Conversation:\n" + render_turns(turns)
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": body}],
                "max_tokens": 200,
                "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
