"""Conversation lifetime: background compaction + fresh sessions after a gap.

s2s keeps the last N exchanges forever with no sense of time, so a question
left dangling by a failed reply got answered 40 minutes later (2026-10-05).
This keeps our own record of the dialogue, and once the room has been quiet
for `compact_after_s` it summarizes it in the background (LLM, idle time).
After `reset_after_s` of silence the shell starts a FRESH s2s session seeded
with that summary instead of the raw history - proactively, while the room is
quiet, so the reconnect never races the next wake word (2026-10-09: doing it
on the wake word lost "Reachy, ..." to a rejected reconnect 20 times out of
22). The wake word still checks, as a fallback. If the summary is not ready,
the existing session is simply kept (no waiting, nothing lost).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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
        extract_facts: Callable[[list[Turn]], Awaitable[list[str]]] | None = None,
        save_fact: Callable[[str], None] | None = None,
    ) -> None:
        self.journal = journal
        self.summarize = summarize
        self.extract_facts = extract_facts  # lasting facts -> long-term memory
        self.save_fact = save_fact
        self._facts_upto = 0  # turns already mined for facts
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

    def tick(self, engaged: bool) -> str | None:
        """Call periodically on the event loop. Starts compaction when idle;
        returns extra instructions when it is time for a fresh session."""
        if engaged or self.compacting or not self.turns:
            return None
        if self.summary is None or self.summary.covers < len(self.turns):
            if self.idle_s >= self.compact_after_s:
                self._task = asyncio.get_event_loop().create_task(self._compact(), name="compaction")
            return None
        return self.on_wake()

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
        await self._mine_facts(turns)

    async def _mine_facts(self, turns: list[Turn]) -> None:
        if self.extract_facts is None or self.save_fact is None:
            return
        fresh = turns[self._facts_upto :]
        if not any(r == "user" for _, r, _ in fresh):
            return
        try:
            facts = await self.extract_facts(fresh)
        except Exception as e:
            log.warning("fact extraction failed: %r", e)
            self.journal.write("conversation.facts_failed", error=repr(e))
            return
        self._facts_upto = len(turns)
        for fact in facts:
            self.save_fact(fact)
        if facts:
            self.journal.write("conversation.facts_saved", facts=facts)

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
        when = time.strftime("%a %d %b %H:%M", time.localtime(s.ended_at))
        return (
            f"\n\nEarlier conversation (it ended {when}, so treat what follows as a new conversation): "
            f"{s.text} Only bring any of this up if the user does."
        )


def render_turns(turns: list[Turn]) -> str:
    from .intents import NAME_RE  # the speech recognizer's spellings of "Reachy"

    return "\n".join(
        f"[{time.strftime('%a %d %b %Y %H:%M', time.localtime(t))}] "
        f"{'User' if r == 'user' else 'Reachy'}: {NAME_RE.sub('Reachy', x) if r == 'user' else x}"
        for t, r, x in turns
    )


# Model-facing prompts, module-level so tests/test_env.py can scan exactly what
# gets sent. These go out in every deployment environment - a hack-day booth has
# no family in it - so they carry no real names and no household framing.
SUMMARY_PROMPT = (
    "Summarize this conversation between a user and Reachy, a small desk robot, in at most 2 short "
    "sentences of plain prose. Call the speaker 'the user' unless they clearly gave their own name "
    "(the speech recognizer mangles names, so ignore names it gave the user that they did not state). "
    "Keep only lasting facts: people's names and things they said about themselves or their "
    "preferences. Do NOT list topics, unanswered questions, failed replies or "
    "promises to follow up - that conversation is over."
)


async def llm_summarize(base_url: str, model: str, prior: str | None, turns: list[Turn]) -> str:
    """Summarize via the local llama.cpp chat endpoint (thinking off)."""
    import httpx

    body = (f"Summary of what came before:\n{prior}\n\n" if prior else "") + "Conversation:\n" + render_turns(turns)
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": body}],
                "max_tokens": 200,
                "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


# The example has to keep showing the shape of a fact `grounded()` can check
# ("The user's ..." + a name the transcript actually says), so it stays a name -
# just an invented one. "Rachel" was a real name on this desk (2026-10-12).
FACT_PROMPT = (
    "You maintain the long-term memory of Reachy, a small desk robot. Below is a "
    "transcript of what people said near it; background TV and talk between other people is mixed in, and "
    "the speech recognizer garbles words. List the LASTING personal facts the user told Reachy about "
    "themselves or the people and places they named: their own name, other people's names and "
    "relationships, ages, pets, where they live, routines, strong preferences. Rules: only facts stated "
    "with 'my', 'our' or 'I' about people or places, never things from TV or stories; skip questions, "
    "jokes, requests, moods and anything "
    "temporary; never invent or correct names (if a name is unclear, skip it); the robot's own name Reachy is "
    "not the user's. Write each fact as one short sentence in the third person starting 'The user' or "
    "'The user's', e.g. \"The user's sister is named Willow.\" At most 8 facts. Answer with a JSON array of "
    "strings only, [] if there is nothing."
)


def parse_facts(raw: str, limit: int = 8) -> list[str]:
    m = re.search(r"\[.*\]", raw or "", re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for x in items if isinstance(items, list) else []:
        if isinstance(x, str) and 8 <= len(x.strip()) <= 200 and x.strip().lower().startswith("the user"):
            out.append(" ".join(x.split()))
    return out[:limit]


_NOT_NAMES = {"the", "user", "user's", "users", "reachy"}


def grounded(facts: list[str], transcript: str) -> list[str]:
    """Drop facts naming anyone/anywhere the transcript never says (a 3B model
    invented a whole spouse - name and all - from a Bible reading, 2026-10-11)."""
    heard = {w.lower() for w in re.findall(r"[A-Za-z0-9']+", transcript)}
    keep = []
    for f in facts:
        words = re.findall(r"[A-Za-z0-9']+", f)
        names = [w for i, w in enumerate(words) if i and w[0].isupper() and w.lower().rstrip("'s") not in _NOT_NAMES]
        names = [w[:-2] if w.endswith("'s") else w for w in names]
        specific = bool(names) or any(w.isdigit() for w in words)  # "has kids" alone is not worth keeping
        if specific and all(n.lower() in heard for n in names):
            keep.append(f)
    return keep


async def llm_extract_facts(base_url: str, model: str, turns: list[Turn], known: list[str]) -> list[str]:
    """Ask the local llama.cpp chat endpoint for lasting user facts (thinking off)."""
    import httpx

    users = [(t, r, x) for t, r, x in turns if r == "user"]
    if not users:
        return []
    body = (f"Already known (do not repeat):\n- " + "\n- ".join(known) + "\n\n" if known else "") + (
        "Transcript (user side only):\n" + render_turns(users)
    )
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "system", "content": FACT_PROMPT}, {"role": "user", "content": body}],
                "max_tokens": 300,
                "temperature": 0.1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        r.raise_for_status()
        facts = parse_facts(r.json()["choices"][0]["message"]["content"] or "")
        return grounded(facts, " ".join(x for _, _, x in users))
