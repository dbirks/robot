"""Direct commands the shell executes itself, without waiting for the LLM.

The small LLMs narrate instead of acting once a session is full of chat:
"Alright, I'm going to sleep now. Goodnight!" - and nothing moves
(2026-10-11: Qwen3.5-4B 37/48 and Granite 4.1 3B 30/48 on a replayed real
session, sleep the most-missed request). Sleep is unambiguous enough to match
on the transcript; the LLM still answers in words, and the handler is
idempotent if it also calls the tool.
"""

from __future__ import annotations

import re

# Parakeet's mis-hearings of "Reachy" observed in the journal, plus near
# spellings (shell/llm_proxy.py rewrites them before the LLM sees them).
NAME_VARIANTS = (
    "Richie|Richy|Ricky|Rickie|Ritchie|Ritchy|Richey|Ritchey|Ricci|Riche|Rachi|Rachie|Reechy|Reachie|Reachey|Reechie"
)
NAME_RE = re.compile(rf"\b(?:{NAME_VARIANTS})\b", re.IGNORECASE)
_NAME = rf"(?:Reachy|{NAME_VARIANTS})"
# Just the wake word, nothing else: "Reachy?", "Hey Richie.", "Okay, Reachy, Reachy!"
WAKE_ONLY_RE = re.compile(
    rf"^\W*(?:(?:hey|hi|hello|ok|okay|yo|oh|um|uh)\W+)*{_NAME}(?:\W+{_NAME})*\W*$",
    re.IGNORECASE,
)


def is_bare_wake(text: str) -> bool:
    return bool(WAKE_ONLY_RE.match(text or ""))


SLEEP_RE = re.compile(
    r"\b(?:go(?:ing)?|get|head|off) (?:back )?to (?:sleep|bed)\b"
    r"|\btime (?:for|to go to) (?:bed|sleep)\b"
    r"|\btake a nap\b|\bnap time\b|\bpower down\b|\bshut ?down\b",
    re.IGNORECASE,
)
# "don't go to sleep", "not going to sleep", "I'm going to bed" (the user, not him)
NEGATED_RE = re.compile(r"\b(?:don'?t|do not|not|never|no)\b[^.?!]{0,20}$", re.IGNORECASE)
SELF_RE = re.compile(r"\b(?:I'?m|I am|I'?ll|we'?re|we'?ll|I need to|I have to|I gotta)\s*$", re.IGNORECASE)


def wants_sleep(text: str) -> bool:
    for m in SLEEP_RE.finditer(text or ""):
        before = text[: m.start()]
        if NEGATED_RE.search(before) or SELF_RE.search(before):
            continue
        return True
    return False
