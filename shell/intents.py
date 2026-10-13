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
# Self-reference: the user talking about their own plans, not an order for him.
# "I will" / "we will" were added 2026-10-12: this regex only knew the contracted "I'll",
# so "I'm going to bed" was correctly ignored while "I will go to bed" put HIM to
# sleep, and "I will be quiet now" read as a demand to stop. Contraction-only
# self-reference detection was always going to lose to the plain form.
SELF_RE = re.compile(
    r"\b(?:I'?m|I am|I'?ll|we'?re|we'?ll|I will|we will|I need to|I have to|I gotta)\s*$", re.IGNORECASE
)


def wants_sleep(text: str) -> bool:
    for m in SLEEP_RE.finditer(text or ""):
        before = text[: m.start()]
        if NEGATED_RE.search(before) or SELF_RE.search(before):
            continue
        return True
    return False


# "stop talking" is a command, not a turn. Measured live on 2026-10-12: the operator
# said it four times; every one DID cut his audio (`audio.speech_started` with
# `cut_playback: true`), and then the phrase arrived at the LLM as a user message,
# which answered it by calling `set_volume` - "stop talking" became "turn it down".
# The cancel half already works; the misroute is the answering half. Mature stacks
# route these around the model rather than giving the model another tool to bump
# into (Home Assistant's `HassNevermind` "does nothing. Used to cancel a request").
STOP_RE = re.compile(
    r"\b(?:stop talking|stop speaking|stop it|stop that|say no more|be quiet"
    r"|shut up|shut your mouth|be still|quiet please|quiet|silence|hush"
    r"|that'?s enough|enough)\b",
    re.IGNORECASE,
)
# A stop command is short and it ends with the command. Both halves matter: "was it
# quiet in Rome?" is 5 words and contains "quiet", and the only thing separating it
# from "be quiet" is the question mark and where the word sits.
STOP_MAX_WORDS = 7
STOP_TAIL_WINDOW = 18


def wants_silence(text: str) -> bool:
    """Is this utterance a demand to stop speaking?

    Deliberately narrow, because a false positive swallows a real turn: short, an
    imperative phrase in the last few words, no question mark, no negation ("don't
    stop talking"), no self-reference ("I'll be quiet"). Even so the CALLER must also
    require that this utterance actually cut his audio - the belt to these braces.
    Suppressing "stop talking" while he is silent would drop a reply to something
    else, and at a booth the phrase that ends a demo is also the phrase someone says
    *about* the demo.
    """
    t = (text or "").strip()
    if not t or t.endswith("?") or "?" in t or len(t.split()) > STOP_MAX_WORDS:
        return False
    for m in STOP_RE.finditer(t):
        if m.start() < len(t) - STOP_TAIL_WINDOW:
            continue  # the phrase is background, not the point ("the quiet room")
        before = t[: m.start()]
        if NEGATED_RE.search(before) or SELF_RE.search(before):
            continue
        return True
    return False
