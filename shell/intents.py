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


# ---------------------------------------------------------------------------
# Loudness is a command too.
#
# Measured on 2026-10-13, in a CLEAN session with no gate denials involved, four plain
# requests went to the model with the tool available and enabled:
#
#     "Turn the volume up a bit."  -> set_volume{louder}, set_volume{current}
#     "Much louder please."        -> (no tool call at all)
#     "Can you turn it up?"        -> set_volume{louder}
#     "Turn it down."              -> (no tool call at all)
#
# Two of four. The earlier gate bug (attention confidence decaying faster than a 4B can
# think) has been fixed, and the model still declines to act about half the time on the
# simplest, most reversible request in the toolset - and when it doesn't act it usually
# says something, which is how a booth hears "I'm afraid I cannot adjust the volume" from
# a robot whose volume button is right there. So: the shell does it, the same way it does
# "stop talking" and "go to sleep", and his reply for that turn is dropped - for volume
# the feedback is the audio itself, which is louder or quieter before he could say so.
# ---------------------------------------------------------------------------

UP_RE = re.compile(
    r"\b(?:(?:much|a lot|way|loads|considerably|even) (?:louder|up)"
    r"|speak up|speak(?:ing)? louder|(?:talk|say that) louder|be louder|get louder"
    r"|louder|turn (?:it|that|the volume|your volume|yourself) up|volume up"
    r"|(?:crank|turn) it(?: all the way)? up|maximum volume|max volume|as loud as (?:it|you|possible))\b",
    re.IGNORECASE,
)
DOWN_RE = re.compile(
    r"\b(?:(?:much|a lot|way|considerably|even) (?:quieter|softer|down|lower)"
    r"|speak (?:quieter|softer|low(?:er)?|more softly)|be quieter|be soft(?:er)?"
    r"|turn (?:it|that|the volume|your volume|yourself) down|volume down"
    r"|quieter|softer|lower (?:it|the volume|your voice)|too loud|that'?s loud"
    r"|(?:tone|turn) it down)\b",
    re.IGNORECASE,
)
INTENSITY_RE = re.compile(r"\b(?:much|a lot|way|considerably|even|really|so)\b", re.IGNORECASE)
PERCENT_RE = re.compile(
    r"\b(?:to|at|set (?:to|at)|set the volume (?:to|at)|in the volume to)\s*(\d{1,3})\s*(?:percent|%)?\b",
    re.IGNORECASE,
)
# Question words that mean "I am asking about the volume", not "change it". The read
# path is `set_volume{level: "current"}`, which the model does reach reliably, and a
# "how loud are you?" that changed the level would be a robot misunderstanding plain
# English. "can you / could you" is NOT in here: that is a request wearing a question.
ASKING_RE = re.compile(r"\b(?:what|why|how|when|who|whether|does|is it|are you|can you hear)\b", re.IGNORECASE)

# "I'll be quieter", "we can be softer" - the speaker describing their OWN level. SELF_RE
# covers "I'll be quiet" for the silence intent but not a comparative, and widening the
# shared pattern would move sleep and silence behaviour to fix a volume one.
SELF_LEVEL_RE = re.compile(
    r"\b(?:i|i'?m|i'll|i will|we|we'?re|we will|let me)\b[^.?!]{0,18}"
    r"\b(?:quieter|quieter still|softer|lower|louder|more softly)\b",
    re.IGNORECASE,
)

VOLUME_MAX_WORDS = 12


def volume_request(text: str) -> str | None:
    """The level word to apply, or None when this is not a loudness command.

    Narrow on purpose, in three ways, because a false positive changes the room's volume
    with nobody having asked:

    * no question words - "what's your volume level?" and "is it too loud?" are questions,
      and the model has a working read path for them (`level: "current"`, which never
      writes);
    * no negation or self-reference, reusing the same guards as sleep and silence -
      "don't turn it up", "I'll be quieter";
    * short. Twelve words, because the wake name is usually bolted on the front.

    The remaining ambiguity is real and accepted: "turn it down" could be about somebody's
    phone. The caller's belt is that the shell only reaches this after the turn was
    accepted into a lease - we are in a conversation with him - and the router still
    applies the attention gate, so ambient room speech cannot move the gain.
    """
    t = (text or "").strip()
    if not t or len(t.split()) > VOLUME_MAX_WORDS:
        return None
    if ASKING_RE.search(t):
        return None
    stripped = NAME_RE.sub(" ", t).strip(" .,!?")
    if not stripped:
        return None  # "Hey Reachy." is a wake word, not a request
    # Strip a courtesy frame so "could you turn it up please" is still recognised as the
    # imperative it is; without this the tier below would have to allow-list every polite
    # phrasing a booth produces.
    core = re.sub(
        r"^\s*(?:hey\s+)?(?:please|kindly|could you|can you|would you|will you|please)\b[\s,]*",
        "",
        stripped,
        flags=re.IGNORECASE,
    ).strip(" .,!?")
    if not core:
        return None
    for before in (core, stripped):
        if NEGATED_RE.search(before) or SELF_RE.search(before) or SELF_LEVEL_RE.search(before):
            continue
        pct = PERCENT_RE.search(before)
        if pct and re.search(r"volume|loud|%", before, re.IGNORECASE):
            return f"{int(pct.group(1))}%"
        up, down = bool(UP_RE.search(before)), bool(DOWN_RE.search(before))
        if up == down:
            continue  # both or neither: "louder or quieter?" is a question about options
        strong = bool(INTENSITY_RE.search(before))
        if up:
            return "much louder" if strong else "louder"
        if down:
            return "much quieter" if strong else "quieter"
    return None
