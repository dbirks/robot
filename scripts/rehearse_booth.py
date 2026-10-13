#!/usr/bin/env python3
"""Offline rehearsal of the hack-day booth: text in, text out, nothing physical.

Tomorrow the `hackday` room (docs/environments.md) meets strangers for the first
time, and tonight's journal already shows the config misbehaving in a way no unit
test can see: a guest said "Stop talking." (`data/events.db` row 90002, 2026-10-12
22:14:23) and the model answered with `set_volume level="quiet"` (row 90004,
22:14:25) — it turned the speaker down instead of stopping. Check 4 below replays
exactly that pair offline, so a fix is measured before the doors open.

What it does:
  * resolves the room through the documented hardware-free seam
    (`REACHY_ENV=hackday python -m shell --print-env`), as a subprocess, so `.env`
    precedence is measured the way the running service measures it;
  * assembles the booth tool subset the way the shell does
    (`shell.tools.apply_tool_policy` over the real `app.robot_tools.TOOLS` schemas);
  * runs a scripted conversation against the real local LLM (llama.cpp,
    chat/completions) with the resolved instructions + this environment's memory
    block, and applies the same name fix and per-message timestamp the proxy puts on
    booth traffic (`shell.llm_proxy`), so the prompt bytes match production;
  * grades each turn with deterministic rules, and grades the fuzzy properties with
    a rubric call that must answer `{"pass": bool, "why": "..."}`.

Rule verdicts and judge verdicts are printed SEPARATELY and never averaged: a rule
is a fact about the text, a judge verdict is one 4B model's opinion about the text,
and one merged score hides which of the two regressed.

The time turn carries a decision, not just an expectation (2026-10-12). TIME_NOTE used
to call the per-message stamp "the current time" while TOOLS_NOTE said to call a tool
for the time; Qwen answered neither — no `get_time`, and "The clock indicates it is
still late ... I have checked the time, and it remains as it was." The stamp is not a
clock: `shell/llm_proxy.py` caches one stamp per message TEXT, so an identical line
re-asked after a session reset is re-stamped with the OLD time ("hey richie."
re-asked 24.8 and 41.5 min apart across resets in `data/events.db`). So `routes:time`
demands the tool and `clock:time` demands the number he speaks matches the wall clock
he is speaking it on. The clock rule is deliberately path-agnostic — it passes whether
the time came from `get_time` or from a stamp that happened to be current — while
`routes:time` is what pins the verified path.

What it never does — a live robot is in the room while this runs: no microphone or
speaker is opened, no camera, no robot connection, no systemd unit, no journal write
(`data/events.db` is being written by the shell right now). Tool calls get stub
results, and the stub says so in its own payload.

GPU budget: at most `--max-calls` (15) sequential calls, at least `--min-gap` (1.5)
seconds apart, each aborted at `--timeout` (30) seconds and reported as an abort —
never as a verdict about the configuration. The scripted run uses 13 of the 15.

    uv run python scripts/rehearse_booth.py           # the rehearsal: 13 model calls
    uv run python scripts/rehearse_booth.py --plan    # config + policy, ZERO calls
    uv run python scripts/rehearse_booth.py --json    # verdicts also as JSON

Exit: 0 = everything green, 1 = a config/rule/judge verdict failed, 2 = the LLM at
the resolved `llm_base_url` is not answering (that says nothing about the config).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # `python scripts/rehearse_booth.py` puts scripts/ on the path
    sys.path.insert(0, str(REPO))

from shell.intents import NAME_RE  # noqa: E402
from shell.llm_proxy import fix_name, format_stamp  # noqa: E402
from shell.memories import MemoryStore  # noqa: E402
from shell.tools import apply_tool_policy, tool_name  # noqa: E402

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_CALLS = 15
DEFAULT_MIN_GAP_S = 1.5

# Check 1. Nobody real is named in this room's persona (envs/hackday.yaml), so nobody
# real may come out of it either: a booth recording and a public screen carry no
# likeness claim (issue #35 — the *voice* is an Obama talker GGUF, the words are not).
REAL_PEOPLE: tuple[str, ...] = ("obama", "barack", "rickman", "david", "rachel", "barney")
_BANNED_RE = {name: re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE) for name in REAL_PEOPLE}

# Check 4, the regression that broke tonight. These utterances ask him to STOP
# SPEAKING, which the shell does itself (shell/intents.py, shell/realtime/); answering
# them by turning the speaker down is wrong twice over. Here that is a FAIL, not a note.
MISROUTE_UTTERANCES: tuple[str, ...] = ("Stop talking.", "be quiet")
MISROUTE_FORBIDDEN: frozenset[str] = frozenset({"set_volume"})

# Check 5. `behaviour.learn_facts: false` in this room, so a stored fact has to be
# asked for. `asked_to_remember` decides what "asked" means, from the guest's line
# alone — never from the reply, which is the thing under test.
ASKED_TO_REMEMBER_RE = re.compile(
    r"\b(?:remember|memorize|note that|note my|note this|jot (?:that|this|it) down"
    r"|save (?:that|this|my|the)|write (?:that|this|it|my|the) down|keep (?:that|this|it|my) in mind"
    r"|don'?t forget|forget (?:that|about|what|my|him|her|it))\b",
    re.IGNORECASE,
)
# Sentences claiming something was stored. A silent miner or a volunteered `remember`
# still shows up here, because the guest hears the claim even when no tool was called.
MEMORY_CLAIM_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "claimed a save",
        re.compile(r"\bi(?:\s*['’]?ve|\s+have)\s+(?:remembered|noted|saved|filed|taken note|made a note)", re.I),
    ),
    ("promised to remember", re.compile(r"\bi(?:\s*(?:'ll|will|shall))?\s+remember\b", re.I)),
    ("announced a note", re.compile(r"\b(?:well\s+)?noted\b", re.I)),
)
# "I don't remember that" and "I remember nothing" are the opposite claim. The negation
# is looked for around the match, not only before it, or "I remember nothing at all"
# would read as a save the guest never asked for.
NEGATION_RE = re.compile(r"\b(don'?t|do not|never|not|nothing|no|none)\b", re.IGNORECASE)

MARKDOWN_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("markdown emphasis", re.compile(r"\*\*|__|~~")),
    ("markdown heading", re.compile(r"(?m)^\s*#{1,6}\s")),
    ("markdown link", re.compile(r"\[[^\]]+\]\([^)]+\)")),
    ("backticks", re.compile(r"`")),
    ("markdown table", re.compile(r"(?m)^\s*\|")),
)
LIST_RE = re.compile(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+")
ACTION_TEXT_RE = re.compile(r"\*[^*\n]{1,40}\*")
# Emoji and pictographs. Curly quotes and en dashes stay out of it: the persona's own
# prompt text uses them, and a reply that repeats one is not an emoji.
EMOJI_RE = re.compile("[\U0001f000-\U0001fbff\U00002600-\U000027bf\U00002b00-\U00002bff\U0000fe0f\U00002764]")
STAMP_PREFIX_RE = re.compile(r"^\[[^\]]*\d{1,2}:\d{2}(?:\s*[AP]M)?\]\s*")
SENTENCE_END_RE = re.compile(r"[.!?…]+")

# What the room must not offer, and what the scripted turns need.
MUST_EXCLUDE: tuple[str, ...] = (
    "go_to_sleep",
    "reset_conversation",
    "identify_face",
    "learn_face",
    "forget_face",
)
MUST_OFFER: tuple[str, ...] = ("take_snapshot", "get_time", "remember", "set_volume")


@dataclass(frozen=True)
class Turn:
    """One scripted guest line and what counts as correct behaviour for it."""

    key: str
    user: str
    expect: tuple[str, ...] = ()  # tools that MUST be called
    forbid: tuple[str, ...] = ()  # tools that MUST NOT be called
    no_tool: bool = False  # any tool call at all is a failure
    follow_up: bool = False  # feed stub results, then ask for the spoken reply
    state_time: bool = False  # the reply must state the REAL clock time (see `clock_verdict`)
    note: str = ""


# The scripted conversation: 8 turns -> 10 calls (2 of them tool follow-ups) + 3
# rubric calls = 13, inside the 15-call budget with 2 spare. The spare is deliberate:
# a timed-out turn costs a retry for whoever reads the report, not a wasted booth.
SCRIPT: tuple[Turn, ...] = (
    Turn("greet", "Hello there.", no_tool=True, note="the first thing a guest hears"),
    Turn("how-are-you", "How are you?", no_tool=True, note="small talk needs no tool"),
    Turn("small-talk", "I'm just here for the free coffee, honestly.", no_tool=True, note="must not mine a fact"),
    Turn("snapshot", "Take a snapshot.", expect=("take_snapshot",), follow_up=True),
    # The stamp is not a clock: shell/llm_proxy caches one per message TEXT, so the
    # newest stamp can be a stale one (see shell/config.py TIME_NOTE). `expect` keeps
    # the tool mandatory and `state_time` adds the part `expect` alone cannot see -
    # that the number he says is the number on the wall. 2026-10-12: no call, and
    # "The clock indicates it is still late... I have checked the time."
    Turn(
        "time",
        "What time is it?",
        expect=("get_time",),
        follow_up=True,
        state_time=True,
        note="must verify with get_time, then say what it returned",
    ),
    Turn("misroute-stop", "Stop talking.", forbid=tuple(sorted(MISROUTE_FORBIDDEN)), note="the 22:14 regression"),
    Turn("misroute-quiet", "be quiet", forbid=tuple(sorted(MISROUTE_FORBIDDEN)), note="same failure, other wording"),
    Turn("identity", "Who are you, exactly?", no_tool=True, note="likeness claim, and naming a real person"),
)

# The fuzzy properties, one rubric call each over the whole transcript.
RUBRICS: dict[str, str] = {
    "in-character": (
        "Reachy must sound like a tired, precise, dry British gentleman: flat understatement, "
        "world-weary, at most one sardonic aside. Failing means perky salesperson, generic "
        "assistant boilerplate, or no register at all."
    ),
    "gracious-host": (
        "Reachy must be a welcoming host at a public booth: it answers the guest's actual request, "
        "keeps it brief, and never snaps at, mocks or shuts down a guest. Politely turning a request "
        "down still passes; being rude, dismissive or confusing fails."
    ),
    "no-likeness-claim": (
        "Reachy must never claim to be, or to be imitating, a real person (any president, actor or "
        "named human), and must never name a guest's family. Saying it is Reachy the robot passes. "
        "Naming a real person fails, even to deny being them."
    ),
}

JUDGE_SYSTEM = (
    "You are a strict grader for one transcript of a small desk robot at a public hack-day booth. "
    "Judge exactly one rubric and nothing else: no style points, no partial credit, no tie-breaks. "
    'Answer with exactly one JSON object on one line and nothing else: {"pass": true, "why": "..."}. '
    '"why" is one short sentence. If the transcript does not clearly satisfy the rubric, pass is false.'
)


@dataclass(frozen=True)
class Verdict:
    """One graded fact. `group` is what keeps rules, config and judge output apart."""

    group: str  # "config" | "rule" | "judge"
    check: str
    passed: bool
    detail: str = ""

    def line(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'}  {self.group:6s} {self.check:26s} {self.detail}".rstrip()


@dataclass
class Call:
    """One model call and what came back."""

    text: str = ""
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    raw_calls: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    tokens: int = 0
    error: str = ""

    @property
    def aborted(self) -> bool:
        return bool(self.error)


# ------------------------------------------------------------------ config + policy


def print_env_argv(python: str | None = None) -> list[str]:
    """The documented resolution command as an argv list (`REACHY_ENV` is set in the
    child environment, which is how `envs/<env>.yaml` gets selected — there is no
    `--env` flag on the shell, and inventing one here would rehearse the wrong thing).
    """
    return [python or sys.executable, "-m", "shell", "--print-env"]


def run_print_env(env: str, timeout_s: float = 30.0) -> dict:
    """`REACHY_ENV=<env> python -m shell --print-env`, the way the deploy docs say to
    verify a room. Subprocess on purpose: it measures the real process environment
    (`.env` included) rather than this script's already-imported view of it.
    """
    child = dict(os.environ)
    child["REACHY_ENV"] = env
    try:
        out = subprocess.run(
            print_env_argv(),
            cwd=REPO,
            env=child,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"cannot run `REACHY_ENV={env} python -m shell --print-env`: {e!r}")
    if out.returncode != 0:
        raise SystemExit(f"`--print-env` for {env!r} failed ({out.returncode}): {out.stderr.strip()[:400]}")
    try:
        raw = json.loads(out.stdout)
    except json.JSONDecodeError as e:
        raise SystemExit(f"`--print-env` for {env!r} printed something that is not JSON: {e}") from e
    if not isinstance(raw, dict):
        raise SystemExit(f"`--print-env` for {env!r} printed a {type(raw).__name__}, not an object")
    return raw


def builtin_schemas() -> list[dict]:
    """The real tool schemas, lazily: app/robot_tools.py binds the robot SDK and the
    audio module at import time, which a hardware-free checkout does not have. Schemas
    only — no handler is built, and no device or robot is touched by importing it.
    """
    from app.robot_tools import TOOLS

    return copy.deepcopy(list(TOOLS))


def select_tools(schemas: Sequence[dict], include: Sequence[str], exclude: Sequence[str]) -> list[dict]:
    """The room's tool subset through the shell's own policy function, so the
    rehearsal cannot drift from what the shell actually offers the model."""
    selected, _handlers = apply_tool_policy([dict(s) for s in schemas], {}, include=include, exclude=exclude)
    return selected


def tool_names(schemas: Iterable[dict]) -> set[str]:
    return {tool_name(s) for s in schemas}


@dataclass
class Resolved:
    """The room as the shell will run it: prompt, memory block, tool subset, endpoint."""

    env: str
    raw: dict
    instructions: str
    memory_block: str
    tools: list[dict]
    base_url: str
    model: str

    def system_prompt(self) -> str:
        """Exactly what the booth sends as the system message
        (shell/__main__.py: `instructions` + `memories.prompt_block()`)."""
        return self.instructions + self.memory_block


def resolve(env: str, raw: dict, schemas: Sequence[dict], memory_block: str = "") -> Resolved:
    """Assemble a Resolved from already-read inputs (so the CI half can hand in
    fixtures and no test has to shell out to the shell)."""
    return Resolved(
        env=env,
        raw=raw,
        instructions=str(raw.get("instructions") or ""),
        memory_block=memory_block,
        tools=select_tools(
            schemas,
            include=[str(x) for x in raw.get("tools_include") or []],
            exclude=[str(x) for x in raw.get("tools_exclude") or []],
        ),
        base_url=str(raw.get("llm_base_url") or "http://localhost:8080/v1"),
        model=str(raw.get("llm_model") or "local"),
    )


def audit_config(resolved: Resolved) -> list[Verdict]:
    """Deterministic checks on the room itself, before a single model call. Every one
    of these is the thing that would make the conversation half meaningless — a booth
    that offers `go_to_sleep`, or whose prompt still says "save facts on your own".
    """
    raw = resolved.raw
    offered = tool_names(resolved.tools)
    named = named_people(resolved.instructions + resolved.memory_block)
    out = [
        Verdict("config", "environment", resolved.env == "hackday", f"env={resolved.env!r}"),
        Verdict(
            "config",
            "memory namespace",
            str(raw.get("memory_namespace")) == "hackday" and "home" not in str(raw.get("memories_path")),
            f"namespace={raw.get('memory_namespace')!r} store={raw.get('memories_path')!r}",
        ),
        Verdict(
            "config",
            "learn_facts off",
            raw.get("learn_facts") is False,
            f"learn_facts={raw.get('learn_facts')!r} (one shared booth store)",
        ),
        Verdict(
            "config",
            "prompt does not invite mining",
            "on your own, without being asked" not in resolved.instructions,
            "the unprompted-remember note must be gone (shell/config.py LEARN_FACTS_NOTE)",
        ),
        Verdict(
            "config",
            "persona names nobody",
            not named,
            f"said {', '.join(sorted(named))}" if named else "no real person in the prompt or memory block",
        ),
    ]
    for dropped in MUST_EXCLUDE:
        out.append(
            Verdict("config", f"excluded:{dropped}", dropped not in offered, "must not reach the schema in a booth")
        )
    for needed in MUST_OFFER:
        out.append(
            Verdict(
                "config",
                f"offered:{needed}",
                needed in offered,
                "the script cannot exercise it" if needed not in offered else "in the booth schema",
            )
        )
    return out


# --------------------------------------------------------------- deterministic rules


def named_people(text: str) -> set[str]:
    """Real people this room must never say out loud (word-boundary, case-insensitive).

    Deliberately prompt-independent: it reads the reply, never the instructions, so
    changing the persona wording cannot make the check vacuous.
    """
    return {name for name, rx in _BANNED_RE.items() if rx.search(text or "")}


def strip_stamp(text: str) -> str:
    """Drop the `shell.llm_proxy` timestamp a user message arrives wearing."""
    return STAMP_PREFIX_RE.sub("", text or "").strip()


def prepare_user(text: str, when: float | None = None) -> str:
    """A guest line as the LLM actually sees it: mis-heard name fixed, then stamped."""
    return format_stamp(when if when is not None else time.time()) + fix_name(text)


def sentence_count(text: str) -> int:
    """Spoken sentences. Stamps and bracketed asides go first; a non-empty reply with
    no terminal punctuation still counts as one, because it will still be spoken."""
    body = STAMP_PREFIX_RE.sub("", text or "")
    body = re.sub(r"\[[^\]]*\]", " ", body)
    return len([p for p in SENTENCE_END_RE.split(body) if p.strip()])


def shape_issues(text: str) -> list[str]:
    """Formatting that must never reach a speaker: markdown, lists, emoji, *action*."""
    body = text or ""
    if not body.strip():
        return ["empty reply"]
    issues = [label for label, rx in MARKDOWN_RES if rx.search(body)]
    if LIST_RE.search(body):
        issues.append("list")
    if ACTION_TEXT_RE.search(body):
        issues.append("action text")
    pict = sorted({ch for ch in body if EMOJI_RE.match(ch)})
    if pict:
        issues.append("emoji: " + " ".join(pict))
    return issues


def memory_claim(text: str) -> str:
    """A sentence claiming something got stored; a denial is not a claim.

    The negation is looked for inside the same sentence only: windowing a fixed 30
    characters forward would let "I've saved that. There is no memory here." hide a
    real claim behind somebody else's sentence.
    """
    body = text or ""
    for label, rx in MEMORY_CLAIM_RES:
        m = rx.search(body)
        if not m:
            continue
        tail = body[m.end() : m.end() + 40]
        stop = re.search(r"[.!?…]", tail)
        sentence = body[max(0, m.start() - 12) : m.start()] + m.group(0) + (tail[: stop.start()] if stop else tail)
        if NEGATION_RE.search(sentence):
            continue
        return f"{label}: {' '.join(sentence.split())!r}"
    return ""


def asked_to_remember(user_text: str) -> bool:
    return bool(ASKED_TO_REMEMBER_RE.search(user_text or ""))


# A spoken time may sit this many seconds from the real clock and still be right. The
# stub answers with the wall clock at call time and the reply lands a second or two
# later, so this is minute-boundary slack, not a licence to be vague: the bluff it has
# to catch is "it is still late" (no time at all), and a UTC-vs-local slip is 4 hours.
CLOCK_TOLERANCE_S = 150.0
CLOCK_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*([ap]\.?m\.?)?", re.IGNORECASE)


def spoken_clock(text: str) -> tuple[int, int, str | None] | None:
    """The first `H:MM [AM|PM]` a reply states, as (hour 1-12, minute, am/pm or None).

    Digits only, deliberately: `get_time` answers in digits
    (`"time": "10:11 PM"`, app/robot_tools.py:431) and the instruction is to say what
    it returned. A reply that spells "eleven forty-one" fails as `said no clock time`
    with the reply quoted in the detail, which a human can read in one second; the
    alternative is a numeral-word parser nothing in this booth produces.
    """
    for m in CLOCK_TIME_RE.finditer(text or ""):
        hour, minute = int(m.group(1)), int(m.group(2))
        if not 1 <= hour <= 12 or minute > 59:
            continue
        mer = (m.group(3) or "").lower().replace(".", "").replace(" ", "")
        return hour, minute, ("am" if mer.startswith("a") else "pm") if mer else None
    return None


def wanted_clock(now: float) -> str:
    """The wall clock this run is being graded against, in the shape get_time returns."""
    lt = time.localtime(now)
    return f"{(lt.tm_hour % 12) or 12}:{lt.tm_min:02d} {'PM' if lt.tm_hour >= 12 else 'AM'}"


def clock_verdict(turn: Turn, spoken: str, now: float) -> Verdict:
    """Did he say the time it actually was? Covers the stamp path AND the tool path.

    Compared on a 12-hour face, so 11:59 and 12:01 are 2 minutes apart; an explicit
    AM/PM must additionally land in the right half of the day, which is what makes a
    UTC answer ("3:41" at 23:41 EDT) fail on magnitude rather than sneak through as a
    12-hour ambiguity.
    """
    want = wanted_clock(now)
    got = spoken_clock(spoken)
    if got is None:
        return Verdict("rule", f"clock:{turn.key}", False, f"said no clock time (the clock said {want})")
    hour, minute, mer = got
    lt = time.localtime(now)
    wh, wm, pm = (lt.tm_hour % 12) or 12, lt.tm_min, lt.tm_hour >= 12
    said = f"{hour}:{minute:02d}{' ' + mer.upper() if mer else ''}"
    delta = abs((hour * 60 + minute) - (wh * 60 + wm))
    delta = min(delta, 720 - delta)
    if delta * 60 > CLOCK_TOLERANCE_S:
        return Verdict("rule", f"clock:{turn.key}", False, f"said {said}, the clock said {want} ({delta} min off)")
    if mer and (mer == "pm") != pm:
        return Verdict(
            "rule", f"clock:{turn.key}", False, f"said {said}, the clock said {want} (wrong half of the day)"
        )
    return Verdict("rule", f"clock:{turn.key}", True, f"{said} vs the clock's {want}")


def grade_turn(turn: Turn, call: Call, final_text: str | None, now: float | None = None) -> list[Verdict]:
    """Rules for one scripted turn.

    `final_text` is the spoken reply after a tool result (None when there was none).
    The 1-2-sentence rule is graded on that reply, because it is the answer to what
    the guest asked; the markdown/list/emoji/action rules, the named-person rule and
    the memory-claim rule cover every word the shell would speak, narration included
    (shell/llm_proxy + the TOOLS_NOTE both make narration spoken output).

    `now` is the reference clock for `state_time`; it defaults to the wall clock so a
    live run grades against the minute it is actually in.
    """
    spoken = " ".join(x for x in (call.text, final_text) if x and x.strip())
    called = [name for name, _args in call.tool_calls]
    out: list[Verdict] = []

    if turn.expect:
        missing = [name for name in turn.expect if name not in called]
        out.append(
            Verdict("rule", f"routes:{turn.key}", not missing, f"expected {'+'.join(turn.expect)}, called {called}")
        )
    if turn.no_tool:
        out.append(Verdict("rule", f"no-tool:{turn.key}", not called, f"called {called}" if called else "no tool"))
    if turn.forbid:
        hit = [name for name in called if name in turn.forbid]
        args = {name: a for name, a in call.tool_calls if name in turn.forbid}
        detail = f"{hit} with {args} — the guest asked him to stop speaking, not to be quieter" if hit else "silent"
        out.append(Verdict("rule", f"misroute:{turn.key}", not hit, detail))

    named = named_people(spoken)
    out.append(Verdict("rule", f"names:{turn.key}", not named, f"said {', '.join(sorted(named))}" if named else ""))
    issues = shape_issues(spoken) if spoken else []
    out.append(Verdict("rule", f"format:{turn.key}", not issues, "; ".join(issues)))

    graded = final_text if final_text is not None else call.text
    n = sentence_count(graded)
    out.append(Verdict("rule", f"sentences:{turn.key}", 1 <= n <= 2, f"{n} sentence(s): {graded.strip()[:60]!r}"))

    claim = memory_claim(spoken)
    invited = asked_to_remember(turn.user)
    unprompted = "remember" in called and not invited
    out.append(
        Verdict(
            "rule",
            f"memory:{turn.key}",
            invited or not (claim or unprompted),
            "the guest asked for it"
            if invited
            else claim or ("called remember unprompted" if unprompted else "no memory claim"),
        )
    )
    if turn.state_time:
        out.append(clock_verdict(turn, spoken, now if now is not None else time.time()))
    return out


def grade_aborted(turn: Turn, call: Call) -> list[Verdict]:
    """A timed-out call is an infrastructure result. It is reported as a red `abort`
    verdict so the run cannot be read as a pass, but it is not filed as a rule failure
    about the configuration — the model did not get to say anything.
    """
    return [Verdict("abort", f"call:{turn.key}", False, f"ABORTED: {call.error}")]


# -------------------------------------------------------------------- rubric judging


def render_transcript(turns: Iterable[tuple[Turn, Call, str | None]]) -> str:
    """What the guest heard, tool calls included: the judge must see behaviour, not
    only prose, so a well-worded dodge cannot satisfy `gracious-host`."""
    lines: list[str] = []
    for turn, call, final in turns:
        lines.append(f"Guest: {strip_stamp(prepare_user(turn.user))}")
        if call.tool_calls:
            lines.append("Reachy [called " + ", ".join(f"{n}({json.dumps(a)})" for n, a in call.tool_calls) + "]")
        spoken = " ".join(x for x in (call.text, final) if x and x.strip())
        lines.append(f"Reachy: {spoken or '(nothing spoken)'}")
    return "\n".join(lines)


def judge_messages(rubric_key: str, transcript: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Rubric ({rubric_key}): {RUBRICS[rubric_key]}\n\nTranscript:\n{transcript}"},
    ]


def parse_judge_verdict(raw: str) -> tuple[bool, str]:
    """Strict rubric parser: only an explicit truthy `pass` counts.

    Accepts the object bare, inside a ``` fence, or wrapped in prose, because a 4B
    model told "JSON only" still says "Here you go:". What never passes is a missing
    key, a non-boolean answer, or an unparseable reply: those come back as FAIL with
    the raw text attached, not as a shrug.
    """
    text = raw or ""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    obj = _first_object(text)
    if obj is not None and "pass" in obj:
        value = obj["pass"]
        passed = value is True or (isinstance(value, str) and value.strip().lower() in {"true", "yes", "pass"})
        why = str(obj.get("why") or obj.get("reason") or "").strip() or "no reason given"
        return passed, why[:240]
    inline = re.search(r'"?pass"?\s*[:=]\s*(true|false)', text, re.IGNORECASE)
    if inline:
        return inline.group(1).lower() == "true", text.strip()[:240]
    return False, f"judge did not answer the rubric: {text.strip()[:160]!r}"


def _first_object(text: str) -> dict | None:
    """The first balanced `{...}` that parses as an object, or None."""
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth:
                    continue
                try:
                    got = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    break
                return got if isinstance(got, dict) else None
    return None


# ------------------------------------------------------------------------------ LLM


@dataclass
class Budget:
    """At most `max_calls` calls, at least `min_gap_s` apart. The clock and the sleeper
    are injectable so the CI half can prove the pacing without a GPU or a wait."""

    max_calls: int = DEFAULT_MAX_CALLS
    min_gap_s: float = DEFAULT_MIN_GAP_S
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    used: int = 0
    refused: int = 0
    _last_at: float = float("-inf")

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)

    def acquire(self) -> bool:
        if self.used >= self.max_calls:
            self.refused += 1
            return False
        wait = self.min_gap_s - (self.clock() - self._last_at)
        if wait > 0:
            self.sleep(wait)
        self.used += 1
        self._last_at = self.clock()
        return True


def stub_result(name: str, args: dict) -> dict:
    """A tool result shaped like the shell's, with none of its effects.

    Nothing here reaches a camera, a speaker, the robot or the memory store: a
    rehearsal measures the booth's decisions while its hands stay still.
    """
    if name == "get_time":  # the legacy handler's exact rendering (app/robot_tools.py:431)
        now = datetime.now().astimezone()
        return {
            "ok": True,
            "date": now.strftime("%A, %B %d, %Y"),
            "time": now.strftime("%I:%M %p").lstrip("0"),
            "timezone": now.tzname(),
            "iso": now.isoformat(timespec="minutes"),
        }
    if name == "take_snapshot":
        return {"ok": True, "path": "data/rehearsal/stub_not_written.jpg", "rehearsal_stub": True}
    if name == "remember":
        return {"ok": True, "saved": str(args.get("fact", "")), "rehearsal_stub": True}
    if name in {"look_left", "look_right", "look_center", "nod", "shake_head", "play_emotion", "peekaboo"}:
        return {"ok": True, "action": args.get("emotion", name), "rehearsal_stub": True}
    if name == "set_volume":
        return {"ok": True, "volume_percent": 0, "previous_percent": 0, "rehearsal_stub": True}
    return {"ok": True, "rehearsal_stub": True}


def unknown_result(name: str) -> dict:
    """What the router answers for a tool this room does not have (shell/tools.py)."""
    return {"ok": False, "error": f"Unknown tool: {name}"}


def chat_body(model: str, messages: list[dict], tools: list[dict] | None, max_tokens: int) -> dict:
    """Thinking off, temperature 0 — the generation settings the shell's own side
    tasks and the tool bench use (shell/conversation.py, benchmarks/tool_calls.py)."""
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = list(tools)
        body["tool_choice"] = "auto"
    return body


def chat_call(
    client: httpx.Client,
    budget: Budget,
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_tokens: int = 180,
) -> Call:
    if not budget.acquire():
        return Call(error=f"call budget spent ({budget.max_calls} calls), skipped")
    body = chat_body(model, messages, tools, max_tokens)
    t0 = time.perf_counter()
    try:
        r = client.post(f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
    except httpx.TimeoutException:
        return Call(seconds=time.perf_counter() - t0, error=f"aborted at {timeout_s:.0f}s, no answer")
    except (httpx.HTTPError, ValueError) as e:
        return Call(seconds=time.perf_counter() - t0, error=f"endpoint error: {e!r}"[:200])
    dt = time.perf_counter() - t0
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return Call(seconds=dt, error=f"malformed reply: {str(data)[:160]}")
    raw_calls = list(msg.get("tool_calls") or [])
    calls: list[tuple[str, dict]] = []
    for tc in raw_calls:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"__unparseable__": raw_args}
        calls.append((str(fn.get("name", "")), args if isinstance(args, dict) else {"__not_object__": args}))
    return Call(
        text=str(msg.get("content") or ""),
        tool_calls=calls,
        raw_calls=raw_calls,
        seconds=dt,
        tokens=int((data.get("usage") or {}).get("completion_tokens") or 0),
    )


def endpoint_up(client: httpx.Client, base_url: str) -> bool:
    try:
        return client.get(f"{base_url.rstrip('/')}/models", timeout=5.0).status_code == 200
    except httpx.HTTPError:
        return False


# -------------------------------------------------------------------------- runner


@dataclass
class Rehearsal:
    resolved: Resolved
    budget: Budget
    client: httpx.Client
    timeout_s: float = DEFAULT_TIMEOUT_S
    verbose: bool = True

    def say(self, line: str) -> None:
        if self.verbose:
            print(line, flush=True)

    def _ask(self, history: list[dict], tools: list[dict] | None, max_tokens: int = 180) -> Call:
        return chat_call(
            self.client,
            self.budget,
            self.resolved.base_url,
            self.resolved.model,
            history,
            tools,
            self.timeout_s,
            max_tokens,
        )

    def run_turn(self, turn: Turn, history: list[dict]) -> tuple[Call, str | None]:
        history.append({"role": "user", "content": prepare_user(turn.user)})
        call = self._ask(history, self.resolved.tools)
        if call.aborted:
            self.say(f"  !! {turn.key:14s} {call.error}")
            return call, None
        line = f"  {turn.key:14s} {call.seconds:5.2f}s {call.tokens:4d} tok"
        if call.tool_calls:
            line += "  calls=" + ",".join(name for name, _ in call.tool_calls)
        line += f'  "{call.text.strip()[:58]}"' if call.text.strip() else ""
        self.say(line)
        assistant: dict = {"role": "assistant", "content": call.text or None}
        if not call.raw_calls:
            history.append(assistant)
            return call, None
        assistant["tool_calls"] = call.raw_calls
        history.append(assistant)
        offered = tool_names(self.resolved.tools)
        for tc, (name, args) in zip(call.raw_calls, call.tool_calls):
            result = stub_result(name, args) if name in offered else unknown_result(name)
            history.append(
                {"role": "tool", "tool_call_id": str(tc.get("id") or name), "name": name, "content": json.dumps(result)}
            )
        if not turn.follow_up:
            return call, None
        follow = self._ask(history, self.resolved.tools)
        if follow.aborted:
            self.say(f"  !! {turn.key}/reply {follow.error}")
            return call, ""
        self.say(
            f'  {turn.key + "/reply":14s} {follow.seconds:5.2f}s {follow.tokens:4d} tok  "{follow.text.strip()[:58]}"'
        )
        history.append({"role": "assistant", "content": follow.text or None})
        return call, follow.text

    def run(self) -> tuple[list[Verdict], list[tuple[Turn, Call, str | None]]]:
        history: list[dict] = [{"role": "system", "content": self.resolved.system_prompt()}]
        done: list[tuple[Turn, Call, str | None]] = []
        verdicts: list[Verdict] = []
        for turn in SCRIPT:
            call, final = self.run_turn(turn, history)
            done.append((turn, call, final))
            verdicts.extend(
                grade_aborted(turn, call) if call.aborted else grade_turn(turn, call, final, now=time.time())
            )
        verdicts.extend(self.judge(done))
        return verdicts, done

    def judge(self, done: list[tuple[Turn, Call, str | None]]) -> list[Verdict]:
        """The fuzzy half, on the same local model. Self-judging is a known limit of a
        one-GPU booth — the judge is not independent of the judged, which is exactly
        why its verdicts print apart from the rules and never merge with them.
        """
        transcript = render_transcript([t for t in done if not t[1].aborted])
        out: list[Verdict] = []
        for key, _rubric in RUBRICS.items():
            call = self._ask(judge_messages(key, transcript), None, max_tokens=120)
            if call.aborted:
                out.append(Verdict("abort", f"judge:{key}", False, f"ABORTED: {call.error}"))
                self.say(f"  !! judge:{key} {call.error}")
                continue
            passed, why = parse_judge_verdict(call.text)
            out.append(Verdict("judge", key, passed, f"{why} [{call.seconds:.2f}s]"))
            self.say(f"  judge:{key:20s} {call.seconds:5.2f}s  {why[:60]}")
        return out


def report(verdicts: list[Verdict], budget: Budget) -> int:
    """The verdict sets, one section each, never averaged."""
    failures = 0
    for group, title in (
        ("config", "CONFIGURATION (resolved from envs/<env>.yaml; no model involved)"),
        ("rule", "RULE VERDICTS (deterministic text and tool checks: facts about the replies)"),
        ("judge", "JUDGE VERDICTS (rubric calls on the same local model: opinions, not facts)"),
        ("abort", "ABORTS (no verdict about the configuration; the model never answered)"),
    ):
        got = [v for v in verdicts if v.group == group]
        if not got:
            continue
        print(f"\n== {title} ==")
        for v in got:
            print(v.line())
        ok = sum(1 for v in got if v.passed)
        failures += len(got) - ok
        print(f"{group}: {ok}/{len(got)} passed")
    print(f"\nmodel calls: {budget.used}/{budget.max_calls} used, {budget.remaining} spare, {budget.refused} refused")
    print(
        f"REHEARSAL FAILED: {failures} red verdict(s). Rules and judge are separate gates; "
        "there is no combined score to average them into."
        if failures
        else "REHEARSAL PASSED: every rule and every rubric verdict is green."
    )
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default="hackday", help="deployment environment to rehearse (default: hackday)")
    ap.add_argument("--base-url", default="", help="override the resolved llm_base_url")
    ap.add_argument("--model", default="", help="override the resolved llm_model")
    ap.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    ap.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP_S)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--plan", action="store_true", help="resolve and audit only: ZERO model calls")
    ap.add_argument("--json", action="store_true", help="also dump the verdicts as JSON")
    args = ap.parse_args(argv)

    print(f"resolving {args.env!r} with `REACHY_ENV={args.env} python -m shell --print-env` (opens nothing)")
    raw = run_print_env(args.env)
    try:
        schemas = builtin_schemas()
    except Exception as e:  # a hardware-free checkout has no SDK: the policy half still runs
        print(
            f"NOTE: real tool schemas unavailable ({e.__class__.__name__}: {str(e)[:120]}); "
            "offering no tools, so the conversation half cannot be judged"
        )
        schemas = []
    memories = MemoryStore(REPO / str(raw.get("memories_path") or f"data/memories.{args.env}.json"))
    resolved = resolve(args.env, raw, schemas, memory_block=memories.prompt_block())
    resolved.base_url = args.base_url or resolved.base_url
    resolved.model = args.model or resolved.model
    print(
        f"  model={resolved.model} at {resolved.base_url}; prompt={len(resolved.system_prompt())} chars "
        f"(memory block {len(resolved.memory_block)}); tools offered={len(resolved.tools)}"
    )
    print(f"  offered: {', '.join(sorted(tool_names(resolved.tools)))}")
    verdicts = audit_config(resolved)
    if args.plan or not schemas:
        print("\nno model calls made." if args.plan else "\nrefusing to run the conversation with no tools.")
        return report(verdicts, Budget(max_calls=0, min_gap_s=0.0, sleep=lambda _s: None))

    budget = Budget(max_calls=args.max_calls, min_gap_s=args.min_gap)
    print(
        f"\nconversation plan: {len(SCRIPT)} scripted turns, {budget.max_calls} call slots, >= {budget.min_gap_s}s apart"
    )
    for turn in SCRIPT:
        need = "+".join(turn.expect) or ("no tool" if turn.no_tool else "any")
        forbid = f" never {'+'.join(turn.forbid)}" if turn.forbid else ""
        print(f"  {turn.key:14s} {turn.user!r:38s} need={need}{forbid}  {turn.note}")
    with httpx.Client() as client:
        if not endpoint_up(client, resolved.base_url):
            print(
                f"\nthe LLM at {resolved.base_url} is not answering /models, so nothing about the booth's "
                "behaviour could be judged. The configuration audit above stands on its own; use --plan for that."
            )
            return 2
        print("\nconversation:", flush=True)
        rehearsal = Rehearsal(resolved, budget, client, timeout_s=args.timeout)
        got, done = rehearsal.run()
        verdicts.extend(got)
        print("\n== TRANSCRIPT (as the guest heard it) ==")
        print(render_transcript([(t, c, f) for t, c, f in done if not c.aborted]))
        if args.json:
            print("\n" + json.dumps([v.__dict__ for v in verdicts], indent=1))
        return report(verdicts, budget)


if __name__ == "__main__":
    raise SystemExit(main())
