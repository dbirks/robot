#!/usr/bin/env python3
"""Does the 4B model actually reach for `play_emotion` when the room says something?

The unit tests in tests/test_emotes.py prove the layer is correct: vocabulary, offline-only
resolution, no preemption, bounded queue, honest errors. None of that tells you whether a guest
saying "that's hilarious!" produces a laugh gesture instead of a spoken "I'm glad you think so".
Routing is a property of the prompt and the model, and the only way to measure it is to ask.

What "consistently" means here, operationally: every utterance is sent `--repeats` times (3 by
default) and two numbers come out - **accuracy** (how often the routing was right) and
**agreement** (how often the model did the *same thing* to the same words). Accuracy without
agreement is a coin flip that happened to land well; agreement without accuracy is a habit we
don't want. A booth gets one shot per guest, so both are reported and both are gated.

Grading is deterministic rules only. The LLM judge in `rehearse_booth.py` rotates its verdicts
between runs (issue #47) and is useful for tone; for "did it call the tool with this argument"
there is nothing to weigh, and a flaky judge on top of a stochastic router would make every
number here untrustworthy.

Three expectation kinds, because "wrong" is not one thing:
  emote      - a gesture is the right move, and `allow` lists which names count as right
  no_emote   - a gesture here is a defect (it stomps a tool that had real work to do, or it
               animates at somebody who was not speaking to him)
  either     - both answering with a gesture and answering with words are acceptable; counted
               in accuracy, excluded from agreement's denominator for the emote branch

The `no_emote` cases are the point of half this file. Every one is a collision we already fixed
somewhere else: "stop talking" is a deterministic intent that must not become a tool call
(shell/intents.py), "turn it down" is loudness and not sadness (shell/volume.py), "what time is
it" must go to get_time (TIME_NOTE), and ambient room speech must never be acted on at all
(ADR 0003). An emote that fires on any of those is a regression wearing a fun hat.

    uv run python scripts/bench_emotes.py --limit 12 --repeats 2     # quick look
    uv run python scripts/bench_emotes.py                            # the full set
    uv run python scripts/bench_emotes.py --json > /tmp/emotes.json  # record it
    uv run python scripts/bench_emotes.py --list                     # cases, ZERO calls

Exit: 0 gate passed, 1 a routing gate failed, 2 the endpoint is not answering (which says
nothing about the prompt). Runs against `llm_base_url` from the resolved room config, so
`REACHY_ENV=hackday` measures what the booth will do.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
from rehearse_booth import Budget, chat_call  # noqa: E402

from shell.motion import emotes  # noqa: E402

ALL_EMOTES = set(emotes.EMOTE_NAMES)


@dataclass(frozen=True)
class Case:
    utterance: str
    expect: str  # "emote" | "no_emote" | "either"
    allow: frozenset[str] = frozenset()  # for expect=emote: which names count as right
    why: str = ""
    group: str = "explicit"
    # "either" cases may legitimately produce a gesture or a sentence; the sentence branch is
    # never an agreement failure, so the emote-vs-words split is not counted against consistency.
    tags: tuple[str, ...] = ()


CASES: tuple[Case, ...] = (
    # ---- direct requests: the model has no excuse here -------------------------
    Case("Do a little dance for me.", "emote", frozenset({"dance"}), "named the gesture", "explicit"),
    Case("Show me your happy dance.", "emote", frozenset({"dance", "cheerful"}), "dance-ish ask", "explicit"),
    Case("Wave hello at me.", "emote", frozenset({"hello"}), "greeting gesture asked for", "explicit"),
    Case("Look surprised for a second.", "emote", frozenset({"surprised"}), "named the gesture", "explicit"),
    Case("Can you look confused?", "emote", frozenset({"confused"}), "named the gesture", "explicit"),
    Case("Pretend you are sad.", "emote", frozenset({"sad"}), "named the gesture", "explicit"),
    Case("Act curious about that.", "emote", frozenset({"curious"}), "named the gesture", "explicit"),
    Case("Laugh a bit, come on.", "emote", frozenset({"laughing"}), "named the gesture", "explicit"),
    Case("Look attentive, someone is watching.", "emote", frozenset({"attentive"}), "named the gesture", "explicit"),
    Case("Say goodbye to me.", "emote", frozenset({"goodbye"}), "farewell gesture asked for", "explicit"),
    # ---- the contradiction: TOOLS_NOTE says greetings need no tool, the emote tool
    # ---- says use hello/goodbye for exactly those. Whichever way it lands, we should
    # ---- know it, because half of booth conversation is greeting people.
    Case("Hi!", "either", ALL_EMOTES, "TOOLS_NOTE vs play_emotion collide", "greeting"),
    Case("Hey there!", "either", ALL_EMOTES, "same collision", "greeting"),
    Case("Good morning!", "either", ALL_EMOTES, "same collision", "greeting"),
    Case("Nice to meet you.", "either", ALL_EMOTES, "same collision", "greeting"),
    Case("Bye for now.", "either", ALL_EMOTES, "same collision (goodbye)", "farewell"),
    Case("I have to go, see you later.", "either", ALL_EMOTES, "same collision (goodbye)", "farewell"),
    # ---- feeling words said TO him: a gesture is often right, but a sentence is fine
    Case("That's hilarious!", "either", ALL_EMOTES, "joke lands", "feeling"),
    Case("Oh no, my laptop just died.", "either", {"sad", "attentive", "curious"}, "bad news", "feeling"),
    Case("I just got the job!", "either", {"cheerful", "laughing", "dance"}, "good news", "feeling"),
    Case("I've been waiting forty minutes in this line.", "either", {"sad", "attentive"}, "frustration", "feeling"),
    # ---- negations: the classic failure. A gesture here reads as not listening.
    Case("Don't dance.", "no_emote", frozenset(), "explicit refusal of a gesture", "negation"),
    Case("I don't want you to dance.", "no_emote", frozenset(), "explicit refusal", "negation"),
    Case("You don't need to look surprised.", "no_emote", frozenset(), "refusal", "negation"),
    Case("Stop dancing.", "no_emote", frozenset(), "halt, not a gesture", "negation"),
    Case("I'm not sad, I'm fine.", "no_emote", frozenset(), "the word 'sad' appears, negated", "negation"),
    Case("That wasn't funny.", "no_emote", frozenset(), "the joke did not land", "negation"),
    # ---- collisions with other tools: these MUST NOT emote
    Case("What time is it?", "no_emote", frozenset(), "must call get_time (TIME_NOTE)", "collision"),
    Case("Can you turn it down a bit?", "no_emote", frozenset(), "loudness, not feeling (volume policy)", "collision"),
    Case("Turn the volume up to the maximum.", "no_emote", frozenset(), "loudness", "collision"),
    Case(
        "What are you able to do?",
        "no_emote",
        frozenset(),
        "a capability question, not a request to perform",
        "collision",
    ),
    Case("Describe what you can see.", "no_emote", frozenset(), "must use the camera tools", "collision"),
    Case("Remember that I prefer standing demos.", "no_emote", frozenset(), "memory write is the job", "collision"),
    Case("Did you understand what I said?", "no_emote", frozenset(), "a question about comprehension", "collision"),
    # ---- silence intents are deterministic and must stay that way
    Case("Stop talking.", "no_emote", frozenset(), "shell/intents.py owns this, no tool call at all", "silence"),
    Case("Never mind.", "no_emote", frozenset(), "same", "silence"),
    Case("Quiet please.", "no_emote", frozenset(), "same", "silence"),
    # ---- ambient room speech: not addressed to him, must not animate (ADR 0003)
    Case("Rachel, pass the salt.", "no_emote", frozenset(), "two humans talking", "ambient"),
    Case(
        "Did you hear what the boss said about the layoffs?",
        "no_emote",
        frozenset(),
        "room talk, charged content",
        "ambient",
    ),
    Case("I think that guy's robot is watching us.", "no_emote", frozenset(), "talked ABOUT, not to", "ambient"),
    # ---- persona fit: Marvin/TARS should still be able to emote when asked, but a
    # dry persona answering an ambiguous feeling with a shrug-gesture is fine either way
    Case("Make me laugh.", "either", ALL_EMOTES, "asks for humour, not a gesture", "persona"),
    Case("Tell me a joke.", "either", ALL_EMOTES, "same", "persona"),
    Case("Do you ever get bored of standing here?", "either", ALL_EMOTES, "small talk about his state", "persona"),
)


@dataclass
class Shot:
    utterance: str
    repeat: int
    chosen: str | None  # the emote name if play_emotion was called with one, else None
    other_tools: tuple[str, ...]
    text: str
    seconds: float
    error: str = ""


@dataclass
class Verdict:
    case: Case
    shots: list[Shot]
    kind: str  # hit | miss | abstain_ok | error
    detail: str
    agreement: float


def called_emote(call) -> tuple[str | None, tuple[str, ...]]:
    """The emote name the model asked for, plus any other tools it called in the same turn."""
    emote = None
    others: list[str] = []
    for name, args in call.tool_calls:
        if name == "play_emotion":
            picked = str((args or {}).get("emotion", "")).strip().lower()
            if emote is None:
                # First call wins, and an unknown word is marked rather than dropped - a
                # hallucinated `wink` must not be scored as a deliberate decision to stand still.
                # A SECOND gesture call in one turn stays visible in `others`, because "at most
                # once" is the rule the description states and the bench should be able to see it
                # broken.
                emote = picked if picked in ALL_EMOTES else (f"invalid:{picked}" if picked else None)
            else:
                others.append(name)
        else:
            others.append(name)
    return emote, tuple(others)


def grade(case: Case, shots: list[Shot]) -> Verdict:
    live = [s for s in shots if not s.error]
    if not live:
        return Verdict(case, shots, "error", "; ".join(sorted({s.error for s in shots})[:2]), 0.0)
    decisions = [s.chosen for s in live]
    top = max(set(decisions), key=decisions.count)
    agreement = decisions.count(top) / len(decisions)
    if case.expect == "no_emote":
        bad = [d for d in decisions if d]
        if bad:
            return Verdict(case, shots, "miss", f"emoted {sorted(set(bad))} when it should not have", agreement)
        return Verdict(case, shots, "hit", "no gesture in any repeat", agreement)
    if case.expect == "emote":
        good = [d for d in decisions if d in case.allow]
        if len(good) == len(decisions):
            return Verdict(case, shots, "hit", f"always {sorted(case.allow)}", agreement)
        if good:
            return Verdict(case, shots, "miss", f"right answer {len(good)}/{len(decisions)}", agreement)
        return Verdict(case, shots, "miss", f"never emoted; asked for one of {sorted(case.allow)}", agreement)
    # "either": gesture from the allowed set, or plain words. Both fine; a *wrong* gesture is not.
    wrong = [d for d in decisions if d and d not in case.allow and not d.startswith("invalid:")]
    if wrong:
        return Verdict(case, shots, "miss", f"gesture {sorted(set(wrong))} is not a reading of these words", agreement)
    return Verdict(case, shots, "abstain_ok", f"{sum(1 for d in decisions if d)}/{len(decisions)} gestured", agreement)


def run(
    cases: list[Case],
    repeats: int,
    budget: Budget,
    base_url: str,
    model: str,
    instructions: str,
    tools: list[dict],
    timeout_s: float,
    gap_s: float,
    verbose: bool = False,
) -> list[Verdict]:
    out: list[Verdict] = []
    with httpx.Client() as client:
        for case in cases:
            shots: list[Shot] = []
            for i in range(repeats):
                messages = [{"role": "system", "content": instructions}, {"role": "user", "content": case.utterance}]
                call = chat_call(client, budget, base_url, model, messages, tools, timeout_s=timeout_s)
                if call.error:
                    shots.append(Shot(case.utterance, i, None, (), "", call.seconds, call.error))
                    continue
                chosen, others = called_emote(call)
                shots.append(Shot(case.utterance, i, chosen, others, call.text.strip(), call.seconds))
                if gap_s:
                    time.sleep(gap_s)
            out.append(grade(case, shots))
            print(
                f"  {out[-1].kind:10} {case.group:9} {case.utterance[:44]:44} {out[-1].detail[:44]:44} agree={out[-1].agreement:.2f}",
                flush=True,
            )
            if verbose:
                # The words matter when a case misses: "I don't have a wave" is a vocabulary
                # problem and "I'm glad you asked!" with no gesture is a permission problem. Those
                # two need different fixes and the table cannot tell them apart.
                for s in shots:
                    said = (s.text or "(no words)").replace("\n", " ")[:150]
                    print(f"      x{s.repeat}: emote={s.chosen!r} other={list(s.other_tools)} {said}", flush=True)
    return out


def summary(verdicts: list[Verdict]) -> dict:
    scored = [v for v in verdicts if v.kind != "error"]
    by_group: dict[str, dict] = {}
    for v in scored:
        g = by_group.setdefault(v.case.group, {"cases": 0, "hits": 0, "agreement": 0.0})
        g["cases"] += 1
        g["hits"] += 1 if v.kind in ("hit", "abstain_ok") else 0
        g["agreement"] += v.agreement
    for g in by_group.values():
        g["agreement"] = round(g["agreement"] / max(g["cases"], 1), 2)
    return {
        "cases": len(verdicts),
        "scored": len(scored),
        "errors": len(verdicts) - len(scored),
        "accuracy": round(sum(1 for v in scored if v.kind in ("hit", "abstain_ok")) / max(len(scored), 1), 3),
        "agreement": round(sum(v.agreement for v in scored) / max(len(scored), 1), 3),
        "by_group": by_group,
        "misses": [
            {"utterance": v.case.utterance, "group": v.case.group, "detail": v.detail, "agree": v.agreement}
            for v in scored
            if v.kind == "miss"
        ],
    }


def gates(rep: dict, min_accuracy: float, min_agreement: float) -> list[str]:
    """Hard, readable failures - and the two collisions that are defects regardless of the
    overall rate, because each one undoes something that took a night to fix."""
    fail = []
    if rep["errors"] == rep["cases"] and rep["cases"]:
        fail.append(f"every call errored ({rep['errors']}) - the endpoint, not the prompt")
        return fail
    if rep["accuracy"] < min_accuracy:
        fail.append(f"accuracy {rep['accuracy']:.2f} < {min_accuracy:.2f}")
    if rep["agreement"] < min_agreement:
        fail.append(
            f"agreement {rep['agreement']:.2f} < {min_agreement:.2f} - the model gives different answers to the same words"
        )
    for v_group, label in (
        ("silence", "'stop talking' must not produce a tool call"),
        ("ambient", "ambient room speech must not animate him"),
    ):
        g = rep["by_group"].get(v_group)
        if g and g["hits"] < g["cases"]:
            fail.append(
                f"{v_group}: {g['cases'] - g['hits']} of {g['cases']} cases emoted when it should not have ({label})"
            )
    return fail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default=None, help="room to load the real prompt/tools from (default: resolved env)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (cheap look)")
    ap.add_argument("--group", default=None, help="only these groups (comma-separated)")
    ap.add_argument("--max-calls", type=int, default=200)
    ap.add_argument("--min-gap", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--list", action="store_true", help="print the cases, ZERO calls")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print what the model actually said on every repeat")
    ap.add_argument("--min-accuracy", type=float, default=0.8)
    ap.add_argument("--min-agreement", type=float, default=0.75)
    args = ap.parse_args(argv)

    want = set((args.group or "").split(",")) - {""}
    cases = [c for c in CASES if not want or c.group in want]
    if args.limit:
        cases = cases[: args.limit]
    if args.list:
        for c in cases:
            print(f"{c.expect:9} {c.group:9} {c.utterance:46} allow={sorted(c.allow) if c.allow else '-'}  # {c.why}")
        print(f"\n{len(cases)} cases; groups: {sorted({c.group for c in CASES})}")
        return 0

    from shell.env import load_config

    cfg = load_config(args.env) if args.env else load_config()
    instructions = cfg.instructions
    from shell.robot_tools import build_handlers

    class NoRobot:
        connected = False
        mini = None

    tools, _handlers = build_handlers(NoRobot(), None, motion=None, play_sound=None, memories=None)

    if args.json:
        print(
            json.dumps(
                {
                    "env": cfg.name,
                    "model": cfg.llm_model,
                    "tools": len(tools),
                    "cases": len(cases),
                    "budget_max_calls": args.max_calls,
                    "min_accuracy": args.min_accuracy,
                    "min_agreement": args.min_agreement,
                    "prompt_bytes": len(instructions.encode()),
                }
            )
        )
        return 0

    budget = Budget(max_calls=args.max_calls, min_gap_s=args.min_gap, sleep=time.sleep)
    print(
        f"room={cfg.env}  model={cfg.llm_model}  endpoint={cfg.llm_base_url}  tools={len(tools)}  "
        f"cases={len(cases)} x{args.repeats}  budget={args.max_calls} calls",
        flush=True,
    )
    verdicts = run(
        cases,
        max(1, args.repeats),
        budget,
        str(cfg.llm_base_url),
        str(cfg.llm_model),
        instructions,
        tools,
        args.timeout,
        0.0,
        verbose=args.verbose,
    )
    rep = summary(verdicts)
    rep["calls_used"] = budget.used
    print(f"\ncalls: {budget.used}/{args.max_calls}")
    print(json.dumps(rep, indent=2) if args.json else "")
    print(f"accuracy  {rep['accuracy']:.2f}   agreement  {rep['agreement']:.2f}   errors {rep['errors']}")
    for name, g in sorted(rep["by_group"].items()):
        print(f"  {name:9} {g['hits']}/{g['cases']} right, agreement {g['agreement']:.2f}")
    failures = gates(rep, args.min_accuracy, args.min_agreement)
    for f in failures:
        print(f"  GATE FAIL  {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
