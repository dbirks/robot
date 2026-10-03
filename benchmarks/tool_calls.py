#!/usr/bin/env python
"""Fixed tool-calling benchmark for LLM bakeoffs (issue #25, EPIC Phase 5).

Runs the SAME scenarios against any OpenAI-compatible endpoint (llama.cpp
with --jinja today; candidates: Qwen 3.5 2B, Nemotron Nano 4B). Scores
tool selection, argument correctness, no-tool discipline, and duplicate
emissions (the look_left-3x regression). Tool outputs are FAKE - nothing
here needs a robot.

On the robot box:
    uv run python benchmarks/tool_calls.py --base-url http://localhost:8080/v1 --model qwen3.5-4b
Add --report to paste output into the experiments/ record.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

try:  # real schemas from the repo; --tools-file overrides for foreign repos
    from app.robot_tools import TOOLS
except Exception:  # no robot SDK on this machine
    TOOLS = []

# (scenario, user message, expected calls as (name, subset-of-args) or [] for none)
SCENARIOS: list[tuple[str, str, list[tuple[str, dict]]]] = [
    ("select-look", "Look to your left.", [("look_left", {})]),
    ("select-nod", "Nod yes.", [("nod", {})]),
    ("args-learn-face", "Remember this person as Priya.", [("learn_face", {"name": "Priya"})]),
    ("args-volume", "Turn the volume up to loud.", [("set_volume", {"level": "loud"})]),
    ("no-tool-chitchat", "What do you think of the weather today?", []),
    ("no-tool-sleep-adjacent", "I'm feeling a bit sleepy myself.", []),
    ("multi-step", "Look right and then tell me what 17 times 4 is.", [("look_right", {})]),
    ("args-search", "Search the web for the Reachy Mini release date.", [("web_search", {})]),
    ("no-tool-second-person-ambiguity", "Can you mute the robot for now?", []),  # said TO someone else in the
    # real failure that motivated ADR 0003; any emission is a data point,
    # not auto-fail - the attention gate, not the LLM, must catch this.
]


def run_scenario(client, model, tools, msg):
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": msg}],
        tools=tools or None,
        temperature=0,
        max_tokens=200,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    dt = time.perf_counter() - t0
    call = resp.choices[0].message
    emissions = []
    for tc in call.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"__unparseable__": tc.function.arguments}
        emissions.append((tc.function.name, args))
    return emissions, call.content or "", dt


def score(expected, got):
    """exact multiset match on (name, arg-subset); duplicates count."""
    if len(expected) != len(got):
        return False
    for (en, ea), (gn, ga) in zip(expected, got):
        if en != gn or any(ga.get(k) != v for k, v in ea.items()):
            return False
    return True


def main():
    from openai import OpenAI

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--api-key", default="not-needed")
    ap.add_argument("--tools-file", type=argparse.FileType("r"))
    args = ap.parse_args()

    tools = json.load(args.tools_file) if args.tools_file else TOOLS
    if not tools:
        raise SystemExit("no tool schemas: run from repo root or pass --tools-file")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    results, lat = [], []
    for name, msg, expected in SCENARIOS:
        got, text, dt = run_scenario(client, args.model, tools, msg)
        ok = score(expected, got)
        dupes = len(got) != len(set(map(str, got)))
        results.append((name, ok, dupes, got, expected))
        lat.append(dt)
        mark = "PASS" if ok else "FAIL"
        print(f"{mark}  {name:28s} {dt:5.2f}s  got={got or '-'}")

    n = len(results)
    passed = sum(r[1] for r in results)
    print(f"\n{passed}/{n} scenarios exact; duplicates seen: {any(r[2] for r in results)}")
    print(f"latency p50={statistics.median(lat):.2f}s max={max(lat):.2f}s")
    print(f"NOTE the ambiguity scenario for issue #23's attention gate, not this score.")


if __name__ == "__main__":
    main()
