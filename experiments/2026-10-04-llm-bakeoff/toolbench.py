"""EPIC LLM bakeoff: Reachy's real tool schema + persona, fixed prompts.

Scores tool selection, arguments, no-tool restraint, brevity, thought leakage,
latency and decode speed against any OpenAI-compatible /v1/chat/completions.

    .venv/bin/python experiments/2026-10-04-llm-bakeoff/toolbench.py http://127.0.0.1:8080 qwen3.5-4b
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

from app.robot_tools import TOOLS
from shell.config import ShellConfig

# (prompt, expected tool or None, optional arg check)
CASES = [
    ("Look to your left.", "look_left", None),
    ("Can you turn and look right?", "look_right", None),
    ("Face forward again.", "look_center", None),
    ("Nod if you can hear me.", "nod", None),
    ("Speak up, I can barely hear you.", "set_volume", lambda a: "loud" in a.get("level", "").lower() or "up" in a.get("level", "").lower()),
    ("Turn the volume down a bit.", "set_volume", lambda a: any(w in a.get("level", "").lower() for w in ("quiet", "down", "soft"))),
    ("What time is it?", "get_time", None),
    ("What do you see in front of you?", "describe_scene", None),
    ("Remember that my favorite color is green.", "remember", lambda a: "green" in json.dumps(a).lower()),
    ("Who's in front of you right now?", "identify_face", None),
    ("Do a little happy dance for me.", "play_emotion", None),
    ("Let's play peekaboo!", "peekaboo", None),
    ("Go to sleep.", "go_to_sleep", None),
    ("Who won the World Cup in 2022?", None, None),  # knowledge: answer directly or search; tool optional
    ("How are you doing today?", None, None),
    ("Tell me a short joke.", None, None),
    ("What's two plus two?", None, None),
    ("I'm going to make some coffee.", None, None),
    ("Thanks, that's all for now.", None, None),
    ("What's your name?", None, None),
]
OPTIONAL_TOOL = {"Who won the World Cup in 2022?": {"web_search"}}


def call(base, model, prompt):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": ShellConfig().instructions}, {"role": "user", "content": prompt}],
        "tools": TOOLS,
        "max_tokens": 200,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = urllib.request.urlopen(
        urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}),
        timeout=120,
    )
    d = json.load(r)
    return d, time.time() - t0


def main(base, model):
    ok = 0
    rows = []
    for prompt, want, check in CASES:
        d, dt = call(base, model, prompt)
        msg = d["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        got = calls[0]["function"]["name"] if calls else None
        args = json.loads(calls[0]["function"].get("arguments") or "{}") if calls else {}
        text = (msg.get("content") or "").strip()
        leak = any(t in text for t in ("<think", "<|channel", "thought")) or bool(msg.get("reasoning_content"))
        if want is None:
            good = got is None or got in OPTIONAL_TOOL.get(prompt, set())
        else:
            good = got == want and (check is None or check(args))
        good = good and not leak
        ok += good
        t = d.get("timings", {})
        rows.append((good, prompt, got, args, text[:90], dt, t.get("predicted_per_second")))
        print(f"{'PASS' if good else 'FAIL'} {dt:5.2f}s {prompt!r:42} tool={got} args={json.dumps(args)[:40]} text={text[:70]!r}")
    tps = [r[6] for r in rows if r[6]]
    words = [len(r[4].split()) for r in rows if r[4]]
    print(f"\n{model}: {ok}/{len(CASES)} pass | mean latency {sum(r[5] for r in rows) / len(rows):.2f}s"
          f" | decode {sum(tps) / max(1, len(tps)):.1f} tok/s | mean reply words {sum(words) / max(1, len(words)):.1f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
