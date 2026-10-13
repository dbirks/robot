# Does the model actually emote? Measured routing for `play_emotion`

`tests/test_emotes.py` proves the emote layer is correct. It cannot prove that a guest saying
"that's hilarious!" produces a laugh. Routing is a property of the prompt and the model, so it was
measured with [`scripts/bench_emotes.py`](../scripts/bench_emotes.py) against the live llama.cpp
(qwen3.5-4b, `REACHY_ENV=hackday`, the resolved booth prompt and all 19 tool schemas), 42 utterances
x 2 repeats = 84 calls, ~9 minutes, 0 errors.

## Why two numbers and not one

"Consistently" needs both, and they disagree:

- **accuracy** — how often the routing was right;
- **agreement** — how often the model did the *same thing* to the *same words* across repeats.

The first measurement was **accuracy 0.59 / agreement 1.00**: perfectly consistent, consistently
wrong. Reporting agreement alone would have called that stable behaviour.

## What the first run found (accuracy 0.59)

1. **0 of 4 greetings produced a gesture, every repeat.** `TOOLS_NOTE` says *"Greetings, small
   talk, jokes, feelings and goodbyes need no tool: just answer"* — while two of the ten gestures
   are named `hello` and `goodbye`. The prompt was contradicting the tool.
2. **7 of 10 explicit requests produced nothing.** And the same note says *"Narrate tool use out
   loud… say a few words about what you are doing"* — so it narrated gestures it never called:
   *"I am currently playing hello"*, *"I shall do my best to appear curious… I am turning my head"*.
   Narration had become a substitute for the call.
3. **The persona refused on philosophical grounds**: *"I am a machine, and I cannot feel surprise"*,
   *"my circuits are wired for dryness, not sorrow"*. Framing the tool as *"express an emotion"*
   hands a veto to any persona that denies feeling — and hackday's persona is Marvin.

## The fix, and the second run

Three changes, each targeting one of the above rather than all of them at once:

- `TOOLS_NOTE`: the greeting ban now excludes `play_emotion`, and states that **narrating a tool
  call is not making one** — the same lesson that fixed the clock (see `TIME_NOTE`, #45): *"'I am
  waving' moves nothing, so a gesture means CALLING the tool."*
- The tool description reframes the call as **a movement, not a claim about feelings** ("the same
  kind of thing as nodding"), so a dry persona can perform it without lying, and adds that the
  call is the **only** way his body moves.
- Home prompt 1765 → 1938 bytes, re-pinned in `tests/parity_baseline.py` with this reason attached.

Result: **accuracy 0.93, agreement 1.00**, and the safety groups are the part that matters:

| group | result | what it protects |
|---|---|---|
| ambient | 3/3, agreement 1.00 | two strangers talking must not animate him (ADR 0003) |
| collision | 7/7 | "turn it down" is loudness, "what time is it" is `get_time`, "what do you see" is the camera |
| negation | 6/6 | "don't dance", "I'm not sad", "that wasn't funny" |
| silence | 3/3 | "stop talking" stays a deterministic intent, no tool call |
| feeling / greeting / farewell / persona | 4/4, 4/4, 2/2, 3/3 | allowed either way; graded on gesture *plausibility* |
| explicit | 7/10 | the model is asked to perform, and should |

## The three misses, left in deliberately

- **"Wave hello at me."** → *"I am a robot, so I cannot wave, but I will give you a cheerful nod
  instead."* The gesture is named `hello`; the word "wave" does not map to it. Fix is vocabulary in
  the description (name the physical shapes), not a new gesture.
- **"Pretend you are sad."** → declines in character: *"wired for dryness, not sorrow"*. For Marvin
  that is arguably correct behaviour and it is graded as a miss anyway, on purpose — the bench
  measures the persona+model pair, and if we want Marvin to perform sadness on request we should
  decide that deliberately (issue #52: persona-scoped vocabularies).
- **"Look attentive, someone is watching."** → *"I am attentive. I am looking."* He argues that he
  already is. Same narration shape as the greeting case, one instance left.

## What this does not prove

- **Not motion.** The desk robot was unpowered: gestures were verified as finite poses with honest
  durations, never observed on servos. Re-run at the booth with `--verbose` and *watch him*.
- **Not the audio path.** The bench goes to llama.cpp over chat-completions with the same
  instructions and tools the shell sends, but it does not traverse the realtime service, the
  per-turn memory block, or ASR text errors ("reachy" heard as "richie").
- **Not stable across personas or model swaps.** Every number here is one model, one prompt, one
  evening, temperature as served. Re-run after any prompt change — the cost is 9 minutes, and
  agreement of 1.00 on 2 repeats is not proof of determinism.
- Grading is deterministic rules only. The LLM judge in `rehearse_booth.py` rotates verdicts between
  runs (#47); for "did it call the tool with this argument" there is nothing to weigh.

Re-run:

```
REACHY_ENV=hackday uv run python scripts/bench_emotes.py --repeats 2      # the recorded run
uv run python scripts/bench_emotes.py --group explicit --verbose          # what he said instead
uv run python scripts/bench_emotes.py --list                              # cases, zero calls
```
