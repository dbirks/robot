# Reachy — booth card

Print this. It answers the four questions every visitor asks, so he can stay busy.

## Who is he

Reachy Mini, a small open-source desk robot. He is running a **fully local voice
assistant** — speech in, conversation out, no cloud inference. One machine:
a 2016 four-core i7 and an 8 GB GTX 1070, running the chat model, the speech
recogniser and the voice synthesiser at the same time.

## Reading the journal on this box

`sqlite3` here is NOT broken, and the note I first wrote here was. What fails is
`datetime(wall,'localtime')`: without the `unixepoch` modifier SQLite reads the number as
a Julian day and returns `1999-12-31 19:00:00`, which looks exactly like a broken clock.
The stored `wall` is a correct unix epoch (`1791857303.45`); the working form is

    sqlite3 -readonly data/events.db \
      "select datetime(wall,'unixepoch','localtime'), type from events order by id desc limit 5"

`mono` is separate and genuinely treacherous: seconds since the *writing process* started
(ADR 0007), so it resets on every restart - never subtract across one. Same symptom, two
different causes, and mine was the rarer of the two: I had written the wrong query and
blamed the tool. Three tool calls and a wrong doc note to learn it, so the correct
diagnosis is written down: (a) missing `unixepoch` -> bogus 1999 date, (b) real clock
reset -> absurd durations.


## Talk to him

Say **"Reachy"** — then your question, in the same breath:

> "Reachy, what's the weather in Indianapolis?"
> "Reachy, what do you see?"
> "Reachy, tell me a joke."

He also says hello first, if you look at him for a couple of seconds.
He answers whoever is talking to him, and deliberately ignores the rest of the room.

## What runs where (the honest version)

- Your speech is transcribed on this machine by NVIDIA Parakeet. Nothing is uploaded.
- He answers with a local Qwen 3.5 4B model running on this machine's GPU.
- He speaks with a locally generated voice. **His voice is synthetic**, and his manner
  is a written persona, not a person.
- **One exception:** if you ask him something current — weather, news, facts he isn't
  sure about — he runs a web search, which leaves the machine and comes back.
- He does not recognise faces at this event. If he asks you to "remember my face",
  say yes and he'll explain that it's switched off today.
- Everything he says is journalled to a local SQLite file for debugging. Ask and we'll
  show you the record.

## The interesting answers

| Ask him | You'll see |
|---|---|
| "What's on your GPU?" | The dashboard's GPU panel — model + voice occupying nearly all 8 GB |
| "What tools do you have?" | The live tool list on screen, including which ones need proof you're talking to him |
| "How did you hear me over the noise?" | The attention lease: he only listens when you have his attention |
| "Show me what that took" | The **Traces** page — one row per turn: listen, transcribe, think, first audio, tools, in milliseconds, read back off the journal |
| "What's running right now?" | `http://localhost:3001` on the booth machine — services, conversation, environment |

## If he stops responding

1. Wake word only, once. If he answers, you were talking over him — he goes quiet while
   he speaks and won't always catch an interruption.
2. Check `:3001` — if a service row is red, restart it from the panel (attendees welcome).
3. Still nothing? Find David. The three numbers worth knowing: wake word = "Reachy",
   dashboard = port 3001, model = Qwen 3.5 4B.

*Hacktoberfest Hack Day Indianapolis × E-gineering · 13 October 2026*
