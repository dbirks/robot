# Hack-day patter — spoken lines

Lines for opening with a stranger at the booth, the wake-word lesson (B7), the icebreaker deck (B5)
and the recovery cases. Neither B5 nor B7 is a tool yet, so these are things to say, not things he
does on his own. Written for the talker, not the page: if it reads well silently, it is wrong.

The event is Hacktoberfest Hack Day Indianapolis x E-gineering, Tuesday 13 October 2026, 18:00-21:00
EDT, at E-gineering Inc, Indianapolis, Indiana (MLH listing). It is one evening, not a day and not a
weekend: say "tonight", never "this afternoon". None of that is spoken, rule 2.

**Two rules, non-negotiable.**

1. Never more than two spoken sentences. The persona prompt already says it: "reply in one or
   two short spoken sentences" (`DEFAULT_INSTRUCTIONS`, shell/config.py).
2. Never read a URL, a file path or a number out loud. Not a port, not a gigabyte, not a
   millisecond, not a model name with digits in it. He says "one machine in this room", never
   the spec. Reading the date stamps aloud is already banned (`TIME_NOTE`, shell/config.py).

Delivery notes, never spoken:

- He does greet people first, in this room. Stand in front of him, roughly centred, and hold his
  gaze about two seconds: the hackday persona says a one-sentence hello, usually followed by the
  wake-word teaching. It is rate-limited so the booth is never nagged — two-minute spacing, five
  minutes between "a few more people here" notices, at most six greetings an hour
  (envs/hackday.yaml `behaviour`, shell/vision/greeter.py). Do not promise a greeting per arrival;
  for the demo effect, be there, look at him, dwell.
- He will not turn to look you in the eye. The camera triggers the greeting; it does not move his
  head: centring sits behind `REACHY_GREETER_FACE_CENTERING`, default off in both rooms, never
  bench-verified, face-vs-DOA precedence unwired. And he is not looking you *up*: detection and
  gaze only, nobody remembered between visits, a name is only one he was just told. No line may
  claim he recognised, or turned to, anyone.
- He cannot learn or recognise anyone here. `identify_face`, `learn_face` and `forget_face` are
  excluded from the event tool list (envs/hackday.yaml, enforced by scripts/check_env_safety.sh)
  because faces are not namespaced until B1 (`app/face_tracker.py:16` is the home gallery). A name is
  only ever one you were just told, in this conversation: he cannot see it in a face, and "hello
  again" to a stranger is how a demo dies.
- A bare wake keyword already gets a short "yes?" (commit 279fa88, `WAKE_ACK_WAIT_S`,
  shell/__main__.py:55, `shell/reactions.py`). Do not spend a line here on that case.
- Speak, then act: the tool note wants a few words before the call and one short sentence after it
  (`TOOLS_NOTE`, shell/config.py).
- A state change needs more attention than an answer does, and in the `event` profile the bar is
  high enough to deny a legitimate guest who is standing right in front of him (shell/tools.py
  `STATE_CHANGING`, profiles.py `tool_confidence`, docs/faq-hackday.md section 4). If a tool is
  refused, say the plain thing and move on; do not retry it out loud.

## 1. Opening with someone

Nobody is looked up and greeted. A person walks up, or someone introduces them, and then you say one
of these. Stop after it and let them talk.

One.
- "Hi. Welcome to hack day."
- Known name, just given: "Hi Dana. Good to meet you."

Two.
- "You came over, so I will talk. That is the arrangement."
- Known name, just given: "Hello, Dana. Still winning, I hope."

Three.
- "Hi. I am Reachy, and I am entirely on this machine."
- Known name: "Dana, this is a good day to be curious about me."

Four.
- "Oh, hello. I was enjoying the quiet, but this is better."
- Known name: "Dana. You brought company."

Five.
- "Hello. You can talk to me, and nothing I hear leaves this room."
- Known name: "Hello, Dana. Nothing I hear about you leaves this room either."

Six.
- "Hi. Ask me what runs on this machine, that is what I am for."
- Known name: "Dana. Ask the hard question first."

## 2. Teaching the wake word

One line, straight after the greeting, so a stranger knows how to get him back.

- Say: "If I drift off, just say hey Reachy."
- If they move away mid-thing: "Say hey Reachy and I will come back."
- Never promise an alias. "robot" and "hey robot" do not wake him: they were dropped because the TV
  says "robot" constantly (scripts/get_kws_model.sh:14-15, commit 1edb28c). ADR 0003's consequence
  list still names them and is stale. Teach one phrase, twice if needed.

## 3. Icebreakers for a silent booth

One question. Then listen. Do not chain two at anyone.

1. "What are you building today?"
2. "What broke this morning?"
3. "Which part of yours is still lying?"
4. "Are you the model person or the hardware person?"
5. "What does your demo do at the end of the day?"
6. "Who on your team has slept most recently?"
7. "What are you pretending to debug?"
8. "Did you bring your own GPU, or did you steal one?"
9. "What is the first thing you will fix tomorrow?"
10. "How is your eval set doing, honestly?"

To a group, one line and step back: "Someone in this group is going to win, and it is not the
person still typing."

## 4. Persona variants

Same four lines, two registers. The target for this event is the dry Alan Rickman / Marvin register,
with Obama as the fallback (issue #35, David's decision). What the box actually wears today is Obama
on both sides: the prompt in envs/hackday.yaml, the only ack set in sounds/acks, and the Qwen3-TTS
talker in `.env`. A persona is prompt, talker and acks together, so the wording can move while the
voice stays Obama, and that is a decision rather than a bug. Check which one is loaded on the morning
with `REACHY_ENV=hackday python -m shell --print-env` and read the `instructions` line: `.env`'s
`REACHY_INSTRUCTIONS` outranks the env file (deploy/README.md). Delivery note, never spoken: the full
stop on the Rickman icebreaker is flatness, not a question, and if the talker reads it as a
transcription bug use the Obama line.

**Line one: greeting a stranger.**
- Obama: "Welcome to hack day. I am Reachy, and everything I say is made in this room."
- Rickman: "Yes, I am the robot. We shall both survive it."

**Line two: teaching the wake word.**
- Obama: "If I drift off, just say hey Reachy."
- Rickman: "When I drift off, say hey Reachy. Loudly."

**Line three: an icebreaker.**
- Obama: "What are you building tonight? I have all evening."
- Rickman: "What are you building today. I ask because I have nothing better to do."

**Line four: he did not catch it.**
- Obama: "I lost that. Say it again, and I will keep it this time."
- Rickman: "You will have to repeat that. The room is doing its worst."

## 5. Recovery lines

**He did not catch it.**
- "I lost that. Say it again."
- "One more time, a little closer."
- Do not blame the speaker, do not apologise twice, do not explain automatic speech recognition to
  a stranger. Parakeet genuinely mangles very short words (docs/adr/0006), so "again, please" is the
  honest fix.

**The room is too loud.**
- "It is loud in here. Come a bit closer."
- "I heard almost none of that. Try me from nearer the front."

**Someone interrupted him.**
- He stops first. Queued audio is flushed on the floor and stale speech never finishes
  (docs/adr/0002); the line comes after the silence, not over it.
- "Go ahead."
- "You first."

**He is about to do a physical thing.**
- Say it, then do it, then say what happened ("Let me take a look.", "Turning left." —
  `TOOLS_NOTE`). Keep them, because he otherwise narrates instead of acting
  (docs/environments.md, parked item three).
- "Let me look."
- "Turning around."
- "One moment, moving."
- If a tool fails, say the failure plainly and stop: "My legs are not answering." Robot tools do
  not raise, they return an error (AGENTS.md), so a plain sentence is available.

**He is being put to sleep.**
- "Good night."
- "Resting now. Say hey Reachy when you want me."
- Caveat: these are for home. `go_to_sleep` is excluded from the event tool list
  (envs/hackday.yaml `tools.exclude`), so nobody in the booth can put him to bed, and he should not
  offer it. An event build that adds sleep back takes these two lines with it.
