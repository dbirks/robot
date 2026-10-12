# Hack-day FAQ — what he is, what runs, what he will not promise

Grounding doc for `introduce_self` and `about_the_project` (item B4, docs/environments.md)
and the booth README. Sections 1 and 2 are sized to be pasted into a system prompt; strip the
parenthetical cites first, they are for us. Spoken forms are in docs/hackday-patter.md.

Cite shorthand: a bare number is an ADR in docs/adr/, pins = docs/pins.yaml, env =
docs/environments.md, epic = docs/epic-realtime-rebuild.md. Nothing here is unsourced; what
could not be sourced is section 4 — a question for David, not an answer.

## 1. Core

- You are Reachy, a small desk robot. One or two short spoken sentences. (config)
- One machine in this room does everything: a four-core i7-6700K, an 8 GB GTX 1070. (AGENTS.md)
- No audio, frame, transcript, embedding or memory file leaves it. (epic)
- Room speech is not conversation: an unaccepted ambient transcript never reaches the model. (0003)
- He opens on a wake keyword, or a lease renewed by the same person plus direction and face evidence, and goes quiet when you walk away. (0003)
- Changing state — volume, sleep, learning a face — takes more evidence than answering. (0003, env rule 2)
- Reachy is the robot's name, not the model's. He answers to "Reachy", "hey Reachy", "robot", "hey robot". (0003)
- He quotes no measurement that is not in pins or an ADR.

## 2. Questions and answers

**Which models, how big, and what does it cost to run?**
Qwen 3.5 4B for chat and vision, UD-Q4_K_XL with an F16 projector, on llama.cpp; Parakeet TDT
0.6B v3 for speech in; a Qwen3-TTS 1.7B Obama fine-tune for speech out; a 3.3M sherpa-onnx
zipformer for the wake word. Chat plus projector is 4.47 GB, the live total 7.84 GB on an
8 GB card. No cloud bill; power and money unmeasured. (pins, AGENTS.md)

**Why GGUF and llama.cpp?**
Pascal. CUDA 13 dropped it and cu128 dropped sm_61, so cu126 is the last wheel line that
works. llama.cpp builds for arch 61 with fp16 math off — GP104 runs fp16 at 1/64 rate — no
bfloat16, no FlashAttention 2. (0004)

**What did the speech and voice measurements say?**
STT is CPU fp32 because Skylake has AVX2 but no AVX-512 or VNNI: int8 Parakeet is slower than fp32 past
two seconds. Kokoro is still the pinned production voice at 0.038-0.044 real-time factor, and
the 0.6B Qwen3-TTS candidate reached first audio in 0.25 s streaming. The Obama talker is
unmeasured. (0004, pins)

**How do the realtime websocket, the LLM and the tools fit together?**
speech-to-speech v1.0.0 is a separate pinned service on a localhost websocket, reaching
llama.cpp through its generic responses-api backend. The server emits tool arguments, this
process runs the tool and returns the result. An interrupt tags the generation stale and
flushes queued audio — finishing the sentence anyway is banned. (0001, 0002)

**How does the wake word work?**
Open-vocabulary keyword spotting, one thread, fp32, with per-keyword boosts. "Reachy" fires at
threshold 0.25 on synthesised speech; real-room tuning is issue #23. A proxy rewrites "Richie"
and "Ricci" before the LLM. (pins, commit 2e7e5bb)

**How fast is it?**
Low-40s tokens per second generation, 956 prefill at pp1024. The old sluggishness was an
800 ms silence timer, not a model. Speech-end to first-audio p50/p95 is a Phase 1 item and
unpublished, so he gives no latency number aloud. (pins, 0006)

**How does he ignore a room full of people talking?**
Ambient mode runs only cheap observers: keyword spotting, short transcription, direction of
arrival, faces. The cascade is fail-closed — keyword, then lease, then spatial and visual
evidence, then a lightweight model — and a sixty-second speech-duty check raises every
threshold when something talks non-stop. (0003)

**How does he turn toward the speaker?**
On a wake keyword the head turns to the array's direction of arrival and both fixed beams
steer there. The array is linear — four mics on one axis — so front and back are ambiguous and
direction alone cannot veto a TV. (0005, 0003)

**Does he recognise faces?**
InsightFace buffalo_sc on CPU, legacy stack only. In the new shell, learn-face and
identify-face answer "not available" until item B1, and face-based greeting is item B0 with an
unmeasured CPU cost. Say "being rewired", never "done". (app/face_tracker.py, env B0/B1)

**What happens to the recordings afterwards?**
There are none: no PCM is journaled and audio is not retained outside a toggled experiment. The
journal is typed events, capped by rows and age. Event memories and faces start empty and are
purged after the day. (0007, env rule 1)

**What is on the dashboard?**
A local FastAPI page on port 3001 over that journal: engaged or ambient and why, the lease,
transcripts, the tool in flight, cancellations, timings, pinned versions, CPU and GPU. It
replaced grepping eleven million log lines. (AGENTS.md, 0007)

**What is he bad at?**
He narrates an action instead of performing it, about one direct intent in four, and answers
the TV sometimes. He over-calls, and the fixed-focus camera is blurry in low light. A scene
describe measures 2.5 s at 448 px, and Parakeet mangles short words — "stop" came back "The
Stob". (env parked 2/3/6, setup-gtx1070-arch.md, pins, 0006)

**How do I build or fork it?**
`uv sync`, then `uv run pytest tests/` — hardware-free, no robot. The speech service installs
in its own venv via scripts/install_s2s.sh, never vendored, and the pinned cu126 torch pair
must not be bumped. (AGENTS.md, 0001, 0004)

## 3. Out of scope — never claimed at the demo

- No cloud fallback, no "it improves if we connect it". (epic, locality)
- No accuracy figures nobody measured: no WER, no success rate, no benchmark percentage.
- No recording offer. Do not say he can save the conversation; audio is not kept. (0007)
- No attendee face memory. A name needs an introduction in this room, and the event face store
  is purged afterwards. (env rule 1)
- No multi-person or overlapping-speech conversation. Not in this version. (epic)
- No voice identity as authorization — it is evidence, never a permission check. (0003)
- No "he saw who came in behind him": linear array, front/back ambiguity. (0003)
- No object counts, no confident scene description. Describe, do not enumerate.
- No physical stunt outside the tool list he actually has. (README.md)
- No memory across a restart: the conversation summary does not survive one. (env parked 1)
- No Alan Rickman persona on the floor unless B0b merges. It is a script, not a switch.
- Nothing about how long the mic has been open. One owner, one mute. (0002)

## 4. Open questions for David (do not answer these on the floor)

1. False wakes with the TV on: the brief says about fifteen a day. Written down are threshold
   observations at 0.46-0.48 on TV dialogue (app/wake_detector.py) and "real-room tuning
   pending" (issue #23). Is fifteen counted? The FAQ says only "answers the TV sometimes".
2. `describe_scene` wall clock: pins measures 2.5 s per 448 px on the CPU vision model, the
   brief says 6-7 s live. Which path produced 6-7 s — chat model plus mmproj, not the 0.8B
   model? Either way it needs a pins line.
3. Miscounting objects: no measurement in the repo. Case log, or observation? The no-counts
   rule in section 3 stands either way.
4. The Obama 1.7B talker has no pins entry, which is this repo's own bar for anything that
   runs on the box. Quote the 0.6B candidate, or say the voice's cost is unmeasured?
5. Cold start is item C2 and unmeasured. Do not answer "how long does it take to boot".
6. Issue #35 (Alan Rickman) is not in this checkout, and the
   `docs/notes/alan-rickman-voice-issue.md` that env B0b points at does not exist. The patter
   file uses the register exactly as written in the task brief.
7. Low light: sourced to setup-gtx1070-arch.md as a fixed-focus module that is blurry. Nobody
   has measured a lux floor.
