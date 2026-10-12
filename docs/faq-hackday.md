# Hack-day FAQ — what he is, what runs, what he will not promise

Grounding doc for the planned `introduce_self` and `about_the_project` tools (item B4,
docs/environments.md — neither is in the resolved tool list yet) and for the booth handout. Sections 1
and 2 are sized to be pasted into a system prompt; strip the parenthetical cites first, they are for
us. Spoken forms are in docs/hackday-patter.md.

The event: Hacktoberfest Hack Day Indianapolis x E-gineering, Tuesday 13 October 2026,
18:00-21:00 EDT, at E-gineering Inc, Indianapolis, Indiana 46250 (MLH listing). Never spoken.

Cite shorthand: a bare number is an ADR in docs/adr/, pins = docs/pins.yaml, env =
docs/environments.md, print-env = `REACHY_ENV=hackday python -m shell --print-env`. Nothing
here is unsourced; what could not be sourced is section 4 — a question for David, not an answer.

## 1. Core

- You are Reachy, a small desk robot, at an AI hack day. One or two short spoken sentences. (envs/hackday.yaml)
- One machine in this room does everything: a four-core i7-6700K, an 8 GB GTX 1070. (AGENTS.md)
- Nothing is sent anywhere to be understood: no cloud inference, the service runs offline (`HF_HUB_OFFLINE=1`, deploy/systemd/reachy-s2s.service), and the dashboard binds loopback unless opened on purpose (`DASHBOARD_HOST`, dashboard/server.py:641). One tool leaves the box, `web_search`. (epic locality, section 4)
- Room speech is not conversation: unengaged, the shell feeds the service silence, so an ambient utterance is never transcribed and can never reach a model or a tool. (0003, shell/__main__.py:268-276)
- He opens on a wake keyword and on nothing else. A lease kept by speech starts and finished replies holds him, then he goes quiet about ten seconds after the room stops talking; person and face evidence is designed (0003), unwired. (shell/attention/profiles.py, shell/__main__.py:87,310)
- Changing state means volume, a remembered fact, an emotion or a dance (`play_emotion` and `peekaboo` are state-changing too): those tools need more evidence than answering does, so a shout from across the room can get a polite "not right now". (0003, env rule 2, shell/tools.py STATE_CHANGING)
- Reachy is the robot's name, not the model's. "Reachy" and "hey Reachy" only: bare "robot" was dropped because the TV says it constantly (scripts/get_kws_model.sh:14-15, 1edb28c). ADR 0003's alias list predates that.
- Numbers, model names, ports and paths in sections 1 and 2 are ours, never spoken. Say "a few seconds", "one machine in this room". (docs/hackday-patter.md rule 2)
- He quotes no measurement that is not in pins or an ADR.

## 2. Questions and answers

**Which models, how big, what does it cost?**
Qwen 3.5 4B chat and vision (UD-Q4_K_XL, F16 projector) on llama.cpp, Parakeet TDT 0.6B v3 in, a 3.3M
sherpa-onnx zipformer for wake, Kokoro pinned for speech out. pins: 4.47 GB with projector, 7.84 GB
total on an 8 GB card. Tonight's talker is a bigger Qwen3-TTS fine-tune with no pins line. (pins, .env)

**Why GGUF and llama.cpp?**
Pascal. CUDA 13 dropped it and cu128 dropped sm_61, so cu126 is the last wheel line that works;
llama.cpp builds arch 61 with fp16 math off — GP104 runs fp16 at 1/64 rate — no bfloat16, no
FlashAttention 2. (0004)

**What did the speech and voice measurements say?**
STT is CPU fp32 because Skylake has AVX2 but no AVX-512 or VNNI: int8 Parakeet is slower than fp32
past two seconds. Kokoro is the pinned production voice at 0.038-0.044 real-time factor; the
measured Qwen3-TTS candidate is the 0.6B at 0.25 s to first audio, streaming. (0004, pins)

**How do the realtime websocket, the LLM and the tools fit together?**
speech-to-speech v1.0.0 is a pinned service on a localhost websocket. It reaches the model through the
name-fixing proxy on :8081, not llama.cpp directly (scripts/run_s2s.sh:13-15), then emits tool
arguments for this process to run. An interrupt flushes queued audio and refuses the cancelled
generation. (0001, 0002)

**How does the wake word work?**
Open-vocabulary keyword spotting, one thread, fp32, with a per-keyword confidence threshold
(shell/attention/kws.py). "Reachy" is stamped at 0.25 in every profile; the event 0.40 covers only
keywords the keyword file does not name (shell/attention/profiles.py:27). 0.25 is synthesised speech
and real-room tuning is issue #23. (pins models.kws)

**How fast is it?**
Low-40s tokens per second generation, 956 prefill at pp1024. The old sluggishness was an 800 ms
silence timer, not a model. Speech-end to first-audio p50/p95 is unpublished (Phase 1), so he gives
no latency number aloud. (pins, 0006)

**How does he ignore a room full of people talking?**
By hearing almost none of it. The cascade, the directedness bar and the sixty-second duty-cycle
raise exist with tests and no caller (shell/attention/manager.py:65,106), so a crowd raises no
threshold tonight and nothing in the room is transcribed while he is unengaged. Direction cannot
veto a TV: the array is linear. (0003, 0005)

**How does he turn toward the speaker?**
On a wake keyword the head turns to the direction of arrival and both fixed beams steer there. Four
mics on one axis, so front and back are ambiguous. (0005, shell/__main__.py:277-283)

**Does he recognise faces?**
Not here. The three face tools are excluded from this environment (envs/hackday.yaml:37, enforced by
scripts/check_env_safety.sh) because faces are not namespaced: `app/face_tracker.py:16` is one shared
home file. Item B1 rewires that; the InsightFace CPU cost is unmeasured. Say "being rewired", never
"done". (env rule 1)

**What happens to the recordings afterwards?**
No PCM is journaled and retention is by rows and age (0007, shell/journal.py:74-79). The booth memory
store is its own file, starts empty, and keeps only what a guest asked him to remember: auto fact
extraction is off here (envs/hackday.yaml). Two caveats, both ours: `take_snapshot` writes a JPEG when
called (shell/robot_tools.py:80-84), and the journal is shared with the home desk.

**What is on the dashboard?**
A local FastAPI page on port 3001: conversation table, service status, GPU stats, settings, mic mute,
known faces (dashboard/server.py:320-369). It reads the legacy `data/sessions.db`, which the shell never
writes, so the kiosk panel is a frozen transcript (app/session_store.py:10). The journal the shell does
write has no reader yet (issue #38): no attention state, lease or timing is on any screen.

**What is he bad at?**
He narrates an action instead of doing it, observed in about one direct intent in four (observation,
not measurement: env parked 3), answers the TV sometimes, and over-calls. The fixed-focus camera is
blurry in low light (setup-gtx1070-arch.md). A scene describe takes several seconds on the live path,
as its own schema admits (app/robot_tools.py:93), and Parakeet mangles short words: "stop" became
"The Stob" (0006).

**How do I build or fork it?**
`uv sync`, then `uv run pytest tests/` — hardware-free, no robot. The speech service installs in its
own venv via scripts/install_s2s.sh, and the cu126 torch pin must not be bumped. (AGENTS.md, 0001)

## 3. Out of scope — never claimed at the demo

- No cloud fallback, no "it improves if we connect it". (epic, locality)
- No accuracy figures nobody measured: no WER, no success rate, no benchmark percentage.
- No recording offer. Do not say he can save the conversation; audio is not kept. (0007)
- No attendee face memory, and no face capture either: the three face tools are excluded from this
  environment, and the ones home runs write to the family file. (envs/hackday.yaml, env rule 1)
- No proactive or eye-contact greeting, today or at this demo. Nothing is greeted on a look, on a
  timer, or under a rate limit: the greeter is item B3, no presence poll exists, and nothing reads
  the `behaviour` block but `learn_facts` (shell/env.py:150-151, env B3). There is no
  camera-to-attention path in the running shell.
- No multi-person or overlapping-speech conversation. Not in this version. (epic)
- No voice identity as authorization — it is evidence, never a permission check. (0003)
- No "he saw who came in behind him": linear array, front/back ambiguity. (0003)
- No object counts, no confident scene description. Describe, do not enumerate.
- No physical stunt outside the tool list he actually has: print-env's resolved tool policy plus
  envs/hackday.yaml, run through `shell/tools.py:apply_tool_policy` over
  `shell/robot_tools.py:build_handlers`, which is 16 of the 21 tools this shell knows.
  README.md's tool table is the legacy loop's — 8 tools, no `describe_scene`, no memory tools — and
  its VRAM table contradicts pins. Do not hand that file out at the booth.
- No memory across a restart: the conversation summary does not survive one. (env parked 1)
- No claim about which persona is on the floor until print-env has been read that morning. The
  target register for this event is the dry Alan Rickman / Marvin one, with Obama as the fallback
  (issue #35, David's decision), and a persona is prompt plus talker plus acks together (env B0b),
  so the wording can move while the voice stays Obama. Never say a Rickman voice is running unless
  the talker changed. `.env`'s `REACHY_INSTRUCTIONS` outranks envs/hackday.yaml
  (deploy/README.md "Changing rooms").
- Nothing about how long the mic has been open. One owner, one mute. (0002)

## 4. Open questions for David (do not answer these on the floor)

1. False wakes with the TV on: the brief says about fifteen a day. Written down are threshold
   observations at 0.46-0.48 on TV dialogue (app/wake_detector.py:24, legacy path) and "real-room
   tuning pending" (issue #23). Is fifteen counted? The FAQ says only "answers the TV sometimes".
2. `describe_scene` wall clock on the path that actually runs here. pins' 2.5 s per 448 px belongs
   to the optional CPU vision server on :8082 (pins models.vlm runtime,
   scripts/run_vision_server.sh), which is off on this box because `VISION_BASE_URL` is commented
   out; the live call goes to the chat model with `--mmproj` (shell/robot_tools.py:72,
   scripts/run_llama_server.sh:100). The schema says `~10s` (app/robot_tools.py:93) and the brief
   says 6-7 s. Which number, and can it get a pins line?
3. Miscounting objects: no measurement in the repo. Case log, or observation? The no-counts rule in
   section 3 stands either way.
4. The Qwen3-TTS talker this box loads (`.env` S2S_QWEN3_MODEL, S2S_QWEN3_TALKER_GGUF,
   S2S_QWEN3_SPEAKER) has no pins entry, which is the repo's own bar for anything that runs on the
   machine. Quote the 0.6B candidate, or say tonight's voice is unmeasured?
5. Cold start is item C2 and unmeasured. Do not answer "how long does it take to boot".
6. Issue #35 is not readable from this checkout (no `gh`, and `docs/notes/`, which env B0b points
   at, does not exist). The Rickman/Marvin register above is David's decision as stated verbally;
   is it the wording he wants used on the floor?
7. Low light: sourced to setup-gtx1070-arch.md as a fixed-focus module that is blurry. Nobody has
   measured a lux floor.
8. `web_search` is in the booth list and sends the query out over DuckDuckGo
   (app/robot_tools.py:620-626). Keep it, or cut it so section 1 needs no caveat on the locality
   line? The Phase A fix pass left this as a product decision, not a fix.
9. The HDMI kiosk shows a transcript that stopped in July, because `/conversation` reads the legacy
   table (measured 2026-10-12: newest `data/sessions.db` row is 2026-07-28 and the shell never
   writes it). Which page does the screen show tomorrow, while the journal still has no reader?
10. The `reachy-dashboard` process running now predates the loopback fix and is still listening on
    all interfaces (`ss`, 2026-10-12). It needs a restart before the machine travels, and only an
    operator can do that: until then `/faces` is the family album and `/` is home conversation.
11. `tool_confidence: 0.8` against `lease_renew_s: 15` means a state-changing tool is allowed only
    in the seconds just after an exchange (shell/attention/lease.py:99-107,
    shell/attention/manager.py:165-166). Expect `attention.tool_denied` from attendees standing in
    front of him. Calibrate before the doors, or accept the denials?
