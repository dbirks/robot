# EPIC: Rebuild Reachy Mini as a fully local realtime social voice agent

## Goal

Rebuild this repository into a **fully local, low-latency, natural conversational stack for Reachy Mini** on the existing machine:

- Intel i7-6700K-class CPU (4C/8T)
- NVIDIA GTX 1070 8 GB (Pascal, compute capability sm_61)
- Arch Linux
- Reachy Mini + XVF3800 microphone array
- **No cloud inference. No remote audio/image processing. No remote storage.**
- Network access is allowed for explicitly networked tools such as web search and for accessing the dashboard over the LAN.
- The robot must still boot and perform its core conversational/perception functions with WAN access blocked once all pinned model/artifact dependencies are installed.

The user-facing priority order is:

1. **Conversational voice naturalness**
2. **Perceived latency**
3. **Intelligence / reliable tool use**
4. **Multi-person awareness**

This issue is intentionally large. It is the canonical implementation spec for the rewrite and must be implementable **without consulting old Beads issues or prior chat history**.

---

## ⚠️ Non-negotiable rewrite rule: do not preserve legacy architecture/state

**There is no production state on the current robot that must be migrated or preserved.**

Do **not** add adapters, migration layers, compatibility shims, dual voice loops, legacy database readers, old configuration translators, fallback orchestration paths, or other complexity merely to preserve the current implementation.

Prefer deleting and replacing code when the new design is cleaner.

The final system should have **one** production voice architecture. The old recorder/STT/barge-in/TTS loop should be removed after the new path passes acceptance tests.

Likewise, existing SQLite schemas, face records, environment variable names, dashboard routes, cached model state, etc. may all be redesigned from scratch.

Where old code contains verified hardware knowledge or measured results, preserve that knowledge in an ADR or experiment record before deleting the code.

---

## Why this rewrite

The current repository grew from a conventional turn-based cascade:

```text
mic -> VAD -> STT -> chat-completions LLM -> TTS -> speaker
```

It now has useful pieces (robot tools, movement, face recognition, dashboard, memory primitives, DOA, sleep/wake, measured Pascal tuning), but the realtime audio lifecycle is fragmented.

Current/previous implementations have had multiple independent audio-device owners, per-turn streams, a separate barge-in listener, stale-output cancellation logic, and ambient-speech heuristics. This is exactly the class of architecture that causes clipped audio, device contention, false barge-ins, stale speech and difficult-to-debug timing races.

A spike already proved the target transport on this hardware: an OpenAI-Realtime-style WebSocket client with **one persistent input stream and one persistent output stream**, server VAD, streamed output audio, and cancellation on interruption.

The rewrite should make that architecture production-quality.

---

## Primary upstream architecture

Use [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) as a **separately installed, version/commit-pinned local service**.

Do not vendor its implementation into this repository and do not maintain a fork unless an experimentally proven blocker requires it.

The Reachy repository remains the **robot embodiment/orchestration layer**:

- audio device ownership
- attention and participant state
- Reachy movement/animations
- robot tool implementations
- face/voice identity
- dashboard / observability
- persistent local memory
- model-lab/experiment harness
- Realtime protocol client

The HF service owns the engaged realtime conversational speech pipeline:

- VAD
- Smart Turn / turn endpointing
- STT
- LLM request lifecycle
- TTS
- generation cancellation
- Realtime events

References:

- HF speech-to-speech: https://github.com/huggingface/speech-to-speech
- Realtime API implementation/docs: https://github.com/huggingface/speech-to-speech/tree/main/src/speech_to_speech/api/openai_realtime
- Smart Turn: use the current upstream-supported Smart Turn integration/version at implementation time, but **pin the exact revision/model afterward**.
- llama.cpp server: https://github.com/ggml-org/llama.cpp/tree/master/tools/server

### Transport decision

Use **WebSocket on localhost** as the canonical transport for this epic.

Do not add WebRTC unless a concrete requirement appears. Keep the client abstraction clean enough that WebRTC could be added later for browser/wireless/network media endpoints.

---

## Architectural invariants

### 1. One persistent microphone owner

Exactly one component opens the physical microphone stream.

No watchdog, barge-in listener, wake detector, recorder, experiment harness, speaker recognizer, or ASD component may independently open the same capture device.

The microphone owner publishes PCM frames to internal subscribers/ring buffers.

### 2. One persistent speaker owner

Exactly one component owns output playback.

Realtime TTS, prerendered acknowledgements, earcons, tool sounds and experiment playback all go through an explicit mixer/arbiter rather than racing `sounddevice.play()` calls.

### 3. Audio fan-out, not duplicated device access

Consumers may subscribe to the same PCM:

- keyword spotting
- local ambient VAD/STT
- HF speech-to-speech while engaged
- active-speaker detection
- speaker embedding/enrollment
- metrics/optional experiment recording

but the hardware stream remains singular.

### 4. Cancellation must stop stale speech

On interruption:

- flush queued output immediately
- cancel in-flight response generation
- prevent late/stale TTS chunks from becoming audible
- preserve the new interrupting utterance, including reasonable pre-roll

No “finish the current sentence anyway.”

---

## Top-level state model

Separate these concepts. Do not collapse them into one “awake/listening” boolean.

```text
Audio observations
    |
    +--> ParticipantResolver
    |      "who / where appears to be speaking?"
    |
    +--> AttentionManager
           "should Reachy engage/respond?"
                 |
                 v
          ConversationEngine
```

Also distinguish:

- **robot awake/asleep**
- **ambient vs engaged audio mode**
- **attention lease**
- **conversation session**
- **current participant**
- **known identity**
- **LLM context window**

These have different lifetimes.

---

## Two-mode attention architecture

### Ambient mode

Ambient room audio must **not** be continuously inserted into the HF conversation history.

This is crucial for home TV audio and a noisy hackathon/event environment.

Ambient mode uses local lightweight components:

```text
persistent mic
  |
  +--> sherpa-onnx open-vocabulary keyword spotting
  +--> local VAD / fast ambient STT when needed
  +--> DOA
  +--> face/participant observations
  +--> AttentionManager
```

If ignored, an ambient utterance disappears from conversational context.

If accepted, the already-transcribed utterance may be inserted into the Realtime conversation as a user item and a response requested, rather than transcribing it twice.

### Engaged mode

While an attention lease is active, stream live PCM to HF speech-to-speech and use its realtime VAD/STT/Smart Turn/LLM/TTS/cancellation lifecycle.

When the lease expires, return to ambient mode.

This avoids background chatter contaminating the LLM while retaining natural multi-turn conversation once someone has Reachy's attention.

---

## Wake / explicit attention fast path

Use **sherpa-onnx open-vocabulary keyword spotting** for explicit attention phrases.

References:

- https://k2-fsa.github.io/sherpa/onnx/kws/index.html
- https://github.com/k2-fsa/sherpa-onnx

Initial aliases:

- `Reachy`
- `hey Reachy`
- `robot`
- `hey robot`

The canonical product/character name in all UI, prompts, logs and docs is **Reachy**. Any phonetic/token spelling used internally by KWS is implementation detail.

Per-keyword boosts/thresholds are allowed and should be tunable.

### Reaction

On a wake phrase:

1. establish/reacquire attention immediately
2. capture current DOA
3. begin turning Reachy toward the likely speaker **immediately**
4. do **not** wait for the LLM
5. schedule a short prerendered acknowledgement after ~250–400 ms
6. if the person continues speaking, suppress the acknowledgement so Reachy does not talk over them
7. if they only say “Reachy?” and pause, play a short response such as `hm?`, `yeah?`, etc.

Create a tiny pool of prerendered acknowledgement sounds/utterances so it is not identical every time. These bypass TTS at runtime.

Do not grunt repeatedly when Reachy is already actively attending to the same participant.

The sub-100-ms-class physical reaction is more important than generating a clever spoken acknowledgement.

---

## Attention leases and profiles

An attention lease means “Reachy currently believes this participant is interacting with him.”

Lease should renew based on evidence, not a single hard timer.

Base behavior:

- explicit wake phrase: immediate lease
- same resolved participant continues talking: renew
- recent accepted dialogue + compatible DOA/face evidence: renew
- person remains visibly engaged/oriented toward Reachy: modest extension
- participant walks away/disappears: expire early
- another explicit participant takes over: transfer
- explicit wake phrase always reacquires

Make the base timeout **profile-driven**.

Ship understandable profiles rather than only a vague sensitivity slider:

- **Quiet room**
- **Home / TV**
- **Event / noisy room**

Each profile sets documented thresholds such as wake sensitivity, ambient directedness threshold, attention lease base timeout, competing-speech penalty, etc.

An Advanced UI can expose the actual values.

Profiles must be versioned/exportable so experiment results are reproducible.

---

## AttentionManager decision cascade

Do not use an LLM for every utterance.

Fastest / strongest signals win first:

1. explicit wake keyword -> engage
2. active lease + same participant -> normally engage
3. strong deterministic/spatial/visual evidence -> respond or ignore
4. only ambiguous ambient utterances -> optional lightweight decision model

The system should be **fail-closed enough that overheard instructions do not execute robot tools**.

A prior real failure mode was overhearing speech addressed to another assistant and performing state-changing actions. State-changing tools therefore require higher directedness/attention confidence than harmless conversational replies.

Log the evidence and decision.

---

## Decision-model experiment (not critical path)

Investigate a bounded local decision model only for ambiguous cases.

Current candidate: **Laya typed decisions** (small encoder/non-autoregressive decision model with ONNX path), rather than the currently ROCm-centric Decision 1.0 Kai runtime.

References:

- https://huggingface.co/convaiinnovations/laya-typed-decisions
- https://huggingface.co/convaiinnovations/laya
- vLLM Semantic Router decision-model concept: https://vllm-semantic-router.netlify.app/blog/decision-models/

Example compact input:

```yaml
recent_accepted_dialogue:
  - user: "Have you seen my glasses?"
  - reachy: "Not from here."

current_transcript: "Can you look on the desk?"

attention:
  lease_active: false
  seconds_since_interaction: 7.4

spatial:
  speech_doa_deg: 11
  last_participant_doa_deg: 14

vision:
  active_visible_face: true
  face_oriented_toward_robot: true

choice:
  - respond
  - ignore
  - ask_if_addressed
```

Requirements:

- benchmark CPU latency on the actual i7-6700K
- compare against simple rules + existing/small embedding classifier
- do not integrate if it adds complexity without measurable accuracy benefit
- fine-tuning a Reachy-specific decision model is **out of scope**
- the dashboard may collect corrected evaluation examples for possible future work

---

## ParticipantResolver

The ParticipantResolver combines independent observations and produces a scored hypothesis, not false certainty.

Potential evidence:

- DOA / audio direction
- persistent face track
- active-speaker probability
- face-recognition identity
- speaker-recognition identity
- recency/continuity
- explicit wake target

Example output:

```json
{
  "track_id": 3,
  "doa_deg": 14,
  "active_speaker_p": 0.93,
  "face_identity": {"name": "David", "confidence": 0.71},
  "voice_identity": {"name": "David", "confidence": 0.62},
  "resolved_identity": "David",
  "resolution_confidence": 0.84
}
```

Identity signals are evidence, not security credentials.

**Never use a remembered voice to authorize sensitive actions.**

---

## Multi-face tracking + audio-visual active speaker detection

After the core realtime voice loop works, immediately run a **time-boxed ASD spike**.

Test both:

- Light-ASD
- LR-ASD

References:

- Light-ASD / related implementations: https://github.com/Junhua-Liao/Light-ASD
- LR-ASD: https://github.com/Junhua-Liao/LR-ASD
- TalkNet pipeline is also useful as a reference for face-detect/track -> ASD structure: https://github.com/TaoRuijie/TalkNet-ASD

Do not run expensive InsightFace recognition at camera frame rate just because ASD wants temporal tracks.

Target pattern:

```text
camera frames
   |
periodic face detection/recognition (InsightFace)
   |
persistent lightweight multi-face track IDs between detections
   |
face crop sequences + synchronized audio
   |
LR-ASD / Light-ASD
   |
active speaker probability per visible track
```

Current face detection around ~3 FPS is not a requirement; redesign if appropriate, but benchmark before optimizing.

### ASD real-world acceptance scenarios

Test on the actual robot, not only benchmark datasets:

- two visible people; A speaks
- two visible people; B speaks
- silent person smiles/moves lips while the other speaks
- TV/background speech
- crowded/noisy room
- speaker partly occluded
- speaker turns sideways
- speaker walks
- Reachy is speaking while visible humans are silent
- human interrupts Reachy
- active speaker temporarily leaves frame

If ASD performs well without materially harming latency/CPU, integrate it into ParticipantResolver. If not, record the experiment honestly and defer it; ASD failure must not block the core voice rewrite.

---

## Speaker / voice memory experiment

After ASD, time-box local speaker recognition using **WeSpeaker**.

References:

- https://github.com/wenet-e2e/wespeaker
- https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md

Goal: store a speaker embedding (“voiceprint”) as supporting identity evidence.

Consent behavior:

- Reachy may proactively offer to remember an unknown person only after a genuine multi-turn interaction and at a natural lull.
- Do not prompt every stranger immediately.
- Do not repeatedly offer within the same interaction after a decline/no response.
- A person may explicitly say “remember me” at any time.
- Ask for their name if it is not already known.
- Explain plainly, if asked, that Reachy stores small numerical representations of face and voice locally so he can try to recognize them later.

After consent, gather enough clean speech from **normal subsequent conversation** while ASD/participant resolution is confident the consenting person is the one speaking.

Then:

- compute speaker embedding
- discard temporary enrollment PCM
- retain embedding locally
- do not retain raw enrollment audio by default

Stored person identity should allow independent removal of:

- face identity
- voice identity
- thumbnail
- associated remembered facts/profile

The UI may also provide one “forget this person entirely” action.

---

## Face memory

Retain the existing useful concept (InsightFace embeddings) but redesign storage/schema freely.

No migration of current face data is required.

Suggested person record:

```text
person
  id
  name
  created_at
  last_seen_at
  consent metadata

  face embeddings (0..n)
  optional local thumbnail

  speaker embeddings (0..n)

  profile / remembered facts
```

Face/voice recognition remains probabilistic evidence.

---

## Nemotron diarization: optional later observation source

Nemotron 3 Diarization is interesting but is not a gate for this epic.

Reference:
https://huggingface.co/docs/transformers/main/model_doc/nemotron3_diarization

It can provide anonymous speaker channels such as `speaker_0`, `speaker_1`, etc. These labels are session-local and are not human identity.

If resources permit after core/ASD/voice-memory work, run a small sidecar experiment consuming the same PCM stream and expose its output through a generic `SpeakerObservation` interface.

Do **not** implement full overlapping multi-speaker transcription in the first version. On this 8 GB / 4-core box it is lower priority than natural voice, latency and reliable tools.

---

## TTS: highest-priority model bakeoff

Natural voice quality is the top product priority.

First production candidate: **Qwen3-TTS 0.6B Base / voice-cloning path**, with a frozen canonical Reachy voice.

References:

- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- Qwen 0.6B Base: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
- qwentts.cpp: https://github.com/ServeurpersoCom/qwentts.cpp
- HF s2s Qwen handler: https://github.com/huggingface/speech-to-speech/tree/main/src/speech_to_speech/TTS

Keep Kokoro only as a **benchmark/control during the experiment**, not as a permanent legacy fallback unless it wins the measured bakeoff.

Pocket TTS may be tested as a lightweight CPU candidate if useful:
https://huggingface.co/kyutai/pocket-tts

### Pascal constraint

Do not trust arbitrary prebuilt CUDA wheels on this GTX 1070.

This machine is **sm_61 Pascal**.

Where a dependency's distributed CUDA artifact omits sm_61, build deterministically from pinned source with explicit Pascal architecture flags rather than downloading random third-party binaries.

Document exact:

- upstream commit/tag
- compiler/CUDA versions
- CMake/build flags
- model revision/hash
- quantization
- measured VRAM
- measured TTFA/RTF
- known limitations

---

## Canonical Reachy voice: design once, then clone

Production voice identity must **not** be re-invented stochastically on every sentence.

Build a Model Lab workflow for one-time voice creation:

1. stop unnecessary production GPU services if needed
2. run the larger Qwen VoiceDesign checkpoint locally
3. prompt for a warm, understated, charming British male voice
4. generate ~10–20 candidate voices
5. expose candidates in the web UI with playback
6. user stars/selects favorites
7. regenerate a fixed test corpus for finalists
8. blind A/B finalists
9. freeze the chosen voice
10. production runtime uses the smaller Base cloning model against that frozen canonical voice

Suggested initial design text (iterate experimentally):

> British man, warm but understated, friendly and a little charming, contemporary southern-British/RP-leaning accent, natural conversational rhythm, not an announcer, not theatrical, moderate pitch, calm confidence.

Freeze enough state to reproduce/audit the voice:

- exact VoiceDesign model + revision/hash
- prompt
- seed
- generation/sampling parameters
- chosen reference WAV
- exact reference transcript
- extracted speaker representation
- any codec/RVQ/reference state used
- qwentts.cpp/HF revision
- build flags

The seed is for reproducibility; it is **not** the definition of voice identity.

### Voice stability is a hard acceptance criterion

The same Reachy voice must remain recognizably consistent across:

- short acknowledgements
- questions
- excited text
- apologetic text
- numbers/names
- long-ish responses
- many consecutive turns

If 0.6B drifts unacceptably, test whether reducing the LLM footprint buys enough VRAM for Qwen3-TTS 1.7B runtime.

Do not accept “sometimes cheerful tenor, sometimes deep serious man” behavior.

---

## Streaming speech

Stream LLM output into TTS, but preserve prosody.

Do not TTS one token at a time.

Implement/test a text coalescer:

```text
LLM token stream
   |
natural short clause / sentence boundary
   |
minimum useful amount of text
   |
TTS request begins
   |
LLM continues generating next chunk in parallel
```

A/B:

- complete sentence boundaries
- clause boundaries
- minimum characters/words
- punctuation heuristics

Naturalness outranks shaving ~150 ms if the faster strategy sounds noticeably worse.

Prerendered wake/earcon/camera sounds bypass this chunker.

---

## STT bakeoff

Baseline/control: current Parakeet TDT path / HF-supported Parakeet configuration.

Candidate: Nemotron 3.5 ASR Streaming if it is practical on this CPU.

Reference:
https://huggingface.co/docs/transformers/main/model_doc/nemotron3_5_asr

Do not switch merely because a model is newer.

Measure:

- end-of-speech -> transcript final
- partial stability
- WER on a small fixed local corpus
- CPU usage on i7-6700K
- effect on TTS/face/ASD contention
- noisy-room behavior

Pick the model from actual measurements.

---

## LLM bakeoff

Known baseline: the current llama.cpp / Qwen3.5 4B configuration, which has already demonstrated reliable tool calling and low-40s tok/s generation on this GTX 1070.

Candidates worth testing:

1. Qwen3.5 4B baseline/current known-good quantization
2. Qwen3.5 2B, primarily to trade some reasoning capacity for TTS VRAM/latency headroom
3. Nemotron 3 Nano 4B if tool reliability and Pascal performance justify it

Do not add a model because of a leaderboard.

Use the exact Reachy tool schema and fixed local tests.

Score at least:

- correct tool selected
- correct arguments
- no tool when none is appropriate
- multi-step tool behavior
- concise spoken response
- no thought-token leakage
- time to first token
- tokens/sec
- prefill latency cached/cold
- peak VRAM
- coexistence with chosen TTS

The selected production model must fit the whole machine, not just an isolated benchmark.

---

## llama.cpp and context budget

Treat context management as a **latency/memory architecture**, not a simple `messages[-20:]`.

The current machine has already measured that the stable prompt/tool prefix is expensive to prefill cold and cheap when cacheable; trimming the conversation can cause multi-second re-prefill penalties.

Current llama.cpp supports prompt/KV caching, cache reuse and explicit context management:
https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

Do not blindly set enormous context lengths. The 8 GB GTX 1070 must simultaneously host the chosen LLM/TTS and required KV/cache state.

### Separate attention lease from conversation lifetime

Example:

- attention lease may end after ~tens of seconds
- conversation session may remain available for several minutes
- if the same person re-engages shortly, conversational continuity remains
- after a longer idle/session boundary, start a new live session while durable/person memory remains

Make these profile/configurable and observable.

### Required long-session context strategy

Use tiers:

```text
stable prefix
  system/character/instructions
  tool schemas

small durable/core state
  current participant identity
  important active facts / task state
  selected relevant memories

rolling conversation summary
  compact summary of older accepted dialogue
  important unresolved commitments/tool effects

recent verbatim tail
  most recent user/assistant/tool turns
```

Do not summarize ambient ignored speech because it never enters the conversation.

### Compaction

Compaction should be **infrequent and high-water-mark-driven**.

When token usage crosses a measured threshold (for example a percentage of per-slot context, not an arbitrary message count):

1. preserve stable prefix
2. preserve recent verbatim turns
3. summarize an older contiguous block into a compact structured/short natural-language summary
4. preserve unresolved tasks, names, commitments and important tool effects
5. do not leave orphaned tool results
6. replace the old segment in one larger operation
7. record a `context.compacted` event with before/after token counts and latency impact

Do not compact every turn.

Benchmark several watermarks on actual hardware because changing earlier prompt text can reduce prefix-cache reuse.

### Summary generation

Prefer doing compaction:

- after a response
- during a quiet/idle opportunity
- before context exhaustion

rather than blocking the first audio of a user-facing reply.

Use the selected local LLM; do not add a cloud summarizer.

Persist enough summary/session state locally to debug and inspect it.

---

## Tool calling with Realtime

Preserve the actual robot tool handler logic where it is good, but adapt transport to the Realtime function-call lifecycle.

Expected pattern:

```text
server emits function-call arguments
        |
Reachy client executes local robot tool
        |
client sends function_call_output item
        |
request follow-up response when appropriate
```

Differentiate:

- **fire-and-forget physical actions** where no extra LLM prose is needed
- **data-returning tools** that require an LLM follow-up (web search, scene description, identity result, etc.)

Do not generate unnecessary verbal filler if a fast physical acknowledgement/earcon is better.

State-changing tools must obey attention confidence rules.

Keep explicit web search allowed. It is an intentionally networked tool, not an inference dependency.

---

## Locality / privacy boundary

“100% local” means:

### Must stay local

- microphone audio
- raw camera frames/photos
- STT
- TTS
- LLM inference
- keyword spotting
- face recognition
- speaker recognition
- ASD
- embeddings
- conversation/session storage
- identity/person profiles
- summaries/memory
- dashboard data

No outside model/API receives audio, images, embeddings or conversation content.

### Allowed network behavior

- user explicitly invokes web search / another intentionally networked information tool
- LAN clients access the dashboard
- installation/update workflow downloads pinned dependencies/models before offline use

Core runtime must be capable of starting with WAN blocked.

Use local model paths and offline modes such as `HF_HUB_OFFLINE=1` after assets are installed.

Include an acceptance test that blocks external egress and restarts all core services.

---

## XVF3800 / hardware audio due diligence

Before tuning software thresholds, verify the microphone DSP path.

In particular, verify the XVF3800 AEC far-end reference actually receives Reachy's playback. If hardware AEC is working, do not stack software AEC blindly on top of it.

Evaluate the existing hardware-level observations in the repository/Beads history before deleting them:

- AEC far-end reference routing
- fixed-beam configuration/gating where appropriate
- speech-energy values as possible competing-source evidence
- AGC/noise suppression settings
- known USB autosuspend / firmware recovery behavior
- PipeWire mute state vs genuinely dead hardware

Fold verified results into an ADR/experiment record.

A watchdog must monitor the **resolved Reachy microphone device**, not blindly the system default, and must distinguish:

- explicitly muted
- silent room
- dead/stalled stream
- missing USB device

It must not open a second competing microphone stream.

---

## Dashboard / observability

Keep **FastAPI**.

Do not rewrite in React merely because this project is becoming interactive.

Preferred frontend:

- server-rendered pages
- SSE for live state/event updates
- HTMX where useful
- targeted local JavaScript for camera overlays, audio meters, timelines and Model Lab
- Tailwind is fine **if compiled at build time into local CSS**
- no runtime CDN dependencies

Remove current runtime dependencies on:

- Tailwind CDN
- unpkg HTMX/SSE
- jsDelivr fonts
- Google Fonts
- any other external dashboard assets

Vendor/build everything needed for runtime locally.

### Live / Inspect page

Create a high-inspectability live page showing at least:

#### Robot/attention
- awake/asleep
- ambient/engaged
- attention profile
- lease state + remaining/base timeout
- why current attention decision was made

#### Participant
- current face track
- known face identity + confidence
- active-speaker probability
- voice identity + confidence
- DOA
- final resolved identity/confidence

#### Audio
- VAD state
- keyword detections
- microphone health/RMS
- Smart Turn state/score where available
- output queue
- current barge-in/cancellation state

#### Turn
- partial/final transcript
- attention decision
- response state
- current tool call

#### Pipeline timing
- speech start/stop
- transcript final
- LLM response created/first token
- TTS first PCM
- first audible sample
- cancellation -> silence

#### Models/resources
- exact pinned versions/hashes
- active STT/LLM/TTS
- CPU/RAM
- GPU utilization/temperature/VRAM
- service health

#### Event stream
live chronological events for debugging.

### Annotated camera

If ASD integrates successfully, include a local live camera debug view with overlays such as:

- face bounding boxes
- persistent track ID
- known identity
- active-speaker probability
- currently attended track
- DOA indicator

Do not upload or remotely process the camera stream.

---

## Structured event journal

Conversation turns alone are not enough.

Create a typed local event journal, likely SQLite WAL-backed, for state transitions and performance observations.

Do not log every PCM frame.

Candidate events:

```text
audio.speech_started
audio.speech_stopped
audio.device_error

kws.detected

attention.decision
attention.lease_started
attention.lease_renewed
attention.lease_expired
attention.profile_changed

participant.face_track_updated
participant.active_speaker_updated
participant.identity_evidence
participant.resolved

turn.transcript_partial
turn.transcript_final
turn.smart_turn_score

response.started
response.first_text
response.first_audio
response.cancelled
response.completed

tool.started
tool.completed
tool.failed

tts.chunk_queued
tts.chunk_started

context.compacted
memory.updated

model.loaded
model.unloaded

system.resource_sample
```

Events need timestamps suitable for computing latency deltas.

The Inspect UI should mostly be a view over this state/event model.

Bound/rotate verbose logs and event retention so the robot does not accumulate multi-GB logs indefinitely.

---

## Model Lab / experiment UI

Create a first-class Model Lab in the dashboard.

It should support:

- selecting candidate model/config
- controlled service restart/reload
- displaying exact config/hash
- fixed test corpora
- blind A/B audio playback
- rating naturalness/responsiveness
- model resource/latency metrics
- saving experiment result metadata
- wake-word threshold testing
- canonical voice creation/selection

### Raw audio retention

Default:

- ordinary conversation: **do not retain raw audio**
- Model Lab experiment: metrics/transcripts only by default
- explicit per-experiment toggle may retain mic/output WAV for that experiment

Make retention state visible.

---

## Experiment records vs ADRs

Create both:

```text
docs/adr/
experiments/
```

### ADRs

ADRs are durable decisions, for example:

- use HF speech-to-speech as external pinned service
- WebSocket Realtime boundary
- single-owner audio I/O
- dual-mode ambient/engaged attention
- ParticipantResolver vs AttentionManager
- canonical Reachy voice
- local privacy boundary
- selected production STT/LLM/TTS after bakeoff

Each ADR should include:

- context
- decision
- alternatives considered
- consequences
- evidence/experiment links

### Experiments

Preserve the messy but valuable learning.

Suggested layout:

```text
experiments/
  README.md
  YYYY-MM-DD-qwen3-tts-pascal/
  YYYY-MM-DD-tts-bakeoff/
  YYYY-MM-DD-wake-kws/
  YYYY-MM-DD-active-speaker/
  YYYY-MM-DD-speaker-id/
  YYYY-MM-DD-attention-decision/
  YYYY-MM-DD-llm-bakeoff/
```

Each experiment records:

- hypothesis
- exact upstream/model revisions
- machine/hardware state
- build flags
- config
- procedure
- raw measurements
- subjective ratings when applicable
- failures
- conclusion
- artifacts location (large audio may remain gitignored)

Do not erase failed experiments.

---

## Benchmark plan

This rewrite is not complete because “a demo worked.”

### Automatic metrics

At minimum:

- speech-end -> transcript-final
- speech-end -> first audible response
- transcript-final -> LLM response-created
- response-created -> first text
- first TTS request -> first PCM
- keyword detection -> beginning of head motion
- interruption -> audible silence
- clipped first-phoneme incidents
- dropped mic frames / overruns
- Smart Turn reopen/correction count
- false wake count
- false directedness/respond count
- tool-call exactness
- tool-argument correctness
- peak VRAM/RAM/CPU
- service crashes/restarts

Report p50/p95 where meaningful, not only one cherry-picked run.

### Human A/B

Naturalness is the highest priority and cannot be selected only by automated scores.

Run live sessions per candidate/config and rate:

- voice naturalness
- identity consistency
- response pacing
- interruptibility
- perceived latency
- annoyance/repetition
- pronunciation errors
- conversational feel

The chosen voice/TTS may lose a small amount of raw TTFA if it sounds meaningfully better.

---

## Noise/attention acceptance corpus

Create repeatable test scenarios from real rooms, including:

- quiet direct conversation
- TV dialogue
- podcast/background speech
- two nearby people talking to each other
- crowded hackathon/event cacophony
- person explicitly says “Reachy”
- stranger says “hey robot”
- direct request without wake word (“Can you help me?”)
- speech to another nearby assistant/device
- second-person TV/dialogue (“can you…”, “what do you…”)
- speaker at different angles/distances
- Reachy already engaged vs ambient
- interruption while Reachy speaks

State-changing tools must never be triggered by clearly non-directed background speech in this corpus.

---

## Context/session acceptance

Test a long conversation substantially beyond the raw recent-tail budget.

Verify:

- conversation remains coherent after compaction
- recent details remain verbatim
- important old commitments survive summary
- stale trivia can fall out
- tool-result relationships are preserved
- no orphan tool messages
- prompt-cache behavior/latency is measured before/after compaction
- compaction does not happen every turn
- attention lease expiry does not erase a still-live conversation session
- conversation session eventually expires/resets on a longer idle boundary
- durable/person memory is separate from ephemeral conversation summary

Expose current approximate token budget and compaction events in Inspect.

---

## Robot personality / proactive behavior

Do not turn Reachy into product onboarding.

After a substantial interaction with an unknown person, at a natural lull while they remain present/engaged, Reachy may make **one contextually relevant offer**, e.g.:

> “By the way, if you'd like, I can remember you for next time.”

He should list broader capabilities only when asked or when context makes it natural.

Tone target: concise, warm, understated, slightly charming; British male voice. Avoid long speeches.

---

## Tool / scene / physical-action behavior

Keep and port useful existing robot capabilities:

- movement/look/nod/shake
- movement manager / smoothing
- emotion/animation behavior
- snapshots
- local scene description if the selected LLM/VLM path remains practical
- face learn/identify/forget
- volume
- sleep/wake
- web search
- reset conversation
- other existing useful tool handlers

The rewrite may redesign schemas/names.

Do not let an LLM fabricate physical actions that did not happen. Tool/state events should make actual execution inspectable.

---

## Model/service lifecycle

The final installation should be deterministic.

Pin:

- HF speech-to-speech revision/version
- model revisions/hashes
- sherpa-onnx model/runtime
- llama.cpp revision/build
- qwentts.cpp/runtime if used
- ASD model/runtime
- WeSpeaker runtime/model if used
- Reachy Mini SDK
- important compiler/CUDA/Python versions

Provide scripts/documentation to:

1. install/download all required assets
2. build Pascal-specific binaries/wheels
3. verify hashes
4. launch services
5. run smoke tests
6. run offline after installation

The production process layout should be simple and observable (systemd user services are fine).

---

## GitHub / issue-tracker cleanup

This repository currently contains a Beads issue database.

The final project should use **GitHub Issues as the issue tracker**.

As part of this epic:

1. inspect `.beads/issues.jsonl`
2. identify open items that are fully subsumed by this epic
3. do not recreate those as separate issues
4. for still-relevant **out-of-scope** open items, create standalone GitHub issues containing the necessary context
5. only after relevant knowledge is preserved, delete `.beads/` and Beads-specific workflow/config
6. future issue references in docs/comments should point to GitHub Issues

This epic specifically supersedes the earlier local realtime/speech-to-speech spike/epic and related voice/ambient-interruption tasks. Do not make implementation depend on looking those up.

---

## Suggested implementation sequence

Order matters. Do not start with experimental multi-speaker features.

### Phase 0 — baseline + evidence preservation

- capture current known-good hardware/resource measurements
- create `docs/adr/` and `experiments/`
- preserve verified historical findings from code/commits/Beads
- define typed event model
- define deterministic dependency/model pinning mechanism

### Phase 1 — single-owner audio + Realtime core

- build persistent mic/speaker owner
- PCM fan-out/ring buffer
- integrate pinned HF speech-to-speech over localhost WebSocket
- basic VAD/STT/LLM/TTS
- streamed output audio
- response cancellation
- barge-in
- port one trivial robot tool end-to-end
- instrument all key timings

Gate: stable natural conversation and repeated interruption on real hardware.

### Phase 2 — destructive cutover

- port remaining robot tools
- remove old AudioRecorder/per-turn input streams
- remove old InterruptiblePlayer / duplicate VAD/STT/TTS loop
- remove old sentence-by-sentence legacy orchestration where replaced
- remove unused TTS servers/config
- no permanent fallback voice loop

### Phase 3 — dual-mode attention

- sherpa KWS fast path
- ambient local VAD/STT
- attention leases/profiles
- ambient directedness rules/classifier
- explicit state-changing-tool confidence gate
- ensure ignored room speech never enters LLM history

### Phase 4 — canonical voice + TTS bakeoff

- deterministic Pascal builds
- Model Lab voice-design UI
- generate candidates
- human blind selection
- freeze canonical Reachy voice
- 0.6B Base clone runtime
- streaming coalescer experiments
- compare Kokoro/Pocket only as controls
- settle production TTS via measurements

### Phase 5 — LLM/STT bakeoffs + context architecture

- run fixed tool benchmark
- test Qwen 3.5 4B vs 2B vs credible alternative
- test STT baseline vs streaming candidate
- implement measured context/session compaction
- choose final per-slot context/VRAM budget
- preserve llama.cpp prompt-cache performance

### Phase 6 — ParticipantResolver + ASD

- persistent multi-face tracks
- periodic InsightFace recognition
- LR-ASD / Light-ASD experiment
- integrate if successful
- annotated camera inspect UI

### Phase 7 — speaker recognition / person memory

- WeSpeaker experiment
- consent flow
- accumulate post-consent normal speech
- persist local voice embedding, discard raw PCM
- face + voice evidence fusion
- dashboard deletion controls

### Phase 8 — optional decision/diarization experiments

- Laya bounded decision experiment for ambiguous attention cases
- optional Nemotron diarization observation
- integrate only if actual benefit exceeds complexity

### Phase 9 — hardening

- noisy-room/event tests
- 50+ repeated barge-ins
- long-session compaction tests
- hardware AEC verification
- mic recovery/watchdog
- log/event retention
- WAN-blocked restart test
- package/build docs
- delete Beads after migrating out-of-scope items
- final ADRs documenting selected stack

---

## Required acceptance criteria

The epic may close only when all critical items below pass on the **actual i7-6700K + GTX 1070 + Reachy Mini**.

### Core realtime

- [ ] one persistent physical mic owner
- [ ] one persistent physical speaker owner
- [ ] HF speech-to-speech pinned and separately installable
- [ ] localhost WebSocket Realtime path
- [ ] streamed assistant speech
- [ ] 50 consecutive short interruptions land correctly without stale speech
- [ ] interruption cancels generation/playback promptly
- [ ] no device-contention/clipped-start regression

### Local/privacy

- [ ] no remote audio/image/model inference
- [ ] no remote storage
- [ ] explicit web-search/network tools still work when invoked
- [ ] WAN-blocked cold restart succeeds after installation
- [ ] dashboard has no runtime CDN/font/script dependency

### Attention

- [ ] “Reachy”, “hey Reachy”, “robot”, “hey robot” fast path tested with multiple speakers
- [ ] immediate head-turn behavior
- [ ] delayed acknowledgement suppressed when user continues speaking
- [ ] attention lease/profile model implemented and inspectable
- [ ] TV/crowd/non-directed speech is normally ignored
- [ ] ignored ambient transcript never contaminates conversation history
- [ ] clearly non-directed speech cannot execute state-changing tools

### Voice

- [ ] canonical British male Reachy voice selected through local Model Lab
- [ ] voice artifact is reproducible/versioned
- [ ] production voice does not exhibit unacceptable timbre/pitch/personality drift
- [ ] streaming/coalescing chosen by measured A/B
- [ ] voice naturalness rated against baseline

### Context

- [ ] attention lease and conversation lifetime are separate
- [ ] long conversation survives beyond recent verbatim tail through compaction
- [ ] compaction is high-water/infrequent, not every turn
- [ ] llama.cpp cache impact is measured
- [ ] current context/token state visible in Inspect

### Observability

- [ ] typed event journal
- [ ] pipeline latency timestamps
- [ ] Inspect page shows robot/attention/participant/audio/turn/model/resource state
- [ ] optional annotated camera overlays if ASD succeeds
- [ ] logs/events are bounded/rotated

### Experiments / documentation

- [ ] ADR folder exists and records durable architecture/model choices
- [ ] experiment folder records bakeoffs including failures
- [ ] exact pinned source/model/build information captured
- [ ] no undocumented random third-party Pascal binaries

### Identity

- [ ] face memory remains fully local
- [ ] proactive consent flow feels natural and is not spammy
- [ ] if speaker recognition experiment succeeds, voice embedding can be enrolled from normal post-consent conversation
- [ ] raw enrollment audio is discarded by default
- [ ] face and voice identity data can be independently deleted
- [ ] voice identity is never treated as authorization

### Cleanup

- [ ] legacy voice loop removed
- [ ] obsolete audio owners/players removed
- [ ] no migration/back-compat scaffolding added for disposable existing robot state
- [ ] Beads knowledge migrated/superseded and `.beads/` removed
- [ ] GitHub Issues is the future issue tracker

---

## Sources / upstream references

The coding agent should re-check current upstream HEAD/releases before implementing, then pin the revisions actually tested.

- HF speech-to-speech  
  https://github.com/huggingface/speech-to-speech

- HF OpenAI Realtime implementation  
  https://github.com/huggingface/speech-to-speech/tree/main/src/speech_to_speech/api/openai_realtime

- llama.cpp server / prompt cache / context controls  
  https://github.com/ggml-org/llama.cpp/tree/master/tools/server

- Sherpa ONNX open-vocabulary keyword spotting  
  https://k2-fsa.github.io/sherpa/onnx/kws/index.html  
  https://github.com/k2-fsa/sherpa-onnx

- Qwen3-TTS  
  https://github.com/QwenLM/Qwen3-TTS  
  https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base

- qwentts.cpp  
  https://github.com/ServeurpersoCom/qwentts.cpp

- LR-ASD / Light-ASD  
  https://github.com/Junhua-Liao/LR-ASD  
  https://github.com/Junhua-Liao/Light-ASD

- WeSpeaker  
  https://github.com/wenet-e2e/wespeaker

- Nemotron 3 Diarization  
  https://huggingface.co/docs/transformers/main/model_doc/nemotron3_diarization

- Nemotron 3.5 ASR Streaming  
  https://huggingface.co/docs/transformers/main/model_doc/nemotron3_5_asr

- Laya typed decisions  
  https://huggingface.co/convaiinnovations/laya-typed-decisions

- vLLM Semantic Router decision-model architecture  
  https://vllm-semantic-router.netlify.app/blog/decision-models/

---

## Final implementation principle

Optimize for how Reachy **feels to talk to**, not for maximizing model novelty.

The right system on this hardware is expected to be a modular local cascade with excellent realtime orchestration:

- rapid physical acknowledgement
- strong turn-taking/barge-in
- natural, consistent voice
- clean attention semantics
- reliable tools
- transparent participant evidence
- aggressive observability
- measured model selection

Do not replace that with a monolithic end-to-end speech model merely because one exists. Revisit native speech models only when the hardware/runtime makes them demonstrably better for this robot.
