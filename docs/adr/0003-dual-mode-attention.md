# ADR 0003: Dual-mode attention — ambient vs engaged, fail-closed, leases over timers

- Status: accepted
- Date: 2026-10-02

## Context

Measured failure of the old ambient filter: across 11M log lines, `classify_utterance`
responded 274 times and ignored **0**. Three bypass rules (<=4 words, second-person
address, "short-ish default respond") short-circuited ~94% of traffic before any
ignore branch — TV dialogue saturates "you/can you" and clipped short lines. The
design comment said "better to respond to background speech occasionally than miss
real requests"; that intent *is* the bug. Real consequence: an overheard request
addressed to a coding assistant ("can you mute the robot?") executed `set_volume`, a
state-changing tool, and left the PipeWire sink at 50% across restarts.

Also structural: the XVF3800 array is **linear** (4 mics on one axis), so DOA has
irreducible front/back ambiguity — `is_from_front()` is really a broadside gate and
DOA alone cannot veto a TV behind the robot.

## Decision

Two audio modes with different lifecycles, plus evidence-based leases:

- **Ambient mode**: room audio does NOT enter the LLM conversation, ever. Only local
  lightweight observers run (KWS, short ambient STT, DOA, face observations). An
  ignored utterance disappears; an accepted one is inserted into the realtime
  conversation as an already-transcribed user item (no double transcription).
- **Engaged mode**: while an attention lease is held, PCM streams to the s2s service
  and it owns VAD/turn/STT/LLM/TTS/cancellation.
- **Attention lease** renews on evidence (wake phrase, same resolved participant
  continuing, recent accepted dialogue + compatible DOA/face), expires when the
  person leaves, transfers between participants, and is always reacquirable by wake
  phrase. Base timeouts are profile-driven (quiet-room / home-TV / event), versioned
  and exportable.
- The decision cascade is fail-closed: wake keyword > active lease + same
  participant > spatial/visual evidence > (only then) a lightweight decision model.
- **State-changing tools require higher directedness confidence than conversation.**
  An ignored ambient transcript never reaches the LLM, so it can never trigger a
  tool at all.

Rejected: continuous ambient STT into history (contaminates the LLM with TV),
single hard idle timer for sleep (too eager / too late), anti-spoof/replay
detection for TV (AASIST/RawNet2 degrade badly against replay per arXiv 2502.20427;
no open broadcast-vs-live classifier exists).

## Consequences

- KWS replaces embedding-similarity wake detection as the fast path
  (sherpa-onnx open-vocabulary; aliases `Reachy`, `hey Reachy`, `robot`,
  `hey robot`, per-keyword boosts).
- Cheap additive heuristics are allowed but may only *raise* thresholds — a rolling
  60 s speech-duty-cycle check (TV talks continuously for minutes, people don't;
  above ~70% raise everything) catches what per-utterance classification cannot.
- Speaker verification (WeSpeaker-class embeddings) is *evidence*, never
  authorization, and never gates sensitive actions on its own.
