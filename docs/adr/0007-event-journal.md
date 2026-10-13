# ADR 0007: Typed local event journal over a growing pile of logs

- Status: accepted
- Date: 2026-10-02

## Context

`agent.log` grew to 794 MB / 11M lines with no rotation, and the interesting signal
(wake decisions, TTS timings, cancellations) was buried under repeated watchdog
lines. Answering "why did it respond to that" or "where does the 1.4 s go" required
grepping prose. The dashboard's Inspect page wants live state; debugging wants
deltas between typed timestamps.

## Decision

One SQLite (WAL) event journal, typed events with monotonic-clock timestamps,
retention-bounded (row cap + age). Prose logs exist but are rotated and carry only
human noise. The dashboard Inspect page is a view over the journal + a small current
-state table, not over log files. No PCM frames are journaled; audio is not retained
by default anywhere outside an explicitly-toggled Model Lab experiment.

Event families (superset grows, never renames — add a type instead):
`audio.*`, `kws.detected`, `attention.*` (decision/lease_*/profile_changed with the
evidence that produced it), `participant.*`, `turn.*` (transcript partial/final,
smart-turn score), `response.*` (created/first_text/first_audio/cancelled/completed/transcript_final),
`tool.*`, `tts.*`, `context.compacted` (before/after tokens + latency),
`memory.updated`, `model.loaded/unloaded`, `system.resource_sample`,
`sleep.enter`/`sleep.exit` (issue #36: sleep is one state that closes the microphone with
the head, so each row carries the `reason` that moved it — `tool`, `auto_sleep`,
`kws:<word>`; the absence of a `sleep.exit` is what "he answered the room while lying
 down" looks like).

`response.transcript_final` (issue #38) is the one row that carries **what he actually said**: `{text, response_id}`
from upstream `response.output_audio_transcript.done`, written by `shell/realtime/client.py` before the
`on_assistant_text` listener runs. Every other `response.*` row is ids and timings only, so reading the live
journal gave a conversation with his side missing entirely (the 3552 turns in `data/sessions.db` are all dead
legacy-agent data, last written 2026-07-28). It is written even when `text` is empty: an empty
`transcript_final` beside a `response.cancelled` is the record of a barge-in cutting a sentence off, which is
what ADR 0003's cancellation acceptance wants queryable. Bubble-rendering readers skip blank text.

`env` is a **reserved payload key**: `shell/env.py` resolves a deployment environment
and `Journal(..., env=cfg.env)` stamps it on every row (docs/environments.md A4), so one
DB holds both rooms and "what happened at the event" is a `json_extract` filter. Writers
must not use `env` for anything else - the stamp is applied first and an explicit payload
key of the same name wins, so a payload can override it but never add it implicitly.

Latency acceptance metrics (EPIC "Benchmark plan") are computed as deltas over these
events — if a metric cannot be computed from the journal, the journal is wrong.

## Consequences

Every subsystem gets the journal injected at construction; emitting is one call,
never raises, drop-on-backpressure. The 274-responds-0-ignores class of bug becomes
a SQL query.
