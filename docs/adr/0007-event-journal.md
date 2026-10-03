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
smart-turn score), `response.*` (created/first_text/first_audio/cancelled/completed),
`tool.*`, `tts.*`, `context.compacted` (before/after tokens + latency),
`memory.updated`, `model.loaded/unloaded`, `system.resource_sample`.

Latency acceptance metrics (EPIC "Benchmark plan") are computed as deltas over these
events — if a metric cannot be computed from the journal, the journal is wrong.

## Consequences

Every subsystem gets the journal injected at construction; emitting is one call,
never raises, drop-on-backpressure. The 274-responds-0-ignores class of bug becomes
a SQL query.
