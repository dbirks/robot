# experiments/

Messy-but-valuable learning, preserved on purpose. One directory per experiment:

    experiments/YYYY-MM-DD-<topic>/README.md

Each records: hypothesis, exact upstream/model revisions, machine state, build flags,
config, procedure, raw measurements, subjective ratings, **failures**, conclusion.
Failed experiments stay. Large audio artifacts stay gitignored; record where they went.

Rules of the house:

- A conclusion that changes durable architecture gets an ADR that links here.
- Numbers without a date, revision, and command line are anecdotes, not results.
- This machine has two performance cliffs worth repeating to every experiment:
  int8-on-CPU regression and torch/cu12x wheel selection (see ADR 0004).

## Index

| Date | Experiment | Verdict |
|---|---|---|
| 2026-07-28 | [realtime-spike](2026-07-28-realtime-spike/README.md) — OpenAI-Realtime WS on this hardware | transport PROVEN; barge-in + load measurements pending |

Planned (Phase 4-8, gated on user sign-off): tts-bakeoff, wake-kws,
active-speaker, speaker-id, llm-bakeoff, stt-bakeoff, attention-decision,
context-compaction.
