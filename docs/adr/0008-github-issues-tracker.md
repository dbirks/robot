# ADR 0008: GitHub Issues replaces beads; the EPIC doc is the spec of record

- Status: accepted
- Date: 2026-10-02

## Context

Beads accumulated ~40 issues, most of which the realtime EPIC explicitly supersedes.
bd requires a local Dolt/JSONL db that drifted (mixed prefixes, no-db mode broken on
this clone), and issue knowledge invisible from GitHub makes handoff between agents
and between machines fragile. The EPIC (docs/epic-realtime-rebuild.md) §"GitHub /
issue-tracker cleanup" directs the cutover.

## Decision

GitHub Issues is the issue tracker. Before deleting `.beads/`, all durable knowledge
was harvested into docs/adr/ and experiments/ (this ADR's sibling files). Open items
not subsumed by the EPIC were recreated as standalone GitHub issues with full
context inline. Docs and code comments reference GitHub issue numbers from now on.

## Consequences

One glanceable public tracker, agents need no local db to plan, and the EPIC doc +
ADRs are self-sufficient: a fresh agent can implement the rewrite without beads or
chat history — which was an explicit EPIC requirement.
