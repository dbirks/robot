# Agent Instructions

This file provides context for AI coding agents (Claude Code, Copilot, Cursor, etc.) working on this repo.

## What this project is

A fully local, realtime, voice-driven social agent for the [Reachy Mini](https://www.reachy-mini.org/) desk robot. No cloud inference — everything runs on a single machine with an NVIDIA GPU. The canonical spec for where this is heading is [`docs/epic-realtime-rebuild.md`](docs/epic-realtime-rebuild.md).

## Two stacks coexist right now

**New stack (`shell/`, the future — Phases 1+3 of the EPIC are on main):**

| Layer | Component | Runs on |
|-------|-----------|---------|
| VAD / turn detection / STT / TTS / cancellation | pinned `huggingface/speech-to-speech` realtime service (ADR 0001) | CPU + GPU (Parakeet + Kokoro) |
| LLM | Qwen 3.5 4B GGUF via llama.cpp, reached by the service over responses-api | GPU |
| Audio device ownership + attention + tools + journal | `python -m shell` | CPU |
| Robot | Reachy Mini SDK 1.8.x via `reachy-mini-daemon` | USB/network |
| Observability | FastAPI dashboard (port 3001) + SQLite event journal (ADR 0007) | CPU |

**Legacy loop (`app/`, being deleted at the Phase 2 cutover, issue #22):** synchronous turn-based cascade (`python -m app`): silero-vad → Parakeet STT → chat-completions LLM → Kokoro/Piper TTS. Do not add features there; port what survives into `shell/`.

## Key architecture rules (read the ADRs)

- One persistent mic owner and one speaker owner; everything else subscribes (ADR 0002). Never open an audio stream outside `shell/audio/`.
- Ambient room speech must never enter LLM history; state-changing tools require attention confidence (ADR 0003, `shell/attention/`).
- Cancellation = flush queued audio + refuse the cancelled generation; "finish the sentence anyway" is banned.
- `llama.cpp` must run with `--jinja` or tool calling silently fails.
- Robot tools never raise; handlers return JSON-serializable dicts (`shell/tools.py` wraps them, dedupes repeats, gates state changes).
- Every service/model needs a revision + measured cost in `docs/pins.yaml` before it runs on the robot box (ADRs 0004-0006 record the measured Pascal/XVF3800/CPU traps — read them before tuning or bumping anything).

## Workflow

- **Commit often.** Commit and push at natural breakpoints.
- **uv only.** `uv sync` to install, but note: sync pulls the cu126 torch pair — never bump those (ADR 0004).
- Tests are hardware-free: `uv run pytest tests/`.
- Lint/format/type gates live in `.github/workflows/lint.yml`; the `ty` check is scoped to `shell/ tests/` until the legacy loop is deleted at cutover.

## Adding a new robot tool (new stack)

1. Schema in `TOOLS` + handler via the legacy bridge for now (`app/robot_tools.py`); Phase 2 moves them natively into `shell/`
2. Handler must return a JSON-serializable dict and never raise
3. If it changes state, add it to `STATE_CHANGING` in `shell/tools.py` — it will require an attention lease

## Running without a robot

Both stacks start with no robot attached; tools return `{"ok": False, "error": "Robot not connected"}` and the audio owners fall back to system default devices.

## Target hardware

i7-6700K (4C/8T Skylake), GTX 1070 8 GB (sm_61 Pascal — no bf16, no FA2, no tensor cores, cu126 is the last torch line), Arch Linux, Reachy Mini with XMOS XVF3800 linear 4-mic array. The machine has hard, measured constraints — see ADRs 0004 (builds), 0005 (mic/watchdog), 0006 (CPU/latency budgets) before adding any model or thread.

## Issue tracking

GitHub Issues is the issue tracker. The canonical work spec is [`docs/epic-realtime-rebuild.md`](docs/epic-realtime-rebuild.md) (EPIC issue #20, phases #21-#28; #24-#28 are gated on maintainer sign-off). Durable decisions live in [`docs/adr/`](docs/adr/), experiment results in [`experiments/`](experiments/).
