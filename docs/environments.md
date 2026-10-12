# Deployment environments (`home` / `hackday`)

Status: proposal being implemented on `main` (started 2026-10-12).
Motivation: Reachy has been living on the home desk and is moving to the office for an
AI hack day, where he has to greet strangers in a noisy room. Rather than a branch, we
put the difference into a **deployment environment** layer, keep iterating on `main`,
and carry the good parts back home afterwards.

## Decision

- **One codebase, one branch.** Environments are data, not branches. `REACHY_ENV` selects one.
- An **environment** is the bundle that changes between rooms:

  | Layer | Home | Hack day |
  |---|---|---|
  | Attention profile (`shell/attention/profiles.py`) | `home_tv` | `event` |
  | Persona / instructions (`REACHY_INSTRUCTIONS`) | family assistant | demo host, no private facts |
  | Tool set | full (incl. `go_to_sleep`, `set_volume`, `remember`) | greeting/demo subset, state changes gated harder |
  | Memory namespace | `data/memories.home.json`, `data/faces.home.json` | separate files — family names, kids, home address **never** loaded |
  | Devices / tuning | current mic/speaker + XVF AGC cap | possibly different devices, higher volume floor |
  | Behaviour flags | today's | proactive greeter on, idle "someone is here" nudge |

- Env files live in `envs/<name>.yaml`, loaded by a new `shell/env.py`.
  **Precedence:** code defaults → env file → real process environment variables, so an
  on-site tweak is a one-off `REACHY_X=... systemd-run`, never an edit under time pressure.
- `home` is the default and must reproduce today's behaviour byte-for-byte (it just moves
  the current `.env` values into `envs/home.yaml`).
- The env name is stamped on every journal event (ADR 0007) so the dashboard can filter
  "what happened at the event" from "what happened at home".
- Anything new that proves useful at the event gets a `[carry-back]` note here and is
  switched on for `home` explicitly, not by accident.

## Env file schema (v1)

```yaml
name: hackday
attention_profile: event          # existing versioned profiles, unchanged
persona:
  instructions: |                 # replaces DEFAULT_INSTRUCTIONS, TOOLS_NOTE/TIME_NOTE still append
    You are Reachy, ...
  ack_dir: sounds/acks/obama
tools:
  include: []                     # empty = everything the shell knows about
  exclude: [go_to_sleep, reset_conversation]
memory:
  namespace: hackday              # memories.<ns>.json + faces.<ns>.json
behaviour:
  greeter: true                   # proactive hello when a face appears
  greeter_cooldown_s: 120         # per person
  greeter_quiet_minutes: 5        # global rate limit: never nag the room
audio:
  mic: null
  speaker: null
  xmos_agc_max_gain: 6            # room noise, not the living room
```

Unknown keys are ignored (forward-compatible), a missing file is a hard startup error,
and `python -m shell --print-env` dumps the resolved config for the dashboard/ADR trail.

## Hard rules

1. **No private home facts in a public room.** Memory + faces are namespaced per env; the
   hack-day store starts empty and is purged after the event unless we decide to keep
   attendees who opted in. Facts auto-extracted during the event stay in `hackday`.
2. **ADR 0003 still applies**: a greeting is speech, a state change is not. In `hackday`,
   `STATE_CHANGING` gets the strictest lease bar (`tool_confidence: 0.8` already in the
   `event` profile) and `go_to_sleep` / `reset_conversation` are excluded from the tool list.
3. **Every new model/runtime on the robot box gets a `docs/pins.yaml` entry with a
   measured cost** (ADRs 0004-0006) — this includes the InsightFace CPU budget for
   face-based greeting.
4. Hardware-free tests only in `tests/`; the greeter is driven by an injected frame, never
   by a real camera.

## Work list

### Phase A — environment layer (no behaviour change)

- [ ] A1 `shell/env.py`: load `envs/<name>.yaml`, layer into `ShellConfig` (defaults < file < env vars)
- [ ] A2 `envs/home.yaml` = today's `.env` values, byte-equivalent behaviour; `.env` keeps working as an override
- [ ] A3 `envs/hackday.yaml` with the `event` attention profile + attendee-safe persona
- [ ] A4 Journal + dashboard carry `env` on every event; `--print-env` flag
- [ ] A5 `MemoryStore` namespaces (`memories.home.json` / `memories.hackday.json`); migrate current `data/memories.json` → home
- [ ] A6 systemd: `Environment=REACHY_ENV=home` in `deploy/systemd/reachy-shell.service`, event swap documented in `deploy/README.md`
- [ ] A7 Tests: precedence, unknown-key tolerance, home parity (resolved instructions match today's)

### Phase B — hack-day capability

- [ ] B1 Wire face recognition natively into the shell (`learn_face` / `identify_face`
      return "not available" today because `make_handlers` gets no `face_tracker`);
      per-env faces file; measured InsightFace CPU cost in `pins.yaml`
- [ ] B2 `greet` tool (name-or-generic hello + look at the person + nod) — the model's hands
- [ ] B3 **Proactive greeter**: low-rate presence poll while idle → un-greeted face in frame →
      `client.insert_user_transcript()` to prompt a spoken hello (the hook already exists).
      Cooldowns per person + globally, never while he is mid-conversation, off in `home` at night
- [ ] B4 `introduce_self` + `about_the_project` Q&A grounded in a short `docs/faq-hackday.md`
      (what he is, what runs locally, what the stack is) — the thing people actually ask a demo
- [ ] B5 `icebreaker` / conversation-starter deck for silences in the booth
- [ ] B6 Attendee persona written up + run through `experiments/2026-10-04-llm-bakeoff/toolbench.py`
      so the new prompt does not cost tool reliability (measured 35/38 baseline)
- [ ] B7 Wake-word-free entry path: in a crowd, "Reachy" is a high bar — the greeter opens the
      conversation and then *teaches* the wake word in its first sentence

### Phase C — office readiness

- [ ] C1 Field checklist: office mic/speaker device names, Wi-Fi/hostname, daemon USB,
      volume range for a louder room, beam/DOA sanity, "can he hear me across the table" test
- [ ] C2 Boot-from-scratch runbook for the event machine (`scripts/` order + which systemd units),
      measured cold-start time
- [ ] C3 Post-event purge: hackday memories/faces, journal export, disk (events.db is already 21 MB)

### Phase D — carry back

- [ ] D1 Review `[carry-back]` items after the event; switch the good ones on for `home`
- [ ] D2 Fold the design into the EPIC + an ADR (0009 deployment environments) once it survives a real room

## Parked backlog from 2026-10-11 (still open, tracked here because this box has no `gh`)

1. Persist the conversation summary across shell restarts (auto fact extraction is done).
2. Stop answering the TV — directedness check + `response.cancel` (ADR 0003 violation today).
3. Direct intents for look/nod/volume like `wants_sleep`; Qwen still narrates ~1 in 4 actions.
4. Measure the 2026-10-11 endpointing changes with the home room + TV on.
5. Verify XVF3800 AEC reference (ADR 0005) — needs playing sound, ask first.
6. Over-calls: "how are you" → `get_robot_status`; narration lines can run long.
7. Pocket TTS (Marvin) on CPU + Parakeet on GPU benchmark (frees ~1.2 GB VRAM).
8. Daemon DOA breaks after an XMOS reboot; sway timing offset; delete the `.beads/` leftover dir.
9. Phase 2 cutover (#22): move remaining tools out of `app/` — B1/B2 above should land natively in `shell/`.
10. Voice candidates review (16 designed voices + Obama), publish the Obama GGUF.
