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
  | Tool set | full (incl. `go_to_sleep`, `set_volume`, `remember`) | 16 of 21 schemas (`web_search` included); all three face tools + `go_to_sleep` + `reset_conversation` excluded; `STATE_CHANGING` needs `tool_confidence` 0.8 |
  | Memory namespace | `data/memories.home.json` | `data/memories.hackday.json` — separate files: family names, kids, home address **never** loaded. Faces are **not** namespaced yet (`data/known_faces.json` is one shared file), so the face tools are excluded from `hackday` instead; per-env faces is B1 |
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

As implemented (Phase A): the keys above plus `motion.hz` and `conversation:
{compact_after_s, reset_after_s, max_turn_s}`, each with the `REACHY_*` process
variable already used by `shell/config.py` as its override. `REACHY_TOOLS_INCLUDE` /
`REACHY_TOOLS_EXCLUDE` (comma-separated) and `REACHY_MEMORY_NAMESPACE` are new.
An explicit `null`/`""` in a file means "no opinion", not "clear the value", so
`audio.mic: null` never fights `REACHY_MIC` — and an *empty* `REACHY_*` in the process
environment means the same thing (`REACHY_INSTRUCTIONS=` in `.env` does not silence the
file). Of the `behaviour:` block `learn_facts` (default true, `REACHY_LEARN_FACTS`) and the
greeter are implemented: `greeter`, `greeter_dwell_s`, `greeter_cooldown_s`,
`greeter_group_cooldown_s`, `greeter_max_per_hour`, `greeter_debug`, `greeter_idle_fps`,
`greeter_engaged_fps`, `greeter_face_centering` — each with its `REACHY_GREETER_*` process
override (`shell/env.py` LAYERS). `greeter_quiet_minutes` above is design sketch and is
ignored by the loader; the shipped global limiter is `greeter_cooldown_s`.
LLM/s2s endpoints deliberately stay
in `.env`: `reachy-s2s` reads the same values and one fact must have one source.

## Hard rules

1. **No private home facts in a public room.** *Memory* is namespaced per env
   (`data/memories.<ns>.json`, and resolution refuses to let a non-`home` env land on the
   `home` store). **Faces are not namespaced until B1**: `app/face_tracker.py` hardcodes
   `data/known_faces.json`, so `identify_face` / `learn_face` / `forget_face` are excluded
   from `hackday` outright rather than pointed at another file — until B1 lands they are the
   one rule-1 hole the layer cannot close, and they must not come back without it. Auto
   fact extraction is off in `hackday` (`behaviour.learn_facts: false`): one shared store
   plus a 5-minute session reset would present attendee A's facts to attendee B, so the
   booth only keeps what a guest explicitly asks him to remember. The hack-day store starts
   empty and is purged after the event unless we decide to keep attendees who opted in.
2. **ADR 0003 still applies**: a greeting is speech, a state change is not. In `hackday`,
   `STATE_CHANGING` gets the strictest lease bar (`tool_confidence: 0.8` already in the
   `event` profile) and `go_to_sleep` / `reset_conversation` are excluded from the tool list.
3. **Every new model/runtime on the robot box gets a `docs/pins.yaml` entry with a
   measured cost** (ADRs 0004-0006) — this includes the InsightFace CPU budget for
   face-based greeting.
4. Hardware-free tests only in `tests/`; the greeter is driven by an injected frame, never
   by a real camera.
5. **An operator-visible switch must not fail open.** A present-but-empty `REACHY_*`
   (`REACHY_INSTRUCTIONS=` in `.env`, an `Environment=REACHY_TOOLS_EXCLUDE=` drop-in) is
   "no opinion", the same as an explicit `null` in the env file — it may not delete the
   attendee-safe persona or re-advertise an excluded tool with exit 0. Measured on this box
   2026-10-12 with a throwaway user unit: systemd applies `EnvironmentFile=` **after**
   `Environment=`, so the `.env` line outranks the unit; `reachy-shell.service` no longer
   pins `REACHY_ENV` at all and `.env` is the single source for the room.

## Work list

### Phase A — environment layer (no behaviour change)

- [x] A1 `shell/env.py`: load `envs/<name>.yaml`, layer into `ShellConfig` (defaults < file < env vars)
- [x] A2 `envs/home.yaml` = today's `.env` values, byte-equivalent behaviour; `.env` keeps working as an override
      — **correction**: the table above says home runs `home_tv`. It does not: the live `.env` never set
      `REACHY_PROFILE`, so today's resolved profile is `quiet` and `envs/home.yaml` says `quiet`. Parity beat
      the table; switching to `home_tv` is a behaviour change and one line away when someone wants it.
      Also note `.env` sits *above* the env file in the precedence, so a leftover `REACHY_INSTRUCTIONS` in
      `.env` silently outranks the attendee-safe persona — `deploy/README.md` says to comment it out for the event.
- [x] A3 `envs/hackday.yaml` with the `event` attention profile + attendee-safe persona
      — excludes are applied at assembly (`apply_tool_policy`), so both the schema and the handler go away.
      Faces are the hole: `identify_face`/`learn_face` still route to `app/face_tracker.py`, which hardcodes
      `data/known_faces.json`. Namespacing them is B1 (it cannot be done from `shell/` without touching `app/`).
      **Done 2026-10-12 (review P0): the face tools are excluded from `hackday`** until B1 lands, and
      `tests/test_env.py` + `scripts/check_env_safety.sh` fail if any `*_face` tool survives that schema.
- [x] A4 Journal carries `env` on every event; `python -m shell --print-env` prints the resolved config.
      Not done: the dashboard badge — it never reads the new layer, tracked as #38
      — every `Journal` row now gets `env` in its payload (explicit payload keys still win). The dashboard has
      no journal reader yet at all (it still shows the legacy `data/sessions.db`), so there was nothing to filter;
      the stamp is in the DB and the UI work lands with whoever wires the Inspect page up to ADR 0007.
- [x] A5 `MemoryStore` namespaces (`memories.home.json` / `memories.hackday.json`); migrate current `data/memories.json` → home
      — namespace defaults to the env name, so an env can never inherit another's facts by omission. The migration
      (`python -m shell --migrate-memory`) copies, never deletes, never overwrites, and refuses to write anywhere
      but `memories.home.json`. **Run on this box 2026-10-12** (shell stopped): 6 facts copied to
      `data/memories.home.json`; the legacy `data/memories.json` is still in place, unarchived.
      Resolution-side guard added with the same rule: a non-`home` env that lands on the `home`
      namespace (`REACHY_MEMORY_NAMESPACE=home`) is a hard `EnvError`, not a leak.
- [x] A6 systemd: event swap documented in `deploy/README.md`
      — the unit no longer carries `Environment=REACHY_ENV=home` (the code default is `home` and `.env` is the
      documented edit, so the line was a second source for the room). Measured 2026-10-12 with a throwaway user
      unit: `EnvironmentFile=` overrides `Environment=`. Booting with `REACHY_ENV` never set now logs a **WARN**
      naming the environment it fell back to, not an INFO.
- [x] A7 Tests: precedence, unknown-key tolerance, home parity (resolved instructions match today's)
      — `tests/test_env.py` (33 tests: precedence both ways including empty values, null, unknown keys, missing
      file, `--print-env` purity + round trip, namespaces, the home-store guard, migration, tool policy, the
      privacy scan of prompts and schemas) plus two journal stamp tests.
- [x] A9 2026-10-12 review fixes (privacy + home-parity reviews), all hardware-free tests plus the new gate:
      face tools out of `hackday`; present-but-empty `REACHY_*` = no opinion (and blank numbers no longer crash
      `float("")`); the home-namespace guard above; `behaviour.learn_facts` off in `hackday` (auto extraction
      bypassed the attention gate and was replayed to the next attendee); no real names or household framing left
      in the fact-extraction prompt / summary prompt / `remember` schema, which the privacy test now scans;
      `persona_instructions` append is idempotent, so `--print-env` output can be pasted back;
      the dashboard binds `127.0.0.1` unless `DASHBOARD_HOST` says otherwise (it served `/faces` and
      `data/sessions.db` to the LAN with no auth); `.env.example` documents the four `REACHY_*` variables that
      outrank the env file and comments `REACHY_PROFILE` out; `scripts/check_env_safety.sh` is the gate that
      runs on the morning of the demo.
      Deferred on purpose: per-env faces files (B1), `events.<env>.db` instead of one shared journal, the
      greeter/`tool_confidence` booth calibration, the bench measurement that completes the
      `docs/pins.yaml` InsightFace entry (detection-only is measured; the per-sample frame read is
      UNMEASURED PENDING BENCH), and the migration
      crash-safety (tmp + `os.replace`) called out in the review.
- [ ] A8 CI drift: `uvx ruff format --check .` currently flags
      `experiments/2026-10-03-voice-design/design.py` and `experiments/2026-10-04-llm-bakeoff/toolbench.py`
      with the latest ruff (the repo has no ruff pin). Pin the lint toolchain or format the files.

### Phase B — hack-day capability

- [ ] B0 **Look people in the eye.** Port the legacy face lock-on (`app/face_tracker.py`:
      InsightFace `buffalo_sc` on the CPU execution provider, head yaw/pitch targets that keep
      a face centred — `OVERSHOOT_SCALE`/`MAX_YAW`/`MAX_PITCH`) natively into `shell/`:
      frames from `shell/camera.py`, targets through `MotionOwner` (the single caller of
      `set_target`), and a "someone is looking at me" trigger that opens with a spoken hello.
      Needs a measured CPU cost in `docs/pins.yaml` (ADRs 0004/0006) and ADR 0003-style gating
      so a face scanning a laptop is not greeted.
- [ ] B0b **Persona picker**: Obama (today) vs Alan Rickman. A persona = talker GGUF + speaker +
      TTS instruct + ack sounds + system prompt, i.e. how he sounds *and* how he behaves.
      Nests under the environment (`persona: obama` / `persona: rickman`). The Alan Rickman
      voice work of 2026-10-08 (`models/voices/marvin-rickman/`) needs its own issue — see
      `docs/notes/alan-rickman-voice-issue.md`.
- [ ] V1 Volume: named levels retuned down 2026-10-12 (cap 2.0 → 1.2, "normal" 1.3 → 0.5) and the
      live sink dropped 1.70 → 0.50. Still open: per-env volume policy (a conference room needs
      headroom the home table does not) and `reachy-audio-init.service`, which hardcodes
      `amixer -c 3` and now fails at boot with `Invalid card number '3'` — match the card by name.
- [ ] B1 Wire face recognition natively into the shell (`learn_face` / `identify_face`
      return "not available" today because `make_handlers` gets no `face_tracker`);
      per-env faces file; measured InsightFace CPU cost in `pins.yaml`
- [ ] B2 `greet` tool (name-or-generic hello + look at the person + nod) — the model's hands
- [x] B3 **Proactive greeter**: shipped 2026-10-12 — detection-only presence poll (2 FPS idle /
      3 FPS engaged, zero while asleep; `shell/camera.py` + `shell/vision/greeter.py`) → a face
      dwelling ~2 s looking at him → `client.insert_user_transcript(origin=greeter)` prompts the
      persona's one-sentence hello, and the greeter origin means that reply can never renew the
      attention lease (`exchange_for_reply`, 0003). Global spacing 120 s, group notice 300 s,
      cap 6/h, per-presence once; on in `hackday`, off in `home`. Not included: head centring —
      gated behind `greeter_face_centering`, default off in both rooms until benched.
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
- [ ] C3 Post-event purge: hackday memories, journal export, disk (events.db is already 21 MB)
      — faces are the trap here: `data/known_faces.json` is the **home** gallery until B1 namespaces it, and
      `hackday` excludes the face tools, so nothing attendee-shaped should be in it. Check before deleting, and
      never purge it blind.

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
