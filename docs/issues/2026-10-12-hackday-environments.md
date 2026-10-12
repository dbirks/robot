# Hack day: deployment environments, a greeter that makes eye contact, and personas

> Draft mirror of the GitHub issue this session files. Canonical design doc:
> [`docs/environments.md`](../environments.md). Delete this file once the issue exists.

**Why:** Reachy has been living on the home desk. He is moving to the office for an AI hack
day, where he has to greet strangers in a noisy room. We want both rooms, no branch, and the
good parts carried back home afterwards.

**How:** `REACHY_ENV` selects a *deployment environment* — a bundle of attention profile +
persona prompt + tool subset + namespaced memories/faces + audio devices, declared in
`envs/home.yaml` / `envs/hackday.yaml`. Iterate on `main`.

## Phase A — environment layer (no behaviour change)

- [ ] A1 `shell/env.py`: load `envs/<name>.yaml`, layered into `ShellConfig` (code defaults < env file < process env vars)
- [ ] A2 `envs/home.yaml` = today's `.env` values, behaviour byte-equivalent
- [ ] A3 `envs/hackday.yaml` on the `event` attention profile + attendee-safe persona
- [ ] A4 Journal + dashboard carry `env` on every event; `python -m shell --print-env`
- [ ] A5 `MemoryStore` namespaces (`memories.home.json` / `memories.hackday.json`); migrate today's `data/memories.json` → home
- [ ] A6 systemd `Environment=REACHY_ENV=home`; event swap documented in `deploy/README.md`
- [ ] A7 Tests: precedence, unknown-key tolerance, home parity
- [ ] A8 Pin the lint toolchain (latest ruff reformats two `experiments/` files CI currently accepts)

**Hard rule:** the family names and the home address in `data/memories.json` must never be
reachable in `hackday`. Namespacing is the privacy control, not a nicety.

## Phase B — hack-day capability

- [ ] B0 **Look people in the eye**: port the legacy face lock-on (`app/face_tracker.py` —
      InsightFace `buffalo_sc` on CPU, yaw/pitch targets that keep a face centred) natively into
      `shell/`: frames from `shell/camera.py`, targets through `MotionOwner` (the only caller of
      `set_target`), and a "someone is looking at me" trigger that opens with *"Hi, how's it
      going? Welcome to the hackathon."* Needs measured CPU cost in `docs/pins.yaml` and
      ADR 0003-style gating so he never nags the whole room.
- [ ] B0b **Persona picker** — Obama (today) vs **Alan Rickman**. A persona = talker GGUF +
      speaker + TTS instruct + ack sounds + system prompt: how he sounds *and* how he behaves.
      Nests under the environment. See the separate Alan Rickman voice issue (#TBD).
- [ ] B1 Wire face recognition natively into the shell (`learn_face`/`identify_face` answer
      "not available" today — `build_handlers` gets no `face_tracker`); per-env faces file
- [ ] B2 `greet` tool (hello + look at the person + nod)
- [ ] B3 Proactive greeter loop: while idle, low-rate presence poll → un-greeted face →
      `RealtimeClient.insert_user_transcript()` to prompt a spoken hello; per-person + global cooldowns
- [ ] B4 `introduce_self` + `about_the_project` Q&A grounded in a short `docs/faq-hackday.md`
- [ ] B5 `icebreaker` deck for booth silences
- [ ] B6 Attendee persona benchmarked through `experiments/2026-10-04-llm-bakeoff/toolbench.py`
      (must not cost tool reliability; baseline 35/38)
- [ ] B7 Wake-word-free entry: in a crowd "Reachy" is a high bar — the greeter opens, then teaches the wake word

## Volume (partly done)

- [x] Named levels retuned down (cap 2.0 → 1.2, "normal" 1.3 → 0.5) and the live sink 1.70 → 0.50 — `b5cca5c`
- [ ] V1b Per-env volume policy: a conference room needs headroom the home table does not
- [ ] V1c `reachy-audio-init.service` hardcodes `amixer -c 3` and fails at boot (`Invalid card number '3'`) — match the card by name

## Phase C — office readiness

- [ ] C1 Field checklist: office mic/speaker device names, network/hostname, daemon USB, volume
      range for a louder room, beam/DOA sanity, "can he hear me across the table" test
- [ ] C2 Boot-from-scratch runbook + measured cold-start time for the event machine
- [ ] C3 Post-event purge: hackday memories/faces, journal export, disk (`data/events.db` is 21 MB)

## Phase D — carry back

- [ ] D1 Review `[carry-back]` items after the event; switch the good ones on for `home` explicitly
- [ ] D2 Fold into the EPIC + ADR 0009 (deployment environments) once it survives a real room

## Parked backlog from 2026-10-11 (carried in so it can't be lost again)

1. Persist the conversation summary across shell restarts (auto fact extraction is done)
2. Stop answering the TV — directedness check + `response.cancel` (ADR 0003 violation today)
3. Direct intents for look/nod/volume like `wants_sleep`; Qwen still narrates ~1 in 4 actions
4. Measure the 2026-10-11 endpointing changes with the home room + TV on
5. Verify XVF3800 AEC reference (ADR 0005) — needs playing sound through the robot
6. Over-calls: "how are you" → `get_robot_status`; narration lines run long
7. Pocket TTS (Marvin) on CPU + Parakeet on GPU benchmark (would free ~1.2 GB VRAM)
8. Daemon DOA breaks after an XMOS reboot; sway timing offset; delete the `.beads/` leftover dir
9. Phase 2 cutover (#22): move the remaining tools out of `app/` — do B0/B1/B2 natively in `shell/`
10. Voice candidates review (16 designed voices + Obama), publish the Obama GGUF

## Constraints

i7-6700K (4C/8T Skylake, no AVX-512), GTX 1070 8 GB sm_61 at ~7.8/8 GB used, Arch, Reachy Mini
with XMOS XVF3800. Every new model/runtime needs a `docs/pins.yaml` entry with a measured cost
(ADRs 0004-0006). Tests stay hardware-free (`uv run pytest tests/`). Attention: a greeting is
speech, a state change is not (ADR 0003).
