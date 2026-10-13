# Knobs — the authoritative table

Every `REACHY_*` / `S2S_*` / `DASHBOARD_*` / `LLAMA_*` name an operator is allowed to
type, **the reader that consumes it** (file:line, or symbol where the file moves), and
what each room resolves today. If a name is not in this table, setting it does nothing:
no code reads it, and nothing will ever warn you. That is how an hour dies at a booth.

Verify any row in one command — hardware-free, opens no audio, connects to no robot:

```bash
REACHY_ENV=hackday .venv/bin/python -m shell --print-env | grep -E '"(env|attention_profile|tools_include|tools_exclude|memory_namespace|memories_path|learn_facts|auto_sleep_s|greeter|motion_hz|xmos_agc_max_gain)"'
```

Precedence, lowest to highest: **code default → `envs/<env>.yaml` → process
environment (`.env` included)**. A name that appears in *both* the env file and the
process environment has a dead line in the file, with no error and no log
(docs/environments.md hard rule 5). `deploy/README.md` "Changing rooms" is the procedure;
this file is the index.

`read by` names the **first** reader — the one that decides whether the name exists at
all. `shell/config.py` was edited twice while this table was being written, so the shell's own
readers are named by **symbol** (the `ShellConfig` field or helper), not by line: a line cite
into a live file is wrong within a day. Same rule for the files that move most
(`shell/__main__.py`, `shell/sleep.py`, `shell/audio/speaker.py`, `shell/motion/**`,
`dashboard/journal_read.py`). Line numbers are kept only where they are quiet and
load-bearing: the launch scripts, the systemd units, `dashboard/server.py`.

## 1. Process environment — the shell (`python -m shell`)

All rows: `shell/config.py` reads the variable, `shell/env.py` `LAYERS` gives the env
file key that fills the same field when the process variable is unset or empty.

| Name | Read by | Effect | home | hackday |
|---|---|---|---|---|
| `REACHY_ENV` | `shell/config.py` `REACHY_ENV_VAR` + `shell/env.py` `resolve_env_name()` | picks `envs/<name>.yaml`; **never** settable from the file itself | unset → `home` | `hackday` (`.env`) |
| `REACHY_INSTRUCTIONS` | `shell/config.py` `instructions` | whole system prompt (`persona_instructions()` appends `TOOLS_NOTE`+`TIME_NOTE`) | `envs/home.yaml` `persona.instructions` | `envs/hackday.yaml:51` |
| `REACHY_PROFILE` | `shell/config.py` `attention_profile` | attention profile name, `shell/attention/profiles.py` | `quiet` | `event` |
| `REACHY_TOOLS_INCLUDE` | `shell/config.py` `tools_include` | comma list; empty = every tool the shell knows (21) | unset | unset |
| `REACHY_TOOLS_EXCLUDE` | `shell/config.py` `tools_exclude` | comma list; always wins over include | unset | `go_to_sleep, reset_conversation, identify_face, learn_face, forget_face` → 16 tools |
| `REACHY_MEMORY_NAMESPACE` | `shell/config.py` `_memory_namespace()` | which `data/memories.<ns>.json`; empty = follow the env name; a non-`home` room landing on `home` is a hard `EnvError` (`guard_memory_namespace`) | `home` | `hackday` |
| `REACHY_LEARN_FACTS` | `shell/config.py` `learn_facts` (`_bool_env`: `0/false/no/off`) | unprompted fact mining on compaction; also strips the standing prompt sentence (`reconcile_learn_facts_prompt`) | true | false |
| `REACHY_GREETER` | `shell/config.py` `greeter_enabled` | proactive hello (`shell/vision/greeter.py`) | false | true |
| `REACHY_GREETER_DEBUG` | `shell/config.py` `greeter_debug` | log every dwell/angle/verdict; no behaviour change | false | false |
| `REACHY_GREETER_MAX_PER_HOUR` | `shell/config.py` `greeter_max_per_hour` | hard room-rate ceiling | 6 | 6 |
| `REACHY_GREETER_DWELL_S` | `shell/config.py` `greeter_dwell_s` | how long a face must look at him | 2.0 | 2.0 |
| `REACHY_GREETER_MIN_SPACING_S` | `shell/config.py` `greeter_min_spacing_s` (file key `behaviour.greeter_cooldown_s`) | global spacing | 120 | 120 |
| `REACHY_GREETER_GROUP_MIN_SPACING_S` | `shell/config.py` `greeter_group_min_spacing_s` (file key `behaviour.greeter_group_cooldown_s`) | group-notice spacing | 300 | 300 |
| `REACHY_GREETER_IDLE_FPS` | `shell/config.py` `greeter_idle_fps` | detector poll on an empty room | 2 | 2 |
| `REACHY_GREETER_ENGAGED_FPS` | `shell/config.py` `greeter_engaged_fps` | detector poll while a face is in frame | 3 | 3 |
| `REACHY_GREETER_FACE_CENTERING` | `shell/config.py` `greeter_face_centering`; consumed in `shell/__main__.py` (`face_centering=cfg.greeter_face_centering`) → `GreeterLimits.face_centering` | head-centring on a qualified face, through `MotionOwner` | **false** | **false** |
| `REACHY_AUTO_SLEEP_S` | `shell/config.py` `auto_sleep_s`; consumer `shell/sleep.py` (`AutoSleep`) | lie down after N s of silence with nobody engaged; 0 = never | 0 | 0 |
| `REACHY_MIC` / `REACHY_SPEAKER` | `shell/config.py` `mic_device`, `speaker_device` | audio device select; the only legal owner is `shell/audio/` (ADR 0002) | system default | system default |
| `REACHY_XMOS_AGC_MAX_GAIN` | `shell/config.py` `xmos_agc_max_gain` (`shell/audio/xmos_tuning.py`) | AGC ceiling; 0 turns AGC off (ADR 0005) | 10 | 6 |
| `REACHY_XMOS_BEAM_FOCUS` | `shell/config.py` `xmos_beam_focus` | steer both fixed beams at the wake-word direction | true | true |
| `REACHY_MOTION_HZ` | `shell/config.py` `motion_hz` | motion control-loop rate; lower it if CPU is tight (ADR 0006) | 50 | 50 |
| `REACHY_COMPACT_AFTER_S` / `REACHY_RESET_AFTER_S` | `shell/config.py` `compact_after_s`, `reset_after_s` (used by `shell/conversation.py`) | summary compaction / fresh-session timers | 180 / 300 | 180 / 300 |
| `REACHY_MAX_TURN_S` | `shell/config.py` `max_turn_s` (also named in `scripts/run_s2s.sh:51`) | shell force-ends a longer turn by feeding silence | 15 | 15 |
| `REACHY_ACK_DIR` | `shell/config.py` `ack_dir` | `<dir>/wake/*.wav`, `<dir>/think/*.wav` | `sounds/acks/obama` | `sounds/acks/obama` |
| `REACHY_PERSONA` | `shell/env.py` `PERSONA_VAR` / `persona_selection()` | names `personas/<name>.yaml`; outranks `persona.name:` in the file. Stamped by `deploy/generated/persona.env`, **never** with `REACHY_INSTRUCTIONS` | `obama` | `marvin` |
| `S2S_URL` | `shell/config.py` `s2s_url` | realtime websocket | `ws://127.0.0.1:8765/v1/realtime` | same |
| `LLM_BASE_URL` / `LLM_MODEL` | `shell/config.py` `llm_base_url`, `llm_model` | side-channel llama.cpp (compaction); `LLM_MODEL` is also the s2s `--model_name` (`scripts/run_s2s.sh:136`) | `.env` | `.env` |
| `DATA_DIR` | `shell/config.py` `data_dir` | moves `events.db` and `memories.*.json` | `data` | `data` |
| `KWS_MODEL_DIR` | `shell/config.py` `kws_model_dir`, `scripts/get_kws_model.sh:10` | wake-word model dir | `models/kws` | `models/kws` |
| `VISION_BASE_URL` / `VISION_MODEL` | `shell/robot_tools.py:72` | separate CPU VLM instead of the chat model's `--mmproj` | unset (commented in `.env`) | unset |

## 2. Process environment — the other processes

| Name | Read by | Effect | Notes |
|---|---|---|---|
| `DASHBOARD_HOST` | `dashboard/server.py:967` | bind address; empty = `127.0.0.1` | Port is **hardcoded 3001** (`dashboard/server.py:968`): there is no `DASHBOARD_PORT`, despite `scripts/booth_recovery.sh` carrying a local shell variable of that name. `/` and `/faces` serve *home* data, so `0.0.0.0` is a deliberate act (deploy/README.md). |
| `REACHY_DASHBOARD_READONLY` | `dashboard/env_panel.py:45,135` via `dashboard/server.py:55` | every mutating route refuses with a visible message | Off by default; the process has no auth at all. |
| `S2S_PORT` | `scripts/run_s2s.sh:121` | websocket port | Only `run_s2s.sh` reads it. `shell/config.py` does **not**: `S2S_URL` is one string, and moving the port without rewriting `S2S_URL` splits the two processes. |
| `S2S_TTS` | `scripts/run_s2s.sh:113` | `kokoro` \| `qwen3` \| anything else = upstream default | tonight `.env` sets `qwen3` |
| `S2S_STT` | `scripts/run_s2s.sh:122` | STT backend | default `parakeet-tdt` |
| `S2S_PARAKEET_MODEL` / `S2S_PARAKEET_DEVICE` | `scripts/run_s2s.sh:123`, `:124` | model id / device (CPU fp32 is the measured choice, ADR 0004) | |
| `S2S_QWEN3_MODEL` / `S2S_QWEN3_TALKER_GGUF` / `S2S_QWEN3_CODEC_GGUF` / `S2S_QWEN3_SPEAKER` / `S2S_QWEN3_QUANT` | `scripts/run_s2s.sh:73-81` | the Qwen3-TTS voice | Read **once at startup**: a voice change costs a `reachy-s2s` restart, measured 20-46 s of dead air. `persona.env` outranks `.env` for these keys (deploy/README.md). No pins line for tonight's 1.7B talker — `docs/pins.yaml` `tts_trial` is the 0.6B candidate. |
| `S2S_QWEN3_INSTRUCT` | `scripts/run_s2s.sh:97-99` → upstream `--qwen3_tts_instruct` | the voice **style** line: `personas/<name>.yaml voice.instruct`, stamped by `scripts/apply_env.py`. Passed **only when non-empty**; empty/absent sends no flag at all, which is upstream's `None` | Every shipped persona carries `instruct: ""` tonight, so the flag is **off** in both rooms: the voice is the Obama talker GGUF plus its speaker embedding, unstyled. It is live plumbing, not decoration, and it is what a second (VoiceDesign) voice will use. |
| `S2S_KOKORO_DEVICE` | `scripts/run_s2s.sh:66` | `cuda` \| `cpu` for the Kokoro path | tonight the engine is qwen3, so this is inert unless `S2S_TTS=kokoro` |
| `KOKORO_VOICE` | `scripts/run_s2s.sh:64` | kokoro voice id | also read by the **legacy** `app/tts_service.py:17` |
| `S2S_LIVE_TRANSCRIPTION`, `S2S_MAX_SPEECH_MS`, `S2S_VAD_THRESH`, `S2S_REOPEN_MS` | `scripts/run_s2s.sh:126-130` | upstream endpointing knobs | CPU cost of live transcription is unmeasured (ADR 0006) |
| `S2S_DIR`, `S2S_REF` | `scripts/run_s2s.sh:18`, `scripts/install_s2s.sh:7,9` | where the pinned service lives (`~/.local/share/reachy/speech-to-speech`) | |
| `HF_HUB_OFFLINE` | `scripts/run_s2s.sh:29`, pinned `Environment=` in `deploy/systemd/reachy-s2s.service` | offline mode | First run on a fresh box needs `HF_HUB_OFFLINE=0` once. |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | `scripts/run_s2s.sh:24,27` | where the service reaches the LLM | **Default is the proxy on :8081.** Put `OPENAI_BASE_URL=http://localhost:8080/v1` in `.env` — as `.env.example:29` shows — and the name-fixing proxy (`reachy-llm-proxy`) is silently bypassed and "Reachy" goes back to being mis-heard. Use `S2S_URL`/`.env` only when you mean to. |
| `LLAMA_MODEL_PATH`, `LLAMA_MMPROJ_PATH` | `scripts/run_llama_server.sh:16`, `:22` | the GGUF + projector | live defaults are `models/gguf/mtp/Qwen3.5-4B-UD-Q4_K_XL.gguf` and `models/gguf/mtp/mmproj-F16.gguf` — **not** the `qwen3.5-4b-q4_k_m.gguf` / `mmproj-BF16.gguf` pair `README.md` still tells you to download (Pascal has no BF16, ADR 0004) |
| `LLAMA_CTX` | `scripts/run_llama_server.sh:36` | total context across `LLAMA_PARALLEL` slots | default **24576** (2 × 12K), not the 4096 `README.md` quotes |
| `LLAMA_PORT`, `LLAMA_HOST`, `LLAMA_PARALLEL`, `LLAMA_GPU_LAYERS`, `LLAMA_THREADS`, `LLAMA_FLASH_ATTN`, `LLAMA_CACHE_TYPE_K/_V`, `LLAMA_CACHE_REUSE`, `LLAMA_SERVER_BIN`, `LLAMA_DIR` | `scripts/run_llama_server.sh:24-96` | the llama.cpp serving flags | KV types must stay matched (ADR 0004) |
| `LLM_API_KEY` | `app/config.py:15` (legacy loop); the service uses `OPENAI_API_KEY` (`scripts/run_s2s.sh:24`) | the local path still needs a dummy string | `shell/config.py` has no `LLM_API_KEY` reader |
| `TTS_ENGINE` / `TTS_SERVER_URL` / `TTS_SPEAKER` / `KOKORO_LANG` / `KOKORO_DEVICE` | `app/tts_service.py:14-22` | **legacy loop only** (`python -m app`); the realtime stack's engine is `S2S_TTS` | live `.env` sets `TTS_ENGINE=kokoro`, which changes nothing for `reachy-shell` |
| `REACHY_DAEMON_URL` | `scripts/booth_recovery.sh:40` | where the recovery script asks `/api/media/status` | `shell/camera.py:86` hardcodes `http://localhost:8000`; the shell does **not** read this variable |
| `REACHY_INSTRUCTIONS` (drop-in caution) | see §1 | `deploy/generated/persona.env` deliberately never sets it | one global file read by both rooms |

## 3. Units an operator will be told to touch

`systemctl --user list-unit-files` on this box, 2026-10-12. The repo ships five stack units
(`deploy/systemd/`) plus two kiosk units (`deploy/kiosk/`); **four more that this box runs
live only in `~/.config/systemd/user/`** and are not versioned here — so a fresh machine
cannot `systemctl --user start` them from a checkout. That is the gap between
`deploy/README.md`'s install block and the box at the venue.

| Unit | Unit file | On this box | Notes |
|---|---|---|---|
| `reachy-mini-daemon` | `~/.config/systemd/user/` (outside the repo, Phase 2) | enabled, running | `ExecStart=uv run reachy-mini-daemon -p /dev/ttyACM0`; SDK 1.8.0; `--sim` is real (`reachy_mini/daemon/app/main.py:476`) |
| `llama-server` | `deploy/systemd/llama-server.service` | enabled, running | |
| `reachy-llm-proxy` | `deploy/systemd/reachy-llm-proxy.service` | enabled, running | :8081, must stay in front of llama.cpp |
| `reachy-s2s` | `deploy/systemd/reachy-s2s.service` | enabled, running | voice flags read once |
| `reachy-shell` | `deploy/systemd/reachy-shell.service` | enabled, running | the one audio owner (ADR 0002) |
| `reachy-vision` | `deploy/systemd/reachy-vision.service` | **disabled** here | optional CPU VLM, only with a text-only chat LLM |
| `reachy-dashboard` | `~/.config/systemd/user/` (outside the repo) | enabled, running | not in `deploy/systemd/`, though the docs tell you to restart it |
| `reachy-agent` | `~/.config/systemd/user/` | **disabled — retired** | see the warning below |
| `qwen3-tts` | `~/.config/systemd/user/` | **disabled — retired** | second model onto a card at 7.84/8 GB |
| `reachy-audio-init` | `/etc/systemd/system/` — a **system** unit | enabled, currently `failed` | see below |
| `reachy-kiosk`, `reachy-kiosk-watch` | `deploy/kiosk/` | **not installed** | staged, never verified on a screen (#37); `scripts/install_kiosk.sh --run` |

### Two facts that cost an hour when they are missing

- **`reachy-shell.service` declares `BindsTo=reachy-mini-daemon.service`
  (`deploy/systemd/reachy-shell.service:5`).** Restarting the daemon takes the shell
  down with it and does not bring it back: `systemctl --user show
  reachy-mini-daemon -p BoundBy` answers `BoundBy=reachy-shell.service`. Every documented
  `restart reachy-mini-daemon` must be followed by `start reachy-shell` — which is why
  `deploy/README.md` "Powering the robot back on" is a three-step block and not two.
- **`systemctl --user start reachy-agent` steals the audio.** It runs the legacy
  `python -m app`, a second mic and speaker owner (ADR 0002). While it is up, the live
  shell hears the room through the wrong device and its own voice stops being
  cancelled — with the shell still `active` in every status line. Both `reachy-agent`
  and `qwen3-tts` are retired (issue #22) and stay disabled; `scripts/booth_recovery.sh`
  never starts them and the dashboard cannot either (`check_env_safety.sh` check 11).

### One more, because `--user` cannot see it

`reachy-audio-init.service` is a **system** unit (`/etc/systemd/system/`), so
`systemctl --user --failed` never lists it. It is `enabled` and it is failing on this box
right now:

```
Process: 481 ExecStart=/usr/bin/amixer -c 3 set PCM,0 60 (code=exited, status=1/FAILURE)
```

Hardcoded card 3 no longer exists (`Invalid card number '3'`); match the card by name.
Until then: `systemctl status reachy-audio-init` (no `--user`), and do not read a
`--user --failed` "all clear" as "everything on this machine is healthy".
Tracked in `docs/environments.md` (V1).

## 4. Paths the docs name

| Path | Status |
|---|---|
| `data/events.db` | exists — the journal (`mode=ro` reads only; `mono` is per-process, never subtract across a restart) |
| `data/memories.home.json`, `data/memories.json` (legacy) | exist |
| `data/memories.hackday.json` | **absent by design** — the booth store starts empty (created on first explicit `remember`); `rm -f` is the documented purge |
| `data/known_faces.json` | exists, and is **one shared home file** — faces are not namespaced until B1, which is why the three face tools are excluded from `hackday` |
| `data/sessions.db` | exists (legacy dashboard store, last row 2026-07-28); never rendered by the new dashboard |
| `data/<...>/faces.<ns>.json` | **never implemented** — `docs/environments.md` schema v1 sketches it; B1 owns it |
| `sounds/acks/obama/{wake,think}` | exists — the only ack pool on this box |
| `models/kws/` | exists (`scripts/get_kws_model.sh`) |
| `models/gguf/mtp/Qwen3.5-4B-UD-Q4_K_XL.gguf`, `models/gguf/mtp/mmproj-F16.gguf` | exist — what the live unit loads |
| `models/gguf/qwen3.5-4b-q4_k_m.gguf`, `models/gguf/mmproj-BF16.gguf` | exist on disk but are the superseded pair that `README.md` still tells you to download |
| `models/qwen3-tts/*.gguf` | exist (the Qwen3-TTS weights) |
| `deploy/generated/persona.env` | **not present on this box** — `scripts/apply_env.py` renders it (gitignored); both units load it with `-` so its absence is legal, and means the persona picker has not been applied since a clean |
| `docs/pins.yaml` keys `services.speech-to-speech`, `services.llama-cpp`, `models.llm`, `models.llm_alternative`, `models.vlm`, `models.stt`, `models.tts_production`, `models.tts_trial`, `models.face_detection`, `models.face_detection_alt`, `models.kws`, `runtime.*` | all exist |
| `models.piper/` | exists — legacy Piper voice, live only when `TTS_ENGINE` is neither `kokoro` nor `qwen3` |

## 5. Script flags this repo advertises

All verified against the scripts' own argument parsing (no script was executed for a side
effect; `bash -n` on the shell ones):

| Command | Real? |
|---|---|
| `scripts/preflight_demo.sh`, `DEMO_ENV=hackday scripts/preflight_demo.sh` | yes (`scripts/preflight_demo.sh:21,101`) — it runs `check_env_safety.sh` and the suite as two of its steps |
| `bash scripts/check_env_safety.sh` | yes — **13** numbered checks; run it and read its own output, it prints one `ok`/`FAIL` line per check |
| `scripts/booth_recovery.sh [--dry-run\|-n] [--only <unit>] [--all] [--yes\|-y] [--no-preflight]` | yes (`scripts/booth_recovery.sh:55-67`) |
| `scripts/tail_journal.py --minutes N [--grep X] [--all] [--json] [--db PATH]` | yes (`scripts/tail_journal.py:169-173`) |
| `scripts/apply_env.py [--env] [--persona] [--out] [--lock] [--check] [--restart] [--strict] [--lock-held]` | yes (`scripts/apply_env.py:59-70`) |
| `.venv/bin/python scripts/bench_aec.py --person --out data/bench-aec.json` | yes — both flags are in `main()`'s parser (`scripts/bench_aec.py:885+`; that file is under active edit, so trust `--help` over a line number here) — opens the mic, so the shell must be stopped |
| `scripts/install_kiosk.sh [--run]` | yes (`scripts/install_kiosk.sh:23`) |
| `python -m shell --print-env \| --print-personas \| --migrate-memory` | yes (`shell/__main__.py` `_parser()`) |

## 6. Names a doc promised and the code never implemented

Called out rather than quietly deleted — each one was copied by somebody at some point:

| Name | Where it was promised | Verdict |
|---|---|---|
| `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE` | `README.md` "GPU Compatibility" / "VRAM Budget", `docs/setup-gtx1070-arch.md:34,97,102-103` | **never implemented** — nothing in `app/`, `shell/` or `scripts/` reads them (the legacy STT is Parakeet via `onnx_asr`, CPU-forced in `app/stt_service.py:12-21`, and the realtime STT device is `S2S_PARAKEET_DEVICE`). The "reliable fallback" these two lines promise does not exist. |
| `DASHBOARD_PORT` | implied by "port 3001" everywhere, and by a local variable in `scripts/booth_recovery.sh:45` | **never implemented** — `dashboard/server.py:968` hardcodes 3001. Changing the port means editing that line, not setting a variable. |
| `REACHY_DAEMON_URL` as a shell knob | `scripts/booth_recovery.sh:40` | real, but only for that script; `shell/camera.py:86` hardcodes `http://localhost:8000` |
| `greeter_quiet_minutes` | `docs/environments.md` schema-v1 sketch | **not read by the loader** (unknown keys are ignored on purpose); the shipped global limiter is `behaviour.greeter_cooldown_s` |
| `memory.namespace` → `faces.<ns>.json` | `docs/environments.md:46` | memories only; per-env faces is B1 and is why `hackday` drops the face tools |
| `REACHY_HEADER_FACE_CENTERING` | quoted in review as a doc'd knob | **never existed**; the real name is `REACHY_GREETER_FACE_CENTERING` (`shell/config.py` `greeter_face_centering`). Verified absent from the whole tree and from `git grep HEAD`. |
| `vision` device env-file keys, `audio.device` | nowhere | do not invent them: use `REACHY_MIC` / `REACHY_SPEAKER` |

Anything added to `shell/env.py` `LAYERS` or `shell/config.py` must gain a row here in the
same PR — this table, `shell/env.py` `LAYERS` and `scripts/check_env_safety.sh` are the
three places one knob lives.
