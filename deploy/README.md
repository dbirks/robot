# Deploy

Production process layout (systemd user services, per the EPIC). Units live
here so they are versioned; install with:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-server reachy-llm-proxy reachy-s2s reachy-shell
```

Services, in startup order:

| Unit | Runs | Notes |
|---|---|---|
| `reachy-mini-daemon` | robot hardware | unchanged; units live outside repo until Phase 2 |
| `llama-server` | chat LLM on :8080 (Qwen3.5-4B + mmproj) | `scripts/run_llama_server.sh`; see issue #33 for pinning |
| `reachy-vision` | optional camera VLM on :8082, CPU (Qwen3.5-0.8B), only with a text-only chat LLM | `scripts/run_vision_server.sh`; set `VISION_BASE_URL=http://127.0.0.1:8082/v1` in `.env` |
| `reachy-llm-proxy` | `python -m shell.llm_proxy` on :8081 | rewrites STT mis-hearings of "Reachy" before llama.cpp |
| `reachy-s2s` | pinned HF speech-to-speech realtime | `scripts/install_s2s.sh` once, then this |
| `reachy-shell` | audio owners + attention + tools (`python -m shell`) | the new stack |

Phase 2 (issue #22) will fold the legacy `reachy-agent` / `qwen3-tts` units
away and teach the dashboard about these names.

## Changing rooms (`REACHY_ENV`)

`reachy-shell` runs a deployment environment: `envs/home.yaml` (the desk) or
`envs/hackday.yaml` (the office demo). The file carries the persona, the attention
profile, the tool subset, the memory namespace and the room's audio/motion tuning;
the design is `docs/environments.md`. A missing file is a hard startup error, and
`REACHY_ENV` never comes from the file itself.

Precedence, lowest to highest: **code defaults → `envs/<env>.yaml` → process
environment**. `.env` is loaded into the process environment, so it outranks the
environment file, and `systemd.service(5)` applies `EnvironmentFile=` over
`Environment=` — meaning the two lines in `.env` beat both the unit and `envs/`.

One line moves the robot to the event and back:

```bash
echo 'REACHY_ENV=hackday' >> .env        # office demo; delete the line to come home
```

**Then comment these two lines out of `.env` as well**, or the home persona wins
over the attendee-safe one — hard rule 1 of the design doc:

```bash
# REACHY_INSTRUCTIONS=...   # Obama family assistant, home facts and all
# REACHY_ACK_DIR=...        # sounds/acks/obama
```

Verify before restarting anything — `--print-env` prints the resolved config as
JSON and exits without opening audio, connecting to the robot, or touching the
network (it works with the whole stack down):

```bash
.venv/bin/python -m shell --print-env | grep -E '"(env|attention_profile|tools_exclude|memories_path)"'
```

### Moving the memories into the `home` namespace (once, per machine)

Memories used to be one file for every environment, `data/memories.json`; they are
namespaced now (`data/memories.home.json`, `data/memories.hackday.json`) so the
event can never read family facts. The migration copies and never deletes, and it
never overwrites a namespaced file that already exists. It only ever writes
`memories.home.json`: run under any other `REACHY_ENV` it refuses, because the
event store has to start empty (design hard rule 1). **Stop the shell first**:
the running process read `data/memories.json` at startup and rewrites that same
path on every saved fact, so anything it learns after the copy is lost. Run it
from the repo root (`DATA_DIR` moves the store elsewhere):

```bash
systemctl --user stop reachy-shell
.venv/bin/python -m shell --migrate-memory     # {"action": "copied", "facts": 6}
systemctl --user start reachy-shell
```

`data/memories.json` is left in place; archive it by hand once
`data/memories.home.json` reads right. The hack-day store is deliberately not
created here — it starts empty on first use and gets purged after the event.

## Prereqs on a fresh machine

```bash
scripts/install_s2s.sh        # clones + pins the service in ~/.local/share/reachy
uv sync --extra kws
scripts/get_kws_model.sh      # wake-word model + tokenized keywords into models/kws
scripts/build_qwentts.sh      # only for S2S_TTS=qwen3: Pascal build of the Qwen3-TTS engine
HF_HUB_OFFLINE=0 scripts/run_s2s.sh   # once, to download Parakeet/Kokoro; Ctrl-C when "Uvicorn running"
```
