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

## Powering the robot back on (travel day)

The daemon does not reliably reattach to a robot that was powered off while it was
running: it keeps serving `/api/media/status` with `available:false` and the greeter
and camera tools go quietly inert. After any power cycle, at the venue, in this order:

```bash
systemctl --user restart reachy-mini-daemon   # re-enumerate the robot, camera, audio
scripts/preflight_demo.sh                     # must be clean before you trust anything
systemctl --user start reachy-shell
```

`scripts/preflight_demo.sh` checks for the stale-daemon state and tells you so; do not
skip it because the audio devices "look present" - the ALSA card enumerates over USB
even with the robot switched off.

## Before the doors open

```bash
scripts/preflight_demo.sh                     # 13 checks: audio route, resolved room,
                                              # safety gate, suite, model files on disk,
                                              # egress for web_search, disk, booth memory
                                              # freshness. Exit nonzero = do not start.
DEMO_ENV=hackday scripts/preflight_demo.sh    # also fails if the room did not take
```

Ordered by how much each failure hurts: the PipeWire route first, because if playback
is not landing on the Reachy card the XVF3800 gets no echo reference and nothing about
barge-in works (issue #40, ADR 0005).

The echo bench is a separate, ten-minute, hands-on test - it needs the robot playing
sound and the shell stopped so nothing else owns the mic:

```bash
systemctl --user stop reachy-shell
.venv/bin/python scripts/bench_aec.py --person --out data/bench-aec.json
systemctl --user start reachy-shell
```

It answers the one question that decides #40: when he is talking, is a person still
audible in the capture? >=~50% of their silent-room level means the problem is our
thresholds and policy; <=~10% means the chip's double-talk suppression is eating them
and no amount of gate-widening will help. Record it either way.

`docs/booth-card.md` is the printable one-pager for the table: who he is, the wake
word, what runs locally versus what does not, and what to do when he stops responding.

## Changing rooms (`REACHY_ENV`)

`reachy-shell` runs a deployment environment: `envs/home.yaml` (the desk) or
`envs/hackday.yaml` (the office demo). The file carries the persona, the attention
profile, the tool subset, the memory namespace and the room's audio/motion tuning;
the design is `docs/environments.md`. A missing file is a hard startup error, and
`REACHY_ENV` never comes from the file itself.

Precedence, lowest to highest: **code defaults → `envs/<env>.yaml` → process
environment**. `.env` is loaded into the process environment, so it outranks the
environment file. An *empty* value counts as unset on both sides: `REACHY_INSTRUCTIONS=`
in `.env` does not silence the env file, and neither does an explicit `null` in the file.

The unit deliberately does not pin `REACHY_ENV`; `shell/config.py` defaults to `home`, so
`.env` is the one source for the room. Measured on this box 2026-10-12 with a throwaway
user unit: **`EnvironmentFile=` overrides `Environment=`** (systemd applies the file after
the inline list), so a `Environment=REACHY_ENV=home` line would have lost to the `.env`
line below — and if that precedence had been the other way round, the documented room
swap would have silently booted the home persona and the family memories in public.
Remove the pin and the question cannot come back.

One line moves the robot to the event and back:

```bash
echo 'REACHY_ENV=hackday' >> .env        # office demo; delete the line to come home
```

**Empty the booth store on purpose.** Auto fact extraction is off in `hackday`, but an
explicit "remember my name" still writes the one shared `data/memories.hackday.json`, and
the next session is handed its last facts as "what you know about the user". Before the
doors open and between demos:

```bash
rm -f data/memories.hackday.json
```

**The dashboard binds loopback by default** (`DASHBOARD_HOST`), and the kiosk uses
`http://127.0.0.1:3001`, so it is unaffected. Opening it to the room is a deliberate act,
not a default — `/faces` and `/` serve *home* data (family names, legacy transcripts):

```bash
systemctl --user set-environment DASHBOARD_HOST=0.0.0.0 && systemctl --user restart reachy-dashboard
```

**Then check `.env` for the variables that outrank the env file.** Comment these two
out, or the home persona wins over the attendee-safe one — hard rule 1 of the design
doc:

```bash
# REACHY_INSTRUCTIONS=...   # Obama family assistant, home facts and all
# REACHY_ACK_DIR=...        # sounds/acks/obama
```

And confirm none of these is set at all, in `.env` or in a `systemctl --user edit`
drop-in — each one silently makes its `envs/hackday.yaml` key dead:
`REACHY_PROFILE` (must not be `quiet` at the event), `REACHY_TOOLS_EXCLUDE`,
`REACHY_TOOLS_INCLUDE`, `REACHY_MEMORY_NAMESPACE`, `REACHY_LEARN_FACTS`.

Verify before restarting anything — `--print-env` prints the resolved config as
JSON and exits without opening audio, connecting to the robot, or touching the
network (it works with the whole stack down). Read the `instructions` line too, not
just the names:

```bash
.venv/bin/python -m shell --print-env | grep -E '"(env|attention_profile|tools_exclude|tools_include|memory_namespace|memories_path|learn_facts|instructions)"'
```

If `REACHY_ENV` was never set the shell logs a WARN naming the environment it fell
back to (`home`), so a machine that wandered into the office on the desk persona is
visible in the journal-less first seconds of the log.

### The dashboard is loopback by default

`dashboard/server.py` serves `data/known_faces.json` (the family's face names and
thumbnails) on `/faces` and hours of home conversation from `data/sessions.db` on `/`,
with no auth in front of any of it. It binds `127.0.0.1` unless you say otherwise, so
the kiosk on this box (`http://127.0.0.1:3001/conversation`) is unaffected while
strangers on the office LAN no longer get the family album. To open it deliberately —
set the host in the unit or the environment and restart `reachy-dashboard`:

```bash
systemctl --user set-environment DASHBOARD_HOST=0.0.0.0   # or a drop-in: Environment=DASHBOARD_HOST=0.0.0.0
systemctl --user restart reachy-dashboard
```

Do not do that from the booth unless you have also cleared `data/known_faces.json`
and `data/sessions.db` out of the way (Phase C3 purge): the faces are still one shared
file until B1 lands.

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

Ran on this box 2026-10-12 (`{"action": "copied", "facts": 6}`); the legacy
`data/memories.json` is still sitting next to the namespaced one, unarchived.

## Prereqs on a fresh machine

```bash
scripts/install_s2s.sh        # clones + pins the service in ~/.local/share/reachy
uv sync --extra kws
scripts/get_kws_model.sh      # wake-word model + tokenized keywords into models/kws
scripts/build_qwentts.sh      # only for S2S_TTS=qwen3: Pascal build of the Qwen3-TTS engine
HF_HUB_OFFLINE=0 scripts/run_s2s.sh   # once, to download Parakeet/Kokoro; Ctrl-C when "Uvicorn running"
```
