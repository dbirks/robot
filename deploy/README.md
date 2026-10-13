# Deploy

Production process layout (systemd user services, per the EPIC). Units live
here so they are versioned; install with:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-server reachy-llm-proxy reachy-s2s reachy-shell
```

**Four units this box runs are not in `deploy/systemd/`** and cannot be installed from a
checkout: `reachy-mini-daemon` and `reachy-dashboard` live only in
`~/.config/systemd/user/`, and the retired `reachy-agent` / `qwen3-tts` are still there.
So the `enable --now` line above is not the whole stack on a fresh machine, and
`reachy-shell` will not come up at all without the daemon unit — it declares
`BindsTo=reachy-mini-daemon.service`. `docs/knobs.md` §3 is the unit table for this box.

Services, in startup order:

| Unit | Runs | Notes |
|---|---|---|
| `reachy-mini-daemon` | robot hardware | unchanged; units live outside repo until Phase 2 |
| `llama-server` | chat LLM on :8080 (Qwen3.5-4B + mmproj) | `scripts/run_llama_server.sh`; see issue #33 for pinning |
| `reachy-vision` | optional camera VLM on :8082, CPU (Qwen3.5-0.8B), only with a text-only chat LLM | `scripts/run_vision_server.sh`; set `VISION_BASE_URL=http://127.0.0.1:8082/v1` in `.env` |
| `reachy-llm-proxy` | `python -m shell.llm_proxy` on :8081 | rewrites STT mis-hearings of "Reachy" before llama.cpp |
| `reachy-s2s` | pinned HF speech-to-speech realtime | `scripts/install_s2s.sh` once, then this |
| `reachy-shell` | audio owners + attention + tools (`python -m shell`) | the new stack |
| `reachy-dashboard` | FastAPI dashboard on :3001 (port is hardcoded in `dashboard/server.py:968`; there is no `DASHBOARD_PORT`) | unit lives outside the repo until Phase 2 |

Phase 2 (issue #22) will fold the legacy `reachy-agent` / `qwen3-tts` units
away and teach the dashboard about these names.

**Do not start `reachy-agent` or `qwen3-tts` to "see what they do".** `reachy-agent` runs
the legacy `python -m app`, which is a second owner of the same microphone and speaker
(ADR 0002): while it is up, the live shell hears the room through the wrong device and
his own voice stops being cancelled — and `reachy-shell` still reads `active` in every
status line. `qwen3-tts` loads a second model onto a card already at 7.84 of 8 GB.
`reachy-shell` also has `BindsTo=reachy-mini-daemon.service`: **restarting the daemon
stops the shell and does not start it again**, so he is deaf until you say
`systemctl --user start reachy-shell`. Both facts are in `docs/knobs.md` §3.

One unit that `systemctl --user --failed` cannot see: `reachy-audio-init.service` is a
**system** unit (`/etc/systemd/system/`), enabled and failing on this box — its
`amixer -c 3` card no longer exists. Check it with `systemctl status reachy-audio-init`.

## Powering the robot back on (travel day)

The daemon does not reliably reattach to a robot that was powered off while it was
running: it keeps serving `/api/media/status` with `available:false` and the greeter
and camera tools go quietly inert. After any power cycle, at the venue, in this order:

```bash
systemctl --user restart reachy-mini-daemon   # re-enumerate the robot, camera, audio
scripts/preflight_demo.sh                     # must be clean before you trust anything
systemctl --user start reachy-shell           # NOT optional: BindsTo= took it down
```

The third line is the whole point of the order. `reachy-shell.service` binds to the
daemon, so the restart in line 1 stopped the shell; without line 3 the box looks healthy
(`reachy-mini-daemon` active, `reachy-s2s` active) and answers nothing.

`scripts/preflight_demo.sh` checks for the stale-daemon state and tells you so; do not
skip it because the audio devices "look present" - the ALSA card enumerates over USB
even with the robot switched off.

## Before the doors open

```bash
scripts/preflight_demo.sh                     # audio route, the robot answering, resolved
                                              # room, safety gate, suite, model files on
                                              # disk, egress for web_search, services,
                                              # disk, booth memory freshness.
                                              # Exit nonzero = do not start.
DEMO_ENV=hackday scripts/preflight_demo.sh    # also fails if the room did not take
bash scripts/check_env_safety.sh             # the 13 numbered safety checks on their own
```

The 13 checks belong to `scripts/check_env_safety.sh`, not to the pre-flight: the
pre-flight runs that gate as **one** of its own steps (and the hardware-free suite as
another), so a pre-flight PASS line is one line for thirteen checks. Read the gate's own
output when you need to know which one moved.

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

**[`docs/knobs.md`](../docs/knobs.md) is the index**: every `REACHY_*` / `S2S_*` /
`DASHBOARD_*` / `LLAMA_*` name that is actually read, the file:line that reads it, and
what each room resolves. A name that is not in that table does nothing when you set it.

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

## Personas (voice + manner, one knob)

A persona is one thing: how Reachy **sounds** (the TTS voice the pinned realtime service
loads) and how he **behaves** (the spoken manner in the system prompt). Both halves live
in `personas/<name>.yaml` — `instructions`, `ack_dir`, and a `voice:` block (`engine`,
`model`, `talker_gguf`, `codec_gguf`, `speaker`, `instruct`, `speed`). A room names one:
`persona: obama`, or `persona: {name: obama, instructions: ...}` to keep its own words.

Tonight's pair: `obama` (the desk — the Obama 1.7B talker GGUF, speaker `laxmikant`, the
ack pool in `sounds/acks/obama`) and `marvin` (the booth's dry, weary gentleman —
**behaviour-only**: the voice block is still the Obama talker, because no Marvin model
exists). Switching to a real second voice means editing `voice:` *and* adding a
measured `docs/pins.yaml` entry first.

Apply it, then restart (the s2s service reads its `--qwen3_tts_*` flags once at startup,
so a voice change is not a reload — measured 20-46 s of dead air):

```bash
uv run python scripts/apply_env.py --check                # what the room's persona renders
uv run python scripts/apply_env.py --persona marvin        # render only; says what to restart
uv run python scripts/apply_env.py --persona marvin --restart
systemctl --user restart reachy-s2s && systemctl --user restart reachy-shell   # by hand
```

That writes `deploy/generated/persona.env` (gitignored, atomic, previous render kept as
`persona.env.prev`), which both `reachy-s2s.service` and `reachy-shell.service` load
**after** `.env`. Measured on this box 2026-10-12 with a throwaway user unit (two files,
overlapping keys, plus an inline `Environment=`): systemd applies `EnvironmentFile=`
lines in listed order with the **last one winning per key**, and every
`EnvironmentFile=` beats `Environment=`; a key only the earlier file sets survives, and
an *empty* value still counts as set. So the persona outranks a stray `S2S_QWEN3_*` line
in `.env` instead of losing to it, and switching engine clears the old talker path.

Three consequences to know before the demo:

- **`persona.env` never sets `REACHY_INSTRUCTIONS`.** It is one global file read in both
  rooms, and a stamped prompt would put the home persona into a public booth the moment
  someone forgot to move the room back (hard rule 1). The room keeps its words.
- **An inline `persona.instructions` in `envs/<env>.yaml` outranks the persona file.**
  Both shipped rooms still carry one (it is the A2 parity bar: `home` resolves byte-identical
  to what it resolved before the picker existed — pinned by `tests/test_env.py`, not by a byte
  count quoted in a README, because the prompt moves and the number goes stale the same day), so
  tonight **applying a persona changes the
  voice and the acknowledgements, not the words**. `apply_env.py` and the dashboard both
  say this out loud. To hand the words to the persona file, delete the `instructions:`
  block from that room's env file — `personas/obama.yaml` and `personas/marvin.yaml`
  already hold byte-identical text.
- **Pin guard.** If a persona names a talker GGUF or model id that `docs/pins.yaml`
  does not mention, `apply_env.py` WARNs (and refuses with `--strict`). The Obama
  1.7B talker *is* currently in that position — its VRAM cost is only quoted inside the
  `models.llm` entry — so expect the warning until someone gives it its own pins block.

The dashboard's `/persona` page is the same script: one Apply at a time (a lock file
refuses a second concurrent apply), a detached `apply_env.py --restart` child, and an
in-progress panel while the units settle. The room itself is **not** a control there —
`REACHY_ENV` stays `.env`-only, and the page shows it as a readout.

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
