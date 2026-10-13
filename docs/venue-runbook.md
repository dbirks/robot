# Venue runbook — symptom → command

One screen for someone who has never opened this repo. Everything runs on the box
under him: a 4-core i7, an 8 GB GTX 1070, no cloud. The dashboard is
`http://127.0.0.1:3001`. The three numbers worth remembering: wake word **Reachy**,
dashboard port **3001**, chat model Qwen 3.5 4B.

## The one command, for anything

```bash
scripts/booth_recovery.sh --dry-run    # read the PLAN lines first, nothing is touched
scripts/booth_recovery.sh              # then run it
```

It diagnoses, then restarts **only what is broken**, in dependency order
(reachy-mini-daemon → llama-server → reachy-llm-proxy → reachy-s2s → reachy-shell →
reachy-dashboard), and finishes with `scripts/preflight_demo.sh` plus the last two
minutes of the journal. `--only <unit>`, `--all` (restart the chain even where it looks
healthy), `--yes` (no prompts), `--no-preflight` (skip the closing pre-flight, so it never
grabs a camera frame).
It never starts the retired `reachy-agent`.

**Do not** `systemctl --user start reachy-agent` or `qwen3-tts`. Both are retired;
`reachy-agent` runs the legacy `python -m app`, a second owner of the same mic and
speaker, and while it is running he hears the room through the wrong device and his own
voice stops being cancelled (ADR 0002). `qwen3-tts` loads a second model onto a card
that is already at 7.84 of 8 GB.

## Symptoms

| Symptom | What it is, and the command |
|---|---|
| **He does not answer** | Say **"hey Reachy"** first, then the question in the same breath. Room chatter never reaches the model — that is a rule, not a bug (ADR 0003: ambient speech is never transcribed into the conversation, so it also cannot trigger a tool). Still nothing: `scripts/booth_recovery.sh --only reachy-shell` (the wake word lives in that process). Check the mute first: `wpctl get-volume @DEFAULT_SOURCE@` — `[MUTED]` there is not a crashed service. |
| **Too quiet / too loud** | Say **"louder"** (+0.1) or **"much louder"** (+0.25), or "set the volume to 150%". Ceiling is **200%**; past ~120% the desk speaker measures as distortion (`shell/volume.py:22-26`). The dashboard slider at `/settings` enforces the same ceiling. By hand: `wpctl set-volume @DEFAULT_SINK@ 1.20` (read it with `wpctl get-volume @DEFAULT_SINK@`). |
| **He will not stop talking** | Say **"stop talking"** — or "be quiet", "shut up", "that's enough". Shipped in 091dcd6: that phrase is routed around the language model, so he goes silent instead of answering the request (before it, "stop talking" was being heard as "turn it down"). It has to be short, an order, and said **while he is talking**; "was it quiet in Rome?" is still a question, and he will answer it. |
| **The screen shows nothing** | `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:3001/conversation` — 200 means the page is fine and the monitor/browser is the problem (the HDMI kiosk, issue #37, is staged but **not installed** on this box; point any browser at that URL). No answer: `systemctl --user restart reachy-dashboard` — the dashboard is decoration, restarting it never takes the voice down. |
| **The camera / "what do you see" is slow** | Normal, and not yours to speed up. The four camera tools get a **35 s** budget (`shell/tools.py:36`); a cold camera open is **2.7-3.0 s** and a vision round trip ~10 s (6852e43). **Do not ask twice**: a call that timed out is still holding the camera, and a second one queues behind it — three retries turned a 3-second job into a 54-second one. Ask once, wait, talk to the guests. |
| **Nobody else can open the dashboard** | It binds loopback by default, on purpose: `/` and `/faces` serve *home* data (family names, old transcripts). Opening it to the room is a deliberate act: `systemctl --user set-environment DASHBOARD_HOST=0.0.0.0 && systemctl --user restart reachy-dashboard` — and back to safe before he goes home: `systemctl --user unset-environment DASHBOARD_HOST && systemctl --user restart reachy-dashboard`. |
| **He stopped after the robot was powered off and on again** | The daemon never reattaches; `GET :8000/api/media/status` keeps answering `available:false` and the greeter and camera go quietly inert. The log signature is `journalctl --user -u reachy-mini-daemon --since "-1 hour" \| grep -i "motor communication error"`. Fix is a restart, and only a restart — `scripts/booth_recovery.sh` does the daemon first for exactly this reason, because the ALSA/USB node keeps enumerating while the robot is off and every check below it will happily pass on a dead speaker. After it restarts the daemon it tests the camera the only way that means anything: ask `shell.camera.daemon_holds_camera()` and then **grab a real frame** — `available:false` in `/api/media/status` is not evidence either way (it says that on a healthy robot right now). |
| **Everything says active and he is still wrong** | `systemctl --user --failed`, then `scripts/preflight_demo.sh` (audio route, resolved room, safety gate, model files on disk, egress, disk, booth-memory freshness). Then David. One blind spot in that first command: `reachy-audio-init.service` is a **system** unit, so `systemctl --user --failed` cannot see it — and it is enabled and failing on this box (`amixer -c 3`: that card is gone). `systemctl status reachy-audio-init` (no `--user`) is the half of the answer the user manager does not have. |

## Where the logs are

```bash
journalctl --user -u reachy-shell --since "-10 min" --no-pager     # same for reachy-mini-daemon,
                                    # llama-server, reachy-llm-proxy, reachy-s2s, reachy-dashboard
scripts/tail_journal.py --minutes 10          # the conversation as a timeline, barge-ins counted
scripts/tail_journal.py --minutes 30 --grep volume
sqlite3 -readonly data/events.db "select datetime(wall,'unixepoch','localtime'), type, payload
                                    from events order by id desc limit 20;"
```

Two traps in that last one (docs/booth-card.md): on this box `datetime(wall)` returns
**nothing** and `datetime(wall,'localtime')` returns **1999-12-31** — the stored epoch is
fine, only the formatter lies. The working form is `datetime(wall,'unixepoch','localtime')`.
And `mono` is seconds since *that process* started, so it resets on every restart: never
subtract two `mono` values from either side of one.

**Zero rows is not a failure.** In a room where nobody has spoken for two minutes there is
nothing to journal, and `tail_journal.py` exits 1 for it. The proof he is still listening is
the `audio.health` row the shell writes every 10 s — `blocks_total` climbing and
`muted: false` means the mic is open and he is simply not being addressed.

## Ports, if you are reading a log that names one

| port | what | unit |
|---|---|---|
| 8000 | robot daemon (camera, media status) | reachy-mini-daemon |
| 8080 | chat LLM, llama.cpp (`--jinja` or tool calling silently fails) | llama-server |
| 8081 | name-fixing proxy — the speech service's only path to the LLM | reachy-llm-proxy |
| 8765 | realtime speech websocket | reachy-s2s |
| 3001 | dashboard | reachy-dashboard |

Restarting the daemon also restarts reachy-shell (`BindsTo=` in
`deploy/systemd/reachy-shell.service:5` — `systemctl --user show reachy-mini-daemon
-p BoundBy` names the shell), so he is deaf for a few
seconds: say something to the room before you do it. Restarting `reachy-s2s` costs 20-46 s
of dead air while the voice reloads. Restarting `llama-server` reloads 4.47 GB onto a card
with ~360 MiB free — if it fails to come up, `systemctl --user stop reachy-s2s` first.

There is no `DASHBOARD_PORT`: 3001 is hardcoded (`dashboard/server.py:968`). Only the
address is a knob (`DASHBOARD_HOST`, `dashboard/server.py:967`). Every name you are allowed
to set — and what each room resolves it to — is in `docs/knobs.md`; a variable that is not
in that table does nothing when you export it.
