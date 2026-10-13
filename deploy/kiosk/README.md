# HDMI kiosk (issue #37)

Plug a screen into the robot box → the dashboard appears, fullscreen, nobody has
to type anything. Attendees can click around, refresh, read the resolved system
prompt and switch personas (#38). Deliberately **not** locked down: we stand
next to it.

**Status: installed and partially working, 2026-10-13.** What the night taught us, replacing the
guesses below:

- **The hotplug watcher works.** It saw `DP-2 connected` and fired within seconds. (The screen
  was on the GTX 1070's DisplayPort; the motherboard's ports are dead on this box because the
  iGPU is disabled, and every connector read `disconnected` until the cable moved to the GPU.)
- **The user service cannot show anything, and that is structural.** `reachy-kiosk.service`
  crash-looped on `/usr/lib/Xorg.wrap: Only console users are allowed to run the X server`: a
  systemd **user** service is not the owner of the foreground VT, so rootless X refuses it. No
  amount of env in the unit fixes that. The watcher was therefore disabled (it is still a good
  sensor with an impossible action).
- **The kiosk starts from the console session instead.** `deploy/kiosk/10-autologin.conf` + a
  marked hook in `~/.bash_profile` that runs `startx ... :0 vt1 -keeptty` when `tty` is
  `/dev/tty1`. Ctrl+Alt+F2 is the escape hatch; delete the hook to stop it.
- **The proper fix, still open:** a *system* unit with `User=david`, `PAMName=login`,
  `TTYPath=/dev/tty1` (and getty masked on that VT), which gives the service a real session and
  seat - that is the shape that would let the watcher's detection drive a real start again.

## Install (needs sudo — David's call)

```bash
scripts/install_kiosk.sh          # prints the plan, changes nothing
scripts/install_kiosk.sh --run    # pacman + autologin drop-in + user units
```

Packages: `xorg-server xorg-xinit xorg-xset openbox chromium xterm`. No `-Syu`:
a coordinated full upgrade is a decision for after the demo, not a dependency of
a kiosk on the eve of one.

## What it does

| file | role |
|---|---|
| `10-autologin.conf` | drop-in for `getty@tty1.service` → david is auto-logged-in on VT1, which is what gives X a seat and `/dev/dri` without a display manager |
| `reachy-kiosk.service` | `xinit kiosk.xinitrc -- :0 vt1 -keeptty` — one X, one Chromium; `Restart=on-failure` so a browser crash is a 3-second flicker, not a dead panel |
| `kiosk.xinitrc` | disables screensaver + DPMS, starts `openbox`, then Chromium `--kiosk http://127.0.0.1:3001/conversation --disable-gpu` |
| `watch_hotplug.sh` (`reachy-kiosk-watch.service`) | starts the kiosk when any connector reads `connected`, stops it when they are all `disconnected` |

Two choices worth defending:

- **`--disable-gpu` is load-bearing.** The GTX 1070 sits at ~7.8 of 8 GB with
  llama.cpp + Qwen3-TTS (`docs/pins.yaml`). A compositor asking the driver for
  GL context is how a demo dies. The dashboard is a table of text; software
  rendering is fine.
- **A 2 s poll instead of a systemd `.path` unit.** `PathChanged=` rides on
  inotify and **sysfs does not implement it**, so a `.path` unit pointed at
  `/sys/class/drm/card1-*/status` looks correct and never fires. A udev rule
  (`SUBSYSTEM=="drm", ACTION=="change"`) is the other right answer and buys us
  another file to debug at 17:40 on demo day. One `cat` per connector per 2 s is
  nothing next to the real threads (ADR 0006).

Escape hatch for us, not for the crowd: `Ctrl+Alt+F2` → a TTY (`xterm` is
installed precisely so a terminal exists inside the session too).

## Verify on a bench (10 minutes, needs a monitor)

1. `systemctl --user start reachy-kiosk` while the screen is already plugged in → dashboard within ~15 s.
2. Unplug → `journalctl --user -u reachy-kiosk-watch -n 20` shows *screen removed → stopping*, and the service stops within ~2 s.
3. Plug back in while llama/s2s/shell are running → *starting*, and **GPU memory did not move** (`nvidia-smi`), because Chromium is not asking for GL.
4. Talk to the robot while the panel is up — the kiosk must not steal the audio or the VT from the voice stack.
5. Walk away for 10 minutes → the panel is still on (`xset -dpms`), not a black rectangle.

## Disable / uninstall

```bash
systemctl --user disable --now reachy-kiosk-watch reachy-kiosk   # tonight
sudo rm /etc/systemd/system/getty@tty1.service.d/10-autologin.conf && sudo systemctl daemon-reload
sudo pacman -Rns chromium openbox xorg-xinit xorg-server        # only if you want it gone
```

## Plan B if this fights us

A laptop or an Android tablet on the LAN pointed at `http://<box>:3001` needs no
install at all, works today, and is more robust than a first-ever X session on a
headless inference box. Cost: one device and one HDMI-or-no-cable. If the kiosk
is still misbehaving on the morning of the demo, take the tablet and stop
sacrificing the evening.

(LAN access is not the default: `dashboard/server.py` binds `127.0.0.1` because the
page serves `data/known_faces.json` and `data/sessions.db` with no auth, so a tablet
needs `DASHBOARD_HOST=0.0.0.0` on the dashboard and a decision about what is in those
files first — `deploy/README.md`, "The dashboard is loopback by default". `DASHBOARD_HOST`
is the only dashboard address knob, and the port is **not** one: 3001 is hardcoded at
`dashboard/server.py:968`. The kiosk's own four variables
(`REACHY_DASHBOARD_URL`, `REACHY_KIOSK_PROFILE` in `kiosk.xinitrc`, `REACHY_KIOSK_SERVICE`,
`REACHY_KIOSK_CONNECTORS`, `REACHY_KIOSK_INTERVAL_S` in `watch_hotplug.sh`) are read by
those scripts and by nothing else — `docs/knobs.md` §2.)

## Not here

Persona switching in the UI and the system-prompt / tool-list views are #38, and
they sit on top of the `envs/` + `personas/` layer from #34.
