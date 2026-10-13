# HDMI kiosk (issue #37)

Plug a screen into the robot box → the dashboard appears, fullscreen, nobody has
to type anything. Attendees can click around, refresh, read the resolved system
prompt and switch personas (#38). Deliberately **not** locked down: we stand
next to it.

**Status: rebuilt as a system unit, 2026-10-13 evening.** The constraint that decided the
shape: **there is no keyboard attached to this machine.** Any design whose diagnosis path is
"press Ctrl+Alt+F1 and read the tty" is unusable, so the kiosk has to be startable,
restartable and observable over ssh + the journal, and nothing may die silently on the
console.

What the day taught us, replacing the guesses below:

- **The hotplug watcher's sensor works.** It saw `DP-2 connected` and fired within seconds. (The screen
  was on the GTX 1070's DisplayPort; the motherboard's ports are dead on this box because the
  iGPU is disabled, and every connector read `disconnected` until the cable moved to the GPU.)
- **The kiosk cannot be a systemd USER unit, and that is structural.** `reachy-kiosk.service`
  crash-looped on `/usr/lib/Xorg.wrap: Only console users are allowed to run the X server`: a user
  service is never the owner of the foreground VT, so rootless X refuses it. No env in the unit fixes
  that. The watcher was therefore disabled (good sensor, impossible action) - pointing it at
  `systemctl start reachy-kiosk` in system scope is the change that would make it useful again.
- **`reachy-kiosk.service` is now a SYSTEM unit** (`User=david`, `PAMName=login`,
  `TTYPath=/dev/tty1`, `StandardInput=tty`, `UtmpMode=user`). `PAMName=login` is not decoration:
  /etc/pam.d/login → system-local-login → system-login line 20 is `pam_systemd.so`, so the service
  gets a real logind session on seat0 - which is also the only thing that puts an ACL on
  `/dev/dri/card1` (measured: `root:video 0660`, `other::---`, while david's ssh session holds no
  seat). Installing it masks `getty@tty1`, because two readers cannot own one tty.
- **Autologin was never the way; the drop-in is demoted to a fallback.** `10-autologin.conf` did
  load (correct bare `ExecStart=` reset, `/sbin/agetty` is a real 77 kB binary), but tty1 kept
  running `/bin/login -p --` - no user, no `-f` - and re-spawned on agetty's own 60 s login
  timeout (`NRestarts=47`). agetty(8): giving `--login-options` replaces the `-f username` that
  the autologin flag would have added, and our copied Arch fragment had `\u` without `-f`. Fixed in
  the file, but `scripts/install_kiosk.sh` no longer installs it: with `getty@tty1` masked it can
  never fire, and a hook that needs a keyboard is not a diagnosis path here.
- **X, not Wayland.** cage has a documented "displays nothing" failure on a GTX 1070 under Arch
  (cage-kiosk #441); this box is a 1070 on driver 580.142 with no keyboard to debug that with.
  nvidia_drv.so and modesetting_drv.so are both installed already.

## Install (needs sudo — David's call)

```bash
scripts/install_kiosk.sh          # prints the plan, changes nothing
scripts/install_kiosk.sh --run    # pacman + SYSTEM unit + mask getty@tty1 + enable
```

Packages: `xorg-server xorg-xinit xorg-xset openbox chromium xterm`. No `-Syu`:
a coordinated full upgrade is a decision for after the demo, not a dependency of
a kiosk on the eve of one.

The installer enables the unit but does **not** start it: with no keyboard, start it by hand
with the journal open in a second terminal (`journalctl -fu reachy-kiosk.service &` then
`sudo systemctl start reachy-kiosk.service`), so a black screen is readable as it happens.

## What it does

| file | role |
|---|---|
| `reachy-kiosk.service` | **SYSTEM** unit: `User=david` + `PAMName=login` + `TTYPath=/dev/tty1` → a real console session that owns the VT, runs `startx kiosk.xinitrc -- :0 vt1 -keeptty`, logs to the journal, `Restart=on-failure` so a browser crash is a 3-second flicker, not a dead panel |
| `kiosk.xinitrc` | waits (bounded, 30 s) for the dashboard, disables screensaver + DPMS, starts `openbox`, then Chromium `--kiosk http://127.0.0.1:3001/conversation --disable-gpu` |
| `10-autologin.conf` | **fallback only**, not installed: autologin for tty1 if you unmask getty and want the console-login path. Needs the marked `~/.bash_profile` hook, and a keyboard |
| `watch_hotplug.sh` (`reachy-kiosk-watch.service`) | user unit, still disabled: starts the kiosk when a connector reads `connected`. Its action has to become `systemctl start reachy-kiosk` (system scope) before it can work |

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

`Ctrl+Alt+F2` still switches VTs and `xterm` is installed, but there is **no keyboard on this
machine**, so the escape hatch that matters is over ssh: `systemctl restart reachy-kiosk.service`
with `journalctl -fu reachy-kiosk.service` open.

## Verify over ssh (no keyboard is attached to this box)

1. `systemd-analyze verify /etc/systemd/system/reachy-kiosk.service` → no `Unknown key` line.
2. `systemctl show reachy-kiosk -p ExecMainPID -p NRestarts -p Result -p StatusText`, then
   `journalctl -b -u reachy-kiosk | tail -40` - X's own output is in the journal because
   `StandardOutput=journal`; the X log is also in `~/.local/share/xorg/`.
3. `loginctl` must now show a session for david with `SEAT=seat0` and `TTY=tty1`, and
   `getfacl -p /dev/dri/card1` must show an ACL entry for david. Those two are the proof the
   PAM session worked; without them Xorg.wrap refuses the process again.
4. `pgrep -af 'Xorg|chromium'` → an X on `:0 vt1` and one chromium with `--kiosk`.
5. `nvidia-smi` → the panel must not have moved GPU memory, because Chromium is not asking for GL.
6. Talk to the robot while the panel is up — the kiosk must not steal the audio or the VT from
   the voice stack. It does not own a sound device and it is `Nice=10` (ADR 0006).
7. The page survives a browser kill: `pkill -f ReachyKiosk` → the unit restarts within ~3 s.

## Disable / uninstall

```bash
sudo systemctl disable --now reachy-kiosk.service
sudo systemctl unmask getty@tty1.service && sudo systemctl start getty@tty1.service   # login prompt back
rm -f ~/.config/systemd/user/reachy-kiosk.service ~/.config/systemd/user/reachy-kiosk-watch.service
systemctl --user daemon-reload
sudo rm /etc/systemd/system/reachy-kiosk.service
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
