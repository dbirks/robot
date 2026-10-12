#!/usr/bin/env bash
# Staged installer for the HDMI kiosk (issue #37). Reviewable on purpose:
# nothing is installed unless you pass --run, because a pacman transaction on
# the robot box the night before a demo is David's call, not a script's.
#
#   scripts/install_kiosk.sh          # print exactly what it would do
#   scripts/install_kiosk.sh --run    # install + wire up
#
# Design notes and the manual fallbacks: deploy/kiosk/README.md
set -euo pipefail

# The whole point of keeping the list here: one place to eyeball before sudo.
# xorg-server/xorg-xinit give us X on VT1; openbox is the window manager that
# gives Chromium a sane fullscreen (and a right-click menu if we ever want
# one); chromium is the kiosk itself; xterm is the escape hatch on tty1.
PKGS=(xorg-server xorg-xinit xorg-xset openbox chromium xterm)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_UID="$(id -u)"
GIT_DIR=/etc/systemd/system/getty@tty1.service.d
REPO_DIR="$HOME/.config/systemd/user"
DRY=1
[ "${1:-}" = "--run" ] && DRY=0

say() { printf '%s\n' "$*"; }
run() { if [ "$DRY" = 1 ]; then say "  would run: $*"; else say "  running: $*"; "$@"; fi; }

say "kiosk install plan (issue #37)"
say "  packages: ${PKGS[*]}"
say "  repo:     $REPO"
say "  drop-in:  $GIT_DIR/10-autologin.conf   (sudo)"
say "  units:    $REPO_DIR/reachy-kiosk.service + reachy-kiosk-watch.service"
say "  target:   http://127.0.0.1:3001/conversation"
say ""

if [ "$DRY" = 1 ]; then
  say "dry run; nothing changed. Re-run with --run."
  exit 0
fi

# 1. X + browser. No -Syu: partial upgrades are Arch's own sin to warn about,
#    and this box is mid-demo-prep. --needed keeps existing packages untouched.
run sudo pacman -S --noconfirm --needed "${PKGS[@]}"

# 2. Autologin on tty1 so X has a session with seat access (this is also what
#    gives the kiosk /dev/dri and the evdev devices without any setuid mess).
run sudo mkdir -p "$GIT_DIR"
run sudo cp "$REPO/deploy/kiosk/10-autologin.conf" "$GIT_DIR/10-autologin.conf"
run sudo systemctl daemon-reload

# 3. The kiosk units (user scope: they die with the user, never with the demo stack).
run mkdir -p "$REPO_DIR"
for u in reachy-kiosk.service reachy-kiosk-watch.service; do
  run ln -sfn "$REPO/deploy/kiosk/$u" "$REPO_DIR/$u"
done
run chmod +x "$REPO/deploy/kiosk/kiosk.xinitrc" "$REPO/deploy/kiosk/watch_hotplug.sh"
run systemctl --user daemon-reload
# The watcher is the only piece enabled at install time: it starts the kiosk
# when a screen appears and leaves it off when one does not.
run systemctl --user enable --now reachy-kiosk-watch.service

say ""
say "done. Verify with a real screen: plug HDMI into card1-HDMI-A-1 and wait <=10 s."
say "Escape hatch: Ctrl+Alt+F2 -> tty2. Kill the kiosk: systemctl --user stop reachy-kiosk"
