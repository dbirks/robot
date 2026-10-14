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
SYS_DIR=/etc/systemd/system
KIOSK_UNIT=reachy-kiosk
# Fallback path only: a console login on tty1 whose ~/.bash_profile hook runs startx by
# hand. NOT installed here - see the note at the bottom of this script and README.
GETTY_DIR=/etc/systemd/system/getty@tty1.service.d
REPO_DIR="$HOME/.config/systemd/user"
DRY=1
[ "${1:-}" = "--run" ] && DRY=0

say() { printf '%s\n' "$*"; }
run() { if [ "$DRY" = 1 ]; then say "  would run: $*"; else say "  running: $*"; "$@"; fi; }

say "kiosk install plan (issue #37)"
say "  packages: ${PKGS[*]}"
say "  repo:     $REPO"
say "  unit:     $SYS_DIR/$KIOSK_UNIT.service   (sudo, SYSTEM scope)"
say "  getty:    sudo systemctl mask getty@tty1.service   (this unit owns the VT)"
say "  cleanup:  unlink $REPO_DIR/$KIOSK_UNIT.service - a user-scope kiosk cannot work"
say "  target:   http://127.0.0.1:3001/conversation"
say ""

# Stage BEFORE the dry-run branch. Two lessons from 2026-10-13.
# (1) The dry-run branch reads $KIOSK_STAGED. Under `set -u`, defining it only in the --run path
#     made the DOCUMENTED review-first invocation - this script with NO arguments, which the
#     header and deploy/kiosk/README.md both tell an operator to run first - exit 1 with
#     "unbound variable". `bash -n` passes: it is a parse-clean RUNTIME failure on the one path
#     whose whole purpose is to be safe to run. The check that catches it is executing the dry
#     path, now asserted in tests/test_kiosk_boot.py.
# (2) Staging early lets the dry run verify the exact bytes it would install, not another file.
#     A temp under $TMPDIR touches nothing privileged, so DRY=1 stays safe.
# The name must END in .service: `systemd-analyze verify` rejects any filename that does
# not look like a unit ("Failed to prepare filename ...: Invalid argument"), and the
# obvious `reachy-kiosk.service.staged.$$` does not.
KIOSK_STAGED="${TMPDIR:-/tmp}/reachy-kiosk-staged-$$.service"
XWRAP_TMP="${TMPDIR:-/tmp}/Xwrapper.config.$$"
trap 'rm -f "$KIOSK_STAGED" "$XWRAP_TMP"' EXIT
sed "s#/home/david/dev/robot#$REPO#g" "$REPO/deploy/kiosk/$KIOSK_UNIT.service" > "$KIOSK_STAGED"

if [ "$DRY" = 1 ]; then
  KIOSK_VERIFY="$KIOSK_STAGED"   # in a dry run nothing reached /etc; verify what we would install
  say ""
  say "  staged unit check (read-only, no sudo):"
  # Explicit rc, not `|| true`: under pipefail a failing verify in a pipeline would kill the dry run
  # for the wrong reason (plumbing), and swallowing it would hide the right one (a broken unit). A
  # non-zero exit here means "the unit you are about to install is defective" - deliberately, and
  # labelled, which is the opposite of the accidental exit 1 this file shipped for an hour.
  if VERIFY_OUT=$(systemd-analyze verify "$KIOSK_STAGED" 2>&1); then
    say "    verify: clean (it still cannot prove the ExecStart path exists - read the line above)"
  else
    printf '%s\n' "$VERIFY_OUT" | sed 's/^/    verify: /'
    VERIFY_FAIL=1
  fi
  if [ "${VERIFY_FAIL:-0}" = 1 ]; then
    say "FAIL: the staged unit does not verify. Do not --run it; fix the unit first."
    exit 1
  fi
  say "dry run; nothing changed. Re-run with --run."
  exit 0
fi

# 1. X + browser. No -Syu: partial upgrades are Arch's own sin to warn about,
#    and this box is mid-demo-prep. --needed keeps existing packages untouched.
run sudo pacman -S --noconfirm --needed "${PKGS[@]}"

# 2. The kiosk as a SYSTEM unit. This is the only shape that can show anything: the unit
#    pairs PAMName=login with TTYPath=/dev/tty1, which gives the process a real logind
#    session on seat0 (pam_systemd is line 20 of /etc/pam.d/system-login) and makes it the
#    foreground-VT owner - the exact thing Xorg.wrap refused when this ran as a user unit.
run sudo mkdir -p "$SYS_DIR"
# The unit's ExecStart has to carry THIS user's repo path. The file in git hardcodes
# /home/david/dev/robot on purpose, because `%h` in a SYSTEM unit expands to the home of whoever
# runs the system manager - /root - and does NOT follow User=david. That one mismatch is what
# left the first booth night staring at a black screen: `systemctl show -p ExecStart` reported
# startx being handed /root/dev/robot/deploy/kiosk/kiosk.xinitrc, which does not exist, X never
# spawned, and `systemd-analyze verify` stayed silent because it validates syntax, not paths.
# Stage a substituted copy so a different $HOME can never ship a /root path again.
# Rootless X is refused by Xorg.wrap unless the caller owns the foreground VT - and on this
# machine that check can NEVER pass, no matter how the unit is written: pam_unix reports
# "session opened for user david", yet `loginctl` shows no tty1 session and /dev/dri/card1
# gains no user ACL, because pam_systemd does not register a seat session for a system
# service here. So the kiosk needs allowed_users=anybody, which runs X as root with chromium
# as david - the old root-X/user-clients split that xinit's -auth cookie handles already.
# Without this line the kiosk is architecturally incapable of drawing on this box.
printf 'allowed_users=anybody\n' > "$XWRAP_TMP"
run sudo cp "$XWRAP_TMP" /etc/X11/Xwrapper.config
run sudo cp "$KIOSK_STAGED" "$SYS_DIR/$KIOSK_UNIT.service"
# `verify` is read-only and needs no root, and it is the only check that catches a typo in a
# directive name before the unit is enabled - the previous version of this file shipped
# `StopPropagationOfFailedJobsFromThisUnitFile=`, which systemd does not know at all.
if [ "$DRY" = 1 ]; then
  say "  would run: systemd-analyze verify $KIOSK_STAGED   (read-only, no sudo)"
else
  say "  running:   systemd-analyze verify $SYS_DIR/$KIOSK_UNIT.service   (read-only, no sudo)"
  systemd-analyze verify "${KIOSK_VERIFY:-$SYS_DIR/$KIOSK_UNIT.service}" 2>&1 | sed 's/^/    verify: /' || true
fi
run sudo systemctl mask getty@tty1.service
run sudo systemctl daemon-reload
run sudo systemctl enable "$KIOSK_UNIT.service"

# 3. Un-link the user-scope copy. It is not just useless ("Only console users are allowed
#    to run the X server", measured), it is a trap: a linked unit with the same name in the
#    user directory reads as "the kiosk is installed" while the screen stays black.
run rm -f "$REPO_DIR/$KIOSK_UNIT.service"

# 4. The hotplug watcher stays a user unit and stays DISABLED. Its sensor is good - it saw
#    DP-2 connect and fired within seconds - but its only action was to start a service that
#    cannot open the VT. Pointing it at `systemctl start reachy-kiosk` (system scope) is the
#    change that would make it usable, and that is not tonight's decision.
run mkdir -p "$REPO_DIR"
run ln -sfn "$REPO/deploy/kiosk/reachy-kiosk-watch.service" "$REPO_DIR/reachy-kiosk-watch.service"
run chmod +x "$REPO/deploy/kiosk/kiosk.xinitrc" "$REPO/deploy/kiosk/watch_hotplug.sh"
run systemctl --user daemon-reload

say ""
say "NOT started from here, on purpose: start it yourself with the journal open, because"
say "  there is no keyboard on this box and a black screen has to be readable as it happens:"
say "    journalctl -fu $KIOSK_UNIT.service &      # other terminal"
say "    sudo systemctl start $KIOSK_UNIT.service"
say "Enable already done ($KIOSK_UNIT.service is WantedBy=multi-user.target), so a reboot"
say "  brings the panel up on its own. getty@tty1 is masked: no login prompt on VT1."
say ""
say "Fallback, only if this unit is disabled and getty@tty1 is unmasked: the autologin"
say "  drop-in in deploy/kiosk/10-autologin.conf + the marked hook in ~/.bash_profile"
say "  (scripts/install_kiosk used to install the drop-in; it no longer does, because with"
say "  getty@tty1 masked it can never fire)."
say "If X itself fails: journalctl -b -u $KIOSK_UNIT | tail -40 and ~/.local/share/xorg/*.log"
say "Undo: sudo systemctl disable --now $KIOSK_UNIT.service; sudo systemctl unmask getty@tty1.service"
