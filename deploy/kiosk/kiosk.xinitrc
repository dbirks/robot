#!/bin/sh
# The kiosk X session (issue #37). One window, fullscreen, no chrome, no blanking.
# Started by deploy/kiosk/reachy-kiosk.service - a SYSTEM unit whose PAMName=login +
# TTYPath=/dev/tty1 pair makes it the console owner, which is what Xorg.wrap requires. It
# can never be started by a systemd USER unit: a user service is never the owner of the
# foreground VT, so Xorg.wrap refuses it (measured 2026-10-13, crash loop). The marked hook
# in ~/.bash_profile runs the same command from a tty1 console login as the fallback path.
# There is no keyboard on this box: stdout/stderr of the unit go to the journal, so keep any
# diagnostic here on stdout, not in a file nobody will open at a booth.

DASH="${REACHY_DASHBOARD_URL:-http://127.0.0.1:3001/conversation}"
PROFILE="${REACHY_KIOSK_PROFILE:-$HOME/.reachy-kiosk}"

# Device pixel ratio for the panel. The booth display is a Samsung LS27A800U - 27" at
# 3840x2160, which X correctly negotiates (log: "Virtual screen size determined to be
# 3840 x 2160") - but a dashboard authored in CSS pixels then paints 1:1 on a 163 PPI panel
# and every word is a thumbnail. Measured on the live screen: 8.3M pixels, 1.59% dark text,
# i.e. a postage stamp in the middle of a 4K monitor. 2 makes the layout canvas a logical
# 1920x1080 and rasterises text at 2x, which is big enough to read from across a table and
# still crisp, because a device-scale-factor is a DPR - not a zoomed screenshot.
# Set REACHY_KIOSK_SCALE=2.5 if the room is loud enough that nobody gets within a metre.
SCALE="${REACHY_KIOSK_SCALE:-2}"

# Wait for the dashboard before navigating. Everything here starts at boot as a peer user
# unit with no ordering against this one, and chromium --kiosk has no retry: win the race
# by a second and the panel shows "This site can't be reached" for the rest of the night
# with nothing to fix it but a keyboard nobody has at a booth. Bounded, and it launches
# anyway - a browser error page tells the next person more than a black screen does.
# Ctrl+R reloads it if the dashboard comes up later.
if command -v curl >/dev/null 2>&1; then
  _i=0
  while [ "$_i" -lt 30 ]; do
    curl -fsS -o /dev/null --max-time 2 "$DASH" 2>/dev/null && break
    _i=$((_i + 1)); sleep 1
  done
  [ "$_i" -ge 30 ] && echo "kiosk: dashboard did not answer $DASH in 30 s; opening anyway" >&2
fi

# A black panel mid-demo is worse than an ugly one.
xset s off >/dev/null 2>&1 || true
xset -dpms >/dev/null 2>&1 || true
xsetroot -solid '#101014' >/dev/null 2>&1 || true

# openbox gives Chromium a real fullscreen parent and sane focus behaviour;
# without a WM, Chromium under bare X tends to sit at a weird size after a
# mode change (which is exactly what plugging in an HDMI cable does).
openbox &
OPENBOX_PID=$!

cleanup() { kill "$OPENBOX_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

# --disable-gpu is load-bearing: 7.8 of 8 GB of VRAM belongs to inference.
# --user-data-dir keeps a throwaway profile so a crash bubble or a stray
# "restore pages" dialog cannot gate the demo, without going --incognito
# (which would drop any state we deliberately leave on the page).
exec chromium \
  --kiosk "$DASH" \
  --user-data-dir="$PROFILE" \
  --no-first-run \
  --disable-gpu \
  --disable-software-rasterizer \
  --disable-dev-shm-usage \
  --force-device-scale-factor="$SCALE" \
  --disable-session-crashed-bubble \
  --disable-infobars \
  --noerrdialogs \
  --disable-component-update \
  --overscroll-history-navigation=0 \
  --start-fullscreen \
  --class=ReachyKiosk
