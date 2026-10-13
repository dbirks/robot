#!/bin/sh
# The kiosk X session (issue #37). One window, fullscreen, no chrome, no blanking.
# Started from the tty1 CONSOLE LOGIN by the hook in ~/.bash_profile - not by
# reachy-kiosk.service, which cannot work: a systemd user service is never the owner of
# the foreground VT, so Xorg.wrap refuses it (measured 2026-10-13, crash loop). Run it by
# hand on another VT if you want to see it; on VT1 it owns the demo screen.

DASH="${REACHY_DASHBOARD_URL:-http://127.0.0.1:3001/conversation}"
PROFILE="${REACHY_KIOSK_PROFILE:-$HOME/.reachy-kiosk}"

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
  --disable-session-crashed-bubble \
  --disable-infobars \
  --noerrdialogs \
  --disable-component-update \
  --overscroll-history-navigation=0 \
  --start-fullscreen \
  --class=ReachyKiosk
