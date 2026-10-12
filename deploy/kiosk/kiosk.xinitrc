#!/bin/sh
# The kiosk X session (issue #37). One window, fullscreen, no chrome, no blanking.
# Started only by reachy-kiosk.service; never run this by hand on VT1 unless you
# mean to fight the demo stack for the console.

DASH="${REACHY_DASHBOARD_URL:-http://127.0.0.1:3001/conversation}"
PROFILE="${REACHY_KIOSK_PROFILE:-$HOME/.reachy-kiosk}"

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
