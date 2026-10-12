#!/usr/bin/env bash
# Poll the DRM connector status and keep the kiosk in step with reality
# (issue #37).
#
# Why a poll and not a systemd .path unit: PathChanged= rides on inotify, and
# sysfs does not implement it, so a .path unit on /sys/class/drm/card1-*/status
# silently never fires. A udev rule would work and costs more moving parts to
# explain at 6 pm before a demo. One stat per connector every 2 s is free
# (ADR 0006: this is not a thread worth worrying about).
set -uo pipefail

SERVICE="${REACHY_KIOSK_SERVICE:-reachy-kiosk.service}"
CONNECTORS_GLOB="${REACHY_KIOSK_CONNECTORS:-/sys/class/drm/card*-*/status}"
INTERVAL_S="${REACHY_KIOSK_INTERVAL_S:-2}"

connected() {
  local f
  for f in $CONNECTORS_GLOB; do
    [ -r "$f" ] || continue
    case "$f" in */*-*) ;; *) continue ;; esac      # skip the bare card dirs
    [ "$(cat "$f" 2>/dev/null)" = "connected" ] && return 0
  done
  return 1
}

active() { systemctl --user is-active --quiet "$SERVICE"; }

was=""
while true; do
  if connected; then want=yes; else want=no; fi
  if [ "$want" = yes ] && ! active; then
    echo "screen attached -> starting $SERVICE"
    systemctl --user start "$SERVICE" || true
  elif [ "$want" = no ] && active; then
    echo "screen removed -> stopping $SERVICE"
    systemctl --user stop "$SERVICE" || true
  fi
  [ "$want" != "$was" ] && was="$want"
  sleep "$INTERVAL_S"
done
