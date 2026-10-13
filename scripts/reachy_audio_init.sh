#!/usr/bin/env bash
# Set the Reachy Mini's hardware playback gains (PCM,0 / PCM,1) by CARD NAME.
#
# Why this exists: the unit this replaces ran `amixer -c 3 set PCM,0 60` and failed at
# boot on 2026-10-12 with `Invalid card number '3'` - the USB sound card had not
# enumerated yet, and the index it got later was not necessarily 3 (the same card was
# card 1 earlier the same day and moved to 3 across a power cycle). A failed hardware
# init is invisible where anyone would look: it is a SYSTEM unit, so
# `systemctl --user --failed` on the operator's machine shows nothing at all.
#
# Contract:
#   - resolves the card by the name in /proc/asound/cards, never a number;
#   - waits a few seconds for USB, because this runs at boot;
#   - exits 0 when no Reachy card is present at all: a desk without the robot attached
#     is a normal state, not a failed unit that haunts `systemctl --failed`.
set -uo pipefail

WANT_SECONDS="${REACHY_AUDIO_WAIT_S:-20}"
VALUE="${REACHY_AUDIO_PCM:-60}"   # the card's own maximum (Limits: Playback 0 - 60)

find_card() {
  # /proc/asound/cards lines look like:
  #  3 [Audio          ]: USB-Audio - Reachy Mini Audio
  awk '/Reachy Mini Audio/ {print $1; exit}' /proc/asound/cards 2>/dev/null
}

card=""
deadline=$((SECONDS + WANT_SECONDS))
while [ "$SECONDS" -lt "$deadline" ]; do
  card="$(find_card)"
  [ -n "$card" ] && break
  sleep 1
done

if [ -z "$card" ]; then
  echo "reachy-audio-init: no Reachy Mini Audio card present; nothing to do"
  exit 0
fi

if ! command -v amixer >/dev/null 2>&1; then
  echo "reachy-audio-init: amixer not installed" >&2
  exit 1
fi

rc=0
for idx in 0 1; do
  if out=$(amixer -c "$card" set "PCM,$idx" "$VALUE" 2>&1); then
    echo "reachy-audio-init: card $card PCM,$idx -> $VALUE"
  else
    # A control missing on a firmware/driver variant is not a boot failure; say which
    # one and keep going so one absent control does not hide the other.
    echo "reachy-audio-init: PCM,$idx on card $card refused: $(head -1 <<<"$out")" >&2
    rc=1
  fi
done

# Report where they ended up, so `journalctl -u reachy-audio-init -b` answers "did the
# gains get set?" without anyone standing at the machine with an amixer prompt.
amixer -c "$card" get PCM,0 2>/dev/null | grep -E "Front Left|Mono:" | sed 's/^/reachy-audio-init:   /'
exit $rc
