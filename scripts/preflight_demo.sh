#!/usr/bin/env bash
# Pre-flight for the demo (issue #34, C1). Read-only: it starts nothing, stops
# nothing, opens no audio device and writes nothing except its own summary.
#
#   scripts/preflight_demo.sh            # check everything, exit nonzero on FAIL
#   DEMO_ENV=hackday scripts/preflight_demo.sh
#
# Ordered by how much it would hurt tomorrow: audio route first (no reference
# means no echo cancellation and #40's whole plan changes), then the resolved
# room, then the things that fail silently (stale .env overrides, missing model
# files, no egress for web_search, a shared booth memory file).
set -uo pipefail
cd "$(dirname "$0")/.."

PASS=0; WARN=0; FAIL=0
ok()   { printf '  PASS  %s\n' "$*"; PASS=$((PASS+1)); }
warn() { printf '  WARN  %s\n' "$*"; WARN=$((WARN+1)); }
bad()  { printf '  FAIL  %s\n' "$*"; FAIL=$((FAIL+1)); }
head_() { printf '\n%s\n' "$*"; }

DEMO_ENV="${DEMO_ENV:-}"
PY="uv run python"
[ -x .venv/bin/python ] && PY=".venv/bin/python"

head_ "audio route (ADR 0005: no reference, no cancellation)"
SINK=$(wpctl status 2>/dev/null | sed -n '/Sinks/,/Sources/p' | grep -m1 -F '* ' || true)
SRC=$(wpctl status 2>/dev/null | sed -n '/Sources/,/Filters/p' | grep -m1 -F '* ' || true)
echo "        sink:   $(tr -s ' ' <<<"$SINK")"
echo "        source: $(tr -s ' ' <<<"$SRC")"
case "$SINK$SRC" in
  *"Reachy Mini Audio"*"Reachy Mini Audio"*) ok "default sink and source are both the Reachy card" ;;
  *"Reachy Mini Audio"*) bad "only one of sink/source is the Reachy card - the other owns the demo" ;;
  *) bad "neither default device is the Reachy card: playback would not reach the chip's AEC reference" ;;
esac
grep -qE '^REACHY_(MIC|SPEAKER)=.+' .env 2>/dev/null \
  && echo "        note: .env pins REACHY_MIC/REACHY_SPEAKER - those override the system default" \
  || ok "no device pin in .env (uses system default, which is what the AEC reference needs)"

head_ "the robot answers (an ALSA node is not a speaker)"
# The card keeps enumerating over USB while the robot is powered OFF, so the
# device check above can pass on a dead speaker. This is the check that catches
# it - found the hard way the night before the demo, when the route section
# reported PASS on a machine with no robot switched on.
MEDIA=$(curl -s -m 2 http://localhost:8000/api/media/status 2>/dev/null || true)
if [ -z "$MEDIA" ]; then
  bad "daemon /api/media/status unreachable: the greeter and camera tools cannot work"
else
  echo "        $MEDIA"
  # Ask the camera owner its own question. Two string-matching guesses at this were
  # both wrong - available:false does NOT mean no camera, daemon_holds_camera() reads
  # it as free - so the check calls the same function the shell calls, then PROVES
  # the answer by grabbing a real frame.
  CAM=$($PY -c "
from shell.camera import CameraGrabber, daemon_holds_camera, find_reachy_camera
st = CameraGrabber('http://localhost:8000')._media_status()
print('held' if daemon_holds_camera(st) else 'free', find_reachy_camera())" 2>/dev/null || echo "unknown -1")
  set -- $CAM
  CAMSTATE="$1"
  CAMNODE="$2"
  if [ "$CAMSTATE" = held ]; then
    warn "the daemon holds the camera: the shell will not open it (no greeter, no describe_scene). At the booth that is the eye-contact feature missing."
  elif [ "$CAMSTATE" = unknown ]; then
    bad "could not ask shell.camera (import failed) - run: $PY -c 'import shell.camera'"
  elif [ "$CAMNODE" = "-1" ]; then
    bad "no Reachy camera node under /sys/class/video4linux: is the robot powered on and enumerated?"
  else
    FRAME=$($PY -c "
from shell.camera import CameraGrabber
try:
    f = CameraGrabber('http://localhost:8000').grab()
    print('ok', getattr(f, 'shape', None))
except Exception as e:
    print('fail', e)" 2>/dev/null || echo "fail import")
    if [ "${FRAME%% *}" = ok ]; then
      ok "camera usable by the shell - grabbed a real frame${FRAME#ok}"
    else
      bad "camera node exists but the shell's own grab failed: ${FRAME#fail }"
    fi
  fi
fi
# The daemon only logs motor errors while it is polling, so an empty recent
# window proves nothing: report the LAST occurrence instead, with a judgement.
LASTMOTOR=$(journalctl --user -u reachy-mini-daemon --since "-6 hours" --no-pager -o short-iso 2>/dev/null | grep -iE "motor communication error|no response from the robot" | tail -1 | cut -d' ' -f1-2)
if [ -n "$LASTMOTOR" ]; then
  RECENT=$(journalctl --user -u reachy-mini-daemon --since "-30 min" --no-pager -o cat 2>/dev/null | grep -icE "motor communication error|no response from the robot" || true)
  if [ "${RECENT:-0}" -gt 0 ]; then
    bad "daemon says the robot is not answering (last at $LASTMOTOR) - power it on, then restart reachy-mini-daemon"
  else
    warn "daemon reported the robot unresponsive at $LASTMOTOR. If you have powered it on since, the daemon does NOT reattach by itself: systemctl --user restart reachy-mini-daemon"
  fi
else
  ok "no motor communication errors in the last 6 hours"
fi

head_ "the room this box will boot into"
RESOLVE=$($PY -m shell --print-env 2>&1) || true
if [ -z "$RESOLVE" ]; then bad "python -m shell --print-env failed (fix before anything else)"; else
  get() { printf '%s' "$RESOLVE" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d.get('$1'))" 2>/dev/null; }
  ENV_NOW=$(get env)
  echo "        REACHY_ENV -> ${ENV_NOW:-?}"
  if [ -n "$DEMO_ENV" ] && [ "$ENV_NOW" != "$DEMO_ENV" ]; then
    bad "resolved $ENV_NOW but the demo needs $DEMO_ENV (add REACHY_ENV=$DEMO_ENV to .env)"
  elif [ "$ENV_NOW" = home ]; then warn "resolving 'home' - at the office you want 'hackday'"; else ok "resolved $ENV_NOW"; fi
  for k in attention_profile memories_path learn_facts greeter_enabled auto_sleep_s tools_exclude; do
    echo "        $k = $(get "$k")"
  done
  case "$(get instructions)" in *"President Barack Obama"*) [ "$ENV_NOW" = hackday ] && bad "hackday is serving the home Obama persona" || ok "Obama persona present (home)";; *) ok "persona carries no named public figure";; esac
  grep -qE '^[[:space:]]*REACHY_INSTRUCTIONS=' .env && [ "$ENV_NOW" = hackday ] \
    && bad ".env still sets REACHY_INSTRUCTIONS: it outranks envs/hackday.yaml (comment it out)" \
    || ok ".env does not override the persona"
fi

head_ "safety gate and suite"
if bash scripts/check_env_safety.sh >/tmp/preflight_gate.log 2>&1; then ok "check_env_safety.sh (faces hidden, namespaces separated)"; else bad "check_env_safety.sh FAILED - see /tmp/preflight_gate.log"; fi
if timeout 300 uv run pytest tests/ -q >/tmp/preflight_tests.log 2>&1; then ok "pytest: $(tail -1 /tmp/preflight_tests.log)"; else bad "pytest failing: $(tail -1 /tmp/preflight_tests.log)"; fi

head_ "models and files on disk (no downloads at the venue)"
for p in $(grep -oE '^(S2S_QWEN3_TALKER_GGUF|LLM_MODEL_PATH)=.*' .env 2>/dev/null | cut -d= -f2-); do
  [ -e "$p" ] && ok "model file present: $(basename "$p")" || bad "model file MISSING: $p"
done
ls models/gguf/mtp/*.gguf >/dev/null 2>&1 && ok "chat LLM GGUFs present" || warn "no models/gguf/mtp/*.gguf found"
ls models/kws >/dev/null 2>&1 && ok "wake-word model dir present" || bad "models/kws missing: he cannot hear his name"
grep -q 'HF_HUB_OFFLINE=1' deploy/systemd/reachy-s2s.service && ok "s2s unit runs offline (no surprise downloads at the venue)" || warn "s2s unit does not set HF_HUB_OFFLINE=1"

head_ "egress for the one tool that leaves the box"
if curl -s -m 6 -o /dev/null "https://duckduckgo.com"; then ok "network reachable: web_search can answer current questions"; else warn "no egress: web_search will fail quietly at the booth (everything else still works)"; fi

head_ "services, robot, space"
for u in reachy-mini-daemon llama-server reachy-llm-proxy reachy-s2s reachy-shell; do
  printf '  %-22s %s\n' "$u" "$(systemctl --user is-active "$u" 2>/dev/null)"
done
[ -e /dev/ttyACM0 ] && ok "/dev/ttyACM0 present (robot attached)" || bad "/dev/ttyACM0 missing: the daemon has no robot to talk to"
AVAIL=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[ "${AVAIL:-0}" -ge 5 ] && ok "disk: ${AVAIL}G free" || bad "disk: ${AVAIL}G free"
DBSZ=$(du -sm data/events.db 2>/dev/null | cut -f1)
echo "        journal: ${DBSZ:-0} MB (attendee transcripts share this file with home history - env-stamped, Phase C decision)"
if [ -f data/memories.hackday.json ]; then
  AGE=$(( ( $(date +%s) - $(stat -c %Y data/memories.hackday.json) ) / 3600 ))
  [ "$AGE" -le 1 ] && ok "booth memory store is fresh (${AGE}h)" || warn "data/memories.hackday.json is ${AGE}h old: rm it before the doors (facts are shared between guests)"
fi
command -v chromium >/dev/null 2>&1 && ok "chromium present (kiosk #37 installable)" || warn "chromium absent: kiosk not installed yet (#37)"

head_ "summary"
printf '  %d pass, %d warn, %d fail\n\n' "$PASS" "$WARN" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
