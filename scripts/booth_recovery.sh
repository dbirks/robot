#!/usr/bin/env bash
# One command that un-breaks him, at the booth, in front of people.
#
#   scripts/booth_recovery.sh                     # diagnose, then restart what is broken
#   scripts/booth_recovery.sh --dry-run           # print the plan, touch nothing
#   scripts/booth_recovery.sh --only reachy-shell # one unit (repeatable, or comma-separated)
#   scripts/booth_recovery.sh --all               # restart the chain even where it looks healthy
#   scripts/booth_recovery.sh --yes               # no prompts (non-interactive / in a hurry)
#   scripts/booth_recovery.sh --no-preflight      # skip the closing pre-flight (suite + camera grab)
#
# Dependency order, because every fault above the line looks like a fault below it:
#
#   reachy-mini-daemon -> llama-server -> reachy-llm-proxy -> reachy-s2s -> reachy-shell -> reachy-dashboard
#
# The rules baked in here are all things that went wrong tonight:
#
# 1. Restart the daemon BEFORE testing the camera. The ALSA/USB node keeps enumerating
#    while the robot is powered OFF, and GET /api/media/status answers "available":false
#    even when the camera is fine (it answers exactly that right now, on a live robot).
#    The only honest camera test is the one scripts/preflight_demo.sh uses: ask
#    shell.camera.daemon_holds_camera(), then grab a real frame.
# 2. NEVER start the retired reachy-agent unit: it runs `python -m app`, a second owner
#    of the mic and speaker the live shell already owns (ADR 0002). This script only ever
#    stops it, if it is found running; it never starts it. qwen3-tts is retired the same
#    way (a torch model onto a card already holding 7.84 of 8 GB - docs/pins.yaml, ADR 0004).
# 3. A stale daemon after a power cycle shows motor communication errors and does not
#    reattach by itself. The line to grep is "motor communication error"; a restart fixes it.
# 4. Restarting a live unit is visible and rude - and restarting the daemon restarts
#    reachy-shell with it (BindsTo=reachy-mini-daemon.service, measured on systemd 260).
#    Every action is printed before it happens; "is he mid-reply?" is answered from the
#    journal (a reply window that opened and never closed), and pauses for confirmation
#    (Ctrl-C waits for silence). --dry-run answers "what would it do" with no side effects
#    at all, and --only <unit> keeps everything else off the table.
#
# Read-only on data/events.db (mode=ro), opens no audio device, moves no head.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CHAIN=(reachy-mini-daemon llama-server reachy-llm-proxy reachy-s2s reachy-shell reachy-dashboard)
DAEMON_URL="${REACHY_DAEMON_URL:-http://localhost:8000}"
LLM_HEALTH="${LLM_HEALTH_URL:-http://127.0.0.1:8080/health}"
PROXY_URL="${LLM_PROXY_URL:-http://127.0.0.1:8081/v1/models}"
S2S_PORT="${S2S_PORT:-8765}"
DASH_URL="${DASHBOARD_URL:-http://127.0.0.1:3001/conversation}"
DASH_PORT="${DASHBOARD_PORT:-3001}"
DB="${EVENTS_DB:-data/events.db}" # overridable so the journal-only logic is testable against a copy
PREFLIGHT_LOG="${TMPDIR:-/tmp}/booth_recovery_preflight.log"

DRY=0 FORCE=0 YES=0 SKIP_PREFLIGHT=0 ONLY=""
usage() { # print this file's own header block
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
}
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run | -n) DRY=1 ;;
    --all) FORCE=1 ;;
    --yes | -y) YES=1 ;;
    --no-preflight) SKIP_PREFLIGHT=1 ;;
    --only)
      shift
      [ $# -gt 0 ] || {
        printf -- '--only needs a unit name\n' >&2
        exit 2
      }
      ONLY="${ONLY:+$ONLY,}$1"
      ;;
    --only=*) ONLY="${ONLY:+$ONLY,}${1#*=}" ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done
if [ -n "$ONLY" ]; then
  IFS=',' read -r -a WANT <<<"$ONLY"
  for w in "${WANT[@]}"; do
    case " ${CHAIN[*]} " in
      *" $w "*) ;;
      *)
        printf 'not a unit in the chain: %s\n  chain: %s\n' "$w" "${CHAIN[*]}" >&2
        exit 2
        ;;
    esac
  done
fi

PASS=0
WARN=0
FAIL=0
RESTARTED=()
ok() {
  printf '  ok    %s\n' "$*"
  PASS=$((PASS + 1))
}
warn() {
  printf '  WARN  %s\n' "$*"
  WARN=$((WARN + 1))
}
bad() {
  printf '  FAIL  %s\n' "$*"
  FAIL=$((FAIL + 1))
}
say() { printf '%s\n' "$*"; }
head_() { printf '\n%s\n' "$*"; }
plan() { printf '  PLAN  %s\n' "$*"; }

PY="${PYTHON:-}"
[ -n "$PY" ] || PY=".venv/bin/python"
[ -x "$PY" ] || PY="uv run python"

unit_state() { systemctl --user is-active "$1" 2>/dev/null || true; }
journal_rows() { # <type> [limit] -> "<unix epoch> <payload>" per row, newest first.
  # cast(wall as integer) + `-separator ' '` so the caller can split on one space:
  # sqlite's default separator is '|' and the payload is JSON full of those.
  # `wall` is a unix epoch, safe to subtract from date(1); `mono` is per-process and
  # resets on every restart, which is why nothing here reads it (ADR 0007).
  sqlite3 -readonly -separator ' ' "$DB" "select cast(wall as integer), payload from events
    where type='$1' order by id desc limit ${2:-1};" 2>/dev/null || true
}
http_code() { curl -s -m "${2:-3}" -o /dev/null -w '%{http_code}' "$1" 2>/dev/null || true; }
http_body() { curl -s -m "${2:-3}" "$1" 2>/dev/null || true; }
port_listening() { ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"; }
selected() {
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in
    *",${1},"*) return 0 ;;
  esac
  return 1
}
wait_for() { # <seconds> <cmd...>
  local deadline=$((SECONDS + $1))
  shift
  while ! "$@" >/dev/null 2>&1; do
    [ "$SECONDS" -lt "$deadline" ] || return 1
    sleep 2
  done
}
restart_unit() { # <unit> <why>
  local unit=$1 why=$2
  if ! polite_to_restart "$unit"; then
    warn "left $unit running: the operator chose not to interrupt a live reply (retry with --yes, or after the silence)"
    return 0
  fi
  plan "systemctl --user restart $unit"
  say "        why: $why"
  if [ "$DRY" = 1 ]; then
    say "        (dry run: not restarted)"
    return 0
  fi
  if systemctl --user restart "$unit" 2>/dev/null; then
    RESTARTED+=("$unit")
    ok "restarted $unit"
  else
    bad "systemctl --user restart $unit failed (unit now $(unit_state "$unit"))"
    return 1
  fi
}
polite_to_restart() { # 0 = restarting this unit now is not an interruption someone can see
  local unit=$1 rc=0
  # Something already dead cannot be made more dead by restarting it.
  [ "$(unit_state "$unit")" = active ] || return 0
  live_reply || return 0
  if [ "$DRY" = 1 ]; then
    say "        (he is mid-reply right now: a real run would pause here for confirmation)"
    return 0
  fi
  if [ "$YES" = 1 ] || [ ! -t 0 ]; then
    say "        (he is mid-reply right now: restarting him without asking)"
    return 0
  fi
  printf '        %s is live and he is mid-reply. [Enter] restarts him now, Ctrl-C waits for silence... ' "$unit"
  read -r _ || rc=1
  printf '\n'
  return "$rc"
}
live_reply() { # 0 = a reply window opened recently and never closed
  local t
  t=$(sqlite3 -readonly "$DB" "select type from events where type in
        ('response.created','response.first_audio','response.completed','response.cancelled')
        and wall > $(( $(date +%s) - ${1:-45} )) order by id desc limit 1;" 2>/dev/null || true)
  case "$t" in
    response.created | response.first_audio) return 0 ;;
  esac
  return 1
}

camera_truth() { # the two steps that are the only honest camera test
  local st
  st=$($PY -c "
from shell.camera import CameraGrabber, daemon_holds_camera, find_reachy_camera
s = CameraGrabber('$DAEMON_URL')._media_status()
print('held' if daemon_holds_camera(s) else 'free', find_reachy_camera())" 2>/dev/null || echo "unknown -1")
  set -- $st
  if [ "$1" = held ]; then
    warn "the daemon holds the camera: no greeter, no describe_scene (available:false in /api/media/status does NOT mean no camera)"
  elif [ "$1" = unknown ]; then
    bad "could not import shell.camera: run $PY -c 'import shell.camera'"
  elif [ "$2" = "-1" ]; then
    bad "no Reachy camera node under /sys/class/video4linux: power the robot on, then restart reachy-mini-daemon"
  else
    local frame
    frame=$($PY -c "
from shell.camera import CameraGrabber
try:
    f = CameraGrabber('$DAEMON_URL').grab()
    print('ok', getattr(f, 'shape', None))
except Exception as e:
    print('fail', e)" 2>/dev/null || echo "fail import")
    if [ "${frame%% *}" = ok ]; then
      ok "camera real: grabbed a frame${frame#ok} (cold open is 2.7-3.0 s, so the first look is slow on purpose)"
    else
      bad "camera node exists but the shell's own grab failed: ${frame#fail } - systemctl --user restart reachy-mini-daemon"
    fi
  fi
}

stage_daemon() {
  selected reachy-mini-daemon || return 0
  head_ "reachy-mini-daemon — robot, camera, audio hardware (:8000)"
  local state media motors recent reason
  state=$(unit_state reachy-mini-daemon)
  media=$(http_body "$DAEMON_URL/api/media/status")
  motors=0
  recent=0
  if command -v journalctl >/dev/null 2>&1; then
    motors=$(journalctl --user -u reachy-mini-daemon --since "-30 min" --no-pager -o cat 2>/dev/null |
      grep -icE "motor communication error" || true)
    # The trigger is recency, not history. Restarting the daemon also restarts
    # reachy-shell (BindsTo=, fact 4 above), so a 20-minute-old burst that already
    # cleared must not take a working conversation down - tonight's journal is exactly
    # that case: 10 errors ending 23:29:58, daemon healthy and answering by 23:52.
    recent=$(journalctl --user -u reachy-mini-daemon --since "-3 min" --no-pager -o cat 2>/dev/null |
      grep -icE "motor communication error" || true)
  fi
  say "        unit: ${state:-unknown}, /api/media/status: ${media:-no answer}, 'motor communication error': ${recent:-0} in the last 3 min (${motors:-0} in 30 min)"
  [ -e /dev/ttyACM0 ] && ok "/dev/ttyACM0 present" || bad "/dev/ttyACM0 missing: the daemon has no robot to talk to"
  reason=""
  if [ "$state" != active ]; then
    reason="unit is ${state:-inactive}; nothing below it can work"
  elif [ -z "$media" ]; then
    reason="active but answering nothing on $DAEMON_URL"
  elif [ "${recent:-0}" -gt 0 ]; then
    reason="$recent 'motor communication error' line(s) in the last 3 min: a power-cycled robot is never reattached, only restarted"
  else
    if [ "${motors:-0}" -gt 0 ]; then
      warn "${motors} motor error(s) in the last 30 min but none in 3: that burst already cleared, and restarting the daemon would take reachy-shell with it (BindsTo) - leaving it alone"
    fi
    ok "daemon healthy right now (motor errors are what to look for after any power cycle)"
  fi
  if [ -n "$reason" ]; then
    # Measured on this box (systemd 260, throwaway unit pair): restarting a unit also
    # restarts the units that BindsTo= it, and they come back by themselves. So a daemon
    # restart IS an interruption of the conversation, and the shell only needs a nudge
    # if it did not come back on its own.
    warn "this also restarts reachy-shell (BindsTo=reachy-mini-daemon.service, deploy/systemd/reachy-shell.service): he is deaf until it reopens the mic"
    restart_unit reachy-mini-daemon "$reason"
    if [ "$DRY" != 1 ]; then
      wait_for 40 sh -c "curl -s -m 2 '$DAEMON_URL/api/media/status' | grep -q ."
      if [ "$(unit_state reachy-shell)" != active ]; then
        plan "systemctl --user start reachy-shell"
        if systemctl --user start reachy-shell 2>/dev/null; then
          ok "reachy-shell started (nothing else owns the mic without it)"
        else
          bad "reachy-shell did not start: journalctl --user -u reachy-shell -n 50"
        fi
      fi
    fi
  fi
  if [ "$DRY" = 1 ]; then
    say "        (dry run: skipping the camera test - it grabs a real frame)"
  else
    camera_truth
  fi
}

stage_llm() {
  selected llama-server || return 0
  head_ "llama-server + reachy-llm-proxy — the chat model (:8080) behind the name-fixing proxy (:8081)"
  local lc pc
  lc=$(http_code "$LLM_HEALTH" 5)
  pc=$(http_code "$PROXY_URL" 5)
  say "        llama-server: $(unit_state llama-server) http $lc, reachy-llm-proxy: $(unit_state reachy-llm-proxy) http $pc"
  if [ "$FORCE" = 1 ] || [ "$(unit_state llama-server)" != active ] || [ "$lc" != 200 ]; then
    restart_unit llama-server "health is '$(unit_state llama-server)'/http $lc (he hears you and never answers when the model is down)"
    if [ "$DRY" != 1 ] && ! wait_for 180 sh -c "[ \"\$(curl -s -m 3 -o /dev/null -w '%{http_code}' '$LLM_HEALTH')\" = 200 ]"; then
      bad "llama-server did not answer $LLM_HEALTH in 180 s. The card is at 7.84/8 GB beside the talker (docs/pins.yaml): a failed load is usually VRAM. Free it with
        systemctl --user stop reachy-s2s   # then restart llama-server, then start reachy-s2s again"
    elif [ "$DRY" != 1 ]; then
      ok "llama-server healthy again"
    fi
  else
    ok "llama-server healthy (do not restart it for a mood: 4.47 GB of GGUF reloads)"
  fi
  if [ "$FORCE" = 1 ] || [ "$(unit_state reachy-llm-proxy)" != active ] || [ "$pc" = 000 ]; then
    restart_unit reachy-llm-proxy "the proxy is the only path from the speech service to the model; down = dead air (:8081 http $pc)"
  elif [ "$pc" != 200 ]; then
    warn "proxy answers but not with 200 (http $pc): it is fine, llama-server behind it is not"
  else
    ok "reachy-llm-proxy healthy"
  fi
}

stage_s2s() {
  selected reachy-s2s || return 0
  head_ "reachy-s2s — the realtime pipeline: VAD, Parakeet, TTS, cancellation (:${S2S_PORT})"
  local state
  state=$(unit_state reachy-s2s)
  if [ "$FORCE" = 1 ] || [ "$state" != active ] || ! port_listening "$S2S_PORT"; then
    restart_unit reachy-s2s "unit ${state:-inactive}, port ${S2S_PORT} $(port_listening "$S2S_PORT" && echo listening || echo not listening)"
    say "        expect 20-46 s of dead air while it reloads the voice (deploy/README.md); do not interrupt it"
  else
    ok "reachy-s2s active and listening on ${S2S_PORT}"
  fi
}

stage_shell() {
  selected reachy-shell || return 0
  head_ "reachy-shell — mic owner, speaker owner, attention, tools, journal"
  local state health newer older age age_txt rows blocks older_blocks pw_mute
  state=$(unit_state reachy-shell)
  health=$(journal_rows audio.health 2)
  newer=$(printf '%s\n' "$health" | head -1)
  older=$(printf '%s\n' "$health" | tail -1)
  age=-1
  age_txt="never"
  rows=$(printf '%s\n' "$health" | grep -c .)
  if [ -n "$health" ]; then
    age=$(( $(date +%s) - ${newer%% *} ))
    age_txt="${age}s"
    blocks=$(printf '%s' "$newer" | grep -oE '"blocks_total": *[0-9]+' | grep -oE '[0-9]+')
    older_blocks=$(printf '%s' "$older" | grep -oE '"blocks_total": *[0-9]+' | grep -oE '[0-9]+')
    pw_mute=$(printf '%s' "$newer" | grep -oE '"muted": *(true|false)' | grep -oE 'true|false')
    say "        last audio.health ${age_txt} ago, blocks_total ${older_blocks:-?} -> ${blocks:-?}, muted=${pw_mute:-?}"
  else
    say "        no audio.health rows in $DB"
  fi
  if [ "$state" != active ]; then
    restart_unit reachy-shell "unit is ${state:-inactive}: nobody owns the mic, so the wake word cannot fire"
    return 0
  fi
  if [ "$age" -ge 0 ] && [ "$age" -le 30 ] && [ "${pw_mute:-}" = true ]; then
    warn "he is MUTED, which is not a restart: the shell's own gate is shut. Unmute from the dashboard mic button, or wpctl set-mute @DEFAULT_SOURCE@ 0"
    return 0
  fi
  if [ "$FORCE" = 1 ]; then
    restart_unit reachy-shell "--all: the wedge that shows up in no status line is a shell that lost its lease"
  elif [ "$age" -lt 0 ] || [ "$age" -gt 30 ]; then
    # The watchdog writes audio.health every 10 s (shell/__main__.py:443-452): silence
    # from it while the unit says "active" is the wedge no status line shows.
    restart_unit reachy-shell "the 10 s watchdog loop last journalled $age_txt while the unit claims to be running"
  elif [ "${rows:-0}" -ge 2 ] && [ -n "${blocks:-}" ] && [ "$blocks" = "${older_blocks:-}" ]; then
    restart_unit reachy-shell "blocks_total frozen at $blocks across two health samples: capture is running in name only (ADR 0005 firmware stall)"
  else
    ok "reachy-shell active, capture loop alive — left alone on purpose (restarting him mid-conversation is visible and rude)"
  fi
}

stage_dashboard() {
  selected reachy-dashboard || return 0
  head_ "reachy-dashboard — the panel people are allowed to click (:${DASH_PORT})"
  local state code bind
  state=$(unit_state reachy-dashboard)
  code=$(http_code "$DASH_URL" 5)
  bind=$(ss -ltnH 2>/dev/null | awk '{print $4}' | grep -E "[:.]${DASH_PORT}$" | head -1)
  say "        unit ${state:-unknown}, GET $DASH_URL -> http $code, bound on ${bind:-nothing}"
  if [ "$FORCE" = 1 ] || [ "$state" != active ] || [ "$code" != 200 ]; then
    restart_unit reachy-dashboard "page is ${state:-inactive}/http $code"
  else
    ok "dashboard serving"
  fi
  case "$bind" in
    "") ;; # nothing listening at all: the FAIL above already says so
    127.0.0.1:* | ::1 | \[::1\]*)
      warn "bound on loopback only: nobody else in the room can open it. Deliberate - / and /faces serve home data. To open it on purpose:
        systemctl --user set-environment DASHBOARD_HOST=0.0.0.0 && systemctl --user restart reachy-dashboard
      and back: systemctl --user unset-environment DASHBOARD_HOST && systemctl --user restart reachy-dashboard"
      ;;
    *) ok "reachable from the room ($bind) - home data is on / and /faces, that was a deliberate act" ;;
  esac
}

retired_guard() {
  local a q
  a=$(unit_state reachy-agent)
  q=$(unit_state qwen3-tts)
  say "        retired units: reachy-agent=${a:-?} qwen3-tts=${q:-?} (never started by this script)"
  if [ "$a" = active ]; then
    bad "reachy-agent is RUNNING: it opens a second mic and speaker owner and steals them from reachy-shell (ADR 0002). Stop it."
    plan "systemctl --user stop reachy-agent"
    if [ "$DRY" != 1 ] && selected reachy-shell; then
      systemctl --user stop reachy-agent 2>/dev/null && RESTARTED+=("reachy-agent (stopped)") && ok "stopped reachy-agent"
    fi
  fi
  if [ "$q" = active ]; then
    warn "qwen3-tts is running: a legacy torch model on a card with ~360 MiB free. Stop it: systemctl --user stop qwen3-tts"
  fi
}

head_ "booth recovery — $(date '+%H:%M:%S'), in dependency order: ${CHAIN[*]}"
[ "$DRY" = 1 ] && say "DRY RUN: nothing is restarted, no frame is grabbed; the journal is read read-only."
[ -n "$ONLY" ] && say "scope: $ONLY"
[ "$FORCE" = 1 ] && say "--all: restarting healthy units too (s2s costs 20-46 s of dead air, llama-server a GGUF reload)"

stage_daemon
stage_llm
stage_s2s
stage_shell
stage_dashboard
head_ "retired units"
retired_guard

if [ "$DRY" = 1 ] || [ -n "$ONLY" ] || [ "$SKIP_PREFLIGHT" = 1 ]; then
  head_ "pre-flight"
  why="skipped"
  [ "$DRY" = 1 ] && why="a dry run neither runs the suite nor grabs a camera frame"
  [ -n "$ONLY" ] && why="scoped to $ONLY"
  [ "$SKIP_PREFLIGHT" = 1 ] && why="asked to skip"
  say "        $why - run it yourself before the doors: scripts/preflight_demo.sh"
else
  head_ "pre-flight (scripts/preflight_demo.sh)"
  say "        running it now: the safety gate, the whole pytest suite, and a real camera frame"
  if timeout 600 scripts/preflight_demo.sh >"$PREFLIGHT_LOG" 2>&1; then
    PREFLIGHT_RC=0
  else
    PREFLIGHT_RC=$?
  fi
  SUMMARY=$(grep -m1 -E '^[[:space:]]*[0-9]+ pass, [0-9]+ warn, [0-9]+ fail$' "$PREFLIGHT_LOG" | tr -s ' ')
  if [ "${PREFLIGHT_RC:-1}" = 0 ]; then
    ok "pre-flight: ${SUMMARY:-no summary line in $PREFLIGHT_LOG}"
  else
    bad "pre-flight FAILED (rc ${PREFLIGHT_RC}): ${SUMMARY:-no summary line in $PREFLIGHT_LOG}"
    grep -E '^  FAIL' "$PREFLIGHT_LOG" | sed 's/^/        /'
  fi
fi

head_ "is the mic producing anything at all (scripts/tail_journal.py --minutes 2)"
if [ -f "$DB" ]; then
  $PY scripts/tail_journal.py --minutes 2 --db "$DB" ||
    say "        ZERO ROWS IS NOT A FAILURE: in a room where nobody has spoken in the last
        two minutes there is nothing to journal. Not proof he is deaf - read the line below."
else
  bad "no journal at $DB: the shell has never written, or the repo moved"
fi
LAST=$(journal_rows audio.health 1)
if [ -n "$LAST" ]; then
  say "        proof of hearing: audio.health $(( $(date +%s) - ${LAST%% *} ))s ago $(printf '%s' "${LAST#* }" | cut -c1-118)"
else
  warn "no audio.health row at all: the shell's capture loop has never reported (restart reachy-shell)"
fi

head_ "summary"
say "        restarted: ${RESTARTED[*]:-nothing}"
printf '  %d ok, %d warn, %d fail\n\n' "$PASS" "$WARN" "$FAIL"
[ "$FAIL" -eq 0 ] || say "        not clean. docs/venue-runbook.md is the symptom -> command page."
[ "$FAIL" -eq 0 ]
