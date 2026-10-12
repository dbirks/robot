#!/usr/bin/env bash
# Environment-safety gate for the deployment environments (docs/environments.md).
#
# Every check below is the repro of a 2026-10-12 review finding that put private home
# data within reach of the hack-day booth. Run it before the robot leaves the house and
# on the morning of a demo, in front of `uv run pytest tests/ -q`:
#
#   bash scripts/check_env_safety.sh          # exit 0 = the room is safe to switch to
#
# It is deliberately honest: revert any one of the fixes and the matching check goes
# red. No dependencies beyond the repo's own venv; it opens no audio device, connects to
# no robot and touches no network (`--print-env` is the documented hardware-free path).
set -uo pipefail
cd "$(dirname "$0")/.." # repo root: envs/, data/, dashboard/ resolve from here

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x .venv/bin/python ]; then
  PY=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  PY="uv run python"
else
  PY="python3"
fi

fails=0
run() { # run <name> <<'PY' ... PY   (the snippet fails = the check fails)
  local name=$1 src out
  src=$(cat)
  if out=$($PY - <<<"$src" 2>&1); then
    printf 'ok   %s\n' "$name"
  else
    printf 'FAIL %s\n%s\n' "$name" "$(printf '%s\n' "$out" | sed 's/^/       /')"
    fails=$((fails + 1))
  fi
}

# Each snippet starts by clearing the overrides the live .env may have set, so the gate
# measures the env files and the code, not whoever's desk it is running on.
PRELUDE='
import os, sys
# Import config FIRST on purpose: shell/config.py runs load_dotenv() at import,
# which repopulates any key our live .env sets (REACHY_INSTRUCTIONS, REACHY_ACK_DIR).
# Popping before that import looked tidy and made check 7 unkillable - the gate was
# measuring the desk it ran on, not the code (2026-10-12 review F-2).
import shell.config  # noqa: F401
for _k in ("REACHY_ENV", "REACHY_INSTRUCTIONS", "REACHY_ACK_DIR", "REACHY_PROFILE",
           "REACHY_TOOLS_INCLUDE", "REACHY_TOOLS_EXCLUDE", "REACHY_MEMORY_NAMESPACE",
           "REACHY_LEARN_FACTS", "REACHY_XMOS_AGC_MAX_GAIN", "REACHY_MOTION_HZ"):
    os.environ.pop(_k, None)
'

run "1. no *_face tool survives the hackday schema" <<PY
$PRELUDE
from shell.env import load_config
from shell.memories import MEMORY_TOOLS
from shell.tools import apply_tool_policy
from app.robot_tools import TOOLS

cfg = load_config("hackday")
replaced = {t["function"]["name"] for t in MEMORY_TOOLS}
schemas = [t for t in TOOLS if t["function"]["name"] not in replaced] + MEMORY_TOOLS
names = {t["function"]["name"] for t in schemas}
assert {"identify_face", "learn_face", "forget_face"} <= names, "face tools vanished upstream"
for excluded in cfg.tools_exclude:
    assert excluded in names, "exclude %s names no real tool (rename? typo?)" % excluded
kept_tools, kept_handlers = apply_tool_policy(
    schemas, {n: (lambda **_k: {}) for n in names}, include=cfg.tools_include, exclude=cfg.tools_exclude
)
kept = {t["function"]["name"] for t in kept_tools}
assert not [n for n in kept if n.endswith("_face")], "face tool in the booth: %s" % sorted(kept)
assert set(kept_handlers) == kept, "a handler survived without its schema"
# faces are one shared file (app/face_tracker.py) until B1 lands; this exclusion is
# the only thing keeping data/known_faces.json out of the room. Do not "simplify" it.
PY

run "2. an empty process env var does not outrank the env file" <<PY
$PRELUDE
os.environ["REACHY_ENV"] = "hackday"
os.environ["REACHY_INSTRUCTIONS"] = ""      # review repro: deletes the attendee-safe persona
os.environ["REACHY_TOOLS_EXCLUDE"] = "  "   # review repro: re-advertises go_to_sleep
os.environ["REACHY_MOTION_HZ"] = ""         # blank used to crash float("")
from shell.env import load_config

cfg = load_config("hackday")
assert "never mention anyone's family" in cfg.instructions, "attendee-safe persona deleted"
assert "go_to_sleep" in cfg.tools_exclude, "go_to_sleep advertised in the booth again"
assert cfg.attention_profile == "event", cfg.attention_profile
PY

run "3. a non-home env may not land on the home memory store" <<PY
$PRELUDE
from shell.env import EnvError, load_config

os.environ["REACHY_ENV"] = "hackday"
os.environ["REACHY_MEMORY_NAMESPACE"] = "home"  # "let me keep my memories at the demo"
try:
    load_config()
except EnvError as e:
    assert "hard rule 1" in str(e), e
else:
    raise AssertionError("REACHY_MEMORY_NAMESPACE=home resolved without an error")
os.environ.pop("REACHY_MEMORY_NAMESPACE")
assert load_config("hackday").memories_path.name == "memories.hackday.json"
PY

run "4. auto fact extraction is off in hackday, on in home" <<PY
$PRELUDE
from dataclasses import replace
from shell.__main__ import fact_mining
from shell.env import load_config
from shell.memories import MemoryStore
import tempfile, pathlib

home = load_config("home")
day = load_config("hackday")
assert home.learn_facts is True, "home stopped learning facts (A2 parity)"
assert day.learn_facts is False, "hackday is mining the room into one shared store"
with tempfile.TemporaryDirectory() as td:
    store = MemoryStore(replace(day, data_dir=pathlib.Path(td)).memories_path)
    assert fact_mining(day, store) == (None, None), "hackday still wired the extractor"
    assert all(c is not None for c in fact_mining(home, store)), "home lost its extractor"
PY

run "5. no real names in the prompts or the tool schemas" <<PY
$PRELUDE
import json
from shell.conversation import FACT_PROMPT, SUMMARY_PROMPT, grounded, parse_facts
from shell.env import load_config
from shell.memories import MEMORY_TOOLS
from app.robot_tools import TOOLS

PRIVATE = ["david", "rachel", "barney", "sarah", "forrest", "timmy", "mccordsville", "46055", "wife"]
sent = {"fact prompt": FACT_PROMPT, "summary prompt": SUMMARY_PROMPT, "schemas": json.dumps(TOOLS + MEMORY_TOOLS)}
for env in ("home", "hackday"):
    sent[env] = load_config(env).instructions
for what, text in sent.items():
    hits = [w for w in PRIVATE if w in text.lower()]
    assert not hits, "%s leaks %s" % (what, hits)
# the replacement example still shows the shape the extractor and the guard expect
example = FACT_PROMPT.split('e.g. "')[1].split('"')[0]
assert parse_facts(json.dumps([example])) == [example], "example no longer parses as a fact"
assert grounded([example], example) == [example], "example no longer survives grounded()"
PY

run "6. the dashboard binds loopback unless told otherwise" <<PY
import re, sys
src = open("dashboard/server.py").read()
assert 'host="0.0.0.0"' not in src, "the dashboard is back on every LAN (no auth in it)"
assert re.search(r'DASHBOARD_HOST",\s*""\)\s*(\.strip\(\)\s*)?or\s*"127\.0\.0\.1"', src), \
    "DASHBOARD_HOST no longer defaults to 127.0.0.1"
assert re.search(r"uvicorn\.run\(\s*app,\s*host=host", src), "uvicorn no longer uses the configured host"
PY

run "7. --print-env output is safe to paste back" <<PY
$PRELUDE
import json, subprocess, tempfile, pathlib, yaml
from shell.env import load_config

out = subprocess.run([sys.executable, "-m", "shell", "--print-env"], capture_output=True, text=True, check=True)
printed = json.loads(out.stdout)["instructions"]
with tempfile.TemporaryDirectory() as td:
    envs = pathlib.Path(td) / "envs"
    envs.mkdir()
    (envs / "lab.yaml").write_text(yaml.safe_dump({"persona": {"instructions": printed}}))
    again = load_config("lab", envs_dir=envs).instructions
assert again == printed, "pasting the printed instructions back changes them"
assert again.count("Speak plain words") == 1, "TOOLS_NOTE appended twice"
assert again.count("Each user message begins") == 1, "TIME_NOTE appended twice"
PY

run "8. one source for the room (unit does not pin REACHY_ENV)" <<PY
import re
unit = open("deploy/systemd/reachy-shell.service").read()
assert not re.search(r"^\s*Environment=REACHY_ENV", unit, re.M), \
    "the unit pins REACHY_ENV again: two sources for the room, precedence decided by systemd"
boot = open("shell/__main__.py").read()
assert "was never set" in boot and "log.warning" in boot, \
    "booting on the default env is back to being silent (or an INFO nobody reads)"
example = open(".env.example").read()
for var in ("REACHY_ENV", "REACHY_MEMORY_NAMESPACE", "REACHY_TOOLS_INCLUDE", "REACHY_TOOLS_EXCLUDE"):
    assert re.search(r"^# ?%s=" % var, example, re.M), "%s missing from .env.example" % var
assert "outrank" in example.lower(), ".env.example does not warn that these outrank the env file"
assert not re.search(r"^\s*REACHY_PROFILE=", example, re.M), \
    "REACHY_PROFILE uncommented: a set line outranks the env file and boots the booth on quiet"
PY

echo
if [ "$fails" -ne 0 ]; then
  echo "check_env_safety: $fails check(s) FAILED - do not switch rooms or start the shell"
  exit 1
fi
echo "check_env_safety: all checks passed"
