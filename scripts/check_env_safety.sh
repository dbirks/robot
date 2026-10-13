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

run "9. no template reaches for an external asset" <<PY
import pathlib
# Filtered egress on this box, none guaranteed at the booth: a page that pulls
# Tailwind/htmx/fonts from a CDN renders unstyled in front of the room (issue #38).
TPL = sorted(pathlib.Path("dashboard/templates").rglob("*.html"))
bad = []
for path in TPL:
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if "http-equiv" in line:
            continue
        if "http://" in line or "https://" in line or '="//' in line:
            bad.append("%s:%s" % (path, lineno))
assert len(TPL) >= 9, "only found %s templates - the glob is wrong" % len(TPL)
assert not bad, "template reaches off-box: %s" % bad
for name in ("tailwind.js", "htmx.min.js", "htmx-ext-sse.js", "fonts.css"):
    assert (pathlib.Path("dashboard/static/vendor") / name).stat().st_size > 1000, "vendored %s is empty" % name
woff = list(pathlib.Path("dashboard/static/vendor/fonts").glob("*.woff2"))
assert len(woff) == 6, "expected 6 vendored latin faces, found %s" % len(woff)
PY

run "10. the journal reader and the panel stay importable without fastapi" <<PY
# CI runs the tests in a minimal env (numpy + pyyaml + pytest). If a server-only
# import sneaks into the chain, every dashboard test errors there instead of here.
import ast, sys
for rel in ("dashboard/journal_read.py", "dashboard/env_panel.py"):
    tree = ast.parse(open(rel).read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    forbidden = mods & {"fastapi", "starlette", "sse_starlette", "uvicorn", "jinja2", "dotenv", "app", "shell"}
    assert not forbidden, "%s imports %s" % (rel, sorted(forbidden))
    extra = mods - set(sys.stdlib_module_names)
    assert not extra, "%s reaches outside the stdlib: %s" % (rel, sorted(extra))
PY

run "11. the dashboard cannot switch the room, show faces, or start a legacy audio owner" <<PY
import ast, pathlib

src = pathlib.Path("dashboard/server.py").read_text()
tree = ast.parse(src)

# (a) the room is a readout: nothing in this process writes REACHY_ENV, and no
#     environment route exists to receive a form post (docs/environments.md hard rule 1)
assert 'write_env_setting("REACHY_ENV"' not in src, "the dashboard can change the room"
assert '"REACHY_ENV"' not in src, "the dashboard resolves the room itself: two sources"
for rel in ("dashboard/templates/environment.html", "dashboard/templates/partials/environment.html"):
    text = pathlib.Path(rel).read_text()
    for control in ("<form", "<select", "<input", "<button", "hx-post"):
        assert control not in text, "%s offers %s - the env panel must be a readout" % (rel, control)

# (b) faces are hidden: the page and every faces route refuse, and the nav does not link them
assert "FACES_VISIBLE = False" in src, "the family faces panel is back on"
bodies = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.AsyncFunctionDef)}
for route in ("faces_page", "partial_faces", "face_thumbnail", "delete_face"):
    assert "faces_hidden()" in bodies[route], "faces route %s renders family data" % route
nav = pathlib.Path("dashboard/templates/base.html").read_text()
assert "/faces" not in nav and "faces.html" not in nav and "agent_button" not in nav, "a hidden panel is back in the nav"

# (c) no systemctl call in this process may name a retired unit: starting reachy-agent
#     opens a second mic/speaker owner and silences the real shell (ADR 0002)
units = set()
for node in ast.walk(tree):
    if isinstance(node, ast.List):
        strs = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if strs[:2] != ["systemctl", "--user"] or strs[2:3] not in (["restart"], ["stop"], ["start"]):
            continue
        last = node.elts[-1]
        if isinstance(last, ast.Constant) and isinstance(last.value, str):
            units.add(last.value)
assert units == set(), "the dashboard can systemctl %s" % sorted(units)
panel_tree = ast.parse(pathlib.Path("dashboard/env_panel.py").read_text())
units_list = ast.literal_eval(
    next(
        ast.unparse(n.value)
        for n in panel_tree.body
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and getattr(n.target if isinstance(n, ast.AnnAssign) else n.targets[0], "id", "") == "SERVICE_UNITS"
    )
)
assert "SERVICES = list(env_panel.SERVICE_UNITS)" in src, "the controllable list was re-hardcoded"
for retired in ("reachy-agent", "qwen3-tts"):
    assert retired not in units_list, "%s is back on the controllable list" % retired
PY

run "12. the booth prompt does not tell him to harvest facts unprompted" <<PY
$PRELUDE
from shell import config as C
from shell import env as E
booth = E.load_config("hackday")
assert booth.learn_facts is False, "hackday stopped disabling the fact miner"
assert C.LEARN_FACTS_NOTE not in booth.instructions, (
    "hackday still instructs him to save 'a name, who someone is, an age, where they "
    "live, a pet' with nobody asking - and 'remember' is in its tool list by design"
)
assert "remember, recall or forget" in booth.instructions, "the explicit-request half must stay"
home = E.load_config("home")
assert C.LEARN_FACTS_NOTE in home.instructions, "home lost the behaviour it shipped with"
assert len(home.instructions.encode()) == 1568, "home persona bytes moved (A2 parity bar)"
assert C.LEARN_FACTS_NOTE in C.TOOLS_NOTE and C.LEARN_FACTS_NOTE in C.SHELL_NOTES, (
    "the note and the shared tail drifted: the reconcile now removes nothing at all"
)
PY

echo
if [ "$fails" -ne 0 ]; then
  echo "check_env_safety: $fails check(s) FAILED - do not switch rooms or start the shell"
  exit 1
fi
echo "check_env_safety: all checks passed"
