"""One home for the A2 parity number, plus the reason it is allowed to move.

docs/environments.md A2: booting with `REACHY_ENV` unset must reproduce the pre-rebuild
home configuration exactly. The byte count of the resolved instructions is the cheap,
brutal version of that check - a single changed word shows up.

It lives here because it was duplicated in three files (tests/test_env.py,
tests/test_persona.py, scripts/check_env_safety.sh). A guard that exists in three
places disagrees with itself the first time a prompt changes *on purpose*: two files get
updated, one is missed, and the safety gate goes red on an unrelated night. One constant,
three readers.

2026-10-13: **1853 -> 1765**, deliberately - the shared `TIME_NOTE` lost its stamp language
when `shell/llm_proxy.py` stopped injecting a per-message timestamp (see below; #45). Earlier
the same day: **1568 -> 1853**.

The shared `TIME_NOTE` told the model that the stamp on the newest message "is today's
date and the current time - use it". But `shell/llm_proxy.py`'s Timestamper caches the
stamp per message *text* inside a long-lived process, so an identical line re-asked
after a `realtime.session_reset` arrives wearing its OLD stamp - measured 24.8 and 41.5
minutes stale in the journal. Obeying the note out loud therefore means stating a wrong
time to a stranger, which is worse than the +209 bytes this change costs. The note now
sends him to `get_time`: the same local clock, read at answer time rather than remembered
from a cache key.

The stale-stamp cache itself is a separate defect (it still misdates anything
date-relative) and is deliberately NOT touched the night before a demo.
"""

HOME_PROMPT_BYTES = 1765
