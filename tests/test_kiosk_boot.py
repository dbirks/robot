"""Guards on the boot path for the attendee screen (issue #37).

These are textual and `sh -n` checks, deliberately. The kiosk is a shell script and two
unit files: nothing here imports the running system, needs a display, or opens a device, so
all of it is CI-safe. What they protect is the two things the first real screen taught us on
2026-10-13, both of which look fine until a monitor is attached.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

KIOSK = Path(__file__).resolve().parent.parent / "deploy" / "kiosk"


def test_the_kiosk_xinitrc_is_valid_sh():
    """It runs as the session's only process, at boot, with nobody able to read a traceback."""
    if shutil.which("sh") is None:  # pragma: no cover - CI has /bin/sh
        return
    assert subprocess.run(["sh", "-n", str(KIOSK / "kiosk.xinitrc")]).returncode == 0


def test_the_kiosk_waits_for_the_dashboard_before_it_navigates():
    """`chromium --kiosk` has no retry, and the panel is a peer user unit with no ordering
    against it: win the boot race by a second and the booth screen reads "This site can't be
    reached" until someone with a keyboard shows up. Assert the retry exists AND is bounded -
    an unbounded wait in a session that owns the console is a black screen instead.
    """
    text = (KIOSK / "kiosk.xinitrc").read_text()
    assert "curl -fsS" in text, "the kiosk no longer probes the panel before navigating"
    assert re.search(r"while\s+\[", text), "and the probe must repeat, not test once"
    assert re.search(r"-lt\s+(1[0-9]|[2-9][0-9])", text), "the wait must be BOUNDED"
    assert "opening anyway" in text, "and it must still open a screen when the wait expires"


def test_the_kiosk_unit_declares_the_dashboard_dependency():
    """The console-login hook is the path that runs today; the unit is the shape the real
    fix takes. Both should carry the ordering so nobody rediscovers the race a second time.
    """
    unit = (KIOSK / "reachy-kiosk.service").read_text()
    assert "After=reachy-dashboard.service" in unit
    assert "Wants=reachy-dashboard.service" in unit


def test_the_kiosk_launcher_is_the_console_session_not_a_user_service():
    """The finding, pinned. Xorg.wrap refuses any process that is not the owner of the
    foreground VT, and a systemd user service never is one - so `xinit` in reachy-kiosk.service
    crash-loops on "Only console users are allowed to run the X server". If someone moves the
    launch back into a unit, this test is where the reason gets re-read.
    """
    hook = Path.home() / ".bash_profile"
    if not hook.exists():  # pragma: no cover - CI has no console
        return
    text = hook.read_text()
    if "reachy kiosk (issue #37)" not in text:
        return  # a machine that never installed the kiosk
    assert '"$(tty 2>/dev/null)" = "/dev/tty1"' in text, "the kiosk must bind to the console session"
    assert "startx" in text
    assert (KIOSK / "10-autologin.conf").read_text().count("--autologin") == 1
