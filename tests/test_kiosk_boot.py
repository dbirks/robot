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


def test_the_kiosk_unit_declares_the_dependency_it_can_actually_see():
    """`reachy-dashboard.service` is a USER unit (`systemctl --user list-unit-files` shows it
    enabled in ~/.config/systemd/user), and systemd does not order across scopes: an
    `After=reachy-dashboard.service` in a system unit is a dangling reference that looks like
    a guarantee. The dependency a SYSTEM unit can really declare is the user manager that
    owns the dashboard, plus systemd-user-sessions (the login/PAM pair needs it). The
    dashboard itself is gated by the bounded curl wait in kiosk.xinitrc, which is the only
    mechanism that works across the scope boundary - so the name must still appear, with the
    reason.
    """
    unit = (KIOSK / "reachy-kiosk.service").read_text()
    assert "After=user@1000.service" in unit, "a system unit can only order against the user manager"
    assert "After=systemd-user-sessions.service" in unit, "PAMName= needs the session machinery up"
    assert "reachy-dashboard.service" in unit, "and the dashboard dependency must be named, not silently dropped"
    assert "USER unit" in unit or "user unit" in unit, "with the scope fact written down next to it"
    assert not re.search(r"^After=reachy-dashboard\.service", unit, re.M), (
        "ordering a system unit on a user-scope unit is a no-op that reads like a dependency"
    )


def test_the_autologin_drop_in_reaches_login_with_a_user_and_no_auth():
    """The 2026-10-13 failure, pinned. The drop-in merged cleanly (`DropInPaths` listed it,
    `ExecStart` contained `--autologin david`) and a reboot did not fix it, because the line
    was copied from Arch's legacy getty fragment, `-o '-p -- \\u'`. agetty(8): `--autologin`
    "causes an -f username option and argument to be added to the /bin/login command line.
    See --login-options, which can be used to modify this option's behavior" - giving `-o`
    replaces that, so login ran as `/bin/login -p --`: no user, no `-f`, an interactive
    prompt that dies on agetty's own 60 s timeout and re-spawns forever.
    """
    conf = (KIOSK / "10-autologin.conf").read_text()
    execs = re.findall(r"^ExecStart=(.*)$", conf, re.M)
    assert len(execs) == 2, (
        "one empty reset then one real line: without the reset the fragment is appended to, not replaced"
    )
    assert execs[0].strip() == "", "the reset line must be empty"
    line = execs[1]
    assert "--autologin david" in line, "and the line must autologin the kiosk user"
    assert line.startswith("-/usr/bin/agetty"), "the leading - keeps getty alive if login exits"

    quoted = re.search(r"-o\s+'([^']*)'", line)
    assert quoted, "login-options must be single-quoted, or systemd splits them into agetty's argv"
    opts = quoted.group(1).split()
    assert "-f" in opts, "-f is what makes login skip authentication; --autologin will not add it once -o is given"
    assert any(o.endswith(r"\u") for o in opts), "and the \\u placeholder is where agetty puts the username"


def test_the_drop_in_targets_the_unit_that_actually_runs_on_this_box():
    """Arch moved the fragment around in 2025, so "is it getty@ or agetty@?" is a real question.
    Measured here (util-linux 2.42, systemd getty@.service with Alias=autovt@.service, enabled
    via /etc/systemd/system/getty.target.wants/getty@tty1.service):
    `systemctl status agetty@tty1.service` -> not found; getty@tty1.service is the live unit.
    If that ever flips, a getty@tty1 drop-in silently applies to nothing - this is the tripwire,
    and it is on the conf because the installer no longer installs the drop-in at all.
    """
    conf = (KIOSK / "10-autologin.conf").read_text()
    assert "getty@tty1.service.d" in conf
    assert "agetty@.service" in conf, "the conf must record which unit name was checked, and how"
    assert "agetty@tty1.service.d" not in conf
    installer = (KIOSK.parent.parent / "scripts" / "install_kiosk.sh").read_text()
    assert 'cp "$REPO/deploy/kiosk/10-autologin.conf"' not in installer, (
        "masking getty@tty1 makes the drop-in unreachable; the installer may name it as the "
        "fallback, but it must not copy it into /etc"
    )


def test_the_kiosk_unit_is_the_console_owner_a_user_unit_can_never_be():
    """The shape that fixes the Xorg.wrap refusal, pinned line by line. A systemd USER service
    is never the owner of the foreground VT, so /usr/lib/Xorg.wrap says "Only console users are
    allowed to run the X server" and the kiosk crash-loops - measured 2026-10-13. The pair that
    makes a SYSTEM service the console owner is `PAMName=` + `TTYPath=` (+ `StandardInput=tty`):
    /etc/pam.d/login -> system-local-login -> system-login, whose line 20 is `pam_systemd.so`,
    so the service gets a real logind session on seat0 - which is also what ACLs /dev/dri/card1
    for david. `UtmpIdentifier`/`UtmpMode=user` is what makes logind treat it as the user's
    session rather than a system job. If someone deletes either half of the pair, X stops
    working and there is no keyboard on this box to discover it with.
    """
    unit = (KIOSK / "reachy-kiosk.service").read_text()
    for directive in (
        "User=david",
        "PAMName=login",
        "TTYPath=/dev/tty1",
        "StandardInput=tty",
        "UtmpIdentifier=tty1",
        "UtmpMode=user",
        "StandardOutput=journal",
        "Environment=XDG_SESSION_TYPE=x11",
        "Restart=always",
    ):
        assert re.search(rf"^{re.escape(directive)}$", unit, re.M), f"missing {directive}"
    assert "Restart=on-failure" not in unit, (
        "a clean exit of the X session (a pkill, a closed window) makes startx exit 0, and "
        "on-failure would then leave the booth with a black panel and no keyboard to fix it"
    )
    assert re.search(r"^ExecStartPre=\+/usr/bin/chvt 1$", unit, re.M), (
        "chvt has to run as root: `+` is the only way to keep it privileged under User=david"
    )
    startx = re.findall(r"^ExecStart=(.*)$", unit, re.M)
    assert len(startx) == 1 and "startx" in startx[0]
    assert "kiosk.xinitrc" in startx[0], "the launcher must be the session that waits for the panel"
    assert "vt1" in startx[0], "X must be told which VT to allocate - that is what vt1 is for"
    # -keeptty is BANNED, deliberately, and this assertion is the reason it stays banned. It pins
    # X's stdio to /dev/tty1, so every word the server says goes to a console on a box with no
    # keyboard - unwatchable for the operator and unreadable for me. The first version of this unit
    # failed eight times with an empty journal for exactly that reason, and -logfile /dev/null then
    # threw away the only other copy. Without them X writes ~/.local/share/xorg/Xorg.0.log, which is
    # reachable over SSH: on this machine a readable artifact is worth more than a kept tty.
    assert "-keeptty" not in startx[0], "X's output must reach a file we can read over SSH"
    assert "-logfile /dev/null" not in startx[0], "never discard the only copy of the failure"
    assert "WantedBy=multi-user.target" in unit, "a SYSTEM unit, so it comes up without anyone logging in"
    assert "Only console users are allowed" in unit, "the measured failure travels with the fix"
    assert "cage" in unit.lower() or "wayland" in unit.lower(), (
        "why this is X and not cage/Wayland on a 1070 with no keyboard (cage-kiosk #441)"
    )


def test_the_console_hook_is_labelled_fallback_and_keeps_a_shell():
    """The hook is no longer the primary path, and it must not pretend to be: it needs a
    keyboard and an unmasked getty@tty1, neither of which exists on this box. Two things are
    pinned. (1) It says FALLBACK PATH ONLY, so nobody reads it as the launcher. (2) It does not
    `exec startx`: exec replaces the login shell, so a startx that dies leaves getty to respawn
    into the same failure - a black monitor with no trace and no shell to read it in.
    Guarded for CI boxes with no console user.
    """
    hook = Path.home() / ".bash_profile"
    if not hook.exists() or "reachy kiosk (issue #37)" not in hook.read_text():  # pragma: no cover
        return
    block = hook.read_text().split("reachy kiosk (issue #37)", 1)[1]
    assert "FALLBACK PATH ONLY" in block, "the hook must not read as the primary launcher"
    assert "exec startx" not in block, "startx must not replace the console login shell"
    assert "startx" in block and "vt1" in block
    assert '"$(tty 2>/dev/null)" = "/dev/tty1"' in block, "and it must still bind to the console session"


def test_the_kiosk_is_installed_in_system_scope_and_unlinked_from_user_scope():
    """The installer must stop pretending the user unit works. Two failures cost the whole
    evening of 2026-10-13: a kiosk unit that systemd-user happily links but Xorg.wrap refuses
    ("Only console users are allowed to run the X server" - a user service never owns the
    foreground VT), and a second reader fighting for the same tty. So: install into
    /etc/systemd/system, mask getty@tty1, verify the merged unit with no root, and delete the
    user-scope link rather than leaving a unit that reads as "installed" while the screen is
    black. The autologin count assertion keeps the fallback conf single-shot: two --autologin
    flags in one ExecStart is how this file got quietly broken in the first place.
    """
    installer = (KIOSK.parent.parent / "scripts" / "install_kiosk.sh").read_text()
    assert "SYS_DIR=/etc/systemd/system" in installer
    # Not a straight copy: `%h` in a SYSTEM unit is the system manager's home (/root) and does
    # NOT follow User=david, so the repo file's path has to be substituted per-install. Shipping
    # /root/dev/robot/.../kiosk.xinitrc is what made the first booth night crash-loop 44 times
    # with no Xorg log, no error in the journal, and `systemd-analyze verify` completely silent
    # - verify checks directive spelling and never opens an ExecStart path.
    assert re.search(r'sed "s#/[^#]*dev/robot#\$REPO#g"', installer), (
        "ExecStart path must be substituted against $REPO, or a different $HOME ships /root again"
    )
    assert 'run sudo cp "$KIOSK_STAGED"' in installer, "install the substituted copy, not the source"
    assert "'rm -f \"$KIOSK_STAGED\"'" in installer, "clean the staged temp up"
    assert "systemctl mask getty@tty1.service" in installer, "two readers cannot own one tty"
    assert 'rm -f "$REPO_DIR/$KIOSK_UNIT.service"' in installer, "the user-scope link has to go"
    assert 'ln -sfn "$REPO/deploy/kiosk/$KIOSK_UNIT.service"' not in installer, (
        "re-linking reachy-kiosk.service into ~/.config/systemd/user puts us back in the crash loop"
    )
    assert "systemd-analyze verify" in installer, "a directive typo must not survive to enable time"
    assert (KIOSK / "10-autologin.conf").read_text().count("--autologin") == 1


def test_the_system_unit_never_uses_the_h_home_specifier():
    """`%h` is the trap that cost twenty minutes of a live booth.

    In a USER unit it means the invoking user's home, which is why the line was correct in the
    file we copied from. In a SYSTEM unit it means the home of whoever runs the system manager -
    `/root` - and it does not follow `User=david`. systemd then runs a command whose argument
    simply does not exist: startx dies in under a second, X never spawns far enough to write a
    log anywhere, and the failure text goes to a console on a machine with no keyboard.
    """
    unit = (KIOSK / "reachy-kiosk.service").read_text()
    body = "\n".join(l for l in unit.splitlines() if not l.lstrip().startswith("#"))
    assert "%h" not in body, "%h in a system unit is /root, not User='s home"
    assert "/home/david/dev/robot/deploy/kiosk/kiosk.xinitrc" in unit, "a real, existing path"
    assert (KIOSK / "kiosk.xinitrc").exists(), "the unit must point at a file that exists"
