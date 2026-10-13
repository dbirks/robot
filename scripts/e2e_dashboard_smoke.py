#!/usr/bin/env python3
"""Real-browser smoke of the booth dashboard — Playwright, not pytest.

Deliberately NOT a pytest test: CI runs `uvx --with numpy --with pyyaml --with
pytest pytest tests/` on a hardware-free, dependency-light tree, so nothing under
tests/ may import fastapi, sse_starlette or playwright (docs/adr/0007, AGENTS.md).
This file lives in scripts/ and runs from the separate Playwright venv, which is
not in the repo lockfile:

    /home/david/.venvs/playwright/bin/python scripts/e2e_dashboard_smoke.py
    ... --serve          # smoke THIS working tree on a throwaway loopback port
    ... --base-url http://127.0.0.1:3001 --only routes,net

Checks (each reported PASS/FAIL):
  routes  every nav target renders with visible text (never asserted on h1)
  net     zero requests off 127.0.0.1 — the proof the vendored assets took
  live    the conversation panel's SSE stream really delivers (or the honest
          empty/quiet state, which is what a stopped shell deserves)
  controls persona apply / service restart / volume are present and readable —
          NEVER clicked: restart can stop llama-server, /settings/save can start
          the dead legacy agent. Any non-GET the page attempts is aborted by the
          route guard and counted as a violation.
  errors  console errors + uncaught page errors (invisible at the venue otherwise)
  shots   full-page screenshot per route into /tmp/e2e/

Read-only against the app: it GETs, it renders, it screenshots, it terminates its
own child process. It never touches systemd and never opens an audio device.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
SHOT_DIR = Path("/tmp/e2e")
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", ""}
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")

# (name, path, visible-text musts, acceptable statuses). Texts are matched
# case-insensitively against the DOM, because the templates uppercase by CSS.
# /faces is 404 on purpose (FACES_VISIBLE=False, issue #38): a family photo is not
# booth content, and "hidden" must not be readable as "nobody seen yet".
ROUTES: list[tuple[str, str, tuple[str, ...], tuple[int, ...]]] = [
    ("index", "/", ("Conversation",), (200, 307)),
    ("conversation", "/conversation", ("Conversation", "SHELL"), (200,)),
    ("traces", "/traces", ("Traces",), (200,)),
    ("services", "/services", ("Services",), (200,)),
    ("settings", "/settings", ("Settings",), (200,)),
    ("environment", "/environment", ("Deployment environment",), (200,)),
    ("persona", "/persona", ("Persona",), (200,)),
    ("faces-hidden", "/faces", ("FACES ARE HIDDEN IN THIS DEPLOYMENT",), (404,)),
]
NAV_TARGETS = ["/conversation", "/traces", "/services", "/settings", "/persona", "/environment"]
# /faces answers 404 on purpose (FACES_VISIBLE=False), so its 404 and the matching
# browser resource-error are expected, not findings.
EXPECTED_ERROR_PATHS = {"/faces"}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []

    def add(self, check: str, ok: bool | None, detail: str) -> None:
        self.rows.append((check, "" if ok is None else "PASS" if ok else "FAIL", detail))

    def dump(self) -> int:
        width = max((len(f"{c} {n}") for c, n, _ in self.rows), default=0)
        failed = 0
        for check, verdict, detail in self.rows:
            if verdict == "FAIL":
                failed += 1
            print(f"[{verdict or 'INFO':>4}] {check + ' ' + ''.ljust(0):<{width - 4}}  {detail}")
        print(f"\n{sum(1 for _, v, _ in self.rows if v == 'PASS')} pass, {failed} fail")
        return failed


def visible_text(page, text: str, timeout_ms: int = 8000) -> bool:
    try:
        page.get_by_text(text, exact=False).first.wait_for(state="visible", timeout=timeout_ms)
        return True
    except Exception:
        return False


def sse_raw_probe(base: str, seconds: float = 6.0) -> tuple[bool, str]:
    """Open /sse/conversation outside the browser and see if frames flow.

    A GET on a stdlib socket: it does not go through the Playwright route guard, so
    it is excluded from the zero-external-request audit (it is loopback anyway).
    """
    u = urlparse(base)
    host, port = u.hostname or "127.0.0.1", u.port or 80
    buf = b""
    try:
        s = socket.create_connection((host, port), timeout=2.0)
    except Exception as e:  # noqa: BLE001
        return False, f"connect failed: {e}"
    s.settimeout(1.0)
    try:
        s.sendall(
            f"GET /sse/conversation HTTP/1.1\r\nHost: {host}:{port}\r\n"
            f"Accept: text/event-stream\r\nConnection: keep-alive\r\n\r\n".encode()
        )
        deadline = time.time() + seconds
        while time.time() < deadline and len(buf) < 200_000:
            try:
                chunk = s.recv(4096)
            except TimeoutError:
                continue
            if not chunk:
                break
            buf += chunk
    except Exception as e:  # noqa: BLE001
        return False, f"{e}; got {len(buf)} bytes"
    finally:
        s.close()
    head = buf.split(b"\r\n\r\n", 1)[0].decode(errors="replace").lower()
    body = buf.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in buf else b""
    status = "200" in head.split("\r\n", 1)[0]
    typed = "text/event-stream" in head
    frames = body.count(b"event: refresh")
    payload = sum(len(m) for m in body.split(b"data: ")[1:])
    ok = status and typed and frames > 0 and payload > 0
    return (
        ok,
        f"status200={status} event-stream={typed} refresh_frames={frames} data_bytes={payload} wire_bytes={len(body)}",
    )


def wait_for_server(base: str, seconds: float) -> bool:
    import urllib.request

    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/conversation", timeout=2.0) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:3001")
    ap.add_argument("--serve", action="store_true", help="spawn THIS tree on a throwaway loopback port")
    ap.add_argument("--port", type=int, default=3999)
    ap.add_argument("--only", default="routes,net,live,controls,errors,shots")
    ap.add_argument("--quiet-live", type=float, default=7.0, help="seconds to watch the conversation panel")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    rep = Report()
    rep.add("target", None, f"base={args.base_url} cwd={REPO} playwright_venv_not_in_lockfile=True")

    child: subprocess.Popen[bytes] | None = None
    base = args.base_url
    if args.serve:
        env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "PYTHONUNBUFFERED": "1"}
        # Same app object the unit runs, on a port that is not the unit's. Loopback
        # bind, no systemd, and the only process this script terminates is its own.
        child = subprocess.Popen(
            [
                str(REPO / "dashboard/.venv/bin/python"),
                "-m",
                "uvicorn",
                "dashboard.server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--log-level",
                "warning",
            ],
            cwd=str(REPO),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base = f"http://127.0.0.1:{args.port}"
        ready = wait_for_server(base, 40.0)
        rep.add("serve", ready, f"spawned dashboard.server:app pid={child.pid} on {base} ready={ready}")
        if not ready:
            child.terminate()
            return rep.dump()

    if SHOT_DIR.exists():
        shutil.rmtree(SHOT_DIR, ignore_errors=True)
    SHOT_DIR.mkdir(parents=True, exist_ok=True)

    requests_log: list[str] = []
    mutations: list[str] = []
    external: list[str] = []
    bad_status: list[str] = []
    failed_reqs: list[str] = []
    console_errors: list[str] = []
    page_errors: list[str] = []
    hosts_seen: set[str] = set()

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 1000})

        def guard(route):
            """Hard read-only guard: nothing leaves this box, nothing mutates."""
            req = route.request
            host = urlparse(req.url).hostname or ""
            if host in ALLOWED_HOSTS:
                hosts_seen.add(host)
            if req.method not in SAFE_METHODS:
                mutations.append(f"{req.method} {req.url}")
                route.abort("blockedbyclient")
                return
            if host not in ALLOWED_HOSTS:
                external.append(f"{req.method} {req.url}")
                route.abort("blockedbyclient")
                return
            requests_log.append(f"{req.method} {req.url}")
            route.continue_()

        ctx.route("**/*", guard)

        def on_request_failed(r) -> None:
            # A request that never got an answer: filtered egress shows up here as
            # an abort, which is the failure mode the vendored assets fixed.
            if urlparse(r.url).hostname not in ALLOWED_HOSTS:
                failed_reqs.append(f"{r.method} {r.url} {r.failure or ''}")

        def on_response(r) -> None:
            if r.status >= 400 and urlparse(r.url).path not in EXPECTED_ERROR_PATHS:
                bad_status.append(f"{r.status} {r.url}")

        def on_console(m) -> None:
            if m.type != "error":
                return
            url = (m.location or {}).get("url") or ""
            if urlparse(url).path in EXPECTED_ERROR_PATHS:
                return
            console_errors.append(f"{m.type}: {m.text} @{url}")

        ctx.on("requestfailed", on_request_failed)
        page = ctx.new_page()
        page.on("console", on_console)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.on("response", on_response)

        # --- routes ---
        for name, path, texts, statuses in ROUTES if "routes" in only else []:
            try:
                resp = page.goto(base + path, wait_until="domcontentloaded", timeout=30_000)
                status = resp.status if resp else 0
                missing = [t for t in texts if not visible_text(page, t)]
                ok = status in statuses and not missing
                rep.add("routes", ok, f"{name:14} {path:14} status={status} missing={missing}")
                if "shots" in only:
                    try:
                        page.wait_for_timeout(300)
                        page.screenshot(path=str(SHOT_DIR / f"{name}.png"), full_page=True)
                    except Exception as e:  # noqa: BLE001
                        rep.add("shots", False, f"{name}: screenshot failed {e}")
            except Exception as e:  # noqa: BLE001
                rep.add("routes", False, f"{name} {path}: {e}")

        if "routes" in only:
            page.goto(base + "/conversation", wait_until="domcontentloaded", timeout=30_000)
            hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.getAttribute('href'))")
            gone = [h for h in NAV_TARGETS if h not in hrefs]
            rep.add("nav", not gone, f"sidebar hrefs present, missing={gone}")

        # --- live conversation ---
        if "live" in only:
            page.goto(base + "/conversation", wait_until="domcontentloaded", timeout=30_000)
            sel = "#conversation-scroll > div"
            before = page.locator(sel).count()
            sse_hits = [u for u in requests_log if "/sse/conversation" in u]
            body = page.locator(sel).first.inner_text() if before else ""
            empty = "No conversation yet" in (body or "")
            quiet = visible_text(page, "SHELL QUIET", 2000)
            alive = visible_text(page, "SHELL ALIVE", 2000)
            page.wait_for_timeout(int(args.quiet_live * 1000))
            after = page.locator(sel).count()
            delivered = after > before
            probe_ok, probe_detail = sse_raw_probe(base)
            # Delivery needs something to deliver: the stream only emits when the
            # shell writes a journal row, and reachy-shell is stopped right now. So
            # the honest assertions are (a) htmx really opened the SSE connection in
            # the page, (b) the endpoint accepts the stream, and (c) the panel shows
            # either journal rows or the empty state plus the liveness badge. If the
            # shell were live, `delivered` would be the real proof.
            ok = bool(sse_hits) and probe_ok and (before > 0 or empty) and (quiet or alive)
            rep.add(
                "live",
                ok,
                f"browser sse connect={len(sse_hits)} rows {before}->{after} in {args.quiet_live:.0f}s "
                f"(delivered={delivered}) empty_state={empty} badge={'ALIVE' if alive else 'QUIET' if quiet else 'none'}",
            )
            rep.add(
                "live",
                None,
                f"raw /sse/conversation probe (GET, outside the browser): {probe_detail} "
                f"— frames only flow when the shell writes rows; reachy-shell is stopped",
            )
            if "shots" in only:
                page.screenshot(path=str(SHOT_DIR / "conversation-live.png"), full_page=True)

        # --- controls (presence + state only, never clicked) ---
        if "controls" in only:
            page.goto(base + "/persona", wait_until="domcontentloaded", timeout=30_000)
            apply_btn = page.locator("button[hx-post='/persona/apply']")
            n_apply = apply_btn.count()
            vals = apply_btn.evaluate_all("els => els.map(e => e.getAttribute('hx-vals'))")
            rep.add(
                "controls",
                n_apply > 0,
                f"persona apply buttons={n_apply} visible={apply_btn.first.is_visible() if n_apply else False} "
                f"hx-post=/persona/apply hx-vals={json.dumps(vals)[:120]} (NOT clicked)",
            )
            if "shots" in only:
                page.screenshot(path=str(SHOT_DIR / "persona.png"), full_page=True)

            page.goto(base + "/services", wait_until="domcontentloaded", timeout=30_000)
            restart = page.locator("button[hx-post*='/restart']")
            stop = page.locator("button[hx-post*='/stop']")
            n_r, n_s = restart.count(), stop.count()
            rep.add(
                "controls",
                n_r + n_s > 0,
                f"service restart buttons={n_r} stop buttons={n_s} "
                f"visible={restart.first.is_visible() if n_r else False} (NOT clicked)",
            )
            units = page.eval_on_selector_all(
                "button[hx-post^='/services/']", "els => els.map(e => e.getAttribute('hx-post'))"
            )
            rep.add("controls", None, f"service actions the page would POST: {units[:12]}")

            page.goto(base + "/settings", wait_until="domcontentloaded", timeout=30_000)
            vol = page.locator("input[type='range'][name='volume_boost']")
            n_v = vol.count()
            vol_state = (
                vol.first.evaluate("e => ({min: e.min, max: e.max, value: e.value, disabled: e.disabled})")
                if n_v
                else {}
            )
            tts = page.locator("select[name='tts_engine']")
            tts_val = tts.first.input_value() if tts.count() else None
            save = page.locator("button[hx-post='/settings/save']")
            rep.add(
                "controls",
                n_v > 0,
                f"volume slider present={n_v > 0} state={vol_state} tts_engine={tts_val!r} "
                f"save button={save.count()} (NOT clicked, NOT dragged)",
            )

        # --- errors / network audit ---
        hosts = sorted(hosts_seen)
        rep.add("net", not external, f"off-box requests={external[:8]} unique hosts={hosts}")
        rep.add("net", not mutations, f"mutation attempts made by pages={mutations[:8]}")
        rep.add("net", not failed_reqs, f"failed/blocked requests (filtered egress shows here)={failed_reqs[:8]}")
        rep.add("net", not bad_status, f"4xx/5xx responses={bad_status[:8]}")
        rep.add("errors", not console_errors, f"console errors={console_errors[:6]}")
        rep.add("errors", not page_errors, f"uncaught page errors={page_errors[:6]}")
        rep.add(
            "shots",
            None,
            "screens: " + ", ".join(sorted(p.name for p in SHOT_DIR.glob("*.png"))) + f" in {SHOT_DIR}",
        )

        browser.close()

    if child:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:  # only ever our own child
            child.kill()
        rep.add("serve", None, f"terminated own child pid={child.pid} (no systemd unit touched)")

    return rep.dump()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
