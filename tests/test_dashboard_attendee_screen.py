"""Guards on the attendee screen as it is actually drawn (issue #37 follow-up).

Textual and headless, in the style of tests/test_kiosk_boot.py: nothing here imports the
running system, needs a robot, or opens an audio device. What they protect is the three
things the operator read off the panel on 2026-10-13, at 3840x2160 with
`chromium --kiosk --force-device-scale-factor=2` - i.e. on a LOGICAL 1920x1080 canvas:

  1. "the conversation needs to track the latest message down below" - the view did not
     follow, and the reason is measurable: the vendored htmx 2.0.4 sse extension swaps via
     the internal api.swap(), which fires htmx:afterSwap carrying detail.elt and NO
     detail.target, so the handler that tested `event.detail.target` never ran.
  2. "the padding on the bottom is really rough" - probed off the live framebuffer, 2.16%
     of the ink on screen sat inside the lowest 120 device pixels.
  3. "if we can detect that it's 4K and then increase the zoom ratio" - a 4K panel driven
     at DPR 2 is a 1920 CSS px canvas, so any `(min-width: 2560px)` rule cannot fire on it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "dashboard" / "templates" / "base.html"
PARTIAL = ROOT / "dashboard" / "templates" / "partials" / "conversation.html"
BODY = ROOT / "dashboard" / "templates" / "partials" / "conversation_body.html"
ATTENDEE_TEMPLATES = (
    ROOT / "dashboard" / "templates" / "conversation.html",
    PARTIAL,
    BODY,
)


def _script() -> str:
    """The inline follow script, which is the house idiom on a page with no build step."""
    blocks = re.findall(r"<script>(.*?)</script>", PARTIAL.read_text(), re.S)
    assert blocks, "the conversation partial lost its inline script"
    return "\n".join(blocks)


def _scroll_tag() -> str:
    """The opening tag of the real scroll container, attributes and all."""
    text = PARTIAL.read_text()
    start = text.index('<div id="conversation-scroll"')
    return text[start : text.index(">", start) + 1]


# --- (1) auto-follow, and the swap event that actually fires ----------------------


def test_the_pane_that_scrolls_is_the_pane_that_swaps():
    """The page updates by a swap INSIDE the container, so a window-level scroll would be
    aiming at the wrong element. The SSE swap target and the overflow box are one node."""
    tag = _scroll_tag()
    assert "overflow-y-auto" in tag, "#conversation-scroll is no longer the scroll container"
    assert "conversation-scroll" in tag, "the padding rule is not applied to the scroll container"
    assert 'sse-swap="refresh"' in tag and 'hx-swap="innerHTML"' in tag, "the swap target moved"
    assert "hx-target" not in tag, "an hx-target here points the follow hook at another element"
    # the follow script reaches the same node by id, not by a window scroll
    assert "getElementById('conversation-scroll')" in PARTIAL.read_text()


def test_the_follow_handler_keys_on_the_field_htmx_actually_sets():
    """htmx 2.0.4 + the vendored sse extension: afterSwap from api.swap() carries detail.elt.

    Checked against the shipped vendor files rather than from memory, because the one
    version of this handler that did nothing read event.detail.target.
    """
    sse_ext = (ROOT / "dashboard" / "static" / "vendor" / "htmx-ext-sse.js").read_text()
    assert "api.swap(target, content, swapSpec)" in sse_ext, "the sse extension stopped swapping"
    assert "contextElement" not in sse_ext, "and it still passes no contextElement"
    htmx = (ROOT / "dashboard" / "static" / "vendor" / "htmx.min.js").read_text()
    assert 'he(e,"htmx:afterSwap",o.eventInfo)' in htmx, (
        "afterSwap is triggered on the swapped element, from the caller's eventInfo"
    )
    assert "n.elt=e" in htmx, "triggerEvent puts the element in detail.elt, which is the only handle the SSE path gives"

    script = _script()
    assert "htmx:afterSwap" in script, "the follow handler is no longer hooked to a swap"
    assert "detail.elt" in script, (
        "detail.target is absent on the SSE swap path; keying on it is the bug that made "
        "the view stop tracking new rows"
    )
    assert re.search(r"el\.scrollTop\s*=\s*el\.scrollHeight", script), "and it does scroll the pane"


def test_the_follow_rule_leaves_a_scrolled_up_reader_alone_and_repins_on_the_way_back():
    """Pinned-to-bottom with slack: within ~100px of the last row we follow, beyond it we
    do not, and scrolling back down re-pins without any click."""
    script = _script()
    slack = re.search(r"PIN_SLACK_PX\s*=\s*(\d+)", script)
    assert slack, "the pinned-to-bottom slack must be a named constant"
    assert 80 <= int(slack.group(1)) <= 120, f"slack {slack.group(1)}px is either yank-everything or never"
    assert re.search(r"scrollHeight\s*-\s*\w+\.scrollTop\s*-\s*\w+\.clientHeight", script), (
        "the distance to the bottom has to be measured, not guessed"
    )
    assert re.search(r"addEventListener\(\s*'scroll'", script), "and it needs the reader's own scrolling"
    assert re.search(r"addEventListener\(\s*'scroll'[\s\S]*?\},\s*true\s*\)", script), (
        "element scroll events do not bubble: without capture the listener never fires"
    )
    guard = re.search(r"pinnedToBottom\s*===\s*'false'\s*\)\s*return", script)
    assert guard, "the swap handler must bail out when the reader is up in history"
    assign = script.index("el.scrollTop = el.scrollHeight")
    assert guard.start() < assign, "the check has to come before the scroll, not after it"
    assert script.count("pinnedToBottom") >= 3, "scrolling back down must re-pin"


def test_the_follow_script_is_bound_once_per_document():
    """The partial is also served on its own at /partials/conversation; re-runs must not
    stack a listener per swap."""
    script = _script()
    assert "conversationFollowBound" in script and "return;" in script


# --- (2) bottom clearance ---------------------------------------------------------


def _style_block() -> str:
    m = re.search(r"<style>(.*?)</style>", BASE.read_text(), re.S)
    assert m, "base.html lost its style block"
    return m.group(1)


def test_the_caption_pane_has_bottom_clearance():
    """No row may sit within 1.5rem of the pane's bottom edge. The condition being aimed at
    is the measured one: 2.16% of the screen's ink inside the lowest 120 device pixels."""
    pad = re.search(r"\.conversation-scroll\s*\{[^}]*padding-bottom:\s*([\d.]+)rem", _style_block())
    assert pad, "the scroll pane lost its bottom padding rule"
    assert float(pad.group(1)) >= 1.5, f"{pad.group(1)}rem is the crampedness we were told to fix"
    # The wrapper must not be able to spill: the journal line used to sit below a
    # hard-coded calc height and land in the same bottom band.
    text = PARTIAL.read_text()
    assert "flex-1" in _scroll_tag() and "min-h-0" in _scroll_tag(), (
        "the pane needs to take the leftover height, not a guessed calc()"
    )
    assert 'class="shrink-0 pt-2 text-xs"' in text, "the journal line is inside the wrapper"


# --- (3) adaptive size, on a feature that is true at DPR 2 -----------------------


def _media_preludes() -> list[str]:
    return re.findall(r"@media([^{]+)\{", _style_block())


def test_the_big_panel_step_uses_a_media_feature_that_is_true_at_dpr_2():
    """The panel reports innerWidth=1920 / devicePixelRatio=2, so 2560px queries are dead
    letters on it. The step is gated on resolution AND width; both were probed true on that
    box, and a 1440x900@1x laptop plus a 3840x2160@1x monitor were probed to reject it."""
    preludes = _media_preludes()
    assert preludes, "base.html has no @media rules to check"
    assert any("min-resolution" in p for p in preludes), "no density-based step for the 4K panel"
    for prelude in preludes:
        for wide in re.findall(r"min-width:\s*(\d+)px", prelude):
            assert int(wide) < 2560, f"{prelude.strip()} can never fire on a 1920 CSS px canvas"
    assert re.search(r"\bclamp\(", _style_block()), "sizing must scale with the viewport, not a fixed step"
    assert re.search(r"html\s*\{\s*font-size:\s*clamp\(", _style_block()), "on the root, so rem utilities follow"
    assert "1920" in BASE.read_text() and "dpr=2" in BASE.read_text(), (
        "the measurement travels with the rule, or the next reader re-adds a 2560px query"
    )


def _chromium_probe(width: int, scale: int, tmp_path: Path) -> dict:
    """Render base.html's real style block in this box's chromium at a given canvas."""
    page = tmp_path / f"probe-{width}x{scale}.html"
    page.write_text(
        "<!doctype html><html><head>"
        "<style>" + _style_block() + "</style></head>"
        '<body><div class="conversation-scroll scroll-area"></div><pre id="probe"></pre>'
        "<script>var cs=getComputedStyle(document.documentElement);"
        "var p=getComputedStyle(document.querySelector('.conversation-scroll'));"
        "document.getElementById('probe').textContent=JSON.stringify({"
        "font:cs.fontSize,pad:p.paddingBottom,dpr:window.devicePixelRatio,w:window.innerWidth});"
        "</script></body></html>"
    )
    out = subprocess.run(
        [
            shutil.which("chromium"),
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--force-device-scale-factor={scale}",
            f"--window-size={width},1080",
            "--virtual-time-budget=1500",
            "--dump-dom",
            f"file://{page}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    found = re.search(r'<pre id="probe">(\{.*?\})</pre>', out.stdout)
    assert found, f"the probe did not run: {out.stderr[-400:]}"
    return json.loads(found.group(1))


_BEHAVIOUR_PROBE = """
window.addEventListener('load', function () {
  var pane = document.getElementById('conversation-scroll');
  var out = {};
  function last() { return pane.children[pane.children.length - 1]; }
  function swap() {
    pane.dispatchEvent(new CustomEvent('htmx:afterSwap', {bubbles: true, composed: true, detail: {elt: pane}}));
  }
  // newest row is in view, and how far it sits from the pane's own bottom edge
  pane.scrollTop = pane.scrollHeight;
  out.pinned_clearance_px = +(pane.getBoundingClientRect().bottom - last().getBoundingClientRect().bottom).toFixed(1);
  out.pane_to_viewport_px = +(window.innerHeight - pane.getBoundingClientRect().bottom).toFixed(1);
  // the reader scrolls up into history, a swap arrives: nothing may move
  pane.scrollTop = 0;
  pane.dispatchEvent(new Event('scroll'));
  swap();
  out.after_swap_while_reading_history = pane.scrollTop;
  // they come back down, the next swap follows again (scrollTop clamps at the max, so
  // what is measurable is the gap to the bottom: 0 means it really followed)
  pane.scrollTop = pane.scrollHeight - pane.clientHeight - 20;
  pane.dispatchEvent(new Event('scroll'));
  swap();
  out.gap_after_repining = pane.scrollHeight - pane.scrollTop - pane.clientHeight;
  out.root_font = getComputedStyle(document.documentElement).fontSize;
  out.bound = document.body.dataset.conversationFollowBound || 'no';
  document.getElementById('probe').textContent = JSON.stringify(out);
});
"""


@pytest.mark.skipif(shutil.which("chromium") is None, reason="needs chromium to measure a viewport")
def test_the_follow_rule_and_the_clearance_rendered_in_chromium_at_the_kiosk_geometry(tmp_path):
    """The behaviour, not the text: real layout at 1920 CSS px / dpr 2, with the page's own
    vendored tailwind so the utility classes mean what they mean on the panel.

    The swap is dispatched the way htmx 2.0.4 dispatches it after an SSE message - on the
    swapped element, bubbling, detail.elt set, no detail.target - which is the exact shape
    the shipped handler used to miss.
    """
    pytest.importorskip("jinja2")
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(ROOT / "dashboard" / "templates")))
    env.filters["format_ts"] = lambda ts: "19:31:02"
    turns = [{"role": "user", "content": f"row {i}", "timestamp": float(i), "tool_calls": None} for i in range(40)]
    body = env.get_template("partials/conversation.html").render(
        turns=turns,
        session_id="deadbeefcafe1234",
        live_env="hackday",
        live={"journal": "events.db", "journal_error": None, "liveness": {"alive": True, "age_s": 3.0}},
    )
    page = tmp_path / "attendee.html"
    page.write_text(
        "<!doctype html><html><head>"
        f'<script src="file://{ROOT / "dashboard" / "static" / "vendor" / "tailwind.js"}"></script>'
        "<style>" + _style_block() + "</style></head>"
        '<body class="min-h-screen font-mono text-black"><main class="max-w-6xl mx-auto px-4 pt-20 pb-8">'
        + body
        + '</main><pre id="probe"></pre><script>'
        + _BEHAVIOUR_PROBE
        + "</script></body></html>"
    )
    out = subprocess.run(
        [
            shutil.which("chromium"),
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--force-device-scale-factor=2",
            "--window-size=1920,1080",
            "--virtual-time-budget=4000",
            "--dump-dom",
            f"file://{page}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    found = re.search(r'<pre id="probe">(\{.*?\})</pre>', out.stdout)
    assert found, f"the probe did not run: {out.stderr[-400:]}"
    r = json.loads(found.group(1))
    assert r["bound"] == "true", r
    assert r["after_swap_while_reading_history"] == 0, f"the reader was yanked: {r}"
    assert r["gap_after_repining"] == 0, f"scrolling back down did not re-pin (20 would mean it did not): {r}"
    # 3rem of pane padding at the booth's 20px root, and the row still off the viewport edge
    assert r["pinned_clearance_px"] >= 45, r
    assert r["pane_to_viewport_px"] >= 20, r


@pytest.mark.skipif(shutil.which("chromium") is None, reason="needs chromium to measure a viewport")
def test_the_adaptive_step_measured_in_chromium_at_the_kiosk_geometry(tmp_path):
    """Not a comment asserting a selector fires: the computed root font-size, in the same
    chromium that drives the panel, at the (innerWidth, dpr) pair that framebuffer says the
    booth has - 3840 physical / DPR 2 = 1920 CSS px.

    `--window-size` is in device-independent pixels in headless mode, so 1920,1080 with
    --force-device-scale-factor=2 reproduces the booth's viewport metrics (1920 CSS px wide,
    dpr 2) rather than its window size.
    """
    booth = _chromium_probe(1920, 2, tmp_path)
    assert booth["w"] == 1920 and booth["dpr"] == 2, booth
    laptop = _chromium_probe(1440, 1, tmp_path)
    assert float(booth["font"].rstrip("px")) > float(laptop["font"].rstrip("px")) > 16, (booth, laptop)
    assert float(booth["pad"].rstrip("px")) >= 1.5 * float(booth["font"].rstrip("px")), booth
    # Non-vacuous: the density step must actually be the rule that fired at this geometry.
    # Swap it for `@media (min-width: 2560px)` and the booth falls back to the base clamp,
    # which is the whole bug this guard exists to catch.
    step = re.search(r"@media[^{]*min-resolution[^{]*\{(?P<body>.*?)\}\s*\}", _style_block(), re.S)
    assert step, "no density-gated step block to check"
    cap = re.search(r"font-size:\s*clamp\([^,]+,[^,]+,\s*([\d.]+)rem\)", step.group("body"))
    assert cap, "the density step no longer ends in a rem cap we can check against"
    assert abs(float(booth["font"].rstrip("px")) - 16 * float(cap.group(1))) < 0.5, (booth, cap.group(1))


# --- nothing attendee-facing is a control, or another room's data -----------------


def test_the_attendee_screen_offers_no_control_at_all():
    """The room switch stays an operator action (test_dashboard_panel.py guards the
    environment page); on the attendee page there must be nothing to switch with."""
    for path in ATTENDEE_TEMPLATES:
        text = path.read_text()
        for control in ("<form", "<select", "<input", "<button", "hx-post", "hx-delete", "hx-confirm"):
            assert control not in text, f"{path.name} offers {control}"


def test_the_attendee_screen_names_no_room_and_shows_only_the_live_one():
    """A hard-coded room on a public screen is another room's data leaking into the badge;
    the env chip has to come off the journal's own stamp."""
    text = PARTIAL.read_text()
    assert "hackday" not in text and "home" not in text, f"a room is hard-coded into {PARTIAL.name}"
    assert "{{ live_env }}" in text, "the env badge stopped reporting the resolved room"
    assert "session_id" in text, "and the session it belongs to"


# --- the page still renders -------------------------------------------------------


def test_the_conversation_partial_still_renders_with_the_fix(tmp_path):
    """A Jinja typo on this page is a blank kiosk. Rendered for real, offline, with the
    same filter the server registers."""
    pytest.importorskip("jinja2")
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(ROOT / "dashboard" / "templates")))
    env.filters["format_ts"] = lambda ts: "19:31:02"
    html = env.get_template("partials/conversation.html").render(
        turns=[
            {"role": "user", "content": "how warm is it", "timestamp": 1.0, "tool_calls": None},
            {
                "role": "assistant",
                "content": "Sixty one.",
                "timestamp": 2.0,
                "tool_calls": [{"function": {"name": "read_temp", "arguments": "{}"}}],
            },
        ],
        session_id="deadbeefcafe1234",
        live_env="hackday",
        live={"journal": "events.db", "journal_error": None, "liveness": {"alive": True, "age_s": 3.0}},
    )
    assert 'id="conversation-scroll"' in html and "overflow-y-auto" in html
    assert "how warm is it" in html and "read_temp" in html
    assert "htmx:afterSwap" in html and "PIN_SLACK_PX" in html, "the follow script left the render"
    assert "conversation-scroll" in html, "the padding rule left the render"
    # the newest row is the last one, so scrollTop=scrollHeight does mean "latest message"
    assert html.index("how warm is it") < html.index("Sixty one.")
    assert "hackday" in html  # the badge comes off the stamped room, and only that room

    empty = env.get_template("partials/conversation.html").render(
        turns=[], session_id=None, live_env=None, live={"liveness": {}, "journal_error": None}
    )
    assert "No conversation yet" in empty and "JOURNAL UNAVAILABLE" not in empty
