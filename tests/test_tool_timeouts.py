"""Per-tool time budgets, pinned against the code they describe.

Tonight's trigger: the operator asked for a snapshot. `take_snapshot` started, the
flat 12 s limit fired, the model was told it failed, and it retried - twice. The
"failed" call was still running (asyncio.to_thread cannot be cancelled), so all three
queued behind each other on one camera handle: 54 seconds, three duplicate captures,
and a robot telling a guest it "is having trouble" with a tool that had succeeded.
"""

from __future__ import annotations

import ast
from pathlib import Path

from shell import camera as CAM
from shell.realtime import client as C
from shell.tools import (
    CAMERA_TIMEOUT_S,
    CAMERA_TOOLS,
    TOOL_TIMEOUT_S,
    VISION_ROUND_TRIP_S,
    ToolRouter,
    tool_timeout,
)

ROOT = Path(__file__).resolve().parent.parent


def _frame_callers() -> set[str]:
    """Names of the *innermost* functions in shell/robot_tools.py that touch the camera.

    The tools are closures inside a `make_*_handlers` factory, so attributing a
    `_frame()` call to the enclosing factory would name a function that is not a tool
    and put every tool in the set.
    """
    tree = ast.parse((ROOT / "shell" / "robot_tools.py").read_text())
    found: set[str] = set()

    def visit(fn: ast.FunctionDef) -> None:
        nested = [n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n is not fn]
        deeper = {id(c) for n in nested for c in ast.walk(n) if isinstance(c, ast.Call)}
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_frame" and id(call) not in deeper:
                found.add(fn.name)
        for n in nested:
            visit(n)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            visit(node)
    return found - {"_frame"}


def test_every_camera_tool_has_the_longer_budget():
    """The set must track the code. A new tool that grabs a frame and is left out of
    CAMERA_TOOLS inherits 12 s and reproduces tonight's pile-up quietly."""
    grabbers = _frame_callers()
    assert grabbers, "no _frame() callers found - did the camera tools move?"
    missing = grabbers - CAMERA_TOOLS
    assert not missing, f"these tools use the camera but keep the 12 s budget: {sorted(missing)}"
    ghost = CAMERA_TOOLS - grabbers
    assert not ghost, f"CAMERA_TOOLS names tools that no longer grab a frame: {sorted(ghost)}"


def test_camera_tools_get_the_measured_budget_and_everything_else_does_not():
    assert tool_timeout("take_snapshot") == CAMERA_TIMEOUT_S
    assert tool_timeout("describe_scene") == CAMERA_TIMEOUT_S
    assert tool_timeout("get_time") == TOOL_TIMEOUT_S
    assert tool_timeout("web_search") == TOOL_TIMEOUT_S  # leaves the box; its own problem
    assert CAMERA_TIMEOUT_S > TOOL_TIMEOUT_S


def test_the_camera_budget_covers_a_bounded_warmup_and_a_vision_round_trip():
    """The budget and the warm-up are one contract: 35 s is the ceiling for the whole
    call, so it has to clear the worst the device can cost (the deadline plus one
    uninterruptible V4L2 select timeout) plus one measured vision round trip. It does
    not have to absorb a 5 x select() warm-up any more, which is what made the old 12 s
    limit look too small; re-widening the warm-up past this goes red here, not at the
    booth."""
    worst_device = CAM.warmup_worst_case_s()
    assert worst_device == CAM.warmup_deadline_s() + CAM.V4L2_SELECT_TIMEOUT_S
    assert CAMERA_TIMEOUT_S >= worst_device + VISION_ROUND_TRIP_S, (
        f"camera tools can cost {worst_device + VISION_ROUND_TRIP_S:.1f}s, the budget is {CAMERA_TIMEOUT_S}s"
    )
    # Margin for jitter, NOT a budgeted retry. The original version of this assertion
    # demanded room for a second vision round trip, which is the thinking that produced
    # tonight's 54-second snapshot: budgeting for a retry designs the pile-in, because
    # the "timed out" call is still holding the camera while the retry queues behind it
    # (asyncio.to_thread is not cancellable). The anti-retry protection is the timeout
    # message ("still running, do not call it again", client.py TIMEOUT_MESSAGE) and the
    # bounded warm-up, not a bigger number.
    assert CAMERA_TIMEOUT_S - worst_device - VISION_ROUND_TRIP_S >= 5.0, (
        f"only {CAMERA_TIMEOUT_S - worst_device - VISION_ROUND_TRIP_S:.1f}s of slack for jitter; raise the "
        "budget or shrink the warm-up - but do not raise it to fit a second attempt"
    )


def test_the_deadline_is_the_bound_now_not_the_tool_timeout():
    """The number that protects the guest is the warm-up deadline, and it has to be
    small enough that the answer arrives while the question is still relevant."""
    assert CAM.WARMUP_DEADLINE_S <= TOOL_TIMEOUT_S
    assert CAM.warmup_worst_case_s() < CAMERA_TIMEOUT_S, "a dead camera must fail inside the budget, not at it"


def test_the_router_owns_the_number_not_the_transport():
    """The client asks the router, because the router is what knows the tools."""
    router = ToolRouter([], {}, None)
    assert router.timeout_for("take_snapshot") == tool_timeout("take_snapshot")
    assert router.timeout_for("nod") == TOOL_TIMEOUT_S


def test_the_timeout_message_does_not_invite_a_retry():
    msg = C.TIMEOUT_MESSAGE.format(name="take_snapshot")
    assert "Do not call it again" in msg, "a timeout that reads like a failure makes the model retry and pile up"
    assert "took too long" not in msg, "the old wording implied the work had stopped"
    assert "still running" in msg, "the honest statement: to_thread is not cancellable"


def test_the_timeout_is_not_copied_into_the_transport_layer():
    """Two definitions of the same number is how 12 s came to be wrong for a camera."""
    src = (ROOT / "shell" / "realtime" / "client.py").read_text()
    assert "TOOL_TIMEOUT_S = " not in src, "client.py re-declared the timeout instead of importing it"
    assert "from ..tools import TOOL_TIMEOUT_S" in src
