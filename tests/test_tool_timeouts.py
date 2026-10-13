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

from shell.realtime import client as C
from shell.tools import CAMERA_TIMEOUT_S, CAMERA_TOOLS, TOOL_TIMEOUT_S, ToolRouter, tool_timeout

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
