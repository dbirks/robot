"""Tool routing for the Realtime function-call lifecycle.

Handlers are plain callables returning JSON-serializable dicts that never
raise (the app/robot_tools contract); they are injected here so this module
stays independent of the legacy app package and unit-testable. The router
adds three things the old direct dispatch never had:

- per-response dedupe of identical (name, arguments) pairs (the old bug
  where look_left fired 3x in one response)
- an attention-confidence gate: state-changing tools need a live lease /
  high directedness, conversational replies do not (ADR 0003)
- fire-and-forget vs data-returning classification, so a nod does not cost
  an extra LLM round-trip
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from . import journal as J

# Tools that change robot/persistent state and must never run on ambiguous
# audio. Physical but transient actions (look/nod) are deliberately NOT here.
# Per-tool time budgets. A hung tool must not hold a reply forever, but one flat
# number for every tool is its own bug: measured on the robot box 2026-10-12, a cold
# camera open plus its 5 warm-up frames is 2.7-3.0 s and a llama-server vision round
# trip ran about 10 s with the TTS and STT models resident - each comfortably inside
# 12 s alone, and not at all when two camera tools queue behind one V4L2 handle. That
# is how a 3-second snapshot became a 54-second turnaround with three duplicate
# captures in it: the limit fired, the model was told it failed, it retried, and the
# "failed" call was still running the whole time (asyncio.to_thread cannot be
# cancelled), so every retry queued another capture behind it.
TOOL_TIMEOUT_S = 12.0
CAMERA_TIMEOUT_S = 35.0
# Every tool that reaches for the camera - pinned against the code by
# tests/test_tool_timeouts.py, so a new _frame() caller cannot be left out.
CAMERA_TOOLS = {"take_snapshot", "describe_scene", "identify_face", "learn_face"}


def tool_timeout(name: str) -> float:
    return CAMERA_TIMEOUT_S if name in CAMERA_TOOLS else TOOL_TIMEOUT_S


STATE_CHANGING = {
    "set_volume",
    "go_to_sleep",
    "learn_face",
    "forget_face",
    "remember",
    "forget_memory",
    "reset_conversation",
    "peekaboo",
    "play_emotion",
}

# Tools whose result gets no spoken follow-up. Empty since 2026-10-11: the
# user wants to hear progress after every call ("Turning left.", "There you
# go."), not silence after physical actions. Kept as a hook.
FIRE_AND_FORGET: set[str] = set()


def tool_name(schema: dict) -> str:
    """The function name of an OpenAI-tool-call schema dict."""
    return str(schema.get("function", {}).get("name", ""))


def apply_tool_policy(
    tools: list[dict],
    handlers: dict,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> tuple[list[dict], dict]:
    """Environment tool subset (docs/environments.md A3): empty `include` keeps
    everything the shell knows about, `exclude` always wins.

    Schema AND handler both go away, so a hallucinated call to an excluded tool
    comes back as an unknown tool and gets journaled as one — never as a side
    effect. Names must match the schema exactly; a name the shell has never
    heard of is a no-op, so an env file from a newer main still boots.
    """
    keep = {str(name).strip() for name in include if str(name).strip()}
    drop = {str(name).strip() for name in exclude if str(name).strip()}
    selected = [tool for tool in tools if (not keep or tool_name(tool) in keep) and tool_name(tool) not in drop]
    allowed = {tool_name(tool) for tool in selected}
    return selected, {name: handler for name, handler in handlers.items() if name in allowed}


class ToolRouter:
    def __init__(
        self,
        tools: list[dict],
        handlers: dict,
        journal,
        *,
        state_changing: set[str] | None = None,
        fire_and_forget: set[str] | None = None,
        confidence_fn=None,  # callable(name: str) -> float in [0,1]
        confidence_threshold: float = 0.6,
    ) -> None:
        self.tools = tools
        self.handlers = handlers
        self.journal = journal
        self.state_changing = state_changing or STATE_CHANGING
        self.fire_and_forget = fire_and_forget or FIRE_AND_FORGET
        self.confidence_fn = confidence_fn or (lambda _name: 1.0)
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def dedupe(calls: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        seen = set()
        out = []
        for name, args in calls:
            key = hashlib.md5(json.dumps([name, args], sort_keys=True, default=str).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append((name, args))
        return out

    def execute(self, name: str, args: dict) -> dict:
        """Never raises; every outcome is journaled."""
        handler = self.handlers.get(name)
        if handler is None:
            self.journal.write(J.TOOL_FAILED, tool=name, error="unknown tool")
            return {"ok": False, "error": f"Unknown tool: {name}"}

        if name in self.state_changing:
            conf = float(self.confidence_fn(name))
            if conf < self.confidence_threshold:
                self.journal.write(
                    J.ATTENTION_TOOL_DENIED,
                    tool=name,
                    confidence=conf,
                    threshold=self.confidence_threshold,
                )
                return {
                    "ok": False,
                    "denied": "attention_confidence_too_low",
                    "hint": "Do not retry. The speaker may not be addressing you.",
                }

        self.journal.write(J.TOOL_STARTED, tool=name, args=args)
        try:
            result = handler(**args)
        except Exception as e:  # handlers promise not to raise; belt AND braces
            self.journal.write(J.TOOL_FAILED, tool=name, error=repr(e))
            return {"ok": False, "error": str(e)}
        if isinstance(result, dict) and result.get("ok") is False:
            self.journal.write(J.TOOL_FAILED, tool=name, error=result.get("error"))
        else:
            self.journal.write(J.TOOL_COMPLETED, tool=name, result=result)
        return result if isinstance(result, dict) else {"ok": True, "result": result}

    def timeout_for(self, name: str) -> float:
        """The budget `RealtimeClient._run_tool` gives this call. A method rather
        than a module constant so the router that knows the tools owns the number,
        and so a fake router in a test can say otherwise."""
        return tool_timeout(name)

    def needs_followup(self, name: str, result: dict) -> bool:
        """Whether to request a spoken response after this tool output."""
        if isinstance(result, dict) and result.get("denied"):
            return False  # do not narrate a denial the model shouldn't retry
        return name not in self.fire_and_forget
