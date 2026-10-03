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

from . import journal as J

# Tools that change robot/persistent state and must never run on ambiguous
# audio. Physical but transient actions (look/nod) are deliberately NOT here.
STATE_CHANGING = {
    "set_volume",
    "go_to_sleep",
    "learn_face",
    "forget_face",
    "remember",
    "reset_conversation",
    "peekaboo",
    "play_emotion",
}

# Physical actions with no informative payload: execute, don't narrate.
FIRE_AND_FORGET = {
    "look_left",
    "look_right",
    "look_center",
    "nod",
    "shake_head",
    "go_to_sleep",
    "play_emotion",
    "peekaboo",
    "set_volume",
}


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

    def needs_followup(self, name: str, result: dict) -> bool:
        """Whether to request a spoken response after this tool output."""
        if isinstance(result, dict) and result.get("denied"):
            return False  # do not narrate a denial the model shouldn't retry
        return name not in self.fire_and_forget
