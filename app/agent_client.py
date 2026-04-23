import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from .config import Config
from .session_store import SessionStore

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful robot assistant controlling a Reachy Mini desk robot.
Keep responses short: 1 to 3 sentences, suitable for spoken output.
Use tools when the user asks for physical actions or sensor readings.
Confirm physical actions briefly. If a tool fails, say so and suggest retrying.
Be friendly, concise, and natural.\
"""

MEMORY_PATH = Path("data/memory.md")

MAX_TOOL_ROUNDS = 5


class AgentClient:
    """Thin wrapper around an OpenAI-compatible chat API with tool calling.

    Talks to llama.cpp (or any OpenAI-compatible server). Designed to be
    replaceable with Hermes or another agent framework later.
    """

    def __init__(self, config: Config, tools: list[dict[str, Any]]):
        self.client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
        self.model = config.llm_model
        self.max_tokens = config.llm_max_tokens
        self.tools = tools or []
        self.tool_handlers: dict[str, Callable] = {}
        self.session = SessionStore()
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self._build_system_prompt()}]

    def register_handlers(self, handlers: dict[str, Callable]):
        self.tool_handlers.update(handlers)

    def send(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        self.session.log_turn("user", content=text)

        for _round in range(MAX_TOOL_ROUNDS):
            start = time.monotonic()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools if self.tools else None,
                max_tokens=self.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            elapsed = time.monotonic() - start
            choice = response.choices[0]
            msg = choice.message

            self.messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                log.info("LLM %.2fs: %r", elapsed, msg.content)
                self.session.log_turn("assistant", content=msg.content)
                return msg.content or ""

            for tc in msg.tool_calls:
                result = self._execute_tool(tc.function.name, tc.function.arguments)
                log.info("Tool %s -> %s", tc.function.name, json.dumps(result))
                self.session.log_turn("tool", content=json.dumps(result), tool_call_id=tc.id)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )

        return self.messages[-1].get("content", "")

    def _execute_tool(self, name: str, arguments: str) -> dict:
        handler = self.tool_handlers.get(name)
        if not handler:
            return {"error": f"Unknown tool: {name}"}
        try:
            args = json.loads(arguments) if arguments else {}
            return handler(**args)
        except Exception as e:
            log.exception("Tool %s failed", name)
            return {"error": str(e)}

    def reset(self):
        self.messages = [{"role": "system", "content": self._build_system_prompt()}]

    def _build_system_prompt(self) -> str:
        parts = [SYSTEM_PROMPT]
        if MEMORY_PATH.exists():
            memory = MEMORY_PATH.read_text().strip()
            if memory:
                parts.append(f"\n\n## Things you remember\n{memory}")
        return "\n".join(parts)

    def save_memory(self, fact: str):
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MEMORY_PATH.open("a") as f:
            f.write(f"- {fact}\n")
        log.info("Memory saved: %s", fact)
