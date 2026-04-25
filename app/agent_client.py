import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Generator

from openai import OpenAI

from .config import Config
from .session_store import SessionStore

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful robot assistant controlling a Reachy Mini desk robot.
Keep responses short: 1 to 3 sentences, suitable for spoken output.
Use tools when the user asks for physical actions or sensor readings.
Confirm physical actions briefly. If a tool fails, say so and suggest retrying.
Be friendly, concise, and natural.
When asked about people (who is here, who do you see, do you recognize me, etc.), \
ALWAYS use identify_face first — it's fast. Only use describe_scene for non-people \
questions about the environment (what's on the table, what color is the wall, etc.).\
"""

MEMORY_PATH = Path("data/memory.md")

MAX_TOOL_ROUNDS = 5

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


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

    def _trim_context(self):
        """Keep conversation under context limit by dropping oldest turns."""
        max_messages = 20
        if len(self.messages) > max_messages:
            system = self.messages[0]
            self.messages = [system] + self.messages[-(max_messages - 1) :]
            log.info("Trimmed conversation to %d messages", len(self.messages))

    def send(self, text: str) -> str:
        self._trim_context()
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
                result = msg.content or ""
                if getattr(self, "_pending_reset", False):
                    self._pending_reset = False
                    self.reset()
                    log.info("Conversation reset")
                return result

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

        log.warning("Max tool rounds (%d) reached", MAX_TOOL_ROUNDS)
        return "Sorry, I got a bit confused there. Could you try asking again?"

    def send_streaming(self, text: str) -> Generator[str, None, None]:
        """Stream LLM response, yielding complete sentences as they form."""
        self._trim_context()
        self.messages.append({"role": "user", "content": text})
        self.session.log_turn("user", content=text)

        for _round in range(MAX_TOOL_ROUNDS):
            start = time.monotonic()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools if self.tools else None,
                max_tokens=self.max_tokens,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            content_buf = ""
            full_content = ""
            tool_calls: dict[int, dict] = {}

            for chunk in response:
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    content_buf += delta.content
                    full_content += delta.content
                    while True:
                        m = _SENTENCE_END.search(content_buf)
                        if not m:
                            break
                        sentence = content_buf[: m.end()].strip()
                        content_buf = content_buf[m.end() :]
                        if sentence:
                            yield sentence

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls:
                            tool_calls[idx] = {"id": "", "name": "", "args": ""}
                        if tc.id:
                            tool_calls[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls[idx]["name"] += tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls[idx]["args"] += tc.function.arguments

            if content_buf.strip():
                yield content_buf.strip()
                full_content += ""

            elapsed = time.monotonic() - start

            if not tool_calls:
                log.info("LLM %.2fs (streamed): %r", elapsed, full_content)
                self.messages.append({"role": "assistant", "content": full_content})
                self.session.log_turn("assistant", content=full_content)
                if getattr(self, "_pending_reset", False):
                    self._pending_reset = False
                    self.reset()
                    log.info("Conversation reset")
                return

            self.messages.append(
                {
                    "role": "assistant",
                    "content": full_content if full_content else None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["args"]},
                        }
                        for tc in tool_calls.values()
                    ],
                }
            )

            for tc in tool_calls.values():
                result = self._execute_tool(tc["name"], tc["args"])
                log.info("Tool %s -> %s", tc["name"], json.dumps(result))
                self.session.log_turn("tool", content=json.dumps(result), tool_call_id=tc["id"])
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)}
                )

        log.warning("Max tool rounds (%d) reached (streaming)", MAX_TOOL_ROUNDS)
        yield "Sorry, I got a bit confused there. Could you try asking again?"

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
