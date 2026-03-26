from dataclasses import dataclass, field
import json
import os
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI  # type: ignore[import-not-found]
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed
from tools import (
    Tool,
)


@dataclass
class EventText:
    text: str
    type: str = "text"


@dataclass
class EventInputJson:
    partial_json: str
    type: str = "input_json"


@dataclass
class EventToolUse:
    tool: Tool
    type: str = "tool_use"


@dataclass
class EventToolResult:
    tool: Tool
    result: str
    type: str = "tool_result"


AgentEvent = EventText | EventInputJson | EventToolUse | EventToolResult


@dataclass
class Agent:
    system_prompt: str
    model: str
    tools: list[Tool]
    messages: list[dict[str, Any]] = field(default_factory=list)
    avaialble_tools: list[dict[str, Any]] = field(default_factory=list)
    _client: AsyncOpenAI = field(init=False, repr=False)

    def __post_init__(self):
        # Build DeepSeek/OpenAI-compatible client.
        api_key = (
            os.getenv("DEEPSEEK_API_KEY")
        )
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # Convert pydantic tool models to OpenAI tool schema.
        self.avaialble_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": tool.__doc__ or "",
                    "parameters": tool.model_json_schema(),
                },
            }  # type: ignore[list-item]
            for tool in self.tools
        ]

    def add_user_message(self, message: str):
        self.messages.append({"role": "user", "content": message})

    async def _call_llm(self) -> tuple[str, list[dict[str, Any]]]:
        """
        Returns (assistant_text, tool_calls).

        tool_calls entries are normalized to OpenAI format:
        {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
        """
        request_messages = [{"role": "system", "content": self.system_prompt}] + self.messages

        # NOTE: DeepSeek supports OpenAI-compatible /v1/chat/completions.
        # We intentionally use non-streaming here to keep tool parsing simple.
        resp: Any = None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3), wait=wait_fixed(3)
        ):
            with attempt:
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    messages=request_messages,  # type: ignore[arg-type]
                    tools=self.avaialble_tools,  # type: ignore[arg-type]
                    tool_choice="auto",
                )
        assert resp is not None

        message = resp.choices[0].message
        content = message.content or ""

        raw_tool_calls = getattr(message, "tool_calls", None) or []
        normalized_tool_calls: list[dict[str, Any]] = []
        for i, tc in enumerate(raw_tool_calls):
            # openai objects are generally attribute-friendly, but we normalize explicitly.
            normalized_tool_calls.append(
                {
                    "id": getattr(tc, "id", None) or f"tool_call_{i}",
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )
        return content, normalized_tool_calls

    async def agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        while True:
            assistant_text, tool_calls = await self._call_llm()
            if assistant_text:
                yield EventText(text=assistant_text)

            # Always append the assistant message, even if tool calls exist.
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_text}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            self.messages.append(assistant_msg)

            if not tool_calls:
                return

            # Execute each requested tool call and feed results back to the model.
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args_raw = tool_call["function"]["arguments"] or "{}"

                # Parse arguments; DeepSeek follows OpenAI's "arguments" as a JSON string.
                try:
                    tool_args = json.loads(tool_args_raw)
                except json.JSONDecodeError:
                    tool_args = {"_raw": tool_args_raw}

                for tool_cls in self.tools:
                    if tool_cls.__name__ != tool_name:
                        continue

                    t = tool_cls.model_validate(tool_args)
                    yield EventToolUse(tool=t)
                    result = await t()
                    yield EventToolResult(tool=t, result=result)

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result,
                        }
                    )
                    break

    async def run(self) -> AsyncGenerator[AgentEvent, None]:
        async for x in self.agentic_loop():
            yield x