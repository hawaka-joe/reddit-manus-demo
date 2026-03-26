"""
This file defines the core Agent class, 
which orchestrates the entire agentic loop. 
It's responsible for managing the agent's state, 
interacting with the language model, and executing tools.
"""

from dataclasses import dataclass


@dataclass
class Agent:
    system_prompt: str
    model: ModelParam
    tools: list[Tool]
    messages: list[MessageParam] = field(default_factory=list)
    avaialble_tools: list[ToolUnionParam] = field(default_factory=list)

    def __post_init__(self):
        self.avaialble_tools = [
            {
                "name": tool.__name__,
                "description": tool.__doc__ or "",
                "input_schema": tool.model_json_schema(),
            }
            for tool in self.tools
        ]

    def add_user_message(self, message: str):
        self.messages.append(MessageParam(role="user", content=message))

    async def agentic_loop(
        self,
    ) -> AsyncGenerator[AgentEvent, None]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3), wait=wait_fixed(3)
        ):
            with attempt:
                async with anthropic_client.messages.stream(
                    max_tokens=8000,
                    messages=self.messages,
                    model=self.model,
                    tools=self.avaialble_tools,
                    system=self.system_prompt,
                ) as stream:

                    async for event in stream:
                        if event.type == "text":
                            event.text
                            yield EventText(text=event.text)
                        if event.type == "input_json":
                            yield EventInputJson(partial_json=event.partial_json)
                            event.partial_json
                            event.snapshot
                        if event.type == "thinking":
                            ...
                        elif event.type == "content_block_stop":
                            ...
                        accumulated = await stream.get_final_message()
                        for content in accumulated.content:
                            if content.type == "tool_use":
                                tool_name = content.name
                                tool_args = content.input

                                for tool in self.tools:
                                    if tool.__name__ == tool_name:
                                        t = tool.model_validate(tool_args)
                                        yield EventToolUse(tool=t)
                                        result = await t()
                                        yield EventToolResult(tool=t, result=result)
                                        self.messages.append(
                                            MessageParam(
                                                role="user",
                                                content=[
                                                    ToolResultBlockParam(
                                                        type="tool_result",
                                                        tool_use_id=content.id,
                                                        content=result,
                                                    )
                                                ],
                                            )
                                        )
                        if accumulated.stop_reason == "tool_use":
                            async for e in self.agentic_loop():
                                yield e
