import asyncio
import os
from typing import Any

from agent import (
    Agent,
    EventText,
    EventToolResult,
    EventToolUse,
)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.status import Status
from tools import (
    ToolRunCommandInDevContainer,
    ToolUpsertFile,
    create_tool_interact_with_user,
    start_python_dev_container,
)


async def get_prompt_from_user(query: str) -> str:
    print()
    res = Prompt.ask(
        f"[italic yellow]{query}[/italic yellow]\n[bold red]User answer[/bold red]"
    )
    print()
    return res


ToolInteractWithUser = create_tool_interact_with_user(get_prompt_from_user)
tools = [
    ToolRunCommandInDevContainer,
    ToolUpsertFile,
    ToolInteractWithUser,
]


async def main():
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    agent = Agent(
        model=deepseek_model,
        tools=tools,
        system_prompt="""
        You are an AI assistant specialized in writing high-quality code. Your goal is to assist users with their coding requests efficiently and accurately.

        Follow these steps to fulfill the user's request:

        1. Analyze the Request:
        <request_analysis>
        - Carefully read and understand the user's query.
        - Break down the query into its main components:
        a. Identify the programming language or framework required.
        b. List the specific functionalities or features requested.
        c. Note any constraints or specific requirements mentioned.
        - Determine if any clarification is needed.
        - Summarize the main coding task or problem to be solved.
        </request_analysis>

        2. Clarification (if needed):
        If the user's request is unclear or lacks necessary details, use the clarify tool to ask for more information. For example:
        <clarify>
        Could you please provide more details about [specific aspect of the request]? This will help me better understand your requirements and provide a more accurate solution.
        </clarify>

        3. Write Tests:
        <test_design>
        - Based on the user's requirements, design appropriate test cases:
        a. Identify the main functionalities to be tested.
        b. Create test cases for normal scenarios.
        c. Design edge cases to test boundary conditions.
        d. Consider potential error scenarios and create tests for them.
        - Choose a suitable testing framework for the language/platform.
        - Write the test code, ensuring each test is clear and focused.
        </test_design>

        Present the tests to the user for validation:
        <test_code>
        [Insert test code here]
        </test_code>

        Ask the user to validate the tests:
        <user_validation>
        Please review the above test cases. Do they adequately cover your requirements? If not, please let me know what changes or additions are needed.
        </user_validation>

        4. Code Implementation:
        <implementation_strategy>
        - Design the solution based on the validated tests:
        a. Break down the problem into smaller, manageable components.
        b. Outline the main functions or classes needed.
        c. Plan the data structures and algorithms to be used.
        - Write clean, efficient, and well-documented code:
        a. Implement each component step by step.
        b. Add clear comments explaining complex logic.
        c. Use meaningful variable and function names.
        - Consider best practices and coding standards for the specific language or framework being used.
        - Implement error handling and input validation where necessary.
        </implementation_strategy>

        5. Run and Iterate:
        <testing_and_refinement>
        - Execute the code and run it against the test cases.
        - For each failing test:
        a. Identify the specific issue causing the failure.
        b. Debug the code to find the root cause.
        c. Make necessary corrections to address the issue.
        - Repeat this process until all tests pass.
        - Refactor the code if needed to improve efficiency or readability.
        </testing_and_refinement>

        6. File Operations:
        If you need to create new files or edit existing ones, always use the ToolUpsertFile tool. For example:
        <tool_usage>
        ToolUpsertFile("filename.ext", "file contents")
        </tool_usage>

        7. Long-running Commands:
        For commands that may take a while to complete, use tmux to run them in the background. 
        You should never ever run long-running commands in the main thread, as it will block the agent and prevent it from responding to the user. Example of long-running command:
        - `python3 -m http.server 8888`
        - `uvicorn main:app --host 0.0.0.0 --port 8888`  

        Here's the process:

        <tmux_setup>
        - Check if tmux is installed.
        - If not, install it using in two steps: `apt update && apt install -y tmux`
        - Use tmux to start a new session for the long-running command.
        </tmux_setup>

        Example tmux usage:
        <tmux_command>
        tmux new-session -d -s mysession "python3 -m http.server 8888"
        </tmux_command>

        8. Present Results:
        Show the final code, test results, and any relevant output to the user. Explain your solution and any important considerations.

        Throughout this process, maintain clear communication with the user, explaining your thoughts and actions. If you encounter any issues or need additional information, don't hesitate to ask for clarification.

        Remember to adhere to best practices in software development, including writing clean, maintainable code, proper error handling, and following language-specific conventions.
        """,
    )

    start_python_dev_container("python-dev")
    console = Console()

    status = Status("")

    while True:
        console.print(Rule("[bold blue]User[/bold blue]"))
        query = input("\nUser: ").strip()
        agent.add_user_message(
            query,
        )
        console.print(Rule("[bold blue]Agentic Loop[/bold blue]"))
        async for x in agent.run():
            match x:
                case EventText(text=t):
                    print(t, end="", flush=True)
                case EventToolUse(tool=t):
                    panel = Panel("")
                    match t:
                        case ToolRunCommandInDevContainer(command=cmd):
                            status.update(f"Tool: {t}")
                            panel = Panel(
                                f"[bold cyan]{t}[/bold cyan]\n\n"
                                + "\n".join(
                                    f"[yellow]{k}:[/yellow] {v}"
                                    for k, v in t.model_dump().items()
                                ),
                                title="Tool Call: ToolRunCommandInDevContainer",
                                border_style="green",
                            )
                            status.start()
                        case ToolUpsertFile(file_path=file_path, content=content):
                            status.update(f"Tool: {t}")
                            md = ""
                            md += f"## File path \n {file_path} \n"
                            md += f"## Content \n ```\n{content}\n``` \n"
                            panel = Panel(
                                Markdown(md),
                                title="Tool Call: ToolUpsertFile",
                                border_style="green",
                            )
                            status.start()
                        case _ if isinstance(t, ToolInteractWithUser):
                            md = ""
                            t_any: Any = t
                            q = "\n".join([f"> {q}" for q in t_any.query.splitlines()])
                            md += f"## Query \n {q} \n"
                            md += f"## Display \n {t_any.display} \n"
                            panel = Panel(
                                Markdown(md),
                                title="Tool Call: ToolInteractWithUser",
                                border_style="green",
                            )
                        case _:
                            print(t)
                    print()
                    status.stop()
                    print()
                    console.print(panel)
                    print()
                case EventToolResult(result=r):
                    pannel = Panel(
                        f"[bold green]{r}[/bold green]",
                        title="Tool Result",
                        border_style="green",
                    )
                    console.print(pannel)
        print()


if __name__ == "__main__":
    asyncio.run(main())