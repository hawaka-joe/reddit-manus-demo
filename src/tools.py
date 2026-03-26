import asyncio
from pathlib import Path
from typing import Awaitable, Callable, Type

from clients import docker_client
from docker import errors as docker_errors
from pydantic import BaseModel, Field


class Tool(BaseModel):
    async def __call__(self) -> str:
        raise NotImplementedError


class ToolRunCommandInDevContainer(Tool):
    """Run a command in the dev container you have at your disposal to test and run code.
    The command will run in the container and the output will be returned.
    The container is a Python development container with Python 3.12 installed.
    It has the port 8888 exposed to the host in case the user asks you to run an http server.
    """

    command: str

    def _run(self) -> str:
        if docker_client is None:
            raise RuntimeError(
                "Docker client is not available (docker socket not found). "
                "Cannot run commands in the dev container."
            )
        container = docker_client.containers.get("python-dev")
        exec_command = f"bash -c '{self.command}'"

        try:
            res = container.exec_run(exec_command)
            output = res.output.decode("utf-8")
        except Exception as e:
            output = f"""Error: {e}\n here is how I run your command: {exec_command}"""

        return output

    async def __call__(self) -> str:
        return await asyncio.to_thread(self._run)


class ToolUpsertFile(Tool):
    """Create a file in the dev container you have at your disposal to test and run code.
    If the file exsits, it will be updated, otherwise it will be created.
    """

    file_path: str = Field(description="The path to the file to create or update")
    content: str = Field(description="The content of the file")

    def _run(self) -> str:
        if docker_client is None:
            raise RuntimeError(
                "Docker client is not available (docker socket not found). "
                "Cannot upsert files in the dev container."
            )
        container = docker_client.containers.get("python-dev")

        # Command to write the file using cat and stdin
        cmd = f'sh -c "cat > {self.file_path}"'

        # Execute the command with stdin enabled
        _, socket = container.exec_run(
            cmd, stdin=True, stdout=True, stderr=True, stream=False, socket=True
        )
        socket._sock.sendall((self.content + "\n").encode("utf-8"))
        socket._sock.close()

        return "File written successfully"

    async def __call__(self) -> str:
        return await asyncio.to_thread(self._run)


class ToolDisplayArtifact(Tool):
    """Use this tool to display important result for the user, such as code or file content.
    It will display on a dedicated panel on the right of the interface.
    The content must be the entire content to display, or an empty string if you don't want to display anything.
    """

    artifact: str

    def _run(self) -> str:
        return "Displayed successfully"

    async def __call__(self) -> str:
        return await asyncio.to_thread(self._run)


def start_python_dev_container(container_name: str) -> None:
    """Start a Python development container"""
    if docker_client is None:
        raise RuntimeError(
            "Docker client is not available (docker socket not reachable). "
            "Please ensure Docker Desktop/engine is running and the current user "
            "can access the Docker socket."
        )
    try:
        existing_container = docker_client.containers.get(container_name)
        if existing_container.status == "running":
            existing_container.kill()
        existing_container.remove()
    except docker_errors.NotFound:
        pass

    volume_path = str(Path(".scratchpad").absolute())

    docker_client.containers.run(
        "python:3.12",
        detach=True,
        name=container_name,
        ports={"8888/tcp": 8888},
        tty=True,
        stdin_open=True,
        working_dir="/app",
        # volumes={
        #     volume_path: {
        #         "bind": "/app",
        #         "mode": "rw",
        #     }
        # },
        command="bash -c 'mkdir -p /app && tail -f /dev/null'",
    )


def create_tool_interact_with_user(
    prompter: Callable[[str], Awaitable[str]],
) -> Type[Tool]:
    class ToolInteractWithUser(Tool):
        """This tool will ask the user to clarify their request, provide your query and it will be asked to the user
        you'll get the answer. Make sure that the content in display is properly markdowned, for instance if you display code, use the triple backticks to display it properly with the language specified for highlighting.
        """

        query: str = Field(description="The query to ask the user")
        display: str = Field(
            description="The interface has a pannel on the right to diaplay artifacts why you asks your query, use this field to display the artifacts, for instance code or file content, you must give the entire content to dispplay, or use an empty string if you don't want to display anything."
        )

        async def __call__(self) -> str:
            res = await prompter(self.query)
            return res

    return ToolInteractWithUser