import os

import docker  # type: ignore[import-not-found]
from dotenv import load_dotenv  # type: ignore[import-not-found]

load_dotenv()

# Docker socket may not exist in every environment (e.g. local lint/test runs).
# We keep this import-safe so that the agent can still be used with DeepSeek.
try:
    # docker-py expects unix socket URLs in the form "unix:///path/to/socket".
    # If DOCKER_HOST is set, prefer it (e.g. tcp://... or unix://...).
    docker_host = os.getenv("DOCKER_HOST")
    if docker_host:
        base_url = docker_host
    else:
        socket_path = os.getenv("DOCKER_SOCKET_PATH", "/var/run/docker.sock")
        # If user passed a raw socket path, convert it to unix:// URL.
        base_url = (
            socket_path if socket_path.startswith("unix://") else f"unix://{socket_path}"
        )

    docker_client = docker.DockerClient(base_url=base_url)
except Exception:  # pragma: no cover
    docker_client = None