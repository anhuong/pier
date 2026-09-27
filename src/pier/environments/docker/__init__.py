import json
from pathlib import Path

from pier.models.trial.config import ServiceVolumeConfig

# Shared compose file paths used by both local Docker and Daytona DinD environments.
COMPOSE_DIR = Path(__file__).parent
COMPOSE_BASE_PATH = COMPOSE_DIR / "docker-compose-base.yaml"
COMPOSE_BUILD_PATH = COMPOSE_DIR / "docker-compose-build.yaml"
COMPOSE_PREBUILT_PATH = COMPOSE_DIR / "docker-compose-prebuilt.yaml"
COMPOSE_NO_NETWORK_PATH = COMPOSE_DIR / "docker-compose-no-network.yaml"
# Host-network override (mirrors Harbor's docker-compose-host-network.yaml).
# Needed under rootless podman without a user D-Bus session: the default compose
# bridge network invokes netavark/aardvark-dns, which fails ("Failed to connect
# to bus: No such file or directory") and the container never starts. network_mode
# host sidesteps it. Applied only when internet is allowed and HARBOR_DOCKER_HOST_NETWORK
# (or PIER_DOCKER_HOST_NETWORK) is truthy.
COMPOSE_HOST_NETWORK_PATH = COMPOSE_DIR / "docker-compose-host-network.yaml"
COMPOSE_WINDOWS_KEEPALIVE_PATH = COMPOSE_DIR / "docker-compose-windows-keepalive.yaml"
RESOURCES_COMPOSE_NAME = "docker-compose-resources.json"


def write_mounts_compose_file(path: Path, mounts: list[ServiceVolumeConfig]) -> Path:
    compose = {"services": {"main": {"volumes": list(mounts)}}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def write_resources_compose_file(
    path: Path,
    *,
    cpu_request: int | None = None,
    cpu_limit: int | None = None,
    memory_request_mb: int | None = None,
    memory_limit_mb: int | None = None,
) -> Path:
    resources: dict[str, dict[str, str]] = {}
    limits: dict[str, str] = {}
    reservations: dict[str, str] = {}

    if cpu_limit is not None:
        limits["cpus"] = str(cpu_limit)
    if memory_limit_mb is not None:
        limits["memory"] = f"{memory_limit_mb}M"
    if cpu_request is not None:
        reservations["cpus"] = str(cpu_request)
    if memory_request_mb is not None:
        reservations["memory"] = f"{memory_request_mb}M"

    if limits:
        resources["limits"] = limits
    if reservations:
        resources["reservations"] = reservations

    main = {"deploy": {"resources": resources}} if resources else {}
    compose = {"services": {"main": main}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path
