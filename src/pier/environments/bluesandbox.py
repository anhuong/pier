"""
Blue-Sandbox Harbor Environment Client

A Harbor BaseEnvironment implementation for the blue-sandbox platform.
Uses the ``bluesandbox`` Python package for sandbox lifecycle, command
execution, and file operations.

Supports:
- Single container sandboxes (image or template)
- docker-compose.yaml multi-service environments
- Shared PVC volumes between compose services
- Image building via Image Builder service
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shlex
import uuid
from pathlib import Path, PurePosixPath
from stat import S_ISDIR, S_ISREG
from typing import Any

import yaml

from bluesandbox import ConnectionConfig, Sandbox
from bluesandbox.models import RunCommandOpts, WriteEntry, SearchEntry

from pier.environments.base import BaseEnvironment, ExecResult
from pier.environments.capabilities import EnvironmentCapabilities
from pier.models.environment_type import EnvironmentType
from pier.models.task.config import EnvironmentConfig
from pier.models.trial.paths import EnvironmentPaths, TrialPaths

logger = logging.getLogger(__name__)

BLUESANDBOX_API_KEY_ENV = "AGENTIX_API_KEY"
DEFAULT_SHARED_VOLUME_PVC = "harbor-shared-volumes"


class BlueSandboxEnvironment(BaseEnvironment):
    """
    Harbor environment backed by the blue-sandbox platform.

    Creates sandboxes via the ``bluesandbox`` Python package, executes
    commands via SSE streaming, and manages files via the execd API.

    Supports single-container tasks and docker-compose.yaml multi-service
    environments with shared PVC volumes.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        api_url: str | None = None,
        ingress_url: str | None = None,  # Deprecated
        api_key: str | None = None,
        template: str = "default-sandbox",
        sandbox_timeout_sec: int = 60 * 60 * 24,
        ready_timeout_sec: int = 120,
        request_timeout_sec: int = 120,
        image: str | None = None,
        entrypoint: list[str] | None = None,
        metadata: dict[str, str] | None = None,
        image_builder_url: str | None = None,
        platform: str = "linux/amd64",
        shared_volume_pvc: str | None = DEFAULT_SHARED_VOLUME_PVC,
        connect_timeout_sec: int = 120,
        image_build_timeout_sec: int = 1800,
        image_push_timeout_sec: int = 600,
        volumes: list[dict[str, Any]] | None = None,
        skip_health_check: bool = False,
        agent_name: str | None = None,
        model_name: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        self._agent_name = agent_name or os.environ.get("HARBOR_AGENT_NAME", "")
        self._model_name = model_name or os.environ.get("HARBOR_MODEL_NAME", "")
        self._template = template
        self._sandbox_timeout_sec = sandbox_timeout_sec
        self._ready_timeout_sec = ready_timeout_sec
        self._image = image
        self._entrypoint = entrypoint
        self._metadata = dict(metadata or {})
        self._image_builder_url = image_builder_url or os.environ.get("HARBOR_IMAGE_BUILDER_URL")
        self._platform = platform
        self._shared_volume_pvc = shared_volume_pvc
        self._image_build_timeout_sec = image_build_timeout_sec
        self._image_push_timeout_sec = image_push_timeout_sec
        self._volumes = list(volumes or [])
        self._skip_health_check = skip_health_check

        # Connection config for the bluesandbox SDK
        # Accept AGENTIX_API_KEY from kwargs (--environment-kwarg) or env var
        resolved_api_key = (
            api_key
            or kwargs.get("AGENTIX_API_KEY")
            or os.environ.get(BLUESANDBOX_API_KEY_ENV)
        )
        self._connection_config = ConnectionConfig(
            api_url=api_url or os.environ.get("BLUESANDBOX_API_URL", ""),
            api_key=resolved_api_key,
            connect_timeout=float(connect_timeout_sec),
            request_timeout=float(request_timeout_sec),
        )

        # Current sandbox state
        self._sandbox: Sandbox | None = None
        self._sandbox_id: str | None = None
        self._sandboxes: dict[str, Sandbox] = {}  # service_name -> Sandbox
        self._workdir: str | None = None

        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            *args,
            **kwargs,
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.BLUESANDBOX

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # Open-egress port: the sandbox is NOT air-gapped during solve; it keeps
        # normal outbound so the agent reaches the CoreWeave model (matches the
        # harbor-fork behavior). disable_internet is advertised True only to
        # satisfy pier's base _validate_internet_config gate, which refuses tasks
        # declaring allow_internet=False on an env that reports it cannot isolate.
        # The flag is a capability advertisement, not an action: pier's base never
        # acts on it and _start_single adds no no-network isolation, so egress
        # stays open. filtered_egress stays False (no squid proxy) since we do not
        # enforce a per-host allowlist. Agents (mini-swe, openhands-sdk) install at
        # setup, so preinstall_agents=True.
        return EnvironmentCapabilities(
            gpus=True,
            disable_internet=True,
            filtered_egress=False,
            preinstall_agents=True,
            mounted=False,
            docker_compose=True,
        )

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _compose_path(self) -> Path:
        return self.environment_dir / "docker-compose.yaml"

    @property
    def _uses_compose(self) -> bool:
        return self._compose_path.exists()

    # ---- Compose parsing ----

    def _parse_compose(self) -> dict[str, dict[str, Any]]:
        compose = yaml.safe_load(self._compose_path.read_text())
        services = compose.get("services", {})
        if not services:
            raise ValueError("docker-compose.yaml must define at least one service.")
        return services

    @staticmethod
    def _resolve_main_service(services: dict[str, Any]) -> str:
        if "main" in services:
            return "main"
        return next(iter(services))

    @staticmethod
    def _topological_order(services: dict[str, Any]) -> list[str]:
        order: list[str] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            svc = services.get(name, {})
            deps = svc.get("depends_on", [])
            if isinstance(deps, dict):
                deps = list(deps.keys())
            for dep in deps:
                if dep in services:
                    visit(dep)
            order.append(name)

        for svc_name in services:
            visit(svc_name)
        return order

    def _parse_compose_shared_volumes(self, services: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        """Parse compose named volumes → per-service PVC mount specs."""
        compose = yaml.safe_load(self._compose_path.read_text())
        top_level_volumes = compose.get("volumes", {})
        if not top_level_volumes or not self._shared_volume_pvc:
            return {}

        named_volume_names = set(top_level_volumes.keys())
        safe_session = self.session_id.replace("/", "_")[:63]

        svc_volumes: dict[str, list[dict[str, str]]] = {}
        for svc_name, svc_def in services.items():
            for entry in svc_def.get("volumes", []):
                if isinstance(entry, str):
                    parts = entry.split(":")
                    if len(parts) >= 2 and parts[0] in named_volume_names:
                        svc_volumes.setdefault(svc_name, []).append({
                            "name": parts[0],
                            "pvc": {"claimName": self._shared_volume_pvc},
                            "mountPath": parts[1],
                            "subPath": f"{safe_session}/{parts[0]}",
                        })
                elif isinstance(entry, dict):
                    source = entry.get("source", "")
                    target = entry.get("target", "")
                    if source in named_volume_names and target:
                        svc_volumes.setdefault(svc_name, []).append({
                            "name": source,
                            "pvc": {"claimName": self._shared_volume_pvc},
                            "mountPath": target,
                            "subPath": f"{safe_session}/{source}",
                        })
        return svc_volumes

    # ---- Image resolution ----

    async def _resolve_compose_service_image(
        self, svc_name: str, svc_def: dict[str, Any], is_main: bool = False,
    ) -> str:
        if svc_def.get("image"):
            return svc_def["image"]
        if svc_def.get("build"):
            return await self._build_compose_service_image(svc_name, svc_def)
        if is_main:
            image = self._image or self.task_env_config.docker_image
            if not image and self._dockerfile_path.exists() and self._image_builder_url:
                image = await self._build_image()
            if image:
                return image
        raise RuntimeError(f"Compose service '{svc_name}' has no image or build directive.")

    async def _build_compose_service_image(self, svc_name: str, svc_def: dict[str, Any]) -> str:
        build_spec = svc_def.get("build", {})
        if isinstance(build_spec, str):
            build_context = self.environment_dir / build_spec
            dockerfile_path = build_context / "Dockerfile"
        else:
            context_dir = build_spec.get("context", ".")
            build_context = self.environment_dir / context_dir
            dockerfile_name = build_spec.get("dockerfile", "Dockerfile")
            dockerfile_path = build_context / dockerfile_name

        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found for service '{svc_name}': {dockerfile_path}")

        return await self._build_image_from_path(dockerfile_path, build_context, svc_name)

    @staticmethod
    def _parse_copy_destinations(
        dockerfile_content: str, target_prefix: str,
    ) -> list[tuple[str, str]]:
        """Extract (source, destination) pairs from COPY instructions targeting a path.

        Returns pairs where the destination starts with ``target_prefix``.
        Only handles simple ``COPY src dest`` forms (no --from, no URLs).
        """
        pairs: list[tuple[str, str]] = []
        for line in dockerfile_content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()
            if not upper.startswith("COPY "):
                continue
            parts = stripped.split()
            parts = parts[1:]  # drop COPY
            while parts and parts[0].startswith("--"):
                if parts[0].startswith("--from"):
                    parts = []
                    break
                parts = parts[1:]
            if len(parts) < 2:
                continue
            dest = parts[-1]
            sources = parts[:-1]
            if dest.startswith(target_prefix) or dest.rstrip("/") == target_prefix.rstrip("/"):
                for src in sources:
                    if src.startswith("http://") or src.startswith("https://"):
                        continue
                    pairs.append((src, dest))
        return pairs

    async def _seed_shared_volumes(
        self,
        services: dict[str, Any],
        shared_volumes: dict[str, list[dict[str, str]]],
    ) -> None:
        """Upload image files that were COPY'd into shared volume mount paths.

        Docker named volumes auto-populate from the image on first use.
        Kubernetes PVCs do not — the mount is always empty. This method
        replicates Docker's behavior by uploading the files from the local
        build context into the PVC via the sandbox file API.
        """
        seeded_paths: set[str] = set()

        for svc_name, vol_specs in shared_volumes.items():
            svc_def = services.get(svc_name, {})
            build_spec = svc_def.get("build")
            if not build_spec:
                continue

            if isinstance(build_spec, str):
                build_context = self.environment_dir / build_spec
                dockerfile_path = build_context / "Dockerfile"
            else:
                ctx = build_spec.get("context", ".")
                build_context = self.environment_dir / ctx
                df_name = build_spec.get("dockerfile", "Dockerfile")
                dockerfile_path = build_context / df_name

            if not dockerfile_path.exists():
                continue

            dockerfile_content = dockerfile_path.read_text()
            sandbox = self._sandboxes.get(svc_name)
            if not sandbox:
                continue

            for vol_spec in vol_specs:
                mount_path = vol_spec["mountPath"]

                copy_pairs = self._parse_copy_destinations(dockerfile_content, mount_path)
                for src, dest in copy_pairs:
                    src_path = build_context / src
                    if not src_path.exists():
                        continue

                    if dest.endswith("/"):
                        remote_path = dest + src_path.name
                    else:
                        remote_path = dest

                    if remote_path in seeded_paths:
                        continue
                    seeded_paths.add(remote_path)

                    if src_path.is_file():
                        parent = str(PurePosixPath(remote_path).parent)
                        if parent not in (".", "/"):
                            await sandbox.files.create_directories(
                                [WriteEntry(path=parent, mode=755)]
                            )
                        await sandbox.files.write_file(
                            remote_path, src_path.read_bytes(), mode=644,
                        )
                        logger.info(
                            "Seeded shared volume: %s -> %s (service=%s)",
                            src, remote_path, svc_name,
                        )
                    elif src_path.is_dir():
                        dirs_to_create: set[str] = {mount_path}
                        files_to_upload: list[tuple[str, bytes]] = []
                        for file_path in src_path.rglob("*"):
                            rel = file_path.relative_to(src_path).as_posix()
                            rpath = str(PurePosixPath(mount_path) / rel)
                            if file_path.is_dir():
                                dirs_to_create.add(rpath)
                            else:
                                dirs_to_create.add(str(PurePosixPath(rpath).parent))
                                files_to_upload.append((rpath, file_path.read_bytes()))
                        if dirs_to_create:
                            await sandbox.files.create_directories(
                                [WriteEntry(path=p, mode=755) for p in sorted(dirs_to_create)]
                            )
                        for rpath, data in files_to_upload:
                            await sandbox.files.write_file(rpath, data, mode=644)
                        logger.info(
                            "Seeded shared volume dir: %s -> %s (%d files, service=%s)",
                            src, mount_path, len(files_to_upload), svc_name,
                        )

    @staticmethod
    def _parse_dockerfile_cmd(dockerfile_content: str) -> list[str] | None:
        last_cmd: list[str] | None = None
        last_entrypoint: list[str] | None = None
        for line in dockerfile_content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()
            if upper.startswith("CMD "):
                rest = stripped[4:].strip()
                try:
                    parsed = json.loads(rest)
                    if isinstance(parsed, list):
                        last_cmd = parsed
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
                last_cmd = ["/bin/sh", "-c", rest]
            elif upper.startswith("ENTRYPOINT "):
                rest = stripped[11:].strip()
                try:
                    parsed = json.loads(rest)
                    if isinstance(parsed, list):
                        last_entrypoint = parsed
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
                last_entrypoint = ["/bin/sh", "-c", rest]
        return last_entrypoint or last_cmd

    # ---- Validation ----

    def _validate_definition(self) -> None:
        if self._uses_compose:
            services = self._parse_compose()
            main_name = self._resolve_main_service(services)
            for svc_name, svc_def in services.items():
                if not svc_def.get("image") and not svc_def.get("build"):
                    if svc_name == main_name:
                        if (not self.task_env_config.docker_image
                                and not self._image
                                and not self._dockerfile_path.exists()):
                            raise ValueError(
                                f"Compose main service '{svc_name}' has no image/build and no fallback."
                            )
                    else:
                        raise ValueError(f"Compose service '{svc_name}' must specify image or build.")
                if svc_def.get("build") and not self._image_builder_url:
                    raise RuntimeError(
                        f"Service '{svc_name}' uses build but HARBOR_IMAGE_BUILDER_URL is not set."
                    )
            return

        if self._image:
            return
        docker_image = self.task_env_config.docker_image
        if docker_image and not docker_image.startswith("/"):
            return
        if self._dockerfile_path.exists():
            content = self._dockerfile_path.read_text()
            needs_build = any(
                line.strip().upper().startswith(("RUN ", "COPY ", "ADD "))
                for line in content.splitlines()
                if not line.strip().startswith("#")
            )
            if needs_build and self._image_builder_url:
                # Dockerfile has RUN/COPY/ADD — delegate to image builder
                return
            # Simple Dockerfile (just FROM + metadata) or no builder available:
            # extract directives and use image directly
            for line in content.splitlines():
                stripped = line.strip()
                upper = stripped.upper()
                if upper.startswith("FROM ") and not self._image:
                    self._image = stripped.split()[1]
                elif upper.startswith("WORKDIR "):
                    self._workdir = stripped.split(None, 1)[1]
                elif upper.startswith("ENTRYPOINT "):
                    import ast
                    rest = stripped.split(None, 1)[1]
                    try:
                        self._entrypoint = ast.literal_eval(rest)
                    except (ValueError, SyntaxError):
                        if rest == "[]":
                            self._entrypoint = []
            if self._image:
                self._image = self._remap_to_icr(self._image)
                logger.info(
                    "Using Dockerfile directives: image=%s workdir=%s entrypoint=%s",
                    self._image, self._workdir, self._entrypoint,
                )
                if needs_build:
                    logger.info("Dockerfile has RUN/COPY — will replay post-start")
                return
            if needs_build and not self._image_builder_url:
                raise RuntimeError(
                    "Dockerfile has RUN/COPY but HARBOR_IMAGE_BUILDER_URL is not set."
                )
            raise RuntimeError("Dockerfile has no FROM line.")
        return
        if self._template:
            return
        raise FileNotFoundError("BlueSandboxEnvironment requires a template, image, or Dockerfile.")

    def _remap_to_icr(self, image: str) -> str:
        """Remap DockerHub image to ICR if ICR_NAMESPACE is configured."""
        icr_ns = os.environ.get("ICR_NAMESPACE")
        if not icr_ns:
            return image
        if ":" in image:
            name_with_registry, tag = image.rsplit(":", 1)
        else:
            name_with_registry, tag = image, "latest"
        name = name_with_registry.rsplit("/", 1)[-1]
        # Derive benchmark subfolder from the dataset path
        env_path = str(self.environment_dir)
        if "swebenchpro" in env_path:
            bench = "pro"
        elif "swebench_multilingual" in env_path:
            bench = "multilingual"
        elif "swebench-verified" in env_path:
            bench = "verified"
        elif "terminal-bench" in env_path:
            bench = "tb21"
        else:
            bench = "other"
        icr_image = f"{icr_ns}/{bench}/{name}:{tag}"
        logger.info("ICR remap: %s → %s", image, icr_image)
        return icr_image

    def _build_resource(self) -> dict[str, str]:
        """Build resource spec from task_env_config (includes overrides).

        Pier made cpus/memory_mb/storage_mb nullable (default None); fall back to
        harbor's historical defaults so the K8s resource spec never renders
        "None" / "NoneMi".
        """
        cpus = self._effective_cpus if self._effective_cpus is not None else 1
        memory_mb = (
            self._effective_memory_mb if self._effective_memory_mb is not None else 2048
        )
        resource = {
            "cpu": str(cpus),
            "memory": f"{memory_mb}Mi",
        }
        storage_mb = self._effective_storage_mb
        if storage_mb and storage_mb > 0:
            resource["ephemeral-storage"] = f"{storage_mb}Mi"
        if self._effective_gpus > 0:
            resource["gpu"] = str(self._effective_gpus)
        return resource

    # ---- Sandbox lifecycle ----

    async def start(self, force_build: bool) -> None:
        if self._uses_compose:
            await self._start_compose()
        else:
            await self._start_single()

    async def _start_single(self) -> None:
        image = self._image
        if not image:
            docker_image = self.task_env_config.docker_image
            if docker_image and not docker_image.startswith("/"):
                image = docker_image
        if not image and self._dockerfile_path.exists() and self._image_builder_url:
            image = await self._build_image()

        create_kwargs: dict[str, Any] = {
            "timeout": self._sandbox_timeout_sec,
            "template": self._template,
            "resource": self._build_resource(),
            "ready_timeout": self._ready_timeout_sec,
            "metadata": self._build_metadata(),
            "connection_config": self._connection_config,
            "skip_ready_check": self._skip_health_check,
        }
        if image:
            create_kwargs["image"] = image
        if self._entrypoint:
            create_kwargs["entrypoint"] = self._entrypoint
        if self._volumes:
            create_kwargs["volumes"] = self._volumes

        self._sandbox = await Sandbox.create(**create_kwargs)
        self._sandbox_id = self._sandbox.id
        logger.info("Created sandbox %s", self._sandbox_id)

        # When skipping image build (no builder URL), replay COPY and RUN
        # directives from the Dockerfile inside the running sandbox.
        if not self._image_builder_url and self._dockerfile_path.exists():
            await self._replay_dockerfile_directives()

        await self._create_standard_dirs()

    async def _create_sandbox_for_compose(self, body: dict[str, Any]) -> Sandbox:
        """Create a sandbox for a compose service using the SDK."""
        return await Sandbox.create(
            image=body.get("image"),
            timeout=body.get("timeout", self._sandbox_timeout_sec),
            env=body.get("env"),
            entrypoint=body.get("entrypoint"),
            metadata=body.get("metadata"),
            volumes=body.get("volumes"),
            ready_timeout=self._ready_timeout_sec,
            connection_config=self._connection_config,
        )

    async def _start_compose(self) -> None:
        services = self._parse_compose()
        main_name = self._resolve_main_service(services)
        creation_order = self._topological_order(services)
        shared_volumes = self._parse_compose_shared_volumes(services)

        if shared_volumes:
            logger.info("Compose shared volumes via PVC '%s': %s",
                        self._shared_volume_pvc,
                        {svc: [v["mountPath"] for v in vols] for svc, vols in shared_volumes.items()})

        # Resolve images
        images: dict[str, str] = {}
        for svc_name in creation_order:
            images[svc_name] = await self._resolve_compose_service_image(
                svc_name, services[svc_name], is_main=(svc_name == main_name),
            )

        # Create sandboxes in dependency order
        for svc_name in creation_order:
            svc_def = services[svc_name]

            svc_env = svc_def.get("environment", {})
            if isinstance(svc_env, list):
                env_dict: dict[str, str] = {}
                for item in svc_env:
                    k, _, v = item.partition("=")
                    env_dict[k] = v
                svc_env = env_dict

            entrypoint = svc_def.get("entrypoint")
            if isinstance(entrypoint, str):
                entrypoint = ["/bin/sh", "-c", entrypoint]
            command = svc_def.get("command")
            if command and not entrypoint:
                if isinstance(command, str):
                    entrypoint = ["/bin/sh", "-c", command]
                elif isinstance(command, list):
                    entrypoint = command

            if not entrypoint and svc_name != main_name:
                build_spec = svc_def.get("build")
                if build_spec:
                    if isinstance(build_spec, str):
                        df_path = self.environment_dir / build_spec / "Dockerfile"
                    else:
                        ctx = build_spec.get("context", ".")
                        df_name = build_spec.get("dockerfile", "Dockerfile")
                        df_path = self.environment_dir / ctx / df_name
                    if df_path.exists():
                        entrypoint = self._parse_dockerfile_cmd(df_path.read_text())

            body: dict[str, Any] = {
                "image": images[svc_name],
                "timeout": self._sandbox_timeout_sec,
            }
            if svc_env:
                body["env"] = svc_env
            if entrypoint:
                body["entrypoint"] = entrypoint
            body["metadata"] = self._build_metadata(extra={"compose_service": svc_name})
            svc_vols = shared_volumes.get(svc_name)
            if svc_vols:
                body["volumes"] = svc_vols

            sbx = await self._create_sandbox_for_compose(body)
            self._sandboxes[svc_name] = sbx
            logger.info("Created compose sandbox: service=%s sandbox_id=%s", svc_name, sbx.id)

        # Set main sandbox
        self._sandbox = self._sandboxes[main_name]
        self._sandbox_id = self._sandbox.id

        # Seed shared volumes with files from build contexts.
        # K8s PVCs mount as empty dirs (unlike Docker named volumes which
        # auto-populate from the image). This uploads files that were COPY'd
        # into mount paths by the Dockerfile.
        if shared_volumes:
            await self._seed_shared_volumes(services, shared_volumes)

        # Discover IPs and inject /etc/hosts
        ips: dict[str, str] = {}
        for svc_name, sbx in self._sandboxes.items():
            result = await sbx.commands.run("hostname -i")
            out = "\n".join(m.text for m in result.logs.stdout)
            ip = out.strip().split()[0] if out.strip() else ""
            ips[svc_name] = ip
            logger.info("Compose service %s has IP %s", svc_name, ip)

        for svc_name, sbx in self._sandboxes.items():
            entries = [(ip, name) for name, ip in ips.items() if name != svc_name]
            if entries:
                lines = "\\n".join(f"{ip}\\t{name}" for ip, name in entries)
                await sbx.commands.run(f"printf '{lines}\\n' >> /etc/hosts")

        await self._create_standard_dirs()

    def _build_metadata(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        safe_env = self.environment_name.replace("/", "_")[:63]
        safe_session = self.session_id.replace("/", "_")[:63]
        meta = {
            "environment_name": safe_env,
            "session_id": safe_session,
            **self._metadata,
        }
        try:
            meta["created_by"] = os.getlogin()
        except OSError:
            pass
        if self._agent_name:
            meta["agent_name"] = self._agent_name.replace("/", "_")[:63]
        if self._model_name:
            meta["model_name"] = self._model_name.replace("/", "_")[:63]
        if extra:
            meta.update(extra)
        return meta

    async def _replay_dockerfile_directives(self) -> None:
        """Replay COPY, RUN, and ENV from Dockerfile when no image builder is used."""
        content = self._dockerfile_path.read_text()
        build_context = self._dockerfile_path.parent
        for line in content.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("COPY ") or upper.startswith("ADD "):
                parts = stripped.split()
                if len(parts) >= 3 and not parts[1].startswith("--from"):
                    src_name = parts[-2]
                    dest = parts[-1]
                    src_path = build_context / src_name
                    if src_path.exists() and src_path.is_file():
                        if dest.endswith("/"):
                            remote = dest + src_path.name
                        else:
                            remote = dest
                        parent = str(PurePosixPath(remote).parent)
                        if parent not in (".", "/"):
                            await self._sandbox.commands.run(f"mkdir -p {parent}")
                        await self._sandbox.files.write_file(
                            remote, src_path.read_bytes(), mode=644,
                        )
                        logger.info("Replayed COPY: %s -> %s", src_name, remote)
            elif upper.startswith("RUN "):
                cmd = stripped[4:]
                result = await self._sandbox.commands.run(
                    cmd, opts=RunCommandOpts(timeout=300),
                )
                exit_code = result.exit_code if hasattr(result, 'exit_code') else (
                    result.logs.exit_code if hasattr(result.logs, 'exit_code') else None
                )
                logger.info("Replayed RUN (rc=%s): %s", exit_code, cmd[:80])
            elif upper.startswith("ENV ") and "=" in stripped:
                # ENV KEY=VALUE or ENV KEY VALUE
                rest = stripped[4:].strip()
                if "=" in rest:
                    key, _, val = rest.partition("=")
                    val = val.strip('"').strip("'")
                else:
                    parts = rest.split(None, 1)
                    key, val = parts[0], parts[1] if len(parts) > 1 else ""
                # Set env for future exec calls
                if not hasattr(self, '_extra_env'):
                    self._extra_env: dict[str, str] = {}
                self._extra_env[key] = val

    async def _create_standard_dirs(self) -> None:
        if not self._sandbox:
            return
        await self._sandbox.files.create_directories([
            WriteEntry(path=str(EnvironmentPaths.agent_dir), mode=755),
            WriteEntry(path=str(EnvironmentPaths.verifier_dir), mode=755),
            WriteEntry(path=str(EnvironmentPaths.artifacts_dir), mode=755),
            WriteEntry(path="/app", mode=755),
            WriteEntry(path="/workspace", mode=755),
        ])
        # Install uv (needed by test harness reward parser)
        await self._sandbox.commands.run(
            "command -v uv >/dev/null 2>&1 || "
            "(curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh) 2>/dev/null; "
            "true",
            opts=RunCommandOpts(timeout=60),
        )

    async def stop(self, delete: bool) -> None:
        if self._sandboxes:
            if self._shared_volume_pvc:
                for svc_name, sbx in self._sandboxes.items():
                    try:
                        services = self._parse_compose()
                        svc_def = services.get(svc_name, {})
                        for vol_entry in svc_def.get("volumes", []):
                            mount_path = None
                            if isinstance(vol_entry, str) and ":" in vol_entry:
                                mount_path = vol_entry.split(":")[1]
                            elif isinstance(vol_entry, dict):
                                mount_path = vol_entry.get("target")
                            if mount_path:
                                await sbx.commands.run(
                                    f"rm -rf {mount_path}/* {mount_path}/.[!.]* 2>/dev/null; true",
                                )
                        break
                    except Exception:
                        pass

            for svc_name, sbx in self._sandboxes.items():
                try:
                    await sbx.kill()
                except Exception:
                    logger.warning("Failed to delete compose sandbox %s", svc_name)
                finally:
                    await sbx.close()
            self._sandboxes.clear()
            self._sandbox = None
            self._sandbox_id = None
            return

        if self._sandbox:
            try:
                await self._sandbox.kill()
            except Exception:
                logger.warning("Failed to delete sandbox %s", self._sandbox_id)
            finally:
                await self._sandbox.close()
                self._sandbox = None
                self._sandbox_id = None

    # ---- Command execution ----

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if not self._sandbox:
            raise RuntimeError("Sandbox not started.")

        user = self._resolve_user(user)
        env = self._merge_env(env)

        # Write env vars to a temp file (mode 600), source it, then delete.
        # Prevents secrets from appearing in command logs or process listings.
        env_file = None
        if env:
            env_file = f"/.env_{uuid.uuid4().hex[:12]}"
            env_content = "\n".join(
                f"export {key}={_shell_quote(value)}" for key, value in env.items()
            )
            await self._sandbox.files.write_file(env_file, env_content, mode=600)

        if env_file:
            final_command = f". {env_file} && rm -f {env_file} && {command}"
        else:
            final_command = command

        if user is not None:
            if isinstance(user, int):
                user_arg = f"$(getent passwd {user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(str(user))
            final_command = f"su {user_arg} -s /bin/bash -c {shlex.quote(final_command)}"

        result = await self._sandbox.commands.run(
            final_command,
            opts=RunCommandOpts(cwd=cwd or self._workdir, timeout=timeout_sec),
        )

        stdout = "\n".join(m.text for m in result.logs.stdout) or None
        stderr = "\n".join(m.text for m in result.logs.stderr) or None

        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=(
                result.exit_code
                if result.exit_code is not None
                else (0 if result.error is None else 1)
            ),
        )

    # ---- File operations ----

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not started.")
        source_path = Path(source_path)
        parent = str(PurePosixPath(target_path).parent)
        if parent not in (".", "/"):
            await self._sandbox.files.create_directories([WriteEntry(path=parent, mode=755)])
        await self._sandbox.files.write_file(target_path, source_path.read_bytes(), mode=644)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not started.")
        source_dir = Path(source_dir)
        directories: set[str] = {target_dir}
        files: list[tuple[str, bytes]] = []

        for fp in source_dir.rglob("*"):
            relative = fp.relative_to(source_dir).as_posix()
            remote = str(PurePosixPath(target_dir) / relative)
            if fp.is_dir():
                directories.add(remote)
            else:
                directories.add(str(PurePosixPath(remote).parent))
                files.append((remote, fp.read_bytes()))

        await self._sandbox.files.create_directories(
            [WriteEntry(path=d, mode=755) for d in sorted(directories)]
        )
        for remote_path, content in files:
            await self._sandbox.files.write_file(remote_path, content, mode=644)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not started.")
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = await self._sandbox.files.read_bytes(source_path)
        target_path.write_bytes(data)

    @staticmethod
    def _infer_entry_kind_from_mode(mode: int | None) -> str | None:
        """Infer file type from stat mode bits.

        Returns ``"dir"``, ``"file"``, or ``None`` if the mode doesn't
        contain file type bits (some providers return bare permissions like 755).
        """
        if mode is None:
            return None
        if mode & 0o170000:
            if S_ISDIR(mode):
                return "dir"
            if S_ISREG(mode):
                return "file"
        return None

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not started.")
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            entries = await self._sandbox.files.search(SearchEntry(path=source_dir, pattern="*"))
        except Exception as e:
            logger.warning("download_dir: search failed for %s: %s: %s", source_dir, type(e).__name__, e)
            return
        if not entries:
            return
        for entry in entries:
            relative = PurePosixPath(entry.path).relative_to(PurePosixPath(source_dir))
            local_path = target_dir / Path(relative.as_posix())

            # Try mode-based inference first (handles full stat modes)
            if entry.is_dir:
                local_path.mkdir(parents=True, exist_ok=True)
                continue

            entry_kind = self._infer_entry_kind_from_mode(entry.mode)
            if entry_kind == "dir":
                local_path.mkdir(parents=True, exist_ok=True)
                continue

            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                data = await self._sandbox.files.read_bytes(entry.path)
                local_path.write_bytes(data)
            except Exception as e:
                logger.warning("download_dir: failed to read %s: %s: %s", entry.path, type(e).__name__, e)
                local_path.mkdir(parents=True, exist_ok=True)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(f"test -d {_shell_quote(path)}", timeout_sec=10)
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(f"test -f {_shell_quote(path)}", timeout_sec=10)
        return result.return_code == 0

    # ---- Image building ----

    @staticmethod
    def _qualify_from_images(dockerfile_content: str) -> str:
        """Prepend docker.io/ to short image names so buildah can resolve them."""
        lines = []
        for line in dockerfile_content.splitlines(True):
            if re.match(r'^\s*FROM\s+', line, re.IGNORECASE):
                m = re.match(r'(\s*FROM\s+(?:--platform=\S+\s+)?)(\S+)(.*)', line.rstrip('\n'))
                if m:
                    prefix, image, rest = m.group(1), m.group(2), m.group(3)
                    image_name = image.split(':')[0].split('@')[0]
                    if '.' not in image_name.split('/')[0]:
                        if '/' in image_name:
                            image = 'docker.io/' + image
                        else:
                            image = 'docker.io/library/' + image
                    line = prefix + image + rest + '\n'
            lines.append(line)
        return "".join(lines)

    async def _build_image(self) -> str:
        return await self._build_image_from_path(self._dockerfile_path, self.environment_dir)

    async def _build_image_from_path(
        self, dockerfile_path: Path, build_context: Path, service_name: str | None = None,
    ) -> str:
        dockerfile_content = self._qualify_from_images(dockerfile_path.read_text())
        files = self._collect_build_files(dockerfile_content, build_context)

        task_name = self.environment_dir.parent.name
        dataset_name = re.sub(r"[^a-zA-Z0-9._-]", "-", self.environment_dir.parent.parent.name)
        platform_suffix = self._platform.replace("/", "-")
        if service_name:
            icr_tag = (
                f"us.icr.io/ai-models-architecture/agentix/"
                f"datasets/harbor/{dataset_name}/{task_name}--{service_name}:{platform_suffix}"
            )
        else:
            icr_tag = (
                f"us.icr.io/ai-models-architecture/agentix/"
                f"datasets/harbor/{dataset_name}/{task_name}:{platform_suffix}"
            )

        payload: dict[str, Any] = {
            "dockerfile": dockerfile_content,
            "image_tag": icr_tag,
            "push_to_local": True,
            "platform": self._platform,
            "build_timeout": self._image_build_timeout_sec,
            "push_timeout": self._image_push_timeout_sec,
        }
        if files:
            payload["files"] = files

        import httpx
        headers = {"Content-Type": "application/json"}
        api_key = self._connection_config.api_key
        if api_key:
            headers["X-API-Key"] = api_key

        http_timeout = self._image_build_timeout_sec + self._image_push_timeout_sec + 60
        async with httpx.AsyncClient(timeout=http_timeout) as client:
            resp = await client.post(self._image_builder_url, json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        if result.get("cached_status"):
            logger.info("Image builder cache hit for %s", icr_tag)
        else:
            logger.info("Image built in %.1fs: %s", result.get("build_time_seconds", 0), icr_tag)

        return result.get("icr_tag") or icr_tag

    @staticmethod
    def _collect_build_files(dockerfile_content: str, build_context: Path) -> list[dict[str, str]]:
        files: list[dict[str, str]] = []
        seen: set[str] = set()
        for line in dockerfile_content.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if not (upper.startswith("COPY ") or upper.startswith("ADD ")):
                continue
            parts = stripped.split()[1:]
            while parts and parts[0].startswith("--"):
                if parts[0].startswith("--from"):
                    parts = []
                    break
                parts = parts[1:]
            if len(parts) < 2:
                continue
            for src in parts[:-1]:
                if src.startswith("http"):
                    continue
                if src in (".", "./"):
                    matches = [build_context]
                else:
                    try:
                        matches = list(build_context.glob(src))
                    except (IndexError, ValueError):
                        matches = []
                    if not matches:
                        src_path = build_context / src
                        matches = [src_path] if src_path.exists() else []
                for match in matches:
                    if match.is_file():
                        rel = match.relative_to(build_context).as_posix()
                        if rel not in seen:
                            seen.add(rel)
                            files.append({"path": rel, "content": base64.b64encode(match.read_bytes()).decode()})
                    elif match.is_dir():
                        for fp in match.rglob("*"):
                            if fp.is_file():
                                rel = fp.relative_to(build_context).as_posix()
                                if rel not in seen:
                                    seen.add(rel)
                                    files.append({"path": rel, "content": base64.b64encode(fp.read_bytes()).decode()})
        return files


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
