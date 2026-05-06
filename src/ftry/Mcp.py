from __future__ import annotations

import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .Tools import (
    FtryCliError,
    _load_yaml_mapping,
    _require_mapping,
    _require_non_empty_string,
    _require_optional_string,
    _require_sequence,
)

MCP_DIR_NAME = "mcp"
MCP_TRANSPORT_STDIO = "stdio"
MCP_TRANSPORT_HTTP = "http"
MCP_TRANSPORT_WEBSOCKET = "websocket"
MCP_TRANSPORT_VALUES = (MCP_TRANSPORT_STDIO, MCP_TRANSPORT_HTTP, MCP_TRANSPORT_WEBSOCKET)


@dataclass(frozen=True)
class McpConfig:
    name: str
    transport: str
    description: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    url: str | None = None
    headers: Mapping[str, str] | None = None
    allowed_tools: tuple[str, ...] = ()
    source_path: Path | None = None


class Mcp:
    def __init__(self, config: McpConfig):
        self._config = config

    @property
    def config(self) -> McpConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._config.name

    @classmethod
    def parse_server_names(cls, raw_value: Any, *, field_name: str, config_kind: str) -> tuple[str, ...]:
        if raw_value is None:
            return ()

        raw_names = _require_sequence(raw_value, field_name, config_kind)
        rendered_names: list[str] = []
        for index, raw_name in enumerate(raw_names):
            rendered_names.append(
                _require_non_empty_string(raw_name, f"{field_name}[{index}]", config_kind)
            )
        return cls.merge_server_names(rendered_names)

    @staticmethod
    def merge_server_names(*groups: Sequence[str]) -> tuple[str, ...]:
        merged_names: list[str] = []
        seen_names: set[str] = set()
        for group in groups:
            for raw_name in group:
                normalized_name = raw_name.strip()
                if not normalized_name or normalized_name in seen_names:
                    continue
                seen_names.add(normalized_name)
                merged_names.append(normalized_name)
        return tuple(merged_names)

    @classmethod
    def get_registry_dir(cls, *, cwd: Path | None = None) -> Path:
        return (Path.cwd() if cwd is None else Path(cwd)) / MCP_DIR_NAME

    @classmethod
    def load_catalog(cls, *, cwd: Path | None = None) -> tuple[McpConfig, ...]:
        registry_dir = cls.get_registry_dir(cwd=cwd)
        if not registry_dir.exists():
            return ()
        if not registry_dir.is_dir():
            raise FtryCliError(f"MCP registry path is not a directory: {registry_dir}")

        descriptor_paths = sorted(registry_dir.glob("*.yaml"))
        configs: list[McpConfig] = []
        seen_by_name: dict[str, Path] = {}
        for descriptor_path in descriptor_paths:
            config = cls.load_config(descriptor_path)
            existing_path = seen_by_name.get(config.name)
            if existing_path is not None:
                raise FtryCliError(
                    "Duplicate MCP descriptor name "
                    f"`{config.name}` found in `{existing_path}` and `{descriptor_path}`."
                )
            seen_by_name[config.name] = descriptor_path
            configs.append(config)
        return tuple(configs)

    @classmethod
    def resolve_configs(
        cls,
        names: Sequence[str],
        *,
        cwd: Path | None = None,
    ) -> tuple[McpConfig, ...]:
        resolved_names = cls.merge_server_names(names)
        if not resolved_names:
            return ()

        registry_dir = cls.get_registry_dir(cwd=cwd)
        catalog = {config.name: config for config in cls.load_catalog(cwd=cwd)}
        missing_names = [name for name in resolved_names if name not in catalog]
        if missing_names:
            missing_fragment = ", ".join(f"`{name}`" for name in missing_names)
            raise FtryCliError(
                f"Referenced MCP descriptor(s) not found in `{registry_dir}`: {missing_fragment}"
            )
        return tuple(catalog[name] for name in resolved_names)

    @classmethod
    def resolve(
        cls,
        names: Sequence[str],
        *,
        cwd: Path | None = None,
    ) -> tuple[Mcp, ...]:
        return tuple(cls(config) for config in cls.resolve_configs(names, cwd=cwd))

    @classmethod
    def load_config(cls, path: str | Path) -> McpConfig:
        descriptor_path = Path(path)
        if not descriptor_path.is_file():
            raise FtryCliError(f"MCP descriptor file not found: {descriptor_path}")

        config = _load_yaml_mapping(descriptor_path, config_kind="MCP descriptor")
        transport = _require_non_empty_string(config.get("transport"), "transport", "MCP descriptor")
        if transport not in MCP_TRANSPORT_VALUES:
            raise FtryCliError(
                "Invalid or missing `transport` in MCP descriptor YAML: "
                f"expected one of {', '.join(MCP_TRANSPORT_VALUES)}."
            )

        allowed_tools = cls.parse_server_names(
            config.get("allowed-tools"),
            field_name="allowed-tools",
            config_kind="MCP descriptor",
        )
        common_kwargs = {
            "name": _require_non_empty_string(config.get("name"), "name", "MCP descriptor"),
            "description": _require_optional_string(config.get("description"), "description", "MCP descriptor"),
            "transport": transport,
            "allowed_tools": allowed_tools,
            "source_path": descriptor_path,
        }

        if transport == MCP_TRANSPORT_STDIO:
            return McpConfig(
                command=_parse_string_required(config.get("command"), field_name="command", config_kind="MCP descriptor"),
                args=_parse_string_list(config.get("args"), field_name="args", config_kind="MCP descriptor"),
                env=_parse_string_mapping(config.get("env"), field_name="env", config_kind="MCP descriptor"),
                **common_kwargs,
            )

        return McpConfig(
            url=_parse_string_required(config.get("url"), field_name="url", config_kind="MCP descriptor"),
            headers=_parse_string_mapping(config.get("headers"), field_name="headers", config_kind="MCP descriptor"),
            **common_kwargs,
        )

    @classmethod
    def render_catalog_for_builder(cls, *, cwd: Path | None = None) -> str:
        catalog = cls.load_catalog(cwd=cwd)
        if not catalog:
            return "Available MCP descriptors:\n- (none)"

        lines = ["Available MCP descriptors:"]
        for config in catalog:
            line = f"- {config.name} | transport: {config.transport}"
            if config.description:
                line += f" | description: {config.description}"
            if config.allowed_tools:
                line += f" | allowed-tools: {', '.join(config.allowed_tools)}"
            lines.append(line)
        return "\n".join(lines)

    @classmethod
    async def open_tools(
        cls,
        mcps: Sequence[Mcp],
        *,
        exit_stack: AsyncExitStack,
    ) -> tuple[Any, ...]:
        tools: list[Any] = []
        for mcp in mcps:
            tools.append(await mcp.enter_tool(exit_stack=exit_stack))
        return tuple(tools)

    async def enter_tool(self, *, exit_stack: AsyncExitStack) -> Any:
        return await exit_stack.enter_async_context(self.create_tool())

    def create_tool(self) -> Any:
        self._require_runtime_dependency()
        try:
            from agent_framework import MCPStdioTool, MCPStreamableHTTPTool, MCPWebsocketTool
        except ImportError as exc:  # pragma: no cover - exercised through CLI error paths
            raise FtryCliError(
                "Microsoft Agent Framework MCP support is required for MCP-enabled runs. "
                "Install the optional `mcp[ws]` package and reinstall the project."
            ) from exc

        common_kwargs: dict[str, Any] = {
            "name": self.config.name,
            "description": self.config.description,
            "approval_mode": "never_require",
        }
        if self.config.allowed_tools:
            common_kwargs["allowed_tools"] = list(self.config.allowed_tools)

        if self.config.transport == MCP_TRANSPORT_STDIO:
            return MCPStdioTool(
                command=self.config.command or "",
                args=list(self.config.args) or None,
                env=self._resolve_runtime_mapping(self.config.env, field_name=f"MCP `{self.config.name}` env"),
                **common_kwargs,
            )

        if self.config.transport == MCP_TRANSPORT_HTTP:
            header_provider = None
            if self.config.headers:
                header_provider = (
                    lambda _kwargs, raw_headers=dict(self.config.headers), name=self.config.name: self._resolve_runtime_mapping(
                        raw_headers,
                        field_name=f"MCP `{name}` headers",
                    )
                    or {}
                )
            return MCPStreamableHTTPTool(
                url=self.config.url or "",
                header_provider=header_provider,
                **common_kwargs,
            )

        websocket_kwargs: dict[str, Any] = {}
        if self.config.headers:
            websocket_kwargs["headers"] = self._resolve_runtime_mapping(
                self.config.headers,
                field_name=f"MCP `{self.config.name}` headers",
            )
        return MCPWebsocketTool(
            url=self.config.url or "",
            **common_kwargs,
            **websocket_kwargs,
        )

    @staticmethod
    def _require_runtime_dependency() -> None:
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise FtryCliError(
                "The Python package `mcp` is required for MCP-enabled runs. "
                "Reinstall the project with `python -m pip install -e .` or install `mcp[ws]`."
            ) from exc

    @staticmethod
    def _resolve_runtime_mapping(
        raw_mapping: Mapping[str, str] | None,
        *,
        field_name: str,
    ) -> dict[str, str] | None:
        if not raw_mapping:
            return None
        return {
            key: Mcp._resolve_runtime_value(value, field_name=f"{field_name}.{key}")
            for key, value in raw_mapping.items()
        }

    @staticmethod
    def _resolve_runtime_value(value: str, *, field_name: str) -> str:
        if not value.startswith("env:"):
            return value

        env_name = value.removeprefix("env:").strip()
        if not env_name:
            raise FtryCliError(f"Invalid `{field_name}`: missing environment variable name after `env:`.")

        resolved_value = os.getenv(env_name)
        if not resolved_value:
            raise FtryCliError(f"Environment variable `{env_name}` is not set for `{field_name}`.")
        return resolved_value


def _parse_string_required(raw_value: Any, *, field_name: str, config_kind: str) -> str:
    return _require_non_empty_string(raw_value, field_name, config_kind)


def _parse_string_list(raw_value: Any, *, field_name: str, config_kind: str) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    raw_items = _require_sequence(raw_value, field_name, config_kind)
    return tuple(
        _require_non_empty_string(raw_item, f"{field_name}[{index}]", config_kind)
        for index, raw_item in enumerate(raw_items)
    )


def _parse_string_mapping(raw_value: Any, *, field_name: str, config_kind: str) -> dict[str, str] | None:
    if raw_value is None:
        return None
    mapping = _require_mapping(raw_value, field_name, config_kind)
    rendered_mapping: dict[str, str] = {}
    for raw_key, raw_item in mapping.items():
        key = _require_non_empty_string(raw_key, f"{field_name}.<key>", config_kind)
        rendered_mapping[key] = _require_non_empty_string(raw_item, f"{field_name}.{key}", config_kind)
    return rendered_mapping
