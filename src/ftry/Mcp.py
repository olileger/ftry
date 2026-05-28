from __future__ import annotations

import json
import os
from collections.abc import Sequence as SequenceCollection
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
LOCAL_MCP_DESCRIPTOR_FILE_PREFIX = "mcp-"
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


@dataclass(frozen=True)
class McpDiscoveredTool:
    name: str
    title: str | None = None
    description: str | None = None
    input_schema_summary: str | None = None


@dataclass(frozen=True)
class McpDiscoveredPrompt:
    name: str
    title: str | None = None
    description: str | None = None
    arguments_summary: str | None = None


@dataclass(frozen=True)
class McpDiscoveredResource:
    name: str
    uri: str
    title: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class McpDiscoveredResourceTemplate:
    name: str
    uri_template: str
    title: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class McpServerCapabilities:
    server_name: str
    transport: str
    tools: tuple[McpDiscoveredTool, ...] = ()
    prompts: tuple[McpDiscoveredPrompt, ...] = ()
    resources: tuple[McpDiscoveredResource, ...] = ()
    resource_templates: tuple[McpDiscoveredResourceTemplate, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class McpRuntimeConnection:
    mcp: Mcp
    tool: Any
    capabilities: McpServerCapabilities


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
        descriptor_paths = cls._list_descriptor_paths(cwd=cwd)
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

        searched_locations = cls._describe_search_locations(cwd=cwd)
        catalog: dict[str, McpConfig] = {}
        for base_dir in _iter_registry_base_dirs(cwd):
            for config in cls.load_catalog(cwd=base_dir):
                catalog.setdefault(config.name, config)
        missing_names = [name for name in resolved_names if name not in catalog]
        if missing_names:
            missing_fragment = ", ".join(f"`{name}`" for name in missing_names)
            registry_fragment = ", ".join(f"`{path}`" for path in searched_locations)
            raise FtryCliError(
                f"Referenced MCP descriptor(s) not found in {registry_fragment}: {missing_fragment}"
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
    def _list_descriptor_paths(cls, *, cwd: Path | None = None) -> tuple[Path, ...]:
        base_dir = Path.cwd() if cwd is None else Path(cwd)
        registry_dir = cls.get_registry_dir(cwd=base_dir)
        local_descriptor_paths = sorted(base_dir.glob(f"{LOCAL_MCP_DESCRIPTOR_FILE_PREFIX}*.yaml"))
        registry_descriptor_paths: list[Path] = []
        if registry_dir.exists():
            if not registry_dir.is_dir():
                raise FtryCliError(f"MCP registry path is not a directory: {registry_dir}")
            registry_descriptor_paths = sorted(registry_dir.glob("*.yaml"))
        return tuple(local_descriptor_paths + registry_descriptor_paths)

    @classmethod
    def _describe_search_locations(cls, *, cwd: Path | None = None) -> tuple[str, ...]:
        locations: list[str] = []
        for base_dir in _iter_registry_base_dirs(cwd):
            base_path = Path(base_dir)
            locations.append(str(base_path / f"{LOCAL_MCP_DESCRIPTOR_FILE_PREFIX}*.yaml"))
            locations.append(str(cls.get_registry_dir(cwd=base_path)))
        deduped_locations: list[str] = []
        seen_locations: set[str] = set()
        for location in locations:
            if location in seen_locations:
                continue
            seen_locations.add(location)
            deduped_locations.append(location)
        return tuple(deduped_locations)

    @classmethod
    async def open_connections(
        cls,
        mcps: Sequence[Mcp],
        *,
        exit_stack: AsyncExitStack,
    ) -> tuple[McpRuntimeConnection, ...]:
        connections: list[McpRuntimeConnection] = []
        for mcp in mcps:
            tool = await mcp.enter_tool(exit_stack=exit_stack)
            connections.append(
                McpRuntimeConnection(
                    mcp=mcp,
                    tool=tool,
                    capabilities=await mcp.discover_capabilities(tool),
                )
            )
        return tuple(connections)

    @classmethod
    async def open_tools(
        cls,
        mcps: Sequence[Mcp],
        *,
        exit_stack: AsyncExitStack,
    ) -> tuple[Any, ...]:
        return tuple(connection.tool for connection in await cls.open_connections(mcps, exit_stack=exit_stack))

    @staticmethod
    def extract_tools(connections: Sequence[McpRuntimeConnection]) -> tuple[Any, ...]:
        return tuple(connection.tool for connection in connections)

    @classmethod
    def render_runtime_context(cls, connections: Sequence[McpRuntimeConnection]) -> str | None:
        if not connections:
            return None

        lines = [
            "<McpRuntimeCapabilities>",
            "The following MCP servers were connected live for this run. Base your tool usage on this live capability directory.",
        ]
        for connection in connections:
            capabilities = connection.capabilities
            lines.append(f"Server `{capabilities.server_name}` (transport: {capabilities.transport})")
            if capabilities.tools:
                lines.append("  Tools:")
                for tool in capabilities.tools:
                    line = f"    - {tool.name}"
                    if tool.title:
                        line += f" | title: {tool.title}"
                    if tool.description:
                        line += f" | description: {tool.description}"
                    if tool.input_schema_summary:
                        line += f" | input-schema: {tool.input_schema_summary}"
                    lines.append(line)
            if capabilities.prompts:
                lines.append("  Prompts:")
                for prompt in capabilities.prompts:
                    line = f"    - {prompt.name}"
                    if prompt.title:
                        line += f" | title: {prompt.title}"
                    if prompt.description:
                        line += f" | description: {prompt.description}"
                    if prompt.arguments_summary:
                        line += f" | arguments: {prompt.arguments_summary}"
                    lines.append(line)
            if capabilities.resources:
                lines.append("  Resources:")
                for resource in capabilities.resources:
                    line = f"    - {resource.name} | uri: {resource.uri}"
                    if resource.title:
                        line += f" | title: {resource.title}"
                    if resource.description:
                        line += f" | description: {resource.description}"
                    lines.append(line)
            if capabilities.resource_templates:
                lines.append("  Resource templates:")
                for resource_template in capabilities.resource_templates:
                    line = f"    - {resource_template.name} | uri-template: {resource_template.uri_template}"
                    if resource_template.title:
                        line += f" | title: {resource_template.title}"
                    if resource_template.description:
                        line += f" | description: {resource_template.description}"
                    lines.append(line)
            if capabilities.warnings:
                lines.append("  Discovery warnings:")
                for warning in capabilities.warnings:
                    lines.append(f"    - {warning}")
        lines.append("</McpRuntimeCapabilities>")
        return "\n".join(lines)

    async def enter_tool(self, *, exit_stack: AsyncExitStack) -> Any:
        return await exit_stack.enter_async_context(self.create_tool())

    async def discover_capabilities(self, tool: Any) -> McpServerCapabilities:
        warnings: list[str] = []
        session = getattr(tool, "session", None)
        if session is None:
            warnings.append("Live MCP session metadata is not exposed by the runtime tool.")
            return McpServerCapabilities(
                server_name=self.config.name,
                transport=self.config.transport,
                warnings=tuple(warnings),
            )

        return McpServerCapabilities(
            server_name=self.config.name,
            transport=self.config.transport,
            tools=await self._discover_tools(session, warnings=warnings),
            prompts=await self._discover_prompts(session, warnings=warnings),
            resources=await self._discover_resources(session, warnings=warnings),
            resource_templates=await self._discover_resource_templates(session, warnings=warnings),
            warnings=tuple(warnings),
        )

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

    async def _discover_tools(self, session: Any, *, warnings: list[str]) -> tuple[McpDiscoveredTool, ...]:
        records = await self._list_session_items(
            session,
            method_name="list_tools",
            items_field="tools",
            warning_label="tools",
            warnings=warnings,
        )
        tools: list[McpDiscoveredTool] = []
        for record in records:
            name = _read_string_field(record, "name")
            if not name or self._is_filtered_out(name):
                continue
            tools.append(
                McpDiscoveredTool(
                    name=name,
                    title=_read_string_field(record, "title"),
                    description=_read_string_field(record, "description"),
                    input_schema_summary=_summarize_json_like(_read_field(record, "inputSchema")),
                )
            )
        return tuple(tools)

    async def _discover_prompts(self, session: Any, *, warnings: list[str]) -> tuple[McpDiscoveredPrompt, ...]:
        records = await self._list_session_items(
            session,
            method_name="list_prompts",
            items_field="prompts",
            warning_label="prompts",
            warnings=warnings,
        )
        prompts: list[McpDiscoveredPrompt] = []
        for record in records:
            name = _read_string_field(record, "name")
            if not name or self._is_filtered_out(name):
                continue
            prompts.append(
                McpDiscoveredPrompt(
                    name=name,
                    title=_read_string_field(record, "title"),
                    description=_read_string_field(record, "description"),
                    arguments_summary=_summarize_prompt_arguments(_read_field(record, "arguments")),
                )
            )
        return tuple(prompts)

    async def _discover_resources(self, session: Any, *, warnings: list[str]) -> tuple[McpDiscoveredResource, ...]:
        records = await self._list_session_items(
            session,
            method_name="list_resources",
            items_field="resources",
            warning_label="resources",
            warnings=warnings,
        )
        resources: list[McpDiscoveredResource] = []
        for record in records:
            name = _read_string_field(record, "name")
            uri = _read_string_field(record, "uri")
            if not name or not uri:
                continue
            resources.append(
                McpDiscoveredResource(
                    name=name,
                    uri=uri,
                    title=_read_string_field(record, "title"),
                    description=_read_string_field(record, "description"),
                )
            )
        return tuple(resources)

    async def _discover_resource_templates(
        self,
        session: Any,
        *,
        warnings: list[str],
    ) -> tuple[McpDiscoveredResourceTemplate, ...]:
        records = await self._list_session_items(
            session,
            method_name="list_resource_templates",
            items_field="resourceTemplates",
            warning_label="resource templates",
            warnings=warnings,
        )
        resource_templates: list[McpDiscoveredResourceTemplate] = []
        for record in records:
            name = _read_string_field(record, "name")
            uri_template = _read_string_field(record, "uriTemplate")
            if not name or not uri_template:
                continue
            resource_templates.append(
                McpDiscoveredResourceTemplate(
                    name=name,
                    uri_template=uri_template,
                    title=_read_string_field(record, "title"),
                    description=_read_string_field(record, "description"),
                )
            )
        return tuple(resource_templates)

    async def _list_session_items(
        self,
        session: Any,
        *,
        method_name: str,
        items_field: str,
        warning_label: str,
        warnings: list[str],
    ) -> tuple[Any, ...]:
        list_method = getattr(session, method_name, None)
        if list_method is None:
            warnings.append(f"Runtime MCP session does not expose {warning_label}.")
            return ()

        items: list[Any] = []
        cursor: str | None = None
        while True:
            try:
                page = await list_method(cursor=cursor)
            except TypeError:
                page = await list_method()
            except Exception as exc:
                warnings.append(f"Could not list {warning_label} from MCP server `{self.config.name}`: {exc}")
                return tuple(items)

            raw_items = _read_field(page, items_field)
            if isinstance(raw_items, SequenceCollection) and not isinstance(raw_items, (str, bytes, bytearray)):
                items.extend(raw_items)
            cursor_value = _read_field(page, "nextCursor")
            cursor = cursor_value if isinstance(cursor_value, str) and cursor_value else None
            if cursor is None:
                break
        return tuple(items)

    def _is_filtered_out(self, remote_name: str) -> bool:
        return bool(self.config.allowed_tools) and remote_name not in self.config.allowed_tools

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


def _iter_registry_base_dirs(preferred_dir: Path | None) -> tuple[Path, ...]:
    candidate_dirs: list[Path] = []
    if preferred_dir is not None:
        preferred_path = Path(preferred_dir)
        candidate_dirs.append(preferred_path)
        parent_dir = preferred_path.parent
        if parent_dir != preferred_path:
            candidate_dirs.append(parent_dir)
    current_dir = Path.cwd()
    if all(candidate.resolve() != current_dir.resolve() for candidate in candidate_dirs if candidate.exists()):
        candidate_dirs.append(current_dir)
    elif not candidate_dirs:
        candidate_dirs.append(current_dir)
    deduped_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    for candidate in candidate_dirs:
        try:
            normalized = candidate.resolve()
        except OSError:
            normalized = candidate
        if normalized in seen_dirs:
            continue
        seen_dirs.add(normalized)
        deduped_dirs.append(candidate)
    return tuple(deduped_dirs)


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


def _read_field(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _read_string_field(value: Any, field_name: str) -> str | None:
    field_value = _read_field(value, field_name)
    return field_value if isinstance(field_value, str) and field_value.strip() else None


def _summarize_json_like(value: Any, *, max_length: int = 220) -> str | None:
    if value in (None, "", [], {}):
        return None
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        rendered = str(value)
    rendered = " ".join(rendered.split())
    if len(rendered) <= max_length:
        return rendered
    return f"{rendered[: max_length - 3]}..."


def _summarize_prompt_arguments(arguments: Any, *, max_length: int = 220) -> str | None:
    if not isinstance(arguments, SequenceCollection) or isinstance(arguments, (str, bytes, bytearray)):
        return None
    rendered_arguments: list[str] = []
    for argument in arguments:
        name = _read_string_field(argument, "name")
        if not name:
            continue
        rendered_argument = name
        description = _read_string_field(argument, "description")
        if description:
            rendered_argument += f": {description}"
        if bool(_read_field(argument, "required")):
            rendered_argument += " (required)"
        rendered_arguments.append(rendered_argument)
    if not rendered_arguments:
        return None
    rendered = "; ".join(rendered_arguments)
    if len(rendered) <= max_length:
        return rendered
    return f"{rendered[: max_length - 3]}..."
