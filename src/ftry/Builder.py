from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .Mcp import (
    MCP_TRANSPORT_HTTP,
    MCP_TRANSPORT_STDIO,
    MCP_TRANSPORT_VALUES,
    MCP_TRANSPORT_WEBSOCKET,
    Mcp,
    McpConfig,
)
from .StandaloneAgent import StandaloneAgent
from .Team import Team
from .Tools import FtryCliError, _sanitize_agent_name

INTERNAL_BUILDER_TEAM_FILE = Path(__file__).resolve().parent / "builder" / "team.yaml"
DEFAULT_BUILD_OUTPUT_DIR_NAME = "output"
DEFAULT_MODEL_NAME = "gpt-4o-2024-08-06"
DEFAULT_MODEL_PROVIDER = "openai"
DEFAULT_MODEL_API_KEY = "env:OAI_API_KEY"
DEFAULT_TEAM_MAX_TURNS = 6
LOCAL_MCP_DESCRIPTOR_FILE_PREFIX = "mcp-"
BUILD_KIND_AGENT = "agent"
BUILD_KIND_TEAM = "team"
BUILD_KIND_VALUES = (BUILD_KIND_AGENT, BUILD_KIND_TEAM)
MAX_BUILDER_CLARIFICATION_ROUNDS = 3
_PLACEHOLDER_STDIO_COMMAND_FRAGMENTS = (
    "path/to/",
    "path\\to\\",
    "existing/command",
    "existing-command",
    "placeholder",
    "your-command",
    "command-here",
    "<command>",
)
_PLACEHOLDER_NETWORK_URL_FRAGMENTS = (
    ".invalid",
    "placeholder",
    "<url>",
    "your-url",
    "your-server",
)


@dataclass(frozen=True)
class _BuiltAgentSpec:
    name: str
    prompt: str
    description: str | None = None
    mcp_servers: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BuiltTeamSpec:
    name: str
    prompt: str
    max_turns: int = DEFAULT_TEAM_MAX_TURNS
    description: str | None = None
    mcp_servers: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BuiltMcpServerSpec:
    name: str
    transport: str
    description: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    url: str | None = None
    headers: Mapping[str, str] | None = None
    allowed_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildSpec:
    kind: str
    agent: _BuiltAgentSpec | None = None
    team: _BuiltTeamSpec | None = None
    agents: tuple[_BuiltAgentSpec, ...] = ()
    mcp_servers: tuple[_BuiltMcpServerSpec, ...] = ()


UserInputProvider = Callable[[str], str]


def build_from_prompt(
    prompt: str,
    *,
    output_dir: str | Path | None = None,
    user_input_provider: UserInputProvider | None = None,
) -> tuple[Path, ...]:
    existing_mcp_catalog = Mcp.load_catalog()
    clarifications: list[tuple[str, str]] = []
    for _ in range(MAX_BUILDER_CLARIFICATION_ROUNDS):
        raw_output = _run_builder_team(
            prompt,
            existing_mcp_catalog=existing_mcp_catalog,
            user_input_provider=user_input_provider,
            clarifications=clarifications,
        )
        spec = _parse_build_spec_output(raw_output)
        mcp_question = _build_mcp_clarification_question(spec)
        if mcp_question is None:
            break
        if user_input_provider is None:
            raise FtryCliError(mcp_question)
        clarifications.append((mcp_question, user_input_provider(mcp_question)))
    else:
        raise FtryCliError(
            "Build could not resolve the required MCP connection details after multiple clarification rounds."
        )

    _validate_build_spec_mcp(spec, existing_catalog=existing_mcp_catalog)
    target_dir = _resolve_build_output_dir(spec, output_dir=output_dir)
    return _write_build_outputs(spec, output_dir=target_dir)


def _run_builder_team(
    prompt: str,
    *,
    existing_mcp_catalog: Sequence[McpConfig],
    user_input_provider: UserInputProvider | None = None,
    clarifications: Sequence[tuple[str, str]] = (),
) -> str:
    builder_team = Team.from_file(INTERNAL_BUILDER_TEAM_FILE)
    return asyncio.run(
        builder_team.run(
            _render_builder_input(
                prompt,
                existing_mcp_catalog=existing_mcp_catalog,
                clarifications=clarifications,
            ),
            user_input_provider=user_input_provider,
        )
    )


def _render_builder_input(
    prompt: str,
    *,
    existing_mcp_catalog: Sequence[McpConfig],
    clarifications: Sequence[tuple[str, str]] = (),
) -> str:
    catalog_lines = ["Available MCP descriptors in .\\mcp:"]
    if not existing_mcp_catalog:
        catalog_lines.append("- (none)")
    else:
        for config in existing_mcp_catalog:
            line = f"- {config.name} | transport: {config.transport}"
            if config.description:
                line += f" | description: {config.description}"
            if config.allowed_tools:
                line += f" | allowed-tools: {', '.join(config.allowed_tools)}"
            catalog_lines.append(line)

    rendered_sections = [f"User request:\n{prompt}", ""]
    if clarifications:
        rendered_sections.extend(["Clarifications collected during build:"])
        for index, (question, answer) in enumerate(clarifications, start=1):
            rendered_sections.extend(
                [
                    f"{index}. Question: {question}",
                    f"   Answer: {answer}",
                ]
            )
        rendered_sections.append("")
    rendered_sections.extend(catalog_lines)
    return "\n".join(rendered_sections)


def _parse_build_spec_output(raw_output: str) -> BuildSpec:
    payload = _parse_json_mapping(raw_output, error_subject="Build team")
    kind = payload.get("kind")
    if not isinstance(kind, str) or kind.strip() not in BUILD_KIND_VALUES:
        raise FtryCliError(
            "Build team output must include a valid `kind` equal to `agent` or `team`."
        )

    mcp_servers = tuple(
        _parse_built_mcp_server_spec(item, field_name="mcp_servers[]")
        for item in _require_output_list(payload.get("mcp_servers"), field_name="mcp_servers")
    )
    normalized_kind = kind.strip()
    if normalized_kind == BUILD_KIND_AGENT:
        return BuildSpec(
            kind=normalized_kind,
            agent=_parse_built_agent_spec(payload.get("agent"), field_name="agent"),
            mcp_servers=mcp_servers,
        )

    team = _parse_built_team_spec(payload.get("team"))
    raw_agents = payload.get("agents")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise FtryCliError("Build team output must include a non-empty `agents` list for `kind: team`.")
    return BuildSpec(
        kind=normalized_kind,
        team=team,
        agents=tuple(_parse_built_agent_spec(item, field_name="agents[]") for item in raw_agents),
        mcp_servers=mcp_servers,
    )


def _parse_json_mapping(raw_output: str, *, error_subject: str) -> Mapping[str, Any]:
    normalized_output = raw_output.strip()
    candidate_payloads = [normalized_output]
    fenced_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", normalized_output, flags=re.DOTALL)
    if fenced_match:
        candidate_payloads.append(fenced_match.group(1).strip())
    if "{" in normalized_output and "}" in normalized_output:
        candidate_payloads.append(normalized_output[normalized_output.find("{"): normalized_output.rfind("}") + 1].strip())

    seen_candidates: set[str] = set()
    for candidate in candidate_payloads:
        if not candidate or candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        for candidate_variant in (candidate, _repair_builder_json_candidate(candidate)):
            if not candidate_variant:
                continue
            try:
                parsed = json.loads(candidate_variant)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, Mapping):
                return parsed

    raise FtryCliError(f"{error_subject} output is missing the structured JSON payload required by `ftry build`.")


def _repair_builder_json_candidate(candidate: str) -> str:
    repaired_candidate = candidate
    repair_patterns = (
        # Common builder-model mistake: missing comma between a closed object/array and the next property.
        (r'([}\]])(\s*)"([A-Za-z0-9_-]+)"\s*:', r'\1,\2"\3":'),
    )
    for pattern, replacement in repair_patterns:
        repaired_candidate = re.sub(pattern, replacement, repaired_candidate)
    return repaired_candidate


def _parse_built_agent_spec(raw_value: Any, *, field_name: str) -> _BuiltAgentSpec:
    if not isinstance(raw_value, Mapping):
        raise FtryCliError(f"Build team output must include a `{field_name}` object.")
    return _BuiltAgentSpec(
        name=_require_output_text(raw_value.get("name"), f"{field_name}.name"),
        description=_require_optional_output_text(raw_value.get("description"), f"{field_name}.description"),
        prompt=_require_output_text(raw_value.get("prompt"), f"{field_name}.prompt"),
        mcp_servers=Mcp.parse_server_names(raw_value.get("mcp"), field_name=f"{field_name}.mcp", config_kind="build output"),
    )


def _parse_built_team_spec(raw_value: Any) -> _BuiltTeamSpec:
    if not isinstance(raw_value, Mapping):
        raise FtryCliError("Build team output must include a `team` object for `kind: team`.")
    max_turns = raw_value.get("max_turns", DEFAULT_TEAM_MAX_TURNS)
    if not isinstance(max_turns, int) or max_turns <= 0:
        raise FtryCliError("Build team output must include a positive integer `team.max_turns` when provided.")
    return _BuiltTeamSpec(
        name=_require_output_text(raw_value.get("name"), "team.name"),
        description=_require_optional_output_text(raw_value.get("description"), "team.description"),
        prompt=_require_output_text(raw_value.get("prompt"), "team.prompt"),
        max_turns=max_turns,
        mcp_servers=Mcp.parse_server_names(raw_value.get("mcp"), field_name="team.mcp", config_kind="build output"),
    )


def _parse_built_mcp_server_spec(raw_value: Any, *, field_name: str) -> _BuiltMcpServerSpec:
    if not isinstance(raw_value, Mapping):
        raise FtryCliError(f"Build team output must include a `{field_name}` object.")

    name = _require_output_text(raw_value.get("name"), f"{field_name}.name")
    transport = _require_output_text(raw_value.get("transport"), f"{field_name}.transport")
    if transport not in MCP_TRANSPORT_VALUES:
        raise FtryCliError(
            f"Build team output must include a supported `{field_name}.transport` "
            f"from {', '.join(MCP_TRANSPORT_VALUES)}."
        )

    common_kwargs = {
        "name": name,
        "transport": transport,
        "description": _require_optional_output_text(raw_value.get("description"), f"{field_name}.description"),
        "allowed_tools": _require_output_name_list(raw_value.get("allowed_tools"), field_name=f"{field_name}.allowed_tools"),
    }
    return _BuiltMcpServerSpec(
        **common_kwargs,
        **_build_transport_specific_mcp_fields(raw_value, field_name=field_name, transport=transport),
    )


def _build_transport_specific_mcp_fields(
    raw_value: Mapping[str, Any],
    *,
    field_name: str,
    transport: str,
) -> dict[str, Any]:
    if transport == MCP_TRANSPORT_STDIO:
        return {
            "command": _require_optional_output_text_or_none(raw_value.get("command"), f"{field_name}.command"),
            "args": _require_output_text_list(raw_value.get("args"), field_name=f"{field_name}.args"),
            "env": _require_optional_output_mapping(raw_value.get("env"), field_name=f"{field_name}.env"),
        }
    return {
        "url": _require_optional_output_text_or_none(raw_value.get("url"), f"{field_name}.url"),
        "headers": _require_optional_output_mapping(raw_value.get("headers"), field_name=f"{field_name}.headers"),
    }


def _validate_build_spec_mcp(spec: BuildSpec, *, existing_catalog: Sequence[McpConfig]) -> None:
    existing_names = {config.name for config in existing_catalog}
    new_names: set[str] = set()
    for mcp_server in spec.mcp_servers:
        if mcp_server.name in existing_names:
            raise FtryCliError(
                f"Build team output attempted to recreate the existing MCP descriptor `{mcp_server.name}`. "
                "Reuse existing descriptors instead of redefining them."
            )
        if mcp_server.name in new_names:
            raise FtryCliError(f"Build team output defines the MCP descriptor `{mcp_server.name}` more than once.")
        _validate_new_mcp_server_spec(mcp_server)
        new_names.add(mcp_server.name)

    referenced_names = set()
    if spec.agent is not None:
        referenced_names.update(spec.agent.mcp_servers)
    if spec.team is not None:
        referenced_names.update(spec.team.mcp_servers)
    for agent in spec.agents:
        referenced_names.update(agent.mcp_servers)

    unresolved_names = sorted(name for name in referenced_names if name not in existing_names and name not in new_names)
    if unresolved_names:
        joined_names = ", ".join(f"`{name}`" for name in unresolved_names)
        raise FtryCliError(
            "Build team output referenced unknown MCP descriptors. "
            f"Reuse an existing descriptor or define it in `mcp_servers`: {joined_names}"
        )


def _validate_new_mcp_server_spec(mcp_server: _BuiltMcpServerSpec) -> None:
    if mcp_server.transport == MCP_TRANSPORT_STDIO:
        _validate_stdio_mcp_command(mcp_server)
        return
    _validate_network_mcp_url(mcp_server)


def _validate_stdio_mcp_command(mcp_server: _BuiltMcpServerSpec) -> None:
    command = (mcp_server.command or "").strip()
    if not command:
        raise FtryCliError(
            f"Build team output defined MCP descriptor `{mcp_server.name}` with an empty stdio command."
        )
    if _is_placeholder_stdio_command(command):
        raise FtryCliError(
            "Build team output defined MCP descriptor "
            f"`{mcp_server.name}` with a placeholder stdio command `{command}`. "
            "Ask the user for the exact MCP server launch command instead of emitting a placeholder."
        )


def _validate_network_mcp_url(mcp_server: _BuiltMcpServerSpec) -> None:
    url = (mcp_server.url or "").strip()
    if not url:
        raise FtryCliError(
            f"Build team output defined MCP descriptor `{mcp_server.name}` with an empty {mcp_server.transport} URL."
        )
    if _is_placeholder_network_url(url):
        raise FtryCliError(
            "Build team output defined MCP descriptor "
            f"`{mcp_server.name}` with a placeholder {mcp_server.transport} URL `{url}`. "
            "Ask the user for the exact MCP server URL instead of emitting a placeholder."
        )


def _build_mcp_clarification_question(spec: BuildSpec) -> str | None:
    for mcp_server in spec.mcp_servers:
        question = _build_mcp_server_clarification_question(mcp_server)
        if question is not None:
            return question
    return None


def _build_mcp_server_clarification_question(mcp_server: _BuiltMcpServerSpec) -> str | None:
    if mcp_server.transport == MCP_TRANSPORT_STDIO:
        command = (mcp_server.command or "").strip()
        if not command or _is_placeholder_stdio_command(command):
            return (
                f"Which MCP server do you use for `{mcp_server.name}`, and what exact command launches it "
                "in your environment?"
            )
        return None

    url = (mcp_server.url or "").strip()
    if not url or _is_placeholder_network_url(url):
        transport_label = "WebSocket" if mcp_server.transport == MCP_TRANSPORT_WEBSOCKET else "HTTP"
        return f"What is the exact {transport_label} URL for MCP server `{mcp_server.name}` in your environment?"
    return None


def _is_placeholder_stdio_command(command: str) -> bool:
    normalized_command = command.strip().lower().replace("\\", "/")
    if not normalized_command:
        return True
    return any(fragment in normalized_command for fragment in _PLACEHOLDER_STDIO_COMMAND_FRAGMENTS)


def _is_placeholder_network_url(url: str) -> bool:
    normalized_url = url.strip().lower()
    if not normalized_url:
        return True
    return any(fragment in normalized_url for fragment in _PLACEHOLDER_NETWORK_URL_FRAGMENTS)


def _require_output_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FtryCliError(f"Build team output is missing a non-empty `{field_name}`.")
    return value.strip()


def _require_optional_output_text_or_none(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _require_output_text(value, field_name)


def _require_optional_output_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_output_text(value, field_name)


def _require_output_list(value: Any, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise FtryCliError(f"Build team output must include `{field_name}` as a list when provided.")
    return value


def _require_output_text_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    return tuple(
        _require_output_text(item, f"{field_name}[{index}]")
        for index, item in enumerate(_require_output_list(value, field_name=field_name))
    )


def _require_output_name_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    return Mcp.merge_server_names(_require_output_text_list(value, field_name=field_name))


def _require_optional_output_mapping(value: Any, *, field_name: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise FtryCliError(f"Build team output must include `{field_name}` as an object when provided.")
    rendered_mapping: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _require_output_text(raw_key, f"{field_name}.<key>")
        rendered_mapping[key] = _require_output_text(raw_value, f"{field_name}.{key}")
    return rendered_mapping


def _write_build_outputs(spec: BuildSpec, *, output_dir: Path) -> tuple[Path, ...]:
    rendered_files = _render_build_files(spec, output_dir=output_dir)
    _ensure_output_files_do_not_exist(rendered_files)
    created_files: list[Path] = []
    try:
        for path, content in rendered_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            created_files.append(path)
        _validate_generated_outputs(spec, output_dir=output_dir)
    except Exception:
        for path in reversed(created_files):
            if path.exists():
                path.unlink()
        raise
    return tuple(path for path, _ in rendered_files)


def _render_build_files(spec: BuildSpec, *, output_dir: Path) -> tuple[tuple[Path, str], ...]:
    solution_files = (
        _render_agent_build_files(spec, output_dir=output_dir)
        if spec.kind == BUILD_KIND_AGENT
        else _render_team_build_files(spec, output_dir=output_dir)
    )
    mcp_files = _render_mcp_descriptor_files(spec, output_dir=output_dir)
    return solution_files + mcp_files


def _resolve_build_output_dir(spec: BuildSpec, *, output_dir: str | Path | None) -> Path:
    root_dir = Path.cwd() / DEFAULT_BUILD_OUTPUT_DIR_NAME if output_dir is None else Path(output_dir)
    if root_dir.exists() and not root_dir.is_dir():
        raise FtryCliError(f"Build output path is not a directory: {root_dir}")
    root_dir.mkdir(parents=True, exist_ok=True)

    target_dir = root_dir / _build_output_folder_name(spec)
    if target_dir.exists() and not target_dir.is_dir():
        raise FtryCliError(f"Build output path is not a directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _build_output_folder_name(spec: BuildSpec) -> str:
    source_name = (
        spec.agent.name
        if spec.kind == BUILD_KIND_AGENT and spec.agent is not None
        else (spec.team.name if spec.team is not None else f"generated-{spec.kind}")
    )
    fallback_prefix = "agent" if spec.kind == BUILD_KIND_AGENT else "team"
    return _sanitize_agent_name(source_name, fallback_prefix=fallback_prefix).strip("-_").lower() or fallback_prefix


def _render_agent_build_files(spec: BuildSpec, *, output_dir: Path) -> tuple[tuple[Path, str], ...]:
    assert spec.agent is not None
    agent_path = output_dir / "agent.yaml"
    return ((agent_path, _render_agent_yaml(spec.agent)),)


def _render_team_build_files(spec: BuildSpec, *, output_dir: Path) -> tuple[tuple[Path, str], ...]:
    assert spec.team is not None
    rendered_files: list[tuple[Path, str]] = []
    planned_agent_paths = _plan_team_agent_paths(spec.agents, output_dir=output_dir)
    for path, agent_spec in zip(planned_agent_paths, spec.agents):
        rendered_files.append((path, _render_agent_yaml(agent_spec)))
    team_path = output_dir / "team.yaml"
    rendered_files.append((team_path, _render_team_yaml(spec.team, planned_agent_paths, spec.agents)))
    return tuple(rendered_files)


def _render_mcp_descriptor_files(spec: BuildSpec, *, output_dir: Path) -> tuple[tuple[Path, str], ...]:
    if not spec.mcp_servers:
        return ()
    return tuple(
        (
            output_dir / _build_mcp_descriptor_file_name(mcp_server.name),
            _render_mcp_descriptor_yaml(mcp_server),
        )
        for mcp_server in spec.mcp_servers
    )


def _build_mcp_descriptor_file_name(server_name: str) -> str:
    stem = _sanitize_agent_name(server_name, fallback_prefix="mcp").strip("-_").lower() or "mcp"
    return f"{LOCAL_MCP_DESCRIPTOR_FILE_PREFIX}{stem}.yaml"


def _ensure_output_files_do_not_exist(rendered_files: Sequence[tuple[Path, str]]) -> None:
    conflicting_paths = [path for path, _ in rendered_files if path.exists()]
    if conflicting_paths:
        joined_paths = ", ".join(str(path) for path in conflicting_paths)
        raise FtryCliError(
            "Build output would overwrite existing files. "
            f"Remove or rename them first: {joined_paths}"
        )


def _validate_generated_outputs(spec: BuildSpec, *, output_dir: Path) -> None:
    if spec.kind == BUILD_KIND_AGENT:
        StandaloneAgent.from_file(output_dir / "agent.yaml")
        return
    Team.from_file(output_dir / "team.yaml")


def _plan_team_agent_paths(agent_specs: Sequence[_BuiltAgentSpec], *, output_dir: Path) -> tuple[Path, ...]:
    used_names: set[str] = set()
    planned_paths: list[Path] = []
    for index, agent_spec in enumerate(agent_specs, start=1):
        stem = _sanitize_agent_name(agent_spec.name, fallback_prefix=f"agent-{index}").strip("-").lower()
        if not stem:
            stem = f"agent-{index}"
        candidate_name = f"agent-{stem}.yaml"
        suffix = 2
        while candidate_name in used_names:
            candidate_name = f"agent-{stem}-{suffix}.yaml"
            suffix += 1
        used_names.add(candidate_name)
        planned_paths.append(output_dir / candidate_name)
    return tuple(planned_paths)


def _render_agent_yaml(agent_spec: _BuiltAgentSpec) -> str:
    lines = [
        f"name: {json.dumps(agent_spec.name, ensure_ascii=False)}",
        "",
    ]
    if agent_spec.description is not None:
        lines.extend(_render_block_field("description", agent_spec.description))
        lines.append("")
    lines.extend(
        [
            "model:",
            f"  name: {json.dumps(DEFAULT_MODEL_NAME, ensure_ascii=False)}",
            f"  provider: {json.dumps(DEFAULT_MODEL_PROVIDER, ensure_ascii=False)}",
            f"  api-key: {json.dumps(DEFAULT_MODEL_API_KEY, ensure_ascii=False)}",
        ]
    )
    if agent_spec.mcp_servers:
        lines.extend(["", "mcp:"])
        lines.extend(f"  - {json.dumps(mcp_server_name, ensure_ascii=False)}" for mcp_server_name in agent_spec.mcp_servers)
    lines.extend([""])
    lines.extend(_render_block_field("prompt", agent_spec.prompt))
    return "\n".join(lines).rstrip() + "\n"


def _render_team_yaml(
    team_spec: _BuiltTeamSpec,
    agent_paths: Sequence[Path],
    agent_specs: Sequence[_BuiltAgentSpec],
) -> str:
    lines = [
        f"name: {json.dumps(team_spec.name, ensure_ascii=False)}",
        "",
    ]
    if team_spec.description is not None:
        lines.extend(_render_block_field("description", team_spec.description))
        lines.append("")
    lines.extend(
        [
            "model:",
            f"  name: {json.dumps(DEFAULT_MODEL_NAME, ensure_ascii=False)}",
            f"  provider: {json.dumps(DEFAULT_MODEL_PROVIDER, ensure_ascii=False)}",
            f"  api-key: {json.dumps(DEFAULT_MODEL_API_KEY, ensure_ascii=False)}",
            "",
        ]
    )
    if team_spec.mcp_servers:
        lines.append("mcp:")
        lines.extend(f"  - {json.dumps(mcp_server_name, ensure_ascii=False)}" for mcp_server_name in team_spec.mcp_servers)
        lines.append("")
    lines.extend(
        [
            "termination:",
            f"  max-turns: {team_spec.max_turns}",
            "",
            "agents:",
        ]
    )
    for agent_path, agent_spec in zip(agent_paths, agent_specs):
        lines.append(f"  - file: ./{agent_path.name}")
        if agent_spec.mcp_servers:
            lines.append("    mcp:")
            lines.extend(
                f"      - {json.dumps(mcp_server_name, ensure_ascii=False)}"
                for mcp_server_name in agent_spec.mcp_servers
            )
    lines.append("")
    lines.extend(_render_block_field("prompt", team_spec.prompt))
    return "\n".join(lines).rstrip() + "\n"


def _render_mcp_descriptor_yaml(mcp_server: _BuiltMcpServerSpec) -> str:
    lines = [
        f"name: {json.dumps(mcp_server.name, ensure_ascii=False)}",
        f"transport: {json.dumps(mcp_server.transport, ensure_ascii=False)}",
    ]
    if mcp_server.description is not None:
        lines.extend(["", * _render_block_field("description", mcp_server.description)])
    if mcp_server.allowed_tools:
        lines.extend(["", "allowed-tools:"])
        lines.extend(
            f"  - {json.dumps(tool_name, ensure_ascii=False)}"
            for tool_name in mcp_server.allowed_tools
        )
    if mcp_server.transport == MCP_TRANSPORT_STDIO:
        lines.extend(
            [
                "",
                f"command: {json.dumps(mcp_server.command or '', ensure_ascii=False)}",
            ]
        )
        if mcp_server.args:
            lines.extend(["args:"])
            lines.extend(f"  - {json.dumps(arg, ensure_ascii=False)}" for arg in mcp_server.args)
        if mcp_server.env:
            lines.extend(["env:"])
            lines.extend(
                f"  {json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}"
                for key, value in mcp_server.env.items()
            )
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(["", f"url: {json.dumps(mcp_server.url or '', ensure_ascii=False)}"])
    if mcp_server.headers:
        lines.extend(["headers:"])
        lines.extend(
            f"  {json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}"
            for key, value in mcp_server.headers.items()
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_block_field(field_name: str, value: str) -> list[str]:
    normalized_lines = value.splitlines() or [value]
    rendered_lines = [f"{field_name}: |"]
    for line in normalized_lines:
        rendered_lines.append(f"  {line}" if line else "  ")
    return rendered_lines
