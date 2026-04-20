from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .StandaloneAgent import StandaloneAgent
from .Team import Team
from .Tools import FtryCliError, _sanitize_agent_name

INTERNAL_BUILDER_TEAM_FILE = Path(__file__).resolve().parent / "builder" / "team.yaml"
DEFAULT_BUILD_OUTPUT_DIR_NAME = "output"
DEFAULT_MODEL_NAME = "gpt-4o-2024-08-06"
DEFAULT_MODEL_PROVIDER = "openai"
DEFAULT_MODEL_API_KEY = "env:OAI_API_KEY"
DEFAULT_TEAM_MAX_TURNS = 6
BUILD_KIND_AGENT = "agent"
BUILD_KIND_TEAM = "team"
BUILD_KIND_VALUES = (BUILD_KIND_AGENT, BUILD_KIND_TEAM)


@dataclass(frozen=True)
class _BuiltAgentSpec:
    name: str
    prompt: str
    description: str | None = None


@dataclass(frozen=True)
class _BuiltTeamSpec:
    name: str
    prompt: str
    max_turns: int = DEFAULT_TEAM_MAX_TURNS
    description: str | None = None


@dataclass(frozen=True)
class BuildSpec:
    kind: str
    agent: _BuiltAgentSpec | None = None
    team: _BuiltTeamSpec | None = None
    agents: tuple[_BuiltAgentSpec, ...] = ()


def build_from_prompt(prompt: str, *, output_dir: str | Path | None = None) -> tuple[Path, ...]:
    raw_output = _run_builder_team(prompt)
    spec = _parse_build_spec_output(raw_output)
    target_dir = _resolve_build_output_dir(spec, output_dir=output_dir)
    return _write_build_outputs(spec, output_dir=target_dir)


def _run_builder_team(prompt: str) -> str:
    builder_team = Team.from_file(INTERNAL_BUILDER_TEAM_FILE)
    return asyncio.run(builder_team.run(prompt))


def _parse_build_spec_output(raw_output: str) -> BuildSpec:
    payload = _parse_json_mapping(raw_output, error_subject="Build team")
    kind = payload.get("kind")
    if not isinstance(kind, str) or kind.strip() not in BUILD_KIND_VALUES:
        raise FtryCliError(
            "Build team output must include a valid `kind` equal to `agent` or `team`."
        )

    normalized_kind = kind.strip()
    if normalized_kind == BUILD_KIND_AGENT:
        return BuildSpec(
            kind=normalized_kind,
            agent=_parse_built_agent_spec(payload.get("agent"), field_name="agent"),
        )

    team = _parse_built_team_spec(payload.get("team"))
    raw_agents = payload.get("agents")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise FtryCliError("Build team output must include a non-empty `agents` list for `kind: team`.")
    return BuildSpec(
        kind=normalized_kind,
        team=team,
        agents=tuple(_parse_built_agent_spec(item, field_name="agents[]") for item in raw_agents),
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
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return parsed

    raise FtryCliError(f"{error_subject} output is missing the structured JSON payload required by `ftry build`.")


def _parse_built_agent_spec(raw_value: Any, *, field_name: str) -> _BuiltAgentSpec:
    if not isinstance(raw_value, Mapping):
        raise FtryCliError(f"Build team output must include a `{field_name}` object.")
    return _BuiltAgentSpec(
        name=_require_output_text(raw_value.get("name"), f"{field_name}.name"),
        description=_require_optional_output_text(raw_value.get("description"), f"{field_name}.description"),
        prompt=_require_output_text(raw_value.get("prompt"), f"{field_name}.prompt"),
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
    )


def _require_output_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FtryCliError(f"Build team output is missing a non-empty `{field_name}`.")
    return value.strip()


def _require_optional_output_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_output_text(value, field_name)


def _write_build_outputs(spec: BuildSpec, *, output_dir: Path) -> tuple[Path, ...]:
    rendered_files = (
        _render_agent_build_files(spec, output_dir=output_dir)
        if spec.kind == BUILD_KIND_AGENT
        else _render_team_build_files(spec, output_dir=output_dir)
    )
    _ensure_output_files_do_not_exist(rendered_files)
    created_files: list[Path] = []
    try:
        for path, content in rendered_files:
            path.write_text(content, encoding="utf-8")
            created_files.append(path)
        _validate_generated_outputs(spec, output_dir=output_dir)
    except Exception:
        for path in reversed(created_files):
            if path.exists():
                path.unlink()
        raise
    return tuple(path for path, _ in rendered_files)


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
    rendered_files.append((team_path, _render_team_yaml(spec.team, planned_agent_paths)))
    return tuple(rendered_files)


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
            "",
        ]
    )
    lines.extend(_render_block_field("prompt", agent_spec.prompt))
    return "\n".join(lines).rstrip() + "\n"


def _render_team_yaml(team_spec: _BuiltTeamSpec, agent_paths: Sequence[Path]) -> str:
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
            "termination:",
            f"  max-turns: {team_spec.max_turns}",
            "",
            "agents:",
        ]
    )
    for agent_path in agent_paths:
        lines.append(f"  - file: ./{agent_path.name}")
    lines.append("")
    lines.extend(_render_block_field("prompt", team_spec.prompt))
    return "\n".join(lines).rstrip() + "\n"


def _render_block_field(field_name: str, value: str) -> list[str]:
    normalized_lines = value.splitlines() or [value]
    rendered_lines = [f"{field_name}: |"]
    for line in normalized_lines:
        rendered_lines.append(f"  {line}" if line else "  ")
    return rendered_lines
