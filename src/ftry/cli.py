"""Command-line interface for ftry."""

from __future__ import annotations

import asyncio
import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RESET = "\033[0m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_PINK = "\033[95m"
PURPLE = "\033[38;5;141m"
ORANGE = "\033[38;5;214m"

MOCK_COMMANDS = ("build", "break", "land")
LINE_COLOR_TOKENS = {
    "[cyan]": BRIGHT_CYAN,
    "[blue]": BRIGHT_BLUE,
    "[pink]": BRIGHT_PINK,
    "[purple]": PURPLE,
    "[yellow]": BRIGHT_YELLOW,
    "[orange]": ORANGE,
    "[green]": BRIGHT_GREEN,
    "[reset]": RESET,
}

TEAM_PATTERN_ALIASES = {
    "concurrent": "concurrent",
    "group-chat": "group-chat",
    "group_chat": "group-chat",
    "groupchat": "group-chat",
    "handoff": "handoff",
    "magentic": "magentic",
    "magentic-one": "magentic",
    "sequential": "sequential",
}
HANDOFF_HINTS = ("handoff", "route", "routing", "triage", "delegate", "delegation", "transfer")
MAGENTIC_HINTS = ("magentic", "manager", "plan", "planner", "replan", "open-ended", "complex task")
CONCURRENT_HINTS = ("parallel", "concurrent", "independent", "simultaneous", "simultaneously")
GROUP_CHAT_HINTS = (
    "collabor",
    "conversation",
    "discussion",
    "iterat",
    "feedback",
    "review",
    "rework",
    "select the most appropriate",
    "shared",
)
SEQUENTIAL_HINTS = ("sequential", "sequence", "pipeline", "stage", "step", "first", "then", "finally", "ensuite")
HANDOFF_START_HINTS = ("triage", "router", "routing", "route", "dispatcher", "coordinator")


class FtryCliError(Exception):
    """Raised when a user-facing CLI error occurs."""


@dataclass(frozen=True)
class AgentModelConfig:
    name: str
    provider: str
    api_key: str


@dataclass(frozen=True)
class AgentConfig:
    name: str
    instructions: str
    model: AgentModelConfig
    description: str | None = None


@dataclass(frozen=True)
class TeamTerminationConfig:
    max_turns: int | None = None


@dataclass(frozen=True)
class TeamConfig:
    name: str
    instructions: str
    agents: tuple[AgentConfig, ...]
    model: AgentModelConfig | None = None
    description: str | None = None
    pattern: str | None = None
    termination: TeamTerminationConfig = field(default_factory=TeamTerminationConfig)


def _load_line_banner() -> str:
    content = files("ftry").joinpath("line.txt").read_text(encoding="utf-8")
    rendered_lines: list[str] = []

    for raw_line in content.splitlines():
        line = raw_line
        uses_color = any(token in raw_line for token in LINE_COLOR_TOKENS if token != "[reset]")
        for token, value in LINE_COLOR_TOKENS.items():
            line = line.replace(token, value)
        if uses_color and not line.endswith(RESET):
            line = f"{line}{RESET}"
        rendered_lines.append(line)

    return "\n".join(rendered_lines).rstrip()


def _run_mock_command(command: str) -> int:
    print(command)
    return 0


def _run_line_command() -> int:
    print(_load_line_banner())
    return 0


def _load_yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - covered by CLI error path
        raise FtryCliError(
            "PyYAML is required for `ftry pop`. Reinstall the project with `python -m pip install -e .`."
        ) from exc

    return yaml


def _load_dotenv_function() -> Callable[..., bool]:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - covered by CLI error path
        raise FtryCliError(
            "python-dotenv is required for `ftry pop` to load `.env` files. "
            "Reinstall the project with `python -m pip install -e .`."
        ) from exc

    return load_dotenv


def _find_dotenv_path(agent_path: Path) -> Path | None:
    search_roots = [Path.cwd(), agent_path.resolve().parent]

    seen: set[Path] = set()
    for root in search_roots:
        for candidate_dir in (root, *root.parents):
            if candidate_dir in seen:
                continue
            seen.add(candidate_dir)

            candidate = candidate_dir / ".env"
            if candidate.is_file():
                return candidate

    return None


def _load_dotenv_for_config(config_path: Path) -> None:
    dotenv_path = _find_dotenv_path(config_path)
    if dotenv_path is None:
        return

    load_dotenv = _load_dotenv_function()
    load_dotenv(dotenv_path=dotenv_path, override=False)


def _require_non_empty_string(value: Any, field_name: str, config_kind: str = "agent") -> str:
    if not isinstance(value, str) or not value.strip():
        raise FtryCliError(f"Invalid or missing `{field_name}` in {config_kind} YAML.")
    return value.strip()


def _require_optional_string(value: Any, field_name: str, config_kind: str = "agent") -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name, config_kind)


def _require_mapping(value: Any, field_name: str, config_kind: str = "agent") -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise FtryCliError(f"Invalid or missing `{field_name}` mapping in {config_kind} YAML.")
    return value


def _require_sequence(value: Any, field_name: str, config_kind: str = "agent") -> Sequence[Any]:
    if not isinstance(value, list):
        raise FtryCliError(f"Invalid or missing `{field_name}` list in {config_kind} YAML.")
    return value


def _require_positive_int(value: Any, field_name: str, config_kind: str = "agent") -> int:
    if not isinstance(value, int) or value <= 0:
        raise FtryCliError(f"Invalid or missing `{field_name}` in {config_kind} YAML: expected a positive integer.")
    return value


def _resolve_secret(value: str) -> str:
    if not value.startswith("env:"):
        return value

    env_name = value.removeprefix("env:").strip()
    if not env_name:
        raise FtryCliError("Invalid `model.api-key`: environment variable name is missing after `env:`.")

    secret = os.getenv(env_name)
    if not secret:
        raise FtryCliError(f"Environment variable `{env_name}` is not set.")
    return secret


def _sanitize_agent_name(value: str, *, fallback_prefix: str = "agent") -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip())
    sanitized = sanitized.strip("-_")
    return sanitized or fallback_prefix


def _resolve_config_path(config_file: str | Path, *, base_dir: Path | None = None) -> Path:
    config_path = Path(config_file).expanduser()
    if config_path.is_absolute() or base_dir is None:
        return config_path

    if config_path.is_file():
        return config_path

    config_path = base_dir / config_path
    return config_path


def _load_yaml_mapping(config_path: Path, *, config_kind: str) -> Mapping[str, Any]:
    yaml = _load_yaml_module()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _require_mapping(raw_config, "root", config_kind)


def _parse_model_config(raw_model: Any, *, config_kind: str, required: bool) -> AgentModelConfig | None:
    if raw_model is None and not required:
        return None

    model_config = _require_mapping(raw_model, "model", config_kind)
    return AgentModelConfig(
        name=_require_non_empty_string(model_config.get("name"), "model.name", config_kind),
        provider=_require_non_empty_string(model_config.get("provider"), "model.provider", config_kind),
        api_key=_resolve_secret(_require_non_empty_string(model_config.get("api-key"), "model.api-key", config_kind)),
    )


def _parse_agent_config(config: Mapping[str, Any], *, config_kind: str) -> AgentConfig:
    model = _parse_model_config(config.get("model"), config_kind=config_kind, required=True)
    assert model is not None
    return AgentConfig(
        name=_require_non_empty_string(config.get("name"), "name", config_kind),
        description=_require_optional_string(config.get("description"), "description", config_kind),
        instructions=_require_non_empty_string(config.get("prompt"), "prompt", config_kind),
        model=model,
    )


def _load_agent_config(agent_file: str | Path, *, base_dir: Path | None = None) -> AgentConfig:
    agent_path = _resolve_config_path(agent_file, base_dir=base_dir)
    if not agent_path.is_file():
        raise FtryCliError(f"Agent file not found: {agent_path}")

    _load_dotenv_for_config(agent_path)
    return _parse_agent_config(_load_yaml_mapping(agent_path, config_kind="agent"), config_kind="agent")


def _normalize_team_pattern(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    pattern = TEAM_PATTERN_ALIASES.get(normalized)
    if pattern is None:
        raise FtryCliError(
            f"Unsupported team pattern `{value}`. Expected one of: sequential, concurrent, handoff, group-chat, magentic."
        )
    return pattern


def _load_team_agent_config(raw_agent: Any, *, team_dir: Path) -> AgentConfig:
    if isinstance(raw_agent, str):
        return _load_agent_config(raw_agent, base_dir=team_dir)

    agent_config = _require_mapping(raw_agent, "agents[]", "team")
    if "file" in agent_config:
        if len(agent_config) != 1:
            raise FtryCliError("Invalid `agents[]` entry in team YAML: `file` references cannot be mixed with inline fields.")
        return _load_agent_config(_require_non_empty_string(agent_config.get("file"), "agents[].file", "team"), base_dir=team_dir)

    return _parse_agent_config(agent_config, config_kind="team agent")


def _parse_team_termination(raw_termination: Any) -> TeamTerminationConfig:
    if raw_termination is None:
        return TeamTerminationConfig()

    termination_config = _require_mapping(raw_termination, "termination", "team")
    max_turns = termination_config.get("max-turns")
    if max_turns is None:
        return TeamTerminationConfig()

    return TeamTerminationConfig(max_turns=_require_positive_int(max_turns, "termination.max-turns", "team"))


def _load_team_config(team_file: str | Path) -> TeamConfig:
    team_path = _resolve_config_path(team_file)
    if not team_path.is_file():
        raise FtryCliError(f"Team file not found: {team_path}")

    _load_dotenv_for_config(team_path)
    config = _load_yaml_mapping(team_path, config_kind="team")
    raw_agents = _require_sequence(config.get("agents"), "agents", "team")
    agents = tuple(_load_team_agent_config(raw_agent, team_dir=team_path.parent) for raw_agent in raw_agents)
    if not agents:
        raise FtryCliError("Invalid or missing `agents` list in team YAML.")

    raw_pattern = config.get("pattern", config.get("orchestration"))
    return TeamConfig(
        name=_require_non_empty_string(config.get("name"), "name", "team"),
        description=_require_optional_string(config.get("description"), "description", "team"),
        instructions=_require_non_empty_string(config.get("prompt"), "prompt", "team"),
        agents=agents,
        model=_parse_model_config(config.get("model"), config_kind="team", required=False),
        pattern=_normalize_team_pattern(raw_pattern) if raw_pattern is not None else None,
        termination=_parse_team_termination(config.get("termination")),
    )


def _render_role_summary(agent: AgentConfig) -> str:
    source = agent.description or agent.instructions
    return " ".join(source.split())


def _render_team_instructions(team: TeamConfig) -> str:
    participants = ", ".join(agent.name for agent in team.agents)
    roles = "\n".join(f"- {agent.name}: {_render_role_summary(agent)}" for agent in team.agents)
    return team.instructions.replace("{participants}", participants).replace("{roles}", roles)


def _compose_pattern_analysis_text(team: TeamConfig) -> str:
    fragments = [team.name, team.description or "", team.instructions]
    for agent in team.agents:
        fragments.extend((agent.name, agent.description or "", agent.instructions))
    return " ".join(fragment for fragment in fragments if fragment).lower()


def _contains_any(text: str, hints: Sequence[str]) -> bool:
    return any(hint in text for hint in hints)


def _has_numbered_steps(text: str) -> bool:
    lowered = text.lower()
    return "1." in lowered and "2." in lowered


def _infer_team_pattern(team: TeamConfig) -> str:
    if team.pattern is not None:
        return team.pattern

    analysis_text = _compose_pattern_analysis_text(team)
    if _contains_any(analysis_text, HANDOFF_HINTS):
        return "handoff"
    if _contains_any(analysis_text, MAGENTIC_HINTS):
        return "magentic"
    if _contains_any(analysis_text, CONCURRENT_HINTS) and not _contains_any(analysis_text, SEQUENTIAL_HINTS):
        return "concurrent"
    if _contains_any(analysis_text, GROUP_CHAT_HINTS):
        return "group-chat"
    if _contains_any(analysis_text, SEQUENTIAL_HINTS) or _has_numbered_steps(team.instructions):
        return "sequential"
    return "group-chat"


def _format_agent_output(result: object, *, author_name_map: Mapping[str, str] | None = None) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    messages = getattr(result, "messages", None)
    if isinstance(messages, list):
        rendered_messages: list[str] = []
        for message in messages:
            message_text = getattr(message, "text", None)
            message_role = getattr(message, "role", None)
            if not isinstance(message_text, str) or not message_text.strip() or message_role == "user":
                continue

            author_name = getattr(message, "author_name", None)
            if isinstance(author_name, str) and author_name.strip():
                display_name = author_name_map.get(author_name.strip(), author_name.strip()) if author_name_map else author_name.strip()
                rendered_messages.append(f"[{display_name}]\n{message_text.strip()}")
            else:
                rendered_messages.append(message_text.strip())

        if rendered_messages:
            return "\n\n".join(rendered_messages)
    return str(result)


def _create_openai_agent(
    config: AgentConfig,
    *,
    extra_instructions: str | None = None,
    name_override: str | None = None,
) -> Any:
    try:
        from agent_framework.openai import OpenAIChatCompletionClient
    except ImportError as exc:  # pragma: no cover - covered by CLI error path
        raise FtryCliError(
            "Microsoft Agent Framework OpenAI support is required for `ftry pop`. "
            "Reinstall the project with `python -m pip install -e .`."
        ) from exc

    instructions = config.instructions
    if extra_instructions:
        instructions = f"{config.instructions}\n\n<TeamContext>\n{extra_instructions}\n</TeamContext>"

    return OpenAIChatCompletionClient(
        model=config.model.name,
        api_key=config.model.api_key,
    ).as_agent(
        name=name_override or _sanitize_agent_name(config.name),
        description=config.description,
        instructions=instructions,
    )


async def _run_openai_agent(config: AgentConfig, prompt: str) -> str:
    agent = _create_openai_agent(config)
    result = await agent.run(prompt)
    return _format_agent_output(result)


async def _run_agent_prompt(config: AgentConfig, prompt: str) -> str:
    provider = config.model.provider.lower()
    if provider != "openai":
        raise FtryCliError(f"Unsupported provider `{config.model.provider}`. Only `openai` is supported for now.")

    return await _run_openai_agent(config, prompt)


def _load_orchestration_builders() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from agent_framework.orchestrations import (
            ConcurrentBuilder,
            GroupChatBuilder,
            HandoffBuilder,
            MagenticBuilder,
            SequentialBuilder,
        )
    except ImportError as exc:  # pragma: no cover - covered by CLI error path
        raise FtryCliError(
            "Microsoft Agent Framework orchestration support is required for `ftry pop -t`. "
            "Reinstall the project with `python -m pip install -e .`."
        ) from exc

    return SequentialBuilder, ConcurrentBuilder, HandoffBuilder, GroupChatBuilder, MagenticBuilder


def _create_team_controller_agent(team: TeamConfig, *, instructions: str) -> Any | None:
    if team.model is None:
        return None

    return _create_openai_agent(
        AgentConfig(
            name=team.name,
            description=team.description,
            instructions=instructions,
            model=team.model,
        ),
        name_override=_sanitize_agent_name(team.name, fallback_prefix="team"),
    )


def _select_handoff_start_agent(agents: Sequence[Any]) -> Any:
    for agent in agents:
        name = getattr(agent, "name", "")
        description = getattr(agent, "description", "")
        instructions = getattr(agent, "instructions", "")
        analysis_text = f"{name} {description} {instructions}".lower()
        if _contains_any(analysis_text, HANDOFF_START_HINTS):
            return agent
    return agents[0]


def _count_assistant_messages(conversation: list[Any]) -> int:
    return sum(1 for message in conversation if getattr(message, "role", None) == "assistant")


def _build_team_participants(team: TeamConfig, *, extra_instructions: str | None) -> tuple[list[Any], dict[str, str]]:
    participants: list[Any] = []
    author_name_map: dict[str, str] = {}
    used_names: set[str] = set()

    for index, agent in enumerate(team.agents, start=1):
        candidate_name = _sanitize_agent_name(agent.name, fallback_prefix=f"agent-{index}")
        unique_name = candidate_name
        suffix = 2
        while unique_name in used_names:
            unique_name = f"{candidate_name}-{suffix}"
            suffix += 1
        used_names.add(unique_name)
        author_name_map[unique_name] = agent.name
        participants.append(_create_openai_agent(agent, extra_instructions=extra_instructions, name_override=unique_name))

    return participants, author_name_map


def _build_team_workflow(team: TeamConfig) -> tuple[str, Any, Mapping[str, str]]:
    pattern = _infer_team_pattern(team)
    rendered_instructions = _render_team_instructions(team)
    inject_team_context = pattern in {"sequential", "concurrent", "handoff"} or team.model is None
    participants, author_name_map = _build_team_participants(
        team,
        extra_instructions=rendered_instructions if inject_team_context else None,
    )
    SequentialBuilder, ConcurrentBuilder, HandoffBuilder, GroupChatBuilder, MagenticBuilder = _load_orchestration_builders()
    author_name_map[_sanitize_agent_name(team.name, fallback_prefix="team")] = team.name

    if pattern == "sequential":
        return pattern, SequentialBuilder(participants=participants).build(), author_name_map

    if pattern == "concurrent":
        return pattern, ConcurrentBuilder(participants=participants).build(), author_name_map

    if pattern == "handoff":
        handoff_builder = HandoffBuilder(name=team.name, participants=participants, description=team.description)
        handoff_builder = handoff_builder.with_start_agent(_select_handoff_start_agent(participants))
        autonomous_kwargs: dict[str, Any] = {"agents": participants}
        if team.termination.max_turns is not None:
            autonomous_kwargs["turn_limits"] = {getattr(agent, "name"): team.termination.max_turns for agent in participants}
            handoff_builder = handoff_builder.with_termination_condition(
                lambda conversation: _count_assistant_messages(conversation) >= team.termination.max_turns
            )
        handoff_builder = handoff_builder.with_autonomous_mode(**autonomous_kwargs)
        return pattern, handoff_builder.build(), author_name_map

    if pattern == "group-chat":
        group_chat_builder = GroupChatBuilder(
            participants=participants,
            orchestrator_agent=_create_team_controller_agent(team, instructions=rendered_instructions),
            max_rounds=team.termination.max_turns,
        )
        return pattern, group_chat_builder.build(), author_name_map

    return (
        pattern,
        MagenticBuilder(
            participants=participants,
            manager_agent=_create_team_controller_agent(team, instructions=rendered_instructions),
            max_round_count=team.termination.max_turns,
        ).build(),
        author_name_map,
    )


async def _run_team_prompt(team: TeamConfig, prompt: str) -> str:
    _, workflow, author_name_map = _build_team_workflow(team)
    team_agent = workflow.as_agent(name=_sanitize_agent_name(team.name, fallback_prefix="team"))
    result = await team_agent.run(prompt)
    return _format_agent_output(result, author_name_map=author_name_map)


def _run_pop_command(agent_file: str | None, team_file: str | None, prompt: str) -> int:
    if team_file is not None:
        config = _load_team_config(team_file)
        result = asyncio.run(_run_team_prompt(config, prompt))
    else:
        if agent_file is None:
            raise FtryCliError("Either `-a/--agent-file` or `-t/--team-file` must be provided.")
        config = _load_agent_config(agent_file)
        result = asyncio.run(_run_agent_prompt(config, prompt))

    print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftry", description="Mock CLI for ftry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in MOCK_COMMANDS:
        subparser = subparsers.add_parser(command, help=f"Mock {command} command.")
        subparser.set_defaults(handler=lambda args, command=command: _run_mock_command(command))

    line_parser = subparsers.add_parser("line", help='Display "First Try" as colored ASCII art.')
    line_parser.set_defaults(handler=lambda args: _run_line_command())

    pop_parser = subparsers.add_parser("pop", help="Run an agent described in a YAML file.")
    pop_source_group = pop_parser.add_mutually_exclusive_group(required=True)
    pop_source_group.add_argument(
        "-a",
        "--agent-file",
        help="Path to the YAML agent description file.",
    )
    pop_source_group.add_argument(
        "-t",
        "--team-file",
        help="Path to the YAML team description file.",
    )
    pop_parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="Prompt to send to the loaded agent.",
    )
    pop_parser.set_defaults(handler=lambda args: _run_pop_command(args.agent_file, args.team_file, args.prompt))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler

    try:
        return handler(args)
    except FtryCliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
