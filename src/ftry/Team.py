from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .Agent import (
    AgentConfig,
    AgentModelConfig,
    _create_openai_agent,
    _load_agent_config,
    _parse_agent_config,
    _parse_model_config,
)
from .Tools import (
    FtryCliError,
    _build_agent_trace_colors,
    _display_name,
    _extract_trace_chunk,
    _format_final_team_output,
    _load_dotenv_for_config,
    _load_yaml_mapping,
    _require_mapping,
    _require_non_empty_string,
    _require_optional_string,
    _require_positive_int,
    _require_sequence,
    _resolve_config_path,
    _sanitize_agent_name,
    _summarize_payload,
    _summarize_trace_text,
    _trace,
    _trace_block,
    _trace_result,
    _trace_route,
    _trace_team_label,
    _trace_team_start,
)


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


def _normalize_team_pattern(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    pattern = TEAM_PATTERN_ALIASES.get(normalized)
    if pattern is None:
        raise FtryCliError(
            f"Unsupported team pattern `{value}`. Expected one of: sequential, concurrent, handoff, group-chat, magentic."
        )
    return pattern


def _load_team_agent_config(
    raw_agent: Any,
    *,
    team_dir: Path,
    load_agent_config: Callable[[str | Path], AgentConfig] | Callable[..., AgentConfig] = _load_agent_config,
    parse_agent_config: Callable[[Mapping[str, Any]], AgentConfig] | Callable[..., AgentConfig] = _parse_agent_config,
) -> AgentConfig:
    if isinstance(raw_agent, str):
        return load_agent_config(raw_agent, base_dir=team_dir)

    agent_config = _require_mapping(raw_agent, "agents[]", "team")
    if "file" in agent_config:
        if len(agent_config) != 1:
            raise FtryCliError("Invalid `agents[]` entry in team YAML: `file` references cannot be mixed with inline fields.")
        return load_agent_config(_require_non_empty_string(agent_config.get("file"), "agents[].file", "team"), base_dir=team_dir)

    return parse_agent_config(agent_config, config_kind="team agent")


def _parse_team_termination(raw_termination: Any) -> TeamTerminationConfig:
    if raw_termination is None:
        return TeamTerminationConfig()

    termination_config = _require_mapping(raw_termination, "termination", "team")
    max_turns = termination_config.get("max-turns")
    if max_turns is None:
        return TeamTerminationConfig()

    return TeamTerminationConfig(max_turns=_require_positive_int(max_turns, "termination.max-turns", "team"))


def _load_team_config(
    team_file: str | Path,
    *,
    resolve_config_path: Callable[[str | Path], Path] | Callable[..., Path] = _resolve_config_path,
    load_dotenv_for_config: Callable[[Path], None] = _load_dotenv_for_config,
    load_yaml_mapping: Callable[[Path], Mapping[str, Any]] | Callable[..., Mapping[str, Any]] = _load_yaml_mapping,
    load_team_agent_config: Callable[[Any], AgentConfig] | Callable[..., AgentConfig] = _load_team_agent_config,
    parse_model_config: Callable[[Any], AgentModelConfig | None] | Callable[..., AgentModelConfig | None] = _parse_model_config,
) -> TeamConfig:
    team_path = resolve_config_path(team_file)
    if not team_path.is_file():
        raise FtryCliError(f"Team file not found: {team_path}")

    load_dotenv_for_config(team_path)
    config = load_yaml_mapping(team_path, config_kind="team")
    raw_agents = _require_sequence(config.get("agents"), "agents", "team")
    agents = tuple(load_team_agent_config(raw_agent, team_dir=team_path.parent) for raw_agent in raw_agents)
    if not agents:
        raise FtryCliError("Invalid or missing `agents` list in team YAML.")

    raw_pattern = config.get("pattern", config.get("orchestration"))
    return TeamConfig(
        name=_require_non_empty_string(config.get("name"), "name", "team"),
        description=_require_optional_string(config.get("description"), "description", "team"),
        instructions=_require_non_empty_string(config.get("prompt"), "prompt", "team"),
        agents=agents,
        model=parse_model_config(config.get("model"), config_kind="team", required=False),
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
    team_internal_name = _sanitize_agent_name(team.name, fallback_prefix="team")
    author_name_map[team_internal_name] = team.name

    if pattern == "sequential":
        return (
            pattern,
            SequentialBuilder(participants=participants, intermediate_outputs=True).build(),
            author_name_map,
        )

    if pattern == "concurrent":
        return (
            pattern,
            ConcurrentBuilder(participants=participants, intermediate_outputs=True).build(),
            author_name_map,
        )

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
            orchestrator_name=team.name,
            max_rounds=team.termination.max_turns,
            intermediate_outputs=True,
        )
        return pattern, group_chat_builder.build(), author_name_map

    return (
        pattern,
        MagenticBuilder(
            participants=participants,
            manager_agent=_create_team_controller_agent(team, instructions=rendered_instructions),
            max_round_count=team.termination.max_turns,
            intermediate_outputs=True,
        ).build(),
        author_name_map,
    )


async def _run_team_prompt(team: TeamConfig, prompt: str) -> str:
    _, workflow, author_name_map = _build_team_workflow(team)
    pattern = _infer_team_pattern(team)
    team_name = team.name
    agent_trace_colors = _build_agent_trace_colors([agent.name for agent in team.agents])
    last_visible_input = prompt
    final_payload: Any = None
    active_executor: str | None = None
    buffered_outputs: list[str] = []
    expected_invoked_executor: str | None = None
    last_agent_output: str | None = None
    last_agent_name: str | None = None
    last_route_source = team_name

    def flush_buffer(*, next_executor: str | None = None) -> None:
        nonlocal last_visible_input, active_executor, buffered_outputs, last_agent_output, last_agent_name, last_route_source
        if active_executor is None or not buffered_outputs:
            active_executor = next_executor
            buffered_outputs = []
            return

        aggregated_output = _summarize_trace_text("".join(buffered_outputs), max_length=600)
        result_target = team_name if pattern in {"group-chat", "concurrent", "magentic"} else (next_executor or team_name)
        _trace_result(
            result_target,
            active_executor,
            aggregated_output,
            team_name=team_name,
            agent_trace_colors=agent_trace_colors,
        )
        last_visible_input = aggregated_output
        last_agent_name = active_executor
        last_agent_output = aggregated_output
        last_route_source = active_executor if pattern in {"sequential", "handoff"} else team_name
        active_executor = next_executor
        buffered_outputs = []

    _trace_team_start(team_name, pattern, prompt)

    stream = workflow.run(prompt, stream=True)
    async for event in stream:
        if event.type == "group_chat":
            participant_name = _display_name(getattr(event.data, "participant_name", None), author_name_map)
            event_type = type(event.data).__name__
            if "RequestSent" in event_type:
                expected_invoked_executor = participant_name
                flush_buffer(next_executor=participant_name)
                _trace_route(
                    team_name,
                    participant_name,
                    _summarize_trace_text(last_visible_input),
                    team_name=team_name,
                    agent_trace_colors=agent_trace_colors,
                )
                last_route_source = team_name
            continue

        if event.type == "handoff_sent":
            source = _display_name(getattr(event.data, "source", None), author_name_map)
            target = _display_name(getattr(event.data, "target", None), author_name_map)
            expected_invoked_executor = target
            flush_buffer(next_executor=target)
            _trace_route(
                source,
                target,
                _summarize_trace_text(last_visible_input),
                team_name=team_name,
                agent_trace_colors=agent_trace_colors,
            )
            last_route_source = source
            continue

        if event.type == "executor_invoked" and event.executor_id:
            invoked_executor = _display_name(event.executor_id, author_name_map)
            if pattern in {"group-chat", "handoff"} and expected_invoked_executor is not None:
                if invoked_executor != expected_invoked_executor:
                    continue
                expected_invoked_executor = None

            flush_buffer(next_executor=invoked_executor)
            if pattern == "sequential":
                _trace_route(
                    last_route_source,
                    invoked_executor,
                    _summarize_trace_text(last_visible_input),
                    team_name=team_name,
                    agent_trace_colors=agent_trace_colors,
                )
                last_route_source = invoked_executor
            elif pattern in {"concurrent", "magentic"}:
                _trace_route(
                    team_name,
                    invoked_executor,
                    _summarize_trace_text(last_visible_input),
                    team_name=team_name,
                    agent_trace_colors=agent_trace_colors,
                )
                last_route_source = team_name
            continue

        if event.type == "output":
            final_payload = event.data
            output_summary = _summarize_payload(event.data, author_name_map=author_name_map)
            if output_summary:
                producer = _display_name(event.executor_id, author_name_map)
                if producer == team_name:
                    flush_buffer(next_executor=producer)
                    last_visible_input = output_summary
                else:
                    if active_executor is None:
                        active_executor = producer
                    if active_executor != producer:
                        flush_buffer(next_executor=producer)
                    raw_chunk = _extract_trace_chunk(event.data)
                    if raw_chunk:
                        buffered_outputs.append(raw_chunk)

    flush_buffer()
    rendered_output = _format_final_team_output(final_payload, author_name_map=author_name_map)
    if last_agent_output and last_agent_name:
        _trace_result(
            team_name,
            last_agent_name,
            _summarize_trace_text(last_agent_output),
            team_name=team_name,
            agent_trace_colors=agent_trace_colors,
            field_name="final-output",
        )
    else:
        _trace('%s | final-output:%s', _trace_team_label(team_name), _trace_block(_summarize_trace_text(rendered_output)))
    return rendered_output
