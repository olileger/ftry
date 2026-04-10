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
    _collect_visible_messages,
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
    _resolve_message_author,
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
TEAM_CONTEXT_PATTERNS = frozenset({"sequential", "concurrent", "handoff"})
TEAM_SHARED_RESULT_PATTERNS = frozenset({"group-chat", "concurrent", "magentic"})
TEAM_REQUEST_DRIVEN_PATTERNS = frozenset({"group-chat", "handoff"})
TEAM_DIRECT_ROUTE_PATTERNS = frozenset({"concurrent", "magentic"})


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


def _build_handoff_workflow(team: TeamConfig, *, participants: Sequence[Any], handoff_builder_type: Any) -> Any:
    handoff_builder = handoff_builder_type(name=team.name, participants=participants, description=team.description)
    handoff_builder = handoff_builder.with_start_agent(_select_handoff_start_agent(participants))
    autonomous_kwargs: dict[str, Any] = {"agents": participants}
    max_turns = team.termination.max_turns
    if max_turns is not None:
        autonomous_kwargs["turn_limits"] = {getattr(agent, "name"): max_turns for agent in participants}
        handoff_builder = handoff_builder.with_termination_condition(
            lambda conversation, limit=max_turns: _count_assistant_messages(conversation) >= limit
        )
    return handoff_builder.with_autonomous_mode(**autonomous_kwargs).build()


def _build_team_workflow(team: TeamConfig) -> tuple[str, Any, Mapping[str, str]]:
    pattern = _infer_team_pattern(team)
    rendered_instructions = _render_team_instructions(team)
    inject_team_context = pattern in TEAM_CONTEXT_PATTERNS or team.model is None
    participants, author_name_map = _build_team_participants(
        team,
        extra_instructions=rendered_instructions if inject_team_context else None,
    )
    SequentialBuilder, ConcurrentBuilder, HandoffBuilder, GroupChatBuilder, MagenticBuilder = _load_orchestration_builders()
    team_internal_name = _sanitize_agent_name(team.name, fallback_prefix="team")
    author_name_map[team_internal_name] = team.name

    if pattern in {"sequential", "concurrent"}:
        builder = {"sequential": SequentialBuilder, "concurrent": ConcurrentBuilder}[pattern]
        return pattern, builder(participants=participants, intermediate_outputs=True).build(), author_name_map

    if pattern == "handoff":
        return pattern, _build_handoff_workflow(team, participants=participants, handoff_builder_type=HandoffBuilder), author_name_map

    controller_agent = _create_team_controller_agent(team, instructions=rendered_instructions)
    if pattern == "group-chat":
        return (
            pattern,
            GroupChatBuilder(
                participants=participants,
                orchestrator_agent=controller_agent,
                orchestrator_name=team.name,
                max_rounds=team.termination.max_turns,
                intermediate_outputs=True,
            ).build(),
            author_name_map,
        )

    return (
        pattern,
        MagenticBuilder(
            participants=participants,
            manager_agent=controller_agent,
            max_round_count=team.termination.max_turns,
            intermediate_outputs=True,
        ).build(),
        author_name_map,
    )


@dataclass
class _TeamTraceState:
    pattern: str
    team_name: str
    agent_trace_colors: Mapping[str, str]
    last_visible_input: str
    final_payload: Any = None
    active_executor: str | None = None
    buffered_outputs: list[str] = field(default_factory=list)
    expected_invoked_executor: str | None = None
    last_agent_output: str | None = None
    last_agent_full_output: str | None = None
    last_agent_name: str | None = None
    last_route_source: str = ""

    def __post_init__(self) -> None:
        self.last_route_source = self.team_name

    def flush_buffer(self, *, next_executor: str | None = None) -> None:
        if self.active_executor is None or not self.buffered_outputs:
            self.active_executor = next_executor
            self.buffered_outputs.clear()
            return

        full_output = "".join(self.buffered_outputs)
        aggregated_output = _summarize_trace_text(full_output, max_length=600)
        result_target = self.team_name if self.pattern in TEAM_SHARED_RESULT_PATTERNS else (next_executor or self.team_name)
        _trace_result(
            result_target,
            self.active_executor,
            aggregated_output,
            team_name=self.team_name,
            agent_trace_colors=self.agent_trace_colors,
        )
        self.last_visible_input = aggregated_output
        self.last_agent_name = self.active_executor
        self.last_agent_output = aggregated_output
        self.last_agent_full_output = full_output
        self.last_route_source = self.active_executor if self.pattern in {"sequential", "handoff"} else self.team_name
        self.active_executor = next_executor
        self.buffered_outputs.clear()

    def trace_route(self, source_name: str, target_name: str) -> None:
        _trace_route(
            source_name,
            target_name,
            _summarize_trace_text(self.last_visible_input),
            team_name=self.team_name,
            agent_trace_colors=self.agent_trace_colors,
        )

    def trace_final_output(self, final_payload: Any, rendered_output: str, author_name_map: Mapping[str, str]) -> None:
        final_messages = _collect_visible_messages(final_payload)
        if final_messages:
            final_author = _resolve_message_author(final_messages[-1], author_name_map=author_name_map)
            if final_author == self.team_name and self.last_agent_full_output and self.last_agent_name:
                _trace_result(
                    self.team_name,
                    self.last_agent_name,
                    self.last_agent_full_output,
                    team_name=self.team_name,
                    agent_trace_colors=self.agent_trace_colors,
                    field_name="final-output",
                )
                return

            _trace_result(
                self.team_name,
                final_author or self.team_name,
                rendered_output,
                team_name=self.team_name,
                agent_trace_colors=self.agent_trace_colors,
                field_name="final-output",
            )
            return

        if self.last_agent_full_output and self.last_agent_name:
            _trace_result(
                self.team_name,
                self.last_agent_name,
                self.last_agent_full_output,
                team_name=self.team_name,
                agent_trace_colors=self.agent_trace_colors,
                field_name="final-output",
            )
            return

        _trace('%s | final-output:%s', _trace_team_label(self.team_name), _trace_block(rendered_output))


def _handle_group_chat_event(event: Any, state: _TeamTraceState, author_name_map: Mapping[str, str]) -> None:
    participant_name = _display_name(getattr(event.data, "participant_name", None), author_name_map)
    if "RequestSent" not in type(event.data).__name__:
        return

    state.expected_invoked_executor = participant_name
    state.flush_buffer(next_executor=participant_name)
    state.trace_route(state.team_name, participant_name)
    state.last_route_source = state.team_name


def _handle_handoff_event(event: Any, state: _TeamTraceState, author_name_map: Mapping[str, str]) -> None:
    source = _display_name(getattr(event.data, "source", None), author_name_map)
    target = _display_name(getattr(event.data, "target", None), author_name_map)
    state.expected_invoked_executor = target
    state.flush_buffer(next_executor=target)
    state.trace_route(source, target)
    state.last_route_source = source


def _handle_executor_invoked_event(event: Any, state: _TeamTraceState, author_name_map: Mapping[str, str]) -> None:
    if not event.executor_id:
        return

    invoked_executor = _display_name(event.executor_id, author_name_map)
    if state.pattern in TEAM_REQUEST_DRIVEN_PATTERNS and state.expected_invoked_executor is not None:
        if invoked_executor != state.expected_invoked_executor:
            return
        state.expected_invoked_executor = None

    state.flush_buffer(next_executor=invoked_executor)
    if state.pattern == "sequential":
        state.trace_route(state.last_route_source, invoked_executor)
        state.last_route_source = invoked_executor
    elif state.pattern in TEAM_DIRECT_ROUTE_PATTERNS:
        state.trace_route(state.team_name, invoked_executor)
        state.last_route_source = state.team_name


def _handle_output_event(event: Any, state: _TeamTraceState, author_name_map: Mapping[str, str]) -> None:
    state.final_payload = event.data
    output_summary = _summarize_payload(event.data, author_name_map=author_name_map)
    if not output_summary:
        return

    producer = _display_name(event.executor_id, author_name_map)
    if producer == state.team_name:
        state.flush_buffer(next_executor=producer)
        state.last_visible_input = output_summary
        return

    if state.active_executor is None:
        state.active_executor = producer
    elif state.active_executor != producer:
        state.flush_buffer(next_executor=producer)

    raw_chunk = _extract_trace_chunk(event.data)
    if raw_chunk:
        state.buffered_outputs.append(raw_chunk)


async def _run_team_prompt(team: TeamConfig, prompt: str) -> str:
    pattern, workflow, author_name_map = _build_team_workflow(team)
    state = _TeamTraceState(
        pattern=pattern,
        team_name=team.name,
        agent_trace_colors=_build_agent_trace_colors([agent.name for agent in team.agents]),
        last_visible_input=prompt,
    )

    _trace_team_start(state.team_name, state.pattern, prompt)

    stream = workflow.run(prompt, stream=True)
    async for event in stream:
        if event.type == "group_chat":
            _handle_group_chat_event(event, state, author_name_map)
            continue

        if event.type == "handoff_sent":
            _handle_handoff_event(event, state, author_name_map)
            continue

        if event.type == "executor_invoked":
            _handle_executor_invoked_event(event, state, author_name_map)
            continue

        if event.type == "output":
            _handle_output_event(event, state, author_name_map)

    state.flush_buffer()
    rendered_output = _format_final_team_output(state.final_payload, author_name_map=author_name_map)
    state.trace_final_output(state.final_payload, rendered_output, author_name_map)
    return rendered_output
