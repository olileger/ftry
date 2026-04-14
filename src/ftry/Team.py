from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence

from .Agent import Agent, AgentConfig, AgentModelConfig
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
    _resolve_config_path,
    _resolve_message_author,
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

TEAM_PATTERN_VALUES = ("sequential", "concurrent", "handoff", "group-chat", "magentic")
TEAM_TYPE_INFERENCE_PROMPT_FILE = "team-type-inference.txt"
TEAM_TYPE_INFERENCE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "team_workflow_inference",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "workflow_type": {
                    "type": "string",
                    "enum": list(TEAM_PATTERN_VALUES),
                },
                "reason": {
                    "type": "string",
                    "minLength": 1,
                },
            },
            "required": ["workflow_type", "reason"],
            "additionalProperties": False,
        },
    },
}
HANDOFF_START_HINTS = ("triage", "router", "routing", "route", "dispatcher", "coordinator")
TEAM_CONTEXT_PATTERNS = frozenset({"sequential", "concurrent", "handoff"})
TEAM_SHARED_RESULT_PATTERNS = frozenset({"group-chat", "concurrent", "magentic"})
TEAM_REQUEST_DRIVEN_PATTERNS = frozenset({"group-chat", "handoff"})
TEAM_DIRECT_ROUTE_PATTERNS = frozenset({"concurrent", "magentic"})


@dataclass(frozen=True)
class TeamTerminationConfig:
    max_turns: int | None = None


@dataclass(frozen=True)
class _TeamPatternInference:
    pattern: str
    reason: str


@dataclass(frozen=True)
class TeamConfig:
    name: str
    instructions: str
    agents: tuple[AgentConfig, ...]
    model: AgentModelConfig | None = None
    description: str | None = None
    termination: TeamTerminationConfig = field(default_factory=TeamTerminationConfig)


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
            team_pattern=self.pattern,
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
            team_pattern=self.pattern,
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
                    team_pattern=self.pattern,
                    field_name="final-output",
                )
                return

            _trace_result(
                self.team_name,
                final_author or self.team_name,
                rendered_output,
                team_name=self.team_name,
                agent_trace_colors=self.agent_trace_colors,
                team_pattern=self.pattern,
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
                team_pattern=self.pattern,
                field_name="final-output",
            )
            return

        _trace("%s | final-output:%s", _trace_team_label(self.team_name, pattern=self.pattern), _trace_block(rendered_output))


class Team:
    def __init__(self, config: TeamConfig):
        self._config = config

    @classmethod
    def from_file(cls, team_file: str | Path) -> Team:
        team_path = _resolve_config_path(team_file)
        if not team_path.is_file():
            raise FtryCliError(f"Team file not found: {team_path}")

        _load_dotenv_for_config(team_path)
        config = _load_yaml_mapping(team_path, config_kind="team")
        return cls(cls._parse_config(config, team_dir=team_path.parent))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], *, team_dir: Path | None = None) -> Team:
        return cls(cls._parse_config(config, team_dir=team_dir))

    @property
    def config(self) -> TeamConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def description(self) -> str | None:
        return self._config.description

    @property
    def instructions(self) -> str:
        return self._config.instructions

    @property
    def agents(self) -> tuple[AgentConfig, ...]:
        return self._config.agents

    @property
    def model(self) -> AgentModelConfig | None:
        return self._config.model

    @property
    def termination(self) -> TeamTerminationConfig:
        return self._config.termination

    async def run(self, prompt: str) -> str:
        pattern, workflow, author_name_map = await self._build_workflow()
        state = _TeamTraceState(
            pattern=pattern,
            team_name=self.name,
            agent_trace_colors=_build_agent_trace_colors([agent.name for agent in self.agents]),
            last_visible_input=prompt,
        )

        _trace_team_start(state.team_name, state.pattern, prompt)

        stream = workflow.run(prompt, stream=True)
        async for event in stream:
            if event.type == "group_chat":
                self._handle_group_chat_event(event, state, author_name_map)
                continue

            if event.type == "handoff_sent":
                self._handle_handoff_event(event, state, author_name_map)
                continue

            if event.type == "executor_invoked":
                self._handle_executor_invoked_event(event, state, author_name_map)
                continue

            if event.type == "output":
                self._handle_output_event(event, state, author_name_map)

        state.flush_buffer()
        rendered_output = _format_final_team_output(state.final_payload, author_name_map=author_name_map)
        state.trace_final_output(state.final_payload, rendered_output, author_name_map)
        return rendered_output

    @classmethod
    def _parse_config(cls, config: Mapping[str, Any], *, team_dir: Path | None) -> TeamConfig:
        raw_agents = _require_sequence(config.get("agents"), "agents", "team")
        agents = tuple(cls._load_agent_config(raw_agent, team_dir=team_dir) for raw_agent in raw_agents)
        if not agents:
            raise FtryCliError("Invalid or missing `agents` list in team YAML.")

        if "pattern" in config or "orchestration" in config:
            raise FtryCliError(
                "Explicit team workflow selection is no longer supported. Remove `pattern`/`orchestration`; "
                "the workflow type is inferred automatically."
            )

        return TeamConfig(
            name=_require_non_empty_string(config.get("name"), "name", "team"),
            description=_require_optional_string(config.get("description"), "description", "team"),
            instructions=_require_non_empty_string(config.get("prompt"), "prompt", "team"),
            agents=agents,
            model=AgentModelConfig.from_mapping(config.get("model"), config_kind="team", required=False),
            termination=cls._parse_termination(config.get("termination")),
        )

    @classmethod
    def _load_agent_config(cls, raw_agent: Any, *, team_dir: Path | None) -> AgentConfig:
        if isinstance(raw_agent, str):
            return Agent.from_file(raw_agent, base_dir=team_dir).config

        agent_config = _require_mapping(raw_agent, "agents[]", "team")
        if "file" in agent_config:
            if len(agent_config) != 1:
                raise FtryCliError("Invalid `agents[]` entry in team YAML: `file` references cannot be mixed with inline fields.")
            return Agent.from_file(
                _require_non_empty_string(agent_config.get("file"), "agents[].file", "team"),
                base_dir=team_dir,
            ).config

        return Agent.from_mapping(agent_config, config_kind="team agent").config

    @staticmethod
    def _parse_termination(raw_termination: Any) -> TeamTerminationConfig:
        if raw_termination is None:
            return TeamTerminationConfig()

        termination_config = _require_mapping(raw_termination, "termination", "team")
        max_turns = termination_config.get("max-turns")
        if max_turns is None:
            return TeamTerminationConfig()

        return TeamTerminationConfig(max_turns=_require_positive_int(max_turns, "termination.max-turns", "team"))

    @staticmethod
    def _render_role_summary(agent: AgentConfig) -> str:
        source = agent.description or agent.instructions
        return " ".join(source.split())

    def _render_instructions(self) -> str:
        participants = ", ".join(agent.name for agent in self.agents)
        roles = "\n".join(f"- {agent.name}: {self._render_role_summary(agent)}" for agent in self.agents)
        return self.instructions.replace("{participants}", participants).replace("{roles}", roles)

    def _render_pattern_inference_input(self, *, rendered_instructions: str) -> str:
        return "\n".join(
            [
                f"Team name: {self.name}",
                f"Team description: {self.description or '(none)'}",
                "",
                "Team prompt:",
                rendered_instructions,
            ]
        )

    @staticmethod
    def _load_team_type_inference_prompt_template() -> str:
        return files("ftry").joinpath(TEAM_TYPE_INFERENCE_PROMPT_FILE).read_text(encoding="utf-8")

    @staticmethod
    def _contains_any(text: str, hints: Sequence[str]) -> bool:
        return any(hint in text for hint in hints)

    def _resolve_pattern_inference_model(self) -> AgentModelConfig:
        return self.model or self.agents[0].model

    @staticmethod
    def _format_pattern_inference_output(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)

    @staticmethod
    def _preview_pattern_inference_prompt(prompt: str, *, max_lines: int = 6) -> str:
        lines = prompt.splitlines()
        if len(lines) <= max_lines:
            return prompt
        return "\n".join([*lines[:max_lines], "..."])

    @classmethod
    def _parse_pattern_inference(cls, value: Any) -> _TeamPatternInference:
        if not isinstance(value, Mapping):
            raise FtryCliError("Team workflow inference did not return a JSON object.")

        workflow_type = value.get("workflow_type")
        if not isinstance(workflow_type, str) or not workflow_type.strip():
            raise FtryCliError("Team workflow inference did not return a valid `workflow_type`.")
        normalized_workflow_type = workflow_type.strip()
        if normalized_workflow_type not in TEAM_PATTERN_VALUES:
            raise FtryCliError(
                "Team workflow inference returned an unsupported `workflow_type` "
                f"`{workflow_type}`. Expected one of: {', '.join(TEAM_PATTERN_VALUES)}."
            )

        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise FtryCliError("Team workflow inference did not return a valid `reason`.")

        return _TeamPatternInference(
            pattern=normalized_workflow_type,
            reason=reason.strip(),
        )

    async def _infer_pattern(self, *, rendered_instructions: str) -> str:
        model = self._resolve_pattern_inference_model()
        if model.provider.lower() != "openai":
            raise FtryCliError(
                f"Unsupported provider `{model.provider}`. Only `openai` is supported for now."
            )

        try:
            from agent_framework.openai import OpenAIChatCompletionClient
        except ImportError as exc:  # pragma: no cover - covered by CLI error path
            raise FtryCliError(
                "Microsoft Agent Framework OpenAI support is required for `ftry pop -t`. "
                "Reinstall the project with `python -m pip install -e .`."
            ) from exc

        inference_agent = OpenAIChatCompletionClient(
            model=model.name,
            api_key=model.api_key,
        ).as_agent(
            name="team-workflow-selector",
            description="Selects the most appropriate Microsoft Agent Framework workflow type.",
            instructions=self._load_team_type_inference_prompt_template(),
        )
        inference_prompt = self._render_pattern_inference_input(rendered_instructions=rendered_instructions)
        _trace(
            "%s | team-type-inference-prompt:%s",
            _trace_team_label(self.name),
            _trace_block(self._preview_pattern_inference_prompt(inference_prompt)),
        )
        inference_response = await inference_agent.run(
            inference_prompt,
            options={"response_format": TEAM_TYPE_INFERENCE_RESPONSE_FORMAT},
        )
        raw_inference_output = getattr(inference_response, "value", None)
        _trace(
            "%s | team-type-inference-output:%s",
            _trace_team_label(self.name),
            _trace_block(self._format_pattern_inference_output(raw_inference_output)),
        )
        inference = self._parse_pattern_inference(raw_inference_output)
        return inference.pattern

    @staticmethod
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

    def _create_controller_agent(self, *, instructions: str) -> Any | None:
        if self.model is None:
            return None

        return Agent(
            AgentConfig(
                name=self.name,
                description=self.description,
                instructions=instructions,
                model=self.model,
            )
        ).create_participant(
            name_override=_sanitize_agent_name(self.name, fallback_prefix="team"),
        )

    @staticmethod
    def _select_handoff_start_agent(agents: Sequence[Any]) -> Any:
        for agent in agents:
            name = getattr(agent, "name", "")
            description = getattr(agent, "description", "")
            instructions = getattr(agent, "instructions", "")
            analysis_text = f"{name} {description} {instructions}".lower()
            if Team._contains_any(analysis_text, HANDOFF_START_HINTS):
                return agent
        return agents[0]

    @staticmethod
    def _count_assistant_messages(conversation: list[Any]) -> int:
        return sum(1 for message in conversation if getattr(message, "role", None) == "assistant")

    def _build_participants(
        self,
        *,
        extra_instructions: str | None,
        require_per_service_call_history_persistence: bool = False,
    ) -> tuple[list[Any], dict[str, str]]:
        participants: list[Any] = []
        author_name_map: dict[str, str] = {}
        used_names: set[str] = set()

        for index, agent in enumerate(self.agents, start=1):
            candidate_name = _sanitize_agent_name(agent.name, fallback_prefix=f"agent-{index}")
            unique_name = candidate_name
            suffix = 2
            while unique_name in used_names:
                unique_name = f"{candidate_name}-{suffix}"
                suffix += 1
            used_names.add(unique_name)
            author_name_map[unique_name] = agent.name
            participants.append(
                Agent(agent).create_participant(
                    extra_instructions=extra_instructions,
                    name_override=unique_name,
                    require_per_service_call_history_persistence=require_per_service_call_history_persistence,
                )
            )

        return participants, author_name_map

    def _build_handoff_workflow(self, *, participants: Sequence[Any], handoff_builder_type: Any) -> Any:
        handoff_builder = handoff_builder_type(name=self.name, participants=participants, description=self.description)
        handoff_builder = handoff_builder.with_start_agent(self._select_handoff_start_agent(participants))
        autonomous_kwargs: dict[str, Any] = {"agents": participants}
        max_turns = self.termination.max_turns
        if max_turns is not None:
            autonomous_kwargs["turn_limits"] = {getattr(agent, "name"): max_turns for agent in participants}
            handoff_builder = handoff_builder.with_termination_condition(
                lambda conversation, limit=max_turns: self._count_assistant_messages(conversation) >= limit
            )
        return handoff_builder.with_autonomous_mode(**autonomous_kwargs).build()

    async def _build_workflow(self) -> tuple[str, Any, Mapping[str, str]]:
        rendered_instructions = self._render_instructions()
        pattern = await self._infer_pattern(rendered_instructions=rendered_instructions)
        inject_team_context = pattern in TEAM_CONTEXT_PATTERNS or self.model is None
        participants, author_name_map = self._build_participants(
            extra_instructions=rendered_instructions if inject_team_context else None,
            require_per_service_call_history_persistence=pattern == "handoff",
        )
        SequentialBuilder, ConcurrentBuilder, HandoffBuilder, GroupChatBuilder, MagenticBuilder = (
            self._load_orchestration_builders()
        )
        team_internal_name = _sanitize_agent_name(self.name, fallback_prefix="team")
        author_name_map[team_internal_name] = self.name

        if pattern in {"sequential", "concurrent"}:
            builder = {"sequential": SequentialBuilder, "concurrent": ConcurrentBuilder}[pattern]
            return pattern, builder(participants=participants, intermediate_outputs=True).build(), author_name_map

        if pattern == "handoff":
            return pattern, self._build_handoff_workflow(participants=participants, handoff_builder_type=HandoffBuilder), author_name_map

        controller_agent = self._create_controller_agent(instructions=rendered_instructions)
        if pattern == "group-chat":
            return (
                pattern,
                GroupChatBuilder(
                    participants=participants,
                    orchestrator_agent=controller_agent,
                    orchestrator_name=self.name,
                    max_rounds=self.termination.max_turns,
                    intermediate_outputs=True,
                ).build(),
                author_name_map,
            )

        return (
            pattern,
            MagenticBuilder(
                participants=participants,
                manager_agent=controller_agent,
                max_round_count=self.termination.max_turns,
                intermediate_outputs=True,
            ).build(),
            author_name_map,
        )

    @staticmethod
    def _handle_group_chat_event(event: Any, state: _TeamTraceState, author_name_map: Mapping[str, str]) -> None:
        participant_name = _display_name(getattr(event.data, "participant_name", None), author_name_map)
        if "RequestSent" not in type(event.data).__name__:
            return

        state.expected_invoked_executor = participant_name
        state.flush_buffer(next_executor=participant_name)
        state.trace_route(state.team_name, participant_name)
        state.last_route_source = state.team_name

    @staticmethod
    def _handle_handoff_event(event: Any, state: _TeamTraceState, author_name_map: Mapping[str, str]) -> None:
        source = _display_name(getattr(event.data, "source", None), author_name_map)
        target = _display_name(getattr(event.data, "target", None), author_name_map)
        state.expected_invoked_executor = target
        state.flush_buffer(next_executor=target)
        state.trace_route(source, target)
        state.last_route_source = source

    @staticmethod
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

    @staticmethod
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
