from __future__ import annotations

from contextlib import AsyncExitStack
import json
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .Agent import AgentConfig, AgentModelConfig
from .Mcp import Mcp
from .TeamAgent import HandoffHilSignalState, TeamAgent
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
TEAM_TYPE_INFERENCE_PROMPT_TEMPLATE = """You are selecting the most appropriate Microsoft Agent Framework workflow type for a team configuration.

The workflow summaries below come from the official Microsoft Agent Framework documentation:

- sequential: agents are organized in a pipeline; each agent processes the task in turn and passes its output to the next. Use this when each step builds on the previous one.
- concurrent: multiple agents work on the same task in parallel and independently, and their results are aggregated. Use this when specialists can work independently on the same input.
- handoff: agents transfer control directly to one another based on the context or user request, without a central orchestrator. Use this when task ownership should move dynamically between specialists.
- group-chat: multiple agents collaborate through an orchestrated conversation with shared context, iterative refinement, and multi-perspective discussion. Use this when agents should review, discuss, and improve together over multiple turns.
- magentic: a manager dynamically plans, selects agents, tracks progress, and replans for complex open-ended tasks. Use this only when the solution path is not known in advance and dynamic planning/replanning is central.

Decision rules:
- Pick exactly one workflow type from: sequential, concurrent, handoff, group-chat, magentic.
- Prefer the simplest workflow that fully matches the team design.
- Do not choose concurrent if later agents depend on earlier agents' outputs.
- Do not choose sequential if agents are meant to work independently on the same input.
- Do not choose handoff unless control should be transferred directly between specialists.
- Do not choose group-chat unless iterative multi-turn collaboration is essential.
- Do not choose magentic unless complex planning or replanning is essential.
- Ignore the human language used in the team definition. The team can be written in any language.

Return JSON only with this exact shape:
{"workflow_type":"sequential|concurrent|handoff|group-chat|magentic","reason":"short justification"}

Rules for the JSON:

- `reason` must be concise and mention the decisive characteristics that led to the workflow choice."""
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
TEAM_CONTEXT_PATTERNS = frozenset({"sequential", "concurrent", "handoff"})
TEAM_SHARED_RESULT_PATTERNS = frozenset({"group-chat", "concurrent", "magentic"})
TEAM_REQUEST_DRIVEN_PATTERNS = frozenset({"group-chat", "handoff"})
TEAM_DIRECT_ROUTE_PATTERNS = frozenset({"concurrent", "magentic"})
UserInputProvider = Callable[[str], str]


@dataclass(frozen=True)
class TeamTerminationConfig:
    max_turns: int | None = None


@dataclass(frozen=True)
class _TeamPatternInference:
    pattern: str
    reason: str


@dataclass(frozen=True)
class _PlannedTeamParticipant:
    internal_name: str
    display_name: str
    config: AgentConfig
    role_summary: str


@dataclass(frozen=True)
class _PendingTeamRequest:
    request_id: str
    prompt: str
    build_response: Callable[[str], Any]
    skip_trace_when_blank: bool = False


@dataclass(frozen=True)
class TeamConfig:
    name: str
    instructions: str
    agents: tuple[AgentConfig, ...]
    model: AgentModelConfig | None = None
    description: str | None = None
    mcp_servers: tuple[str, ...] = ()
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
    def mcp_servers(self) -> tuple[str, ...]:
        return self._config.mcp_servers

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

    async def run(self, prompt: str, *, user_input_provider: UserInputProvider | None = None) -> str:
        pattern, workflow, author_name_map, handoff_hil_signal_state, workflow_resources = await self._build_workflow_with_resources(prompt)
        initial_prompt = prompt
        state = _TeamTraceState(
            pattern=pattern,
            team_name=self.name,
            agent_trace_colors=_build_agent_trace_colors([agent.name for agent in self.agents]),
            last_visible_input=prompt,
        )

        _trace_team_start(state.team_name, state.pattern, prompt)

        pending_responses: dict[str, Any] | None = None
        first_run = True
        next_prompt: str | None = prompt
        handoff_user_inputs: list[tuple[str, str]] = []
        handoff_agent_context: list[tuple[str, str]] = []

        try:
            while True:
                if first_run:
                    if handoff_hil_signal_state is not None:
                        handoff_hil_signal_state.clear()
                    stream = workflow.run(next_prompt, stream=True)
                    first_run = False
                    next_prompt = None
                elif next_prompt is not None:
                    if handoff_hil_signal_state is not None:
                        handoff_hil_signal_state.clear()
                    stream = workflow.run(next_prompt, stream=True)
                    next_prompt = None
                elif pending_responses is not None:
                    stream = workflow.run(stream=True, responses=pending_responses)
                else:
                    break

                pending_responses = await self._process_workflow_stream(
                    stream,
                    state=state,
                    author_name_map=author_name_map,
                    user_input_provider=user_input_provider,
                )

                if handoff_hil_signal_state is not None:
                    if handoff_hil_signal_state.action == "request_user_input":
                        request_prompt, user_input, requesting_agent = self._resolve_handoff_user_input(
                            signal_state=handoff_hil_signal_state,
                            state=state,
                            user_input_provider=user_input_provider,
                        )
                        if state.last_agent_name and state.last_agent_full_output:
                            handoff_agent_context.append((author_name_map.get(state.last_agent_name, state.last_agent_name), state.last_agent_full_output))
                        handoff_user_inputs.append((request_prompt, user_input))
                        next_prompt = self._build_handoff_resume_prompt(
                            initial_prompt=initial_prompt,
                            agent_context=handoff_agent_context,
                            user_inputs=handoff_user_inputs,
                        )
                        await workflow_resources.aclose()
                        pattern, workflow, author_name_map, handoff_hil_signal_state, workflow_resources = await self._build_workflow_with_resources(
                            next_prompt,
                            forced_pattern=pattern,
                            forced_handoff_start_agent=requesting_agent,
                        )
                        pending_responses = None
                        state.final_payload = None
                        continue

                    if handoff_hil_signal_state.action == "final_answer":
                        state.final_payload = self._build_handoff_signal_payload(handoff_hil_signal_state)

            state.flush_buffer()
            rendered_output = _format_final_team_output(state.final_payload, author_name_map=author_name_map)
            state.trace_final_output(state.final_payload, rendered_output, author_name_map)
            return rendered_output
        finally:
            await workflow_resources.aclose()

    @classmethod
    def _parse_config(cls, config: Mapping[str, Any], *, team_dir: Path | None) -> TeamConfig:
        shared_mcp_servers = Mcp.parse_server_names(config.get("mcp"), field_name="mcp", config_kind="team")
        raw_agents = _require_sequence(config.get("agents"), "agents", "team")
        agents = tuple(
            cls._load_agent_config(raw_agent, team_dir=team_dir, shared_mcp_servers=shared_mcp_servers)
            for raw_agent in raw_agents
        )
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
            mcp_servers=shared_mcp_servers,
            termination=cls._parse_termination(config.get("termination")),
        )

    @classmethod
    def _load_agent_config(
        cls,
        raw_agent: Any,
        *,
        team_dir: Path | None,
        shared_mcp_servers: Sequence[str] = (),
    ) -> AgentConfig:
        if isinstance(raw_agent, str):
            return cls._apply_mcp_defaults(
                TeamAgent.from_file(raw_agent, base_dir=team_dir).config,
                shared_mcp_servers=shared_mcp_servers,
                mcp_registry_dir=team_dir,
            )

        agent_config = _require_mapping(raw_agent, "agents[]", "team")
        if "file" in agent_config:
            unexpected_fields = set(agent_config).difference({"file", "mcp"})
            if unexpected_fields:
                raise FtryCliError(
                    "Invalid `agents[]` entry in team YAML: `file` references cannot be mixed with inline fields "
                    "other than optional `mcp`."
                )
            loaded_config = TeamAgent.from_file(
                _require_non_empty_string(agent_config.get("file"), "agents[].file", "team"),
                base_dir=team_dir,
            ).config
            return cls._apply_mcp_defaults(
                loaded_config,
                shared_mcp_servers=shared_mcp_servers,
                local_mcp_servers=Mcp.parse_server_names(
                    agent_config.get("mcp"),
                    field_name="agents[].mcp",
                    config_kind="team",
                ),
                mcp_registry_dir=team_dir,
            )

        return cls._apply_mcp_defaults(
            TeamAgent.from_mapping(agent_config, config_kind="team agent").config,
            shared_mcp_servers=shared_mcp_servers,
            mcp_registry_dir=team_dir,
        )

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

    def _render_pattern_inference_input(
        self,
        *,
        current_input: str,
        rendered_instructions: str,
        participant_plans: Sequence[_PlannedTeamParticipant],
    ) -> str:
        agent_lines = [
            f"- id: {plan.internal_name} | name: {plan.display_name} | role: {plan.role_summary}"
            for plan in participant_plans
        ]
        return "\n".join(
            [
                f"Team name: {self.name}",
                f"Team description: {self.description or '(none)'}",
                f"Current user request: {current_input}",
                "",
                "Agents:",
                *agent_lines,
                "",
                "Team prompt:",
                rendered_instructions,
            ]
        )

    @staticmethod
    def _load_team_type_inference_prompt_template() -> str:
        return TEAM_TYPE_INFERENCE_PROMPT_TEMPLATE

    @staticmethod
    def _extract_request_prompt(source: Any) -> str:
        if source is None:
            raise FtryCliError("Team request_info did not include a visible prompt for the user.")

        text = getattr(source, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        messages = source if isinstance(source, Sequence) and not isinstance(source, (str, bytes)) else getattr(source, "messages", None)
        if isinstance(messages, Sequence):
            visible_messages = [
                message_text.strip()
                for message in messages
                if isinstance((message_text := getattr(message, "text", None)), str) and message_text.strip()
            ]
            if visible_messages:
                return visible_messages[-1]

        raise FtryCliError("Team request_info did not include a visible prompt for the user.")

    @staticmethod
    def _trace_request_info(prompt: str, state: _TeamTraceState) -> None:
        _trace(
            "%s | request-info:%s",
            _trace_team_label(state.team_name, pattern=state.pattern),
            _trace_block(prompt),
        )

    @staticmethod
    def _build_handoff_signal_payload(signal_state: HandoffHilSignalState) -> list[Any]:
        if not signal_state.message:
            return []
        return [
            types.SimpleNamespace(
                role="assistant",
                text=signal_state.message,
                author_name=signal_state.actor_name,
            )
        ]

    @staticmethod
    def _build_handoff_resume_prompt(
        *,
        initial_prompt: str,
        agent_context: Sequence[tuple[str, str]],
        user_inputs: Sequence[tuple[str, str]],
    ) -> str:
        if not agent_context and not user_inputs:
            return initial_prompt

        rendered_agent_context = "\n".join(
            [
                f"{index}. {agent_name}: {message}"
                for index, (agent_name, message) in enumerate(agent_context, start=1)
            ]
        )
        rendered_user_inputs = "\n".join(
            [
                f"{index}. Question: {question}\n   User answer: {answer}"
                for index, (question, answer) in enumerate(user_inputs, start=1)
            ]
        )
        sections = [f"Original user request:\n{initial_prompt}"]
        if rendered_agent_context:
            sections.append(f"Context already produced by the team before the clarification pause:\n{rendered_agent_context}")
        if rendered_user_inputs:
            sections.append(f"Additional user information collected during the handoff workflow:\n{rendered_user_inputs}")
        sections.append(
            "Resume the workflow from the latest relevant specialist context. "
            "Do not ask again for information that the user already provided unless it is still missing or contradictory."
        )
        return "\n\n".join(sections)

    def _resolve_handoff_user_input(
        self,
        *,
        signal_state: HandoffHilSignalState,
        state: _TeamTraceState,
        user_input_provider: UserInputProvider | None,
    ) -> tuple[str, str, str | None]:
        prompt = (signal_state.prompt or "").strip()
        requesting_agent = signal_state.actor_name
        if not prompt:
            raise FtryCliError("Handoff Human in the Loop requested user input without a visible question.")
        if user_input_provider is None:
            raise FtryCliError(
                f"Team `{self.name}` is awaiting user input, but no interactive user input provider is configured."
            )

        state.flush_buffer()
        state.last_visible_input = prompt
        self._trace_request_info(prompt, state)
        user_input = user_input_provider(prompt)
        _trace_team_start(state.team_name, state.pattern, user_input)
        state.last_visible_input = user_input
        signal_state.clear()
        return prompt, user_input, requesting_agent

    @classmethod
    def _create_pending_request(cls, event: Any, *, state: _TeamTraceState) -> _PendingTeamRequest:
        request_id = getattr(event, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            raise FtryCliError("Team request_info event is missing a valid request identifier.")

        data = getattr(event, "data", None)
        state.flush_buffer()

        if bool(getattr(data, "user_input_request", False)):
            prompt = cls._extract_request_prompt(data)
            state.last_visible_input = prompt
            cls._trace_request_info(prompt, state)
            return _PendingTeamRequest(
                request_id=request_id,
                prompt=prompt,
                build_response=lambda user_input: type(data).from_text(user_input),
            )

        raise FtryCliError(
            "Team request_info emitted an unsupported payload. "
            "Only Human in the Loop request/response is supported; tool approval is not supported."
        )

    async def _process_workflow_stream(
        self,
        stream: Any,
        *,
        state: _TeamTraceState,
        author_name_map: Mapping[str, str],
        user_input_provider: UserInputProvider | None,
    ) -> dict[str, Any] | None:
        pending_requests: list[_PendingTeamRequest] = []

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

            if event.type == "request_info":
                pending_requests.append(self._create_pending_request(event, state=state))
                continue

            if event.type == "output":
                self._handle_output_event(event, state, author_name_map)

        if not pending_requests:
            return None

        if user_input_provider is None:
            raise FtryCliError(
                f"Team `{self.name}` is awaiting user input, but no interactive user input provider is configured."
            )

        responses: dict[str, Any] = {}
        for request in pending_requests:
            user_input = user_input_provider(request.prompt)
            if not (request.skip_trace_when_blank and not user_input.strip()):
                _trace_team_start(state.team_name, state.pattern, user_input)
                state.last_visible_input = user_input
            responses[request.request_id] = request.build_response(user_input)

        return responses

    def _resolve_pattern_inference_model(self) -> AgentModelConfig:
        return self.model or self.agents[0].model

    @staticmethod
    def _format_pattern_inference_output(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)

    @staticmethod
    def _preview_pattern_inference_prompt(prompt: str, *, max_lines: int = 9) -> str:
        lines = prompt.splitlines()
        if len(lines) <= max_lines:
            return prompt
        return "\n".join([*lines[:max_lines], "..."])

    @classmethod
    def _parse_pattern_inference(
        cls,
        value: Any,
        *,
        valid_agent_names: Sequence[str] = (),
    ) -> _TeamPatternInference:
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

    async def _infer_pattern(
        self,
        *,
        current_input: str,
        rendered_instructions: str,
        participant_plans: Sequence[_PlannedTeamParticipant],
    ) -> _TeamPatternInference:
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
        inference_prompt = self._render_pattern_inference_input(
            current_input=current_input,
            rendered_instructions=rendered_instructions,
            participant_plans=participant_plans,
        )
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
        return self._parse_pattern_inference(
            raw_inference_output,
            valid_agent_names=[plan.internal_name for plan in participant_plans],
        )

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

        return TeamAgent(
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
    def _select_handoff_start_agent(agents: Sequence[Any], preferred_agent_name: str | None = None) -> Any:
        if preferred_agent_name:
            for agent in agents:
                if getattr(agent, "name", None) == preferred_agent_name:
                    return agent
        return agents[0]

    @staticmethod
    def _count_assistant_messages(conversation: list[Any]) -> int:
        return sum(1 for message in conversation if getattr(message, "role", None) == "assistant")

    def _build_participant_plans(self) -> tuple[_PlannedTeamParticipant, ...]:
        participant_plans: list[_PlannedTeamParticipant] = []
        used_names: set[str] = set()

        for index, agent in enumerate(self.agents, start=1):
            candidate_name = _sanitize_agent_name(agent.name, fallback_prefix=f"agent-{index}")
            unique_name = candidate_name
            suffix = 2
            while unique_name in used_names:
                unique_name = f"{candidate_name}-{suffix}"
                suffix += 1
            used_names.add(unique_name)
            participant_plans.append(
                _PlannedTeamParticipant(
                    internal_name=unique_name,
                    display_name=agent.name,
                    config=agent,
                    role_summary=self._render_role_summary(agent),
                )
            )

        return tuple(participant_plans)

    def _build_participants(
        self,
        *,
        participant_plans: Sequence[_PlannedTeamParticipant],
        extra_instructions: str | None,
        use_managed_participants: bool,
        enforce_structured_output: bool = True,
        require_per_service_call_history_persistence: bool = False,
        handoff_hil_signal_state: HandoffHilSignalState | None = None,
    ) -> tuple[list[Any], dict[str, str]]:
        participants: list[Any] = []
        author_name_map: dict[str, str] = {}

        for plan in participant_plans:
            author_name_map[plan.internal_name] = plan.display_name
            team_agent = TeamAgent(plan.config)
            participant_factory = (
                team_agent.create_managed_participant if use_managed_participants else team_agent.create_participant
            )
            participant_kwargs: dict[str, Any] = {
                "extra_instructions": extra_instructions,
                "name_override": plan.internal_name,
                "require_per_service_call_history_persistence": require_per_service_call_history_persistence,
            }
            if use_managed_participants:
                participant_kwargs["enforce_structured_output"] = enforce_structured_output
                participant_kwargs["handoff_hil_signal_state"] = handoff_hil_signal_state
            participants.append(
                participant_factory(**participant_kwargs)
            )

        return participants, author_name_map

    async def _build_participants_with_mcp(
        self,
        *,
        participant_plans: Sequence[_PlannedTeamParticipant],
        extra_instructions: str | None,
        exit_stack: AsyncExitStack,
        use_managed_participants: bool,
        enforce_structured_output: bool = True,
        require_per_service_call_history_persistence: bool = False,
        handoff_hil_signal_state: HandoffHilSignalState | None = None,
    ) -> tuple[list[Any], dict[str, str]]:
        participants: list[Any] = []
        author_name_map: dict[str, str] = {}

        for plan in participant_plans:
            author_name_map[plan.internal_name] = plan.display_name
            team_agent = TeamAgent(plan.config)
            participant_factory = (
                team_agent.create_managed_participant if use_managed_participants else team_agent.create_participant
            )
            mcp_tools, mcp_context = await team_agent._prepare_mcp_runtime(exit_stack)
            participant_kwargs: dict[str, Any] = {
                "extra_instructions": team_agent._merge_extra_instructions(extra_instructions, mcp_context),
                "name_override": plan.internal_name,
                "require_per_service_call_history_persistence": require_per_service_call_history_persistence,
                "tools": mcp_tools,
            }
            if use_managed_participants:
                participant_kwargs["enforce_structured_output"] = enforce_structured_output
                participant_kwargs["handoff_hil_signal_state"] = handoff_hil_signal_state
            participants.append(participant_factory(**participant_kwargs))

        return participants, author_name_map

    def _build_handoff_workflow(
        self,
        *,
        participants: Sequence[Any],
        handoff_builder_type: Any,
        handoff_hil_signal_state: HandoffHilSignalState,
        preferred_start_agent_name: str | None = None,
    ) -> Any:
        handoff_builder = handoff_builder_type(name=self.name, participants=participants, description=self.description)
        handoff_builder = handoff_builder.with_start_agent(
            self._select_handoff_start_agent(participants, preferred_agent_name=preferred_start_agent_name)
        )
        handoff_builder = handoff_builder.with_autonomous_mode()
        max_turns = self.termination.max_turns
        handoff_builder = handoff_builder.with_termination_condition(
            lambda conversation, limit=max_turns, signal_state=handoff_hil_signal_state: (
                signal_state.action is not None
                or (limit is not None and self._count_assistant_messages(conversation) >= limit)
            )
        )
        workflow = handoff_builder.build()
        if hasattr(workflow, "__dict__"):
            setattr(workflow, "handoff_hil_signal_state", handoff_hil_signal_state)
        if hasattr(workflow, "kwargs") and isinstance(getattr(workflow, "kwargs"), dict):
            workflow.kwargs["handoff_hil_signal_state"] = handoff_hil_signal_state
        return workflow

    async def _build_workflow(
        self,
        current_input: str,
        *,
        forced_pattern: str | None = None,
        forced_handoff_start_agent: str | None = None,
    ) -> tuple[str, Any, Mapping[str, str], HandoffHilSignalState | None]:
        pattern, workflow, author_name_map, handoff_hil_signal_state, workflow_resources = await self._build_workflow_with_resources(
            current_input,
            forced_pattern=forced_pattern,
            forced_handoff_start_agent=forced_handoff_start_agent,
        )
        await workflow_resources.aclose()
        return pattern, workflow, author_name_map, handoff_hil_signal_state

    async def _build_workflow_with_resources(
        self,
        current_input: str,
        *,
        forced_pattern: str | None = None,
        forced_handoff_start_agent: str | None = None,
    ) -> tuple[str, Any, Mapping[str, str], HandoffHilSignalState | None, AsyncExitStack]:
        workflow_resources = AsyncExitStack()
        try:
            rendered_instructions = self._render_instructions()
            participant_plans = self._build_participant_plans()
            if forced_pattern is None:
                inference = await self._infer_pattern(
                    current_input=current_input,
                    rendered_instructions=rendered_instructions,
                    participant_plans=participant_plans,
                )
                pattern = inference.pattern
            else:
                pattern = forced_pattern
            inject_team_context = pattern in TEAM_CONTEXT_PATTERNS or self.model is None
            handoff_hil_signal_state = HandoffHilSignalState() if pattern == "handoff" else None
            participants, author_name_map = await self._build_participants_with_mcp(
                participant_plans=participant_plans,
                extra_instructions=rendered_instructions if inject_team_context else None,
                exit_stack=workflow_resources,
                use_managed_participants=pattern != "concurrent",
                enforce_structured_output=pattern != "handoff",
                require_per_service_call_history_persistence=pattern == "handoff",
                handoff_hil_signal_state=handoff_hil_signal_state,
            )
            SequentialBuilder, ConcurrentBuilder, HandoffBuilder, GroupChatBuilder, MagenticBuilder = (
                self._load_orchestration_builders()
            )
            team_internal_name = _sanitize_agent_name(self.name, fallback_prefix="team")
            author_name_map[team_internal_name] = self.name

            if pattern in {"sequential", "concurrent"}:
                builder = {"sequential": SequentialBuilder, "concurrent": ConcurrentBuilder}[pattern]
                workflow_builder = builder(participants=participants, intermediate_outputs=True)
                return pattern, workflow_builder.build(), author_name_map, None, workflow_resources

            if pattern == "handoff":
                return (
                    pattern,
                    self._build_handoff_workflow(
                        participants=participants,
                        handoff_builder_type=HandoffBuilder,
                        handoff_hil_signal_state=handoff_hil_signal_state or HandoffHilSignalState(),
                        preferred_start_agent_name=forced_handoff_start_agent,
                    ),
                    author_name_map,
                    handoff_hil_signal_state,
                    workflow_resources,
                )

            controller_agent = self._create_controller_agent(instructions=rendered_instructions)
            if pattern == "group-chat":
                workflow_builder = GroupChatBuilder(
                    participants=participants,
                    orchestrator_agent=controller_agent,
                    orchestrator_name=self.name,
                    max_rounds=self.termination.max_turns,
                    intermediate_outputs=True,
                )
                return (
                    pattern,
                    workflow_builder.build(),
                    author_name_map,
                    None,
                    workflow_resources,
                )

            return (
                pattern,
                MagenticBuilder(
                    participants=participants,
                    enable_plan_review=False,
                    manager_agent=controller_agent,
                    max_round_count=self.termination.max_turns,
                    intermediate_outputs=True,
                ).build(),
                author_name_map,
                None,
                workflow_resources,
            )
        except Exception:
            await workflow_resources.aclose()
            raise

    @staticmethod
    def _apply_mcp_defaults(
        agent_config: AgentConfig,
        *,
        shared_mcp_servers: Sequence[str] = (),
        local_mcp_servers: Sequence[str] = (),
        mcp_registry_dir: Path | None = None,
    ) -> AgentConfig:
        merged_mcp_servers = Mcp.merge_server_names(shared_mcp_servers, agent_config.mcp_servers, local_mcp_servers)
        if merged_mcp_servers == agent_config.mcp_servers and mcp_registry_dir == agent_config.mcp_registry_dir:
            return agent_config
        return replace(agent_config, mcp_servers=merged_mcp_servers, mcp_registry_dir=mcp_registry_dir or agent_config.mcp_registry_dir)

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
