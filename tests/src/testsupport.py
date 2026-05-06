from __future__ import annotations

import io
import re
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "samples"
SAMPLE_AGENT_FILE = SAMPLES_DIR / "agents" / "poete.yaml"
SAMPLE_TEAM_FILE = SAMPLES_DIR / "teams" / "better-prompt" / "team.yaml"
SEQUENTIAL_SAMPLE_TEAM_FILE = SAMPLES_DIR / "teams" / "seq-support-brief-team" / "team.yaml"
CONCURRENT_SAMPLE_TEAM_FILE = SAMPLES_DIR / "teams" / "con-release-readiness-team" / "team.yaml"
GROUP_CHAT_SAMPLE_TEAM_FILE = SAMPLES_DIR / "teams" / "grp-feature-debate-team" / "team.yaml"
HANDOFF_SAMPLE_TEAM_FILE = SAMPLES_DIR / "teams" / "han-support-routing-team" / "team.yaml"
MAGENTIC_SAMPLE_TEAM_FILE = SAMPLES_DIR / "teams" / "mag-launch-planning-team" / "team.yaml"


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class FakeResult:
    def __init__(self, text: str, *, value: object | None = None) -> None:
        self.text = text
        self.value = value


class FakeAgent:
    last_prompt: str | None = None
    last_options: object | None = None
    next_value: object | None = None
    queued_results: list[FakeResult] = []
    run_calls: list[dict[str, object]] = []
    created_sessions: list[object] = []

    def __init__(
        self,
        name: str,
        instructions: str,
        description: str | None = None,
        require_per_service_call_history_persistence: bool = False,
        tools: object | None = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.description = description
        self.require_per_service_call_history_persistence = require_per_service_call_history_persistence
        self.tools = tools
        self.default_options: dict[str, object] = {}
        self.middleware: list[object] = []
        self.agent_middleware: list[object] = []
        self._cached_agent_middleware_pipeline: object | None = None

    def create_session(self) -> object:
        session = object()
        FakeAgent.created_sessions.append(session)
        return session

    async def run(self, prompt: str, *, options: object | None = None, **kwargs: object) -> FakeResult:
        effective_options = dict(self.default_options)
        if isinstance(options, dict):
            effective_options.update(options)
            options = effective_options
        elif effective_options:
            options = effective_options
        FakeAgent.last_prompt = prompt
        FakeAgent.last_options = options
        FakeAgent.run_calls.append(
            {
                "prompt": prompt,
                "options": options,
                "kwargs": dict(kwargs),
            }
        )
        if isinstance(options, dict) and "response_format" in options:
            if FakeAgent.queued_results:
                return FakeAgent.queued_results.pop(0)
            response_format = options["response_format"]
            schema_name = response_format.get("json_schema", {}).get("name") if isinstance(response_format, dict) else None
            if schema_name == "agent_turn_response":
                return FakeResult(
                    "Structured agent turn",
                    value={"status": "done", "message": "Poeme genere"},
                )
            return FakeResult(
                "Structured workflow inference",
                value=FakeAgent.next_value
                or {
                    "workflow_type": "magentic",
                    "reason": "Stubbed workflow inference result.",
                    "human_in_the_loop": {
                        "enabled": False,
                        "reason": "No human input needed.",
                        "agent_names": [],
                    },
                },
            )
        return FakeResult("Poeme genere")


class FakeTool:
    def __init__(self, func, *, name: str | None = None, description: str | None = None, approval_mode: str | None = None) -> None:
        self.func = func
        self.name = name or func.__name__
        self.description = description
        self.approval_mode = approval_mode

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.func(*args, **kwargs)


class _FakeMcpToolBase:
    created_tools: list["_FakeMcpToolBase"] = []
    entered_tools: list["_FakeMcpToolBase"] = []
    closed_tools: list["_FakeMcpToolBase"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.name = kwargs.get("name")
        self.description = kwargs.get("description")
        self.closed = False
        type(self).created_tools.append(self)

    async def __aenter__(self) -> "_FakeMcpToolBase":
        type(self).entered_tools.append(self)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        self.closed = True
        type(self).closed_tools.append(self)


class FakeMCPStdioTool(_FakeMcpToolBase):
    pass


class FakeMCPStreamableHTTPTool(_FakeMcpToolBase):
    pass


class FakeMCPWebsocketTool(_FakeMcpToolBase):
    pass


class FakeOpenAIChatCompletionClient:
    last_model: str | None = None
    last_api_key: str | None = None
    last_agent: FakeAgent | None = None

    def __init__(self, *, model: str, api_key: str) -> None:
        FakeOpenAIChatCompletionClient.last_model = model
        FakeOpenAIChatCompletionClient.last_api_key = api_key

    def as_agent(
        self,
        *,
        name: str,
        instructions: str,
        description: str | None = None,
        tools: object | None = None,
        require_per_service_call_history_persistence: bool = False,
    ) -> FakeAgent:
        agent = FakeAgent(
            name=name,
            instructions=instructions,
            description=description,
            tools=tools,
            require_per_service_call_history_persistence=require_per_service_call_history_persistence,
        )
        FakeOpenAIChatCompletionClient.last_agent = agent
        return agent


class FakeTtyStream(io.StringIO):
    def isatty(self) -> bool:
        return True


class FakeWorkflowMessage:
    def __init__(self, role: str, text: str, author_name: str | None = None) -> None:
        self.role = role
        self.text = text
        self.author_name = author_name


class FakeWorkflowResult:
    def __init__(self, messages: list[FakeWorkflowMessage]) -> None:
        self.messages = messages
        self.text = next(
            (
                message.text
                for message in reversed(messages)
                if isinstance(message.text, str) and message.text.strip()
            ),
            "",
        )


class FakeAgentResponse:
    def __init__(self, *, messages: list[FakeWorkflowMessage] | None = None, value: object | None = None, **kwargs: object) -> None:
        self.messages = messages or []
        self.value = value
        self.user_input_requests = kwargs.get("user_input_requests", [])
        self.response_id = kwargs.get("response_id")
        self.agent_id = kwargs.get("agent_id")
        self.created_at = kwargs.get("created_at")
        self.usage_details = kwargs.get("usage_details")
        self.raw_representation = kwargs.get("raw_representation")
        self.additional_properties = kwargs.get("additional_properties")


class FakeMessage(FakeWorkflowMessage):
    def __init__(
        self,
        role: str,
        contents: list[object] | None = None,
        *,
        author_name: str | None = None,
        additional_properties: dict[str, object] | None = None,
    ) -> None:
        text_parts = [
            content if isinstance(content, str) else getattr(content, "text", None)
            for content in (contents or [])
        ]
        text_parts = [part for part in text_parts if isinstance(part, str)]
        super().__init__(role, " ".join(text_parts), author_name=author_name)
        self.contents = contents or []
        self.additional_properties = additional_properties or {}


class FakeContent:
    def __init__(
        self,
        type: str = "text",
        *,
        content_id: str = "content-id",
        text: str | None = None,
        additional_properties: dict[str, object] | None = None,
    ) -> None:
        self.id = content_id
        self.type = type
        self.text = text
        self.user_input_request = False
        self.additional_properties = additional_properties or {}

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        additional_properties: dict[str, object] | None = None,
        raw_representation: object | None = None,
    ) -> "FakeContent":
        del raw_representation
        return cls(text=text, additional_properties=additional_properties)


class FakeResponseStream:
    def __init__(self, stream: object, *, finalizer: object | None = None) -> None:
        self.stream = stream
        self.finalizer = finalizer



class FakeAgentExecutorResponse:
    def __init__(
        self,
        executor_id: str,
        agent_response: FakeWorkflowResult,
        *,
        full_conversation: list[FakeWorkflowMessage] | None = None,
    ) -> None:
        self.executor_id = executor_id
        self.agent_response = agent_response
        self.full_conversation = full_conversation or []


class FakeAgentRequestInfoResponse:
    @staticmethod
    def approve() -> tuple[str]:
        return ("approve",)

    @staticmethod
    def from_strings(values: list[str]) -> tuple[str, list[str]]:
        return ("feedback", values)


class FakeAgentInputRequest:
    def __init__(
        self,
        *,
        target_agent_id: str | None,
        conversation: list[FakeWorkflowMessage] | None = None,
        instruction: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.target_agent_id = target_agent_id
        self.conversation = conversation or []
        self.instruction = instruction
        self.metadata = metadata or {}


class FakeHandoffAgentUserRequest:
    def __init__(self, agent_response: FakeWorkflowResult) -> None:
        self.agent_response = agent_response

    def create_response(self, text: str) -> tuple[str, str]:
        return ("handoff-response", text)

    @staticmethod
    def terminate() -> tuple[str]:
        return ("terminate",)


class FakePlan:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeMagenticPlanReviewRequest:
    def __init__(self, plan_text: str) -> None:
        self.plan = FakePlan(plan_text)
        self.current_progress = None

    def approve(self) -> tuple[str]:
        return ("approve-plan",)

    def revise(self, feedback: str) -> tuple[str, str]:
        return ("revise-plan", feedback)


class FakeWorkflowEvent:
    def __init__(
        self,
        type: str,
        data: object = None,
        executor_id: str | None = None,
        *,
        request_id: str | None = None,
        request_type: object | None = None,
        state: object | None = None,
    ) -> None:
        self.type = type
        self.data = data
        self.executor_id = executor_id
        self.request_id = request_id
        self.request_type = request_type
        self.state = state


class FakeGroupChatRequestSentEvent:
    def __init__(self, participant_name: str) -> None:
        self.participant_name = participant_name


class FakeGroupChatResponseReceivedEvent:
    def __init__(self, participant_name: str) -> None:
        self.participant_name = participant_name


class FakeHandoffSentEvent:
    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target


class FakeWorkflowStream:
    def __init__(self, events: list[FakeWorkflowEvent]) -> None:
        self._events = events
        self._index = 0

    def __aiter__(self) -> "FakeWorkflowStream":
        return self

    async def __anext__(self) -> FakeWorkflowEvent:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


class FakeWorkflow:
    last_prompt: str | None = None
    last_responses: dict[str, object] | None = None
    run_calls: list[dict[str, object]] = []

    def __init__(self, pattern: str, **kwargs: object) -> None:
        self.pattern = pattern
        self.kwargs = kwargs
        self._last_prompt: str | None = None
        self._request_phase = 0

    def _participant_names(self) -> list[str]:
        participants = self.kwargs.get("participants", [])
        return [
            getattr(participant, "name", None) or getattr(participant, "id", None) or "participant"
            for participant in participants
        ]

    def _team_name(self) -> str:
        if self.pattern == "group-chat":
            orchestrator_name = self.kwargs.get("orchestrator_name")
            if isinstance(orchestrator_name, str) and orchestrator_name:
                return orchestrator_name.replace(" ", "-")
            orchestrator = self.kwargs.get("orchestrator_agent")
            return getattr(orchestrator, "name", "team")
        if self.pattern == "magentic":
            manager = self.kwargs.get("manager_agent")
            return getattr(manager, "name", "team")
        participant_names = self._participant_names()
        if participant_names:
            return participant_names[-1]
        return "team"

    def _request_target(self) -> str | None:
        if self.kwargs.get("request_info_enabled"):
            request_info_agents = self.kwargs.get("request_info_agents")
            if isinstance(request_info_agents, list) and request_info_agents:
                target = request_info_agents[0]
                if isinstance(target, str) and target:
                    return target
        participant_names = self._participant_names()
        return participant_names[0] if participant_names else None

    def _build_final_output_events(self, prompt: str) -> list[FakeWorkflowEvent]:
        participant_names = self._participant_names()
        events: list[FakeWorkflowEvent] = []

        if self.pattern == "group-chat" and len(participant_names) >= 2:
            first, second = participant_names[:2]
            events.extend(
                [
                    FakeWorkflowEvent("group_chat", FakeGroupChatRequestSentEvent(first)),
                    FakeWorkflowEvent("executor_invoked", executor_id=first),
                    FakeWorkflowEvent(
                        "output",
                        FakeWorkflowResult([FakeWorkflowMessage("assistant", "Draft prompt", author_name=first)]),
                        executor_id=first,
                    ),
                    FakeWorkflowEvent("group_chat", FakeGroupChatResponseReceivedEvent(first)),
                    FakeWorkflowEvent("group_chat", FakeGroupChatRequestSentEvent(second)),
                    FakeWorkflowEvent("executor_invoked", executor_id=second),
                    FakeWorkflowEvent(
                        "output",
                        FakeWorkflowResult([FakeWorkflowMessage("assistant", "Review feedback", author_name=second)]),
                        executor_id=second,
                    ),
                ]
            )
        elif self.pattern == "handoff" and len(participant_names) >= 2:
            source, target = participant_names[:2]
            events.extend(
                [
                    FakeWorkflowEvent("handoff_sent", FakeHandoffSentEvent(source, target)),
                    FakeWorkflowEvent("executor_invoked", executor_id=target),
                    FakeWorkflowEvent(
                        "output",
                        FakeWorkflowResult([FakeWorkflowMessage("assistant", "Specialist answer", author_name=target)]),
                        executor_id=target,
                    ),
                ]
            )
        else:
            for participant_name in participant_names:
                events.append(FakeWorkflowEvent("executor_invoked", executor_id=participant_name))
                events.append(
                    FakeWorkflowEvent(
                        "output",
                        FakeWorkflowResult(
                            [FakeWorkflowMessage("assistant", f"{participant_name} handled {prompt}", author_name=participant_name)]
                        ),
                        executor_id=participant_name,
                    )
                )

        events.append(
            FakeWorkflowEvent(
                "output",
                [FakeWorkflowMessage("assistant", f"{self.pattern}:{prompt}", author_name=self._team_name())],
                executor_id=self._team_name(),
            )
        )
        return events

    def run(
        self,
        prompt: str | None = None,
        *,
        stream: bool = False,
        responses: dict[str, object] | None = None,
    ) -> FakeWorkflowStream:
        if prompt is not None:
            FakeWorkflow.last_prompt = prompt
            self._last_prompt = prompt
        FakeWorkflow.last_responses = responses
        FakeWorkflow.run_calls.append(
            {
                "prompt": prompt,
                "responses": responses,
                "stream": stream,
            }
        )
        assert stream is True

        effective_prompt = self._last_prompt or ""
        request_target = self._request_target()
        is_handoff_resume_prompt = (
            self.pattern == "handoff"
            and isinstance(effective_prompt, str)
            and "Additional user information collected during the handoff workflow:" in effective_prompt
        )

        if (
            self.pattern == "sequential"
            and request_target is not None
            and self.kwargs.get("request_info_enabled")
            and responses is None
            and self._request_phase == 0
        ):
            self._request_phase = 1
            request = FakeContent.from_text(f"{request_target} needs feedback about {effective_prompt}")
            request.user_input_request = True
            return FakeWorkflowStream(
                [
                    FakeWorkflowEvent(
                        "request_info",
                        request,
                        executor_id=request_target,
                        request_id="req-sequential",
                    ),
                ]
            )

        if (
            self.pattern == "group-chat"
            and request_target is not None
            and self.kwargs.get("request_info_enabled")
            and responses is None
            and self._request_phase == 0
        ):
            self._request_phase = 1
            request = FakeContent.from_text(f"{request_target} needs guidance before speaking about {effective_prompt}")
            request.user_input_request = True
            return FakeWorkflowStream(
                [
                    FakeWorkflowEvent("group_chat", FakeGroupChatRequestSentEvent(request_target)),
                    FakeWorkflowEvent(
                        "request_info",
                        request,
                        executor_id=request_target,
                        request_id="req-group-chat",
                    ),
                ]
            )

        if (
            self.pattern == "handoff"
            and request_target is not None
            and responses is None
            and self._request_phase == 0
            and not is_handoff_resume_prompt
        ):
            self._request_phase = 1
            signal_state = self.kwargs.get("handoff_hil_signal_state")
            if signal_state is not None:
                signal_state.request_user_input("Initial triage", request_target)
            return FakeWorkflowStream(
                [
                    FakeWorkflowEvent("executor_invoked", executor_id=request_target),
                    FakeWorkflowEvent(
                        "output",
                        FakeWorkflowResult([FakeWorkflowMessage("assistant", "Initial triage", author_name=request_target)]),
                        executor_id=request_target,
                    ),
                ]
            )

        if (
            self.pattern == "handoff"
            and responses is None
            and (self._request_phase == 1 or is_handoff_resume_prompt)
        ):
            participant_names = self._participant_names()
            source = participant_names[0] if participant_names else "router"
            target = participant_names[1] if len(participant_names) > 1 else source
            signal_state = self.kwargs.get("handoff_hil_signal_state")
            if signal_state is not None:
                signal_state.finalize("Specialist answer", target)
            self._request_phase = 2
            return FakeWorkflowStream(
                [
                    FakeWorkflowEvent("handoff_sent", FakeHandoffSentEvent(source, target)),
                    FakeWorkflowEvent("executor_invoked", executor_id=target),
                    FakeWorkflowEvent(
                        "output",
                        FakeWorkflowResult([FakeWorkflowMessage("assistant", "Specialist answer", author_name=target)]),
                        executor_id=target,
                    ),
                ]
            )

        if self.pattern == "magentic" and self.kwargs.get("enable_plan_review") and responses is None and self._request_phase == 0:
            self._request_phase = 1
            return FakeWorkflowStream(
                [
                    FakeWorkflowEvent(
                        "request_info",
                        FakeMagenticPlanReviewRequest(f"Plan for {effective_prompt}"),
                        executor_id=self._team_name(),
                        request_id="req-magentic",
                        request_type=FakeMagenticPlanReviewRequest,
                    )
                ]
            )

        return FakeWorkflowStream(self._build_final_output_events(effective_prompt))


class FakeSequentialBuilder:
    last_kwargs: dict[str, object] | None = None

    def __new__(cls, **kwargs: object) -> "FakeSequentialBuilder":
        instance = super().__new__(cls)
        cls.last_kwargs = dict(kwargs)
        return instance

    def build(self) -> FakeWorkflow:
        return FakeWorkflow("sequential", **(self.last_kwargs or {}))


class FakeConcurrentBuilder:
    last_kwargs: dict[str, object] | None = None

    def __new__(cls, **kwargs: object) -> "FakeConcurrentBuilder":
        instance = super().__new__(cls)
        cls.last_kwargs = dict(kwargs)
        return instance

    def build(self) -> FakeWorkflow:
        return FakeWorkflow("concurrent", **(self.last_kwargs or {}))


class FakeGroupChatBuilder:
    last_kwargs: dict[str, object] | None = None

    def __new__(cls, **kwargs: object) -> "FakeGroupChatBuilder":
        instance = super().__new__(cls)
        cls.last_kwargs = dict(kwargs)
        return instance

    def build(self) -> FakeWorkflow:
        return FakeWorkflow("group-chat", **(self.last_kwargs or {}))


class FakeMagenticBuilder:
    last_kwargs: dict[str, object] | None = None

    def __new__(cls, **kwargs: object) -> "FakeMagenticBuilder":
        instance = super().__new__(cls)
        cls.last_kwargs = dict(kwargs)
        return instance

    def build(self) -> FakeWorkflow:
        return FakeWorkflow("magentic", **(self.last_kwargs or {}))


class FakeHandoffBuilder:
    last_kwargs: dict[str, object] | None = None
    last_start_agent: FakeAgent | None = None
    last_autonomous_kwargs: dict[str, object] | None = None
    last_termination_condition: object | None = None

    def __new__(cls, **kwargs: object) -> "FakeHandoffBuilder":
        instance = super().__new__(cls)
        cls.last_kwargs = dict(kwargs)
        cls.last_start_agent = None
        cls.last_autonomous_kwargs = None
        cls.last_termination_condition = None
        return instance

    def with_start_agent(self, agent: FakeAgent) -> "FakeHandoffBuilder":
        FakeHandoffBuilder.last_start_agent = agent
        return self

    def with_autonomous_mode(self, **kwargs: object) -> "FakeHandoffBuilder":
        FakeHandoffBuilder.last_autonomous_kwargs = dict(kwargs)
        assert self.last_kwargs is not None
        self.last_kwargs["autonomous_mode"] = True
        return self

    def with_request_info(self, *, agents: list[str] | None = None) -> "FakeHandoffBuilder":
        assert self.last_kwargs is not None
        self.last_kwargs["request_info_enabled"] = True
        if agents is not None:
            self.last_kwargs["request_info_agents"] = list(agents)
        return self

    def with_termination_condition(self, termination_condition: object) -> "FakeHandoffBuilder":
        FakeHandoffBuilder.last_termination_condition = termination_condition
        return self

    def build(self) -> FakeWorkflow:
        return FakeWorkflow("handoff", **(self.last_kwargs or {}))


def make_fake_agent_framework_modules() -> dict[str, types.ModuleType]:
    def fake_response_handler(func=None, **kwargs: object):
        def decorator(inner):
            return inner

        if func is not None and callable(func) and not kwargs:
            return decorator(func)
        return decorator

    def fake_agent_middleware(func):
        func._middleware_type = "agent"
        return func

    def fake_tool(*, name: str | None = None, description: str | None = None, approval_mode: str | None = None):
        def decorator(func):
            return FakeTool(
                func,
                name=name,
                description=description,
                approval_mode=approval_mode,
            )

        return decorator

    fake_package = types.ModuleType("agent_framework")
    fake_package.AgentExecutorResponse = FakeAgentExecutorResponse
    fake_package.AgentInputRequest = FakeAgentInputRequest
    fake_package.AgentExecutor = type(
        "FakeAgentExecutor",
        (),
        {
            "__init__": lambda self, agent, **kwargs: (
                setattr(self, "_agent", agent),
                setattr(self, "_session", object()),
                setattr(self, "_pending_agent_requests", {}),
                setattr(self, "_cache", []),
                setattr(self, "_full_conversation", []),
                setattr(self, "id", kwargs.get("id") or getattr(agent, "name", "agent")),
                setattr(self, "name", kwargs.get("id") or getattr(agent, "name", "agent")),
            )[-1],
            "_prepare_agent_run_args": lambda self, raw_kwargs: (None, None),
        },
    )
    fake_package.AgentResponse = FakeAgentResponse
    fake_package.Content = FakeContent
    fake_package.MCPStdioTool = FakeMCPStdioTool
    fake_package.MCPStreamableHTTPTool = FakeMCPStreamableHTTPTool
    fake_package.MCPWebsocketTool = FakeMCPWebsocketTool
    fake_package.Message = FakeMessage
    fake_package.ResponseStream = FakeResponseStream
    fake_package.agent_middleware = fake_agent_middleware
    fake_package.response_handler = fake_response_handler
    fake_package.tool = fake_tool
    fake_openai_module = types.ModuleType("agent_framework.openai")
    fake_openai_module.OpenAIChatCompletionClient = FakeOpenAIChatCompletionClient
    fake_orchestrations_module = types.ModuleType("agent_framework.orchestrations")
    fake_orchestrations_module.SequentialBuilder = FakeSequentialBuilder
    fake_orchestrations_module.ConcurrentBuilder = FakeConcurrentBuilder
    fake_orchestrations_module.GroupChatBuilder = FakeGroupChatBuilder
    fake_orchestrations_module.HandoffAgentUserRequest = FakeHandoffAgentUserRequest
    fake_orchestrations_module.HandoffBuilder = FakeHandoffBuilder
    fake_orchestrations_module.MagenticBuilder = FakeMagenticBuilder
    fake_mcp_module = types.ModuleType("mcp")
    return {
        "agent_framework": fake_package,
        "agent_framework.openai": fake_openai_module,
        "agent_framework.orchestrations": fake_orchestrations_module,
        "mcp": fake_mcp_module,
    }


def reset_fakes() -> None:
    FakeAgent.last_prompt = None
    FakeAgent.last_options = None
    FakeAgent.next_value = None
    FakeAgent.queued_results = []
    FakeAgent.run_calls = []
    FakeAgent.created_sessions = []
    FakeOpenAIChatCompletionClient.last_model = None
    FakeOpenAIChatCompletionClient.last_api_key = None
    FakeOpenAIChatCompletionClient.last_agent = None
    FakeMCPStdioTool.created_tools = []
    FakeMCPStdioTool.entered_tools = []
    FakeMCPStdioTool.closed_tools = []
    FakeMCPStreamableHTTPTool.created_tools = []
    FakeMCPStreamableHTTPTool.entered_tools = []
    FakeMCPStreamableHTTPTool.closed_tools = []
    FakeMCPWebsocketTool.created_tools = []
    FakeMCPWebsocketTool.entered_tools = []
    FakeMCPWebsocketTool.closed_tools = []
    FakeWorkflow.last_prompt = None
    FakeWorkflow.last_responses = None
    FakeWorkflow.run_calls = []
    FakeSequentialBuilder.last_kwargs = None
    FakeConcurrentBuilder.last_kwargs = None
    FakeGroupChatBuilder.last_kwargs = None
    FakeMagenticBuilder.last_kwargs = None
    FakeHandoffBuilder.last_kwargs = None
    FakeHandoffBuilder.last_start_agent = None
    FakeHandoffBuilder.last_autonomous_kwargs = None
    FakeHandoffBuilder.last_termination_condition = None
