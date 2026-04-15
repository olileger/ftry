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
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.description = description
        self.require_per_service_call_history_persistence = require_per_service_call_history_persistence

    def create_session(self) -> object:
        session = object()
        FakeAgent.created_sessions.append(session)
        return session

    async def run(self, prompt: str, *, options: object | None = None, **kwargs: object) -> FakeResult:
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
                or {"workflow_type": "magentic", "reason": "Stubbed workflow inference result."},
            )
        return FakeResult("Poeme genere")


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
        require_per_service_call_history_persistence: bool = False,
    ) -> FakeAgent:
        agent = FakeAgent(
            name=name,
            instructions=instructions,
            description=description,
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


class FakeWorkflowEvent:
    def __init__(self, type: str, data: object = None, executor_id: str | None = None) -> None:
        self.type = type
        self.data = data
        self.executor_id = executor_id


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

    def __init__(self, pattern: str, **kwargs: object) -> None:
        self.pattern = pattern
        self.kwargs = kwargs

    def _participant_names(self) -> list[str]:
        participants = self.kwargs.get("participants", [])
        return [participant.name for participant in participants]

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

    def run(self, prompt: str, *, stream: bool = False) -> FakeWorkflowStream:
        FakeWorkflow.last_prompt = prompt
        assert stream is True

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
                    FakeWorkflowEvent("executor_invoked", executor_id=source),
                    FakeWorkflowEvent(
                        "output",
                        FakeWorkflowResult([FakeWorkflowMessage("assistant", "Initial triage", author_name=source)]),
                        executor_id=source,
                    ),
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
        return FakeWorkflowStream(events)


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
        return instance

    def with_start_agent(self, agent: FakeAgent) -> "FakeHandoffBuilder":
        FakeHandoffBuilder.last_start_agent = agent
        return self

    def with_autonomous_mode(self, **kwargs: object) -> "FakeHandoffBuilder":
        FakeHandoffBuilder.last_autonomous_kwargs = dict(kwargs)
        return self

    def with_termination_condition(self, termination_condition: object) -> "FakeHandoffBuilder":
        FakeHandoffBuilder.last_termination_condition = termination_condition
        return self

    def build(self) -> FakeWorkflow:
        return FakeWorkflow("handoff", **(self.last_kwargs or {}))


def make_fake_agent_framework_modules() -> dict[str, types.ModuleType]:
    fake_package = types.ModuleType("agent_framework")
    fake_openai_module = types.ModuleType("agent_framework.openai")
    fake_openai_module.OpenAIChatCompletionClient = FakeOpenAIChatCompletionClient
    fake_orchestrations_module = types.ModuleType("agent_framework.orchestrations")
    fake_orchestrations_module.SequentialBuilder = FakeSequentialBuilder
    fake_orchestrations_module.ConcurrentBuilder = FakeConcurrentBuilder
    fake_orchestrations_module.GroupChatBuilder = FakeGroupChatBuilder
    fake_orchestrations_module.HandoffBuilder = FakeHandoffBuilder
    fake_orchestrations_module.MagenticBuilder = FakeMagenticBuilder
    return {
        "agent_framework": fake_package,
        "agent_framework.openai": fake_openai_module,
        "agent_framework.orchestrations": fake_orchestrations_module,
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
    FakeWorkflow.last_prompt = None
    FakeSequentialBuilder.last_kwargs = None
    FakeConcurrentBuilder.last_kwargs = None
    FakeGroupChatBuilder.last_kwargs = None
    FakeMagenticBuilder.last_kwargs = None
    FakeHandoffBuilder.last_kwargs = None
    FakeHandoffBuilder.last_start_agent = None
    FakeHandoffBuilder.last_autonomous_kwargs = None
    FakeHandoffBuilder.last_termination_condition = None
