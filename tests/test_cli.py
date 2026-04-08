from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ftry import cli


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeAgent:
    last_prompt: str | None = None

    def __init__(self, name: str, instructions: str, description: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.description = description

    async def run(self, prompt: str) -> FakeResult:
        FakeAgent.last_prompt = prompt
        return FakeResult("Poeme genere")


class FakeOpenAIChatCompletionClient:
    last_model: str | None = None
    last_api_key: str | None = None
    last_agent: FakeAgent | None = None

    def __init__(self, *, model: str, api_key: str) -> None:
        FakeOpenAIChatCompletionClient.last_model = model
        FakeOpenAIChatCompletionClient.last_api_key = api_key

    def as_agent(self, *, name: str, instructions: str, description: str | None = None) -> FakeAgent:
        agent = FakeAgent(name=name, instructions=instructions, description=description)
        FakeOpenAIChatCompletionClient.last_agent = agent
        return agent


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


class CliTests(unittest.TestCase):
    def _write_agent_file(self, provider: str = "openai") -> str:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False) as handle:
            handle.write(
                "\n".join(
                    [
                        "name: Poete",
                        "model:",
                        "  name: gpt-4o-2024-08-06",
                        f"  provider: {provider}",
                        "  api-key: env:OAI_API_KEY",
                        "prompt: |",
                        "  Tu es un poete.",
                    ]
                )
            )
            return handle.name

    def test_build_parser_parses_pop_arguments(self) -> None:
        args = cli.build_parser().parse_args(["pop", "-a", r"samples\poete.yaml", "-p", "Bonjour"])
        self.assertEqual(args.command, "pop")
        self.assertEqual(args.agent_file, r"samples\poete.yaml")
        self.assertIsNone(args.team_file)
        self.assertEqual(args.prompt, "Bonjour")

    def test_build_parser_parses_pop_team_arguments(self) -> None:
        args = cli.build_parser().parse_args(["pop", "-t", r"samples\team.yaml", "-p", "Bonjour"])
        self.assertEqual(args.command, "pop")
        self.assertEqual(args.team_file, r"samples\team.yaml")
        self.assertIsNone(args.agent_file)
        self.assertEqual(args.prompt, "Bonjour")

    def test_infer_team_pattern_selects_expected_agent_framework_pattern(self) -> None:
        agent = cli.AgentConfig(
            name="Agent",
            description="Helpful specialist.",
            instructions="Do the work.",
            model=cli.AgentModelConfig(name="gpt-4o", provider="openai", api_key="secret"),
        )

        self.assertEqual(
            cli._infer_team_pattern(cli.TeamConfig(name="Triage", instructions="Route the request and handoff to specialists.", agents=(agent,))),
            "handoff",
        )
        self.assertEqual(
            cli._infer_team_pattern(cli.TeamConfig(name="Research swarm", instructions="Work in parallel on independent aspects.", agents=(agent,))),
            "concurrent",
        )
        self.assertEqual(
            cli._infer_team_pattern(cli.TeamConfig(name="Planner", instructions="Create a plan, replan if needed, and manage a complex task.", agents=(agent,))),
            "magentic",
        )
        self.assertEqual(
            cli._infer_team_pattern(cli.TeamConfig(name="Workshop", instructions="Discuss ideas, review them, and iterate together.", agents=(agent,))),
            "group-chat",
        )
        self.assertEqual(
            cli._infer_team_pattern(cli.TeamConfig(name="Pipeline", instructions="First collect data, then write the answer, and finally polish it.", agents=(agent,))),
            "sequential",
        )

    def test_sanitize_agent_name_returns_openai_compatible_name(self) -> None:
        self.assertEqual(cli._sanitize_agent_name("Better Prompt team"), "Better-Prompt-team")
        self.assertEqual(cli._sanitize_agent_name("agent/triage<v1>"), "agent-triage-v1")
        self.assertEqual(cli._sanitize_agent_name("   "), "agent")

    def test_summarize_trace_text_preserves_useful_newlines(self) -> None:
        self.assertEqual(
            cli._summarize_trace_text("# Type of problemCe sujet demande un poeme."),
            "# Type of problem\nCe sujet demande un poeme.",
        )

    def test_trace_block_indents_multiline_text(self) -> None:
        self.assertEqual(cli._trace_block("a\nb"), "\n\ta\n\tb")

    def test_build_agent_trace_colors_assigns_stable_colors(self) -> None:
        colors = cli._build_agent_trace_colors(["Prompter", "Reviewer", "Runner", "Prompter"])
        self.assertEqual(colors["Prompter"], cli.BRIGHT_PINK)
        self.assertEqual(colors["Reviewer"], cli.BRIGHT_BLUE)
        self.assertEqual(colors["Runner"], cli.PURPLE)

    def test_pop_runs_agent_loaded_from_yaml(self) -> None:
        agent_file = self._write_agent_file()
        fake_package = types.ModuleType("agent_framework")
        fake_openai_module = types.ModuleType("agent_framework.openai")
        fake_openai_module.OpenAIChatCompletionClient = FakeOpenAIChatCompletionClient
        stdout = io.StringIO()
        stderr = io.StringIO()

        try:
            with (
                patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
                patch.dict(
                    sys.modules,
                    {"agent_framework": fake_package, "agent_framework.openai": fake_openai_module},
                    clear=False,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main(["pop", "-a", agent_file, "-p", "Ecris un poeme sur la pluie"])
        finally:
            Path(agent_file).unlink(missing_ok=True)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "Poeme genere")
        self.assertEqual(FakeOpenAIChatCompletionClient.last_model, "gpt-4o-2024-08-06")
        self.assertEqual(FakeOpenAIChatCompletionClient.last_api_key, "secret-key")
        self.assertIsNotNone(FakeOpenAIChatCompletionClient.last_agent)
        self.assertEqual(FakeOpenAIChatCompletionClient.last_agent.name, "Poete")
        self.assertEqual(FakeOpenAIChatCompletionClient.last_agent.instructions, "Tu es un poete.")
        self.assertEqual(FakeAgent.last_prompt, "Ecris un poeme sur la pluie")
        plain_stderr = _strip_ansi(stderr.getvalue())
        self.assertIn("AGENT Poete | input:", plain_stderr)
        self.assertIn("AGENT Poete | output:", plain_stderr)

    def test_pop_runs_team_loaded_from_yaml_file_references(self) -> None:
        fake_package = types.ModuleType("agent_framework")
        fake_openai_module = types.ModuleType("agent_framework.openai")
        fake_openai_module.OpenAIChatCompletionClient = FakeOpenAIChatCompletionClient
        fake_orchestrations_module = types.ModuleType("agent_framework.orchestrations")
        fake_orchestrations_module.SequentialBuilder = FakeSequentialBuilder
        fake_orchestrations_module.ConcurrentBuilder = FakeConcurrentBuilder
        fake_orchestrations_module.GroupChatBuilder = FakeGroupChatBuilder
        fake_orchestrations_module.HandoffBuilder = FakeHandoffBuilder
        fake_orchestrations_module.MagenticBuilder = FakeMagenticBuilder
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for file_name, name, description in (
                ("pr.yaml", "Prompter", "Builds prompts."),
                ("re.yaml", "Reviewer", "Reviews prompts."),
                ("ru.yaml", "Runner", "Runs prompts."),
            ):
                (temp_path / file_name).write_text(
                    "\n".join(
                        [
                            f"name: {name}",
                            "description: |",
                            f"  {description}",
                            "model:",
                            "  name: gpt-4o-2024-08-06",
                            "  provider: openai",
                            "  api-key: env:OAI_API_KEY",
                            "prompt: |",
                            f"  You are {name}.",
                        ]
                    ),
                    encoding="utf-8",
                )

            team_file = temp_path / "team.yaml"
            team_file.write_text(
                "\n".join(
                    [
                        "name: Better Prompt team",
                        "model:",
                        "  name: gpt-4o-2024-08-06",
                        "  provider: openai",
                        "  api-key: env:OAI_API_KEY",
                        "termination:",
                        "  max-turns: 10",
                        "agents:",
                        "  - file: ./pr.yaml",
                        "  - file: ./re.yaml",
                        "  - file: ./ru.yaml",
                        "prompt: |",
                        "  You have these tools: {participants}.",
                        "  {roles}",
                        "  Select the most appropriate tool and iterate with review if needed.",
                    ]
                ),
                encoding="utf-8",
            )

            with (
                patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
                patch.dict(
                    sys.modules,
                    {
                        "agent_framework": fake_package,
                        "agent_framework.openai": fake_openai_module,
                        "agent_framework.orchestrations": fake_orchestrations_module,
                    },
                    clear=False,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main(["pop", "-t", str(team_file), "-p", "Ameliore ce prompt"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "[Better Prompt team]\ngroup-chat:Ameliore ce prompt")
        self.assertEqual(FakeWorkflow.last_prompt, "Ameliore ce prompt")
        self.assertIsNotNone(FakeGroupChatBuilder.last_kwargs)
        self.assertEqual(FakeGroupChatBuilder.last_kwargs["max_rounds"], 10)
        orchestrator_agent = FakeGroupChatBuilder.last_kwargs["orchestrator_agent"]
        self.assertIsInstance(orchestrator_agent, FakeAgent)
        self.assertEqual(orchestrator_agent.name, "Better-Prompt-team")
        self.assertIn("Prompter, Reviewer, Runner", orchestrator_agent.instructions)
        self.assertIn("- Prompter: Builds prompts.", orchestrator_agent.instructions)
        plain_stderr = _strip_ansi(stderr.getvalue())
        self.assertIn("TEAM Better Prompt team | pattern: group-chat | input:", plain_stderr)
        self.assertIn("TEAM Better Prompt team --> Prompter | input:", plain_stderr)
        self.assertIn("TEAM Better Prompt team <-- Prompter | output:", plain_stderr)

    def test_pop_runs_team_loaded_from_project_relative_file_references(self) -> None:
        fake_package = types.ModuleType("agent_framework")
        fake_openai_module = types.ModuleType("agent_framework.openai")
        fake_openai_module.OpenAIChatCompletionClient = FakeOpenAIChatCompletionClient
        fake_orchestrations_module = types.ModuleType("agent_framework.orchestrations")
        fake_orchestrations_module.SequentialBuilder = FakeSequentialBuilder
        fake_orchestrations_module.ConcurrentBuilder = FakeConcurrentBuilder
        fake_orchestrations_module.GroupChatBuilder = FakeGroupChatBuilder
        fake_orchestrations_module.HandoffBuilder = FakeHandoffBuilder
        fake_orchestrations_module.MagenticBuilder = FakeMagenticBuilder
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            samples_dir = temp_path / "samples"
            samples_dir.mkdir()
            for file_name, name in (("pr.yaml", "Prompter"), ("re.yaml", "Reviewer"), ("ru.yaml", "Runner")):
                (samples_dir / file_name).write_text(
                    "\n".join(
                        [
                            f"name: {name}",
                            "model:",
                            "  name: gpt-4o-2024-08-06",
                            "  provider: openai",
                            "  api-key: env:OAI_API_KEY",
                            "prompt: |",
                            f"  You are {name}.",
                        ]
                    ),
                    encoding="utf-8",
                )

            team_file = samples_dir / "team.yaml"
            team_file.write_text(
                "\n".join(
                    [
                        "name: Better Prompt team",
                        "agents:",
                        "  - file: ./samples/pr.yaml",
                        "  - file: ./samples/re.yaml",
                        "  - file: ./samples/ru.yaml",
                        "prompt: |",
                        "  Select the most appropriate tool and iterate with review if needed.",
                    ]
                ),
                encoding="utf-8",
            )

            previous_cwd = Path.cwd()
            os.chdir(temp_path)
            try:
                with (
                    patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
                    patch.dict(
                        sys.modules,
                        {
                            "agent_framework": fake_package,
                            "agent_framework.openai": fake_openai_module,
                            "agent_framework.orchestrations": fake_orchestrations_module,
                        },
                        clear=False,
                    ),
                    redirect_stdout(stdout),
                    redirect_stderr(stderr),
                ):
                    exit_code = cli.main(["pop", "-t", r".\samples\team.yaml", "-p", "Ameliore ce prompt"])
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "[Better Prompt team]\ngroup-chat:Ameliore ce prompt")
        self.assertIn("TEAM Better Prompt team | pattern: group-chat | input:", _strip_ansi(stderr.getvalue()))

    def test_pop_runs_team_with_inline_agents_and_sequential_pattern(self) -> None:
        fake_package = types.ModuleType("agent_framework")
        fake_openai_module = types.ModuleType("agent_framework.openai")
        fake_openai_module.OpenAIChatCompletionClient = FakeOpenAIChatCompletionClient
        fake_orchestrations_module = types.ModuleType("agent_framework.orchestrations")
        fake_orchestrations_module.SequentialBuilder = FakeSequentialBuilder
        fake_orchestrations_module.ConcurrentBuilder = FakeConcurrentBuilder
        fake_orchestrations_module.GroupChatBuilder = FakeGroupChatBuilder
        fake_orchestrations_module.HandoffBuilder = FakeHandoffBuilder
        fake_orchestrations_module.MagenticBuilder = FakeMagenticBuilder
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as temp_dir:
            team_file = Path(temp_dir) / "team-inline.yaml"
            team_file.write_text(
                "\n".join(
                    [
                        "name: Pipeline team",
                        "agents:",
                        "  - name: Researcher",
                        "    model:",
                        "      name: gpt-4o-2024-08-06",
                        "      provider: openai",
                        "      api-key: env:OAI_API_KEY",
                        "    prompt: |",
                        "      You gather facts.",
                        "  - name: Writer",
                        "    model:",
                        "      name: gpt-4o-2024-08-06",
                        "      provider: openai",
                        "      api-key: env:OAI_API_KEY",
                        "    prompt: |",
                        "      You write the final answer.",
                        "prompt: |",
                        "  First gather the facts, then draft the answer, and finally produce the final response.",
                    ]
                ),
                encoding="utf-8",
            )

            with (
                patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
                patch.dict(
                    sys.modules,
                    {
                        "agent_framework": fake_package,
                        "agent_framework.openai": fake_openai_module,
                        "agent_framework.orchestrations": fake_orchestrations_module,
                    },
                    clear=False,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main(["pop", "-t", str(team_file), "-p", "Sujet"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "[Writer]\nsequential:Sujet")
        self.assertIsNotNone(FakeSequentialBuilder.last_kwargs)
        participants = FakeSequentialBuilder.last_kwargs["participants"]
        self.assertEqual(len(participants), 2)
        self.assertIn("<TeamContext>", participants[0].instructions)
        self.assertIn("First gather the facts", participants[0].instructions)
        plain_stderr = _strip_ansi(stderr.getvalue())
        self.assertIn("TEAM Pipeline team --> Researcher | input:", plain_stderr)
        self.assertIn("Writer <-- Researcher | output:", plain_stderr)
        self.assertIn("TEAM Pipeline team <-- Writer | final-output:", plain_stderr)

    def test_pop_loads_api_key_from_dotenv_file(self) -> None:
        fake_package = types.ModuleType("agent_framework")
        fake_openai_module = types.ModuleType("agent_framework.openai")
        fake_openai_module.OpenAIChatCompletionClient = FakeOpenAIChatCompletionClient
        stdout = io.StringIO()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            agent_file = temp_path / "poete.yaml"
            agent_file.write_text(
                "\n".join(
                    [
                        "name: Poete",
                        "model:",
                        "  name: gpt-4o-2024-08-06",
                        "  provider: openai",
                        "  api-key: env:OAI_API_KEY",
                        "prompt: |",
                        "  Tu es un poete.",
                    ]
                ),
                encoding="utf-8",
            )
            (temp_path / ".env").write_text("OAI_API_KEY=dotenv-secret\n", encoding="utf-8")

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.dict(
                    sys.modules,
                    {"agent_framework": fake_package, "agent_framework.openai": fake_openai_module},
                    clear=False,
                ),
                patch("ftry.cli.Path.cwd", return_value=temp_path),
                redirect_stdout(stdout),
            ):
                exit_code = cli.main(["pop", "-a", str(agent_file), "-p", "Ecris un poeme sur la pluie"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "Poeme genere")
        self.assertEqual(FakeOpenAIChatCompletionClient.last_api_key, "dotenv-secret")

    def test_pop_returns_error_for_unsupported_provider(self) -> None:
        agent_file = self._write_agent_file(provider="anthropic")
        stderr = io.StringIO()

        try:
            with (
                patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main(["pop", "-a", agent_file, "-p", "Bonjour"])
        finally:
            Path(agent_file).unlink(missing_ok=True)

        self.assertEqual(exit_code, 1)
        self.assertIn("Only `openai` is supported for now.", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
