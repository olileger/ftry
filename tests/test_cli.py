from __future__ import annotations

import asyncio
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

import ftry.Team as team_module
from ftry import cli
from ftry.Tools import _detect_yaml_config_kind


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "samples"


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

    def _make_agent_config(
        self,
        *,
        name: str = "Agent",
        description: str | None = "Helpful specialist.",
        instructions: str = "Do the work.",
        provider: str = "openai",
    ) -> cli.AgentConfig:
        return cli.AgentConfig(
            name=name,
            description=description,
            instructions=instructions,
            model=cli.AgentModelConfig(name="gpt-4o", provider=provider, api_key="secret"),
        )

    def _make_team_config(
        self,
        *agents: cli.AgentConfig,
        name: str = "Team",
        description: str | None = "Helpful team.",
        instructions: str = "Discuss and solve the request.",
        pattern: str | None = None,
        with_model: bool = False,
        max_turns: int | None = None,
    ) -> cli.TeamConfig:
        return cli.TeamConfig(
            name=name,
            description=description,
            instructions=instructions,
            agents=agents or (self._make_agent_config(),),
            model=cli.AgentModelConfig(name="gpt-4o", provider="openai", api_key="secret") if with_model else None,
            pattern=pattern,
            termination=cli.TeamTerminationConfig(max_turns=max_turns),
        )

    def _patch_agent_framework(self) -> tuple[dict[str, types.ModuleType], types.ModuleType, types.ModuleType]:
        fake_package = types.ModuleType("agent_framework")
        fake_openai_module = types.ModuleType("agent_framework.openai")
        fake_openai_module.OpenAIChatCompletionClient = FakeOpenAIChatCompletionClient
        fake_orchestrations_module = types.ModuleType("agent_framework.orchestrations")
        fake_orchestrations_module.SequentialBuilder = FakeSequentialBuilder
        fake_orchestrations_module.ConcurrentBuilder = FakeConcurrentBuilder
        fake_orchestrations_module.GroupChatBuilder = FakeGroupChatBuilder
        fake_orchestrations_module.HandoffBuilder = FakeHandoffBuilder
        fake_orchestrations_module.MagenticBuilder = FakeMagenticBuilder
        return (
            {
                "agent_framework": fake_package,
                "agent_framework.openai": fake_openai_module,
                "agent_framework.orchestrations": fake_orchestrations_module,
            },
            fake_openai_module,
            fake_orchestrations_module,
        )

    def _reset_fakes(self) -> None:
        FakeAgent.last_prompt = None
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

    def test_load_line_banner_and_mock_commands_render_expected_output(self) -> None:
        banner = cli._load_line_banner()
        self.assertNotIn("[pink]", banner)
        self.assertIn(cli.RESET, banner)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            mock_exit_code = cli._run_mock_command("build")
            line_exit_code = cli._run_line_command()

        self.assertEqual(mock_exit_code, 0)
        self.assertEqual(line_exit_code, 0)
        rendered = stdout.getvalue()
        self.assertIn("build", rendered)
        self.assertIn("______", _strip_ansi(rendered))

    def test_validation_helpers_handle_edge_cases(self) -> None:
        self.assertEqual(cli._require_non_empty_string("  value  ", "name"), "value")
        self.assertIsNone(cli._require_optional_string(None, "description"))
        self.assertEqual(cli._require_optional_string("  note  ", "description"), "note")
        self.assertEqual(cli._require_mapping({"ok": True}, "root"), {"ok": True})
        self.assertEqual(cli._require_sequence(["a"], "agents"), ["a"])
        self.assertEqual(cli._require_positive_int(3, "termination.max-turns", "team"), 3)

        with self.assertRaisesRegex(cli.FtryCliError, "Invalid or missing `name`"):
            cli._require_non_empty_string("   ", "name")
        with self.assertRaisesRegex(cli.FtryCliError, "Invalid or missing `description`"):
            cli._require_optional_string("", "description")
        with self.assertRaisesRegex(cli.FtryCliError, "Invalid or missing `root` mapping"):
            cli._require_mapping([], "root")
        with self.assertRaisesRegex(cli.FtryCliError, "Invalid or missing `agents` list"):
            cli._require_sequence({}, "agents")
        with self.assertRaisesRegex(cli.FtryCliError, "expected a positive integer"):
            cli._require_positive_int(0, "termination.max-turns", "team")

    def test_find_dotenv_path_and_loader_use_expected_sources(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cwd_dir = temp_path / "cwd"
            cwd_dir.mkdir()
            config_dir = temp_path / "configs" / "nested"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "agent.yaml"
            config_path.write_text("name: Agent\n", encoding="utf-8")

            cwd_dotenv = cwd_dir / ".env"
            cwd_dotenv.write_text("FROM=cwd\n", encoding="utf-8")
            parent_dotenv = temp_path / "configs" / ".env"
            parent_dotenv.write_text("FROM=parent\n", encoding="utf-8")

            with patch("ftry.cli.Path.cwd", return_value=cwd_dir):
                self.assertEqual(cli._find_dotenv_path(config_path), cwd_dotenv)

            cwd_dotenv.unlink()

            with patch("ftry.cli.Path.cwd", return_value=cwd_dir):
                self.assertEqual(cli._find_dotenv_path(config_path), parent_dotenv)

            loaded_calls: list[tuple[Path, bool]] = []

            def fake_load_dotenv(*, dotenv_path: Path, override: bool) -> bool:
                loaded_calls.append((dotenv_path, override))
                return True

            with (
                patch("ftry.cli.Path.cwd", return_value=cwd_dir),
                patch("ftry.cli._load_dotenv_function", return_value=fake_load_dotenv),
            ):
                cli._load_dotenv_for_config(config_path)

            self.assertEqual(loaded_calls, [(parent_dotenv, False)])

            parent_dotenv.unlink()
            with patch("ftry.cli.Path.cwd", return_value=cwd_dir):
                self.assertIsNone(cli._find_dotenv_path(config_path))

    def test_secret_and_team_pattern_helpers_cover_aliases_and_errors(self) -> None:
        self.assertEqual(cli._resolve_secret("literal-secret"), "literal-secret")

        with patch.dict(os.environ, {"OAI_API_KEY": "env-secret"}, clear=False):
            self.assertEqual(cli._resolve_secret("env:OAI_API_KEY"), "env-secret")

        with self.assertRaisesRegex(cli.FtryCliError, "environment variable name is missing"):
            cli._resolve_secret("env:   ")
        with self.assertRaisesRegex(cli.FtryCliError, "Environment variable `UNSET_KEY` is not set."):
            cli._resolve_secret("env:UNSET_KEY")

        self.assertEqual(cli._normalize_team_pattern("Group_Chat"), "group-chat")
        self.assertEqual(cli._normalize_team_pattern("magentic-one"), "magentic")
        with self.assertRaisesRegex(cli.FtryCliError, "Unsupported team pattern `swarm`"):
            cli._normalize_team_pattern("swarm")

        self.assertEqual(cli._parse_team_termination(None), cli.TeamTerminationConfig())
        self.assertEqual(cli._parse_team_termination({"max-turns": 4}), cli.TeamTerminationConfig(max_turns=4))
        with self.assertRaisesRegex(cli.FtryCliError, "expected a positive integer"):
            cli._parse_team_termination({"max-turns": -1})

    def test_message_and_output_helpers_render_expected_text(self) -> None:
        text_message = types.SimpleNamespace(text="  direct answer  ")
        content_message = types.SimpleNamespace(contents=[types.SimpleNamespace(text="Hello"), " world"])
        empty_message = types.SimpleNamespace()

        self.assertEqual(cli._extract_message_text(text_message), "direct answer")
        self.assertEqual(cli._extract_message_text(content_message), "Hello world")
        self.assertEqual(cli._extract_message_text(empty_message), "")

        messages = [types.SimpleNamespace(role="assistant", text="Hi", author_name="agent-1")]
        payload = types.SimpleNamespace(messages=messages)
        author_name_map = {"agent-1": "Prompter", "team-1": "Better Prompt team"}

        self.assertEqual(cli._extract_messages(messages), messages)
        self.assertEqual(cli._extract_messages(payload), messages)
        self.assertEqual(cli._extract_messages(object()), [])

        self.assertEqual(cli._display_name(None), "unknown")
        self.assertEqual(cli._display_name("agent-1", author_name_map), "Prompter")
        self.assertEqual(cli._display_name("missing", author_name_map), "missing")

        summarized = cli._summarize_payload(
            types.SimpleNamespace(
                messages=[
                    types.SimpleNamespace(role="user", text="ignore me", author_name="user"),
                    types.SimpleNamespace(role="assistant", text="Draft prompt", author_name="agent-1"),
                ]
            ),
            author_name_map=author_name_map,
        )
        self.assertEqual(summarized, "[Prompter] Draft prompt")

        chunk = cli._extract_trace_chunk(
            types.SimpleNamespace(
                messages=[
                    types.SimpleNamespace(role="assistant", text="One", author_name="agent-1"),
                    types.SimpleNamespace(role="assistant", contents=["Two"], author_name="agent-1"),
                ]
            )
        )
        self.assertEqual(chunk, "One\n\nTwo")

        formatted_agent = cli._format_agent_output(
            types.SimpleNamespace(
                messages=[
                    types.SimpleNamespace(role="assistant", text="Done", author_name="agent-1"),
                    types.SimpleNamespace(role="user", text="ignored"),
                ]
            ),
            author_name_map=author_name_map,
        )
        self.assertEqual(formatted_agent, "[Prompter]\nDone")

        final_output = cli._format_final_team_output(
            types.SimpleNamespace(
                messages=[
                    types.SimpleNamespace(role="assistant", text="Draft", author_name="agent-1"),
                    types.SimpleNamespace(role="assistant", text="Final", author_name="team-1"),
                ]
            ),
            author_name_map=author_name_map,
        )
        self.assertEqual(final_output, "[Better Prompt team]\nFinal")
        self.assertEqual(cli._format_final_team_output(types.SimpleNamespace(text="fallback")), "fallback")

    def test_summarize_trace_text_truncates_and_normalizes_whitespace(self) -> None:
        long_text = "Line 1\r\n\r\n\r\n" + ("x" * 260)
        summarized = cli._summarize_trace_text(long_text, max_length=40)
        self.assertEqual(summarized[:6], "Line 1")
        self.assertTrue(summarized.endswith("..."))
        self.assertLessEqual(len(summarized), 40)

    def test_render_team_instructions_and_analysis_helpers_include_roles(self) -> None:
        agent = self._make_agent_config(
            name="Prompter",
            description=" Builds prompts.\nWith care. ",
            instructions="Build a prompt.",
        )
        reviewer = self._make_agent_config(name="Reviewer", description=None, instructions="Review drafts.")
        team = self._make_team_config(
            agent,
            reviewer,
            name="Workshop",
            description="Discuss together.",
            instructions="Use {participants}.\n{roles}\n1. Draft.\n2. Review.",
        )

        self.assertEqual(cli._render_role_summary(agent), "Builds prompts. With care.")
        rendered = cli._render_team_instructions(team)
        self.assertIn("Prompter, Reviewer", rendered)
        self.assertIn("- Reviewer: Review drafts.", rendered)

        analysis_text = cli._compose_pattern_analysis_text(team)
        self.assertIn("workshop", analysis_text)
        self.assertTrue(cli._contains_any(analysis_text, ("review",)))
        self.assertTrue(cli._has_numbered_steps(team.instructions))

    def test_team_config_helpers_cover_string_file_refs_missing_defaults_and_fallbacks(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            agent_file = temp_path / "agent.yaml"
            agent_file.write_text(
                "\n".join(
                    [
                        "name: String Agent",
                        "model:",
                        "  name: gpt-4o",
                        "  provider: openai",
                        "  api-key: env:OAI_API_KEY",
                        "prompt: |",
                        "  Work from a string file reference.",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
                loaded_agent = team_module._load_team_agent_config(f".\\{agent_file.name}", team_dir=temp_path)

            self.assertEqual(loaded_agent.name, "String Agent")
            self.assertEqual(team_module._parse_team_termination({}), cli.TeamTerminationConfig())
            with self.assertRaisesRegex(cli.FtryCliError, "Team file not found"):
                team_module._load_team_config(temp_path / "missing-team.yaml")

        neutral_agent = self._make_agent_config(
            name="Helper",
            description="Answers the request.",
            instructions="Provide a useful answer.",
        )
        neutral_team = self._make_team_config(
            neutral_agent,
            name="Generalists",
            description=None,
            instructions="Help with the request.",
        )
        self.assertEqual(team_module._infer_team_pattern(neutral_team), "group-chat")

        first_agent = FakeAgent(name="Alpha", instructions="Handle the task.", description="First specialist.")
        second_agent = FakeAgent(name="Beta", instructions="Continue the work.", description="Second specialist.")
        self.assertEqual(team_module._select_handoff_start_agent([first_agent, second_agent]), first_agent)

    def test_load_agent_and_team_config_validate_nominal_and_error_cases(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
                agent_file = temp_path / "agent.yaml"
                agent_file.write_text(
                    "\n".join(
                        [
                            "name: Test Agent",
                            "description: |",
                            "  Helpful agent.",
                            "model:",
                            "  name: gpt-4o",
                            "  provider: openai",
                            "  api-key: env:OAI_API_KEY",
                            "prompt: |",
                            "  Do the work.",
                        ]
                    ),
                    encoding="utf-8",
                )

                config = cli._load_agent_config(agent_file)
                self.assertEqual(config.name, "Test Agent")
                self.assertEqual(config.description, "Helpful agent.")

                team_file = temp_path / "team.yaml"
                team_file.write_text(
                    "\n".join(
                        [
                            "name: Explicit Team",
                            "pattern: group_chat",
                            "agents:",
                            f"  - file: .\\{agent_file.name}",
                            "prompt: |",
                            "  Coordinate the work.",
                        ]
                    ),
                    encoding="utf-8",
                )

                team = cli._load_team_config(team_file)
                self.assertEqual(team.pattern, "group-chat")
                self.assertEqual(team.agents[0].name, "Test Agent")

                bad_team_file = temp_path / "bad-team.yaml"
                bad_team_file.write_text(
                    "\n".join(
                        [
                            "name: Broken Team",
                            "agents:",
                            f"  - file: .\\{agent_file.name}",
                            "    name: Invalid mix",
                            "prompt: |",
                            "  Broken.",
                        ]
                    ),
                    encoding="utf-8",
                )

                empty_agents_file = temp_path / "empty-team.yaml"
                empty_agents_file.write_text(
                    "\n".join(
                        [
                            "name: Empty Team",
                            "agents: []",
                            "prompt: |",
                            "  Nothing to do.",
                        ]
                    ),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(cli.FtryCliError, "Agent file not found"):
                    cli._load_agent_config(temp_path / "missing.yaml")
                with self.assertRaisesRegex(cli.FtryCliError, "`file` references cannot be mixed with inline fields"):
                    cli._load_team_config(bad_team_file)
                with self.assertRaisesRegex(cli.FtryCliError, "Invalid or missing `agents` list"):
                    cli._load_team_config(empty_agents_file)

    def test_create_openai_agent_and_controller_agent_apply_team_context(self) -> None:
        self._reset_fakes()
        module_patch, _, _ = self._patch_agent_framework()
        agent_config = self._make_agent_config(name="Runner", description="Executes prompts.", instructions="Run it.")
        team = self._make_team_config(
            agent_config,
            name="Better Prompt team",
            instructions="Use {participants}.\n{roles}",
            with_model=True,
        )

        with patch.dict(sys.modules, module_patch, clear=False):
            created_agent = cli._create_openai_agent(
                agent_config,
                extra_instructions="Shared context",
                name_override="Runner-2",
            )
            self.assertIsInstance(created_agent, FakeAgent)
            self.assertEqual(created_agent.name, "Runner-2")
            self.assertIn("<TeamContext>", created_agent.instructions)
            self.assertIn("Shared context", created_agent.instructions)

            controller = cli._create_team_controller_agent(team, instructions=cli._render_team_instructions(team))
            self.assertIsInstance(controller, FakeAgent)
            self.assertEqual(controller.name, "Better-Prompt-team")
            self.assertEqual(cli._create_team_controller_agent(self._make_team_config(agent_config), instructions="ctx"), None)

    def test_build_team_participants_and_workflows_cover_pattern_specific_logic(self) -> None:
        self._reset_fakes()
        module_patch, _, _ = self._patch_agent_framework()
        duplicate_a = self._make_agent_config(name="Agent", instructions="Do A.")
        duplicate_b = self._make_agent_config(name="Agent", instructions="Do B.")
        router = self._make_agent_config(name="Router", description="Route the request.", instructions="Triage the task.")
        specialist = self._make_agent_config(name="Specialist", instructions="Answer the request.")

        with patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False), patch.dict(sys.modules, module_patch, clear=False):
            participants, author_name_map = cli._build_team_participants(
                self._make_team_config(duplicate_a, duplicate_b, router, instructions="Use {participants}."),
                extra_instructions="Shared context",
            )
            self.assertEqual([participant.name for participant in participants], ["Agent", "Agent-2", "Router"])
            self.assertEqual(author_name_map["Agent-2"], "Agent")
            self.assertIn("<TeamContext>", participants[0].instructions)
            self.assertEqual(cli._select_handoff_start_agent(participants).name, "Router")
            self.assertEqual(
                cli._count_assistant_messages(
                    [
                        types.SimpleNamespace(role="assistant"),
                        types.SimpleNamespace(role="user"),
                        types.SimpleNamespace(role="assistant"),
                    ]
                ),
                2,
            )

            cli._build_team_workflow(
                self._make_team_config(duplicate_a, specialist, pattern="sequential", name="Pipeline", instructions="First draft, then refine.")
            )
            self.assertTrue(FakeSequentialBuilder.last_kwargs["intermediate_outputs"])

            cli._build_team_workflow(
                self._make_team_config(duplicate_a, specialist, pattern="concurrent", name="Swarm", instructions="Work in parallel.")
            )
            self.assertTrue(FakeConcurrentBuilder.last_kwargs["intermediate_outputs"])

            cli._build_team_workflow(
                self._make_team_config(
                    router,
                    specialist,
                    pattern="handoff",
                    name="Triage",
                    instructions="Route and handoff.",
                    max_turns=3,
                )
            )
            self.assertEqual(FakeHandoffBuilder.last_start_agent.name, "Router")
            self.assertEqual(FakeHandoffBuilder.last_autonomous_kwargs["turn_limits"]["Router"], 3)
            self.assertTrue(callable(FakeHandoffBuilder.last_termination_condition))

            cli._build_team_workflow(
                self._make_team_config(
                    duplicate_a,
                    specialist,
                    pattern="group-chat",
                    name="Workshop",
                    instructions="Discuss together.",
                    with_model=True,
                    max_turns=5,
                )
            )
            self.assertEqual(FakeGroupChatBuilder.last_kwargs["max_rounds"], 5)
            self.assertIsInstance(FakeGroupChatBuilder.last_kwargs["orchestrator_agent"], FakeAgent)

            cli._build_team_workflow(
                self._make_team_config(
                    duplicate_a,
                    specialist,
                    pattern="magentic",
                    name="Planner",
                    instructions="Plan and replan a complex task.",
                    with_model=True,
                    max_turns=4,
                )
            )
            self.assertEqual(FakeMagenticBuilder.last_kwargs["max_round_count"], 4)
            self.assertIsInstance(FakeMagenticBuilder.last_kwargs["manager_agent"], FakeAgent)

    def test_run_team_prompt_handles_handoff_event_stream(self) -> None:
        self._reset_fakes()
        module_patch, _, _ = self._patch_agent_framework()
        stderr = io.StringIO()
        team = self._make_team_config(
            self._make_agent_config(name="Router", description="Route the request.", instructions="Triage the task."),
            self._make_agent_config(name="Specialist", description="Handles the final answer.", instructions="Solve the task."),
            name="Handoff squad",
            instructions="Route the request and handoff to specialists.",
            pattern="handoff",
            max_turns=2,
        )

        with (
            patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
            patch.dict(sys.modules, module_patch, clear=False),
            redirect_stderr(stderr),
        ):
            output = asyncio.run(cli._run_team_prompt(team, "Route this request"))

        self.assertEqual(output, "[Specialist]\nhandoff:Route this request")
        plain_stderr = _strip_ansi(stderr.getvalue())
        self.assertIn("TEAM Handoff squad | pattern: handoff | input:", plain_stderr)
        self.assertIn("Router --> Specialist | input:", plain_stderr)
        self.assertIn("TEAM Handoff squad <-- Specialist | final-output:", plain_stderr)

    def test_run_team_prompt_prefers_last_agent_output_when_team_authors_final_message(self) -> None:
        self._reset_fakes()
        module_patch, _, _ = self._patch_agent_framework()
        stderr = io.StringIO()
        team = self._make_team_config(
            self._make_agent_config(name="Prompter", instructions="Draft the prompt."),
            self._make_agent_config(name="Reviewer", instructions="Review the draft."),
            name="Better Prompt team",
            instructions="Discuss together and choose the best prompt.",
            pattern="group-chat",
            with_model=True,
        )

        with (
            patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
            patch.dict(sys.modules, module_patch, clear=False),
            redirect_stderr(stderr),
        ):
            output = asyncio.run(cli._run_team_prompt(team, "Ameliore ce prompt"))

        self.assertEqual(output, "[Better Prompt team]\ngroup-chat:Ameliore ce prompt")
        plain_stderr = _strip_ansi(stderr.getvalue())
        final_output_log = plain_stderr.split("TEAM Better Prompt team <-- Reviewer | final-output:", maxsplit=1)[1]
        self.assertIn("Review feedback", final_output_log)
        self.assertNotIn("[Better Prompt team]\n\tgroup-chat:Ameliore ce prompt", final_output_log)

    def test_run_team_prompt_logs_full_final_output_without_truncation(self) -> None:
        self._reset_fakes()
        module_patch, _, _ = self._patch_agent_framework()
        stderr = io.StringIO()
        long_prompt = "x" * 260
        team = self._make_team_config(
            self._make_agent_config(name="Researcher", instructions="Gather the facts."),
            self._make_agent_config(name="Writer", instructions="Write the final answer."),
            name="Pipeline team",
            instructions="First gather the facts, then draft the answer, and finally produce the final response.",
            pattern="sequential",
        )

        with (
            patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
            patch.dict(sys.modules, module_patch, clear=False),
            redirect_stderr(stderr),
        ):
            output = asyncio.run(cli._run_team_prompt(team, long_prompt))

        self.assertEqual(output, f"[Writer]\nsequential:{long_prompt}")
        plain_stderr = _strip_ansi(stderr.getvalue())
        final_output_log = plain_stderr.split("TEAM Pipeline team <-- Writer | final-output:", maxsplit=1)[1]
        self.assertIn(f"[Writer]\n\tsequential:{long_prompt}", final_output_log)
        self.assertNotIn(f"sequential:{long_prompt[:237]}...", final_output_log)

    def test_team_trace_state_final_output_covers_no_visible_message_fallbacks(self) -> None:
        state = team_module._TeamTraceState(
            pattern="sequential",
            team_name="Pipeline team",
            agent_trace_colors={},
            last_visible_input="Prompt",
        )
        state.last_agent_name = "Writer"
        state.last_agent_full_output = "Full writer output"

        with (
            patch("ftry.Team._collect_visible_messages", return_value=[]),
            patch("ftry.Team._trace_result") as trace_result,
        ):
            state.trace_final_output(object(), "Rendered output", {})

        trace_result.assert_called_once_with(
            "Pipeline team",
            "Writer",
            "Full writer output",
            team_name="Pipeline team",
            agent_trace_colors={},
            field_name="final-output",
        )

        fallback_state = team_module._TeamTraceState(
            pattern="sequential",
            team_name="Pipeline team",
            agent_trace_colors={},
            last_visible_input="Prompt",
        )
        with (
            patch("ftry.Team._collect_visible_messages", return_value=[]),
            patch("ftry.Team._trace_team_label", return_value="TEAM Pipeline team"),
            patch("ftry.Team._trace_block", return_value="\n\tRendered output"),
            patch("ftry.Team._trace") as trace_message,
        ):
            fallback_state.trace_final_output(object(), "Rendered output", {})

        trace_message.assert_called_once_with(
            "%s | final-output:%s",
            "TEAM Pipeline team",
            "\n\tRendered output",
        )

    def test_team_event_helpers_cover_guard_paths_and_executor_transitions(self) -> None:
        request_driven_state = team_module._TeamTraceState(
            pattern="group-chat",
            team_name="Workshop",
            agent_trace_colors={},
            last_visible_input="Prompt",
        )

        team_module._handle_executor_invoked_event(types.SimpleNamespace(executor_id=None), request_driven_state, {})
        self.assertIsNone(request_driven_state.active_executor)

        request_driven_state.expected_invoked_executor = "Reviewer"
        with patch.object(request_driven_state, "flush_buffer") as flush_buffer:
            team_module._handle_executor_invoked_event(
                types.SimpleNamespace(executor_id="Prompter"),
                request_driven_state,
                {},
            )
        flush_buffer.assert_not_called()
        self.assertEqual(request_driven_state.expected_invoked_executor, "Reviewer")

        direct_route_state = team_module._TeamTraceState(
            pattern="concurrent",
            team_name="Swarm",
            agent_trace_colors={},
            last_visible_input="Prompt",
        )
        with (
            patch.object(direct_route_state, "flush_buffer", wraps=direct_route_state.flush_buffer) as flush_buffer,
            patch.object(direct_route_state, "trace_route") as trace_route,
        ):
            team_module._handle_executor_invoked_event(
                types.SimpleNamespace(executor_id="Worker"),
                direct_route_state,
                {},
            )

        flush_buffer.assert_called_once_with(next_executor="Worker")
        trace_route.assert_called_once_with("Swarm", "Worker")
        self.assertEqual(direct_route_state.last_route_source, "Swarm")

        output_state = team_module._TeamTraceState(
            pattern="sequential",
            team_name="Pipeline",
            agent_trace_colors={},
            last_visible_input="Prompt",
        )
        empty_output_event = types.SimpleNamespace(data=object(), executor_id="Researcher")
        with patch("ftry.Team._summarize_payload", return_value=""):
            team_module._handle_output_event(empty_output_event, output_state, {})
        self.assertIsNone(output_state.active_executor)

        first_output_event = types.SimpleNamespace(data=object(), executor_id="Researcher")
        with (
            patch("ftry.Team._summarize_payload", return_value="Draft"),
            patch("ftry.Team._extract_trace_chunk", return_value="Chunk A"),
        ):
            team_module._handle_output_event(first_output_event, output_state, {})
        self.assertEqual(output_state.active_executor, "Researcher")
        self.assertEqual(output_state.buffered_outputs, ["Chunk A"])

        second_output_event = types.SimpleNamespace(data=object(), executor_id="Writer")
        with (
            patch("ftry.Team._summarize_payload", return_value="Review"),
            patch("ftry.Team._extract_trace_chunk", return_value="Chunk B"),
            patch.object(output_state, "flush_buffer", wraps=output_state.flush_buffer) as flush_buffer,
        ):
            team_module._handle_output_event(second_output_event, output_state, {})
        flush_buffer.assert_called_once_with(next_executor="Writer")
        self.assertEqual(output_state.active_executor, "Writer")
        self.assertEqual(output_state.buffered_outputs, ["Chunk B"])

    def test_main_reports_errors_and_direct_pop_requires_a_source(self) -> None:
        with self.assertRaisesRegex(cli.FtryCliError, "Either `-a/--agent-file` or `-t/--team-file` must be provided."):
            cli._run_pop_command(None, None, "Bonjour")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = cli.main(["pop", "-a", "missing.yaml", "-p", "Bonjour"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: Agent file not found:", stderr.getvalue())

    def test_build_parser_parses_pop_arguments(self) -> None:
        args = cli.build_parser().parse_args(["pop", "-a", r"samples\agents\poete.yaml", "-p", "Bonjour"])
        self.assertEqual(args.command, "pop")
        self.assertEqual(args.agent_file, r"samples\agents\poete.yaml")
        self.assertIsNone(args.team_file)
        self.assertEqual(args.prompt, "Bonjour")

    def test_build_parser_parses_pop_team_arguments(self) -> None:
        args = cli.build_parser().parse_args(["pop", "-t", r"samples\teams\better-prompt\team.yaml", "-p", "Bonjour"])
        self.assertEqual(args.command, "pop")
        self.assertEqual(args.team_file, r"samples\teams\better-prompt\team.yaml")
        self.assertIsNone(args.agent_file)
        self.assertEqual(args.prompt, "Bonjour")

    def test_detect_yaml_config_kind_returns_none_for_unknown_shape(self) -> None:
        self.assertIsNone(_detect_yaml_config_kind({"name": "Config", "prompt": "Bonjour"}))

    def test_pop_rejects_team_file_passed_as_agent_file(self) -> None:
        stderr = io.StringIO()
        team_file = str(SAMPLES_DIR / "team.yaml")

        with redirect_stderr(stderr):
            exit_code = cli.main(["pop", "-a", team_file, "-p", "Bonjour"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Invalid agent YAML", stderr.getvalue())
        self.assertIn("defines `agents` at the root", stderr.getvalue())
        self.assertIn("Use `-t/--team-file` instead of `-a/--agent-file`", stderr.getvalue())

    def test_pop_rejects_agent_file_passed_as_team_file(self) -> None:
        stderr = io.StringIO()
        agent_file = str(SAMPLES_DIR / "poete.yaml")

        with redirect_stderr(stderr):
            exit_code = cli.main(["pop", "-t", agent_file, "-p", "Bonjour"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Invalid team YAML", stderr.getvalue())
        self.assertIn("matches an agent configuration (`name`, `model`, `prompt`)", stderr.getvalue())
        self.assertIn("does not define `agents` at the root", stderr.getvalue())
        self.assertIn("Use `-a/--agent-file` instead of `-t/--team-file`", stderr.getvalue())

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

    def test_render_pop_animation_only_runs_in_a_tty(self) -> None:
        non_tty_stream = io.StringIO()
        cli._render_pop_animation(stream=non_tty_stream, sleep=lambda _: self.fail("sleep should not be called"))
        self.assertEqual(non_tty_stream.getvalue(), "")

        tty_stream = FakeTtyStream()
        delays: list[float] = []

        cli._render_pop_animation(stream=tty_stream, sleep=delays.append)

        rendered = tty_stream.getvalue()
        plain_rendered = _strip_ansi(rendered)
        self.assertTrue(rendered.startswith(cli.POP_ANIMATION_CURSOR_HIDE))
        self.assertTrue(rendered.endswith(f"\n{cli.POP_ANIMATION_CURSOR_SHOW}"))
        self.assertIn(cli.BRIGHT_PINK, rendered)
        self.assertIn(_strip_ansi(cli._load_pop_banner()).splitlines()[0], plain_rendered)
        self.assertIn("         .  .", plain_rendered)
        self.assertIn(r"         \______/>", plain_rendered)
        self.assertIn("          o    o", plain_rendered)
        self.assertIn("_" * 40, plain_rendered)
        self.assertIn(cli.POP_ANIMATION_CLEAR_LINE, rendered)
        self.assertNotIn(f"{cli.POP_ANIMATION_CLEAR_LINE}{cli.POP_ANIMATION_CURSOR_SHOW}", rendered)
        self.assertEqual(delays, [cli.POP_ANIMATION_STEP_SECONDS] * len(cli.POP_ANIMATION_FRAMES))

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
        self.assertEqual(stdout.getvalue().strip(), "")
        self.assertEqual(FakeOpenAIChatCompletionClient.last_model, "gpt-4o-2024-08-06")
        self.assertEqual(FakeOpenAIChatCompletionClient.last_api_key, "secret-key")
        self.assertIsNotNone(FakeOpenAIChatCompletionClient.last_agent)
        self.assertEqual(FakeOpenAIChatCompletionClient.last_agent.name, "Poete")
        self.assertEqual(FakeOpenAIChatCompletionClient.last_agent.instructions, "Tu es un poete.")
        self.assertEqual(FakeAgent.last_prompt, "Ecris un poeme sur la pluie")
        plain_stderr = _strip_ansi(stderr.getvalue())
        self.assertIn("AGENT Poete | input:", plain_stderr)
        self.assertIn("AGENT Poete | final-output:", plain_stderr)
        self.assertIn("Poeme genere", plain_stderr)

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
        self.assertEqual(stdout.getvalue().strip(), "")
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
                    exit_code = cli.main(["pop", "-t", r".\samples\teams\better-prompt\team.yaml", "-p", "Ameliore ce prompt"])
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "")
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
        self.assertEqual(stdout.getvalue().strip(), "")
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
        stderr = io.StringIO()

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
                redirect_stderr(stderr),
            ):
                exit_code = cli.main(["pop", "-a", str(agent_file), "-p", "Ecris un poeme sur la pluie"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "")
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
