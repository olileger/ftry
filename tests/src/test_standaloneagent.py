from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ftry.Agent as agent_module
import ftry.StandaloneAgent as standalone_agent_module
from tests.src.testsupport import (
    FakeAgent,
    FakeMCPStdioTool,
    FakeOpenAIChatCompletionClient,
    FakeResult,
    SAMPLE_AGENT_FILE,
    make_fake_agent_framework_modules,
    reset_fakes,
    strip_ansi,
)


class StandaloneAgentTests(unittest.TestCase):
    def _write_agent_file(self, provider: str = "openai") -> str:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False) as handle:
            handle.write(
                "\n".join(
                    [
                        "name: Poete",
                        "description: |",
                        "  Helpful agent.",
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
    ) -> agent_module.AgentConfig:
        return agent_module.AgentConfig(
            name=name,
            description=description,
            instructions=instructions,
            model=agent_module.AgentModelConfig(name="gpt-4o", provider=provider, api_key="secret"),
        )

    def test_from_mapping_builds_inline_agent(self) -> None:
        agent = standalone_agent_module.StandaloneAgent.from_mapping(
            {
                "name": "Inline Agent",
                "description": "Inline specialist.",
                "model": {
                    "name": "gpt-4o",
                    "provider": "openai",
                    "api-key": "secret",
                },
                "mcp": ["file-system"],
                "prompt": "Work inline.",
            },
            config_kind="team agent",
        )

        self.assertEqual(agent.name, "Inline Agent")
        self.assertEqual(agent.description, "Inline specialist.")
        self.assertEqual(agent.instructions, "Work inline.")
        self.assertEqual(agent.model.name, "gpt-4o")
        self.assertEqual(agent.config.mcp_servers, ("file-system",))

    def test_from_file_validates_nominal_and_error_cases(self) -> None:
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

                agent = standalone_agent_module.StandaloneAgent.from_file(agent_file)
                self.assertEqual(agent.name, "Test Agent")
                self.assertEqual(agent.description, "Helpful agent.")
                self.assertEqual(agent.instructions, "Do the work.")
                self.assertEqual(agent.model.api_key, "secret-key")

            with self.assertRaisesRegex(agent_module.FtryCliError, "Agent file not found"):
                standalone_agent_module.StandaloneAgent.from_file(temp_path / "missing.yaml")

        with patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
            repo_sample = standalone_agent_module.StandaloneAgent.from_file(SAMPLE_AGENT_FILE)
        self.assertEqual(repo_sample.name, "Poete")

    def test_create_participant_adds_structured_console_contract(self) -> None:
        reset_fakes()
        agent = standalone_agent_module.StandaloneAgent(
            self._make_agent_config(name="Runner", description="Executes prompts.", instructions="Run it.")
        )

        with patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False):
            created_agent = agent.create_participant(extra_instructions="Shared context")

        self.assertIsInstance(created_agent, FakeAgent)
        self.assertIn("<TeamContext>", created_agent.instructions)
        self.assertIn("ConsoleInteractionContract", created_agent.instructions)

    def test_run_returns_rendered_output_and_traces(self) -> None:
        reset_fakes()
        stderr = io.StringIO()
        agent = standalone_agent_module.StandaloneAgent(self._make_agent_config(name="Poete", instructions="Tu es un poete."))

        with (
            patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False),
            redirect_stderr(stderr),
        ):
            output = asyncio.run(agent.run("Ecris un poeme sur la pluie"))

        self.assertEqual(output, "Poeme genere")
        self.assertEqual(FakeOpenAIChatCompletionClient.last_model, "gpt-4o")
        self.assertEqual(FakeOpenAIChatCompletionClient.last_api_key, "secret")
        self.assertEqual(FakeAgent.last_prompt, "Ecris un poeme sur la pluie")
        self.assertEqual(len(FakeAgent.created_sessions), 1)
        self.assertEqual(FakeAgent.last_options, {"response_format": standalone_agent_module.AGENT_TURN_RESPONSE_FORMAT})
        plain_stderr = strip_ansi(stderr.getvalue())
        self.assertIn("AGENT Poete | input:", plain_stderr)
        self.assertIn("AGENT Poete | final-output:", plain_stderr)

    def test_run_attaches_and_closes_mcp_tools_from_current_registry(self) -> None:
        reset_fakes()
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_dir = temp_path / "mcp"
            registry_dir.mkdir()
            (registry_dir / "file-system.yaml").write_text(
                "\n".join(
                    [
                        'name: "file-system"',
                        'transport: "stdio"',
                        'command: "uvx"',
                        "args:",
                        '  - "mcp-server-filesystem"',
                    ]
                ),
                encoding="utf-8",
            )
            agent = standalone_agent_module.StandaloneAgent.from_mapping(
                {
                    "name": "Workspace Agent",
                    "model": {
                        "name": "gpt-4o",
                        "provider": "openai",
                        "api-key": "secret",
                    },
                    "mcp": ["file-system"],
                    "prompt": "Inspect files.",
                }
            )

            with (
                patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False),
                patch("ftry.Mcp.Path.cwd", return_value=temp_path),
            ):
                output = asyncio.run(agent.run("List the files"))

        self.assertEqual(output, "Poeme genere")
        self.assertIsNotNone(FakeOpenAIChatCompletionClient.last_agent)
        self.assertEqual(len(FakeOpenAIChatCompletionClient.last_agent.tools), 1)
        self.assertEqual(FakeOpenAIChatCompletionClient.last_agent.tools[0].name, "file-system")
        self.assertEqual(len(FakeMCPStdioTool.entered_tools), 1)
        self.assertEqual(len(FakeMCPStdioTool.closed_tools), 1)

    def test_run_loops_when_agent_awaits_user_input(self) -> None:
        reset_fakes()
        stderr = io.StringIO()
        agent = standalone_agent_module.StandaloneAgent(
            self._make_agent_config(name="Detective", instructions="Trouve la bonne personne.")
        )
        FakeAgent.queued_results = [
            FakeResult(
                "Structured agent turn",
                value={"status": "await_user_input", "message": "Pense a une personnalite et dis-moi quand tu es pret."},
            ),
            FakeResult(
                "Structured agent turn",
                value={"status": "done", "message": "Je pense que c'est Marie Curie."},
            ),
        ]
        prompts_seen: list[str] = []

        def user_input_provider(agent_message: str) -> str:
            prompts_seen.append(agent_message)
            return "Je suis pret."

        with (
            patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False),
            redirect_stderr(stderr),
        ):
            output = asyncio.run(agent.run("On joue a qui est-ce qui est ?", user_input_provider=user_input_provider))

        self.assertEqual(output, "Je pense que c'est Marie Curie.")
        self.assertEqual(prompts_seen, ["Pense a une personnalite et dis-moi quand tu es pret."])
        self.assertEqual(len(FakeAgent.created_sessions), 1)
        self.assertEqual(len(FakeAgent.run_calls), 2)
        self.assertEqual(FakeAgent.run_calls[0]["kwargs"]["session"], FakeAgent.run_calls[1]["kwargs"]["session"])
        self.assertEqual(FakeAgent.run_calls[1]["prompt"], "Je suis pret.")
        plain_stderr = strip_ansi(stderr.getvalue())
        self.assertIn("AGENT Detective | output [AWAIT USER INPUT]:", plain_stderr)
        self.assertIn("AGENT Detective | final-output:", plain_stderr)
        self.assertIn("AGENT Detective | input:\n\tJe suis pret.", plain_stderr)

    def test_run_fails_when_agent_awaits_user_input_without_provider(self) -> None:
        reset_fakes()
        agent = standalone_agent_module.StandaloneAgent(
            self._make_agent_config(name="Detective", instructions="Trouve la bonne personne.")
        )
        FakeAgent.queued_results = [
            FakeResult(
                "Structured agent turn",
                value={"status": "await_user_input", "message": "Pense a une personnalite et dis-moi quand tu es pret."},
            )
        ]

        with patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False):
            with self.assertRaisesRegex(agent_module.FtryCliError, "awaiting user input"):
                asyncio.run(agent.run("On joue a qui est-ce qui est ?"))

    def test_parse_turn_response_rejects_missing_structured_payload(self) -> None:
        with self.assertRaisesRegex(agent_module.FtryCliError, "missing the structured control payload"):
            standalone_agent_module.StandaloneAgent._parse_turn_response(FakeResult("Structured agent turn", value="not-a-mapping"))

    def test_parse_turn_response_rejects_invalid_status(self) -> None:
        with self.assertRaisesRegex(agent_module.FtryCliError, "invalid structured status"):
            standalone_agent_module.StandaloneAgent._parse_turn_response(
                FakeResult(
                    "Structured agent turn",
                    value={"status": "continue", "message": "Je continue."},
                )
            )

    def test_parse_turn_response_rejects_blank_message(self) -> None:
        with self.assertRaisesRegex(agent_module.FtryCliError, "missing a non-empty structured message"):
            standalone_agent_module.StandaloneAgent._parse_turn_response(
                FakeResult(
                    "Structured agent turn",
                    value={"status": "done", "message": "   "},
                )
            )

    def test_from_file_loads_api_key_from_dotenv_file(self) -> None:
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
                patch("ftry.Tools.Path.cwd", return_value=temp_path),
            ):
                agent = standalone_agent_module.StandaloneAgent.from_file(agent_file)

        self.assertEqual(agent.model.api_key, "dotenv-secret")

    def test_run_rejects_unsupported_provider(self) -> None:
        agent_file = self._write_agent_file(provider="anthropic")

        try:
            with patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
                agent = standalone_agent_module.StandaloneAgent.from_file(agent_file)
            with self.assertRaisesRegex(agent_module.FtryCliError, "Only `openai` is supported for now."):
                asyncio.run(agent.run("Bonjour"))
        finally:
            Path(agent_file).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
