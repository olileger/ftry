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
from tests.src.testsupport import (
    FakeAgent,
    FakeOpenAIChatCompletionClient,
    SAMPLE_AGENT_FILE,
    make_fake_agent_framework_modules,
    reset_fakes,
    strip_ansi,
)


class AgentTests(unittest.TestCase):
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

    def test_agent_model_config_returns_none_when_model_is_optional(self) -> None:
        self.assertIsNone(agent_module.AgentModelConfig.from_mapping(None, config_kind="team", required=False))

    def test_agent_from_mapping_builds_an_inline_agent(self) -> None:
        agent = agent_module.Agent.from_mapping(
            {
                "name": "Inline Agent",
                "description": "Inline specialist.",
                "model": {
                    "name": "gpt-4o",
                    "provider": "openai",
                    "api-key": "secret",
                },
                "prompt": "Work inline.",
            },
            config_kind="team agent",
        )

        self.assertEqual(agent.name, "Inline Agent")
        self.assertEqual(agent.description, "Inline specialist.")
        self.assertEqual(agent.instructions, "Work inline.")
        self.assertEqual(agent.model.name, "gpt-4o")

    def test_agent_from_file_validates_nominal_and_error_cases(self) -> None:
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

                agent = agent_module.Agent.from_file(agent_file)
                self.assertEqual(agent.name, "Test Agent")
                self.assertEqual(agent.description, "Helpful agent.")
                self.assertEqual(agent.instructions, "Do the work.")
                self.assertEqual(agent.model.api_key, "secret-key")

            with self.assertRaisesRegex(agent_module.FtryCliError, "Agent file not found"):
                agent_module.Agent.from_file(temp_path / "missing.yaml")

        with patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
            repo_sample = agent_module.Agent.from_file(SAMPLE_AGENT_FILE)
        self.assertEqual(repo_sample.name, "Poete")

    def test_create_participant_applies_team_context_and_name_override(self) -> None:
        reset_fakes()
        agent = agent_module.Agent(
            self._make_agent_config(name="Runner", description="Executes prompts.", instructions="Run it.")
        )

        with patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False):
            created_agent = agent.create_participant(
                extra_instructions="Shared context",
                name_override="Runner-2",
            )

        self.assertIsInstance(created_agent, FakeAgent)
        self.assertEqual(created_agent.name, "Runner-2")
        self.assertIn("<TeamContext>", created_agent.instructions)
        self.assertIn("Shared context", created_agent.instructions)

    def test_run_returns_rendered_output_and_traces(self) -> None:
        reset_fakes()
        stderr = io.StringIO()
        agent = agent_module.Agent(self._make_agent_config(name="Poete", instructions="Tu es un poete."))

        with (
            patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False),
            redirect_stderr(stderr),
        ):
            output = asyncio.run(agent.run("Ecris un poeme sur la pluie"))

        self.assertEqual(output, "Poeme genere")
        self.assertEqual(FakeOpenAIChatCompletionClient.last_model, "gpt-4o")
        self.assertEqual(FakeOpenAIChatCompletionClient.last_api_key, "secret")
        self.assertEqual(FakeAgent.last_prompt, "Ecris un poeme sur la pluie")
        plain_stderr = strip_ansi(stderr.getvalue())
        self.assertIn("AGENT Poete | input:", plain_stderr)
        self.assertIn("AGENT Poete | final-output:", plain_stderr)

    def test_agent_from_file_loads_api_key_from_dotenv_file(self) -> None:
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
                agent = agent_module.Agent.from_file(agent_file)

        self.assertEqual(agent.model.api_key, "dotenv-secret")

    def test_run_rejects_unsupported_provider(self) -> None:
        agent_file = self._write_agent_file(provider="anthropic")

        try:
            with patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
                agent = agent_module.Agent.from_file(agent_file)
            with self.assertRaisesRegex(agent_module.FtryCliError, "Only `openai` is supported for now."):
                asyncio.run(agent.run("Bonjour"))
        finally:
            Path(agent_file).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
