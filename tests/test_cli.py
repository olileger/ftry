from __future__ import annotations

import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ftry import cli


class FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeAgent:
    last_prompt: str | None = None

    def __init__(self, name: str, instructions: str) -> None:
        self.name = name
        self.instructions = instructions

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

    def as_agent(self, *, name: str, instructions: str) -> FakeAgent:
        agent = FakeAgent(name=name, instructions=instructions)
        FakeOpenAIChatCompletionClient.last_agent = agent
        return agent


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
        self.assertEqual(args.prompt, "Bonjour")

    def test_pop_runs_agent_loaded_from_yaml(self) -> None:
        agent_file = self._write_agent_file()
        fake_package = types.ModuleType("agent_framework")
        fake_openai_module = types.ModuleType("agent_framework.openai")
        fake_openai_module.OpenAIChatCompletionClient = FakeOpenAIChatCompletionClient
        stdout = io.StringIO()

        try:
            with (
                patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
                patch.dict(
                    sys.modules,
                    {"agent_framework": fake_package, "agent_framework.openai": fake_openai_module},
                    clear=False,
                ),
                redirect_stdout(stdout),
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
