from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ftry.Agent as agent_module
import ftry.TeamAgent as team_agent_module
from tests.src.testsupport import FakeAgent, SAMPLE_AGENT_FILE, make_fake_agent_framework_modules, reset_fakes


class TeamAgentTests(unittest.TestCase):
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
        agent = team_agent_module.TeamAgent.from_mapping(
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

                agent = team_agent_module.TeamAgent.from_file(agent_file)
                self.assertEqual(agent.name, "Test Agent")
                self.assertEqual(agent.description, "Helpful agent.")
                self.assertEqual(agent.instructions, "Do the work.")
                self.assertEqual(agent.model.api_key, "secret-key")

            with self.assertRaisesRegex(agent_module.FtryCliError, "Agent file not found"):
                team_agent_module.TeamAgent.from_file(temp_path / "missing.yaml")

        with patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
            repo_sample = team_agent_module.TeamAgent.from_file(SAMPLE_AGENT_FILE)
        self.assertEqual(repo_sample.name, "Poete")

    def test_create_participant_applies_team_context_and_name_override(self) -> None:
        reset_fakes()
        agent = team_agent_module.TeamAgent(
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
        self.assertNotIn("ConsoleInteractionContract", created_agent.instructions)

    def test_create_participant_forwards_history_persistence_flag(self) -> None:
        reset_fakes()
        agent = team_agent_module.TeamAgent(self._make_agent_config(name="Historian", instructions="Remember context."))

        with patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False):
            created_agent = agent.create_participant(require_per_service_call_history_persistence=True)

        self.assertIsInstance(created_agent, FakeAgent)
        self.assertTrue(created_agent.require_per_service_call_history_persistence)

    def test_create_managed_participant_enables_turn_control_contract_and_middleware(self) -> None:
        reset_fakes()
        agent = team_agent_module.TeamAgent(
            self._make_agent_config(name="Researcher", description="Asks clarifying questions.", instructions="Research.")
        )

        with patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False):
            created_agent = agent.create_managed_participant(name_override="Researcher-2")

        self.assertIsInstance(created_agent, FakeAgent)
        self.assertEqual(created_agent.name, "Researcher-2")
        self.assertIn("ConsoleInteractionContract", created_agent.instructions)
        self.assertEqual(
            created_agent.default_options.get("response_format"),
            team_agent_module.AGENT_TURN_RESPONSE_FORMAT,
        )
        self.assertEqual(len(created_agent.middleware), 1)
        self.assertEqual(len(created_agent.agent_middleware), 1)
        self.assertIsNone(created_agent._cached_agent_middleware_pipeline)

    def test_create_managed_participant_can_skip_strict_response_format(self) -> None:
        reset_fakes()
        agent = team_agent_module.TeamAgent(self._make_agent_config(name="Router", instructions="Route requests."))
        signal_state = team_agent_module.HandoffHilSignalState()

        with patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False):
            created_agent = agent.create_managed_participant(
                name_override="Router-2",
                enforce_structured_output=False,
                handoff_hil_signal_state=signal_state,
            )

        self.assertNotIn("ConsoleInteractionContract", created_agent.instructions)
        self.assertIn("HandoffInteractionContract", created_agent.instructions)
        self.assertEqual(
            {tool.name for tool in created_agent.default_options.get("tools", [])},
            {"request_user_input", "final_answer"},
        )
        self.assertEqual(created_agent.middleware, [])
        self.assertEqual(created_agent.agent_middleware, [])


if __name__ == "__main__":
    unittest.main()
