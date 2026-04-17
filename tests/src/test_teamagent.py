from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ftry.Agent as agent_module
import ftry.TeamAgent as team_agent_module
from tests.src.testsupport import (
    FakeAgent,
    FakeAgentResponse,
    FakeContent,
    FakeMessage,
    FakeResponseStream,
    FakeResult,
    SAMPLE_AGENT_FILE,
    make_fake_agent_framework_modules,
    reset_fakes,
)


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

    def test_configure_team_managed_participant_reuses_existing_agent_middleware_entries(self) -> None:
        reset_fakes()
        agent = FakeAgent(name="Researcher", instructions="Research.")

        with patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False):
            team_agent_module._configure_team_managed_participant(
                agent,
                enforce_structured_output=True,
                handoff_hil_signal_state=None,
            )
            first_middleware = list(agent.agent_middleware)
            team_agent_module._configure_team_managed_participant(
                agent,
                enforce_structured_output=True,
                handoff_hil_signal_state=None,
            )

        self.assertEqual(len(first_middleware), 1)
        self.assertEqual(len(agent.middleware), 2)
        self.assertEqual(len(agent.agent_middleware), 2)
        self.assertIs(agent.agent_middleware[0], first_middleware[0])

    def test_create_team_turn_control_middleware_transforms_streaming_responses(self) -> None:
        def fake_agent_middleware(func):
            func._middleware_type = "agent"
            return func

        middleware = team_agent_module._create_team_turn_control_middleware(
            FakeAgentResponse,
            FakeContent,
            FakeMessage,
            FakeResponseStream,
            fake_agent_middleware,
        )
        context = types.SimpleNamespace(
            stream=True,
            result=None,
            agent=types.SimpleNamespace(name="Researcher"),
        )

        async def call_next() -> None:
            context.result = FakeResult(
                "Structured response",
                value={"status": "await_user_input", "message": "Need a date"},
            )

        asyncio.run(middleware(context, call_next))

        self.assertTrue(context.stream)
        self.assertIsInstance(context.result, FakeResponseStream)
        transformed_response = context.result.finalizer(None)
        self.assertIsInstance(transformed_response, FakeAgentResponse)
        self.assertTrue(transformed_response.messages[0].contents[0].user_input_request)
        self.assertEqual(transformed_response.messages[0].author_name, "Researcher")

    def test_transform_team_agent_response_returns_original_response_for_tool_payload(self) -> None:
        response = types.SimpleNamespace(
            messages=[
                FakeMessage(
                    "assistant",
                    [FakeContent(type="function_call", text="handoff_to_specialist")],
                )
            ]
        )

        transformed_response = team_agent_module._transform_team_agent_response(
            response,
            agent_response_type=FakeAgentResponse,
            content_type=FakeContent,
            message_type=FakeMessage,
            author_name="Router",
        )

        self.assertIs(transformed_response, response)
        self.assertTrue(team_agent_module._response_contains_tool_payload(response))
        self.assertFalse(
            team_agent_module._response_contains_tool_payload(
                types.SimpleNamespace(messages=[FakeMessage("assistant", [FakeContent.from_text("hello")])])
            )
        )

    def test_attach_handoff_hil_tools_records_requests_and_final_answers(self) -> None:
        signal_state = team_agent_module.HandoffHilSignalState()
        agent = types.SimpleNamespace(name="Billing Specialist", default_options={})

        def fake_tool_decorator(*, name: str, description: str, approval_mode: str):
            def decorate(func):
                func.name = name
                func.description = description
                func.approval_mode = approval_mode
                return func

            return decorate

        team_agent_module._attach_handoff_hil_tools(
            agent,
            signal_state=signal_state,
            tool_decorator=fake_tool_decorator,
        )
        team_agent_module._attach_handoff_hil_tools(
            agent,
            signal_state=signal_state,
            tool_decorator=fake_tool_decorator,
        )

        tools = {tool.name: tool for tool in agent.default_options["tools"]}
        self.assertEqual(set(tools), {"request_user_input", "final_answer"})
        self.assertEqual(tools["request_user_input"](" Which date? "), "User input request recorded.")
        self.assertEqual(signal_state.action, "request_user_input")
        self.assertEqual(signal_state.prompt, "Which date?")
        self.assertEqual(signal_state.actor_name, "Billing Specialist")
        self.assertEqual(tools["final_answer"](" Refund initiated. "), "Final answer recorded.")
        self.assertEqual(signal_state.action, "final_answer")
        self.assertEqual(signal_state.message, "Refund initiated.")
        self.assertEqual(len(agent.default_options["tools"]), 2)


if __name__ == "__main__":
    unittest.main()
