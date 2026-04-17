from __future__ import annotations

import unittest

import ftry.AgentTurnControl as agent_turn_control_module
from tests.src.testsupport import FakeResult


class AgentTurnControlTests(unittest.TestCase):
    def test_parse_agent_turn_response_parses_json_from_text(self) -> None:
        result = FakeResult('{"status":"await_user_input","message":"Need one detail"}')

        parsed = agent_turn_control_module.parse_agent_turn_response(
            result,
            error_subject="Team agent",
        )

        self.assertEqual(parsed.status, agent_turn_control_module.AGENT_RESPONSE_STATUS_AWAIT_USER_INPUT)
        self.assertEqual(parsed.message, "Need one detail")
        self.assertTrue(parsed.awaits_user_input)

    def test_parse_agent_turn_response_rejects_invalid_json_text(self) -> None:
        result = FakeResult("{not-json")

        with self.assertRaisesRegex(
            agent_turn_control_module.FtryCliError,
            "structured control payload required for console interaction",
        ):
            agent_turn_control_module.parse_agent_turn_response(
                result,
                error_subject="Team agent",
            )


if __name__ == "__main__":
    unittest.main()
