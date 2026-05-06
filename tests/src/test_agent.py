from __future__ import annotations

import unittest

import ftry.Agent as agent_module


class AgentTests(unittest.TestCase):
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

    def test_agent_config_from_mapping_builds_common_config(self) -> None:
        config = agent_module.AgentConfig.from_mapping(
            {
                "name": "Inline Agent",
                "description": "Inline specialist.",
                "model": {
                    "name": "gpt-4o",
                    "provider": "openai",
                    "api-key": "secret",
                },
                "mcp": ["file-system", "github"],
                "prompt": "Work inline.",
            },
            config_kind="team agent",
        )

        self.assertEqual(config.name, "Inline Agent")
        self.assertEqual(config.description, "Inline specialist.")
        self.assertEqual(config.instructions, "Work inline.")
        self.assertEqual(config.model.name, "gpt-4o")
        self.assertEqual(config.mcp_servers, ("file-system", "github"))

    def test_agent_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            agent_module.Agent(self._make_agent_config())


if __name__ == "__main__":
    unittest.main()
