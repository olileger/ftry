from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ftry.Builder as builder_module
from ftry.StandaloneAgent import StandaloneAgent
from ftry.Team import Team


class BuilderTests(unittest.TestCase):
    def test_build_from_prompt_rejects_missing_or_non_directory_output_path(self) -> None:
        spec = builder_module.BuildSpec(
            kind=builder_module.BUILD_KIND_AGENT,
            agent=builder_module._BuiltAgentSpec(
                name="Writer",
                description="Writes content.",
                prompt="Write the content.",
            ),
        )
        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "not-a-dir.txt"
            file_path.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(builder_module.FtryCliError, "not a directory"):
                builder_module._resolve_build_output_dir(spec, output_dir=file_path)

    def test_parse_build_spec_output_accepts_fenced_json(self) -> None:
        spec = builder_module._parse_build_spec_output(
            """```json
            {"kind":"agent","agent":{"name":"Writer","description":"Writes short copy.","prompt":"Write the requested copy."}}
            ```"""
        )

        self.assertEqual(spec.kind, builder_module.BUILD_KIND_AGENT)
        assert spec.agent is not None
        self.assertEqual(spec.agent.name, "Writer")
        self.assertEqual(spec.agent.prompt, "Write the requested copy.")
        self.assertEqual(builder_module._require_optional_output_text(None, "agent.description"), None)

    def test_parse_build_spec_output_rejects_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(builder_module.FtryCliError, "valid `kind`"):
            builder_module._parse_build_spec_output('{"kind":"workflow"}')

        with self.assertRaisesRegex(builder_module.FtryCliError, "must include a `agent` object"):
            builder_module._parse_build_spec_output('{"kind":"agent"}')

        with self.assertRaisesRegex(builder_module.FtryCliError, "non-empty `agent.prompt`"):
            builder_module._parse_build_spec_output(
                '{"kind":"agent","agent":{"name":"Writer","description":"Desc","prompt":"   "}}'
            )

        with self.assertRaisesRegex(builder_module.FtryCliError, "must include a `team` object"):
            builder_module._parse_build_spec_output('{"kind":"team","agents":[{"name":"A","description":"D","prompt":"P"}]}')

        with self.assertRaisesRegex(builder_module.FtryCliError, "positive integer `team.max_turns`"):
            builder_module._parse_build_spec_output(
                '{"kind":"team","team":{"name":"Team","description":"Desc","prompt":"Prompt","max_turns":0},"agents":[{"name":"A","description":"D","prompt":"P"}]}'
            )

        with self.assertRaisesRegex(builder_module.FtryCliError, "non-empty `agents` list"):
            builder_module._parse_build_spec_output(
                '{"kind":"team","team":{"name":"Team","description":"Desc","prompt":"Prompt","max_turns":4},"agents":[]}'
            )

    def test_build_from_prompt_writes_agent_yaml_and_validates_it(self) -> None:
        with TemporaryDirectory() as temp_dir, patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
            temp_path = Path(temp_dir)
            with patch(
                "ftry.Builder._run_builder_team",
                return_value=json.dumps(
                    {
                        "kind": "agent",
                        "agent": {
                            "name": "Release Writer",
                            "description": "Writes short release updates.",
                            "prompt": "Write a concise release note from the user's request.",
                        },
                    }
                ),
            ):
                created_files = builder_module.build_from_prompt("Create a release note agent.", output_dir=temp_path)

            agent_output_dir = temp_path / "release-writer"
            self.assertEqual(created_files, (agent_output_dir / "agent.yaml",))
            rendered_agent = (agent_output_dir / "agent.yaml").read_text(encoding="utf-8")
            self.assertIn('name: "Release Writer"', rendered_agent)
            self.assertIn("prompt: |", rendered_agent)
            self.assertEqual(StandaloneAgent.from_file(agent_output_dir / "agent.yaml").name, "Release Writer")

    def test_build_from_prompt_writes_team_yaml_and_agent_files(self) -> None:
        with TemporaryDirectory() as temp_dir, patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
            temp_path = Path(temp_dir)
            with patch(
                "ftry.Builder._run_builder_team",
                return_value=json.dumps(
                    {
                        "kind": "team",
                        "team": {
                            "name": "Feature Launch team",
                            "description": "Coordinates a launch brief.",
                            "prompt": "Use {participants} and {roles} to build a lightweight launch brief.",
                            "max_turns": 5,
                        },
                        "agents": [
                            {
                                "name": "Scope Lead",
                                "description": "Keeps the scope realistic.",
                                "prompt": "Define the smallest useful scope.",
                            },
                            {
                                "name": "Risk Lead",
                                "description": "Highlights launch risks.",
                                "prompt": "Identify the main delivery risks.",
                            },
                        ],
                    }
                ),
            ):
                created_files = builder_module.build_from_prompt("Build a launch team.", output_dir=temp_path)

            team_output_dir = temp_path / "feature-launch-team"
            self.assertEqual(created_files[0].name, "agent-scope-lead.yaml")
            self.assertEqual(created_files[1].name, "agent-risk-lead.yaml")
            self.assertEqual(created_files[2].name, "team.yaml")
            rendered_team = (team_output_dir / "team.yaml").read_text(encoding="utf-8")
            self.assertIn("max-turns: 5", rendered_team)
            self.assertIn("  - file: ./agent-scope-lead.yaml", rendered_team)
            self.assertEqual(Team.from_file(team_output_dir / "team.yaml").name, "Feature Launch team")

    def test_build_from_prompt_rejects_existing_output_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "writer"
            output_dir.mkdir()
            (output_dir / "agent.yaml").write_text("name: Existing\n", encoding="utf-8")
            with patch(
                "ftry.Builder._run_builder_team",
                return_value=json.dumps(
                    {
                        "kind": "agent",
                        "agent": {
                            "name": "Writer",
                            "description": "Writes content.",
                            "prompt": "Write the content.",
                        },
                    }
                ),
            ):
                with self.assertRaisesRegex(builder_module.FtryCliError, "overwrite existing files"):
                    builder_module.build_from_prompt("Create an agent.", output_dir=temp_path)

    def test_build_from_prompt_rejects_invalid_builder_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch("ftry.Builder._run_builder_team", return_value="not json at all"):
                with self.assertRaisesRegex(builder_module.FtryCliError, "structured JSON payload"):
                    builder_module.build_from_prompt("Create something.", output_dir=temp_dir)

    def test_write_build_outputs_cleans_up_files_when_validation_fails(self) -> None:
        spec = builder_module.BuildSpec(
            kind=builder_module.BUILD_KIND_AGENT,
            agent=builder_module._BuiltAgentSpec(
                name="Writer",
                description="Writes content.",
                prompt="Write the content.",
            ),
        )
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with patch("ftry.Builder.StandaloneAgent.from_file", side_effect=builder_module.FtryCliError("boom")):
                with self.assertRaisesRegex(builder_module.FtryCliError, "boom"):
                    builder_module._write_build_outputs(spec, output_dir=temp_path)
            self.assertFalse((temp_path / "agent.yaml").exists())

    def test_plan_team_agent_paths_handles_duplicate_and_fallback_names(self) -> None:
        specs = (
            builder_module._BuiltAgentSpec(name="Reviewer", description="A", prompt="P"),
            builder_module._BuiltAgentSpec(name="Reviewer", description="B", prompt="P"),
        )
        planned_paths = builder_module._plan_team_agent_paths(specs, output_dir=Path(r"C:\tmp"))
        self.assertEqual(planned_paths[0].name, "agent-reviewer.yaml")
        self.assertEqual(planned_paths[1].name, "agent-reviewer-2.yaml")

        with patch("ftry.Builder._sanitize_agent_name", return_value="----"):
            fallback_paths = builder_module._plan_team_agent_paths(specs[:1], output_dir=Path(r"C:\tmp"))
        self.assertEqual(fallback_paths[0].name, "agent-agent-1.yaml")

    def test_build_from_prompt_uses_default_output_directory_under_current_directory(self) -> None:
        with TemporaryDirectory() as temp_dir, patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
            temp_path = Path(temp_dir)
            with (
                patch("ftry.Builder._run_builder_team", return_value=json.dumps(
                    {
                        "kind": "agent",
                        "agent": {
                            "name": "Idea Builder",
                            "description": "Develops an idea.",
                            "prompt": "Develop the idea and identify the technical aspects.",
                        },
                    }
                )),
                patch("ftry.Builder.Path.cwd", return_value=temp_path),
            ):
                created_files = builder_module.build_from_prompt("Create an ideation agent.")

            self.assertEqual(created_files, (temp_path / "output" / "idea-builder" / "agent.yaml",))
            self.assertTrue((temp_path / "output").is_dir())

    def test_resolve_build_output_dir_creates_root_and_name_subdirectory(self) -> None:
        spec = builder_module.BuildSpec(
            kind=builder_module.BUILD_KIND_TEAM,
            team=builder_module._BuiltTeamSpec(
                name="Launch Squad",
                description="Coordinates launch work.",
                prompt="Coordinate the work.",
            ),
        )
        with TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir) / "custom-output"
            target_dir = builder_module._resolve_build_output_dir(spec, output_dir=root_dir)

            self.assertEqual(target_dir, root_dir / "launch-squad")
            self.assertTrue(root_dir.is_dir())
            self.assertTrue(target_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
