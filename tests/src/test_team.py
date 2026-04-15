from __future__ import annotations

import asyncio
import io
import os
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ftry.Team as team_module
from tests.src.testsupport import (
    CONCURRENT_SAMPLE_TEAM_FILE,
    FakeAgent,
    FakeConcurrentBuilder,
    FakeGroupChatBuilder,
    FakeHandoffBuilder,
    FakeMagenticBuilder,
    FakeOpenAIChatCompletionClient,
    FakeSequentialBuilder,
    GROUP_CHAT_SAMPLE_TEAM_FILE,
    HANDOFF_SAMPLE_TEAM_FILE,
    MAGENTIC_SAMPLE_TEAM_FILE,
    SAMPLE_TEAM_FILE,
    SEQUENTIAL_SAMPLE_TEAM_FILE,
    make_fake_agent_framework_modules,
    reset_fakes,
    strip_ansi,
)


class TeamTests(unittest.TestCase):
    def _make_agent_config(
        self,
        *,
        name: str = "Agent",
        description: str | None = "Helpful specialist.",
        instructions: str = "Do the work.",
        provider: str = "openai",
    ) -> team_module.AgentConfig:
        return team_module.AgentConfig(
            name=name,
            description=description,
            instructions=instructions,
            model=team_module.AgentModelConfig(name="gpt-4o", provider=provider, api_key="secret"),
        )

    def _make_team_config(
        self,
        *agents: team_module.AgentConfig,
        name: str = "Team",
        description: str | None = "Helpful team.",
        instructions: str = "Discuss and solve the request.",
        with_model: bool = False,
        max_turns: int | None = None,
    ) -> team_module.TeamConfig:
        return team_module.TeamConfig(
            name=name,
            description=description,
            instructions=instructions,
            agents=agents or (self._make_agent_config(),),
            model=team_module.AgentModelConfig(name="gpt-4o", provider="openai", api_key="secret") if with_model else None,
            termination=team_module.TeamTerminationConfig(max_turns=max_turns),
        )

    def _make_team(self, *agents: team_module.AgentConfig, **kwargs: object) -> team_module.Team:
        return team_module.Team(self._make_team_config(*agents, **kwargs))

    def test_termination_helper_parses_and_validates_max_turns(self) -> None:
        self.assertEqual(team_module.Team._parse_termination(None), team_module.TeamTerminationConfig())
        self.assertEqual(team_module.Team._parse_termination({}), team_module.TeamTerminationConfig())
        self.assertEqual(
            team_module.Team._parse_termination({"max-turns": 4}),
            team_module.TeamTerminationConfig(max_turns=4),
        )
        with self.assertRaisesRegex(team_module.FtryCliError, "expected a positive integer"):
            team_module.Team._parse_termination({"max-turns": -1})

    def test_render_team_instructions_and_analysis_helpers_include_roles(self) -> None:
        agent = self._make_agent_config(
            name="Prompter",
            description=" Builds prompts.\nWith care. ",
            instructions="Build a prompt.",
        )
        reviewer = self._make_agent_config(name="Reviewer", description=None, instructions="Review drafts.")
        team = self._make_team(
            agent,
            reviewer,
            name="Workshop",
            description="Discuss together.",
            instructions="Use {participants}.\n{roles}\n1. Draft.\n2. Review.",
        )

        self.assertEqual(team_module.Team._render_role_summary(agent), "Builds prompts. With care.")
        rendered = team._render_instructions()
        self.assertIn("Prompter, Reviewer", rendered)
        self.assertIn("- Reviewer: Review drafts.", rendered)
        inference_input = team._render_pattern_inference_input(rendered_instructions=rendered)
        self.assertIn("Team name: Workshop", inference_input)
        self.assertIn("Team prompt:", inference_input)
        self.assertNotIn("Raw team prompt template:", inference_input)
        self.assertNotIn("Build a prompt.", inference_input)

    def test_from_mapping_and_config_property_expose_parsed_team_config(self) -> None:
        raw_config = {
            "name": "Inline Team",
            "agents": [
                {
                    "name": "Inline Agent",
                    "model": {
                        "name": "gpt-4o",
                        "provider": "openai",
                        "api-key": "env:OAI_API_KEY",
                    },
                    "prompt": "Handle the request.",
                }
            ],
            "prompt": "Coordinate the work.",
        }

        with patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
            team = team_module.Team.from_mapping(raw_config)

        self.assertEqual(team.config.name, "Inline Team")
        self.assertEqual(team.config.instructions, "Coordinate the work.")
        self.assertEqual(team.config.agents[0].name, "Inline Agent")

    def test_team_loader_and_inference_helpers_cover_string_file_refs_missing_defaults_and_fallbacks(self) -> None:
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
                loaded_agent = team_module.Team._load_agent_config(f".\\{agent_file.name}", team_dir=temp_path)
                inline_agent = team_module.Team._load_agent_config(
                    {
                        "name": "Inline Agent",
                        "model": {
                            "name": "gpt-4o",
                            "provider": "openai",
                            "api-key": "env:OAI_API_KEY",
                        },
                        "prompt": "Work inline.",
                    },
                    team_dir=temp_path,
                )

            self.assertEqual(loaded_agent.name, "String Agent")
            self.assertEqual(inline_agent.name, "Inline Agent")
            with self.assertRaisesRegex(team_module.FtryCliError, "Team file not found"):
                team_module.Team.from_file(temp_path / "missing-team.yaml")

        first_agent = FakeAgent(name="Alpha", instructions="Handle the task.", description="First specialist.")
        second_agent = FakeAgent(name="Beta", instructions="Continue the work.", description="Second specialist.")
        self.assertEqual(team_module.Team._select_handoff_start_agent([first_agent, second_agent]), first_agent)

    def test_team_from_file_validates_nominal_and_error_cases(self) -> None:
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

                team_file = temp_path / "team.yaml"
                team_file.write_text(
                    "\n".join(
                        [
                            "name: Explicit Team",
                            "agents:",
                            f"  - file: .\\{agent_file.name}",
                            "prompt: |",
                            "  Coordinate the work.",
                        ]
                    ),
                    encoding="utf-8",
                )

                team = team_module.Team.from_file(team_file)
                self.assertEqual(team.name, "Explicit Team")
                self.assertEqual(team.agents[0].name, "Test Agent")

                explicit_pattern_file = temp_path / "pattern-team.yaml"
                explicit_pattern_file.write_text(
                    "\n".join(
                        [
                            "name: Pattern Team",
                            "pattern: sequential",
                            "agents:",
                            f"  - file: .\\{agent_file.name}",
                            "prompt: |",
                            "  Coordinate the work.",
                        ]
                    ),
                    encoding="utf-8",
                )

                explicit_orchestration_file = temp_path / "orchestration-team.yaml"
                explicit_orchestration_file.write_text(
                    "\n".join(
                        [
                            "name: Orchestration Team",
                            "orchestration: concurrent",
                            "agents:",
                            f"  - file: .\\{agent_file.name}",
                            "prompt: |",
                            "  Coordinate the work.",
                        ]
                    ),
                    encoding="utf-8",
                )

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

                with self.assertRaisesRegex(team_module.FtryCliError, "`file` references cannot be mixed with inline fields"):
                    team_module.Team.from_file(bad_team_file)
                with self.assertRaisesRegex(team_module.FtryCliError, "Invalid or missing `agents` list"):
                    team_module.Team.from_file(empty_agents_file)
                with self.assertRaisesRegex(team_module.FtryCliError, "Explicit team workflow selection is no longer supported"):
                    team_module.Team.from_file(explicit_pattern_file)
                with self.assertRaisesRegex(team_module.FtryCliError, "Explicit team workflow selection is no longer supported"):
                    team_module.Team.from_file(explicit_orchestration_file)

            with patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
                repo_sample = team_module.Team.from_file(SAMPLE_TEAM_FILE)
                sequential_repo_sample = team_module.Team.from_file(SEQUENTIAL_SAMPLE_TEAM_FILE)
                concurrent_repo_sample = team_module.Team.from_file(CONCURRENT_SAMPLE_TEAM_FILE)
                group_chat_repo_sample = team_module.Team.from_file(GROUP_CHAT_SAMPLE_TEAM_FILE)
                handoff_repo_sample = team_module.Team.from_file(HANDOFF_SAMPLE_TEAM_FILE)
                magentic_repo_sample = team_module.Team.from_file(MAGENTIC_SAMPLE_TEAM_FILE)
            self.assertEqual(repo_sample.name, "Better Prompt team")
            self.assertEqual(len(repo_sample.agents), 3)
            self.assertEqual(sequential_repo_sample.name, "Support Brief team")
            self.assertEqual(
                [agent.name for agent in sequential_repo_sample.agents],
                ["Fact Extractor", "Update Drafter", "Final Checker"],
            )
            self.assertEqual(concurrent_repo_sample.name, "Release Readiness team")
            self.assertEqual(
                [agent.name for agent in concurrent_repo_sample.agents],
                ["Value Analyst", "Risk Reviewer", "Launch Note Drafter"],
            )
            self.assertEqual(group_chat_repo_sample.name, "Feature Debate team")
            self.assertEqual(
                [agent.name for agent in group_chat_repo_sample.agents],
                ["Product Lead", "Implementation Lead", "Customer Advocate"],
            )
            self.assertEqual(handoff_repo_sample.name, "Support Routing team")
            self.assertEqual(
                [agent.name for agent in handoff_repo_sample.agents],
                ["Triage Router", "Billing Specialist", "Access Specialist", "Response Finisher"],
            )
            self.assertEqual(magentic_repo_sample.name, "Launch Planning team")
            self.assertEqual(
                [agent.name for agent in magentic_repo_sample.agents],
                ["Scope Analyst", "Risk Reviewer", "Brief Writer"],
            )

    def test_team_from_file_resolves_project_relative_file_references(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            team_dir = temp_path / "samples" / "teams" / "better-prompt"
            team_dir.mkdir(parents=True)

            for file_name, name in (
                ("agent-prompter.yaml", "Prompter"),
                ("agent-reviewer.yaml", "Reviewer"),
                ("agent-runner.yaml", "Runner"),
            ):
                (team_dir / file_name).write_text(
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

            team_file = team_dir / "team.yaml"
            team_file.write_text(
                "\n".join(
                    [
                        "name: Better Prompt team",
                        "agents:",
                        "  - file: ./agent-prompter.yaml",
                        "  - file: ./agent-reviewer.yaml",
                        "  - file: ./agent-runner.yaml",
                        "prompt: |",
                        "  Select the most appropriate tool and iterate with review if needed.",
                    ]
                ),
                encoding="utf-8",
            )

            previous_cwd = Path.cwd()
            os.chdir(temp_path)
            try:
                with patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False):
                    team = team_module.Team.from_file(r".\samples\teams\better-prompt\team.yaml")
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(team.name, "Better Prompt team")
        self.assertEqual([agent.name for agent in team.agents], ["Prompter", "Reviewer", "Runner"])

    def test_create_controller_agent_applies_team_context(self) -> None:
        reset_fakes()
        agent_config = self._make_agent_config(name="Runner", description="Executes prompts.", instructions="Run it.")
        team = self._make_team(
            agent_config,
            name="Better Prompt team",
            instructions="Use {participants}.\n{roles}",
            with_model=True,
        )

        with patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False):
            controller = team._create_controller_agent(instructions=team._render_instructions())
            self.assertIsInstance(controller, FakeAgent)
            self.assertEqual(controller.name, "Better-Prompt-team")
            self.assertEqual(self._make_team(agent_config)._create_controller_agent(instructions="ctx"), None)

    def test_infer_team_pattern_uses_structured_llm_output(self) -> None:
        reset_fakes()
        agent = self._make_agent_config(
            description="Helpful specialist.",
            instructions="AGENT PROMPT SHOULD NOT APPEAR",
        )
        with patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False):
            FakeAgent.next_value = {
                "workflow_type": "concurrent",
                "reason": "Agents can work independently on the same input.",
            }
            inferred_team = self._make_team(
                agent,
                name="研究团队",
                description="Analyse plusieurs aspects.",
                instructions=(
                    "Participants: {participants}\n"
                    "Roles:\n"
                    "{roles}\n"
                    "Step 1\n"
                    "Step 2\n"
                    "Step 3\n"
                    "Step 4\n"
                    "请协作完成这个任务。"
                ),
            )
            rendered = inferred_team._render_instructions()
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(asyncio.run(inferred_team._infer_pattern(rendered_instructions=rendered)), "concurrent")
            self.assertEqual(FakeOpenAIChatCompletionClient.last_model, "gpt-4o")
            self.assertIsInstance(FakeAgent.last_options, dict)
            self.assertIn("response_format", FakeAgent.last_options)
            self.assertEqual(FakeAgent.last_options["response_format"]["type"], "json_schema")
            self.assertIn("Team prompt:", FakeAgent.last_prompt or "")
            self.assertIn("Participants: Agent", FakeAgent.last_prompt or "")
            self.assertIn("- Agent: Helpful specialist.", FakeAgent.last_prompt or "")
            self.assertIn("请协作完成这个任务。", FakeAgent.last_prompt or "")
            self.assertNotIn("AGENT PROMPT SHOULD NOT APPEAR", FakeAgent.last_prompt or "")
            plain_stderr = strip_ansi(stderr.getvalue())
            self.assertIn("TEAM 研究团队 | team-type-inference-prompt:", plain_stderr)
            self.assertIn("Team prompt:", plain_stderr)
            self.assertIn("Participants: Agent", plain_stderr)
            self.assertIn("Roles:", plain_stderr)
            self.assertNotIn("Step 1", plain_stderr)
            self.assertNotIn("Step 4", plain_stderr)
            self.assertNotIn("请协作完成这个任务。", plain_stderr)
            self.assertIn("\n\t...", stderr.getvalue())
            self.assertNotIn("AGENT PROMPT SHOULD NOT APPEAR", plain_stderr)
            self.assertIn("TEAM 研究团队 | team-type-inference-output:", plain_stderr)
            self.assertIn('"workflow_type": "concurrent"', plain_stderr)
            self.assertIn('"reason": "Agents can work independently on the same input."', plain_stderr)

    def test_inference_helpers_cover_validation_and_provider_errors(self) -> None:
        class UnserializableValue:
            def __str__(self) -> str:
                return "custom-unserializable-value"

        self.assertEqual(
            team_module.Team._format_pattern_inference_output(UnserializableValue()),
            "custom-unserializable-value",
        )

        with self.assertRaisesRegex(team_module.FtryCliError, "JSON object"):
            team_module.Team._parse_pattern_inference("invalid")

        with self.assertRaisesRegex(team_module.FtryCliError, "valid `reason`"):
            team_module.Team._parse_pattern_inference({"workflow_type": "sequential", "reason": "   "})

        unsupported_team = self._make_team(
            self._make_agent_config(provider="anthropic"),
            name="Unsupported Team",
            instructions="Coordinate the work.",
        )
        with self.assertRaisesRegex(team_module.FtryCliError, "Unsupported provider `anthropic`"):
            asyncio.run(unsupported_team._infer_pattern(rendered_instructions=unsupported_team._render_instructions()))

    def test_infer_team_pattern_rejects_invalid_structured_output(self) -> None:
        reset_fakes()
        agent = self._make_agent_config()
        team = self._make_team(agent, name="Team", instructions="A prompt.")

        with patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False):
            FakeAgent.next_value = {"reason": "Missing workflow type."}
            with self.assertRaisesRegex(team_module.FtryCliError, "valid `workflow_type`"):
                asyncio.run(team._infer_pattern(rendered_instructions=team._render_instructions()))

            FakeAgent.next_value = {"workflow_type": "swarm", "reason": "Unsupported."}
            with self.assertRaisesRegex(team_module.FtryCliError, "unsupported `workflow_type` `swarm`"):
                asyncio.run(team._infer_pattern(rendered_instructions=team._render_instructions()))

    def test_build_team_participants_and_workflows_cover_pattern_specific_logic(self) -> None:
        reset_fakes()
        duplicate_a = self._make_agent_config(name="Agent", instructions="Do A.")
        duplicate_b = self._make_agent_config(name="Agent", instructions="Do B.")
        router = self._make_agent_config(name="Router", description="Route the request.", instructions="Triage the task.")
        specialist = self._make_agent_config(name="Specialist", instructions="Answer the request.")

        with (
            patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
            patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False),
        ):
            participants, author_name_map = self._make_team(
                duplicate_a,
                duplicate_b,
                router,
                instructions="Use {participants}.",
            )._build_participants(extra_instructions="Shared context")
            self.assertEqual([participant.name for participant in participants], ["Agent", "Agent-2", "Router"])
            self.assertEqual(author_name_map["Agent-2"], "Agent")
            self.assertIn("<TeamContext>", participants[0].instructions)
            self.assertEqual(team_module.Team._select_handoff_start_agent(participants).name, "Router")
            self.assertEqual(
                team_module.Team._count_assistant_messages(
                    [
                        type("Message", (), {"role": "assistant"})(),
                        type("Message", (), {"role": "user"})(),
                        type("Message", (), {"role": "assistant"})(),
                    ]
                ),
                2,
            )

            FakeAgent.next_value = {"workflow_type": "sequential", "reason": "Pipeline."}
            asyncio.run(
                self._make_team(
                    duplicate_a,
                    specialist,
                    name="Pipeline",
                    instructions="First draft, then refine.",
                )._build_workflow()
            )
            self.assertTrue(FakeSequentialBuilder.last_kwargs["intermediate_outputs"])

            FakeAgent.next_value = {"workflow_type": "concurrent", "reason": "Independent."}
            asyncio.run(
                self._make_team(
                    duplicate_a,
                    specialist,
                    name="Swarm",
                    instructions="Work in parallel.",
                )._build_workflow()
            )
            self.assertTrue(FakeConcurrentBuilder.last_kwargs["intermediate_outputs"])

            FakeAgent.next_value = {"workflow_type": "handoff", "reason": "Routing."}
            asyncio.run(
                self._make_team(
                    router,
                    specialist,
                    name="Triage",
                    instructions="Route and handoff.",
                    max_turns=3,
                )._build_workflow()
            )
            self.assertEqual(FakeHandoffBuilder.last_start_agent.name, "Router")
            self.assertEqual(FakeHandoffBuilder.last_autonomous_kwargs["turn_limits"]["Router"], 3)
            self.assertTrue(callable(FakeHandoffBuilder.last_termination_condition))
            self.assertTrue(
                all(
                    not participant.require_per_service_call_history_persistence
                    for participant in FakeHandoffBuilder.last_autonomous_kwargs["agents"]
                )
            )

            FakeAgent.next_value = {"workflow_type": "group-chat", "reason": "Collaboration."}
            asyncio.run(
                self._make_team(
                    duplicate_a,
                    specialist,
                    name="Workshop",
                    instructions="Discuss together.",
                    with_model=True,
                    max_turns=5,
                )._build_workflow()
            )
            self.assertEqual(FakeGroupChatBuilder.last_kwargs["max_rounds"], 5)
            self.assertIsInstance(FakeGroupChatBuilder.last_kwargs["orchestrator_agent"], FakeAgent)

            FakeAgent.next_value = {"workflow_type": "magentic", "reason": "Planning."}
            asyncio.run(
                self._make_team(
                    duplicate_a,
                    specialist,
                    name="Planner",
                    instructions="Plan and replan a complex task.",
                    with_model=True,
                    max_turns=4,
                )._build_workflow()
            )
            self.assertEqual(FakeMagenticBuilder.last_kwargs["max_round_count"], 4)
            self.assertIsInstance(FakeMagenticBuilder.last_kwargs["manager_agent"], FakeAgent)

    def test_run_handles_handoff_event_stream(self) -> None:
        reset_fakes()
        stderr = io.StringIO()
        team = self._make_team(
            self._make_agent_config(name="Router", description="Route the request.", instructions="Triage the task."),
            self._make_agent_config(name="Specialist", description="Handles the final answer.", instructions="Solve the task."),
            name="Handoff squad",
            instructions="Route the request and handoff to specialists.",
            max_turns=2,
        )

        with (
            patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
            patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False),
            redirect_stderr(stderr),
        ):
            FakeAgent.next_value = {"workflow_type": "handoff", "reason": "Routing."}
            output = asyncio.run(team.run("Route this request"))

        self.assertEqual(output, "[Specialist]\nhandoff:Route this request")
        plain_stderr = strip_ansi(stderr.getvalue())
        self.assertIn("TEAM (H) Handoff squad | pattern: handoff | input:", plain_stderr)
        self.assertIn("Router --> Specialist | input:", plain_stderr)
        self.assertIn("TEAM (H) Handoff squad <-- Specialist | final-output:", plain_stderr)

    def test_run_prefers_last_agent_output_when_team_authors_final_message(self) -> None:
        reset_fakes()
        stderr = io.StringIO()
        team = self._make_team(
            self._make_agent_config(name="Prompter", instructions="Draft the prompt."),
            self._make_agent_config(name="Reviewer", instructions="Review the draft."),
            name="Better Prompt team",
            instructions="Discuss together and choose the best prompt.",
            with_model=True,
        )

        with (
            patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
            patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False),
            redirect_stderr(stderr),
        ):
            FakeAgent.next_value = {"workflow_type": "group-chat", "reason": "Collaboration."}
            output = asyncio.run(team.run("Ameliore ce prompt"))

        self.assertEqual(output, "[Better Prompt team]\ngroup-chat:Ameliore ce prompt")
        plain_stderr = strip_ansi(stderr.getvalue())
        final_output_log = plain_stderr.split("TEAM (G) Better Prompt team <-- Reviewer | final-output:", maxsplit=1)[1]
        self.assertIn("Review feedback", final_output_log)
        self.assertNotIn("[Better Prompt team]\n\tgroup-chat:Ameliore ce prompt", final_output_log)

    def test_run_logs_full_final_output_without_truncation(self) -> None:
        reset_fakes()
        stderr = io.StringIO()
        long_prompt = "x" * 260
        team = self._make_team(
            self._make_agent_config(name="Researcher", instructions="Gather the facts."),
            self._make_agent_config(name="Writer", instructions="Write the final answer."),
            name="Pipeline team",
            instructions="First gather the facts, then draft the answer, and finally produce the final response.",
        )

        with (
            patch.dict(os.environ, {"OAI_API_KEY": "secret-key"}, clear=False),
            patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False),
            redirect_stderr(stderr),
        ):
            FakeAgent.next_value = {"workflow_type": "sequential", "reason": "Pipeline."}
            output = asyncio.run(team.run(long_prompt))

        self.assertEqual(output, f"[Writer]\nsequential:{long_prompt}")
        plain_stderr = strip_ansi(stderr.getvalue())
        final_output_log = plain_stderr.split("TEAM (S) Pipeline team <-- Writer | final-output:", maxsplit=1)[1]
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
            team_pattern="sequential",
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
            patch("ftry.Team._trace_team_label", return_value="TEAM (S) Pipeline team"),
            patch("ftry.Team._trace_block", return_value="\n\tRendered output"),
            patch("ftry.Team._trace") as trace_message,
        ):
            fallback_state.trace_final_output(object(), "Rendered output", {})

        trace_message.assert_called_once_with(
            "%s | final-output:%s",
            "TEAM (S) Pipeline team",
            "\n\tRendered output",
        )

    def test_team_event_helpers_cover_guard_paths_and_executor_transitions(self) -> None:
        request_driven_state = team_module._TeamTraceState(
            pattern="group-chat",
            team_name="Workshop",
            agent_trace_colors={},
            last_visible_input="Prompt",
        )

        team_module.Team._handle_executor_invoked_event(type("Event", (), {"executor_id": None})(), request_driven_state, {})
        self.assertIsNone(request_driven_state.active_executor)

        request_driven_state.expected_invoked_executor = "Reviewer"
        with patch.object(request_driven_state, "flush_buffer") as flush_buffer:
            team_module.Team._handle_executor_invoked_event(
                type("Event", (), {"executor_id": "Prompter"})(),
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
            team_module.Team._handle_executor_invoked_event(
                type("Event", (), {"executor_id": "Worker"})(),
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
        empty_output_event = type("Event", (), {"data": object(), "executor_id": "Researcher"})()
        with patch("ftry.Team._summarize_payload", return_value=""):
            team_module.Team._handle_output_event(empty_output_event, output_state, {})
        self.assertIsNone(output_state.active_executor)

        first_output_event = type("Event", (), {"data": object(), "executor_id": "Researcher"})()
        with (
            patch("ftry.Team._summarize_payload", return_value="Draft"),
            patch("ftry.Team._extract_trace_chunk", return_value="Chunk A"),
        ):
            team_module.Team._handle_output_event(first_output_event, output_state, {})
        self.assertEqual(output_state.active_executor, "Researcher")
        self.assertEqual(output_state.buffered_outputs, ["Chunk A"])

        second_output_event = type("Event", (), {"data": object(), "executor_id": "Writer"})()
        with (
            patch("ftry.Team._summarize_payload", return_value="Review"),
            patch("ftry.Team._extract_trace_chunk", return_value="Chunk B"),
            patch.object(output_state, "flush_buffer", wraps=output_state.flush_buffer) as flush_buffer,
        ):
            team_module.Team._handle_output_event(second_output_event, output_state, {})
        flush_buffer.assert_called_once_with(next_executor="Writer")
        self.assertEqual(output_state.active_executor, "Writer")
        self.assertEqual(output_state.buffered_outputs, ["Chunk B"])


if __name__ == "__main__":
    unittest.main()
