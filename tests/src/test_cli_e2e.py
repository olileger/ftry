from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"


def _strip_ansi(text: str) -> str:
    return __import__("re").sub(r"\x1b\[[0-9;]*m", "", text)


def _write_stub_agent_framework(root: Path) -> None:
    package_dir = root / "agent_framework"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "openai.py").write_text(
        textwrap.dedent(
            """
            class Result:
                def __init__(self, text, value=None):
                    self.text = text
                    self.value = value


            class Agent:
                def __init__(
                    self,
                    name,
                    instructions,
                    description=None,
                    require_per_service_call_history_persistence=False,
                ):
                    self.name = name
                    self.instructions = instructions
                    self.description = description
                    self.require_per_service_call_history_persistence = require_per_service_call_history_persistence

                async def run(self, prompt, *, options=None, **kwargs):
                    if isinstance(options, dict) and "response_format" in options:
                        return Result(
                            f"{self.name}:{prompt}",
                            value={
                                "workflow_type": "magentic",
                                "reason": "Stubbed workflow inference result.",
                            },
                        )
                    return Result(f"{self.name}:{prompt}")


            class OpenAIChatCompletionClient:
                def __init__(self, *, model, api_key):
                    self.model = model
                    self.api_key = api_key

                def as_agent(
                    self,
                    *,
                    name,
                    instructions,
                    description=None,
                    require_per_service_call_history_persistence=False,
                ):
                    return Agent(
                        name=name,
                        instructions=instructions,
                        description=description,
                        require_per_service_call_history_persistence=require_per_service_call_history_persistence,
                    )
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (package_dir / "orchestrations.py").write_text(
        textwrap.dedent(
            """
            class WorkflowMessage:
                def __init__(self, role, text, author_name=None):
                    self.role = role
                    self.text = text
                    self.author_name = author_name


            class WorkflowResult:
                def __init__(self, messages):
                    self.messages = messages


            class WorkflowEvent:
                def __init__(self, type, data=None, executor_id=None):
                    self.type = type
                    self.data = data
                    self.executor_id = executor_id


            class GroupChatRequestSentEvent:
                def __init__(self, participant_name):
                    self.participant_name = participant_name


            class GroupChatResponseReceivedEvent:
                def __init__(self, participant_name):
                    self.participant_name = participant_name


            class HandoffSentEvent:
                def __init__(self, source, target):
                    self.source = source
                    self.target = target


            class WorkflowStream:
                def __init__(self, events):
                    self._events = events
                    self._index = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self._index >= len(self._events):
                        raise StopAsyncIteration
                    event = self._events[self._index]
                    self._index += 1
                    return event


            class Workflow:
                def __init__(self, pattern, **kwargs):
                    self.pattern = pattern
                    self.kwargs = kwargs

                def _participant_names(self):
                    participants = self.kwargs.get("participants", [])
                    return [participant.name for participant in participants]

                def _team_name(self):
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

                def run(self, prompt, *, stream=False):
                    participant_names = self._participant_names()
                    events = []

                    if self.pattern == "group-chat" and len(participant_names) >= 2:
                        first, second = participant_names[:2]
                        events.extend(
                            [
                                WorkflowEvent("group_chat", GroupChatRequestSentEvent(first)),
                                WorkflowEvent("executor_invoked", executor_id=first),
                                WorkflowEvent(
                                    "output",
                                    WorkflowResult([WorkflowMessage("assistant", "Draft prompt", author_name=first)]),
                                    executor_id=first,
                                ),
                                WorkflowEvent("group_chat", GroupChatResponseReceivedEvent(first)),
                                WorkflowEvent("group_chat", GroupChatRequestSentEvent(second)),
                                WorkflowEvent("executor_invoked", executor_id=second),
                                WorkflowEvent(
                                    "output",
                                    WorkflowResult([WorkflowMessage("assistant", "Review feedback", author_name=second)]),
                                    executor_id=second,
                                ),
                            ]
                        )
                    elif self.pattern == "handoff" and len(participant_names) >= 2:
                        source, target = participant_names[:2]
                        events.extend(
                            [
                                WorkflowEvent("executor_invoked", executor_id=source),
                                WorkflowEvent(
                                    "output",
                                    WorkflowResult([WorkflowMessage("assistant", "Initial triage", author_name=source)]),
                                    executor_id=source,
                                ),
                                WorkflowEvent("handoff_sent", HandoffSentEvent(source, target)),
                                WorkflowEvent("executor_invoked", executor_id=target),
                                WorkflowEvent(
                                    "output",
                                    WorkflowResult([WorkflowMessage("assistant", "Specialist answer", author_name=target)]),
                                    executor_id=target,
                                ),
                            ]
                        )
                    else:
                        for participant_name in participant_names:
                            events.append(WorkflowEvent("executor_invoked", executor_id=participant_name))
                            events.append(
                                WorkflowEvent(
                                    "output",
                                    WorkflowResult(
                                        [WorkflowMessage("assistant", f"{participant_name} handled {prompt}", author_name=participant_name)]
                                    ),
                                    executor_id=participant_name,
                                )
                            )

                    events.append(
                        WorkflowEvent(
                            "output",
                            [WorkflowMessage("assistant", f"{self.pattern}:{prompt}", author_name=self._team_name())],
                            executor_id=self._team_name(),
                        )
                    )
                    return WorkflowStream(events)


            class SequentialBuilder:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def build(self):
                    return Workflow("sequential", **self.kwargs)


            class ConcurrentBuilder:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def build(self):
                    return Workflow("concurrent", **self.kwargs)


            class GroupChatBuilder:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def build(self):
                    return Workflow("group-chat", **self.kwargs)


            class HandoffBuilder:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def with_start_agent(self, agent):
                    self.kwargs["start_agent"] = agent
                    return self

                def with_autonomous_mode(self, **kwargs):
                    self.kwargs["autonomous_mode"] = kwargs
                    return self

                def with_termination_condition(self, termination_condition):
                    self.kwargs["termination_condition"] = termination_condition
                    return self

                def build(self):
                    return Workflow("handoff", **self.kwargs)


            class MagenticBuilder:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def build(self):
                    return Workflow("magentic", **self.kwargs)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


class CliEndToEndTests(unittest.TestCase):
    def _run_cli(self, *args: str, with_agent_framework: bool = False) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.pop("OAI_API_KEY", None)

        pythonpath_entries = [str(SRC_DIR)]
        with TemporaryDirectory() as temp_dir:
            if with_agent_framework:
                stub_dir = Path(temp_dir)
                _write_stub_agent_framework(stub_dir)
                pythonpath_entries.insert(0, str(stub_dir))

            existing_pythonpath = env.get("PYTHONPATH")
            if existing_pythonpath:
                pythonpath_entries.append(existing_pythonpath)
            env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

            command = [sys.executable, "-m", "ftry.cli", *args]
            if env.get("FTRY_E2E_COVERAGE") == "1":
                command = [
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--parallel-mode",
                    "--source",
                    "ftry",
                    "-m",
                    "ftry.cli",
                    *args,
                ]

            return subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

    def test_mock_and_line_commands_run_through_cli_subprocess(self) -> None:
        build_result = self._run_cli("build")
        self.assertEqual(build_result.returncode, 0)
        self.assertEqual(build_result.stdout.strip(), "build")

        line_result = self._run_cli("line")
        self.assertEqual(line_result.returncode, 0)
        self.assertIn("______", _strip_ansi(line_result.stdout))

    def test_pop_agent_sample_runs_end_to_end_with_samples_and_dotenv(self) -> None:
        result = self._run_cli(
            "pop",
            "-a",
            r".\samples\agents\poete.yaml",
            "-p",
            "Ecris un poeme sur la pluie",
            with_agent_framework=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "")
        plain_stderr = _strip_ansi(result.stderr)
        self.assertIn("AGENT Poete | input:", plain_stderr)
        self.assertIn("AGENT Poete | final-output:", plain_stderr)

    def test_pop_team_sample_runs_end_to_end_with_samples_and_dotenv(self) -> None:
        result = self._run_cli(
            "pop",
            "-t",
            r".\samples\teams\better-prompt\team.yaml",
            "-p",
            "Ameliore ce prompt",
            with_agent_framework=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "")
        plain_stderr = _strip_ansi(result.stderr)
        self.assertIn("TEAM Better Prompt team | team-type-inference-prompt:", plain_stderr)
        self.assertIn("Team prompt:", plain_stderr)
        self.assertIn("You are managing a team of agents to build the best possible prompt.", plain_stderr)
        self.assertIn("...", plain_stderr)
        self.assertNotIn("Prompter, Reviewer, Runner", plain_stderr)
        self.assertNotIn("agent-prompter.yaml", plain_stderr)
        self.assertIn("TEAM Better Prompt team | team-type-inference-output:", plain_stderr)
        self.assertIn('"workflow_type": "magentic"', plain_stderr)
        self.assertIn("TEAM Better Prompt team | pattern: magentic | input:", plain_stderr)
        self.assertIn("TEAM Better Prompt team --> Prompter | input:", plain_stderr)
        self.assertIn("TEAM Better Prompt team <-- Runner | final-output:", plain_stderr)

