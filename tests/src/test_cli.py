from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import AsyncMock, Mock, patch

from ftry import cli
from tests.src.testsupport import FakeTtyStream, SAMPLE_AGENT_FILE, SAMPLE_TEAM_FILE, strip_ansi


class CliTests(unittest.TestCase):
    def test_mock_and_line_commands_render_expected_output(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            mock_exit_code = cli._run_mock_command("build")
            line_exit_code = cli._run_line_command()

        self.assertEqual(mock_exit_code, 0)
        self.assertEqual(line_exit_code, 0)
        rendered = stdout.getvalue()
        self.assertIn("build", rendered)
        self.assertIn("______", strip_ansi(rendered))

    def test_run_pop_command_dispatches_agent_prompts(self) -> None:
        loaded_agent = Mock()
        loaded_agent.run = AsyncMock(return_value="done")

        with (
            patch("ftry.cli.StandaloneAgent.from_file", return_value=loaded_agent) as load_agent,
            patch("ftry.cli._print_pop_banner") as print_pop_banner,
        ):
            exit_code = cli._run_pop_command("agent.yaml", None, "Bonjour")

        self.assertEqual(exit_code, 0)
        print_pop_banner.assert_called_once_with()
        load_agent.assert_called_once_with("agent.yaml")
        loaded_agent.run.assert_awaited_once_with("Bonjour", user_input_provider=cli._read_agent_follow_up_input)

    def test_run_pop_command_dispatches_team_prompts(self) -> None:
        loaded_team = Mock()
        loaded_team.run = AsyncMock(return_value="done")

        with (
            patch("ftry.cli.Team.from_file", return_value=loaded_team) as load_team,
            patch("ftry.cli._print_pop_banner") as print_pop_banner,
        ):
            exit_code = cli._run_pop_command(None, "team.yaml", "Bonjour")

        self.assertEqual(exit_code, 0)
        print_pop_banner.assert_called_once_with()
        load_team.assert_called_once_with("team.yaml")
        loaded_team.run.assert_awaited_once_with("Bonjour", user_input_provider=cli._read_agent_follow_up_input)

    def test_main_reports_errors_and_direct_pop_requires_a_source(self) -> None:
        with self.assertRaisesRegex(cli.FtryCliError, "Either `-a/--agent-file` or `-t/--team-file` must be provided."):
            cli._run_pop_command(None, None, "Bonjour")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = cli.main(["pop", "-a", "missing.yaml", "-p", "Bonjour"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: Agent file not found:", stderr.getvalue())

    def test_build_parser_parses_pop_arguments(self) -> None:
        args = cli.build_parser().parse_args(["pop", "-a", r"samples\agents\poete.yaml", "-p", "Bonjour"])

        self.assertEqual(args.command, "pop")
        self.assertEqual(args.agent_file, r"samples\agents\poete.yaml")
        self.assertIsNone(args.team_file)
        self.assertEqual(args.prompt, "Bonjour")

    def test_build_parser_parses_pop_team_arguments(self) -> None:
        args = cli.build_parser().parse_args(["pop", "-t", r"samples\teams\better-prompt\team.yaml", "-p", "Bonjour"])

        self.assertEqual(args.command, "pop")
        self.assertEqual(args.team_file, r"samples\teams\better-prompt\team.yaml")
        self.assertIsNone(args.agent_file)
        self.assertEqual(args.prompt, "Bonjour")

    def test_print_pop_banner_only_runs_in_a_tty(self) -> None:
        non_tty_stream = io.StringIO()
        cli._print_pop_banner(stream=non_tty_stream)
        self.assertEqual(non_tty_stream.getvalue(), "")

        tty_stream = FakeTtyStream()
        cli._print_pop_banner(stream=tty_stream)

        rendered = tty_stream.getvalue()
        plain_rendered = strip_ansi(rendered)
        self.assertIn("======", plain_rendered)
        self.assertIn("_____", plain_rendered)

    def test_pop_rejects_team_file_passed_as_agent_file(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = cli.main(["pop", "-a", str(SAMPLE_TEAM_FILE), "-p", "Bonjour"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Invalid agent YAML", stderr.getvalue())
        self.assertIn("defines `agents` at the root", stderr.getvalue())
        self.assertIn("Use `-t/--team-file` instead of `-a/--agent-file`", stderr.getvalue())

    def test_pop_rejects_agent_file_passed_as_team_file(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = cli.main(["pop", "-t", str(SAMPLE_AGENT_FILE), "-p", "Bonjour"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Invalid team YAML", stderr.getvalue())
        self.assertIn("matches an agent configuration (`name`, `model`, `prompt`)", stderr.getvalue())
        self.assertIn("does not define `agents` at the root", stderr.getvalue())
        self.assertIn("Use `-a/--agent-file` instead of `-t/--team-file`", stderr.getvalue())

    def test_read_agent_follow_up_input_accepts_interactive_terminal(self) -> None:
        response = cli._read_agent_follow_up_input(
            "Question",
            input_stream=FakeTtyStream("oui\n"),
            output_stream=FakeTtyStream(),
        )

        self.assertEqual(response, "oui")

    def test_read_agent_follow_up_input_rejects_non_interactive_terminal(self) -> None:
        with self.assertRaisesRegex(cli.FtryCliError, "interactive terminal"):
            cli._read_agent_follow_up_input(
                "Question",
                input_stream=io.StringIO("oui\n"),
                output_stream=io.StringIO(),
            )


if __name__ == "__main__":
    unittest.main()
