from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

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

    def test_find_dotenv_path_and_loader_use_expected_sources(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cwd_dir = temp_path / "cwd"
            cwd_dir.mkdir()
            config_dir = temp_path / "configs" / "nested"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "agent.yaml"
            config_path.write_text("name: Agent\n", encoding="utf-8")

            cwd_dotenv = cwd_dir / ".env"
            cwd_dotenv.write_text("FROM=cwd\n", encoding="utf-8")
            parent_dotenv = temp_path / "configs" / ".env"
            parent_dotenv.write_text("FROM=parent\n", encoding="utf-8")

            with patch("ftry.cli.Path.cwd", return_value=cwd_dir):
                self.assertEqual(cli._find_dotenv_path(config_path), cwd_dotenv)

            cwd_dotenv.unlink()

            with patch("ftry.cli.Path.cwd", return_value=cwd_dir):
                self.assertEqual(cli._find_dotenv_path(config_path), parent_dotenv)

            loaded_calls: list[tuple[Path, bool]] = []

            def fake_load_dotenv(*, dotenv_path: Path, override: bool) -> bool:
                loaded_calls.append((dotenv_path, override))
                return True

            with (
                patch("ftry.cli.Path.cwd", return_value=cwd_dir),
                patch("ftry.cli._load_dotenv_function", return_value=fake_load_dotenv),
            ):
                cli._load_dotenv_for_config(config_path)

            self.assertEqual(loaded_calls, [(parent_dotenv, False)])

            parent_dotenv.unlink()
            with (
                patch("ftry.cli.Path.cwd", return_value=cwd_dir),
                patch("ftry.cli._load_dotenv_function", side_effect=self.fail),
            ):
                cli._load_dotenv_for_config(config_path)
                self.assertIsNone(cli._find_dotenv_path(config_path))

    def test_run_pop_command_dispatches_agent_prompts(self) -> None:
        config = object()
        run_agent_prompt = AsyncMock(return_value="done")

        with (
            patch("ftry.cli._load_agent_config", return_value=config) as load_agent_config,
            patch("ftry.cli._run_agent_prompt", run_agent_prompt),
            patch("ftry.cli._render_pop_animation") as render_animation,
        ):
            exit_code = cli._run_pop_command("agent.yaml", None, "Bonjour")

        self.assertEqual(exit_code, 0)
        load_agent_config.assert_called_once_with("agent.yaml")
        run_agent_prompt.assert_awaited_once_with(config, "Bonjour")
        render_animation.assert_called_once_with()

    def test_run_pop_command_dispatches_team_prompts(self) -> None:
        config = object()
        run_team_prompt = AsyncMock(return_value="done")

        with (
            patch("ftry.cli._load_team_config", return_value=config) as load_team_config,
            patch("ftry.cli._run_team_prompt", run_team_prompt),
            patch("ftry.cli._render_pop_animation") as render_animation,
        ):
            exit_code = cli._run_pop_command(None, "team.yaml", "Bonjour")

        self.assertEqual(exit_code, 0)
        load_team_config.assert_called_once_with("team.yaml")
        run_team_prompt.assert_awaited_once_with(config, "Bonjour")
        render_animation.assert_called_once_with()

    def test_load_team_agent_config_delegates_to_team_loader(self) -> None:
        config = object()
        team_dir = Path("C:\\tmp")

        with patch("ftry.cli._team_load_team_agent_config", return_value=config) as load_team_agent_config:
            result = cli._load_team_agent_config({"name": "Agent"}, team_dir=team_dir)

        self.assertIs(result, config)
        load_team_agent_config.assert_called_once()

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

    def test_render_pop_animation_only_runs_in_a_tty(self) -> None:
        non_tty_stream = io.StringIO()
        cli._render_pop_animation(stream=non_tty_stream, sleep=lambda _: self.fail("sleep should not be called"))
        self.assertEqual(non_tty_stream.getvalue(), "")

        tty_stream = FakeTtyStream()
        delays: list[float] = []

        cli._render_pop_animation(stream=tty_stream, sleep=delays.append)

        rendered = tty_stream.getvalue()
        plain_rendered = strip_ansi(rendered)
        self.assertTrue(rendered.startswith(cli.POP_ANIMATION_CURSOR_HIDE))
        self.assertTrue(rendered.endswith(f"\n{cli.POP_ANIMATION_CURSOR_SHOW}"))
        self.assertIn(cli.BRIGHT_PINK, rendered)
        self.assertIn(strip_ansi(cli._load_pop_banner()).splitlines()[0], plain_rendered)
        self.assertIn("         .  .", plain_rendered)
        self.assertIn(r"         \______/>", plain_rendered)
        self.assertIn("          o    o", plain_rendered)
        self.assertIn("_" * 40, plain_rendered)
        self.assertIn(cli.POP_ANIMATION_CLEAR_LINE, rendered)
        self.assertNotIn(f"{cli.POP_ANIMATION_CLEAR_LINE}{cli.POP_ANIMATION_CURSOR_SHOW}", rendered)
        self.assertEqual(delays, [cli.POP_ANIMATION_STEP_SECONDS] * len(cli.POP_ANIMATION_FRAMES))

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


if __name__ == "__main__":
    unittest.main()
