from __future__ import annotations

import os
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ftry.Tools as tools_module
from tests.src.testsupport import SAMPLE_AGENT_FILE, SAMPLE_TEAM_FILE


class ToolsTests(unittest.TestCase):
    def test_load_ascii_banners_replace_color_tokens(self) -> None:
        line_banner = tools_module._load_line_banner()
        pop_banner = tools_module._load_pop_banner()

        self.assertNotIn("[pink]", line_banner)
        self.assertNotIn("[cyan]", pop_banner)
        self.assertIn(tools_module.RESET, line_banner)
        self.assertIn(tools_module.RESET, pop_banner)

    def test_validation_helpers_handle_edge_cases(self) -> None:
        self.assertEqual(tools_module._require_non_empty_string("  value  ", "name"), "value")
        self.assertIsNone(tools_module._require_optional_string(None, "description"))
        self.assertEqual(tools_module._require_optional_string("  note  ", "description"), "note")
        self.assertEqual(tools_module._require_mapping({"ok": True}, "root"), {"ok": True})
        self.assertEqual(tools_module._require_sequence(["a"], "agents"), ["a"])
        self.assertEqual(tools_module._require_positive_int(3, "termination.max-turns", "team"), 3)

        with self.assertRaisesRegex(tools_module.FtryCliError, "Invalid or missing `name`"):
            tools_module._require_non_empty_string("   ", "name")
        with self.assertRaisesRegex(tools_module.FtryCliError, "Invalid or missing `description`"):
            tools_module._require_optional_string("", "description")
        with self.assertRaisesRegex(tools_module.FtryCliError, "Invalid or missing `root` mapping"):
            tools_module._require_mapping([], "root")
        with self.assertRaisesRegex(tools_module.FtryCliError, "Invalid or missing `agents` list"):
            tools_module._require_sequence({}, "agents")
        with self.assertRaisesRegex(tools_module.FtryCliError, "expected a positive integer"):
            tools_module._require_positive_int(0, "termination.max-turns", "team")

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

            with patch("ftry.Tools.Path.cwd", return_value=cwd_dir):
                self.assertEqual(tools_module._find_dotenv_path(config_path), cwd_dotenv)

            cwd_dotenv.unlink()

            with patch("ftry.Tools.Path.cwd", return_value=cwd_dir):
                self.assertEqual(tools_module._find_dotenv_path(config_path), parent_dotenv)

            loaded_calls: list[tuple[Path, bool]] = []

            def fake_load_dotenv(*, dotenv_path: Path, override: bool) -> bool:
                loaded_calls.append((dotenv_path, override))
                return True

            with (
                patch("ftry.Tools.Path.cwd", return_value=cwd_dir),
                patch("ftry.Tools._load_dotenv_function", return_value=fake_load_dotenv),
            ):
                tools_module._load_dotenv_for_config(config_path)

            self.assertEqual(loaded_calls, [(parent_dotenv, False)])

            parent_dotenv.unlink()
            with patch("ftry.Tools.Path.cwd", return_value=cwd_dir):
                self.assertIsNone(tools_module._find_dotenv_path(config_path))

    def test_secret_helpers_cover_success_and_error_cases(self) -> None:
        self.assertEqual(tools_module._resolve_secret("literal-secret"), "literal-secret")

        with patch.dict(os.environ, {"OAI_API_KEY": "env-secret"}, clear=False):
            self.assertEqual(tools_module._resolve_secret("env:OAI_API_KEY"), "env-secret")

        with self.assertRaisesRegex(tools_module.FtryCliError, "environment variable name is missing"):
            tools_module._resolve_secret("env:   ")
        with self.assertRaisesRegex(tools_module.FtryCliError, "Environment variable `UNSET_KEY` is not set."):
            tools_module._resolve_secret("env:UNSET_KEY")

    def test_message_and_output_helpers_render_expected_text(self) -> None:
        class FallbackPayload:
            def __str__(self) -> str:
                return "fallback payload"

        text_message = types.SimpleNamespace(text="  direct answer  ")
        content_message = types.SimpleNamespace(contents=[types.SimpleNamespace(text="Hello"), " world"])
        empty_message = types.SimpleNamespace()

        self.assertEqual(tools_module._extract_message_text(text_message), "direct answer")
        self.assertEqual(tools_module._extract_message_text(content_message), "Hello world")
        self.assertEqual(tools_module._extract_message_text(empty_message), "")

        messages = [types.SimpleNamespace(role="assistant", text="Hi", author_name="agent-1")]
        payload = types.SimpleNamespace(messages=messages)
        author_name_map = {"agent-1": "Prompter", "team-1": "Better Prompt team"}

        self.assertEqual(tools_module._extract_messages(messages), messages)
        self.assertEqual(tools_module._extract_messages(payload), messages)
        self.assertEqual(tools_module._extract_messages(object()), [])

        self.assertEqual(tools_module._display_name(None), "unknown")
        self.assertEqual(tools_module._display_name("agent-1"), "agent-1")
        self.assertEqual(tools_module._display_name("agent-1", author_name_map), "Prompter")
        self.assertEqual(tools_module._display_name("missing", author_name_map), "missing")

        self.assertEqual(tools_module._summarize_payload(types.SimpleNamespace(text=" Direct text ")), "Direct text")
        summarized = tools_module._summarize_payload(
            types.SimpleNamespace(
                messages=[
                    types.SimpleNamespace(role="user", text="ignore me", author_name="user"),
                    types.SimpleNamespace(role="assistant", text="Draft prompt", author_name="agent-1"),
                ]
            ),
            author_name_map=author_name_map,
        )
        self.assertEqual(summarized, "[Prompter] Draft prompt")
        self.assertEqual(tools_module._summarize_payload(FallbackPayload()), "fallback payload")

        self.assertEqual(tools_module._extract_trace_chunk(types.SimpleNamespace(text="Raw chunk")), "Raw chunk")
        chunk = tools_module._extract_trace_chunk(
            types.SimpleNamespace(
                messages=[
                    types.SimpleNamespace(role="assistant", text="One", author_name="agent-1"),
                    types.SimpleNamespace(role="assistant", contents=["Two"], author_name="agent-1"),
                ]
            )
        )
        self.assertEqual(chunk, "One\n\nTwo")
        self.assertEqual(tools_module._extract_trace_chunk(FallbackPayload()), "fallback payload")

        formatted_agent = tools_module._format_agent_output(
            types.SimpleNamespace(
                messages=[
                    types.SimpleNamespace(role="assistant", text="Done", author_name="agent-1"),
                    types.SimpleNamespace(role="user", text="ignored"),
                ]
            ),
            author_name_map=author_name_map,
        )
        self.assertEqual(formatted_agent, "[Prompter]\nDone")
        self.assertEqual(tools_module._format_agent_output(FallbackPayload()), "fallback payload")

        final_output = tools_module._format_final_team_output(
            types.SimpleNamespace(
                messages=[
                    types.SimpleNamespace(role="assistant", text="Draft", author_name="agent-1"),
                    types.SimpleNamespace(role="assistant", text="Final", author_name="team-1"),
                ]
            ),
            author_name_map=author_name_map,
        )
        self.assertEqual(final_output, "[Better Prompt team]\nFinal")
        self.assertEqual(tools_module._format_final_team_output(types.SimpleNamespace(text="fallback")), "fallback")
        self.assertIsNone(tools_module._resolve_message_author(types.SimpleNamespace(text="No author")))
        self.assertIsNone(tools_module._render_message(types.SimpleNamespace()))
        self.assertEqual(tools_module._render_message(types.SimpleNamespace(text="Anonymous answer")), "Anonymous answer")

    def test_load_yaml_mapping_supports_current_sample_layout_and_validates_kinds(self) -> None:
        agent_config = tools_module._load_yaml_mapping(SAMPLE_AGENT_FILE, config_kind="agent")
        team_config = tools_module._load_yaml_mapping(SAMPLE_TEAM_FILE, config_kind="team")

        self.assertEqual(agent_config["name"], "Poete")
        self.assertEqual(team_config["name"], "Better Prompt team")

        with self.assertRaisesRegex(tools_module.FtryCliError, "Invalid agent YAML"):
            tools_module._load_yaml_mapping(SAMPLE_TEAM_FILE, config_kind="agent")
        with self.assertRaisesRegex(tools_module.FtryCliError, "Invalid team YAML"):
            tools_module._load_yaml_mapping(SAMPLE_AGENT_FILE, config_kind="team")

    def test_trace_and_yaml_helpers_cover_expected_cases(self) -> None:
        self.assertIsNone(tools_module._detect_yaml_config_kind({"name": "Config", "prompt": "Bonjour"}))
        self.assertEqual(tools_module._sanitize_agent_name("Better Prompt team"), "Better-Prompt-team")
        self.assertEqual(tools_module._sanitize_agent_name("agent/triage<v1>"), "agent-triage-v1")
        self.assertEqual(tools_module._sanitize_agent_name("   "), "agent")
        self.assertEqual(tools_module._trace_block("a\nb"), "\n\ta\n\tb")
        self.assertEqual(
            tools_module._summarize_trace_text("# Type of problemCe sujet demande un poeme."),
            "# Type of problem\nCe sujet demande un poeme.",
        )

        colors = tools_module._build_agent_trace_colors(["Prompter", "Reviewer", "Runner", "Prompter"])
        self.assertEqual(colors["Prompter"], tools_module.BRIGHT_PINK)
        self.assertEqual(colors["Reviewer"], tools_module.BRIGHT_BLUE)
        self.assertEqual(colors["Runner"], tools_module.PURPLE)

    def test_summarize_trace_text_truncates_and_normalizes_whitespace(self) -> None:
        long_text = "Line 1\r\n\r\n\r\n" + ("x" * 260)
        summarized = tools_module._summarize_trace_text(long_text, max_length=40)

        self.assertEqual(summarized[:6], "Line 1")
        self.assertTrue(summarized.endswith("..."))
        self.assertLessEqual(len(summarized), 40)

    def test_resolve_config_path_prefers_existing_relative_files_before_base_dir(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            existing_file = temp_path / "agent.yaml"
            existing_file.write_text("name: Agent\n", encoding="utf-8")
            base_dir = temp_path / "configs"
            base_dir.mkdir()

            previous_cwd = Path.cwd()
            os.chdir(temp_path)
            try:
                resolved = tools_module._resolve_config_path("agent.yaml", base_dir=base_dir)
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(resolved, Path("agent.yaml"))


if __name__ == "__main__":
    unittest.main()
