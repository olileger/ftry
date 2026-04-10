from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from ftry import Tools


class ToolsTests(unittest.TestCase):
    def test_find_dotenv_path_prefers_first_matching_search_root_and_can_return_none(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cwd_dir = temp_path / "cwd"
            cwd_dir.mkdir()
            config_dir = temp_path / "configs" / "nested"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "agent.yaml"
            config_path.write_text("name: Agent\n", encoding="utf-8")

            parent_dotenv = temp_path / "configs" / ".env"
            parent_dotenv.write_text("FROM=parent\n", encoding="utf-8")

            with patch("ftry.Tools.Path.cwd", return_value=cwd_dir):
                self.assertEqual(Tools._find_dotenv_path(config_path), parent_dotenv)

            parent_dotenv.unlink()

            with patch("ftry.Tools.Path.cwd", return_value=cwd_dir):
                self.assertIsNone(Tools._find_dotenv_path(config_path))

    def test_load_dotenv_for_config_skips_missing_file_and_loads_existing_file(self) -> None:
        config_path = Path(r"C:\repo\agent.yaml")
        load_calls: list[tuple[Path, bool]] = []

        def fake_load_dotenv(*, dotenv_path: Path, override: bool) -> bool:
            load_calls.append((dotenv_path, override))
            return True

        with patch("ftry.Tools._find_dotenv_path", return_value=None), patch("ftry.Tools._load_dotenv_function") as loader:
            Tools._load_dotenv_for_config(config_path)
            loader.assert_not_called()

        dotenv_path = Path(r"C:\repo\.env")
        with (
            patch("ftry.Tools._find_dotenv_path", return_value=dotenv_path),
            patch("ftry.Tools._load_dotenv_function", return_value=fake_load_dotenv),
        ):
            Tools._load_dotenv_for_config(config_path)

        self.assertEqual(load_calls, [(dotenv_path, False)])

    def test_display_name_returns_input_when_no_mapping_is_provided(self) -> None:
        self.assertEqual(Tools._display_name("Reviewer"), "Reviewer")

    def test_summarize_payload_uses_direct_text_and_string_fallback(self) -> None:
        direct_payload = SimpleNamespace(text="  Direct response  ")
        self.assertEqual(Tools._summarize_payload(direct_payload), "Direct response")

        fallback_payload = SimpleNamespace(value=42)
        self.assertEqual(Tools._summarize_payload(fallback_payload), str(fallback_payload))

    def test_extract_trace_chunk_uses_text_and_string_fallback(self) -> None:
        direct_payload = SimpleNamespace(text="raw chunk")
        self.assertEqual(Tools._extract_trace_chunk(direct_payload), "raw chunk")

        fallback_payload = SimpleNamespace(value=42)
        self.assertEqual(Tools._extract_trace_chunk(fallback_payload), str(fallback_payload))

    def test_format_agent_output_falls_back_to_string_representation(self) -> None:
        result = SimpleNamespace(value=42)
        self.assertEqual(Tools._format_agent_output(result), str(result))

    def test_iter_unique_parent_dirs_deduplicates_overlapping_roots(self) -> None:
        root = Path(r"C:\workspace\project")
        roots = (root, root / "child")

        unique_dirs = Tools._iter_unique_parent_dirs(roots)

        self.assertEqual(unique_dirs[0], root)
        self.assertEqual(unique_dirs[1], root.parent)
        self.assertEqual(len(unique_dirs), len(set(unique_dirs)))
        self.assertIn(root / "child", unique_dirs)
        self.assertIn(Path(root.anchor), unique_dirs)

    def test_resolve_message_author_returns_none_without_author_unless_unknowns_are_requested(self) -> None:
        message = SimpleNamespace(role="assistant", text="Done")

        self.assertIsNone(Tools._resolve_message_author(message))
        self.assertEqual(Tools._resolve_message_author(message, include_unknown_author=True), Tools.UNKNOWN_DISPLAY_NAME)

    def test_render_message_handles_empty_text_and_missing_author(self) -> None:
        empty_message = SimpleNamespace(role="assistant")
        self.assertIsNone(Tools._render_message(empty_message))

        anonymous_message = SimpleNamespace(role="assistant", text="Done")
        self.assertEqual(Tools._render_message(anonymous_message), "Done")


if __name__ == "__main__":
    unittest.main()
