"""Command-line interface for ftry."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .Agent import (
    AgentConfig,
    AgentModelConfig,
    _create_openai_agent,
    _load_agent_config as _agent_load_agent_config,
    _parse_agent_config,
    _parse_model_config,
    _run_agent_prompt,
    _run_openai_agent,
)
from .Team import (
    TeamConfig,
    TeamTerminationConfig,
    _build_team_participants,
    _build_team_workflow,
    _compose_pattern_analysis_text,
    _contains_any,
    _count_assistant_messages,
    _create_team_controller_agent,
    _has_numbered_steps,
    _infer_team_pattern,
    _load_orchestration_builders,
    _load_team_agent_config as _team_load_team_agent_config,
    _load_team_config as _team_load_team_config,
    _normalize_team_pattern,
    _parse_team_termination,
    _render_role_summary,
    _render_team_instructions,
    _run_team_prompt,
    _select_handoff_start_agent,
)
from .Tools import (
    AGENT_TRACE_COLORS,
    BOLD,
    BRIGHT_BLUE,
    BRIGHT_CYAN,
    BRIGHT_GREEN,
    BRIGHT_PINK,
    BRIGHT_YELLOW,
    LINE_COLOR_TOKENS,
    ORANGE,
    PURPLE,
    RESET,
    FtryCliError,
    _build_agent_trace_colors,
    _colorize,
    _display_name,
    _ensure_trace_logger,
    _extract_message_text,
    _extract_messages,
    _extract_trace_chunk,
    _format_agent_output,
    _format_final_team_output,
    _load_dotenv_function,
    _load_line_banner,
    _load_yaml_mapping,
    _load_yaml_module,
    _require_mapping,
    _require_non_empty_string,
    _require_optional_string,
    _require_positive_int,
    _require_sequence,
    _resolve_config_path,
    _resolve_secret,
    _sanitize_agent_name,
    _summarize_payload,
    _summarize_trace_text,
    _trace,
    _trace_agent_label,
    _trace_agent_output,
    _trace_agent_start,
    _trace_block,
    _trace_node_label,
    _trace_result,
    _trace_route,
    _trace_team_label,
    _trace_team_start,
)


MOCK_COMMANDS = ("build", "break", "land")


def _run_mock_command(command: str) -> int:
    print(command)
    return 0


def _run_line_command() -> int:
    print(_load_line_banner())
    return 0


def _find_dotenv_path(agent_path: Path) -> Path | None:
    search_roots = [Path.cwd(), agent_path.resolve().parent]

    seen: set[Path] = set()
    for root in search_roots:
        for candidate_dir in (root, *root.parents):
            if candidate_dir in seen:
                continue
            seen.add(candidate_dir)

            candidate = candidate_dir / ".env"
            if candidate.is_file():
                return candidate

    return None


def _load_dotenv_for_config(config_path: Path) -> None:
    dotenv_path = _find_dotenv_path(config_path)
    if dotenv_path is None:
        return

    load_dotenv = _load_dotenv_function()
    load_dotenv(dotenv_path=dotenv_path, override=False)


def _load_agent_config(agent_file: str | Path, *, base_dir: Path | None = None) -> AgentConfig:
    return _agent_load_agent_config(
        agent_file,
        base_dir=base_dir,
        resolve_config_path=_resolve_config_path,
        load_dotenv_for_config=_load_dotenv_for_config,
        load_yaml_mapping=_load_yaml_mapping,
    )


def _load_team_agent_config(raw_agent: Any, *, team_dir: Path) -> AgentConfig:
    return _team_load_team_agent_config(
        raw_agent,
        team_dir=team_dir,
        load_agent_config=_load_agent_config,
        parse_agent_config=_parse_agent_config,
    )


def _load_team_config(team_file: str | Path) -> TeamConfig:
    return _team_load_team_config(
        team_file,
        resolve_config_path=_resolve_config_path,
        load_dotenv_for_config=_load_dotenv_for_config,
        load_yaml_mapping=_load_yaml_mapping,
        load_team_agent_config=_load_team_agent_config,
        parse_model_config=_parse_model_config,
    )


def _run_pop_command(agent_file: str | None, team_file: str | None, prompt: str) -> int:
    if team_file is not None:
        config = _load_team_config(team_file)
        asyncio.run(_run_team_prompt(config, prompt))
        return 0
    else:
        if agent_file is None:
            raise FtryCliError("Either `-a/--agent-file` or `-t/--team-file` must be provided.")
        config = _load_agent_config(agent_file)
        result = asyncio.run(_run_agent_prompt(config, prompt))

    print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftry", description="Mock CLI for ftry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in MOCK_COMMANDS:
        subparser = subparsers.add_parser(command, help=f"Mock {command} command.")
        subparser.set_defaults(handler=lambda args, command=command: _run_mock_command(command))

    line_parser = subparsers.add_parser("line", help='Display "First Try" as colored ASCII art.')
    line_parser.set_defaults(handler=lambda args: _run_line_command())

    pop_parser = subparsers.add_parser("pop", help="Run an agent described in a YAML file.")
    pop_source_group = pop_parser.add_mutually_exclusive_group(required=True)
    pop_source_group.add_argument(
        "-a",
        "--agent-file",
        help="Path to the YAML agent description file.",
    )
    pop_source_group.add_argument(
        "-t",
        "--team-file",
        help="Path to the YAML team description file.",
    )
    pop_parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="Prompt to send to the loaded agent.",
    )
    pop_parser.set_defaults(handler=lambda args: _run_pop_command(args.agent_file, args.team_file, args.prompt))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler

    try:
        return handler(args)
    except FtryCliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
