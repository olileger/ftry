"""Command-line interface for ftry."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Callable, Sequence, TextIO

from .Builder import build_from_prompt
from .StandaloneAgent import StandaloneAgent
from .Team import (
    Team,
)
from .Tools import (
    FtryCliError,
    _load_build_banner,
    _load_line_banner,
    _load_pop_banner,
)

AGENT_INPUT_PROMPT = "You> "


def _run_line_command() -> int:
    print(_load_line_banner())
    return 0


def _run_build_command(prompt: str, *, output_dir: str | None = None) -> int:
    _print_build_banner()
    created_files = build_from_prompt(
        prompt,
        output_dir=output_dir,
        user_input_provider=_read_agent_follow_up_input,
    )
    print("\n".join(str(path) for path in created_files))
    return 0


def _print_banner(banner_text: str, *, stream: TextIO | None = None) -> None:
    target_stream = sys.stderr if stream is None else stream
    if not getattr(target_stream, "isatty", lambda: False)():
        return
    target_stream.write(f"{banner_text}\n")
    target_stream.flush()


def _print_build_banner(*, stream: TextIO | None = None) -> None:
    _print_banner(_load_build_banner(), stream=stream)


def _print_pop_banner(*, stream: TextIO | None = None) -> None:
    _print_banner(_load_pop_banner(), stream=stream)


def _read_agent_follow_up_input(
    _: str,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> str:
    target_input = sys.stdin if input_stream is None else input_stream
    target_output = sys.stderr if output_stream is None else output_stream
    if not getattr(target_input, "isatty", lambda: False)() or not getattr(target_output, "isatty", lambda: False)():
        raise FtryCliError(
            "Interactive request/response conversations require an interactive terminal on stdin and stderr for `ftry pop` and `ftry build`."
        )

    target_output.write(f"{AGENT_INPUT_PROMPT}")
    target_output.flush()
    response = target_input.readline()
    if response == "":
        raise FtryCliError("Interactive agent conversation ended before the next user input was provided.")
    return response.rstrip("\r\n")


def _run_pop_command(agent_file: str | None, team_file: str | None, prompt: str) -> int:
    _print_pop_banner()
    if team_file is not None:
        team = Team.from_file(team_file)
        asyncio.run(team.run(prompt, user_input_provider=_read_agent_follow_up_input))
        return 0

    if agent_file is None:
        raise FtryCliError("Either `-a/--agent-file` or `-t/--team-file` must be provided.")
    agent = StandaloneAgent.from_file(agent_file)
    asyncio.run(agent.run(prompt, user_input_provider=_read_agent_follow_up_input))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftry", description="Mock CLI for ftry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Generate an agent or team YAML configuration from a prompt.")
    build_parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="Prompt describing what should be built.",
    )
    build_parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory where the generated YAML files should be written. Defaults to the current directory.",
    )
    build_parser.set_defaults(handler=lambda args: _run_build_command(args.prompt, output_dir=args.output_dir))

    line_parser = subparsers.add_parser("line", help='Display "First Try" as colored ASCII art.')
    line_parser.set_defaults(handler=lambda args: _run_line_command())

    pop_parser = subparsers.add_parser("pop", help="Run an agent or team described in a YAML file.")
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
