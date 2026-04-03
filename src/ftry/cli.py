"""Command-line interface for ftry."""

from __future__ import annotations

import argparse
from importlib.resources import files
from typing import Callable, Sequence


RESET = "\033[0m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_PINK = "\033[95m"
PURPLE = "\033[38;5;141m"
ORANGE = "\033[38;5;214m"

MOCK_COMMANDS = ("build", "break", "pop", "land")
LINE_COLOR_TOKENS = {
    "[cyan]": BRIGHT_CYAN,
    "[blue]": BRIGHT_BLUE,
    "[pink]": BRIGHT_PINK,
    "[purple]": PURPLE,
    "[yellow]": BRIGHT_YELLOW,
    "[orange]": ORANGE,
    "[green]": BRIGHT_GREEN,
    "[reset]": RESET,
}


def _load_line_banner() -> str:
    content = files("ftry").joinpath("line.txt").read_text(encoding="utf-8")
    rendered_lines: list[str] = []

    for raw_line in content.splitlines():
        line = raw_line
        uses_color = any(token in raw_line for token in LINE_COLOR_TOKENS if token != "[reset]")
        for token, value in LINE_COLOR_TOKENS.items():
            line = line.replace(token, value)
        if uses_color and not line.endswith(RESET):
            line = f"{line}{RESET}"
        rendered_lines.append(line)

    return "\n".join(rendered_lines).rstrip()


def _run_mock_command(command: str) -> int:
    print(command)
    return 0


def _run_line_command() -> int:
    print(_load_line_banner())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftry", description="Mock CLI for ftry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in MOCK_COMMANDS:
        subparsers.add_parser(command, help=f"Mock {command} command.")

    subparsers.add_parser("line", help='Display "First Try" as colored ASCII art.')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers: dict[str, Callable[[], int]] = {
        **{command: lambda command=command: _run_mock_command(command) for command in MOCK_COMMANDS},
        "line": _run_line_command,
    }
    return handlers[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
