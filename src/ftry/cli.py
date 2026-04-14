"""Command-line interface for ftry."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Callable, Sequence, TextIO

from .Agent import Agent
from .Team import (
    _load_team_config,
    _run_team_prompt,
)
from .Tools import (
    BRIGHT_CYAN,
    BRIGHT_PINK,
    BRIGHT_YELLOW,
    ORANGE,
    PURPLE,
    FtryCliError,
    _colorize,
    _load_line_banner,
    _load_pop_banner,
)


MOCK_COMMANDS = ("build", "break", "land")
POP_ANIMATION_STEP_SECONDS = 0.16
POP_ANIMATION_SKATE_COLORS = (ORANGE, PURPLE, BRIGHT_YELLOW)
POP_ANIMATION_BANNER = _load_pop_banner()
POP_ANIMATION_SKATEBOARD_LINES = (
    "         .  .",
    "         \\______/>",
    "          o    o",
)
POP_ANIMATION_GROUND = "________________________________________________________________________________"
POP_ANIMATION_FRAME_SPECS = (
    (0, 0),
    (5, 0),
    (10, 0),
    (15, 0),
    (20, 0),
    (25, 1),
    (30, 2),
    (36, 1),
    (43, 0),
)
POP_ANIMATION_CLEAR_LINE = "\x1b[2K"
POP_ANIMATION_CURSOR_HIDE = "\x1b[?25l"
POP_ANIMATION_CURSOR_SHOW = "\x1b[?25h"
POP_ANIMATION_CURSOR_UP = "\x1b[1A"


def _run_mock_command(command: str) -> int:
    print(command)
    return 0


def _run_line_command() -> int:
    print(_load_line_banner())
    return 0


def _clear_pop_animation_frame(line_count: int, *, stream: TextIO) -> None:
    for index in range(line_count):
        stream.write(f"\r{POP_ANIMATION_CLEAR_LINE}")
        if index != line_count - 1:
            stream.write(POP_ANIMATION_CURSOR_UP)


def _build_pop_animation_frame(horizontal_offset: int, vertical_lift: int, max_lift: int) -> str:
    skateboard_lines = tuple(
        f"{' ' * horizontal_offset}{_colorize(line, POP_ANIMATION_SKATE_COLORS[index % len(POP_ANIMATION_SKATE_COLORS)])}"
        for index, line in enumerate(POP_ANIMATION_SKATEBOARD_LINES)
    )
    ground_line = _colorize(POP_ANIMATION_GROUND, BRIGHT_CYAN)
    top_padding = ("",) * (max_lift - vertical_lift)
    bottom_padding = ("",) * vertical_lift
    return "\n".join((POP_ANIMATION_BANNER, "", *top_padding, *skateboard_lines, *bottom_padding, ground_line))


POP_ANIMATION_MAX_LIFT = max(vertical_lift for _, vertical_lift in POP_ANIMATION_FRAME_SPECS)
POP_ANIMATION_FRAMES = tuple(
    _build_pop_animation_frame(horizontal_offset, vertical_lift, POP_ANIMATION_MAX_LIFT)
    for horizontal_offset, vertical_lift in POP_ANIMATION_FRAME_SPECS
)


def _render_pop_animation(*, stream: TextIO | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
    target_stream = sys.stderr if stream is None else stream
    if not getattr(target_stream, "isatty", lambda: False)():
        return

    line_count = len(POP_ANIMATION_FRAMES[0].splitlines())
    frame_was_rendered = False
    target_stream.write(POP_ANIMATION_CURSOR_HIDE)
    target_stream.flush()

    try:
        for frame in POP_ANIMATION_FRAMES:
            if frame_was_rendered:
                _clear_pop_animation_frame(line_count, stream=target_stream)
            target_stream.write(frame)
            target_stream.flush()
            frame_was_rendered = True
            sleep(POP_ANIMATION_STEP_SECONDS)

        if frame_was_rendered:
            target_stream.write("\n")
            target_stream.flush()
    finally:
        target_stream.write(POP_ANIMATION_CURSOR_SHOW)
        target_stream.flush()


def _run_pop_command(agent_file: str | None, team_file: str | None, prompt: str) -> int:
    if team_file is not None:
        config = _load_team_config(team_file)
        _render_pop_animation()
        asyncio.run(_run_team_prompt(config, prompt))
        return 0

    if agent_file is None:
        raise FtryCliError("Either `-a/--agent-file` or `-t/--team-file` must be provided.")
    agent = Agent.from_file(agent_file)
    _render_pop_animation()
    asyncio.run(agent.run(prompt))
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
