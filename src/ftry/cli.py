"""Command-line interface for ftry."""

from __future__ import annotations

import argparse
from typing import Sequence


COMMANDS = ("build", "break", "pop", "land")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftry", description="Mock CLI for ftry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in COMMANDS:
        subparsers.add_parser(command, help=f"Mock {command} command.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
