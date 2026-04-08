"""Command-line interface for ftry."""

from __future__ import annotations

import asyncio
import argparse
import os
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RESET = "\033[0m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_PINK = "\033[95m"
PURPLE = "\033[38;5;141m"
ORANGE = "\033[38;5;214m"

MOCK_COMMANDS = ("build", "break", "land")
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


class FtryCliError(Exception):
    """Raised when a user-facing CLI error occurs."""


@dataclass(frozen=True)
class AgentModelConfig:
    name: str
    provider: str
    api_key: str


@dataclass(frozen=True)
class AgentConfig:
    name: str
    instructions: str
    model: AgentModelConfig


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


def _load_yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - covered by CLI error path
        raise FtryCliError(
            "PyYAML is required for `ftry pop`. Reinstall the project with `python -m pip install -e .`."
        ) from exc

    return yaml


def _load_dotenv_function() -> Callable[..., bool]:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - covered by CLI error path
        raise FtryCliError(
            "python-dotenv is required for `ftry pop` to load `.env` files. "
            "Reinstall the project with `python -m pip install -e .`."
        ) from exc

    return load_dotenv


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


def _load_dotenv_for_agent(agent_path: Path) -> None:
    dotenv_path = _find_dotenv_path(agent_path)
    if dotenv_path is None:
        return

    load_dotenv = _load_dotenv_function()
    load_dotenv(dotenv_path=dotenv_path, override=False)


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FtryCliError(f"Invalid or missing `{field_name}` in agent YAML.")
    return value.strip()


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise FtryCliError(f"Invalid or missing `{field_name}` mapping in agent YAML.")
    return value


def _resolve_secret(value: str) -> str:
    if not value.startswith("env:"):
        return value

    env_name = value.removeprefix("env:").strip()
    if not env_name:
        raise FtryCliError("Invalid `model.api-key`: environment variable name is missing after `env:`.")

    secret = os.getenv(env_name)
    if not secret:
        raise FtryCliError(f"Environment variable `{env_name}` is not set.")
    return secret


def _load_agent_config(agent_file: str) -> AgentConfig:
    agent_path = Path(agent_file).expanduser()
    if not agent_path.is_file():
        raise FtryCliError(f"Agent file not found: {agent_path}")

    _load_dotenv_for_agent(agent_path)

    yaml = _load_yaml_module()
    raw_config = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
    config = _require_mapping(raw_config, "root")
    model_config = _require_mapping(config.get("model"), "model")

    return AgentConfig(
        name=_require_non_empty_string(config.get("name"), "name"),
        instructions=_require_non_empty_string(config.get("prompt"), "prompt"),
        model=AgentModelConfig(
            name=_require_non_empty_string(model_config.get("name"), "model.name"),
            provider=_require_non_empty_string(model_config.get("provider"), "model.provider"),
            api_key=_resolve_secret(_require_non_empty_string(model_config.get("api-key"), "model.api-key")),
        ),
    )


def _format_agent_output(result: object) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str) and text:
        return text
    return str(result)


async def _run_openai_agent(config: AgentConfig, prompt: str) -> str:
    try:
        from agent_framework.openai import OpenAIChatCompletionClient
    except ImportError as exc:  # pragma: no cover - covered by CLI error path
        raise FtryCliError(
            "Microsoft Agent Framework OpenAI support is required for `ftry pop`. "
            "Reinstall the project with `python -m pip install -e .`."
        ) from exc

    agent = OpenAIChatCompletionClient(
        model=config.model.name,
        api_key=config.model.api_key,
    ).as_agent(
        name=config.name,
        instructions=config.instructions,
    )
    result = await agent.run(prompt)
    return _format_agent_output(result)


async def _run_agent_prompt(config: AgentConfig, prompt: str) -> str:
    provider = config.model.provider.lower()
    if provider != "openai":
        raise FtryCliError(f"Unsupported provider `{config.model.provider}`. Only `openai` is supported for now.")

    return await _run_openai_agent(config, prompt)


def _run_pop_command(agent_file: str, prompt: str) -> int:
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
    pop_parser.add_argument(
        "-a",
        "--agent-file",
        required=True,
        help="Path to the YAML agent description file.",
    )
    pop_parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="Prompt to send to the loaded agent.",
    )
    pop_parser.set_defaults(handler=lambda args: _run_pop_command(args.agent_file, args.prompt))
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
