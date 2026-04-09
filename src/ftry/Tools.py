from __future__ import annotations

import logging
import os
import re
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RESET = "\033[0m"
BOLD = "\033[1m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_PINK = "\033[95m"
PURPLE = "\033[38;5;141m"
ORANGE = "\033[38;5;214m"
TRACE_LOGGER_NAME = "ftry.trace"
AGENT_TRACE_COLORS = (BRIGHT_PINK, BRIGHT_BLUE, PURPLE, ORANGE, BRIGHT_GREEN, BRIGHT_YELLOW)

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


def _load_dotenv_for_config(config_path: Path) -> None:
    dotenv_path = _find_dotenv_path(config_path)
    if dotenv_path is None:
        return

    load_dotenv = _load_dotenv_function()
    load_dotenv(dotenv_path=dotenv_path, override=False)


def _require_non_empty_string(value: Any, field_name: str, config_kind: str = "agent") -> str:
    if not isinstance(value, str) or not value.strip():
        raise FtryCliError(f"Invalid or missing `{field_name}` in {config_kind} YAML.")
    return value.strip()


def _require_optional_string(value: Any, field_name: str, config_kind: str = "agent") -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name, config_kind)


def _require_mapping(value: Any, field_name: str, config_kind: str = "agent") -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise FtryCliError(f"Invalid or missing `{field_name}` mapping in {config_kind} YAML.")
    return value


def _require_sequence(value: Any, field_name: str, config_kind: str = "agent") -> Sequence[Any]:
    if not isinstance(value, list):
        raise FtryCliError(f"Invalid or missing `{field_name}` list in {config_kind} YAML.")
    return value


def _require_positive_int(value: Any, field_name: str, config_kind: str = "agent") -> int:
    if not isinstance(value, int) or value <= 0:
        raise FtryCliError(f"Invalid or missing `{field_name}` in {config_kind} YAML: expected a positive integer.")
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


def _sanitize_agent_name(value: str, *, fallback_prefix: str = "agent") -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip())
    sanitized = sanitized.strip("-_")
    return sanitized or fallback_prefix


def _ensure_trace_logger() -> logging.Logger:
    logger = logging.getLogger(TRACE_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stderr)

    return logger


def _trace(message: str, *args: object) -> None:
    _ensure_trace_logger().info(message, *args)


def _colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _trace_block(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    indented = normalized.replace("\n", "\n\t")
    return f"\n\t{indented}"


def _display_name(name: str | None, author_name_map: Mapping[str, str] | None = None) -> str:
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    if author_name_map is None:
        return name
    return author_name_map.get(name, name)


def _build_agent_trace_colors(agent_names: Sequence[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    palette_size = len(AGENT_TRACE_COLORS)
    for index, agent_name in enumerate(agent_names):
        colors.setdefault(agent_name, AGENT_TRACE_COLORS[index % palette_size])
    return colors


def _trace_team_label(team_name: str) -> str:
    return _colorize(f"TEAM {team_name}", f"{BOLD}{BRIGHT_CYAN}")


def _trace_node_label(
    name: str,
    *,
    team_name: str,
    agent_trace_colors: Mapping[str, str],
) -> str:
    if name == team_name:
        return _trace_team_label(name)
    return _trace_agent_label(name, agent_trace_colors)


def _trace_agent_label(agent_name: str, agent_trace_colors: Mapping[str, str]) -> str:
    return _colorize(agent_name, agent_trace_colors.get(agent_name, BRIGHT_PINK))


def _trace_team_start(team_name: str, pattern: str, prompt: str) -> None:
    _trace('%s | pattern: %s | input:%s', _trace_team_label(team_name), pattern, _trace_block(prompt))


def _trace_route(
    source_name: str,
    target_name: str,
    payload: str,
    *,
    team_name: str,
    agent_trace_colors: Mapping[str, str],
) -> None:
    _trace(
        '%s %s %s | input:%s',
        _trace_node_label(source_name, team_name=team_name, agent_trace_colors=agent_trace_colors),
        _colorize("-->", BRIGHT_YELLOW),
        _trace_node_label(target_name, team_name=team_name, agent_trace_colors=agent_trace_colors),
        _trace_block(payload),
    )


def _trace_result(
    receiver_name: str,
    producer_name: str,
    payload: str,
    *,
    team_name: str,
    agent_trace_colors: Mapping[str, str],
    field_name: str = "output",
) -> None:
    _trace(
        '%s %s %s | %s:%s',
        _trace_node_label(receiver_name, team_name=team_name, agent_trace_colors=agent_trace_colors),
        _colorize("<--", BRIGHT_GREEN),
        _trace_node_label(producer_name, team_name=team_name, agent_trace_colors=agent_trace_colors),
        field_name,
        _trace_block(payload),
    )


def _trace_agent_start(agent_name: str, prompt: str) -> None:
    agent_trace_colors = _build_agent_trace_colors([agent_name])
    _trace('%s | input:%s', _colorize(f"AGENT {agent_name}", f"{BOLD}{agent_trace_colors[agent_name]}"), _trace_block(prompt))


def _trace_agent_output(agent_name: str, output: str) -> None:
    agent_trace_colors = _build_agent_trace_colors([agent_name])
    _trace('%s | output:%s', _colorize(f"AGENT {agent_name}", f"{BOLD}{agent_trace_colors[agent_name]}"), _trace_block(output))


def _extract_message_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    contents = getattr(message, "contents", None)
    if not isinstance(contents, list):
        return ""

    rendered_parts: list[str] = []
    for content in contents:
        if isinstance(content, str):
            rendered_parts.append(content)
            continue

        content_text = getattr(content, "text", None)
        if isinstance(content_text, str):
            rendered_parts.append(content_text)

    return "".join(rendered_parts).strip()


def _extract_messages(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload

    messages = getattr(payload, "messages", None)
    if isinstance(messages, list):
        return messages

    return []


def _summarize_trace_text(text: str, *, max_length: int = 240) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?m)(^#{1,6} [^\n]*[a-zà-ÿ])(?=[A-ZÀ-ÖØ-Þ])", r"\1\n", normalized)
    normalized = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3]}..."


def _summarize_payload(payload: Any, *, author_name_map: Mapping[str, str] | None = None) -> str:
    text = getattr(payload, "text", None)
    if isinstance(text, str) and text.strip():
        return _summarize_trace_text(text)

    messages = _extract_messages(payload)
    rendered: list[str] = []
    for message in messages:
        message_text = _extract_message_text(message)
        if not message_text or getattr(message, "role", None) == "user":
            continue

        author_name = _display_name(getattr(message, "author_name", None), author_name_map)
        rendered.append(f"[{author_name}] {message_text}")

    if rendered:
        return _summarize_trace_text("\n\n".join(rendered))

    return _summarize_trace_text(str(payload))


def _extract_trace_chunk(payload: Any) -> str:
    text = getattr(payload, "text", None)
    if isinstance(text, str):
        return text

    messages = _extract_messages(payload)
    rendered: list[str] = []
    for message in messages:
        if getattr(message, "role", None) == "user":
            continue
        message_text = _extract_message_text(message)
        if message_text:
            rendered.append(message_text)

    if rendered:
        return "\n\n".join(rendered)

    return str(payload)


def _format_agent_output(result: object, *, author_name_map: Mapping[str, str] | None = None) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    messages = _extract_messages(result)
    if messages:
        rendered_messages: list[str] = []
        for message in messages:
            message_text = _extract_message_text(message)
            message_role = getattr(message, "role", None)
            if not message_text or message_role == "user":
                continue

            author_name = getattr(message, "author_name", None)
            if isinstance(author_name, str) and author_name.strip():
                display_name = author_name_map.get(author_name.strip(), author_name.strip()) if author_name_map else author_name.strip()
                rendered_messages.append(f"[{display_name}]\n{message_text.strip()}")
            else:
                rendered_messages.append(message_text.strip())

        if rendered_messages:
            return "\n\n".join(rendered_messages)
    return str(result)


def _format_final_team_output(payload: Any, *, author_name_map: Mapping[str, str] | None = None) -> str:
    messages = _extract_messages(payload)
    if messages:
        for message in reversed(messages):
            if getattr(message, "role", None) == "user":
                continue
            message_text = _extract_message_text(message)
            if not message_text:
                continue
            author_name = getattr(message, "author_name", None)
            if isinstance(author_name, str) and author_name.strip():
                display_name = _display_name(author_name.strip(), author_name_map)
                return f"[{display_name}]\n{message_text}"
            return message_text

    return _format_agent_output(payload, author_name_map=author_name_map)


def _resolve_config_path(config_file: str | Path, *, base_dir: Path | None = None) -> Path:
    config_path = Path(config_file).expanduser()
    if config_path.is_absolute() or base_dir is None:
        return config_path

    if config_path.is_file():
        return config_path

    config_path = base_dir / config_path
    return config_path


def _load_yaml_mapping(config_path: Path, *, config_kind: str) -> Mapping[str, Any]:
    yaml = _load_yaml_module()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _require_mapping(raw_config, "root", config_kind)
