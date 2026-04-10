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
AGENT_LABEL_PREFIX = "AGENT "
ELLIPSIS = "..."
ENV_FILE_NAME = ".env"
ENV_SECRET_PREFIX = "env:"
TRACE_INPUT_FIELD = "input"
TRACE_OUTPUT_FIELD = "output"
UNKNOWN_DISPLAY_NAME = "unknown"
USER_ROLE = "user"
YAML_ROOT_FIELD = "root"
AGENT_YAML_ROOT_FIELDS = frozenset({"name", "model", "prompt"})
TEAM_YAML_ROOT_FIELDS = frozenset({"name", "agents", "prompt"})
LINE_RESET_TOKEN = "[reset]"

LINE_COLOR_TOKENS = {
    "[cyan]": BRIGHT_CYAN,
    "[blue]": BRIGHT_BLUE,
    "[pink]": BRIGHT_PINK,
    "[purple]": PURPLE,
    "[yellow]": BRIGHT_YELLOW,
    "[orange]": ORANGE,
    "[green]": BRIGHT_GREEN,
    LINE_RESET_TOKEN: RESET,
}
LINE_COLOR_MARKERS = tuple(token for token in LINE_COLOR_TOKENS if token != LINE_RESET_TOKEN)


class FtryCliError(Exception):
    """Raised when a user-facing CLI error occurs."""


def _load_ascii_banner(file_name: str) -> str:
    content = files("ftry").joinpath(file_name).read_text(encoding="utf-8")
    rendered_lines: list[str] = []

    for raw_line in content.splitlines():
        line = raw_line
        uses_color = any(token in raw_line for token in LINE_COLOR_MARKERS)
        for token, value in LINE_COLOR_TOKENS.items():
            line = line.replace(token, value)
        if uses_color and not line.endswith(RESET):
            line = f"{line}{RESET}"
        rendered_lines.append(line)

    return "\n".join(rendered_lines).rstrip()


def _load_line_banner() -> str:
    return _load_ascii_banner("line.txt")


def _load_pop_banner() -> str:
    return _load_ascii_banner("pop.txt")


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
    for candidate_dir in _iter_unique_parent_dirs((Path.cwd(), agent_path.resolve().parent)):
        candidate = candidate_dir / ENV_FILE_NAME
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
    if not value.startswith(ENV_SECRET_PREFIX):
        return value

    env_name = value.removeprefix(ENV_SECRET_PREFIX).strip()
    if not env_name:
        raise FtryCliError(f"Invalid `model.api-key`: environment variable name is missing after `{ENV_SECRET_PREFIX}`.")

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
    normalized = _normalize_newlines(text)
    indented = normalized.replace("\n", "\n\t")
    return f"\n\t{indented}"


def _display_name(name: str | None, author_name_map: Mapping[str, str] | None = None) -> str:
    if not isinstance(name, str) or not name.strip():
        return UNKNOWN_DISPLAY_NAME
    if author_name_map is None:
        return name
    return author_name_map.get(name, name)


def _build_agent_trace_colors(agent_names: Sequence[str]) -> dict[str, str]:
    palette_size = len(AGENT_TRACE_COLORS)
    unique_agent_names = tuple(dict.fromkeys(agent_names))
    return {
        agent_name: AGENT_TRACE_COLORS[index % palette_size]
        for index, agent_name in enumerate(unique_agent_names)
    }


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
    _trace("%s | pattern: %s | %s:%s", _trace_team_label(team_name), pattern, TRACE_INPUT_FIELD, _trace_block(prompt))


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
    field_name: str = TRACE_OUTPUT_FIELD,
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
    _trace_agent_event(agent_name, TRACE_INPUT_FIELD, prompt)


def _trace_agent_output(agent_name: str, output: str, *, field_name: str = TRACE_OUTPUT_FIELD) -> None:
    _trace_agent_event(agent_name, field_name, output)


def _extract_message_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    contents = getattr(message, "contents", None)
    if not isinstance(contents, list):
        return ""

    return "".join(_iter_message_content_text(contents)).strip()


def _extract_messages(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload

    messages = getattr(payload, "messages", None)
    if isinstance(messages, list):
        return messages

    return []


def _summarize_trace_text(text: str, *, max_length: int = 240) -> str:
    normalized = _normalize_newlines(text)
    normalized = re.sub(r"(?m)(^#{1,6} [^\n]*[a-zà-ÿ])(?=[A-ZÀ-ÖØ-Þ])", r"\1\n", normalized)
    normalized = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - len(ELLIPSIS)]}{ELLIPSIS}"


def _summarize_payload(payload: Any, *, author_name_map: Mapping[str, str] | None = None) -> str:
    text = getattr(payload, "text", None)
    if isinstance(text, str) and text.strip():
        return _summarize_trace_text(text)

    rendered = _render_visible_messages(
        payload,
        author_name_map=author_name_map,
        author_separator=" ",
        include_unknown_author=True,
    )

    if rendered:
        return _summarize_trace_text("\n\n".join(rendered))

    return _summarize_trace_text(str(payload))


def _extract_trace_chunk(payload: Any) -> str:
    text = getattr(payload, "text", None)
    if isinstance(text, str):
        return text

    rendered = _collect_visible_message_texts(payload)

    if rendered:
        return "\n\n".join(rendered)

    return str(payload)


def _format_agent_output(result: object, *, author_name_map: Mapping[str, str] | None = None) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    rendered_messages = _render_visible_messages(result, author_name_map=author_name_map)
    if rendered_messages:
        return "\n\n".join(rendered_messages)
    return str(result)


def _format_final_team_output(payload: Any, *, author_name_map: Mapping[str, str] | None = None) -> str:
    visible_messages = _collect_visible_messages(payload)
    if visible_messages:
        final_message = visible_messages[-1]
        rendered_message = _render_message(final_message, author_name_map=author_name_map)
        if rendered_message is not None:
            return rendered_message

    return _format_agent_output(payload, author_name_map=author_name_map)


def _resolve_config_path(config_file: str | Path, *, base_dir: Path | None = None) -> Path:
    config_path = Path(config_file).expanduser()
    if config_path.is_absolute() or base_dir is None:
        return config_path

    if config_path.is_file():
        return config_path

    config_path = base_dir / config_path
    return config_path


def _detect_yaml_config_kind(config: Mapping[str, Any]) -> str | None:
    root_fields = set(config)
    if TEAM_YAML_ROOT_FIELDS.issubset(root_fields):
        return "team"
    if AGENT_YAML_ROOT_FIELDS.issubset(root_fields) and "agents" not in root_fields:
        return "agent"
    return None


def _validate_yaml_config_kind(config: Mapping[str, Any], *, config_path: Path, config_kind: str) -> None:
    actual_kind = _detect_yaml_config_kind(config)
    if actual_kind is None or actual_kind == config_kind:
        return

    if config_kind == "agent":
        raise FtryCliError(
            f"Invalid agent YAML in `{config_path}`: this file defines `agents` at the root, "
            "so it is a team configuration. Use `-t/--team-file` instead of `-a/--agent-file`."
        )

    raise FtryCliError(
        f"Invalid team YAML in `{config_path}`: this file matches an agent configuration "
        "(`name`, `model`, `prompt`) and does not define `agents` at the root. "
        "Use `-a/--agent-file` instead of `-t/--team-file`."
    )


def _load_yaml_mapping(config_path: Path, *, config_kind: str) -> Mapping[str, Any]:
    yaml = _load_yaml_module()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _require_mapping(raw_config, YAML_ROOT_FIELD, config_kind)
    _validate_yaml_config_kind(config, config_path=config_path, config_kind=config_kind)
    return config


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _iter_unique_parent_dirs(roots: Sequence[Path]) -> Sequence[Path]:
    seen: set[Path] = set()
    unique_dirs: list[Path] = []
    for root in roots:
        for candidate_dir in (root, *root.parents):
            if candidate_dir in seen:
                continue
            seen.add(candidate_dir)
            unique_dirs.append(candidate_dir)
    return unique_dirs


def _trace_agent_event(agent_name: str, field_name: str, payload: str) -> None:
    agent_trace_colors = _build_agent_trace_colors([agent_name])
    agent_label = _colorize(
        f"{AGENT_LABEL_PREFIX}{agent_name}",
        f"{BOLD}{agent_trace_colors[agent_name]}",
    )
    _trace("%s | %s:%s", agent_label, field_name, _trace_block(payload))


def _iter_message_content_text(contents: Sequence[Any]) -> Sequence[str]:
    rendered_parts: list[str] = []
    for content in contents:
        if isinstance(content, str):
            rendered_parts.append(content)
            continue

        content_text = getattr(content, "text", None)
        if isinstance(content_text, str):
            rendered_parts.append(content_text)
    return rendered_parts


def _collect_visible_messages(payload: Any) -> list[Any]:
    return [message for message in _extract_messages(payload) if getattr(message, "role", None) != USER_ROLE]


def _collect_visible_message_texts(payload: Any) -> list[str]:
    return [
        message_text
        for message in _collect_visible_messages(payload)
        if (message_text := _extract_message_text(message))
    ]


def _resolve_message_author(
    message: Any,
    *,
    author_name_map: Mapping[str, str] | None = None,
    include_unknown_author: bool = False,
) -> str | None:
    author_name = getattr(message, "author_name", None)
    if not include_unknown_author and (not isinstance(author_name, str) or not author_name.strip()):
        return None
    return _display_name(author_name, author_name_map)


def _render_message(
    message: Any,
    *,
    author_name_map: Mapping[str, str] | None = None,
    author_separator: str = "\n",
    include_unknown_author: bool = False,
) -> str | None:
    message_text = _extract_message_text(message)
    if not message_text:
        return None

    author_name = _resolve_message_author(
        message,
        author_name_map=author_name_map,
        include_unknown_author=include_unknown_author,
    )
    if author_name is None:
        return message_text
    return f"[{author_name}]{author_separator}{message_text}"


def _render_visible_messages(
    payload: Any,
    *,
    author_name_map: Mapping[str, str] | None = None,
    author_separator: str = "\n",
    include_unknown_author: bool = False,
) -> list[str]:
    rendered_messages: list[str] = []
    for message in _collect_visible_messages(payload):
        rendered_message = _render_message(
            message,
            author_name_map=author_name_map,
            author_separator=author_separator,
            include_unknown_author=include_unknown_author,
        )
        if rendered_message is not None:
            rendered_messages.append(rendered_message)
    return rendered_messages
