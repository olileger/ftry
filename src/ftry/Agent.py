from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .Tools import (
    FtryCliError,
    _format_agent_output,
    _load_dotenv_for_config,
    _load_yaml_mapping,
    _require_mapping,
    _require_non_empty_string,
    _require_optional_string,
    _resolve_config_path,
    _resolve_secret,
    _sanitize_agent_name,
    _trace_agent_output,
    _trace_agent_start,
)


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
    description: str | None = None


_OPENAI_PROVIDER = "openai"


def _parse_model_config(raw_model: Any, *, config_kind: str, required: bool) -> AgentModelConfig | None:
    if raw_model is None and not required:
        return None

    model_config = _require_mapping(raw_model, "model", config_kind)
    return AgentModelConfig(
        name=_require_non_empty_string(model_config.get("name"), "model.name", config_kind),
        provider=_require_non_empty_string(model_config.get("provider"), "model.provider", config_kind),
        api_key=_resolve_secret(_require_non_empty_string(model_config.get("api-key"), "model.api-key", config_kind)),
    )


def _parse_agent_config(config: Mapping[str, Any], *, config_kind: str) -> AgentConfig:
    model = _parse_model_config(config.get("model"), config_kind=config_kind, required=True)
    assert model is not None
    return AgentConfig(
        name=_require_non_empty_string(config.get("name"), "name", config_kind),
        description=_require_optional_string(config.get("description"), "description", config_kind),
        instructions=_require_non_empty_string(config.get("prompt"), "prompt", config_kind),
        model=model,
    )


def _load_agent_config(
    agent_file: str | Path,
    *,
    base_dir: Path | None = None,
    resolve_config_path: Callable[[str | Path], Path] | Callable[..., Path] = _resolve_config_path,
    load_dotenv_for_config: Callable[[Path], None] = _load_dotenv_for_config,
    load_yaml_mapping: Callable[[Path], Mapping[str, Any]] | Callable[..., Mapping[str, Any]] = _load_yaml_mapping,
) -> AgentConfig:
    agent_path = resolve_config_path(agent_file, base_dir=base_dir)
    if not agent_path.is_file():
        raise FtryCliError(f"Agent file not found: {agent_path}")

    load_dotenv_for_config(agent_path)
    return _parse_agent_config(load_yaml_mapping(agent_path, config_kind="agent"), config_kind="agent")


def _create_openai_agent(
    config: AgentConfig,
    *,
    extra_instructions: str | None = None,
    name_override: str | None = None,
) -> Any:
    try:
        from agent_framework.openai import OpenAIChatCompletionClient
    except ImportError as exc:  # pragma: no cover - covered by CLI error path
        raise FtryCliError(
            "Microsoft Agent Framework OpenAI support is required for `ftry pop`. "
            "Reinstall the project with `python -m pip install -e .`."
        ) from exc

    instructions = config.instructions
    if extra_instructions:
        instructions = f"{instructions}\n\n<TeamContext>\n{extra_instructions}\n</TeamContext>"

    return OpenAIChatCompletionClient(
        model=config.model.name,
        api_key=config.model.api_key,
    ).as_agent(
        name=name_override or _sanitize_agent_name(config.name),
        description=config.description,
        instructions=instructions,
    )


async def _run_openai_agent(config: AgentConfig, prompt: str) -> str:
    agent = _create_openai_agent(config)
    _trace_agent_start(config.name, prompt)
    result = await agent.run(prompt)
    rendered_output = _format_agent_output(result)
    _trace_agent_output(config.name, rendered_output, field_name="final-output")
    return rendered_output


async def _run_agent_prompt(config: AgentConfig, prompt: str) -> str:
    provider = config.model.provider.lower()
    if provider != _OPENAI_PROVIDER:
        raise FtryCliError(
            f"Unsupported provider `{config.model.provider}`. Only `{_OPENAI_PROVIDER}` is supported for now."
        )

    return await _run_openai_agent(config, prompt)
