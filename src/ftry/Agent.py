from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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

    @classmethod
    def from_mapping(
        cls,
        raw_model: Any,
        *,
        config_kind: str,
        required: bool = True,
    ) -> AgentModelConfig | None:
        if raw_model is None and not required:
            return None

        model_config = _require_mapping(raw_model, "model", config_kind)
        return cls(
            name=_require_non_empty_string(model_config.get("name"), "model.name", config_kind),
            provider=_require_non_empty_string(model_config.get("provider"), "model.provider", config_kind),
            api_key=_resolve_secret(_require_non_empty_string(model_config.get("api-key"), "model.api-key", config_kind)),
        )


@dataclass(frozen=True)
class AgentConfig:
    name: str
    instructions: str
    model: AgentModelConfig
    description: str | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], *, config_kind: str = "agent") -> AgentConfig:
        model = AgentModelConfig.from_mapping(config.get("model"), config_kind=config_kind, required=True)
        assert model is not None
        return cls(
            name=_require_non_empty_string(config.get("name"), "name", config_kind),
            description=_require_optional_string(config.get("description"), "description", config_kind),
            instructions=_require_non_empty_string(config.get("prompt"), "prompt", config_kind),
            model=model,
        )

    @classmethod
    def from_file(cls, agent_file: str | Path, *, base_dir: Path | None = None) -> AgentConfig:
        agent_path = _resolve_config_path(agent_file, base_dir=base_dir)
        if not agent_path.is_file():
            raise FtryCliError(f"Agent file not found: {agent_path}")

        _load_dotenv_for_config(agent_path)
        return cls.from_mapping(_load_yaml_mapping(agent_path, config_kind="agent"), config_kind="agent")


_OPENAI_PROVIDER = "openai"


class Agent:
    def __init__(self, config: AgentConfig):
        self._config = config

    @classmethod
    def from_file(cls, agent_file: str | Path, *, base_dir: Path | None = None) -> Agent:
        return cls(AgentConfig.from_file(agent_file, base_dir=base_dir))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], *, config_kind: str = "agent") -> Agent:
        return cls(AgentConfig.from_mapping(config, config_kind=config_kind))

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def description(self) -> str | None:
        return self._config.description

    @property
    def instructions(self) -> str:
        return self._config.instructions

    @property
    def model(self) -> AgentModelConfig:
        return self._config.model

    def create_participant(
        self,
        *,
        extra_instructions: str | None = None,
        name_override: str | None = None,
        require_per_service_call_history_persistence: bool = False,
    ) -> Any:
        self._require_supported_provider()
        return self._create_openai_participant(
            extra_instructions=extra_instructions,
            name_override=name_override,
            require_per_service_call_history_persistence=require_per_service_call_history_persistence,
        )

    async def run(self, prompt: str) -> str:
        participant = self.create_participant()
        _trace_agent_start(self.name, prompt)
        result = await participant.run(prompt)
        rendered_output = _format_agent_output(result)
        _trace_agent_output(self.name, rendered_output, field_name="final-output")
        return rendered_output

    def _require_supported_provider(self) -> None:
        provider = self.model.provider.lower()
        if provider != _OPENAI_PROVIDER:
            raise FtryCliError(
                f"Unsupported provider `{self.model.provider}`. Only `{_OPENAI_PROVIDER}` is supported for now."
            )

    def _build_instructions(self, extra_instructions: str | None) -> str:
        if not extra_instructions:
            return self.instructions
        return f"{self.instructions}\n\n<TeamContext>\n{extra_instructions}\n</TeamContext>"

    def _create_openai_participant(
        self,
        *,
        extra_instructions: str | None = None,
        name_override: str | None = None,
        require_per_service_call_history_persistence: bool = False,
    ) -> Any:
        try:
            from agent_framework.openai import OpenAIChatCompletionClient
        except ImportError as exc:  # pragma: no cover - covered by CLI error path
            raise FtryCliError(
                "Microsoft Agent Framework OpenAI support is required for `ftry pop`. "
                "Reinstall the project with `python -m pip install -e .`."
            ) from exc

        return OpenAIChatCompletionClient(
            model=self.model.name,
            api_key=self.model.api_key,
        ).as_agent(
            name=name_override or _sanitize_agent_name(self.name),
            description=self.description,
            instructions=self._build_instructions(extra_instructions),
            require_per_service_call_history_persistence=require_per_service_call_history_persistence,
        )
