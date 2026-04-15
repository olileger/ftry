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
_AGENT_RESPONSE_STATUS_DONE = "done"
_AGENT_RESPONSE_STATUS_AWAIT_USER_INPUT = "await_user_input"
_AGENT_AWAIT_USER_INPUT_TRACE_FIELD = "output [AWAIT USER INPUT]"
_AGENT_RESPONSE_STATUS_VALUES = (
    _AGENT_RESPONSE_STATUS_DONE,
    _AGENT_RESPONSE_STATUS_AWAIT_USER_INPUT,
)
AGENT_TURN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_turn_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": list(_AGENT_RESPONSE_STATUS_VALUES),
                },
                "message": {
                    "type": "string",
                    "minLength": 1,
                },
            },
            "required": ["status", "message"],
            "additionalProperties": False,
        },
    },
}
AGENT_TURN_CONTROL_PROMPT = """<ConsoleInteractionContract>
Tu dois repondre en respectant le schema JSON fourni par `response_format`.

Regles de controle:
- Mets dans `message` uniquement le texte visible par l'utilisateur.
- Mets `status` a `await_user_input` quand tu attends explicitement la prochaine reponse de l'utilisateur pour continuer.
- Mets `status` a `done` quand ta reponse doit clore l'execution courante de l'agent.
- N'ecris jamais de JSON, de schema, ni d'explication meta dans `message`.
</ConsoleInteractionContract>"""


@dataclass(frozen=True)
class AgentTurnResponse:
    message: str
    status: str

    @property
    def awaits_user_input(self) -> bool:
        return self.status == _AGENT_RESPONSE_STATUS_AWAIT_USER_INPUT


UserInputProvider = Callable[[str], str]


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
        require_structured_turn_response: bool = False,
    ) -> Any:
        self._require_supported_provider()
        return self._create_openai_participant(
            extra_instructions=extra_instructions,
            name_override=name_override,
            require_per_service_call_history_persistence=require_per_service_call_history_persistence,
            require_structured_turn_response=require_structured_turn_response,
        )

    async def run(self, prompt: str, *, user_input_provider: UserInputProvider | None = None) -> str:
        participant = self.create_participant(require_structured_turn_response=True)
        session = participant.create_session()
        pending_prompt = prompt
        _trace_agent_start(self.name, prompt)

        while True:
            result = await participant.run(
                pending_prompt,
                session=session,
                options={"response_format": AGENT_TURN_RESPONSE_FORMAT},
            )
            turn_response = self._parse_turn_response(result)
            field_name = _AGENT_AWAIT_USER_INPUT_TRACE_FIELD if turn_response.awaits_user_input else "final-output"
            _trace_agent_output(self.name, turn_response.message, field_name=field_name)

            if not turn_response.awaits_user_input:
                return turn_response.message

            if user_input_provider is None:
                raise FtryCliError(
                    f"Agent `{self.name}` is awaiting user input, but no interactive user input provider is configured."
                )

            pending_prompt = user_input_provider(turn_response.message)
            _trace_agent_start(self.name, pending_prompt)

    def _require_supported_provider(self) -> None:
        provider = self.model.provider.lower()
        if provider != _OPENAI_PROVIDER:
            raise FtryCliError(
                f"Unsupported provider `{self.model.provider}`. Only `{_OPENAI_PROVIDER}` is supported for now."
            )

    def _build_instructions(
        self,
        extra_instructions: str | None,
        *,
        require_structured_turn_response: bool = False,
    ) -> str:
        rendered_instructions = self.instructions
        if extra_instructions:
            rendered_instructions = f"{rendered_instructions}\n\n<TeamContext>\n{extra_instructions}\n</TeamContext>"
        if require_structured_turn_response:
            rendered_instructions = f"{rendered_instructions}\n\n{AGENT_TURN_CONTROL_PROMPT}"
        return rendered_instructions

    @staticmethod
    def _parse_turn_response(result: Any) -> AgentTurnResponse:
        raw_value = getattr(result, "value", None)
        if not isinstance(raw_value, Mapping):
            raise FtryCliError("Agent response is missing the structured control payload required for console interaction.")

        status = raw_value.get("status")
        if status not in _AGENT_RESPONSE_STATUS_VALUES:
            raise FtryCliError(
                "Agent response has an invalid structured status. "
                f"Expected one of: {', '.join(_AGENT_RESPONSE_STATUS_VALUES)}."
            )

        message = raw_value.get("message")
        if not isinstance(message, str) or not message.strip():
            raise FtryCliError("Agent response is missing a non-empty structured message.")

        return AgentTurnResponse(message=message.strip(), status=status)

    def _create_openai_participant(
        self,
        *,
        extra_instructions: str | None = None,
        name_override: str | None = None,
        require_per_service_call_history_persistence: bool = False,
        require_structured_turn_response: bool = False,
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
            instructions=self._build_instructions(
                extra_instructions,
                require_structured_turn_response=require_structured_turn_response,
            ),
            require_per_service_call_history_persistence=require_per_service_call_history_persistence,
        )
