from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .Agent import Agent
from .Tools import FtryCliError, _trace_agent_output, _trace_agent_start

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


class StandaloneAgent(Agent):
    async def run(self, prompt: str, *, user_input_provider: UserInputProvider | None = None) -> str:
        participant = self.create_participant()
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

    def _build_participant_instructions(self, extra_instructions: str | None) -> str:
        return f"{self._build_instructions(extra_instructions)}\n\n{AGENT_TURN_CONTROL_PROMPT}"

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
