from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .Tools import FtryCliError

AGENT_RESPONSE_STATUS_DONE = "done"
AGENT_RESPONSE_STATUS_AWAIT_USER_INPUT = "await_user_input"
AGENT_RESPONSE_STATUS_VALUES = (
    AGENT_RESPONSE_STATUS_DONE,
    AGENT_RESPONSE_STATUS_AWAIT_USER_INPUT,
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
                    "enum": list(AGENT_RESPONSE_STATUS_VALUES),
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
        return self.status == AGENT_RESPONSE_STATUS_AWAIT_USER_INPUT


def parse_agent_turn_response(result: Any, *, error_subject: str) -> AgentTurnResponse:
    raw_value = getattr(result, "value", None)
    if not isinstance(raw_value, Mapping):
        raise FtryCliError(
            f"{error_subject} response is missing the structured control payload required for console interaction."
        )

    status = raw_value.get("status")
    if status not in AGENT_RESPONSE_STATUS_VALUES:
        raise FtryCliError(
            f"{error_subject} response has an invalid structured status. "
            f"Expected one of: {', '.join(AGENT_RESPONSE_STATUS_VALUES)}."
        )

    message = raw_value.get("message")
    if not isinstance(message, str) or not message.strip():
        raise FtryCliError(f"{error_subject} response is missing a non-empty structured message.")

    return AgentTurnResponse(message=message.strip(), status=status)
