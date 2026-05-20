from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, Callable

from .Agent import Agent
from .AgentTurnControl import (
    AGENT_TURN_CONTROL_PROMPT,
    AGENT_TURN_RESPONSE_FORMAT,
    parse_agent_turn_response,
)
from .Tools import FtryCliError, _trace_agent_output, _trace_agent_start
_AGENT_AWAIT_USER_INPUT_TRACE_FIELD = "output [AWAIT USER INPUT]"


UserInputProvider = Callable[[str], str]


class StandaloneAgent(Agent):
    async def run(self, prompt: str, *, user_input_provider: UserInputProvider | None = None) -> str:
        async with AsyncExitStack() as exit_stack:
            mcp_tools, mcp_context = await self._prepare_mcp_runtime(exit_stack)
            participant = self.create_participant(
                extra_instructions=mcp_context,
                tools=mcp_tools,
            )
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
    def _parse_turn_response(result: Any):
        return parse_agent_turn_response(result, error_subject="Agent")
