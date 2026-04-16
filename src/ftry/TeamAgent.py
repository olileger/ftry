from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .Agent import Agent
from .AgentTurnControl import (
    AGENT_TURN_CONTROL_PROMPT,
    AGENT_TURN_RESPONSE_FORMAT,
    parse_agent_turn_response,
)
from .Tools import FtryCliError


@dataclass(frozen=True)
class TeamAgentUserInputRequest:
    prompt: str
    agent_name: str


class TeamAgent(Agent):
    def _build_participant_instructions(self, extra_instructions: str | None) -> str:
        return self._build_instructions(extra_instructions)

    def _build_managed_participant_instructions(self, extra_instructions: str | None) -> str:
        return f"{self._build_instructions(extra_instructions)}\n\n{AGENT_TURN_CONTROL_PROMPT}"

    def create_managed_participant(
        self,
        *,
        extra_instructions: str | None = None,
        name_override: str | None = None,
        require_per_service_call_history_persistence: bool = False,
    ) -> Any:
        self._require_supported_provider()
        participant = self._create_openai_participant(
            rendered_instructions=self._build_managed_participant_instructions(extra_instructions),
            name_override=name_override,
            require_per_service_call_history_persistence=require_per_service_call_history_persistence,
        )
        executor_id = name_override or getattr(participant, "name", None)
        return _create_team_managed_agent_executor(participant, executor_id=executor_id)


def _create_team_managed_agent_executor(agent: Any, *, executor_id: str | None) -> Any:
    try:
        from agent_framework import AgentExecutor, AgentExecutorResponse, AgentResponse, Content, Message, response_handler
    except ImportError as exc:  # pragma: no cover - covered by CLI error path
        raise FtryCliError(
            "Microsoft Agent Framework workflow executor support is required for `ftry pop -t`. "
            "Reinstall the project with `python -m pip install -e .`."
        ) from exc

    class TeamManagedAgentExecutor(AgentExecutor):
        async def _run_agent_and_emit(self, ctx: Any) -> None:
            function_invocation_kwargs, client_kwargs = self._prepare_agent_run_args(
                ctx.get_state("workflow_run_kwargs", {})
            )

            response = await self._agent.run(
                self._cache,
                stream=False,
                session=self._session,
                options={"response_format": AGENT_TURN_RESPONSE_FORMAT},
                function_invocation_kwargs=function_invocation_kwargs,
                client_kwargs=client_kwargs,
            )

            if getattr(response, "user_input_requests", None):
                for user_input_request in response.user_input_requests:
                    self._pending_agent_requests[user_input_request.id] = user_input_request
                    await ctx.request_info(user_input_request, Content)
                return

            turn_response = parse_agent_turn_response(response, error_subject="Team agent")
            visible_response = AgentResponse(
                messages=[Message("assistant", [turn_response.message], author_name=self.id)],
                response_id=getattr(response, "response_id", None),
                agent_id=getattr(response, "agent_id", None) or self.id,
                created_at=getattr(response, "created_at", None),
                usage_details=getattr(response, "usage_details", None),
                value=getattr(response, "value", None),
                raw_representation=getattr(response, "raw_representation", None),
                additional_properties=getattr(response, "additional_properties", None),
            )
            await ctx.yield_output(visible_response)

            self._full_conversation = [*self._cache, *visible_response.messages]
            if turn_response.awaits_user_input:
                await ctx.request_info(
                    TeamAgentUserInputRequest(prompt=turn_response.message, agent_name=self.id),
                    str,
                )
                return

            await ctx.send_message(
                AgentExecutorResponse(self.id, visible_response, full_conversation=self._full_conversation)
            )
            self._cache.clear()

        @response_handler(request=TeamAgentUserInputRequest, response=str)
        async def handle_team_input_response(
            self,
            original_request: TeamAgentUserInputRequest,
            response: str,
            ctx,
        ) -> None:
            self._cache.append(
                Message(
                    "user",
                    [response],
                    additional_properties={"team_agent_request_prompt": original_request.prompt},
                )
            )
            await self._run_agent_and_emit(ctx)

    return TeamManagedAgentExecutor(agent, id=executor_id)
