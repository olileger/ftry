from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import uuid4
from typing import Any

from .Agent import Agent
from .AgentTurnControl import (
    AGENT_TURN_CONTROL_PROMPT,
    AGENT_TURN_RESPONSE_FORMAT,
    parse_agent_turn_response,
)
from .Tools import FtryCliError


@dataclass
class HandoffHilSignalState:
    action: str | None = None
    prompt: str | None = None
    message: str | None = None
    actor_name: str | None = None

    def request_user_input(self, prompt: str, actor_name: str) -> None:
        self.action = "request_user_input"
        self.prompt = prompt
        self.message = None
        self.actor_name = actor_name

    def finalize(self, message: str, actor_name: str) -> None:
        self.action = "final_answer"
        self.prompt = None
        self.message = message
        self.actor_name = actor_name

    def clear(self) -> None:
        self.action = None
        self.prompt = None
        self.message = None
        self.actor_name = None


HANDOFF_HIL_CONTROL_PROMPT = """<HandoffInteractionContract>
Quand tu travailles dans une team handoff:
- Utilise les tools de handoff natifs du framework pour transferer le controle a un autre agent quand c'est necessaire.
- Si tu as besoin d'une information supplementaire de l'utilisateur, appelle le tool `request_user_input` avec une seule question claire.
- Si le traitement est termine et qu'il ne faut pas redonner la main a l'utilisateur, appelle le tool `final_answer` avec la reponse finale visible.
- N'attends pas un nouvel input utilisateur en texte libre sans appeler `request_user_input`.
- N'essaie pas de clore la conversation en texte libre sans appeler `final_answer`.
</HandoffInteractionContract>"""


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
        enforce_structured_output: bool = True,
        handoff_hil_signal_state: HandoffHilSignalState | None = None,
    ) -> Any:
        self._require_supported_provider()
        rendered_instructions = (
            self._build_instructions(extra_instructions)
            if handoff_hil_signal_state is not None
            else self._build_managed_participant_instructions(extra_instructions)
        )
        if handoff_hil_signal_state is not None:
            rendered_instructions = f"{rendered_instructions}\n\n{HANDOFF_HIL_CONTROL_PROMPT}"
        participant = self._create_openai_participant(
            rendered_instructions=rendered_instructions,
            name_override=name_override,
            require_per_service_call_history_persistence=require_per_service_call_history_persistence,
        )
        _configure_team_managed_participant(
            participant,
            enforce_structured_output=enforce_structured_output,
            handoff_hil_signal_state=handoff_hil_signal_state,
        )
        return participant


def _configure_team_managed_participant(
    agent: Any,
    *,
    enforce_structured_output: bool,
    handoff_hil_signal_state: HandoffHilSignalState | None,
) -> None:
    try:
        from agent_framework import AgentResponse, Content, Message, ResponseStream, agent_middleware, tool
    except ImportError as exc:  # pragma: no cover - covered by CLI error path
        raise FtryCliError(
            "Microsoft Agent Framework middleware support is required for `ftry pop -t`. "
            "Reinstall the project with `python -m pip install -e .`."
        ) from exc

    if enforce_structured_output:
        default_options = dict(getattr(agent, "default_options", {}) or {})
        default_options["response_format"] = AGENT_TURN_RESPONSE_FORMAT
        agent.default_options = default_options
    elif handoff_hil_signal_state is not None:
        _attach_handoff_hil_tools(
            agent,
            signal_state=handoff_hil_signal_state,
            tool_decorator=tool,
        )
        if hasattr(agent, "_cached_agent_middleware_pipeline"):
            agent._cached_agent_middleware_pipeline = None
        return

    existing_middleware = list(getattr(agent, "middleware", []) or [])
    existing_middleware.append(_create_team_turn_control_middleware(AgentResponse, Content, Message, ResponseStream, agent_middleware))
    agent.middleware = existing_middleware
    current_agent_middleware = list(getattr(agent, "agent_middleware", []) or [])
    for middleware in existing_middleware:
        if getattr(middleware, "_middleware_type", None) is None or middleware in current_agent_middleware:
            continue
        current_agent_middleware.append(middleware)
    agent.agent_middleware = current_agent_middleware
    if hasattr(agent, "_cached_agent_middleware_pipeline"):
        agent._cached_agent_middleware_pipeline = None


def _create_team_turn_control_middleware(
    agent_response_type: Any,
    content_type: Any,
    message_type: Any,
    response_stream_type: Any,
    middleware_decorator: Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]],
) -> Callable[..., Awaitable[None]]:
    @middleware_decorator
    async def team_turn_control_middleware(context: Any, call_next: Callable[[], Awaitable[None]]) -> None:
        requested_stream = bool(getattr(context, "stream", False))
        if requested_stream:
            context.stream = False

        await call_next()
        transformed_response = _transform_team_agent_response(
            getattr(context, "result", None),
            agent_response_type=agent_response_type,
            content_type=content_type,
            message_type=message_type,
            author_name=getattr(getattr(context, "agent", None), "name", None),
        )

        if requested_stream:
            context.stream = True
            context.result = response_stream_type(_empty_async_iterable(), finalizer=lambda _: transformed_response)
            return

        context.result = transformed_response

    return team_turn_control_middleware


def _transform_team_agent_response(
    response: Any,
    *,
    agent_response_type: Any,
    content_type: Any,
    message_type: Any,
    author_name: str | None,
) -> Any:
    if _response_contains_tool_payload(response):
        return response

    turn_response = parse_agent_turn_response(response, error_subject="Team agent")
    message_content = content_type.from_text(turn_response.message)
    if turn_response.awaits_user_input:
        message_content.id = f"team-user-input-{uuid4()}"
        message_content.user_input_request = True

    return agent_response_type(
        messages=[message_type("assistant", [message_content], author_name=author_name)],
        response_id=getattr(response, "response_id", None),
        agent_id=getattr(response, "agent_id", None) or author_name,
        created_at=getattr(response, "created_at", None),
        usage_details=getattr(response, "usage_details", None),
        value=getattr(response, "value", None),
        raw_representation=getattr(response, "raw_representation", None),
        additional_properties=getattr(response, "additional_properties", None),
    )


async def _empty_async_iterable() -> Any:
    if False:  # pragma: no cover
        yield None


def _response_contains_tool_payload(response: Any) -> bool:
    for message in getattr(response, "messages", []) or []:
        for content in getattr(message, "contents", []) or []:
            if getattr(content, "type", None) in {"function_call", "function_result"}:
                return True
    return False


def _attach_handoff_hil_tools(
    agent: Any,
    *,
    signal_state: HandoffHilSignalState,
    tool_decorator: Callable[..., Any],
) -> None:
    agent_name = getattr(agent, "name", "agent")
    default_options = dict(getattr(agent, "default_options", {}) or {})
    existing_tools = list(default_options.get("tools") or [])
    existing_tool_names = {getattr(tool, "name", "") for tool in existing_tools}

    if "request_user_input" not in existing_tool_names:
        @tool_decorator(name="request_user_input", description="Ask the end user for one missing detail.", approval_mode="never_require")
        def request_user_input(question: str) -> str:
            signal_state.request_user_input(question.strip(), agent_name)
            return "User input request recorded."

        existing_tools.append(request_user_input)

    if "final_answer" not in existing_tool_names:
        @tool_decorator(name="final_answer", description="Mark the task as complete with the final user-visible answer.", approval_mode="never_require")
        def final_answer(message: str) -> str:
            signal_state.finalize(message.strip(), agent_name)
            return "Final answer recorded."

        existing_tools.append(final_answer)

    default_options["tools"] = existing_tools
    agent.default_options = default_options
